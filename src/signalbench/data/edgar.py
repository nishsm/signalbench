"""Resumable EDGAR access including older submissions and EX-99.1 discovery."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import httpx
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

from signalbench.core import artifact, digest
from signalbench.db import upsert


class Edgar:
    """One sequential client, five requests per second, persistent URL cache."""

    def __init__(self, cache: str):
        agent = os.environ.get("EDGAR_USER_AGENT", "")
        if "@" not in agent or "example.com" in agent:
            raise ValueError("Set EDGAR_USER_AGENT to your name and real email")
        self.client = httpx.Client(headers={"User-Agent": agent}, timeout=60, follow_redirects=True)
        self.cache = Path(cache) / "edgar"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.last = 0.0

    def get(self, url: str) -> str:
        """Fetch with backoff; never cache an HTTP error as a filing."""
        path = self.cache / digest(url)
        if path.exists():
            return path.read_text()
        for attempt in range(5):
            time.sleep(max(0.0, 0.2 - (time.monotonic() - self.last)))
            self.last = time.monotonic()
            response = self.client.get(url)
            if response.status_code in {429, 500, 502, 503, 504}:
                time.sleep(2**attempt)
                continue
            response.raise_for_status()
            temporary = path.with_suffix(".tmp")
            temporary.write_text(response.text)
            temporary.replace(path)
            return response.text
        raise RuntimeError(f"EDGAR retries exhausted: {url}")


def exhibit_url(html: str, index_url: str) -> str:
    """Find EX-99.1 by document type in the filing index table."""
    soup = BeautifulSoup(html, "html.parser")
    for row in soup.select("tr"):
        cells = row.find_all("td")
        if any(re.fullmatch(r"EX-99\.1(?:\s.*)?", c.get_text(strip=True), re.I) for c in cells):
            link = row.find("a", href=True)
            if link:
                url = urljoin(index_url, link["href"])
                return url.replace("/ixviewer/doc/action?doc=", "").replace("/ix?doc=", "")
    raise ValueError("missing_ex99_1")


def extract_text(html: str) -> str:
    """Remove HTML, inline XBRL hidden nodes and tables, retaining paragraph breaks."""
    soup = BeautifulSoup(html, "html.parser")
    for node in soup.select("script,style,table,ix\\:hidden"):
        node.decompose()
    return "\n".join(
        line
        for line in (re.sub(r"\s+", " ", line).strip() for line in soup.get_text("\n").splitlines())
        if line
    )


def run(cfg: dict, con) -> None:
    """Acquire explicit universe; require benchmark checkpoint before broad collection."""
    if not cfg["edgar"]["scaleup_approved"]:
        raise RuntimeError(
            "Benchmark checkpoint: review results with Nish before EDGAR scale-up; "
            "then set edgar.scaleup_approved=true"
        )
    universe = pd.read_csv(cfg["edgar"]["universe"], dtype={"cik": str, "ticker": str})
    if universe.ticker.duplicated().any():
        raise ValueError("Universe must have one row per ticker")
    client = Edgar(cfg["paths"]["cache"])
    artifact(Path(cfg["paths"]["results"]) / "universe.csv", cfg, universe.to_csv(index=False))
    for company in tqdm(list(universe.itertuples()), desc="EDGAR companies"):
        cik = str(int(company.cik)).zfill(10)
        main = json.loads(client.get(f"https://data.sec.gov/submissions/CIK{cik}.json"))
        blocks = [main["filings"]["recent"]]
        for old in main["filings"].get("files", []):
            if old["filingTo"] >= cfg["edgar"]["start"]:
                blocks.append(
                    json.loads(client.get("https://data.sec.gov/submissions/" + old["name"]))
                )
        for block in blocks:
            for row in pd.DataFrame(block).itertuples():
                if row.form != "8-K" or not (
                    cfg["edgar"]["start"] <= row.filingDate <= cfg["edgar"]["end"]
                ):
                    continue
                accession = row.accessionNumber
                if con.execute(
                    "SELECT 1 FROM filing_text WHERE accession=?", [accession]
                ).fetchone():
                    continue
                compact = accession.replace("-", "")
                base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{compact}/"
                index = base + accession + "-index.html"
                try:
                    timestamp = pd.Timestamp(row.acceptanceDateTime)
                    if timestamp.tzinfo is None:
                        raise ValueError("ambiguous_acceptance_timezone")
                    upsert(
                        con,
                        "filings",
                        pd.DataFrame(
                            [
                                {
                                    "cik": cik,
                                    "ticker": company.ticker,
                                    "accession": accession,
                                    "accepted_at": timestamp,
                                    "items": getattr(row, "items", ""),
                                    "url": index,
                                }
                            ]
                        ),
                    )
                    url = exhibit_url(client.get(index), index)
                    text = extract_text(client.get(url))
                    if len(text.split()) < 20:
                        raise ValueError("insufficient_exhibit_text")
                    upsert(
                        con, "filing_text", pd.DataFrame([{"accession": accession, "text": text}])
                    )
                    con.execute("DELETE FROM failures WHERE accession=?", [accession])
                except (ValueError, httpx.HTTPError, RuntimeError) as exc:
                    upsert(
                        con,
                        "failures",
                        pd.DataFrame(
                            [{"accession": accession, "stage": "edgar", "reason": str(exc)}]
                        ),
                    )
    coverage = con.execute("SELECT stage,reason,COUNT(*) AS n FROM failures GROUP BY ALL").df()
    summary = {
        "filings_found": con.execute("SELECT COUNT(*) FROM filings").fetchone()[0],
        "exhibits_parsed": con.execute("SELECT COUNT(*) FROM filing_text").fetchone()[0],
    }
    artifact(
        Path(cfg["paths"]["results"]) / "coverage.md",
        cfg,
        json.dumps(summary, indent=2) + "\n\n" + coverage.to_markdown(index=False),
    )
