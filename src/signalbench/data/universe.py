"""Explicit current-constituent proxy; historical membership is NOT inferred."""

from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd
from bs4 import BeautifulSoup

from signalbench.core import artifact
from signalbench.data.edgar import Edgar

SOURCE = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def run(cfg: dict) -> None:
    """Cache a current S&P 500 snapshot and join SEC ticker-to-CIK identifiers.

    This is intentionally labeled survivorship-biased. Unmatched tickers are saved for
    manual review, never silently substituted with unrelated historical securities.
    """
    cache = Path(cfg["paths"]["cache"]) / "universe.html"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        response = httpx.get(
            SOURCE,
            follow_redirects=True,
            timeout=60,
            headers={"User-Agent": "signalbench research"},
        )
        response.raise_for_status()
        cache.write_text(response.text)
    soup = BeautifulSoup(cache.read_text(), "html.parser")
    table = soup.find("table", id="constituents")
    if table is None:
        raise ValueError("Constituent table changed; review source before building universe")
    import json

    mapping = json.loads(
        Edgar(cfg["paths"]["cache"]).get("https://www.sec.gov/files/company_tickers.json")
    )
    ciks = {
        row["ticker"].replace(".", "-"): str(row["cik_str"]).zfill(10) for row in mapping.values()
    }
    # Snapshot date follows cached file creation, never the historical backtest date.
    as_of = datetime.fromtimestamp(cache.stat().st_mtime, timezone.utc).date().isoformat()
    rows, unmatched = [], []
    for row in table.select("tr")[1:]:
        cells = row.find_all("td")
        if not cells:
            continue
        ticker = cells[0].get_text(strip=True).replace(".", "-")
        if ticker not in ciks:
            unmatched.append(ticker)
            continue
        rows.append(
            {
                "ticker": ticker,
                "cik": ciks[ticker],
                "as_of": as_of,
                "membership_source": SOURCE,
                "survivorship_biased": True,
            }
        )
    if len(rows) < 300:
        raise ValueError("Unexpectedly small universe; review source and CIK joins")
    artifact(cfg["edgar"]["universe"], cfg, pd.DataFrame(rows).to_csv(index=False))
    artifact(
        Path(cfg["paths"]["results"]) / "universe_unmatched.json",
        cfg,
        json.dumps(unmatched, indent=2),
    )
