"""Sentence scoring and filing-level aggregation."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from signalbench.core import artifact
from signalbench.db import upsert
from signalbench.models.registry import model


def sentences(text: str, maximum: int = 40) -> list[str]:
    """Deterministic sentence heuristic excluding short headings and boilerplate fragments."""
    candidates = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [s.strip() for s in candidates if len(s.split()) >= 6][:maximum]


def run(cfg: dict, con) -> None:
    """Score cached exhibits, selecting API approach on validation performance only."""
    frame = con.execute(
        "SELECT f.*,t.text FROM filings f JOIN filing_text t USING(accession) ORDER BY accession"
    ).df()
    if frame.empty:
        raise RuntimeError("No exhibits; run make edgar after benchmark review")
    train = con.execute(
        "SELECT * FROM labeled_text WHERE split='train' AND source='sentences_allagree' ORDER BY id"
    ).df()
    names = cfg["signal"]["models"].copy()
    selection = Path(cfg["paths"]["results"]) / "selection.json"
    if selection.exists():
        best = json.loads(selection.read_text())["best_api"]
        if best and best not in names:
            names.append(best)
    coverage = []
    for name in names:
        estimator = model(name, cfg, train)
        subset = frame
        if name.startswith("api_"):
            all_texts = [
                s for text in frame.text for s in sentences(text, cfg["signal"]["sentences"])
            ]
            estimate = estimator.estimate(all_texts)
            print(json.dumps({"model": name, "full_filing_estimate_usd": estimate}))
            spent = estimator.db.execute("SELECT COALESCE(SUM(cost),0) FROM responses").fetchone()[
                0
            ]
            if estimate + spent > cfg["api"]["budget"]:
                subset = frame.sample(
                    min(len(frame), cfg["signal"]["api_subsample"]), random_state=cfg["seed"]
                )
                print(
                    "Budget guard: using seeded API subsample; ledger still enforces total budget"
                )
        for row in tqdm(list(subset.itertuples()), desc=name):
            parts = sentences(row.text, cfg["signal"]["sentences"])
            if not parts:
                coverage.append(
                    {"accession": row.accession, "model": name, "status": "no_sentences"}
                )
                continue
            predictions = estimator.predict(parts)
            score = np.mean([r["score"] for r in predictions])
            upsert(
                con,
                "filing_scores",
                pd.DataFrame(
                    [
                        {
                            "accession": row.accession,
                            "model": name,
                            "score": score,
                            "fraction_positive": np.mean(
                                [r["label"] == "positive" for r in predictions]
                            ),
                            "fraction_negative": np.mean(
                                [r["label"] == "negative" for r in predictions]
                            ),
                            "sentences": len(parts),
                        }
                    ]
                ),
            )
            coverage.append({"accession": row.accession, "model": name, "status": "scored"})
        del estimator
    artifact(
        Path(cfg["paths"]["results"]) / "signal_coverage.csv",
        cfg,
        pd.DataFrame(coverage).to_csv(index=False),
    )
