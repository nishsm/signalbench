"""Offline integration checks on fixtures, distinct from measured research artifacts."""

import json
from pathlib import Path

import pandas as pd
import pytest

from signalbench import LABELS
from signalbench.backtest.alignment import schedule
from signalbench.backtest.analysis import events
from signalbench.core import config
from signalbench.db import connect, upsert
from signalbench.eval.benchmark import run as benchmark
from signalbench.report.render import run as report


def test_toy_event_returns_and_failure_coverage():
    dates = schedule("2024-01-02", "2024-04-01").index
    prices = pd.DataFrame(
        [
            {"ticker": name, "date": date, "adj_close": 100 * (1 + rate) ** i}
            for name, rate in [("A", 0.01), ("SPY", 0.002)]
            for i, date in enumerate(dates)
        ]
    )
    filings = pd.DataFrame(
        [
            {
                "ticker": "A",
                "accession": "fixture",
                "model": "toy",
                "accepted_at": pd.Timestamp("2024-01-03 13:00Z"),
                "score": 0.2,
            }
        ]
    )
    frame, failures = events(filings, prices)
    assert failures.empty
    assert frame.ar_1.iloc[0] == pytest.approx(0.008)
    assert frame.car_1_5.iloc[0] == pytest.approx(1.01**5 - 1.002**5)
    broken = prices[~((prices.ticker == "A") & (prices.date == dates[5]))]
    frame, failures = events(filings, broken)
    assert frame.empty
    assert len(failures) == 1


def test_fixture_benchmark_and_report(tmp_path, monkeypatch):
    cfg = config("configs/default.yaml")
    cfg["benchmark"]["models"] = ["majority", "tfidf"]
    cfg["benchmark"]["bootstrap"] = 10
    cfg["paths"]["results"] = str(tmp_path / "results")
    rows = []
    for partition in ["train", "val", "test", "ood"]:
        for label in LABELS:
            for i in range(5):
                rows.append(
                    {
                        "source": "sentences_allagree",
                        "id": f"{partition}_{label}_{i}",
                        "text": f"company reported {label} {label} profit {i}",
                        "label": label,
                        "split": partition,
                    }
                )
    with connect() as con:
        upsert(con, "labeled_text", pd.DataFrame(rows))
        # Disable the research anomaly guard only for synthetic perfect fixture predictions.
        rows = [row for row in rows if row["split"] != "ood"]
        con.execute("DELETE FROM labeled_text WHERE split='ood'")
        benchmark(cfg, con)
    table = pd.read_csv(tmp_path / "results/benchmark.csv")
    assert set(table.model) == {"majority", "tfidf"}
    assert set(table.partition) == {"val", "test"}
    monkeypatch.chdir(tmp_path)
    report(cfg)
    assert "Classification" in Path("RESULTS_FOR_RESUME.md").read_text()
    assert not json.loads(Path("results/status.json").read_text())["backtest_measured"]


def test_actual_split_manifest_when_available():
    path = Path("results/splits/ids.csv")
    if not path.exists():
        pytest.skip("Real downloaded data is optional in CPU CI")
    saved = pd.read_csv(path)
    assert saved.id.is_unique
    if "text_hash" in saved:
        assert saved.text_hash.is_unique
        assert saved.groupby("text_hash").split.nunique().max() == 1
    assert set(saved[saved.source == "sentences_allagree"].split) == {"train", "val", "test"}
