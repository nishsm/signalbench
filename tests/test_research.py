"""Small offline methodology checks; no API calls or model downloads."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from signalbench import LABELS
from signalbench.backtest.alignment import abnormal, align, schedule
from signalbench.backtest.analysis import buckets, portfolio
from signalbench.data.datasets import deduplicate, normalize, split
from signalbench.data.edgar import exhibit_url, extract_text
from signalbench.db import connect, upsert
from signalbench.eval.metrics import bootstrap, calibration, mcnemar, metrics
from signalbench.models.common import output
from signalbench.models.rag_fewshot import Examples


def test_dedup_and_split():
    rows = [
        {
            "id": f"{label}{i:03}",
            "text": f"{label} company {i * 293} revenue {i * 17}",
            "label": label,
            "source": "test",
        }
        for label in LABELS
        for i in range(30)
    ]
    rows += [
        {
            "id": "zzduplicate",
            "text": rows[0]["text"].upper(),
            "label": rows[0]["label"],
            "source": "test",
        }
    ]
    frame = deduplicate(pd.DataFrame(rows))
    result = split(frame, 42)
    assert result.text.map(normalize).nunique() == len(result)
    assert result.groupby(result.text.map(normalize)).split.nunique().max() == 1
    assert set(result.split) == {"train", "val", "test"}
    pd.testing.assert_frame_equal(result, split(frame, 42))


def test_conflicting_duplicates_excluded():
    frame = pd.DataFrame(
        {"id": ["a", "b"], "text": ["Profit grew strongly."] * 2, "label": ["positive", "negative"]}
    )
    assert deduplicate(frame).empty


def test_near_duplicates():
    text = "The company reported strong operating profits and rising revenue this quarter"
    frame = pd.DataFrame({"id": ["a", "b"], "text": [text, text + "."], "label": ["positive"] * 2})
    assert len(deduplicate(frame)) == 1


def test_label_contract():
    rows = output(np.eye(3))
    assert [r["label"] for r in rows] == list(LABELS)
    assert [r["score"] for r in rows] == [-1, 0, 1]
    with pytest.raises(ValueError):
        output([[0.4, 0.4, 0.4]])


def test_metrics_hand_computed():
    y = ["negative", "neutral", "positive"]
    pred = ["negative", "neutral", "neutral"]
    result = metrics(y, pred)
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["macro_f1"] == pytest.approx((1 + 2 / 3) / 3)
    assert mcnemar(y, pred, pred)["p_value"] == 1


def test_bootstrap_determinism():
    y = list(LABELS) * 3
    a = bootstrap(y, y[::-1], seed=7, n=50)
    assert a == bootstrap(y, y[::-1], seed=7, n=50)
    assert a["macro_f1_low"] <= a["macro_f1_high"]


def test_calibration():
    ece, _ = calibration(LABELS, np.eye(3))
    assert ece == 0
    with pytest.raises(ValueError):
        calibration(LABELS, [[1, 1, 1]] * 3)


@pytest.mark.parametrize(
    "accepted,expected",
    [
        ("2024-07-01 09:00", "2024-07-01"),
        ("2024-07-01 10:00", "2024-07-02"),
        ("2024-07-01 16:30", "2024-07-02"),
        ("2024-07-05 17:00", "2024-07-08"),
        ("2024-07-04 08:00", "2024-07-05"),
        ("2024-07-01 09:30", "2024-07-02"),
        ("2024-07-03 13:30", "2024-07-05"),
    ],
)
def test_point_in_time(accepted, expected):
    ts = pd.Timestamp(accepted, tz="America/New_York")
    date, entry = align(ts, schedule("2024-07-01", "2024-07-12"))
    assert str(date.date()) == expected
    assert entry > ts


def test_naive_timestamp_rejected():
    with pytest.raises(ValueError):
        align("2024-07-01", schedule("2024-07-01", "2024-07-12"))


def test_abnormal_return():
    assert abnormal(pd.Series([100, 110]), pd.Series([100, 102]), 0, 1) == pytest.approx(0.08)


def test_edgar_fixture():
    html = Path("tests/fixtures/filing-index.html").read_text()
    assert (
        exhibit_url(html, "https://www.sec.gov/test")
        == "https://www.sec.gov/Archives/example/press.htm"
    )
    assert (
        extract_text("<p>Revenue rose.</p><table><tr><td>99</td></tr></table><script>bad</script>")
        == "Revenue rose."
    )
    with pytest.raises(ValueError, match="missing_ex99_1"):
        exhibit_url("<p>no exhibits</p>", "https://www.sec.gov")


def test_schema():
    with connect() as con:
        assert len(con.execute("SHOW TABLES").fetchall()) == 6
        row = pd.DataFrame(
            [
                {
                    "source": "fixture",
                    "id": "x",
                    "text": "profit",
                    "label": "positive",
                    "split": "train",
                }
            ]
        )
        upsert(con, "labeled_text", row)
        upsert(con, "labeled_text", row)
        assert con.execute("SELECT COUNT(*) FROM labeled_text").fetchone()[0] == 1


def test_rag_rejects_test_data():
    frame = pd.DataFrame({"split": ["test"]})
    with pytest.raises(ValueError, match="training-only"):
        Examples(frame, "few", 1, 42)


def test_bucket_prefix_invariance():
    frame = pd.DataFrame(
        {"event_date": pd.date_range("2020-01-01", periods=50), "score": np.linspace(-1, 1, 50)}
    )
    before = buckets(frame)
    frame.loc[40:, "score"] = -100
    pd.testing.assert_series_equal(before.iloc[:40], buckets(frame).iloc[:40])


def test_portfolio_no_prior_close_capture_and_costs():
    dates = schedule("2024-07-01", "2024-07-12").index
    prices = pd.DataFrame(
        [
            {"date": date, "ticker": ticker, "open": 200.0, "close": 200.0, "adj_close": 200.0}
            for date in dates
            for ticker in ["A", "B"]
        ]
    )
    # Huge event-day gap is already in the open; a flat open-to-close path earns nothing.
    frame = pd.DataFrame(
        {"event_date": [dates[0], dates[0]], "ticker": ["A", "B"], "bucket": [1, -1]}
    )
    _, gross = portfolio(frame, prices, 5, 0)
    _, net = portfolio(frame, prices, 5, 10)
    assert gross.gross_pnl.sum() == 0
    assert gross.turnover.sum() == pytest.approx(0.4)
    assert net.equity.iloc[-1] == pytest.approx(1 - 0.4 * 0.001)


def test_prompt_truncation_preserves_assistant_prefix():
    from signalbench.models.qlora import bounded_prompt

    class Tokenizer:
        def encode(self, text, **kwargs):
            return [ord(char) for char in text]

        def decode(self, tokens, **kwargs):
            return "".join(chr(token) for token in tokens)

        def apply_chat_template(self, messages, **kwargs):
            return "instruction|" + messages[-1]["content"] + "|assistant:"

    prefix = bounded_prompt(Tokenizer(), "x" * 1000, 100)
    assert prefix.startswith("instruction|")
    assert prefix.endswith("|assistant:")
    assert len(prefix) <= 92
