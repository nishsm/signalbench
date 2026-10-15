"""Cached adjusted daily price acquisition."""

from pathlib import Path

import pandas as pd

from signalbench.core import artifact
from signalbench.db import upsert


def run(cfg: dict, con) -> None:
    """Use Yahoo adjusted closes explicitly; raw opens remain unadjusted in storage.

    Yahoo is used instead of Stooq because total-return adjustment semantics must be explicit.
    Missing/delisted histories are retained in coverage, never forward-filled.
    """
    import yfinance as yf

    tickers = sorted(set(con.execute("SELECT DISTINCT ticker FROM filings").df().ticker) | {"SPY"})
    cache = Path(cfg["paths"]["cache"]) / "prices"
    cache.mkdir(parents=True, exist_ok=True)
    coverage = []
    for ticker in tickers:
        path = cache / f"{ticker}_{cfg['edgar']['start']}_{cfg['edgar']['end']}.csv"
        try:
            if path.exists():
                frame = pd.read_csv(path, parse_dates=["date"])
            else:
                raw = yf.Ticker(ticker).history(
                    start=pd.Timestamp(cfg["edgar"]["start"]) - pd.Timedelta(days=40),
                    end=pd.Timestamp(cfg["edgar"]["end"]) + pd.Timedelta(days=60),
                    auto_adjust=False,
                )
                if raw.empty:
                    raise ValueError("empty history")
                frame = pd.DataFrame(
                    {
                        "date": raw.index.tz_localize(None).normalize(),
                        "open": raw.Open.to_numpy(),
                        "close": raw.Close.to_numpy(),
                        "adj_close": raw["Adj Close"].to_numpy(),
                        "volume": raw.Volume.to_numpy(),
                    }
                )
                frame["ticker"] = ticker
                frame.to_csv(path, index=False)
            upsert(con, "prices", frame)
            coverage.append({"ticker": ticker, "rows": len(frame), "error": ""})
        except (ValueError, KeyError) as exc:
            coverage.append({"ticker": ticker, "rows": 0, "error": str(exc)})
    artifact(
        Path(cfg["paths"]["results"]) / "price_coverage.csv",
        cfg,
        pd.DataFrame(coverage).to_csv(index=False),
    )
