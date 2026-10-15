"""Event studies, monthly IC and overlapping daily executable cohorts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

from signalbench.backtest.alignment import abnormal, align, schedule


def events(filings: pd.DataFrame, prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build paired horizons, excluding incomplete windows with explicit reasons."""
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices.date)
    calendar = schedule(prices.date.min(), prices.date.max())
    close = prices.pivot(index="date", columns="ticker", values="adj_close").reindex(calendar.index)
    rows, failures = [], []
    for filing in filings.itertuples():
        try:
            date, entry = align(filing.accepted_at, calendar)
            i = close.index.get_loc(date)
            if i < 1 or i + 20 >= len(close):
                raise ValueError("incomplete_forward_window")
            stock, spy = close[filing.ticker], close["SPY"]
            row = filing._asdict() | {"event_date": date, "entry_at": entry}
            row.update({f"car_{h}": abnormal(stock, spy, i - 1, i + h) for h in range(21)})
            row["ar_1"] = row["car_0"]
            row["car_1_5"] = abnormal(stock, spy, i, i + 5)
            row["car_1_20"] = abnormal(stock, spy, i, i + 20)
            rows.append(row)
        except (ValueError, KeyError) as exc:
            failures.append(
                {"accession": filing.accession, "model": filing.model, "reason": str(exc)}
            )
    return pd.DataFrame(rows), pd.DataFrame(failures)


def buckets(frame: pd.DataFrame) -> pd.Series:
    """Expanding historical tertile thresholds, never full-sample or same-day future ranks.

    First 30 events are warm-up. Ties at a cutpoint stay neutral to avoid arbitrary sorting.
    """
    assigned = pd.Series(0, index=frame.index, dtype=int)
    for date, group in frame.groupby("event_date", sort=True):
        history = frame.loc[frame.event_date < date, "score"]
        if len(history) < 30:
            continue
        lo, hi = history.quantile([1 / 3, 2 / 3])
        assigned.loc[group.index[group.score < lo]] = -1
        assigned.loc[group.index[group.score > hi]] = 1
    return assigned


def clustered_mean(frame: pd.DataFrame, value: str) -> dict:
    """Intercept-only regression with standard errors clustered by event date."""
    if len(frame) < 2 or frame.event_date.nunique() < 2:
        return {"mean": float(frame[value].mean()), "se": None, "t": None, "n": len(frame)}
    fit = sm.OLS(frame[value].to_numpy(), np.ones((len(frame), 1))).fit(
        cov_type="cluster", cov_kwds={"groups": frame.event_date.to_numpy()}
    )
    return {
        "mean": float(fit.params[0]),
        "se": float(fit.bse[0]),
        "t": float(fit.tvalues[0]),
        "n": len(frame),
    }


def information_coefficient(frame: pd.DataFrame, minimum: int = 10) -> tuple[dict, pd.DataFrame]:
    """Monthly Spearman IC; HAC mean t-stat allows serial dependence across months."""
    rows = []
    for month, group in frame.groupby(frame.event_date.dt.to_period("M")):
        if len(group) >= minimum and group.score.nunique() > 1 and group.car_1_5.nunique() > 1:
            rows.append(
                {
                    "month": str(month),
                    "ic": float(spearmanr(group.score, group.car_1_5).statistic),
                    "n": len(group),
                }
            )
    monthly = pd.DataFrame(rows, columns=["month", "ic", "n"])
    if len(monthly) < 2:
        return {"mean_ic": None, "ic_t": None, "hit_rate": None, "months": len(monthly)}, monthly
    fit = sm.OLS(monthly.ic, np.ones((len(monthly), 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": 3}
    )
    return {
        "mean_ic": float(monthly.ic.mean()),
        "ic_t": float(fit.tvalues.iloc[0]),
        "hit_rate": float((monthly.ic > 0).mean()),
        "months": len(monthly),
    }, monthly


def portfolio(frame: pd.DataFrame, prices: pd.DataFrame, hold: int, bps: float):
    """Fixed-capital daily cohorts, each side 50% gross, hold open-to-close for five days.

    Each cohort receives 1/hold of total capital. Empty sides remain cash. Shares drift
    rather than rebalancing daily. Costs apply at entry and exit actual traded notional.
    Same-ticker same-day filings are collapsed by mean bucket to avoid duplicated exposure.
    """
    p = prices.copy()
    p["date"] = pd.to_datetime(p.date)
    dates = schedule(p.date.min(), p.date.max()).index
    adj = p.pivot(index="date", columns="ticker", values="adj_close").reindex(dates)
    raw_close = p.pivot(index="date", columns="ticker", values="close").reindex(dates)
    opens = p.pivot(index="date", columns="ticker", values="open").reindex(dates)
    adj_open = opens * adj / raw_close
    pnl, traded = pd.Series(0.0, index=dates), pd.Series(0.0, index=dates)
    for date, group in frame.groupby("event_date"):
        i = dates.get_loc(date)
        if i + hold > len(dates):
            continue
        signals = group.groupby("ticker").bucket.mean()
        for side in [-1, 1]:
            names = signals.index[signals * side > 0]
            if len(names) == 0:
                continue
            weight = side * 0.5 / hold / len(names)
            for name in names:
                entry = adj_open.loc[date, name]
                path = adj[name].iloc[i : i + hold]
                if not np.isfinite(entry) or entry <= 0 or path.isna().any():
                    raise ValueError(f"Missing held price for {name} at {date}; cannot drop trades")
                # Entry and exits both at observable prices after publication.
                marks = np.r_[entry, path.to_numpy()]
                pnl.iloc[i : i + hold] += weight * np.diff(marks) / entry
                traded.iloc[i] += abs(weight)
                traded.iloc[i + hold - 1] += abs(weight) * marks[-1] / entry
    net = pnl - traded * bps / 10000
    # Fixed-capital daily PnL is divided by evolving equity for return statistics.
    equity = 1 + net.cumsum()
    if (equity <= 0).any():
        raise ValueError("Portfolio insolvent")
    returns = net / equity.shift(1, fill_value=1)
    volatility = returns.std(ddof=1) * np.sqrt(252)
    stats = {
        "annualized_return": float(equity.iloc[-1] ** (252 / len(equity)) - 1),
        "volatility": float(volatility),
        "sharpe": float(returns.mean() * 252 / volatility) if volatility else None,
        "max_drawdown": float((equity / equity.cummax().clip(lower=1) - 1).min()),
        "annual_turnover": float(traded.mean() * 252),
        "cost_bps": bps,
    }
    return stats, pd.DataFrame(
        {
            "date": dates,
            "gross_pnl": pnl.values,
            "turnover": traded.values,
            "net_return": returns.values,
            "equity": equity.values,
        }
    )
