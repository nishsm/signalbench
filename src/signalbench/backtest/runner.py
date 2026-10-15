"""Backtest artifact production with declared robustness subsets."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from signalbench.backtest.analysis import (
    buckets,
    clustered_mean,
    events,
    information_coefficient,
    portfolio,
)
from signalbench.core import artifact
from signalbench.eval.benchmark import savefig


def run(cfg: dict, con) -> None:
    """Write event studies, IC, transaction-cost sensitivities and coverage."""
    filings = con.execute(
        "SELECT f.*,s.model,s.score FROM filings f JOIN filing_scores s USING(accession)"
    ).df()
    prices = con.execute("SELECT * FROM prices ORDER BY date,ticker").df()
    if filings.empty or prices.empty:
        raise RuntimeError("Need filing scores and prices before backtesting")
    frame, failures = events(filings, prices)
    if frame.empty:
        raise RuntimeError("No complete events; inspect price coverage")
    out = Path(cfg["paths"]["results"])
    artifact(out / "backtest_exclusions.csv", cfg, failures.to_csv(index=False))
    all_summary, event_rows, ic_rows = [], [], []
    midpoint = (
        pd.Timestamp(cfg["edgar"]["start"])
        + (pd.Timestamp(cfg["edgar"]["end"]) - pd.Timestamp(cfg["edgar"]["start"])) / 2
    )
    counts = frame.groupby("accession").model.nunique()
    common_ids = counts.index[counts == frame.model.nunique()]
    artifact(
        out / "backtest_coverage.csv",
        cfg,
        frame.groupby("model").accession.nunique().rename("events").to_csv(),
    )
    for name, group in frame.groupby("model"):
        group = group.copy()
        group["bucket"] = buckets(group)
        cutoff = cfg["backtest"]["training_cutoffs"].get(name)
        subsets = {
            "all": group,
            "common_coverage": group[group.accession.isin(common_ids)],
            "earnings": group[group["items"].str.contains("2.02", regex=False)],
            "non_earnings": group[~group["items"].str.contains("2.02", regex=False)],
            "early": group[group.event_date < midpoint],
            "late": group[group.event_date >= midpoint],
        }
        if cutoff:
            subsets["post_cutoff"] = group[
                pd.to_datetime(group.accepted_at, utc=True) > pd.Timestamp(cutoff, tz="UTC")
            ]
        for subset, data in subsets.items():
            if data.empty:
                continue
            ic, monthly = information_coefficient(data, cfg["backtest"]["minimum_month_events"])
            monthly["model"], monthly["subset"] = name, subset
            ic_rows.append(monthly)
            for bucket, bucket_frame in data.groupby("bucket"):
                for value in ["ar_1", "car_1_5", "car_1_20"]:
                    event_rows.append(
                        {
                            "model": name,
                            "subset": subset,
                            "bucket": bucket,
                            "horizon": value,
                            **clustered_mean(bucket_frame, value),
                        }
                    )
            for bps in cfg["backtest"]["costs_bps"]:
                stats, daily = portfolio(data, prices, cfg["backtest"]["hold_days"], bps)
                all_summary.append(
                    {
                        "model": name,
                        "subset": subset,
                        **ic,
                        **stats,
                        "filings": data.accession.nunique(),
                        "companies": data.ticker.nunique(),
                        "start": str(data.event_date.min().date()),
                        "end": str(data.event_date.max().date()),
                        "cutoff_verified": bool(cutoff),
                    }
                )
                artifact(
                    out / f"portfolio/{name}_{subset}_{bps}bps.csv", cfg, daily.to_csv(index=False)
                )
                if stats["sharpe"] is not None and stats["sharpe"] > 2:
                    artifact(
                        out / "REVIEW_REQUIRED.json",
                        cfg,
                        json.dumps({"model": name, "subset": subset, **stats}, indent=2),
                    )
                    raise RuntimeError("Sharpe >2: investigate alignment/leakage before reporting")
                if subset == "all" and bps == 10:
                    fig, ax = plt.subplots()
                    ax.plot(daily.date, daily.equity)
                    ax.set(title=f"{name}: long-short after 10 bps", ylabel="Equity")
                    savefig(fig, out / f"figures/{name}_pnl.png", cfg)
            if subset == "all":
                fig, ax = plt.subplots()
                for bucket, part in data.groupby("bucket"):
                    ax.plot(
                        range(21), [part[f"car_{h}"].mean() for h in range(21)], label=str(bucket)
                    )
                ax.legend(title="Historical sentiment bucket")
                ax.set(xlabel="Session offset", ylabel="Market-adjusted return", title=name)
                savefig(fig, out / f"figures/{name}_car.png", cfg)
                fig, ax = plt.subplots()
                ax.plot(monthly.month, monthly.ic)
                ax.tick_params(axis="x", rotation=90)
                ax.set(title=f"{name}: monthly IC")
                savefig(fig, out / f"figures/{name}_ic.png", cfg)
    summary = pd.DataFrame(all_summary)
    artifact(out / "backtest.csv", cfg, summary.to_csv(index=False))
    artifact(out / "event_study.csv", cfg, pd.DataFrame(event_rows).to_csv(index=False))
    artifact(out / "monthly_ic.csv", cfg, pd.concat(ic_rows).to_csv(index=False))
    artifact(
        out / "backtest.md",
        cfg,
        "# Measured backtest\n\n"
        + summary.to_markdown(index=False)
        + "\n\nMarket-cap robustness unavailable unless historical "
        "point-in-time caps are supplied. "
        "Current caps must not be used. Compare API and local models on common coverage as well. "
        "A historical event study is descriptive; portfolio entries are at the next eligible open.",
    )
