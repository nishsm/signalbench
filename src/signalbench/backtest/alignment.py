"""Exchange-calendar event alignment and strictly post-publication entry."""

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal


def schedule(start, end) -> pd.DataFrame:
    """NYSE schedule in UTC, including holidays and early closes."""
    return mcal.get_calendar("NYSE").schedule(start_date=start, end_date=end)


def align(accepted_at, sessions: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return event session and entry open; equality at the open rolls forward.

    A scheduled open must be strictly later than public acceptance. Day-0 close returns
    from the prior close are descriptive event returns, not executable portfolio PnL.
    """
    accepted = pd.Timestamp(accepted_at)
    if accepted.tzinfo is None:
        raise ValueError("Acceptance timestamp must include its timezone")
    valid = sessions[sessions.market_open > accepted.tz_convert("UTC")]
    if valid.empty:
        raise ValueError("Calendar ends before event")
    return pd.Timestamp(valid.index[0]).normalize(), valid.market_open.iloc[0]


def abnormal(stock: pd.Series, market: pd.Series, start: int, end: int) -> float:
    """Buy-and-hold stock return minus market return over identical endpoints."""
    values = np.array([stock.iloc[start], stock.iloc[end], market.iloc[start], market.iloc[end]])
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Missing/nonpositive endpoint price")
    return float(values[1] / values[0] - values[3] / values[2])
