"""US holiday calendar features.

Holidays are matched on their actual calendar date (not the "observed" weekday), because demand
follows the real day. is_holiday_adjacent marks the day before or after a holiday, which covers
Christmas Eve, the day after Christmas, the Friday after Thanksgiving, and so on.
"""
import datetime as dt

import pandas as pd


def nth_weekday(year: int, month: int, weekday: int, n: int) -> dt.date:
    """n-th given weekday of a month (Monday=0). n=-1 means the last one."""
    if n > 0:
        first = dt.date(year, month, 1)
        day = 1 + (weekday - first.weekday()) % 7 + 7 * (n - 1)
        return dt.date(year, month, day)
    last = (pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)).date()
    return last - dt.timedelta(days=(last.weekday() - weekday) % 7)


def holiday_dates(years) -> set:
    out = set()
    for y in years:
        out |= {
            dt.date(y, 1, 1), dt.date(y, 7, 4), dt.date(y, 12, 25),
            nth_weekday(y, 1, 0, 3),    # Martin Luther King Jr. Day
            nth_weekday(y, 2, 0, 3),    # Presidents Day
            nth_weekday(y, 5, 0, -1),   # Memorial Day
            nth_weekday(y, 9, 0, 1),    # Labor Day
            nth_weekday(y, 11, 3, 4),   # Thanksgiving
        }
    return out


def holiday_table(start, end) -> pd.DataFrame:
    """One row per calendar date: date, is_holiday, is_holiday_adjacent."""
    days = pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(), freq="D")
    hols = holiday_dates(range(days.min().year - 1, days.max().year + 2))
    is_hol = pd.Series([d.date() in hols for d in days], index=days)
    adjacent = (is_hol.shift(1, fill_value=False) | is_hol.shift(-1, fill_value=False)) & ~is_hol
    return pd.DataFrame({"date": days, "is_holiday": is_hol.to_numpy(),
                         "is_holiday_adjacent": adjacent.to_numpy()})
