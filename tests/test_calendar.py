import datetime as dt

import pandas as pd

from src.calendar_features import holiday_dates, holiday_table, nth_weekday


def test_floating_holidays_2025():
    assert nth_weekday(2025, 5, 0, -1) == dt.date(2025, 5, 26)   # Memorial Day
    assert nth_weekday(2025, 9, 0, 1) == dt.date(2025, 9, 1)     # Labor Day
    assert nth_weekday(2024, 11, 3, 4) == dt.date(2024, 11, 28)  # Thanksgiving
    assert nth_weekday(2024, 1, 0, 3) == dt.date(2024, 1, 15)    # MLK Day


def test_fixed_holidays_use_actual_date_not_observed():
    h = holiday_dates([2023])
    assert dt.date(2023, 1, 1) in h          # a Sunday; observed Monday is not used
    assert dt.date(2023, 1, 2) not in h


def test_holiday_table_flags_and_adjacent():
    t = holiday_table("2024-11-20", "2024-12-31").set_index("date")
    assert t.loc["2024-11-28", "is_holiday"] and not t.loc["2024-11-28", "is_holiday_adjacent"]
    assert t.loc["2024-11-29", "is_holiday_adjacent"]            # Friday after Thanksgiving
    assert t.loc["2024-12-24", "is_holiday_adjacent"] and t.loc["2024-12-26", "is_holiday_adjacent"]
    assert not t.loc["2024-12-10", ["is_holiday", "is_holiday_adjacent"]].any()
    assert len(t) == (pd.Timestamp("2024-12-31") - pd.Timestamp("2024-11-20")).days + 1
