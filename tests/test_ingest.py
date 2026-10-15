"""Offline tests: no network or API key needed."""
import pandas as pd
import pytest

from src.ingest import find_missing_hours, parse_eia_rows, parse_weather


def test_parse_eia_rows_cleans_and_sorts():
    rows = [
        {"period": "2024-01-01T02", "value": "300"},
        {"period": "2024-01-01T01", "value": "200"},
        {"period": "2024-01-01T01", "value": "200"},  # duplicate
    ]
    df = parse_eia_rows(rows)
    assert list(df["demand_mwh"]) == [200, 300]
    assert df["ts"].is_monotonic_increasing
    assert str(df["ts"].dt.tz) == "UTC"


def test_parse_eia_rows_empty_raises():
    with pytest.raises(ValueError):
        parse_eia_rows([])


def test_parse_weather_adds_city_and_utc():
    payload = {"hourly": {
        "time": ["2024-01-01T00:00", "2024-01-01T01:00"],
        "temperature_2m": [1.0, 2.0],
    }}
    df = parse_weather(payload, "philadelphia")
    assert list(df.columns) == ["ts", "city", "temperature_2m"]
    assert (df["city"] == "philadelphia").all()
    assert str(df["ts"].dt.tz) == "UTC"


def test_find_missing_hours():
    ts = pd.Series(pd.to_datetime(
        ["2024-01-01T00", "2024-01-01T01", "2024-01-01T04"], utc=True))
    assert len(find_missing_hours(ts)) == 2
