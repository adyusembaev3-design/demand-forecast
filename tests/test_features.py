"""Offline test of the SQL pipeline using synthetic data with a known answer."""
import duckdb
import numpy as np
import pandas as pd

from src.build_features import load_raw_tables, run_sql

DAYS = 28
GAP_DAY = 14       # a full missing day
SPIKE_IDX = 10 * 24 + 12  # one bad value on day 10


def _synthetic():
    ts = pd.date_range("2024-01-01", periods=DAYS * 24, freq="h", tz="UTC")
    truth = 100 + 20 * np.sin(2 * np.pi * ts.hour / 24) + 0.5 * ts.dayofweek
    demand = pd.DataFrame({"ts": ts, "demand_mwh": truth.astype(float)})
    demand.loc[SPIKE_IDX, "demand_mwh"] = 20.0
    demand = demand.drop(index=range(GAP_DAY * 24, GAP_DAY * 24 + 24))
    weather = pd.concat([
        pd.DataFrame({"ts": ts, "city": "a", "temperature_2m": 10.0,
                      "relative_humidity_2m": 50.0, "wind_speed_10m": 5.0}),
        pd.DataFrame({"ts": ts, "city": "b", "temperature_2m": 20.0,
                      "relative_humidity_2m": 70.0, "wind_speed_10m": 15.0}),
    ])
    return truth, demand, weather


def _build():
    truth, demand, weather = _synthetic()
    con = duckdb.connect(":memory:")
    load_raw_tables(con, demand, weather)
    run_sql(con)
    return con, truth


def test_gaps_and_spikes_are_repaired_and_flagged():
    con, truth = _build()
    dh = con.execute("SELECT * FROM demand_hourly ORDER BY ts").df()
    assert len(dh) == DAYS * 24
    assert dh["demand_mwh"].notna().all()
    np.testing.assert_allclose(dh["demand_mwh"].to_numpy(), np.asarray(truth))
    assert int(dh["is_suspect"].sum()) == 1
    assert int(dh["is_imputed"].sum()) == 25  # 24 missing hours + 1 suspect value


def test_features_table():
    con, _ = _build()
    f = con.execute("SELECT * FROM features_hourly").df()
    assert len(f) == DAYS * 24 - 168
    assert f["lag_168"].notna().all()
    assert np.allclose(f["temp_c"], 15.0)      # average of the two cities
    assert np.allclose(f["cdd"], 0.0)           # 15C is below the 18.3C base
    assert np.allclose(f["hdd"], 3.3)


def test_holiday_features_are_joined():
    import datetime as dt
    con, _ = _build()
    f = con.execute("SELECT * FROM features_hourly").df()
    assert {"is_holiday", "is_holiday_adjacent"} <= set(f.columns)
    mlk = f[f["ts_local"].dt.date == dt.date(2024, 1, 15)]       # Martin Luther King Jr. Day
    assert len(mlk) > 0 and mlk["is_holiday"].all()
    eve = f[f["ts_local"].dt.date == dt.date(2024, 1, 14)]
    assert eve["is_holiday_adjacent"].all() and not eve["is_holiday"].any()
