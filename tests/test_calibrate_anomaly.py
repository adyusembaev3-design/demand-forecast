import numpy as np
import pandas as pd

from src.anomaly import find_events, robust_z
from src.calibrate import calibrate_intervals, conformal_adjustment
from src.evaluation_extra import coverage


def test_conformal_adjustment_known_value():
    assert conformal_adjustment(list(range(1, 100)), alpha=0.2) == 80


def test_calibration_widens_a_too_narrow_band():
    rng = np.random.default_rng(1)
    ts = pd.date_range("2025-01-01", periods=24 * 30 * 6, freq="h")
    df = pd.DataFrame({"ts": ts, "demand_mwh": rng.normal(size=len(ts)),
                       "is_imputed": False, "q10": -0.3, "q90": 0.3})
    cal = calibrate_intervals(df)
    ev = cal[cal["q10_cal"].notna()]
    raw = coverage(ev["demand_mwh"], ev["q10"], ev["q90"])
    fixed = coverage(ev["demand_mwh"], ev["q10_cal"], ev["q90_cal"])
    assert raw < 40 and 74 < fixed < 86
    assert cal["q10_cal"].iloc[:24].isna().all()  # first month stays uncalibrated


def test_find_events_groups_consecutive_hours():
    ts = pd.date_range("2025-01-01", periods=30, freq="h")
    resid = np.zeros(30)
    resid[[3, 4, 5, 20]] = [10, 12, -8, -15]
    df = pd.DataFrame({"ts": ts, "resid": resid})
    df["z"] = resid
    df["flag"] = np.abs(resid) > 5
    ev = find_events(df)
    assert list(ev["hours"]) == [3, 1]
    assert ev["direction"].iloc[1] == "demand below forecast"


def test_robust_z_ignores_outliers_in_scale():
    r = pd.Series([0.0, 1, -1, 1, -1, 0, 1, -1, 1000])
    assert abs(robust_z(r).iloc[-1]) > 100
