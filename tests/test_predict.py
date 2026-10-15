import joblib
import numpy as np
import pandas as pd

from src.model import FEATURES
from src.predict import build_bundle, latest_margin, predict_frame


def _frame(n=700):
    rng = np.random.default_rng(0)
    d = pd.DataFrame({c: rng.normal(size=n) for c in FEATURES})
    for c in ("is_weekend", "is_holiday", "is_holiday_adjacent"):
        d[c] = rng.integers(0, 2, n).astype(bool)
    d["ts"] = pd.date_range("2024-01-01", periods=n, freq="h")
    d["demand_mwh"] = 100 + 10 * d["temp_c"] + rng.normal(scale=1.0, size=n)
    d["is_imputed"] = False
    return d


def test_bundle_predicts_with_ordered_band_and_survives_roundtrip(tmp_path):
    d = _frame()
    bundle = build_bundle(d, margin_mwh=5.0)
    path = tmp_path / "bundle.joblib"
    joblib.dump(bundle, path)
    out = predict_frame(joblib.load(path), d.tail(50))
    assert (out["lower_mwh"] <= out["upper_mwh"]).all()
    assert (out["upper_mwh"] - out["lower_mwh"] >= 10.0 - 1e-9).all()   # at least 2 x margin wide
    assert np.abs(out["forecast_mwh"] - d["demand_mwh"].tail(50)).mean() < 5


def test_imputed_rows_are_not_used_for_training():
    d = _frame()
    d.loc[:100, "is_imputed"] = True
    assert build_bundle(d, 0.0)["n_train_hours"] == len(d) - 101


def test_latest_margin_uses_only_recent_months_and_skips_imputed():
    ts = pd.date_range("2025-01-01", periods=24 * 30 * 6, freq="h")
    p = pd.DataFrame({"ts": ts, "demand_mwh": 0.0, "q10": -1.0, "q90": 1.0, "is_imputed": False})
    p.loc[p["ts"] < "2025-03-01", "demand_mwh"] = 100.0        # huge misses, but in old months
    p.loc[p.index[-5:], ["demand_mwh", "is_imputed"]] = [100.0, True]   # huge misses on imputed hours
    # every scored recent point is inside the band by 1 MWh, so the margin is -1 (band tightens)
    assert latest_margin(p, n_months=3) == -1.0


def test_forecast_is_always_inside_its_band_even_on_unusual_inputs():
    d = _frame()
    bundle = build_bundle(d, margin_mwh=-0.5)      # a negative margin tightens the band
    odd = d.tail(50).copy()
    odd["temp_c"] = 500.0                          # far outside anything seen in training
    out = predict_frame(bundle, odd)
    assert (out["lower_mwh"] <= out["forecast_mwh"]).all()
    assert (out["forecast_mwh"] <= out["upper_mwh"]).all()
