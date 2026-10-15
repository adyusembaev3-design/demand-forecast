import numpy as np
import pandas as pd

from src.evaluation_extra import coverage, mean_width, pinball
from src.model import FEATURES, fit_predict


def test_interval_metrics():
    assert coverage([1, 2, 3, 4], [0, 0, 0, 0], [2, 2, 2, 2]) == 50.0
    assert mean_width([1, 2], [3, 6]) == 3.0
    assert abs(pinball([10], [8], 0.9) - 1.8) < 1e-9


def _synthetic(n=900):
    rng = np.random.default_rng(0)
    d = pd.DataFrame({c: rng.normal(size=n) for c in FEATURES})
    d["is_weekend"] = rng.integers(0, 2, n).astype(bool)
    d["demand_mwh"] = 100 + 10 * d["temp_c"] + rng.normal(scale=1.0, size=n)
    d["is_imputed"] = False
    return d


def test_fit_predict_learns_signal_and_orders_quantiles():
    d = _synthetic()
    res = fit_predict(d.iloc[:700], d.iloc[700:])
    assert (res["q10"] <= res["q90"]).all()
    err = np.abs(res["pred"] - d["demand_mwh"].iloc[700:]).mean()
    assert err < 0.5 * d["demand_mwh"].std()
