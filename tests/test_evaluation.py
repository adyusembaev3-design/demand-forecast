import numpy as np
import pandas as pd

from src.evaluation import (evaluate_by_fold, evaluate_overall, mae, mape,
                            monthly_folds, rmse)


def test_metrics_known_values():
    y, p = [100, 200], [110, 180]
    assert abs(mape(y, p) - 10.0) < 1e-9
    assert abs(mae(y, p) - 15.0) < 1e-9
    assert abs(rmse(y, p) - np.sqrt((100 + 400) / 2)) < 1e-9


def _frame():
    ts = pd.Series(pd.date_range("2024-01-01", periods=24 * 400, freq="h"))
    df = pd.DataFrame({"ts": ts, "demand_mwh": 100.0, "is_imputed": False})
    return df


def test_monthly_folds_are_time_ordered_and_disjoint():
    df = _frame()
    folds = monthly_folds(df["ts"], n_months=6)
    assert len(folds) == 6
    seen = set()
    for _, train, test in folds:
        assert df["ts"].iloc[train].max() < df["ts"].iloc[test].min()
        assert not (set(test) & seen)
        seen |= set(test)


def test_imputed_hours_are_excluded_from_scoring():
    df = _frame()
    folds = monthly_folds(df["ts"], n_months=3)
    preds = np.full(len(df), 100.0)
    bad = folds[0][2][0]
    preds[bad] = 1000.0                      # a huge error...
    df.loc[bad, "is_imputed"] = True         # ...on an imputed hour
    assert evaluate_overall(df, preds, folds)["mape_pct"] == 0.0
    assert (evaluate_by_fold(df, preds, folds)["mape_pct"] == 0.0).all()
