"""Metrics and time-series cross-validation folds shared by every model in this project."""
import numpy as np
import pandas as pd


def mape(y_true, y_pred) -> float:
    """Mean absolute percentage error, in percent."""
    y, p = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y - p) / y)) * 100)


def mae(y_true, y_pred) -> float:
    y, p = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y - p)))


def rmse(y_true, y_pred) -> float:
    y, p = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def score(y_true, y_pred) -> dict:
    return {"mape_pct": mape(y_true, y_pred), "mae_mwh": mae(y_true, y_pred),
            "rmse_mwh": rmse(y_true, y_pred), "n": int(len(y_true))}


def monthly_folds(ts: pd.Series, n_months: int = 12):
    """Rolling-origin folds: test on each of the last n months, train on everything before it.

    Returns a list of (month_label, train_positions, test_positions).
    """
    periods = ts.dt.to_period("M")
    months = sorted(periods.unique())[-n_months:]
    folds = []
    for m in months:
        test_idx = np.flatnonzero((periods == m).to_numpy())
        train_idx = np.flatnonzero((ts < m.start_time).to_numpy())
        folds.append((str(m), train_idx, test_idx))
    return folds


def evaluate_by_fold(df: pd.DataFrame, preds, folds, target: str = "demand_mwh") -> pd.DataFrame:
    """Score predictions on each fold. Imputed hours are excluded from scoring."""
    preds = np.asarray(preds, dtype=float)
    y = df[target].to_numpy(dtype=float)
    keep_all = ~df["is_imputed"].to_numpy(dtype=bool)
    rows = []
    for name, _, test_idx in folds:
        keep = test_idx[keep_all[test_idx]]
        rows.append({"fold": name, **score(y[keep], preds[keep])})
    return pd.DataFrame(rows)


def evaluate_overall(df: pd.DataFrame, preds, folds, target: str = "demand_mwh") -> dict:
    """Score predictions pooled over all test folds (imputed hours excluded)."""
    preds = np.asarray(preds, dtype=float)
    idx = np.concatenate([test for _, _, test in folds])
    keep = idx[~df["is_imputed"].to_numpy(dtype=bool)[idx]]
    return score(df[target].to_numpy(dtype=float)[keep], preds[keep])
