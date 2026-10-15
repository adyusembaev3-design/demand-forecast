"""Build and use the deployable model bundle: point forecast plus a calibrated 80% band."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.calibrate import conformal_adjustment
from src.model import FEATURES, PARAMS, QUANTILES, make_X

CAL_MONTHS = 3


def build_bundle(df: pd.DataFrame, margin_mwh: float) -> dict:
    """Train the point and 10th/90th percentile models on all non-imputed rows of df."""
    train = df[~df["is_imputed"]]
    X, y = make_X(train), train["demand_mwh"]
    models = {"point": HistGradientBoostingRegressor(**PARAMS).fit(X, y)}
    for q in QUANTILES:
        models[f"q{int(q * 100)}"] = HistGradientBoostingRegressor(
            loss="quantile", quantile=q, **PARAMS).fit(X, y)
    return {
        "models": models,
        "features": list(FEATURES),
        "margin_mwh": float(margin_mwh),
        "trained_through": str(train["ts"].max()),
        "n_train_hours": int(len(train)),
    }


def predict_frame(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Forecast and calibrated band for rows of df (must contain the feature columns)."""
    X = make_X(df)[bundle["features"]]
    lo = bundle["models"]["q10"].predict(X)
    hi = bundle["models"]["q90"].predict(X)
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    m = bundle["margin_mwh"]
    point = bundle["models"]["point"].predict(X)
    # Separate models can disagree on unusual inputs; keep the forecast inside its band.
    return pd.DataFrame({
        "forecast_mwh": point,
        "lower_mwh": np.minimum(lo - m, point),
        "upper_mwh": np.maximum(hi + m, point),
    }, index=df.index)


def latest_margin(preds: pd.DataFrame, n_months: int = CAL_MONTHS) -> float:
    """Calibration margin (MWh) from the most recent months of out-of-sample predictions."""
    p = preds[~preds["is_imputed"]]
    months = sorted(p["ts"].dt.to_period("M").unique())[-n_months:]
    cal = p[p["ts"].dt.to_period("M").isin(months)]
    scores = np.maximum(cal["q10"] - cal["demand_mwh"], cal["demand_mwh"] - cal["q90"])
    return conformal_adjustment(scores)
