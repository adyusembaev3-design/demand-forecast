"""Step 5: gradient-boosted point forecast plus 10th/90th percentile interval forecasts.

Each of the 12 monthly folds trains only on data before that month (expanding window).

Run from the project root:
    python -m src.model

Caveat: features use observed weather. In production you would use weather forecasts,
so real-world error would be somewhat higher. Documented in the README.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.config import DATA_PROCESSED, ROOT
from src.evaluation import evaluate_by_fold, evaluate_overall, monthly_folds
from src.evaluation_extra import coverage, mean_width, pinball

RESULTS = ROOT / "docs" / "results"
FIGURES = ROOT / "docs" / "figures"

FEATURES = [
    "hour_local", "dow", "month_num", "is_weekend", "is_holiday", "is_holiday_adjacent",
    "temp_c", "temp_c_max_city", "temp_c_min_city", "humidity_pct", "wind_kmh",
    "cdd", "hdd", "lag_24", "lag_48", "lag_168", "roll24_mean",
]
QUANTILES = (0.1, 0.9)
PARAMS = dict(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=40, random_state=0)


def make_X(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES].copy()
    for c in ("is_weekend", "is_holiday", "is_holiday_adjacent"):
        X[c] = X[c].astype(int)
    return X


def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Fit point and quantile models on train (non-imputed rows) and predict test."""
    train = train[~train["is_imputed"]]
    Xtr, ytr, Xte = make_X(train), train["demand_mwh"], make_X(test)
    out = {"pred": HistGradientBoostingRegressor(**PARAMS).fit(Xtr, ytr).predict(Xte)}
    for q in QUANTILES:
        m = HistGradientBoostingRegressor(loss="quantile", quantile=q, **PARAMS).fit(Xtr, ytr)
        out[f"q{int(q * 100)}"] = m.predict(Xte)
    res = pd.DataFrame(out, index=test.index)
    lo, hi = np.minimum(res["q10"], res["q90"]), np.maximum(res["q10"], res["q90"])  # no crossing
    res["q10"], res["q90"] = lo, hi
    return res


def main() -> None:
    df = pd.read_parquet(DATA_PROCESSED / "features.parquet").sort_values("ts").reset_index(drop=True)
    folds = monthly_folds(df["ts"], n_months=12)

    preds = pd.DataFrame(index=df.index, columns=["pred", "q10", "q90"], dtype=float)
    for name, tr, te in folds:
        preds.loc[df.index[te]] = fit_predict(df.iloc[tr], df.iloc[te])
        print(f"fitted fold {name}")

    overall = evaluate_overall(df, preds["pred"], folds)
    by_month = evaluate_by_fold(df, preds["pred"], folds)

    idx = np.concatenate([t for _, _, t in folds])
    idx = idx[~df["is_imputed"].to_numpy()[idx]]
    y = df["demand_mwh"].to_numpy()[idx]
    lo, hi = preds["q10"].to_numpy()[idx], preds["q90"].to_numpy()[idx]
    interval = {
        "coverage_80_pct": coverage(y, lo, hi),
        "mean_width_mwh": mean_width(lo, hi),
        "pinball_q10": pinball(y, lo, 0.1),
        "pinball_q90": pinball(y, hi, 0.9),
    }

    print("\n== Gradient boosting, overall (12 test months) ==")
    print({k: round(v, 2) for k, v in overall.items()})
    print("Baseline to beat (same hour yesterday): mape 4.97, mae 4850.18")
    print("\n== 10th-90th percentile interval (nominal coverage 80%) ==")
    print({k: round(v, 2) for k, v in interval.items()})
    print("\n== MAPE by month (%) ==")
    print(by_month[["fold", "mape_pct"]].round(2).to_string(index=False))

    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"model": "hist_gradient_boosting", **overall, **interval}]).to_csv(
        RESULTS / "gbm_overall.csv", index=False)
    by_month.to_csv(RESULTS / "gbm_by_month.csv", index=False)
    out = df[["ts", "demand_mwh", "is_imputed"]].join(preds)
    out[out["pred"].notna()].to_parquet(DATA_PROCESSED / "predictions.parquet", index=False)

    w = out[(out["ts"] >= "2025-06-20") & (out["ts"] < "2025-06-28")]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(w["ts"], w["q10"], w["q90"], alpha=0.25, label="10th-90th percentile")
    ax.plot(w["ts"], w["demand_mwh"], label="Actual", linewidth=1.8)
    ax.plot(w["ts"], w["pred"], label="Forecast", linestyle="--")
    ax.set_title("June 2025 heat wave: forecast with 80% prediction interval")
    ax.set_ylabel("MWh (UTC time)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGURES / "gbm_heatwave.png", dpi=150)
    print(f"\nSaved results to {RESULTS} and {FIGURES / 'gbm_heatwave.png'}")


if __name__ == "__main__":
    main()
