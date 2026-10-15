"""Step 8: explain what drives the forecast, using SHAP and a permutation-importance cross-check.

The model is trained on data before 2025-07-01 and explained on a random sample of hours from
2025-07 onward (out-of-sample), so the explanation reflects how it behaves on unseen data.

Run from the project root:
    python -m src.explain
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from src.config import DATA_PROCESSED, ROOT
from src.evaluation import mape
from src.model import FEATURES, PARAMS, make_X

RESULTS = ROOT / "docs" / "results"
FIGURES = ROOT / "docs" / "figures"
SPLIT = "2025-07-01"
N_EXPLAIN = 800


def permutation_table(model, X: pd.DataFrame, y, n_repeats: int = 5) -> pd.Series:
    """How much the mean absolute error (MWh) grows when one feature is shuffled."""
    r = permutation_importance(model, X, y, scoring="neg_mean_absolute_error",
                               n_repeats=n_repeats, random_state=0)
    return pd.Series(r.importances_mean, index=X.columns, name="permutation_mae_increase_mwh")


def importance_table(shap_values, X: pd.DataFrame, perm: pd.Series) -> pd.DataFrame:
    """Combine mean |SHAP| (MWh) and permutation importance, sorted by SHAP."""
    out = pd.DataFrame({
        "mean_abs_shap_mwh": np.abs(np.asarray(shap_values)).mean(axis=0),
        "permutation_mae_increase_mwh": perm.reindex(X.columns).to_numpy(),
    }, index=X.columns)
    return out.sort_values("mean_abs_shap_mwh", ascending=False)


def compute_shap(model, X_background: pd.DataFrame, X_explain: pd.DataFrame) -> np.ndarray:
    """SHAP values via the fast tree explainer; falls back to the model-agnostic explainer."""
    import shap
    try:
        return np.asarray(shap.TreeExplainer(model).shap_values(X_explain))
    except Exception as e:  # tree support for this model type can vary by shap version
        print(f"TreeExplainer unavailable ({type(e).__name__}); using the model-agnostic explainer (slower).")
        explainer = shap.Explainer(model.predict, X_background.sample(100, random_state=0),
                                   algorithm="permutation")
        return np.asarray(explainer(X_explain).values)


def main() -> None:
    df = pd.read_parquet(DATA_PROCESSED / "features.parquet").sort_values("ts").reset_index(drop=True)
    df = df[~df["is_imputed"]]
    train, test = df[df["ts"] < SPLIT], df[df["ts"] >= SPLIT]
    Xtr, ytr = make_X(train), train["demand_mwh"]
    model = HistGradientBoostingRegressor(**PARAMS).fit(Xtr, ytr)

    sample = test.sample(min(N_EXPLAIN, len(test)), random_state=0).sort_values("ts")
    Xs, ys = make_X(sample), sample["demand_mwh"]
    print(f"Trained on {len(train):,} hours before {SPLIT}; explaining {len(sample)} out-of-sample hours.")
    print(f"MAPE on the full out-of-sample period: {mape(test['demand_mwh'], model.predict(make_X(test))):.2f}%")

    perm = permutation_table(model, Xs, ys)
    shap_values = compute_shap(model, Xtr, Xs)
    table = importance_table(shap_values, Xs, perm)

    print("\n== Feature importance (MWh) ==")
    print(table.round(0).to_string())

    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS / "feature_importance.csv")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    t = table.iloc[::-1]
    ax[0].barh(t.index, t["mean_abs_shap_mwh"])
    ax[0].set_title("Mean |SHAP| (average push on the forecast, MWh)")
    ax[1].barh(t.index, t["permutation_mae_increase_mwh"], color="tab:orange")
    ax[1].set_title("Permutation importance (error increase, MWh)")
    fig.tight_layout()
    fig.savefig(FIGURES / "feature_importance.png", dpi=150)

    cols = list(Xs.columns)
    s_temp = shap_values[:, cols.index("temp_c")]
    s_hour = pd.Series(shap_values[:, cols.index("hour_local")], index=Xs.index)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].scatter(Xs["temp_c"], s_temp, s=8, alpha=0.6)
    ax[0].axhline(0, color="gray", linewidth=0.6)
    ax[0].set_xlabel("Regional average temperature (C)")
    ax[0].set_ylabel("SHAP effect on forecast (MWh)")
    ax[0].set_title("How temperature moves the forecast")
    by_hour = s_hour.groupby(Xs["hour_local"]).mean()
    ax[1].plot(by_hour.index, by_hour.values, marker="o")
    ax[1].axhline(0, color="gray", linewidth=0.6)
    ax[1].set_xlabel("Hour of day (Eastern)")
    ax[1].set_ylabel("Average SHAP effect (MWh)")
    ax[1].set_title("How hour of day moves the forecast")
    fig.tight_layout()
    fig.savefig(FIGURES / "shap_dependence.png", dpi=150)
    print(f"\nSaved results and figures to {RESULTS} and {FIGURES}")


if __name__ == "__main__":
    main()
