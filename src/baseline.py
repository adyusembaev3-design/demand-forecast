"""Step 4: baseline forecasts and the evaluation protocol every later model must beat.

Run from the project root:
    python -m src.baseline
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED, ROOT
from src.evaluation import evaluate_by_fold, evaluate_overall, monthly_folds, score

RESULTS = ROOT / "docs" / "results"
FIGURES = ROOT / "docs" / "figures"


def main() -> None:
    df = pd.read_parquet(DATA_PROCESSED / "features.parquet").sort_values("ts").reset_index(drop=True)
    folds = monthly_folds(df["ts"], n_months=12)
    print(f"{len(folds)} monthly test folds: {folds[0][0]} -> {folds[-1][0]}")

    baselines = {
        "naive_lag24 (same hour yesterday)": df["lag_24"].to_numpy(),
        "seasonal_naive_lag168 (same hour last week)": df["lag_168"].to_numpy(),
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    overall_rows, per_fold = [], []
    for name, preds in baselines.items():
        overall_rows.append({"model": name, **evaluate_overall(df, preds, folds)})
        f = evaluate_by_fold(df, preds, folds)
        f.insert(0, "model", name)
        per_fold.append(f)
    overall = pd.DataFrame(overall_rows)
    per_fold = pd.concat(per_fold, ignore_index=True)

    print("\n== Overall (12 test months, imputed hours excluded) ==")
    print(overall.round(2).to_string(index=False))
    print("\n== MAPE by month (%) ==")
    print(per_fold.pivot(index="fold", columns="model", values="mape_pct").round(2).to_string())

    # Breakdown for the stronger baseline: by local hour of day
    best = overall.sort_values("mape_pct").iloc[0]["model"]
    preds = baselines[best]
    test_idx = np.concatenate([t for _, _, t in folds])
    t = df.iloc[test_idx].copy()
    t["pred"] = preds[test_idx]
    t = t[~t["is_imputed"]]
    by_hour = t.groupby("hour_local").apply(
        lambda g: pd.Series(score(g["demand_mwh"], g["pred"])), include_groups=False
    )
    print(f"\n== MAPE by local hour of day, best baseline: {best} ==")
    print(by_hour["mape_pct"].round(2).to_string())

    overall.to_csv(RESULTS / "baseline_overall.csv", index=False)
    per_fold.to_csv(RESULTS / "baseline_by_month.csv", index=False)
    by_hour.to_csv(RESULTS / "baseline_by_hour.csv")

    # Figure: June 2025 heat wave, where a naive forecast should struggle
    w = df[(df["ts"] >= "2025-06-20") & (df["ts"] < "2025-06-28")]
    if len(w):
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(w["ts"], w["demand_mwh"], label="Actual", linewidth=1.8)
        ax.plot(w["ts"], w["lag_168"], label="Seasonal naive (last week)", linestyle="--")
        ax.set_title("PJM hourly demand during the June 2025 heat wave: actual vs seasonal-naive baseline")
        ax.set_ylabel("MWh (UTC time)")
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(FIGURES / "baseline_heatwave.png", dpi=150)
        print(f"\nSaved {FIGURES / 'baseline_heatwave.png'}")
    print(f"Saved CSVs to {RESULTS}")


if __name__ == "__main__":
    main()
