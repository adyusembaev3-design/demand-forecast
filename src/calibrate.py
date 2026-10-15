"""Step 6a: conformal calibration of the 10th-90th percentile band.

For each month, the band is widened by a margin learned from the preceding months' out-of-sample
errors (those predictions came from models that had not seen those months, so they are honest).
The first test month has no history, so it is left uncalibrated and excluded from the comparison.

Run from the project root:
    python -m src.calibrate
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED, ROOT
from src.evaluation_extra import coverage, mean_width

RESULTS = ROOT / "docs" / "results"
FIGURES = ROOT / "docs" / "figures"


def conformal_adjustment(scores, alpha: float = 0.2) -> float:
    """Margin that makes the band cover (1 - alpha) of past points (finite-sample corrected)."""
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))   # the k-th smallest past error
    return float(s[min(k, n) - 1])


def calibrate_intervals(df: pd.DataFrame, n_cal_months: int = 3, alpha: float = 0.2) -> pd.DataFrame:
    """Add q10_cal, q90_cal and margin columns (NaN for the first month)."""
    df = df.copy()
    df["month"] = df["ts"].dt.to_period("M")
    months = sorted(df["month"].unique())
    df["q10_cal"], df["q90_cal"], df["margin"] = np.nan, np.nan, np.nan
    for i, m in enumerate(months):
        if i == 0:
            continue
        prior = months[max(0, i - n_cal_months):i]
        cal = df[df["month"].isin(prior) & ~df["is_imputed"]]
        scores = np.maximum(cal["q10"] - cal["demand_mwh"], cal["demand_mwh"] - cal["q90"])
        margin = conformal_adjustment(scores, alpha)
        rows = df["month"] == m
        df.loc[rows, "margin"] = margin
        df.loc[rows, "q10_cal"] = df.loc[rows, "q10"] - margin
        df.loc[rows, "q90_cal"] = df.loc[rows, "q90"] + margin
    return df


def main() -> None:
    df = pd.read_parquet(DATA_PROCESSED / "predictions.parquet").sort_values("ts").reset_index(drop=True)
    cal = calibrate_intervals(df)
    ev = cal[cal["q10_cal"].notna() & ~cal["is_imputed"]]
    y = ev["demand_mwh"]

    summary = pd.DataFrame([
        {"band": "raw 10th-90th", "coverage_pct": coverage(y, ev["q10"], ev["q90"]),
         "mean_width_mwh": mean_width(ev["q10"], ev["q90"])},
        {"band": "calibrated", "coverage_pct": coverage(y, ev["q10_cal"], ev["q90_cal"]),
         "mean_width_mwh": mean_width(ev["q10_cal"], ev["q90_cal"])},
    ])
    print(f"== Interval calibration (target coverage 80%), {ev['month'].nunique()} months ==")
    print(summary.round(1).to_string(index=False))

    by_month = ev.groupby("month").apply(
        lambda g: pd.Series({
            "raw_cov": coverage(g["demand_mwh"], g["q10"], g["q90"]),
            "cal_cov": coverage(g["demand_mwh"], g["q10_cal"], g["q90_cal"]),
            "margin_mwh": g["margin"].iloc[0],
        }), include_groups=False)
    print("\n== Coverage by month (%) ==")
    print(by_month.round(1).to_string())

    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS / "interval_calibration.csv", index=False)
    by_month.to_csv(RESULTS / "interval_calibration_by_month.csv")
    cal.drop(columns="month").to_parquet(DATA_PROCESSED / "predictions_calibrated.parquet", index=False)

    w = cal[(cal["ts"] >= "2025-06-20") & (cal["ts"] < "2025-06-28")]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(w["ts"], w["q10_cal"], w["q90_cal"], alpha=0.2, label="Calibrated 80% band")
    ax.fill_between(w["ts"], w["q10"], w["q90"], alpha=0.35, label="Raw 10th-90th band")
    ax.plot(w["ts"], w["demand_mwh"], label="Actual", linewidth=1.8)
    ax.set_title("June 2025 heat wave: raw vs calibrated prediction band")
    ax.set_ylabel("MWh (UTC time)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGURES / "calibrated_heatwave.png", dpi=150)
    print(f"\nSaved results and {FIGURES / 'calibrated_heatwave.png'}")


if __name__ == "__main__":
    main()
