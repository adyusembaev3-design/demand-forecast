"""Step 12: export tidy CSVs for the Tableau Public dashboard.

Reads the pipeline outputs (data/processed/predictions_calibrated.parquet and features.parquet) and
writes four small CSVs to dashboard/data/. Each is one clean table, so Tableau needs no data prep.

Run from the project root, after src.model and src.calibrate:
    python -m src.export_dashboard

Times: `ts_local` is US Eastern wall-clock time (what a grid operator would read); `ts_utc` is kept
for joining. `month` is the UTC month, matching the evaluation folds, so monthly figures agree with
docs/results. Error metrics skip imputed hours, as everywhere else in the project.
"""
import numpy as np
import pandas as pd

from src.anomaly import MAX_GAP_HOURS, Z_THRESHOLD, find_events, robust_z
from src.config import DATA_PROCESSED, ROOT

OUT = ROOT / "dashboard" / "data"
TZ = "America/New_York"


def to_local(ts: pd.Series) -> pd.Series:
    """Naive-UTC timestamps -> naive US Eastern wall-clock timestamps."""
    return ts.dt.tz_localize("UTC").dt.tz_convert(TZ).dt.tz_localize(None)


def build_hourly(pred: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """One row per test hour: actual, forecast, bands, naive baseline, errors and anomaly flag.

    pred needs: ts, demand_mwh, is_imputed, pred, q10, q90, q10_cal, q90_cal
    feats needs: ts, temp_c, is_holiday, lag_24
    """
    d = pred.merge(feats[["ts", "temp_c", "is_holiday", "lag_24"]], on="ts", how="left")
    d = d.sort_values("ts").reset_index(drop=True)
    local = to_local(d["ts"])

    ok = ~d["is_imputed"]
    resid = (d["demand_mwh"] - d["pred"]).where(ok)
    z = pd.Series(np.nan, index=d.index)
    z[ok] = robust_z(resid[ok])

    inside = ((d["demand_mwh"] >= d["q10_cal"]) & (d["demand_mwh"] <= d["q90_cal"])).where(
        d["q10_cal"].notna() & ok)
    return pd.DataFrame({
        "ts_utc": d["ts"],
        "ts_local": local,
        "date_local": local.dt.date,
        "month": d["ts"].dt.strftime("%Y-%m"),    # UTC month, same grouping as the evaluation folds
        "hour_local": local.dt.hour,
        "day_of_week": local.dt.day_name(),
        "is_holiday": d["is_holiday"].astype(bool),
        "temp_c": d["temp_c"],
        "actual_mwh": d["demand_mwh"],
        "forecast_mwh": d["pred"],
        "baseline_mwh": d["lag_24"],              # naive: same hour yesterday
        "raw_lower_mwh": d["q10"],
        "raw_upper_mwh": d["q90"],
        "band_lower_mwh": d["q10_cal"],           # calibrated 80% band (empty for the first month)
        "band_upper_mwh": d["q90_cal"],
        "error_mwh": resid,                       # actual - forecast
        "abs_pct_error": (resid.abs() / d["demand_mwh"] * 100).where(ok),
        "baseline_abs_pct_error": ((d["demand_mwh"] - d["lag_24"]).abs() / d["demand_mwh"] * 100).where(ok),
        "inside_band": inside,
        "z_score": z,
        "is_anomaly": (z.abs() > Z_THRESHOLD).fillna(False),
        "is_imputed": d["is_imputed"],
    })


def monthly_summary(hourly: pd.DataFrame) -> pd.DataFrame:
    """Per-month accuracy, band coverage and anomaly counts."""
    h = hourly[~hourly["is_imputed"]]
    g = h.groupby("month")
    out = pd.DataFrame({
        "model_mape_pct": g["abs_pct_error"].mean(),
        "baseline_mape_pct": g["baseline_abs_pct_error"].mean(),
        "model_mae_mwh": g["error_mwh"].apply(lambda s: s.abs().mean()),
        "band_coverage_pct": g["inside_band"].apply(lambda s: s.dropna().astype(float).mean() * 100
                                                    if s.notna().any() else np.nan),
        "anomaly_hours": g["is_anomaly"].sum().astype(int),
        "hours": g.size(),
    })
    return out.reset_index()


def hour_profile(hourly: pd.DataFrame) -> pd.DataFrame:
    """Error by hour of day, model vs baseline."""
    h = hourly[~hourly["is_imputed"]]
    g = h.groupby("hour_local")
    return pd.DataFrame({
        "model_mape_pct": g["abs_pct_error"].mean(),
        "baseline_mape_pct": g["baseline_abs_pct_error"].mean(),
        "mean_actual_mwh": g["actual_mwh"].mean(),
    }).reset_index()


def events_local(hourly: pd.DataFrame) -> pd.DataFrame:
    """Anomaly events with Eastern start/end times, grouped as in src.anomaly."""
    h = hourly[~hourly["is_imputed"]]
    frame = pd.DataFrame({"ts": h["ts_utc"], "resid": h["error_mwh"], "z": h["z_score"],
                          "flag": h["is_anomaly"]})
    ev = find_events(frame, MAX_GAP_HOURS)
    for c in ("start", "end"):
        ev[c] = to_local(ev[c])
    return ev.rename(columns={"start": "start_local", "end": "end_local"})


def main() -> None:
    pred = pd.read_parquet(DATA_PROCESSED / "predictions_calibrated.parquet")
    feats = pd.read_parquet(DATA_PROCESSED / "features.parquet")
    hourly = build_hourly(pred, feats)

    OUT.mkdir(parents=True, exist_ok=True)
    tables = {
        "hourly_forecast.csv": hourly,
        "monthly_summary.csv": monthly_summary(hourly),
        "error_by_hour.csv": hour_profile(hourly),
        "anomaly_events.csv": events_local(hourly),
    }
    for name, t in tables.items():
        t.to_csv(OUT / name, index=False)
        print(f"{name:22s} {len(t):>6,} rows x {t.shape[1]} columns")
    print(f"\nSaved to {OUT}")

    ok = hourly[~hourly["is_imputed"]]
    print(f"\nCheck: model MAPE {ok['abs_pct_error'].mean():.2f}%  "
          f"baseline MAPE {ok['baseline_abs_pct_error'].mean():.2f}%  "
          f"(expected about 2.22% and 4.97%)")


if __name__ == "__main__":
    main()
