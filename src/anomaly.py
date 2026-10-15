"""Step 6b: flag hours where actual demand departs sharply from the forecast, and group them into events.

An hour is anomalous if its forecast error is more than Z_THRESHOLD robust standard deviations
(median/MAD based, so the big errors do not inflate their own yardstick). Imputed hours are skipped.

Run from the project root:
    python -m src.anomaly
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED, ROOT

RESULTS = ROOT / "docs" / "results"
FIGURES = ROOT / "docs" / "figures"
Z_THRESHOLD = 4.0
MAX_GAP_HOURS = 2


def robust_z(resid: pd.Series) -> pd.Series:
    med = resid.median()
    sigma = 1.4826 * (resid - med).abs().median()
    return (resid - med) / sigma


def find_events(df: pd.DataFrame, max_gap_hours: int = MAX_GAP_HOURS) -> pd.DataFrame:
    """Group flagged hours (df must have ts, resid, z, flag) into events separated by > max_gap_hours."""
    f = df[df["flag"]].sort_values("ts")
    if f.empty:
        return pd.DataFrame(columns=["start", "end", "hours", "peak_resid_mwh", "peak_z", "direction"])
    gap = f["ts"].diff() > pd.Timedelta(hours=max_gap_hours)
    f = f.assign(event=gap.cumsum())
    rows = []
    for _, g in f.groupby("event"):
        peak = g.loc[g["z"].abs().idxmax()]
        rows.append({
            "start": g["ts"].min(), "end": g["ts"].max(), "hours": len(g),
            "peak_resid_mwh": peak["resid"], "peak_z": peak["z"],
            "direction": "demand above forecast" if peak["resid"] > 0 else "demand below forecast",
        })
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_parquet(DATA_PROCESSED / "predictions.parquet").sort_values("ts").reset_index(drop=True)
    df = df[~df["is_imputed"]].copy()
    df["resid"] = df["demand_mwh"] - df["pred"]
    df["z"] = robust_z(df["resid"])
    df["flag"] = df["z"].abs() > Z_THRESHOLD

    events = find_events(df)
    print(f"{int(df['flag'].sum())} anomalous hours of {len(df):,} ({df['flag'].mean() * 100:.2f}%), "
          f"grouped into {len(events)} events")

    if len(events):
        show = events.copy()
        for c in ("start", "end"):
            show[c] = show[c].dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d %H:%M")
        top = show.reindex(events["peak_z"].abs().sort_values(ascending=False).index).head(15)
        print("\n== Top events by size (Eastern time) ==")
        print(top.round(1).to_string(index=False))

    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    events.to_csv(RESULTS / "anomaly_events.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(df["ts"], df["resid"], linewidth=0.4, label="Forecast error (actual - forecast)")
    fl = df[df["flag"]]
    ax.scatter(fl["ts"], fl["resid"], s=8, color="red", label=f"Flagged (|z| > {Z_THRESHOLD:g})", zorder=3)
    ax.axhline(0, color="gray", linewidth=0.6)
    ax.set_title("Forecast error and flagged anomalous hours, 2025")
    ax.set_ylabel("MWh")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGURES / "anomalies.png", dpi=150)
    print(f"\nSaved results and {FIGURES / 'anomalies.png'}")


if __name__ == "__main__":
    main()
