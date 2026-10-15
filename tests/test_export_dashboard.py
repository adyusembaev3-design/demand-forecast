import numpy as np
import pandas as pd

from src.export_dashboard import build_hourly, events_local, hour_profile, monthly_summary


def _inputs(n=24 * 70):
    rng = np.random.default_rng(1)
    ts = pd.date_range("2025-01-01", periods=n, freq="h")          # naive UTC, like the pipeline
    actual = 100_000 + 5_000 * np.sin(np.arange(n) / 24 * 2 * np.pi) + rng.normal(0, 500, n)
    pred = pd.DataFrame({"ts": ts, "demand_mwh": actual, "is_imputed": False,
                         "pred": actual + rng.normal(0, 500, n)})
    pred["q10"], pred["q90"] = pred["pred"] - 900, pred["pred"] + 900
    pred["q10_cal"], pred["q90_cal"] = pred["q10"] - 100, pred["q90"] + 100
    pred.loc[pred["ts"] < "2025-02-01", ["q10_cal", "q90_cal"]] = np.nan   # first month uncalibrated
    pred.loc[1500, "demand_mwh"] += 30_000                                  # one obvious anomaly
    pred.loc[10:12, "is_imputed"] = True
    feats = pd.DataFrame({"ts": ts, "temp_c": 10.0, "is_holiday": False,
                          "lag_24": pd.Series(actual).shift(24).bfill()})
    return pred, feats


def test_hourly_has_one_row_per_hour_and_eastern_time():
    pred, feats = _inputs()
    h = build_hourly(pred, feats)
    assert len(h) == len(pred)
    # 2025-01-01 00:00 UTC is 2024-12-31 19:00 Eastern
    assert h["ts_local"].iloc[0] == pd.Timestamp("2024-12-31 19:00")
    assert h["month"].iloc[0] == "2025-01"          # month follows UTC, like the evaluation folds


def test_imputed_hours_have_no_error_and_uncalibrated_month_has_no_coverage():
    pred, feats = _inputs()
    h = build_hourly(pred, feats)
    assert h.loc[10:12, "abs_pct_error"].isna().all()
    assert h.loc[10:12, "is_anomaly"].eq(False).all()
    assert h.loc[:100, "inside_band"].isna().all()          # January: no calibrated band


def test_injected_spike_is_flagged_and_forms_one_event():
    pred, feats = _inputs()
    h = build_hourly(pred, feats)
    assert bool(h.loc[1500, "is_anomaly"])
    ev = events_local(h)
    spike = h.loc[1500, "ts_local"]                      # the injected hour, in Eastern time
    assert ((ev["start_local"] <= spike) & (spike <= ev["end_local"])).any()
    assert {"start_local", "end_local", "hours", "direction"} <= set(ev.columns)


def test_monthly_and_hourly_summaries_have_expected_shape():
    pred, feats = _inputs()
    h = build_hourly(pred, feats)
    m = monthly_summary(h)
    assert m["hours"].sum() == (~h["is_imputed"]).sum()
    assert m["band_coverage_pct"].dropna().between(0, 100).all()
    assert m["model_mape_pct"].gt(0).all()
    p = hour_profile(h)
    assert list(p["hour_local"]) == list(range(24))
