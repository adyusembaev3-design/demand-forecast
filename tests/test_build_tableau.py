import xml.etree.ElementTree as ET

import pandas as pd

from src.build_tableau import (BASELINE_LABEL, MODEL_LABEL, SERIES, build_workbook, make_forecast_long,
                               make_monthly_long)


def _hourly(n=48):
    ts = pd.date_range("2025-01-01", periods=n, freq="h")
    d = pd.DataFrame({"ts_local": ts.astype(str), "actual_mwh": 100.0, "forecast_mwh": 101.0,
                      "band_lower_mwh": 99.0, "band_upper_mwh": 103.0})
    d.loc[:5, ["band_lower_mwh", "band_upper_mwh"]] = None      # uncalibrated start
    return d


def test_forecast_long_has_one_row_per_hour_and_series_and_skips_missing():
    long = make_forecast_long(_hourly())
    assert list(long.columns) == ["ts_local", "series", "value"]
    assert set(long["series"]) == set(SERIES.values())
    assert long["value"].notna().all()
    assert len(long) == 48 * 2 + 42 * 2                          # actual+forecast always, band after row 5


def test_monthly_long_stacks_model_and_baseline():
    m = pd.DataFrame({"month": ["2025-01", "2025-02"], "model_mape_pct": [2.0, 2.1], "baseline_mape_pct": [5.0, 4.9]})
    long = make_monthly_long(m)
    assert list(long.columns) == ["month", "series", "mape_pct"]
    assert sorted(long["series"].unique()) == sorted([MODEL_LABEL, BASELINE_LABEL])
    assert len(long) == 4


def test_workbook_is_well_formed_and_references_only_declared_columns():
    root = ET.fromstring(build_workbook("/tmp/some dir/dashboard/data"))
    assert root.tag == "workbook"
    sources = {d.get("name"): d for d in root.find("datasources")}
    assert len(sources) == 3
    sheets = root.find("worksheets")
    assert [w.get("name") for w in sheets] == ["Forecast vs actual", "Monthly error", "Anomaly events"]
    assert [w.get("name") for w in root.find("windows")] == [w.get("name") for w in sheets]
    for ws in sheets:
        dep = ws.find(".//datasource-dependencies")
        src = sources[dep.get("datasource")]
        declared = {c.get("name") for c in src.iter("column")}
        used = {c.get("column").strip("[]") for c in dep.iter("column-instance")}
        assert used <= declared, (ws.get("name"), used - declared)
        # every shelf reference points at an instance that exists
        inst = {c.get("name") for c in dep.iter("column-instance")}
        for shelf in (ws.find(".//rows").text, ws.find(".//cols").text):
            for part in shelf.strip("()").split(" * "):
                assert part.split("].", 1)[1] in inst, part
    # the directory with a space survives XML attribute quoting
    assert any("/tmp/some dir/dashboard/data" in (c.get("directory") or "") for c in root.iter("connection"))


def test_time_axes_are_continuous_not_discrete():
    xml = build_workbook("/x/data")
    assert "[none:ts_local:qk]" in xml and "[none:start_local:qk]" in xml
    assert ":ok]" not in xml                      # no discrete (blue-pill) date instances
