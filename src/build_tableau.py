"""Step 13: generate a ready-to-open Tableau workbook (.twb) from the dashboard CSVs.

Run from the project root, after src.export_dashboard:
    python -m src.build_tableau
then open dashboard/demand_forecast.twb in Tableau Public Desktop.

It writes two reshaped CSVs (long format makes one clean line chart possible) and one workbook with
three sheets: forecast vs actual, monthly error vs baseline, and anomaly events. The workbook stores
the absolute path to dashboard/data on THIS computer, so do not commit it; publish from Tableau
Public instead (File > Save to Tableau Public), which packages the data into the workbook.
"""
import uuid
from xml.sax.saxutils import quoteattr

import pandas as pd

from src.config import ROOT

DATA = ROOT / "dashboard" / "data"
OUT = ROOT / "dashboard" / "demand_forecast.twb"

SERIES = {                       # column in hourly_forecast.csv -> label shown in the legend
    "actual_mwh": "Actual",
    "forecast_mwh": "Forecast",
    "band_lower_mwh": "80% band, lower",
    "band_upper_mwh": "80% band, upper",
}
BASELINE_LABEL = "Baseline (same hour yesterday)"
MODEL_LABEL = "Model"


# ---------------------------------------------------------------- reshaped CSVs
def make_forecast_long(hourly: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for col, label in SERIES.items():
        p = hourly[["ts_local", col]].rename(columns={col: "value"}).dropna(subset=["value"])
        p.insert(1, "series", label)
        parts.append(p)
    return pd.concat(parts, ignore_index=True).sort_values(["ts_local", "series"]).reset_index(drop=True)


def make_monthly_long(monthly: pd.DataFrame) -> pd.DataFrame:
    a = monthly[["month", "model_mape_pct"]].rename(columns={"model_mape_pct": "mape_pct"})
    a.insert(1, "series", MODEL_LABEL)
    b = monthly[["month", "baseline_mape_pct"]].rename(columns={"baseline_mape_pct": "mape_pct"})
    b.insert(1, "series", BASELINE_LABEL)
    return pd.concat([a, b], ignore_index=True)


# ---------------------------------------------------------------- workbook XML
ROLE = {"string": ("dimension", "nominal"), "datetime": ("dimension", "ordinal"),
        "real": ("measure", "quantitative"), "integer": ("measure", "quantitative")}
SUFFIX = {"nominal": "nk", "quantitative": "qk", "ordinal": "ok"}


def _q(s: str) -> str:
    return quoteattr(s)


class Source:
    """One CSV file as a Tableau data source."""

    def __init__(self, filename: str, columns: dict[str, str]):
        self.filename, self.columns = filename, columns
        tag = uuid.uuid4().hex[:20]
        self.name = f"federated.{tag}"
        self.conn = f"textscan.{tag}"
        self.table = filename.replace(".", "#")

    def xml(self, directory: str) -> str:
        cols = "\n".join(
            f"          <column datatype={_q(t)} name={_q(c)} ordinal='{i}' />"
            for i, (c, t) in enumerate(self.columns.items()))
        return f"""    <datasource caption={_q(self.filename)} inline='true' name={_q(self.name)} version='18.1'>
      <connection class='federated'>
        <named-connections>
          <named-connection caption={_q(directory.rsplit('/', 1)[-1])} name={_q(self.conn)}>
            <connection class='textscan' directory={_q(directory)} filename={_q(self.filename)} password='' server='' />
          </named-connection>
        </named-connections>
        <relation connection={_q(self.conn)} name={_q(self.filename)} table={_q('[' + self.table + ']')} type='table'>
          <columns character-set='UTF-8' header='yes' locale='en_US' separator=','>
{cols}
          </columns>
        </relation>
      </connection>
      <aliases enabled='yes' />
    </datasource>"""

    def instance(self, col: str, derivation: str) -> tuple[str, str]:
        """Return (instance name, column-instance xml) for a column used in a sheet."""
        _, kind = ROLE[self.columns[col]]
        if derivation == "Sum" or self.columns[col] == "datetime":
            kind = "quantitative"      # datetimes are used as continuous "exact date" axes (green pills)
        name = f"[{derivation.lower()}:{col}:{SUFFIX[kind]}]"
        xml = (f"<column-instance column={_q('[' + col + ']')} derivation={_q(derivation)} "
               f"name={_q(name)} pivot='key' type={_q(kind)} />")
        return name, xml

    def ref(self, inst_name: str) -> str:
        return f"[{self.name}].{inst_name}"


def sheet_xml(title: str, src: Source, cols: list[str], rows: list[str], mark: str,
              color: str, filter_range: tuple[str, str, str] | None = None) -> tuple[str, str]:
    """Return (worksheet xml, window xml). cols/rows/color are plain column names; rows are summed."""
    used = {}                                              # column -> derivation
    for c in cols + [color]:
        used[c] = "None"
    for r in rows:
        used[r] = "Sum"
    deps, names = [], {}
    for c, d in used.items():
        role, typ = ROLE[src.columns[c]]
        deps.append(f"<column datatype={_q(src.columns[c])} name={_q('[' + c + ']')} "
                    f"role={_q(role)} type={_q(typ)} />")
        n, x = src.instance(c, d)
        names[c] = n
        deps.append(x)
    flt = ""
    if filter_range:
        col, lo, hi = filter_range
        flt = (f"<filter class='quantitative' column={_q(src.ref(names[col]))} included-values='in-range'>"
               f"<min>#{lo}#</min><max>#{hi}#</max></filter>")

    def shelf(parts):
        refs = [src.ref(names[p]) for p in parts]
        return refs[0] if len(refs) == 1 else "(" + " * ".join(refs) + ")"

    nl = "\n            "
    ws = f"""    <worksheet name={_q(title)}>
      <table>
        <view>
          <datasources>
            <datasource caption={_q(src.filename)} name={_q(src.name)} />
          </datasources>
          <datasource-dependencies datasource={_q(src.name)}>
            {nl.join(deps)}
          </datasource-dependencies>
          {flt}
          <slices />
          <aggregation value='true' />
        </view>
        <style />
        <panes>
          <pane selection-relaxation-option='selection-relaxation-allow'>
            <view>
              <breakdown value='auto' />
            </view>
            <mark class={_q(mark)} />
            <encodings>
              <color column={_q(src.ref(names[color]))} />
            </encodings>
          </pane>
        </panes>
        <rows>{shelf(rows)}</rows>
        <cols>{shelf(cols)}</cols>
      </table>
    </worksheet>"""
    win = f"""    <window class='worksheet' maximized='true' name={_q(title)}>
      <cards>
        <edge name='left'>
          <strip size='160'>
            <card type='pages' />
            <card type='filters' />
            <card type='marks' />
          </strip>
        </edge>
        <edge name='top'>
          <strip size='2147483647'>
            <card type='columns' />
          </strip>
          <strip size='2147483647'>
            <card type='rows' />
          </strip>
          <strip size='31'>
            <card type='title' />
          </strip>
        </edge>
        <edge name='right'>
          <strip size='160'>
            <card pane-specification-id='0' param={_q('[' + src.name + '].' + names[color])} type='color' />
          </strip>
        </edge>
      </cards>
    </window>"""
    return ws, win


def build_workbook(directory: str) -> str:
    fc = Source("forecast_long.csv", {"ts_local": "datetime", "series": "string", "value": "real"})
    mo = Source("monthly_long.csv", {"month": "string", "series": "string", "mape_pct": "real"})
    an = Source("anomaly_events.csv", {"start_local": "datetime", "end_local": "datetime", "hours": "integer",
                                       "peak_resid_mwh": "real", "peak_z": "real", "direction": "string"})
    s1, w1 = sheet_xml("Forecast vs actual", fc, ["ts_local"], ["value"], "Line", "series",
                       ("ts_local", "2025-06-20 00:00:00", "2025-06-27 23:59:59"))
    s2, w2 = sheet_xml("Monthly error", mo, ["month", "series"], ["mape_pct"], "Bar", "series")
    s3, w3 = sheet_xml("Anomaly events", an, ["start_local"], ["peak_resid_mwh"], "Circle", "direction")
    sources = "\n".join(s.xml(directory) for s in (fc, mo, an))
    return f"""<?xml version='1.0' encoding='utf-8' ?>
<workbook original-version='18.1' source-build='2024.1.0 (20241.24.0312.1146)' source-platform='mac' version='18.1' xmlns:user='http://www.tableausoftware.com/xml/user'>
  <preferences>
    <preference name='ui.encoding.shelf.height' value='24' />
    <preference name='ui.shelf.height' value='26' />
  </preferences>
  <datasources>
{sources}
  </datasources>
  <worksheets>
{s1}
{s2}
{s3}
  </worksheets>
  <windows source-height='30'>
{w1}
{w2}
{w3}
  </windows>
</workbook>
"""


def main() -> None:
    hourly = pd.read_csv(DATA / "hourly_forecast.csv")
    monthly = pd.read_csv(DATA / "monthly_summary.csv")
    make_forecast_long(hourly).to_csv(DATA / "forecast_long.csv", index=False)
    make_monthly_long(monthly).to_csv(DATA / "monthly_long.csv", index=False)
    OUT.write_text(build_workbook(DATA.as_posix()), encoding="utf-8")
    print(f"Wrote {DATA / 'forecast_long.csv'}\nWrote {DATA / 'monthly_long.csv'}\nWrote {OUT}")
    print("\nNext: open the .twb file in Tableau Public Desktop. Do not commit it (it holds a local path).")


if __name__ == "__main__":
    main()
