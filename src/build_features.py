"""Step 3: build clean hourly demand, regional weather and a feature table with DuckDB SQL.

Run from the project root:
    python -m src.build_features
"""
import duckdb
import pandas as pd

from src.calendar_features import holiday_table
from src.config import DATA_PROCESSED, DATA_RAW, DB_PATH, ROOT

SQL_DIR = ROOT / "sql"
SQL_FILES = ["01_demand_hourly.sql", "02_weather_hourly.sql", "03_features_hourly.sql"]
LOCAL_TZ = "America/New_York"


def load_raw_tables(con: duckdb.DuckDBPyConnection, demand: pd.DataFrame, weather: pd.DataFrame) -> None:
    """Load raw frames into DuckDB as naive-UTC tables, plus a complete hourly time grid."""
    demand = demand.copy()
    weather = weather.copy()
    demand["ts"] = demand["ts"].dt.tz_convert("UTC").dt.tz_localize(None)
    weather["ts"] = weather["ts"].dt.tz_convert("UTC").dt.tz_localize(None)

    grid = pd.DataFrame({"ts": pd.date_range(demand["ts"].min(), demand["ts"].max(), freq="h")})
    grid["ts_local"] = (
        grid["ts"].dt.tz_localize("UTC").dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    )

    hol = holiday_table(grid["ts_local"].min(), grid["ts_local"].max())

    for name, df in [("raw_demand", demand), ("raw_weather", weather), ("time_dim", grid), ("holidays", hol)]:
        con.register("tmp_df", df)
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM tmp_df")
        con.unregister("tmp_df")


def _strip_comments(text: str) -> str:
    """Drop full-line SQL comments so semicolons inside comments cannot split statements."""
    return "\n".join(l for l in text.splitlines() if not l.strip().startswith("--"))


def run_sql(con: duckdb.DuckDBPyConnection) -> None:
    """Run the SQL files in order, one statement at a time."""
    for fname in SQL_FILES:
        text = _strip_comments((SQL_DIR / fname).read_text())
        for stmt in text.split(";"):
            if stmt.strip():
                con.execute(stmt)


def report(con: duckdb.DuckDBPyConnection) -> None:
    n, first, last, imputed, suspect = con.execute(
        "SELECT count(*), min(ts), max(ts), sum(is_imputed::INT), sum(is_suspect::INT) FROM demand_hourly"
    ).fetchone()
    print(f"\ndemand_hourly: {n:,} rows, {first} -> {last}")
    print(f"imputed hours: {imputed}   suspect values removed: {suspect}")

    print("\nImputed hours by date:")
    print(con.execute(
        "SELECT CAST(ts AS DATE) AS date, count(*) AS hours FROM demand_hourly "
        "WHERE is_imputed GROUP BY 1 ORDER BY 1"
    ).df().to_string(index=False))

    print("\nSuspect values (raw value vs. imputed replacement):")
    print(con.execute(
        "SELECT ts, demand_raw, demand_mwh FROM demand_hourly WHERE is_suspect ORDER BY ts"
    ).df().to_string(index=False))

    print("\nRemaining nulls in demand_mwh:",
          con.execute("SELECT count(*) FROM demand_hourly WHERE demand_mwh IS NULL").fetchone()[0])

    f = con.execute("SELECT count(*), min(ts), max(ts) FROM features_hourly").fetchone()
    print(f"\nfeatures_hourly: {f[0]:,} rows, {f[1]} -> {f[2]}")
    print("null counts in features:")
    cols = [r[0] for r in con.execute("DESCRIBE features_hourly").fetchall()]
    nulls = con.execute(
        "SELECT " + ", ".join(f"sum(({c} IS NULL)::INT) AS {c}" for c in cols) + " FROM features_hourly"
    ).df().T
    print(nulls[nulls[0] > 0].to_string() if (nulls[0] > 0).any() else "  none")


def main() -> None:
    demand = pd.read_parquet(DATA_RAW / "demand.parquet")
    weather = pd.read_parquet(DATA_RAW / "weather.parquet")

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    load_raw_tables(con, demand, weather)
    run_sql(con)
    report(con)

    out = DATA_PROCESSED / "features.parquet"
    con.execute(f"COPY features_hourly TO '{out}' (FORMAT PARQUET)")
    print(f"\nSaved {out}")
    con.close()


if __name__ == "__main__":
    main()
