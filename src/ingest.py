"""Step 2: ingest hourly demand (EIA) and weather (Open-Meteo) into data/raw/.

Run from the project root:
    python -m src.ingest
    python -m src.ingest --start 2023-01-01 --end 2025-12-31
"""
import argparse

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import DATA_RAW, get_env

EIA_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
PAGE_SIZE = 5000

# Large load centers inside the PJM footprint. Weather is averaged across them later in SQL.
CITIES = {
    "philadelphia": (39.95, -75.16),
    "washington_dc": (38.91, -77.04),
    "pittsburgh": (40.44, -79.99),
    "columbus": (39.96, -83.00),
    "chicago": (41.88, -87.63),
}
WEATHER_VARS = "temperature_2m,relative_humidity_2m,wind_speed_10m"


def _session() -> requests.Session:
    """HTTP session that retries on temporary server errors and rate limits."""
    s = requests.Session()
    retry = Retry(total=5, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def parse_eia_rows(rows: list[dict]) -> pd.DataFrame:
    """Turn EIA API rows into a clean hourly frame: ts (UTC), demand_mwh."""
    if not rows:
        raise ValueError("EIA returned no rows. Check the date range, respondent and API key.")
    df = pd.DataFrame(rows)
    out = pd.DataFrame({
        "ts": pd.to_datetime(df["period"], utc=True),
        "demand_mwh": pd.to_numeric(df["value"], errors="coerce"),
    })
    return out.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def fetch_eia_demand(start: str, end: str, respondent: str = "PJM") -> pd.DataFrame:
    """Fetch hourly demand for one balancing authority. start/end are YYYY-MM-DD."""
    key = get_env("EIA_API_KEY")
    session = _session()
    rows, offset = [], 0
    while True:
        params = {
            "api_key": key,
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": respondent,
            "facets[type][]": "D",  # D = demand
            "start": f"{start}T00",
            "end": f"{end}T23",
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": offset,
            "length": PAGE_SIZE,
        }
        r = session.get(EIA_URL, params=params, timeout=60)
        r.raise_for_status()
        batch = r.json()["response"]["data"]
        rows += batch
        if len(batch) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return parse_eia_rows(rows)


def parse_weather(payload: dict, city: str) -> pd.DataFrame:
    """Turn an Open-Meteo response into an hourly frame for one city."""
    h = payload["hourly"]
    df = pd.DataFrame(h)
    df["ts"] = pd.to_datetime(df.pop("time"), utc=True)
    df["city"] = city
    return df[["ts", "city"] + [c for c in df.columns if c not in ("ts", "city")]]


def fetch_weather(start: str, end: str) -> pd.DataFrame:
    """Fetch hourly weather for every city in CITIES (UTC timestamps)."""
    session = _session()
    frames = []
    for city, (lat, lon) in CITIES.items():
        r = session.get(
            WEATHER_URL,
            params={
                "latitude": lat, "longitude": lon,
                "start_date": start, "end_date": end,
                "hourly": WEATHER_VARS, "timezone": "UTC",
            },
            timeout=60,
        )
        r.raise_for_status()
        frames.append(parse_weather(r.json(), city))
    return pd.concat(frames, ignore_index=True)


def find_missing_hours(ts: pd.Series) -> pd.DatetimeIndex:
    """Return the hourly timestamps absent between the first and last timestamp."""
    full = pd.date_range(ts.min(), ts.max(), freq="h")
    return full.difference(pd.DatetimeIndex(ts))


def summarize(name: str, df: pd.DataFrame) -> None:
    """Print a quick data-quality report."""
    print(f"\n== {name} ==")
    print(f"rows: {len(df):,}   range: {df['ts'].min()} -> {df['ts'].max()}")
    print("null counts:")
    print(df.isna().sum().to_string())
    if "city" in df.columns:
        for city, g in df.groupby("city"):
            print(f"  {city}: {len(g):,} rows, {len(find_missing_hours(g['ts']))} missing hours")
    else:
        print(f"missing hours: {len(find_missing_hours(df['ts']))}")


def main() -> None:
    p = argparse.ArgumentParser(description="Ingest EIA demand and Open-Meteo weather.")
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--respondent", default="PJM")
    args = p.parse_args()

    DATA_RAW.mkdir(parents=True, exist_ok=True)

    demand = fetch_eia_demand(args.start, args.end, args.respondent)
    demand.to_parquet(DATA_RAW / "demand.parquet", index=False)
    summarize("demand", demand)

    weather = fetch_weather(args.start, args.end)
    weather.to_parquet(DATA_RAW / "weather.parquet", index=False)
    summarize("weather", weather)

    print("\nSaved data/raw/demand.parquet and data/raw/weather.parquet")


if __name__ == "__main__":
    main()
