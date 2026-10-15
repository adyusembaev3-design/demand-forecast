import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import create_app
from src.model import FEATURES
from src.predict import build_bundle

PAYLOAD = {
    "hour_local": 15, "dow": 3, "month_num": 7, "is_weekend": False,
    "is_holiday": False, "is_holiday_adjacent": False,
    "temp_c": 30.0, "temp_c_max_city": 34.0, "temp_c_min_city": 26.0,
    "humidity_pct": 60.0, "wind_kmh": 10.0,
    "lag_24": 120000.0, "lag_48": 118000.0, "lag_168": 119000.0, "roll24_mean": 105000.0,
}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    rng = np.random.default_rng(0)
    d = pd.DataFrame({c: rng.normal(size=600) for c in FEATURES})
    for c in ("is_weekend", "is_holiday", "is_holiday_adjacent"):
        d[c] = rng.integers(0, 2, 600).astype(bool)
    d["ts"] = pd.date_range("2024-01-01", periods=600, freq="h")
    d["demand_mwh"] = 100 + 10 * d["temp_c"] + rng.normal(size=600)
    d["is_imputed"] = False
    path = tmp_path_factory.mktemp("m") / "bundle.joblib"
    joblib.dump(build_bundle(d, margin_mwh=7.0), path)
    with TestClient(create_app(path)) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert r.json()["n_features"] == len(FEATURES)


def test_predict_returns_ordered_band(client):
    r = client.post("/predict", json=PAYLOAD)
    assert r.status_code == 200
    j = r.json()
    assert j["lower_mwh"] <= j["upper_mwh"]
    assert j["upper_mwh"] - j["lower_mwh"] >= 14.0 - 1e-9   # at least 2 x the 7 MWh margin


def test_forecast_always_lies_inside_its_band(client):
    # unusual inputs make the separate point and quantile models disagree; the band must still hold it
    for temp in (-30.0, 0.0, 30.0, 55.0):
        j = client.post("/predict", json={**PAYLOAD, "temp_c": temp, "temp_c_max_city": temp,
                                          "temp_c_min_city": temp}).json()
        assert j["lower_mwh"] <= j["forecast_mwh"] <= j["upper_mwh"]


def test_cdd_hdd_derived_from_temperature_when_omitted(client):
    derived = client.post("/predict", json=PAYLOAD).json()
    explicit = client.post("/predict", json={**PAYLOAD, "cdd": 11.7, "hdd": 0.0}).json()
    assert derived["forecast_mwh"] == pytest.approx(explicit["forecast_mwh"])


def test_invalid_inputs_are_rejected(client):
    assert client.post("/predict", json={**PAYLOAD, "hour_local": 25}).status_code == 422
    assert client.post("/predict", json={**PAYLOAD, "lag_24": -5}).status_code == 422
    # implausible demand values (e.g. 1 MWh) are rejected instead of producing a meaningless forecast
    assert client.post("/predict", json={**PAYLOAD, "lag_24": 1}).status_code == 422
    assert client.post("/predict", json={**PAYLOAD, "roll24_mean": 5_000_000}).status_code == 422
    assert client.post("/predict", json={**PAYLOAD, "dow": 0, "is_weekend": False}).status_code == 422
    missing = {k: v for k, v in PAYLOAD.items() if k != "temp_c"}
    assert client.post("/predict", json=missing).status_code == 422
