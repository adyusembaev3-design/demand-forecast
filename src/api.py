"""FastAPI service: one hour's features in, a forecast and a calibrated 80% band out.

Self-contained on purpose: it needs only the saved model bundle, not the training code, so the
Docker image stays small. Run locally with:
    uvicorn src.api:app --reload
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field, model_validator

DEFAULT_MODEL_PATH = "models/model_bundle.joblib"
BASE_TEMP_C = 18.3  # base temperature for heating / cooling degrees
BOOL_FEATURES = ("is_weekend", "is_holiday", "is_holiday_adjacent")
# Plausible range for PJM hourly demand (MWh). The model was trained on roughly 60,000-160,000;
# values far outside that are almost certainly input errors, and tree models cannot extrapolate.
LAG_MIN_MWH = 20_000
LAG_MAX_MWH = 250_000


class Features(BaseModel):
    """Inputs for one target hour. Demand lags are in MWh; temperatures in Celsius."""
    hour_local: int = Field(ge=0, le=23, description="Hour of day, US Eastern time")
    dow: int = Field(ge=0, le=6, description="Day of week, Sunday=0 ... Saturday=6")
    month_num: int = Field(ge=1, le=12)
    is_weekend: bool
    is_holiday: bool = False
    is_holiday_adjacent: bool = False
    temp_c: float = Field(ge=-50, le=60, description="Regional average temperature")
    temp_c_max_city: float = Field(ge=-50, le=60)
    temp_c_min_city: float = Field(ge=-50, le=60)
    humidity_pct: float = Field(ge=0, le=100)
    wind_kmh: float = Field(ge=0, le=300)
    cdd: float | None = Field(default=None, ge=0, description="Optional; derived from temp_c if omitted")
    hdd: float | None = Field(default=None, ge=0, description="Optional; derived from temp_c if omitted")
    lag_24: float = Field(ge=LAG_MIN_MWH, le=LAG_MAX_MWH, description="Demand 24 hours before the target hour")
    lag_48: float = Field(ge=LAG_MIN_MWH, le=LAG_MAX_MWH, description="Demand 48 hours before")
    lag_168: float = Field(ge=LAG_MIN_MWH, le=LAG_MAX_MWH, description="Demand 168 hours (one week) before")
    roll24_mean: float = Field(ge=LAG_MIN_MWH, le=LAG_MAX_MWH,
                               description="Mean demand over the 24 hours ending 24h before the target")

    @model_validator(mode="after")
    def weekend_matches_day(self):
        if self.is_weekend != (self.dow in (0, 6)):
            raise ValueError("is_weekend does not match dow (Sunday=0 ... Saturday=6)")
        return self


class Forecast(BaseModel):
    forecast_mwh: float
    lower_mwh: float
    upper_mwh: float
    interval: str = "calibrated ~80% prediction interval"
    model_trained_through: str


def create_app(model_path: str | os.PathLike | None = None) -> FastAPI:
    path = Path(model_path or os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.bundle = joblib.load(path)  # fails fast with a clear error if the file is missing
        yield

    app = FastAPI(title="Demand Forecast API", version="1.0", lifespan=lifespan)

    @app.get("/health")
    def health(request: Request):
        b = request.app.state.bundle
        return {"status": "ok", "model_trained_through": b["trained_through"],
                "n_features": len(b["features"])}

    @app.post("/predict", response_model=Forecast)
    def predict(f: Features, request: Request):
        b = request.app.state.bundle
        row = f.model_dump()
        if row["cdd"] is None:
            row["cdd"] = max(row["temp_c"] - BASE_TEMP_C, 0.0)
        if row["hdd"] is None:
            row["hdd"] = max(BASE_TEMP_C - row["temp_c"], 0.0)
        for c in BOOL_FEATURES:
            row[c] = int(row[c])
        X = pd.DataFrame([row])[b["features"]]
        lo = float(b["models"]["q10"].predict(X)[0])
        hi = float(b["models"]["q90"].predict(X)[0])
        lo, hi = min(lo, hi), max(lo, hi)
        m = b["margin_mwh"]
        point = float(b["models"]["point"].predict(X)[0])
        # The point and quantile models are separate, so on unusual inputs they can disagree.
        # Widen the band if needed so the forecast always lies inside it.
        return Forecast(
            forecast_mwh=point,
            lower_mwh=min(lo - m, point), upper_mwh=max(hi + m, point),
            model_trained_through=b["trained_through"],
        )

    return app


app = create_app()
