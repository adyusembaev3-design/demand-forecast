# Hourly Electricity Demand Forecasting Platform

End-to-end forecasting project: public demand and weather data -> SQL feature tables -> time-series models with prediction intervals -> anomaly detection and explainability -> Dockerized API on AWS -> monitoring and dashboard.

> **Status:** in progress. See the roadmap below.

## Problem
Forecast hourly electricity demand for the PJM region (Philadelphia area) using calendar and weather features, and quantify uncertainty with prediction intervals.

## Planned architecture
```
EIA API (demand) + Open-Meteo (weather)
        -> Python ingest -> raw Parquet (local + S3)
        -> DuckDB SQL transforms -> hourly feature table
        -> scikit-learn gradient-boosted models (point + quantile) -> MLflow tracking
        -> anomaly detection + SHAP explainability
        -> FastAPI in Docker -> AWS
        -> monitoring + BI dashboard
```

## Roadmap
- [x] Step 1: Repo scaffold and environment
- [ ] Step 2: Ingest demand and weather data
- [ ] Step 3: SQL feature table
- [ ] Step 4: Baseline and time-series backtesting
- [ ] Step 5: Gradient-boosted model and quantile forecasts
- [ ] Step 6: Anomaly detection and SHAP
- [ ] Step 7: MLflow tracking
- [ ] Step 8: FastAPI service and Docker
- [ ] Step 9: AWS deployment (S3, SageMaker/ECS)
- [ ] Step 10: Monitoring, dashboard and write-up

## Setup
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then add your EIA API key
pytest
```

## Results
_To be added._

## Limitations
_To be added (e.g., weather-forecast leakage, holidays, outages)._
