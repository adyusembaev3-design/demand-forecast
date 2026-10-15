"""Step 9: train the final model on all data, save the bundle, and log the run to MLflow.

Run from the project root:
    python -m src.train
Then browse runs with:
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
import joblib
import mlflow
import pandas as pd

from src.config import DATA_PROCESSED, ROOT
from src.model import FEATURES, PARAMS
from src.predict import CAL_MONTHS, build_bundle, latest_margin, predict_frame

RESULTS = ROOT / "docs" / "results"
FIGURES = ROOT / "docs" / "figures"
MODELS = ROOT / "models"
TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"


def log_csv_metrics(path, prefix: str, label_col: str | None = None) -> None:
    """Log every numeric column of a small results CSV as MLflow metrics."""
    if not path.exists():
        return
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        tag = f"{prefix}_{str(row[label_col]).split(' ')[0]}" if label_col else prefix
        for col in df.select_dtypes("number").columns:
            mlflow.log_metric(f"{tag}_{col}", float(row[col]))


def main() -> None:
    df = pd.read_parquet(DATA_PROCESSED / "features.parquet").sort_values("ts").reset_index(drop=True)
    preds = pd.read_parquet(DATA_PROCESSED / "predictions.parquet")
    margin = latest_margin(preds)

    bundle = build_bundle(df, margin)
    MODELS.mkdir(exist_ok=True)
    bundle_path = MODELS / "model_bundle.joblib"
    joblib.dump(bundle, bundle_path)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("demand-forecast")
    with mlflow.start_run(run_name="final_model"):
        mlflow.log_params({**{f"hgb_{k}": v for k, v in PARAMS.items()},
                           "n_features": len(FEATURES),
                           "trained_through": bundle["trained_through"],
                           "n_train_hours": bundle["n_train_hours"],
                           "calibration_months": CAL_MONTHS})
        mlflow.log_metric("calibration_margin_mwh", margin)
        log_csv_metrics(RESULTS / "gbm_overall.csv", "backtest_gbm")
        log_csv_metrics(RESULTS / "baseline_overall.csv", "backtest_baseline", label_col="model")
        log_csv_metrics(RESULTS / "interval_calibration.csv", "interval", label_col="band")
        mlflow.log_artifact(str(bundle_path))
        for f in [RESULTS / "feature_importance.csv", RESULTS / "gbm_by_month.csv",
                  FIGURES / "feature_importance.png", FIGURES / "shap_dependence.png",
                  FIGURES / "calibrated_heatwave.png"]:
            if f.exists():
                mlflow.log_artifact(str(f))
        print(f"Logged run {mlflow.active_run().info.run_id} to {TRACKING_URI}")

    sample = df.tail(3)
    out = predict_frame(bundle, sample)
    print(f"\nSaved {bundle_path}  (calibration margin {margin:,.0f} MWh)")
    print("In-sample sanity check (last 3 hours; the model has seen these, so this is not a test):")
    print(pd.concat([sample[["ts", "demand_mwh"]], out], axis=1)
          .round({"demand_mwh": 0, "forecast_mwh": 0, "lower_mwh": 0, "upper_mwh": 0}).to_string(index=False))


if __name__ == "__main__":
    main()
