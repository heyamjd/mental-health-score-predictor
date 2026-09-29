"""
ml/train.py
============
End-to-end training script for the Mental Health Score Predictor.

Steps:
  1. Generate synthetic data (if raw CSV is missing)
  2. Load and clean data
  3. IQR outlier removal on training split only (no leakage)
  4. Train Linear Regression (baseline)
  5. Tune Random Forest with RandomizedSearchCV
  6. Evaluate both models on held-out test set
  7. Auto-select the best model by R²
  8. Persist: pipeline.joblib, metrics.json, metadata.json

Run with:
    python ml/train.py
    # or
    make train
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)

from ml.config import (
    ALL_FEATURE_COLS,
    ALLOWED_COUNTRIES,
    ALLOWED_PLATFORMS,
    CV_FOLDS,
    FIGURES_DIR,
    METADATA_PATH,
    METRICS_PATH,
    N_ITER_SEARCH,
    NUMERIC_COLS,
    NUMERIC_RANGES,
    ORDINAL_CATEGORIES,
    ORDINAL_COL,
    PIPELINE_PATH,
    PROCESSED_CSV,
    RANDOM_STATE,
    RAW_CSV,
    RF_PARAM_DIST,
    SCORE_BANDS,
    SCORE_MESSAGES,
    TARGET_COL,
    TEST_SIZE,
)
from ml.data_cleaning import clean, iqr_outlier_removal, load_raw, report_missing
from ml.evaluate import (
    print_comparison_table,
    regression_metrics,
    save_comparison_plot,
    save_metrics,
)
from ml.features import build_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: Ensure raw data exists
# ---------------------------------------------------------------------------

def ensure_data() -> None:
    if not RAW_CSV.exists():
        logger.info("No raw CSV found — generating synthetic data …")
        import subprocess
        subprocess.run(
            [sys.executable, "scripts/generate_synthetic_data.py"],
            check=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
    else:
        logger.info("Raw CSV found: %s", RAW_CSV)


# ---------------------------------------------------------------------------
# Step 2-3: Load, clean, split, IQR
# ---------------------------------------------------------------------------

def load_and_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    raw = load_raw()
    report_missing(raw)
    df = clean(raw)

    # Save processed CSV
    PROCESSED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_CSV, index=False)

    X = df[ALL_FEATURE_COLS]
    y = df[TARGET_COL]

    # ⚠️  Split BEFORE any IQR removal — IQR is computed only on training rows
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logger.info("Train size: %d  |  Test size: %d", len(X_train), len(X_test))

    # IQR removal on training set only
    train_df = X_train.copy()
    train_df[TARGET_COL] = y_train.values

    train_df_clean, iqr_report = iqr_outlier_removal(train_df, NUMERIC_COLS)
    logger.info("IQR removal report: %s", iqr_report)
    logger.info("Training rows after IQR: %d (removed %d)",
                len(train_df_clean), len(train_df) - len(train_df_clean))

    X_train_clean = train_df_clean[ALL_FEATURE_COLS]
    y_train_clean = train_df_clean[TARGET_COL]

    return X_train_clean, X_test, y_train_clean, y_test


# ---------------------------------------------------------------------------
# Step 4: Linear Regression baseline
# ---------------------------------------------------------------------------

def train_linear_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> object:
    logger.info("Training Linear Regression …")
    lr_pipeline = build_pipeline(LinearRegression())
    lr_pipeline.fit(X_train, y_train)

    cv_scores = cross_val_score(
        lr_pipeline, X_train, y_train, cv=CV_FOLDS, scoring="r2", n_jobs=-1
    )
    logger.info("  LR CV R² scores: %s  (mean=%.4f)", cv_scores.round(4), cv_scores.mean())
    return lr_pipeline, float(cv_scores.mean())


# ---------------------------------------------------------------------------
# Step 5: Random Forest + RandomizedSearchCV
# ---------------------------------------------------------------------------

def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[object, float]:
    logger.info("Tuning Random Forest with RandomizedSearchCV (n_iter=%d, cv=%d) …",
                N_ITER_SEARCH, CV_FOLDS)

    base_pipeline = build_pipeline(RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))

    search = RandomizedSearchCV(
        estimator=base_pipeline,
        param_distributions=RF_PARAM_DIST,
        n_iter=N_ITER_SEARCH,
        cv=CV_FOLDS,
        scoring="r2",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    search.fit(X_train, y_train)

    logger.info("  Best RF params: %s", search.best_params_)
    logger.info("  Best RF CV R²: %.4f", search.best_score_)
    return search.best_estimator_, float(search.best_score_)


# ---------------------------------------------------------------------------
# Step 6: Evaluate and persist
# ---------------------------------------------------------------------------

def evaluate_and_persist(
    lr_pipeline: object,
    rf_pipeline: object,
    lr_cv: float,
    rf_cv: float,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> None:
    y_pred_lr = lr_pipeline.predict(X_test)
    y_pred_rf = rf_pipeline.predict(X_test)

    # Clip predictions to valid range
    y_pred_lr = np.clip(y_pred_lr, 0, 100)
    y_pred_rf = np.clip(y_pred_rf, 0, 100)

    lr_metrics = regression_metrics(y_test, y_pred_lr)
    rf_metrics = regression_metrics(y_test, y_pred_rf)

    lr_metrics["cv_r2"] = lr_cv
    rf_metrics["cv_r2"] = rf_cv

    print_comparison_table(lr_metrics, rf_metrics)

    # Auto-select best model by test R²
    best_name = "Random Forest" if rf_metrics["r2"] >= lr_metrics["r2"] else "Linear Regression"
    best_pipeline = rf_pipeline if best_name == "Random Forest" else lr_pipeline
    best_metrics  = rf_metrics  if best_name == "Random Forest" else lr_metrics

    logger.info("[BEST] Best model: %s  (R2=%.4f, MAE=%.4f, RMSE=%.4f)",
                best_name, best_metrics["r2"], best_metrics["mae"], best_metrics["rmse"])

    # Save predicted-vs-actual plot
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save_comparison_plot(np.array(y_test), y_pred_lr, y_pred_rf, FIGURES_DIR)

    # Save full metrics
    all_metrics = {
        "best_model": best_name,
        "linear_regression": lr_metrics,
        "random_forest": rf_metrics,
    }
    save_metrics(all_metrics, METRICS_PATH)

    # Save pipeline artifact
    PIPELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, PIPELINE_PATH)
    logger.info("Saved pipeline → %s", PIPELINE_PATH)

    # Build numeric ranges from training data (for API validation)
    numeric_ranges_from_data: dict = {}
    for col in NUMERIC_COLS:
        lo, hi = NUMERIC_RANGES[col]
        numeric_ranges_from_data[col] = {"min": lo, "max": hi}

    # Save metadata (consumed by the API /metadata endpoint)
    metadata = {
        "model_name":      best_name,
        "training_date":   datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "features":        ALL_FEATURE_COLS,
        "numeric_features": NUMERIC_COLS,
        "numeric_ranges":  numeric_ranges_from_data,
        "ordinal_feature": ORDINAL_COL,
        "ordinal_categories": ORDINAL_CATEGORIES,
        "nominal_features": list(NUMERIC_COLS),   # kept for schema completeness
        "allowed_countries":  ALLOWED_COUNTRIES,
        "allowed_platforms":  ALLOWED_PLATFORMS,
        "score_bands":     SCORE_BANDS,
        "score_messages":  SCORE_MESSAGES,
        "metrics":         best_metrics,
        "is_synthetic_data": True,
        "data_warning": (
            "⚠️  Trained on SYNTHETIC data. Results are for demonstration only. "
            "Replace data/raw/mental_health_data.csv with a real dataset and run `make train`."
        ),
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved metadata → %s", METADATA_PATH)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=" * 60)
    logger.info("  Mental Health Score Predictor - Training Script")
    logger.info("=" * 60)

    ensure_data()
    X_train, X_test, y_train, y_test = load_and_split()
    lr_pipeline, lr_cv = train_linear_regression(X_train, y_train)
    rf_pipeline, rf_cv = train_random_forest(X_train, y_train)
    evaluate_and_persist(
        lr_pipeline, rf_pipeline, lr_cv, rf_cv,
        X_test, y_test, X_train, y_train
    )
    logger.info("[DONE] Training complete!")


if __name__ == "__main__":
    main()
