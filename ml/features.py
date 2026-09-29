"""
ml/features.py
===============
Feature engineering and sklearn Pipeline construction.

Pipeline structure
------------------
ColumnTransformer
├── numeric_pipeline
│   ├── ClipTransformer   – hard-clips to valid range (prevents leakage of test extremes)
│   ├── Log1pTransformer  – log(1+x) on right-skewed columns
│   └── StandardScaler    – zero mean, unit variance (helps Linear Regression)
├── ordinal_pipeline
│   └── OrdinalEncoder    – Stress_Level: Low=0, Medium=1, High=2
└── nominal_pipeline
    └── OneHotEncoder      – Country, Platform (handle_unknown="ignore")

Then → estimator (LinearRegression or RandomForest)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import (
    NOMINAL_COLS,
    NUMERIC_COLS,
    NUMERIC_RANGES,
    ORDINAL_CATEGORIES,
    ORDINAL_COL,
)

# ---------------------------------------------------------------------------
# Custom transformer: clip numeric columns to their valid ranges
# ---------------------------------------------------------------------------

class ClipTransformer(BaseEstimator, TransformerMixin):
    """
    Clip each column to [lo, hi] defined in NUMERIC_RANGES.
    This is placed FIRST in the numeric sub-pipeline so that downstream
    steps (log1p, scaler) never receive values outside valid bounds.
    """

    def __init__(self, ranges: dict[str, tuple[float, float]] | None = None) -> None:
        self.ranges = ranges or NUMERIC_RANGES

    def fit(self, X: np.ndarray, y: object = None) -> ClipTransformer:
        # Nothing to learn; ranges are known upfront
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        # Convert DataFrame to numpy array if needed (sklearn passes DataFrames)
        if hasattr(X, "to_numpy"):
            X_out = X.to_numpy(dtype=float).copy()
        else:
            X_out = np.array(X, dtype=float).copy()
        for i, col in enumerate(NUMERIC_COLS):
            lo, hi = self.ranges[col]
            X_out[:, i] = np.clip(X_out[:, i], lo, hi)
        return X_out

    def get_feature_names_out(
        self, input_features: list[str] | None = None
    ) -> list[str]:
        return NUMERIC_COLS if input_features is None else list(input_features)


# ---------------------------------------------------------------------------
# log1p transformer (wraps numpy for sklearn compatibility)
# ---------------------------------------------------------------------------

def _log1p(X: np.ndarray) -> np.ndarray:
    """Apply log(1 + x) element-wise. Safe because ClipTransformer ensures x >= 0."""
    return np.log1p(X)


log1p_transformer = FunctionTransformer(
    func=_log1p,
    feature_names_out="one-to-one",
    validate=True,
)


# ---------------------------------------------------------------------------
# Sub-pipelines
# ---------------------------------------------------------------------------

def _numeric_pipeline() -> Pipeline:
    """Clip → log1p → StandardScaler for all numeric columns."""
    return Pipeline([
        ("clip",   ClipTransformer()),
        ("log1p",  log1p_transformer),
        ("scaler", StandardScaler()),
    ])


def _ordinal_pipeline() -> Pipeline:
    """OrdinalEncoder for Stress_Level with explicit ordering."""
    return Pipeline([
        ("enc", OrdinalEncoder(
            categories=[ORDINAL_CATEGORIES],
            handle_unknown="use_encoded_value",
            unknown_value=-1,         # unseen stress level → -1 (safe default)
        )),
    ])


def _nominal_pipeline() -> Pipeline:
    """OneHotEncoder for Country and Platform; ignores unseen categories."""
    return Pipeline([
        ("enc", OneHotEncoder(
            handle_unknown="ignore",  # test set may have rare countries → all-zeros row
            sparse_output=False,
        )),
    ])


# ---------------------------------------------------------------------------
# Public API: build the full ColumnTransformer preprocessor
# ---------------------------------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    """
    Returns a ColumnTransformer that preprocesses all feature columns.
    Plug this into a Pipeline with an estimator.
    """
    return ColumnTransformer(
        transformers=[
            ("numeric",  _numeric_pipeline(),  NUMERIC_COLS),
            ("ordinal",  _ordinal_pipeline(),  [ORDINAL_COL]),
            ("nominal",  _nominal_pipeline(),  NOMINAL_COLS),
        ],
        remainder="drop",       # Drop any unexpected extra columns
        verbose_feature_names_out=True,
    )


def build_pipeline(estimator: object) -> Pipeline:
    """
    Wraps the preprocessor and a regressor in a single sklearn Pipeline.

    Usage
    -----
    pipeline = build_pipeline(RandomForestRegressor())
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    """
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model",        estimator),
    ])
