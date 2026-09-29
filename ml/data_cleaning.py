"""
ml/data_cleaning.py
====================
Data cleaning utilities for the Mental Health Score Predictor.

Steps performed:
  1. Load raw CSV
  2. Report missing values
  3. Remove duplicate rows
  4. Validate and fix dtypes
  5. Remove/clip unrealistic values (e.g. sleep > 24h)
  6. Impute or drop remaining missing values
  7. Save clean CSV to data/processed/

IQR-based outlier removal is performed ONLY on training data inside the
training script (ml/train.py) to prevent data leakage.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Resolve imports whether run directly or as a module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import (
    NOMINAL_COLS,
    NUMERIC_COLS,
    NUMERIC_RANGES,
    ORDINAL_CATEGORIES,
    ORDINAL_COL,
    PROCESSED_CSV,
    RAW_CSV,
    TARGET_COL,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_raw(path: Path = RAW_CSV) -> pd.DataFrame:
    """Load raw CSV and return a DataFrame."""
    logger.info("Loading raw data from %s", path)
    df = pd.read_csv(path)
    logger.info("  Shape: %s", df.shape)
    return df


def report_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy DataFrame with missing-value counts and percentages."""
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df) * 100).round(2)
    report = pd.DataFrame(
        {"missing_count": missing_count, "missing_pct": missing_pct}
    ).query("missing_count > 0")
    logger.info("Missing values:\n%s", report.to_string())
    return report


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows and log how many were removed."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    logger.info("Duplicates removed: %d", removed)
    return df


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce columns to expected dtypes."""
    for col in NUMERIC_COLS + [TARGET_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalise string columns: strip whitespace, title-case
    for col in [ORDINAL_COL] + NOMINAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            # Replace 'nan' strings introduced by .astype(str)
            df[col] = df[col].replace("nan", np.nan)
    return df


def clip_unrealistic_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clip numeric columns to their valid physical ranges.
    Values outside these ranges are biologically / behaviourally impossible
    (e.g. 30 hours of sleep per day).
    """
    for col, (lo, hi) in NUMERIC_RANGES.items():
        before_outliers = ((df[col] < lo) | (df[col] > hi)).sum()
        df[col] = df[col].clip(lower=lo, upper=hi)
        if before_outliers:
            logger.info("  %s: clipped %d unrealistic values to [%s, %s]",
                        col, before_outliers, lo, hi)
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strategy:
      - Numeric cols: fill with column median (robust to remaining skew)
      - Ordinal / Nominal: fill with the mode
      - Target: drop rows where target is missing (cannot learn without a label)
    """
    # Drop rows with missing target
    before = len(df)
    df = df.dropna(subset=[TARGET_COL])
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d rows with missing target.", dropped)

    # Impute numeric columns with median
    for col in NUMERIC_COLS:
        median = df[col].median()
        n_filled = df[col].isnull().sum()
        if n_filled:
            df[col] = df[col].fillna(median)
            logger.info("  %s: filled %d missing with median=%.2f", col, n_filled, median)

    # Impute categorical columns with mode
    for col in [ORDINAL_COL] + NOMINAL_COLS:
        if col in df.columns:
            mode = df[col].mode(dropna=True)
            if not mode.empty:
                n_filled = df[col].isnull().sum()
                if n_filled:
                    df[col] = df[col].fillna(mode.iloc[0])
                    logger.info("  %s: filled %d missing with mode='%s'",
                                col, n_filled, mode.iloc[0])
    return df


def validate_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map unseen ordinal values to NaN, then re-impute.
    This ensures the ordinal encoder never sees unexpected values.
    """
    valid = set(ORDINAL_CATEGORIES)
    unknown_mask = ~df[ORDINAL_COL].isin(valid)
    n_unknown = unknown_mask.sum()
    if n_found := n_unknown:
        logger.warning("  %s: %d rows with unknown category → replacing with mode",
                       ORDINAL_COL, n_found)
        mode = df.loc[~unknown_mask, ORDINAL_COL].mode().iloc[0]
        df.loc[unknown_mask, ORDINAL_COL] = mode
    return df


def clip_target(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure target is in [0, 100]."""
    df[TARGET_COL] = df[TARGET_COL].clip(0, 100)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline and return a clean DataFrame."""
    logger.info("=== Starting data cleaning ===")
    df = remove_duplicates(df)
    df = fix_dtypes(df)
    df = clip_unrealistic_values(df)
    df = handle_missing(df)
    df = validate_categories(df)
    df = clip_target(df)
    logger.info("=== Cleaning complete. Final shape: %s ===", df.shape)
    return df


def iqr_outlier_removal(
    df: pd.DataFrame,
    cols: list[str],
    factor: float = 1.5,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Remove rows where any numeric column has a value outside
    [Q1 - factor*IQR, Q3 + factor*IQR].

    IMPORTANT: Call this ONLY on the training split to avoid leakage.

    Returns
    -------
    df_clean : filtered DataFrame
    report   : {column: n_rows_removed}
    """
    mask_keep = pd.Series(True, index=df.index)
    report: dict[str, int] = {}

    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        col_mask = (df[col] >= lower) & (df[col] <= upper)
        n_removed = (~col_mask).sum()
        report[col] = int(n_removed)
        mask_keep &= col_mask
        logger.info("  IQR outlier — %s: removed %d rows  (bounds=[%.2f, %.2f])",
                    col, n_removed, lower, upper)

    return df[mask_keep].reset_index(drop=True), report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    raw_df = load_raw()
    report_missing(raw_df)
    clean_df = clean(raw_df)
    PROCESSED_CSV.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(PROCESSED_CSV, index=False)
    logger.info("Saved clean data → %s", PROCESSED_CSV)
