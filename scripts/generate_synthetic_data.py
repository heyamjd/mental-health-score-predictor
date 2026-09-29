"""
scripts/generate_synthetic_data.py
====================================
⚠️  SYNTHETIC DATA NOTICE
This script generates FAKE data for demonstration purposes only.
The mental health scores produced here are mathematically generated
and DO NOT represent real clinical observations.
Drop in a real dataset (same column names) and re-run `make train` to
get meaningful results.
"""

import random

# Add project root to path so we can import ml.config
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import (
    ALLOWED_COUNTRIES,
    ALLOWED_PLATFORMS,
    NUMERIC_COLS,
    ORDINAL_CATEGORIES,
    ORDINAL_COL,
    RANDOM_STATE,
    RAW_CSV,
    TARGET_COL,
)

N_ROWS = 5_000
rng = np.random.default_rng(RANDOM_STATE)
random.seed(RANDOM_STATE)

# ---------------------------------------------------------------------------
# Helper: generate base score from lifestyle inputs
# ---------------------------------------------------------------------------
def compute_score(
    activity: float,
    sleep: float,
    screen: float,
    stress: str,
) -> float:
    """
    A hand-crafted scoring formula that creates realistic-looking relationships
    between inputs and mental health score (0-100).

    The formula is intentionally non-linear so Random Forest has something
    meaningful to learn beyond a linear relationship.
    """
    stress_penalty = {"Low": 0, "Medium": 15, "High": 30}[stress]

    # Activity contributes positively up to ~3h/day, then plateaus
    activity_score = min(activity * 8, 25)

    # Sleep has an optimal range (7-9h): too little or too much reduces score
    sleep_score = max(0.0, 25.0 - abs(sleep - 8.0) * 4.0)

    # Screen time reduces score; heavy users (>8h) get a bigger penalty
    screen_penalty = min(screen * 2.5, 25)

    raw = 50 + activity_score + sleep_score - screen_penalty - stress_penalty
    return float(np.clip(raw, 0, 100))


# ---------------------------------------------------------------------------
# Generate synthetic features
# ---------------------------------------------------------------------------
def generate() -> pd.DataFrame:
    # Stress levels with realistic distribution (High is less common)
    stress_levels = rng.choice(
        ORDINAL_CATEGORIES, size=N_ROWS, p=[0.35, 0.45, 0.20]
    )

    # Physical activity: right-skewed (most people exercise less)
    activity = rng.exponential(scale=1.5, size=N_ROWS).clip(0, 10)

    # Sleep: roughly normal centred at 7.5h, some people sleep very little
    sleep = rng.normal(loc=7.5, scale=1.5, size=N_ROWS).clip(2, 12)

    # Screen time: skewed right (heavy users pull mean up)
    screen = rng.lognormal(mean=1.8, sigma=0.6, size=N_ROWS).clip(0, 16)

    # Country distribution: India-heavy (mirrors user base)
    country_weights = [0.40, 0.15, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02]
    countries = rng.choice(ALLOWED_COUNTRIES, size=N_ROWS, p=country_weights)

    # Platform distribution
    platform_weights = [0.25, 0.15, 0.12, 0.18, 0.12, 0.08, 0.05, 0.05]
    platforms = rng.choice(ALLOWED_PLATFORMS, size=N_ROWS, p=platform_weights)

    # Compute base scores with added Gaussian noise (realistic variance)
    scores = np.array([
        compute_score(a, s, sc, st)
        for a, s, sc, st in zip(activity, sleep, screen, stress_levels)
    ])
    # Add realistic noise and clip
    scores += rng.normal(0, 5, size=N_ROWS)
    scores = np.clip(scores, 0, 100)

    # --- Inject outliers (~2%) ---
    outlier_idx = rng.choice(N_ROWS, size=int(N_ROWS * 0.02), replace=False)
    activity[outlier_idx] = rng.uniform(10, 20, size=len(outlier_idx))  # unrealistically high
    sleep[outlier_idx]    = rng.uniform(12, 20, size=len(outlier_idx))  # too much sleep

    df = pd.DataFrame({
        "Physical_Activity_Hours": activity,
        "Sleep_Hours":             sleep,
        "Screen_Time_Hours":       screen,
        ORDINAL_COL:               stress_levels,
        "Country":                 countries,
        "Platform":                platforms,
        TARGET_COL:                scores,
    })

    # --- Inject missing values (~3% per numeric column) ---
    for col in NUMERIC_COLS:
        mask = rng.random(N_ROWS) < 0.03
        df.loc[mask, col] = np.nan

    # --- Inject missing stress levels (~1%) ---
    mask = rng.random(N_ROWS) < 0.01
    df.loc[mask, ORDINAL_COL] = np.nan

    return df


if __name__ == "__main__":
    print("[INFO] Generating synthetic data...")
    df = generate()

    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_CSV, index=False)

    print(f"[OK] Saved {len(df):,} rows -> {RAW_CSV}")
    print(f"   Missing values per column:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\n   Target stats:\n{df[TARGET_COL].describe().round(2)}")
    print("\n[WARNING] SYNTHETIC DATA -- for demonstration only. Use real data for meaningful results.")
