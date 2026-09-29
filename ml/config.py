"""
ml/config.py
============
Central configuration for the Mental Health Score Predictor ML pipeline.
All column names, file paths, and hyperparameter search spaces are defined
HERE so that changing a column name never requires touching multiple files.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root & data paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"
FIGURES_DIR = DOCS_DIR / "figures"

RAW_CSV = DATA_RAW_DIR / "mental_health_data.csv"
PROCESSED_CSV = DATA_PROCESSED_DIR / "mental_health_clean.csv"
PIPELINE_PATH = MODELS_DIR / "pipeline.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
METADATA_PATH = MODELS_DIR / "metadata.json"

# ---------------------------------------------------------------------------
# Column names  ← only place they are defined
# ---------------------------------------------------------------------------
TARGET_COL = "Mental_Health_Score"

NUMERIC_COLS = [
    "Physical_Activity_Hours",
    "Sleep_Hours",
    "Screen_Time_Hours",
]

ORDINAL_COL = "Stress_Level"
ORDINAL_CATEGORIES = ["Low", "Medium", "High"]   # explicit order

NOMINAL_COLS = ["Country", "Platform"]

ALL_FEATURE_COLS = NUMERIC_COLS + [ORDINAL_COL] + NOMINAL_COLS

# ---------------------------------------------------------------------------
# Realistic valid ranges for numeric columns (used for clipping & validation)
# ---------------------------------------------------------------------------
NUMERIC_RANGES: dict[str, tuple[float, float]] = {
    "Physical_Activity_Hours": (0.0, 10.0),   # hours per day
    "Sleep_Hours":             (0.0, 12.0),   # hours per day
    "Screen_Time_Hours":       (0.0, 16.0),   # hours per day
}

# Target score range
TARGET_MIN = 0.0
TARGET_MAX = 100.0

# ---------------------------------------------------------------------------
# Allowed categorical values (used by API and validation)
# ---------------------------------------------------------------------------
ALLOWED_COUNTRIES = [
    "India", "USA", "UK", "Canada", "Australia",
    "Germany", "Brazil", "Japan", "South Korea", "Other",
]

ALLOWED_PLATFORMS = [
    "Instagram", "Twitter", "Facebook", "TikTok",
    "YouTube", "Reddit", "LinkedIn", "None",
]

# ---------------------------------------------------------------------------
# Score interpretation bands
# ---------------------------------------------------------------------------
SCORE_BANDS = [
    {"label": "Thriving",       "min": 80, "max": 100, "color": "#2ecc71"},
    {"label": "Good",           "min": 60, "max": 79,  "color": "#27ae60"},
    {"label": "Moderate",       "min": 40, "max": 59,  "color": "#f39c12"},
    {"label": "Needs Attention","min": 20, "max": 39,  "color": "#e67e22"},
    {"label": "Seek Support",   "min": 0,  "max": 19,  "color": "#c0392b"},
]

# Short supportive messages keyed by band label
SCORE_MESSAGES: dict[str, str] = {
    "Thriving":        "You seem to be in great shape! Keep up those healthy habits.",
    "Good":            "You're doing well. Small, consistent habits will keep you here.",
    "Moderate":        "Some areas could use attention. Consider prioritising sleep and reducing stress.",
    "Needs Attention": "It looks like some lifestyle factors may be affecting your wellbeing. Please consider speaking to someone you trust or a professional.",
    "Seek Support":    "Your responses suggest you may be going through a tough time. Please reach out to a mental health professional or a helpline listed below.",
}

# ---------------------------------------------------------------------------
# Training hyper-parameters
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.30

# RandomizedSearchCV settings
N_ITER_SEARCH = 30
CV_FOLDS = 5

# Hyperparameter distributions for RandomForest
RF_PARAM_DIST: dict = {
    "model__n_estimators":      [100, 200, 300, 500],
    "model__max_depth":         [None, 5, 10, 15, 20],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf":  [1, 2, 4],
    "model__max_features":      ["sqrt", "log2", None],
}
