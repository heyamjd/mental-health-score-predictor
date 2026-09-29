"""
backend/tests/test_model.py
============================
Unit tests for ML pipeline artifacts and data cleaning logic.

Covers:
  - Artifact loading (pipeline.joblib exists and predicts)
  - ClipTransformer
  - IQR outlier removal
  - Pipeline with unseen categorical values (OneHotEncoder handle_unknown)
  - Data cleaning pipeline
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.config import (
    ALL_FEATURE_COLS,
    NUMERIC_COLS,
    ORDINAL_COL,
    PIPELINE_PATH,
    TARGET_COL,
)
from ml.data_cleaning import (
    clip_unrealistic_values,
    fix_dtypes,
    handle_missing,
    iqr_outlier_removal,
    remove_duplicates,
)
from ml.features import ClipTransformer, build_pipeline, build_preprocessor

# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------

class TestArtifactLoading:
    def test_pipeline_file_exists(self):
        assert PIPELINE_PATH.exists(), (
            f"Model artifact not found at {PIPELINE_PATH}. Run `make train` first."
        )

    def test_pipeline_can_predict(self):
        import joblib
        pipeline = joblib.load(PIPELINE_PATH)
        row = {
            "Physical_Activity_Hours": 1.5,
            "Sleep_Hours":             7.0,
            "Screen_Time_Hours":       4.0,
            "Stress_Level":            "Medium",
            "Country":                 "India",
            "Platform":                "Instagram",
        }
        df = pd.DataFrame([row])
        pred = pipeline.predict(df)
        assert len(pred) == 1
        assert 0.0 <= float(pred[0]) <= 100.0

    def test_pipeline_batch_prediction(self):
        import joblib
        pipeline = joblib.load(PIPELINE_PATH)
        rows = [
            {"Physical_Activity_Hours": 0.5, "Sleep_Hours": 5.0,
             "Screen_Time_Hours": 10.0, "Stress_Level": "High",
             "Country": "India", "Platform": "TikTok"},
            {"Physical_Activity_Hours": 3.0, "Sleep_Hours": 8.0,
             "Screen_Time_Hours": 2.0, "Stress_Level": "Low",
             "Country": "USA", "Platform": "None"},
        ]
        df = pd.DataFrame(rows)
        preds = pipeline.predict(df)
        assert len(preds) == 2
        assert all(0.0 <= p <= 100.0 for p in preds)


# ---------------------------------------------------------------------------
# ClipTransformer
# ---------------------------------------------------------------------------

class TestClipTransformer:
    def _make_array(self, pa=1.0, sl=7.0, sc=4.0) -> np.ndarray:
        return np.array([[pa, sl, sc]])

    def test_clips_activity_above_max(self):
        ct = ClipTransformer()
        arr = self._make_array(pa=25.0)  # above 10
        result = ct.fit_transform(arr)
        assert result[0, 0] == 10.0

    def test_clips_sleep_below_min(self):
        ct = ClipTransformer()
        arr = self._make_array(sl=-5.0)  # below 0
        result = ct.fit_transform(arr)
        assert result[0, 1] == 0.0

    def test_valid_values_unchanged(self):
        ct = ClipTransformer()
        arr = self._make_array(pa=2.0, sl=7.0, sc=5.0)
        result = ct.fit_transform(arr)
        np.testing.assert_array_almost_equal(result, arr)

    def test_get_feature_names_out(self):
        ct = ClipTransformer()
        names = ct.get_feature_names_out()
        assert names == NUMERIC_COLS


# ---------------------------------------------------------------------------
# IQR Outlier Removal
# ---------------------------------------------------------------------------

class TestIQROutlierRemoval:
    def _make_df(self) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        n = 200
        data = {
            "Physical_Activity_Hours": rng.normal(1.5, 0.5, n).clip(0, 10),
            "Sleep_Hours":             rng.normal(7.5, 1.0, n).clip(0, 12),
            "Screen_Time_Hours":       rng.normal(4.0, 1.5, n).clip(0, 16),
            TARGET_COL:                rng.uniform(30, 80, n),
        }
        # Inject obvious outliers
        data["Physical_Activity_Hours"][0] = 50.0
        data["Sleep_Hours"][1]             = 100.0
        return pd.DataFrame(data)

    def test_removes_outliers(self):
        df = self._make_df()
        before = len(df)
        df_clean, _ = iqr_outlier_removal(df, NUMERIC_COLS)
        assert len(df_clean) < before

    def test_report_keys(self):
        df = self._make_df()
        _, report = iqr_outlier_removal(df, NUMERIC_COLS)
        for col in NUMERIC_COLS:
            assert col in report

    def test_no_false_positives_on_clean_data(self):
        """Tight normal data should have very few removed rows."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "Physical_Activity_Hours": rng.normal(2.0, 0.1, 500).clip(0, 10),
            "Sleep_Hours":             rng.normal(7.5, 0.1, 500).clip(0, 12),
            "Screen_Time_Hours":       rng.normal(4.0, 0.1, 500).clip(0, 16),
            TARGET_COL:                rng.uniform(40, 70, 500),
        })
        df_clean, _ = iqr_outlier_removal(df, NUMERIC_COLS)
        # Should keep at least 95% of data
        assert len(df_clean) / len(df) >= 0.95


# ---------------------------------------------------------------------------
# Data Cleaning Pipeline
# ---------------------------------------------------------------------------

class TestDataCleaning:
    def _make_raw(self) -> pd.DataFrame:
        return pd.DataFrame({
            "Physical_Activity_Hours": [1.5, None, 30.0, 2.0],
            "Sleep_Hours":             [7.0, 6.0, None,  8.0],
            "Screen_Time_Hours":       [4.0, 5.0, 6.0,   None],
            ORDINAL_COL:               ["Low", "High", None, "Medium"],
            "Country":                 ["India", "USA", "UK", "Germany"],
            "Platform":                ["Instagram", "TikTok", "None", "Reddit"],
            TARGET_COL:                [65.0, 45.0, 75.0, 80.0],
        })

    def test_fix_dtypes_converts_numeric(self):
        df = self._make_raw()
        df["Physical_Activity_Hours"] = df["Physical_Activity_Hours"].astype(str)
        df = fix_dtypes(df)
        assert pd.api.types.is_float_dtype(df["Physical_Activity_Hours"])

    def test_handle_missing_fills_median(self):
        df = self._make_raw()
        df = fix_dtypes(df)
        df = handle_missing(df)
        assert df[NUMERIC_COLS].isnull().sum().sum() == 0

    def test_clip_unrealistic_values(self):
        df = self._make_raw()
        df = fix_dtypes(df)
        df = clip_unrealistic_values(df)
        assert df["Physical_Activity_Hours"].max() <= 10.0

    def test_remove_duplicates(self):
        df = pd.DataFrame({
            "Physical_Activity_Hours": [1.0, 1.0],
            "Sleep_Hours":             [7.0, 7.0],
            "Screen_Time_Hours":       [4.0, 4.0],
            ORDINAL_COL:               ["Low", "Low"],
            "Country":                 ["India", "India"],
            "Platform":                ["Instagram", "Instagram"],
            TARGET_COL:                [65.0, 65.0],
        })
        df_clean = remove_duplicates(df)
        assert len(df_clean) == 1

    def test_clean_data_pipeline(self):
        from ml.data_cleaning import clean
        df = self._make_raw()
        df_clean = clean(df)
        assert len(df_clean) >= 1
        assert df_clean[NUMERIC_COLS].isna().sum().sum() == 0




# ---------------------------------------------------------------------------
# Pipeline with unseen categories
# ---------------------------------------------------------------------------

class TestPipelineUnseenCategories:
    def test_unseen_country_does_not_raise(self):
        """OneHotEncoder with handle_unknown='ignore' should return zeros for unseen."""
        preprocessor = build_preprocessor()
        train_data = pd.DataFrame([{
            "Physical_Activity_Hours": 1.5,
            "Sleep_Hours":             7.0,
            "Screen_Time_Hours":       4.0,
            ORDINAL_COL:               "Medium",
            "Country":                 "India",
            "Platform":                "Instagram",
        }])
        preprocessor.fit(train_data)

        test_data = pd.DataFrame([{
            "Physical_Activity_Hours": 2.0,
            "Sleep_Hours":             6.5,
            "Screen_Time_Hours":       5.0,
            ORDINAL_COL:               "High",
            "Country":                 "Atlantis",   # unseen!
            "Platform":                "Instagram",
        }])
        # Should not raise
        result = preprocessor.transform(test_data)
        assert result is not None


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------

class TestEvaluation:
    def test_regression_metrics(self):
        from ml.evaluate import regression_metrics
        y_true = np.array([50.0, 60.0, 70.0, 80.0])
        y_pred = np.array([50.0, 60.0, 70.0, 80.0])
        m = regression_metrics(y_true, y_pred)
        assert m["r2"] == 1.0
        assert m["mae"] == 0.0
        assert m["rmse"] == 0.0

    def test_save_metrics(self, tmp_path):
        from ml.evaluate import save_metrics
        path = tmp_path / "test_metrics.json"
        data = {"r2": 0.85, "mae": 4.8}
        save_metrics(data, path)
        assert path.exists()
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["r2"] == 0.85

    def test_print_comparison_table(self, capsys):
        from ml.evaluate import print_comparison_table
        lr = {"r2": 0.77, "mae": 5.6, "rmse": 7.8}
        rf = {"r2": 0.85, "mae": 4.8, "rmse": 6.3}
        print_comparison_table(lr, rf)
        captured = capsys.readouterr()
        assert "Linear Regression" in captured.out
        assert "Random Forest" in captured.out

    def test_save_comparison_plot(self, tmp_path):
        from ml.evaluate import save_comparison_plot
        y_true = np.array([50.0, 60.0, 70.0, 80.0])
        y_lr = np.array([48.0, 62.0, 68.0, 82.0])
        y_rf = np.array([49.0, 61.0, 70.0, 79.0])
        fig_dir = tmp_path / "figures"
        save_comparison_plot(y_true, y_lr, y_rf, fig_dir)
        assert (fig_dir / "predicted_vs_actual.png").exists()


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------

class TestTrainingPipeline:
    def test_build_pipeline(self):
        from sklearn.linear_model import LinearRegression
        pipe = build_pipeline(LinearRegression())
        assert hasattr(pipe, "fit")
        assert hasattr(pipe, "predict")

    def test_train_linear_regression(self):
        from sklearn.model_selection import train_test_split

        from ml.train import train_linear_regression

        df = pd.DataFrame({
            "Physical_Activity_Hours": [1.0, 2.0, 3.0, 1.5, 2.5] * 10,
            "Sleep_Hours":             [7.0, 8.0, 6.0, 7.5, 8.5] * 10,
            "Screen_Time_Hours":       [4.0, 3.0, 5.0, 4.5, 3.5] * 10,
            ORDINAL_COL:               ["Low", "Medium", "High", "Low", "Medium"] * 10,
            "Country":                 ["India", "USA", "UK", "Germany", "India"] * 10,
            "Platform":                ["Instagram", "YouTube", "Reddit", "None", "Twitter"] * 10,
            TARGET_COL:                [70.0, 80.0, 50.0, 75.0, 85.0] * 10,
        })
        X = df[ALL_FEATURE_COLS]
        y = df[TARGET_COL]
        X_train, X_test, y_train, _y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        pipe, cv_mean = train_linear_regression(X_train, y_train)
        assert hasattr(pipe, "predict")
        preds = pipe.predict(X_test)
        assert len(preds) == len(X_test)
        assert isinstance(cv_mean, float)

    def test_load_and_split(self):
        from ml.train import load_and_split
        X_train, X_test, y_train, y_test = load_and_split()
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)




