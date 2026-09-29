"""
ml/evaluate.py
===============
Evaluation utilities: compute metrics and save comparison plots.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Return R², MAE, and RMSE."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    return {
        "r2":   round(float(r2_score(y_true, y_pred)), 4),
        "mae":  round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
    }


def save_comparison_plot(
    y_true: np.ndarray,
    y_pred_lr: np.ndarray,
    y_pred_rf: np.ndarray,
    figures_dir: Path,
) -> None:
    """Save a predicted-vs-actual scatter plot comparing both models."""
    import matplotlib
    matplotlib.use("Agg")   # headless backend for server environments
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Predicted vs Actual — Mental Health Score", fontsize=14, fontweight="bold")

    for ax, y_pred, title, color in [
        (axes[0], y_pred_lr, "Linear Regression", "#3498db"),
        (axes[1], y_pred_rf, "Random Forest",     "#2ecc71"),
    ]:
        ax.scatter(y_true, y_pred, alpha=0.4, s=15, color=color)
        lims = [0, 100]
        ax.plot(lims, lims, "r--", linewidth=1, label="Perfect fit")
        ax.set_xlabel("Actual Score")
        ax.set_ylabel("Predicted Score")
        ax.set_title(title)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.legend()

    plt.tight_layout()
    out = figures_dir / "predicted_vs_actual.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved predicted-vs-actual plot → %s", out)


def save_metrics(
    metrics: dict,
    path: Path,
) -> None:
    """Persist metrics dict as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics → %s", path)


def print_comparison_table(lr_metrics: dict, rf_metrics: dict) -> None:
    """Print a formatted comparison table to stdout."""
    header = f"{'Metric':<10} {'Linear Regression':>20} {'Random Forest':>20}"
    sep = "─" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for metric in ["r2", "mae", "rmse"]:
        print(f"{metric.upper():<10} {lr_metrics[metric]:>20.4f} {rf_metrics[metric]:>20.4f}")
    print(sep)
    print()
