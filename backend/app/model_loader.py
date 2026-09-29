"""
backend/app/model_loader.py
============================
Loads the trained sklearn pipeline and metadata at application startup.

The model is loaded ONCE (singleton pattern) and stored in module-level
variables so that each request handler simply calls `get_pipeline()`.

If the artifact is missing or corrupted, the app fails fast at startup
with a clear error message rather than returning silent 500s at predict time.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib

from .settings import settings

logger = logging.getLogger(__name__)

# Module-level singletons
_pipeline: Any = None
_metadata: dict = {}


def load_model() -> None:
    """
    Load the sklearn pipeline and metadata JSON.
    Called once during the FastAPI lifespan startup event.
    Raises RuntimeError on failure so the app refuses to start.
    """
    global _pipeline, _metadata

    model_path: Path = settings.model_path
    meta_path:  Path = settings.metadata_path

    if not model_path.exists():
        raise RuntimeError(
            f"Model artifact not found at {model_path}. "
            "Run `make train` to generate the artifact."
        )

    if not meta_path.exists():
        raise RuntimeError(
            f"Metadata file not found at {meta_path}. "
            "Run `make train` to generate it."
        )

    logger.info("Loading model pipeline from %s …", model_path)
    _pipeline = joblib.load(model_path)
    logger.info("Pipeline loaded: %s", type(_pipeline).__name__)

    with open(meta_path) as f:
        _metadata = json.load(f)

    logger.info(
        "Metadata loaded | model=%s | trained=%s | sklearn=%s",
        _metadata.get("model_name"),
        _metadata.get("training_date"),
        _metadata.get("sklearn_version"),
    )


def get_pipeline() -> Any:
    """Return the loaded pipeline. Raises if not yet loaded."""
    if _pipeline is None:
        raise RuntimeError("Model not loaded. This is a startup bug.")
    return _pipeline


def get_metadata() -> dict:
    """Return the loaded metadata dict."""
    return _metadata


def is_loaded() -> bool:
    """Return True if the pipeline has been successfully loaded."""
    return _pipeline is not None
