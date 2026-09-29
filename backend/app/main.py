"""
backend/app/main.py
====================
FastAPI application entry point.

Endpoints
---------
GET  /health       – liveness check; returns model_loaded flag
GET  /metadata     – allowed inputs and score bands (form is built from this)
POST /predict      – submit lifestyle inputs, get mental health score
GET  /model-info   – model name, metrics, training date

Design decisions
----------------
- Model loads once in lifespan() to avoid per-request I/O
- CORS origins driven by ALLOWED_ORIGINS env var (no * in prod)
- Rate limiting via slowapi (30 req/min per IP by default)
- Security + request-ID headers on every response
- Structured logging; user input values are NEVER logged
- No user data is stored anywhere
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from .model_loader import get_metadata, get_pipeline, is_loaded, load_model
from .schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from .security import RequestIDMiddleware, SecurityHeadersMiddleware
from .settings import settings

# ---------------------------------------------------------------------------
# Logging (structured, no PII)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address, default_limits=[settings.rate_limit])

# ---------------------------------------------------------------------------
# Lifespan: startup / shutdown events
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    """Load the model at startup; log on shutdown."""
    logger.info("Starting up %s v%s …", settings.app_name, settings.app_version)
    load_model()   # Fails fast with RuntimeError if artifact is missing
    logger.info("✅  Model loaded successfully.")
    yield
    logger.info("Shutting down %s.", settings.app_name)


# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "## Mental Health Score Predictor API\n\n"
        "Estimates a wellness score (0–100) from lifestyle inputs.\n\n"
        "> ⚠️ **This is an informational screening tool, NOT a medical diagnosis.**\n\n"
        "No user data is stored. Predictions are stateless."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# Middleware (order matters: outermost runs first)
# ---------------------------------------------------------------------------
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Request-ID"],
)

# Rate limiter exception handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ---------------------------------------------------------------------------
# Global exception handler (never expose internals)
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled exception on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Please try again later."},
    )


# ---------------------------------------------------------------------------
# Score interpretation helper
# ---------------------------------------------------------------------------

def _interpret_score(score: float) -> tuple[str, str]:
    """
    Return (status_flag, message) for a given score.
    Falls back to the lowest band if score is below all defined bands.
    """
    meta = get_metadata()
    bands = meta.get("score_bands", [])
    messages = meta.get("score_messages", {})

    for band in sorted(bands, key=lambda b: b["min"], reverse=True):
        if score >= band["min"]:
            label = band["label"]
            return label, messages.get(label, "Take care of yourself.")

    # Fallback
    label = bands[-1]["label"] if bands else "Unknown"
    return label, messages.get(label, "Please seek professional support.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

DISCLAIMER = (
    "This score is estimated by a machine-learning model trained on synthetic data. "
    "It is NOT a medical diagnosis and should not replace professional mental health advice. "
    "If you are struggling, please reach out to a qualified professional or helpline."
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
@limiter.limit("60/minute")
async def health(request: Request) -> HealthResponse:
    """Liveness check. Returns model_loaded=True when the artifact is ready."""
    return HealthResponse(
        status="ok",
        model_loaded=is_loaded(),
        version=settings.app_version,
    )


@app.get("/metadata", tags=["Model"])
@limiter.limit("60/minute")
async def metadata(request: Request) -> dict:
    """
    Returns allowed input values and score interpretation bands.
    The frontend builds its form dynamically from this response.
    """
    meta = get_metadata()
    # Return only the safe subset (no internal paths, etc.)
    return {
        "numeric_features": meta.get("numeric_features", []),
        "numeric_ranges":   meta.get("numeric_ranges", {}),
        "ordinal_feature":  meta.get("ordinal_feature"),
        "ordinal_categories": meta.get("ordinal_categories", []),
        "allowed_countries":  meta.get("allowed_countries", []),
        "allowed_platforms":  meta.get("allowed_platforms", []),
        "score_bands":        meta.get("score_bands", []),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
@limiter.limit(settings.rate_limit)
async def predict(request: Request, body: PredictionRequest) -> PredictionResponse:
    """
    Estimate a mental health wellness score from lifestyle inputs.

    - Validates all inputs strictly (422 on invalid values)
    - Returns score 0–100, a status flag, and a supportive message
    - User input values are NEVER logged
    - No data is stored
    """
    if not is_loaded():
        raise HTTPException(status_code=503, detail="Model not available. Please try again shortly.")

    pipeline = get_pipeline()

    # Build a single-row DataFrame matching training feature order
    row: dict[str, Any] = {
        "Physical_Activity_Hours": body.Physical_Activity_Hours,
        "Sleep_Hours":             body.Sleep_Hours,
        "Screen_Time_Hours":       body.Screen_Time_Hours,
        "Stress_Level":            body.Stress_Level,
        "Country":                 body.Country,
        "Platform":                body.Platform,
    }
    df_input = pd.DataFrame([row])

    try:
        raw_score = float(pipeline.predict(df_input)[0])
        score = round(max(0.0, min(100.0, raw_score)), 1)
    except Exception as exc:  # noqa: BLE001
        logger.error("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail="Prediction error. Please try again.")

    status_flag, message = _interpret_score(score)
    prediction_id = uuid.uuid4()

    # Log only non-PII data (no input values)
    logger.info(
        "Prediction complete | id=%s | score=%.1f | flag=%s",
        prediction_id, score, status_flag,
    )

    return PredictionResponse(
        prediction_id=prediction_id,
        mental_health_score=score,
        status_flag=status_flag,
        message=message,
        disclaimer=DISCLAIMER,
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
@limiter.limit("30/minute")
async def model_info(request: Request) -> ModelInfoResponse:
    """Returns the model name, training date, and evaluation metrics."""
    meta = get_metadata()
    return ModelInfoResponse(
        model_name=meta.get("model_name", "Unknown"),
        training_date=meta.get("training_date", "Unknown"),
        metrics=meta.get("metrics", {}),
        is_synthetic=meta.get("is_synthetic_data", True),
        data_warning=meta.get("data_warning"),
    )
