# docs/ARCHITECTURE.md
# Architecture — Mental Health Score Predictor

## System Overview

The application is split into two independently deployable services:

| Service | Technology | Deployed As |
|---------|-----------|-------------|
| API Backend | Python, FastAPI | Render Web Service |
| Frontend | HTML5, CSS3, Vanilla JS | Render Static Site |

There is **no database**. The API is stateless — no user data is stored anywhere.

---

## Level-0 DFD (Context Diagram)

```mermaid
graph LR
    USER["👤 User\n(Browser)"]
    SYS["Mental Health\nScore Predictor\nSystem"]
    USER -- "Lifestyle inputs\n(Activity, Sleep, Screen,\nStress, Country, Platform)" --> SYS
    SYS -- "Wellness score (0-100)\nStatus flag\nSupportive message" --> USER
```

---

## Level-1 DFD

```mermaid
graph TD
    USER["👤 User"]
    FE["Frontend\n(index.html / app.js)"]
    API["FastAPI Backend\n(main.py)"]
    ML["Sklearn Pipeline\n(pipeline.joblib)"]
    META["Metadata\n(metadata.json)"]
    CFG["config.js\n(API base URL)"]

    USER -- "form submit" --> FE
    CFG  -- "API_BASE_URL" --> FE
    FE   -- "GET /metadata" --> API
    API  -- "allowed values, score bands" --> FE
    FE   -- "POST /predict\n(JSON body)" --> API
    API  -- "validate (Pydantic)" --> API
    API  -- "predict(X)" --> ML
    ML   -- "raw score" --> API
    API  -- "load at startup" --> META
    META -- "score bands, messages" --> API
    API  -- "PredictionResponse\n(score, flag, message)" --> FE
    FE   -- "animated gauge + message" --> USER
```

---

## Component Diagram

```mermaid
graph LR
    subgraph Frontend ["Frontend (Static Site)"]
        HTML["index.html"]
        CSS["styles.css"]
        JS["app.js"]
        CFG2["config.js"]
    end

    subgraph Backend ["Backend (Web Service)"]
        MAIN["main.py\n(FastAPI app)"]
        SCHEMA["schemas.py\n(Pydantic v2)"]
        LOADER["model_loader.py\n(singleton)"]
        SECURITY["security.py\n(headers, request-ID)"]
        SETTINGS["settings.py\n(env vars)"]
    end

    subgraph ML ["ML Artifacts (models/)"]
        PIPE["pipeline.joblib\n(sklearn Pipeline)"]
        METRICS["metrics.json"]
        META2["metadata.json"]
    end

    JS --> MAIN
    MAIN --> SCHEMA
    MAIN --> LOADER
    MAIN --> SECURITY
    MAIN --> SETTINGS
    LOADER --> PIPE
    LOADER --> META2
```

---

## Request / Response Entity Diagrams

```mermaid
erDiagram
    PredictionRequest {
        float Physical_Activity_Hours "0.0 - 10.0 hrs/day"
        float Sleep_Hours             "0.0 - 12.0 hrs"
        float Screen_Time_Hours       "0.0 - 16.0 hrs/day"
        string Stress_Level           "Low | Medium | High"
        string Country                "enum[10 values]"
        string Platform               "enum[8 values]"
    }

    PredictionResponse {
        UUID  prediction_id          "unique request ID"
        float mental_health_score    "0.0 - 100.0"
        string status_flag           "Thriving|Good|Moderate|Needs Attention|Seek Support"
        string message               "supportive text"
        string disclaimer            "not a diagnosis"
    }

    ModelMetadata {
        string model_name            "e.g. Random Forest"
        string training_date         "ISO 8601"
        string sklearn_version       "e.g. 1.9.1"
        list   features              "feature column names"
        dict   numeric_ranges        "min/max per column"
        list   score_bands           "label, min, max, color"
        dict   metrics               "r2, mae, rmse, cv_r2"
        bool   is_synthetic_data     "true for demo"
    }

    PredictionRequest ||--|| PredictionResponse : "POST /predict"
    PredictionResponse }o--|| ModelMetadata : "score interpreted via"
```

---

## Data Flow — Training Pipeline

```mermaid
graph LR
    RAW["data/raw/\nmental_health_data.csv\n(5,000 rows)"]
    CLEAN["ml/data_cleaning.py\n(dedup, dtype, clip,\nimpute, validate)"]
    SPLIT["train_test_split\n70% train / 30% test\nrandom_state=42"]
    IQR["IQR Outlier Removal\n(training split only)"]
    PIPE["sklearn Pipeline\nColumnTransformer\n+ estimator"]
    LR["Linear Regression\n(baseline)"]
    RF["Random Forest\n+ RandomizedSearchCV"]
    EVAL["Evaluate on test set\nR2, MAE, RMSE"]
    BEST["Auto-select best\nby test R2"]
    SAVE["models/\npipeline.joblib\nmetrics.json\nmetadata.json"]

    RAW --> CLEAN --> SPLIT
    SPLIT -- "X_train, y_train" --> IQR
    SPLIT -- "X_test, y_test" --> EVAL
    IQR --> PIPE
    PIPE --> LR
    PIPE --> RF
    LR --> EVAL
    RF --> EVAL
    EVAL --> BEST --> SAVE
```

---

## Security Architecture

| Layer | Mechanism |
|-------|-----------|
| CORS | `ALLOWED_ORIGINS` env var — whitelist-based |
| Rate limiting | slowapi — 30 req/min per IP |
| Security headers | `X-Frame-Options`, `X-Content-Type-Options`, etc. |
| Request tracing | UUID `X-Request-ID` on every response |
| Data privacy | No user input logged; no storage |
| Input validation | Pydantic v2 strict mode — 422 on bad values |
