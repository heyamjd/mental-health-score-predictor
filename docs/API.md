# API Reference — Mental Health Score Predictor

Base URL (Local): `http://127.0.0.1:8000`  
Base URL (Production): `https://mhsp-backend.onrender.com` (or configured custom domain)  
Interactive OpenAPI Documentation: `/docs` (Swagger UI) or `/redoc` (ReDoc)

---

## Headers & Security

| Header | Description | Required |
|---|---|---|
| `Content-Type` | `application/json` | Required for `POST /predict` |
| `X-Request-ID` | Unique tracking ID generated per request if not provided | Auto-generated |

**Rate Limiting:**  
- `/predict`: Max 60 requests per minute per client IP (`429 Too Many Requests` returned when exceeded).
- Global rate limits can be configured via environment variables.

---

## Endpoints

### 1. Health Check
`GET /health`

Verifies that the backend server is running and checks whether the ML model artifact is loaded and ready for inference.

#### Response: `200 OK`
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2026-09-29T15:25:00.000Z",
  "version": "1.0.0"
}
```

---

### 2. Model Metadata
`GET /metadata`

Returns the allowed categories, numeric valid min/max ranges, model features, and score bands. The frontend dynamically renders and validates input form fields based on this endpoint.

#### Response: `200 OK`
```json
{
  "model_name": "Random Forest Regressor",
  "trained_date": "2026-09-29T15:28:40",
  "sklearn_version": "1.4.2",
  "features": {
    "numeric": [
      {
        "name": "Physical_Activity_Hours",
        "label": "Physical Activity (Hours/Day)",
        "min": 0.0,
        "max": 10.0,
        "default": 1.5,
        "step": 0.1,
        "unit": "hours"
      },
      {
        "name": "Sleep_Hours",
        "label": "Sleep Duration (Hours/Day)",
        "min": 0.0,
        "max": 12.0,
        "default": 7.0,
        "step": 0.5,
        "unit": "hours"
      },
      {
        "name": "Screen_Time_Hours",
        "label": "Screen Time (Hours/Day)",
        "min": 0.0,
        "max": 24.0,
        "default": 6.0,
        "step": 0.5,
        "unit": "hours"
      }
    ],
    "categorical": [
      {
        "name": "Stress_Level",
        "label": "Current Stress Level",
        "options": ["Low", "Medium", "High"],
        "default": "Medium"
      },
      {
        "name": "Platform",
        "label": "Most Used Social Platform",
        "options": ["Instagram", "YouTube", "Twitter/X", "TikTok", "Reddit", "Facebook", "Other"],
        "default": "Instagram"
      }
    ]
  },
  "score_bands": [
    {
      "min": 75,
      "max": 100,
      "status": "Good",
      "color": "#10B981",
      "description": "Balanced lifestyle indicators and lower stress markers."
    },
    {
      "min": 50,
      "max": 74,
      "status": "Moderate",
      "color": "#F59E0B",
      "description": "Moderate wellness indicators with opportunities for lifestyle adjustments."
    },
    {
      "min": 0,
      "max": 49,
      "status": "Needs Attention",
      "color": "#EF4444",
      "description": "High stress or lifestyle imbalance indicated; self-care and support recommended."
    }
  ]
}
```

---

### 3. Model Information & Metrics
`GET /model-info`

Returns training metrics, cross-validation scores, and configuration details.

#### Response: `200 OK`
```json
{
  "model_type": "Random Forest Regressor",
  "pipeline_version": "1.0.0",
  "metrics": {
    "r2": 0.8545,
    "mae": 4.8635,
    "rmse": 6.3409,
    "cv_r2_mean": 0.8585
  },
  "baseline_metrics": {
    "linear_regression": {
      "r2": 0.7786,
      "mae": 5.6245,
      "rmse": 7.8212
    }
  },
  "training_sample_count": 3244,
  "test_sample_count": 1500
}
```

---

### 4. Mental Health Score Prediction
`POST /predict`

Predicts wellness score (0-100) based on user lifestyle inputs.

#### Request Body
```json
{
  "Physical_Activity_Hours": 2.5,
  "Sleep_Hours": 8.0,
  "Screen_Time_Hours": 4.5,
  "Stress_Level": "Low",
  "Platform": "YouTube"
}
```

#### Field Validation Rules
| Field | Type | Rules | Example |
|---|---|---|---|
| `Physical_Activity_Hours` | float | 0.0 <= val <= 10.0 | `2.5` |
| `Sleep_Hours` | float | 0.0 <= val <= 12.0 | `8.0` |
| `Screen_Time_Hours` | float | 0.0 <= val <= 24.0 | `4.5` |
| `Stress_Level` | string | One of: `"Low"`, `"Medium"`, `"High"` | `"Low"` |
| `Platform` | string | String max 50 chars | `"YouTube"` |

#### Response: `200 OK`
```json
{
  "prediction_id": "4b68e9bf-05a8-4444-a69d-cfd9d20c5765",
  "mental_health_score": 84.2,
  "status_flag": "Good",
  "color_code": "#10B981",
  "message": "Your lifestyle indicators reflect healthy sleep and low stress patterns. Keep prioritizing regular physical activity!",
  "timestamp": "2026-09-29T15:35:12.123Z"
}
```

#### Error Responses
- **422 Unprocessable Entity:** Input validation failure (e.g. `Sleep_Hours` > 12).
```json
{
  "detail": [
    {
      "loc": ["body", "Sleep_Hours"],
      "msg": "Input should be less than or equal to 12.0",
      "type": "less_than_equal"
    }
  ]
}
```
- **429 Too Many Requests:** Rate limit exceeded.
- **503 Service Unavailable:** ML model not loaded or internal inference error.

---

## Example cURL Commands

### Predict Mental Health Score:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Physical_Activity_Hours": 2.0,
    "Sleep_Hours": 7.5,
    "Screen_Time_Hours": 5.0,
    "Stress_Level": "Medium",
    "Platform": "Instagram"
  }'
```

### Health Check:
```bash
curl -i "http://127.0.0.1:8000/health"
```
