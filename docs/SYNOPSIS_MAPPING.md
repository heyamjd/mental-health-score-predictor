# Synopsis Mapping — Mental Health Score Predictor

**Project:** Mental Health Score Predictor (B.Tech Minor Project)  
**Institution:** Poornima University, Jaipur  
**Department:** Department of Computer Science & Engineering  

---

## Synopsis Module to Implementation Mapping

This document establishes the traceability matrix connecting the B.Tech project synopsis modules to the production codebase implementation.

| Synopsis Module | Synopsis Description | Implemented Files / Components | Key Features & Implementation Highlights |
|---|---|---|---|
| **Module 1: Data Collection & Synthetic Modeling** | Dataset acquisition, feature schema, lifestyle variables | `scripts/generate_synthetic_data.py`, `ml/config.py`, `data/raw/mental_health_data.csv` | • Realistic synthetic population generation ($N=5,000$)<br>• Defined schema: 3 continuous numeric, 1 ordinal, 2 nominal categorical features<br>• Controlled skewness, domain limits, and realistic missing rates (~3%) |
| **Module 2: Exploratory Data Analysis & Outlier Handling** | Missing value analysis, distributions, correlation heatmap, IQR filtering | `notebooks/01_eda.ipynb`, `ml/data_cleaning.py`, `docs/figures/` | • Missing imputation (median for numeric, mode for categorical)<br>• IQR outlier detection (1.5 × IQR) on training split only (prevents data leakage)<br>• Domain clipping (e.g. Activity [0, 10], Sleep [0, 12])<br>• Visualizations: `correlation_heatmap.png`, `feature_distributions.png`, `eda_relationships.png` |
| **Module 3: Preprocessing & Scikit-Learn Pipeline** | Feature transformations, encoding, scaling | `ml/features.py` | • Custom `ClipTransformer` with `get_feature_names_out`<br>• `FunctionTransformer(np.log1p)` for skewed distributions<br>• `StandardScaler` for numeric columns<br>• `OrdinalEncoder` for `Stress_Level` (Low < Medium < High)<br>• `OneHotEncoder(handle_unknown="ignore")` for nominal categories |
| **Module 4: Model Training & Hyperparameter Optimization** | Model fitting, evaluation, RandomizedSearchCV, persistence | `ml/train.py`, `ml/evaluate.py`, `models/pipeline.joblib`, `models/metrics.json`, `models/metadata.json` | • 70/30 train/test split with `random_state=42`<br>• Baseline: Linear Regression ($R^2 = 0.7786$)<br>• Tuned Model: Random Forest Regressor ($R^2 = 0.8545$, $MAE = 4.8635$, $RMSE = 6.3409$)<br>• Automated best-model selection and artifact serialization |
| **Module 5: Backend REST API** | Decoupled API service, request validation, middleware | `backend/app/main.py`, `backend/app/schemas.py`, `backend/app/model_loader.py`, `backend/app/settings.py`, `backend/app/security.py` | • FastAPI + Uvicorn with lifespan startup model loading<br>• Strict Pydantic v2 schemas and validation ranges<br>• Rate limiting (`slowapi`), Security headers middleware, Request-ID tracing<br>• Endpoints: `GET /health`, `GET /metadata`, `GET /model-info`, `POST /predict` |
| **Module 6: Frontend User Interface** | Responsive web client, interactive inputs, score visualization, crisis resources | `frontend/index.html`, `frontend/styles.css`, `frontend/app.js`, `frontend/config.js` | • Dynamic form generation from `GET /metadata`<br>• Real-time client-side validation<br>• Animated SVG circular score meter / progress ring<br>• Render cold-start detection & retry banner<br>• Crisis helplines section (Tele-MANAS, Vandrevala, KIRAN) & ethical disclaimer |
| **Module 7: Cloud Deployment & CI/CD** | Production hosting, Blueprint automation, Continuous Integration | `render.yaml`, `.github/workflows/ci.yml`, `docs/DEPLOYMENT.md`, `Makefile` | • Render.com Blueprint: Web Service (FastAPI) + Static Site (Frontend)<br>• GitHub Actions CI pipeline running linting, training check, and pytest coverage suite |

---

## Architectural Evolution Note: Synopsis vs. Final Implementation

### Extended Input Features
The preliminary synopsis DFD depicted a simplified model with 3 basic inputs. The final production implementation extends this to 6 comprehensive lifestyle and behavioral dimensions:
1. `Physical_Activity_Hours` (Numeric)
2. `Sleep_Hours` (Numeric)
3. `Screen_Time_Hours` (Numeric)
4. `Stress_Level` (Ordinal Categorical: Low, Medium, High)
5. `Country` (Nominal Categorical)
6. `Platform` (Nominal Categorical)

This expansion significantly enhances prediction accuracy and realism while allowing robust handling of unseen demographic inputs via scikit-learn's `OneHotEncoder(handle_unknown="ignore")`.
