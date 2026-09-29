# Model Card — Mental Health Score Predictor

## Model Details

- **Model Name:** Mental Health Score Predictor
- **Version:** 1.0.0
- **Model Type:** Tuned Random Forest Regressor within a scikit-learn `Pipeline` (with ColumnTransformer preprocessing)
- **Baseline Model:** Ordinary Least Squares Linear Regression
- **Frameworks:** Python 3.11+, Scikit-Learn 1.4.2, NumPy, Pandas, Joblib
- **Training Date:** September 2026
- **License:** MIT License (Educational / Non-commercial demonstration)
- **Academic Context:** B.Tech Minor Project, Department of Computer Science & Engineering, Poornima University, Jaipur.

---

## Intended Use

### Primary Use Case
The model serves as an educational and informational self-screening web application. It takes basic daily lifestyle metrics (physical activity, sleep duration, screen time, perceived stress level, country, and primary social platform) and outputs an estimated mental wellness score between 0 and 100 with supportive lifestyle feedback.

### Out-of-Scope & Prohibited Uses
- **NOT A DIAGNOSTIC TOOL:** This software cannot diagnose clinical depression, anxiety disorders, psychiatric conditions, or any medical condition.
- **NOT CLINICAL DECISION SUPPORT:** Must not be used by medical practitioners or institutions for triage, diagnosis, prescription, or clinical evaluation.
- **NO HIGH-STAKES DECISIONS:** Must not be used for employment, insurance underwriting, academic evaluations, or healthcare eligibility.

---

## Dataset Description

### Data Source & Synthetic Disclaimer
> **IMPORTANT:** The dataset bundled with this repository was generated using `scripts/generate_synthetic_data.py` (5,000 samples) to demonstrate the complete machine learning engineering lifecycle without violating privacy regulations (HIPAA/GDPR). Results on this synthetic dataset are strictly for academic demonstration. For production deployment, users should replace `data/raw/mental_health_data.csv` with a verified, ethically consented dataset and re-run `make train`.

### Input Features & Types

| Feature Name | Type | Unit / Allowed Values | Preprocessing Pipeline Step |
|---|---|---|---|
| `Physical_Activity_Hours` | Continuous Numeric | Hours per day `[0.0, 10.0]` | Domain Clipping `[0.0, 10.0]`, Log1p Transform (`log(1+x)`), Standard Scaling |
| `Sleep_Hours` | Continuous Numeric | Hours per day `[0.0, 12.0]` | Domain Clipping `[0.0, 12.0]`, Standard Scaling |
| `Screen_Time_Hours` | Continuous Numeric | Hours per day `[0.0, 16.0]` | Domain Clipping `[0.0, 16.0]`, Standard Scaling |
| `Stress_Level` | Ordinal Categorical | `Low` < `Medium` < `High` | `OrdinalEncoder(categories=[['Low', 'Medium', 'High']])` |
| `Country` | Nominal Categorical | 10 countries / `Other` | `OneHotEncoder(handle_unknown='ignore')` |
| `Platform` | Nominal Categorical | 8 platforms / `None` | `OneHotEncoder(handle_unknown='ignore')` |

### Target Variable
- **`Mental_Health_Score`:** Continuous score normalized from 0.0 to 100.0, where higher scores represent higher self-reported positive wellness and lifestyle balance.

---

## Training and Evaluation Methodology

### Data Splitting Strategy
- **Split Ratio:** 70% Training (3,500 samples), 30% Test (1,500 samples)
- **Random Seed:** Fixed `random_state=42`
- **Data Leakage Prevention:** All scalers, imputers, and encoders are fit *strictly on the training set* inside the scikit-learn Pipeline. Outlier filtering using the IQR method (1.5 x IQR) is applied exclusively to training data (3,244 samples retained after removing 256 outlier rows).

### Hyperparameter Tuning
- Tuned via `RandomizedSearchCV` with 5-fold cross-validation (`cv=5`, `n_iter=30`, `n_jobs=-1`, `random_state=42`).
- **Optimal Hyperparameters Found:**
  - `n_estimators`: 500
  - `max_depth`: 10
  - `min_samples_split`: 10
  - `min_samples_leaf`: 2
  - `max_features`: `None`

---

## Performance Metrics (Actual Run Results)

| Metric | Baseline (Linear Regression) | Tuned Random Forest (Selected Best) |
|---|---|---|
| **Test $R^2$ (Coefficient of Determination)** | 0.7786 | **0.8545** |
| **5-Fold CV $R^2$ (Mean)** | 0.8361 | **0.8585** |
| **Test Mean Absolute Error (MAE)** | 5.6245 | **4.8635** |
| **Test Root Mean Squared Error (RMSE)** | 7.8212 | **6.3409** |

---

## Bias, Fairness & Limitations

1. **Self-Reporting Bias:** Lifestyle inputs (sleep, screen time, stress) are subjective and subject to recall bias.
2. **Simplified Representation:** Mental wellness is multifaceted (genetics, trauma, socioeconomic conditions, medical history) and cannot be captured solely through basic lifestyle metrics.
3. **Synthetic Skew:** Synthetic distributions may not capture non-linear interactions present in diverse global clinical populations.
4. **Platform & Country Representation:** The categorical features include popular global platforms, but local demographics may differ. `OneHotEncoder(handle_unknown='ignore')` ensures zero errors when unknown categories are provided at inference time.

---

## Crisis Resources & Support

If you or someone you know is struggling or in distress:
- **India:** Tele-MANAS (`14416` or `1800 891 4416`), Vandrevala Foundation (`9999 666 555`), KIRAN (`1800-599-0019`)
- **United States:** `988` Suicide & Crisis Lifeline
- **United Kingdom:** `111` NHS Mental Health Services or `116 123` (Samaritans)
- **International:** Find local support at [https://findahelpline.com](https://findahelpline.com)
