# Viva Voce Q&A Cheat Sheet — Mental Health Score Predictor

**Project:** Mental Health Score Predictor  
**Department:** Computer Science & Engineering, Poornima University, Jaipur  
**Student Reference:** Final Minor Project Evaluation & Defense  

---

## 1. Machine Learning & Feature Engineering

### Q1: Why did you choose the Interquartile Range (IQR) method for outlier removal?
> **Answer:**  
> The IQR method ($IQR = Q_3 - Q_1$, with bounds $[Q_1 - 1.5 \times IQR, Q_3 + 1.5 \times IQR]$) is a non-parametric, robust outlier filtering technique. Unlike Z-score methods, IQR does not assume a strictly Gaussian (normal) distribution and is resilient to extreme skewness. Furthermore, we apply IQR outlier removal **strictly on the training dataset split**, never on the test split or at live inference time, ensuring that data leakage is prevented.

### Q2: Why is the `log1p` transformation used on `Physical_Activity_Hours`?
> **Answer:**  
> Daily physical activity exhibits strong positive (right) skewness because most individuals report low-to-moderate values (0.5–2 hours) while a smaller subset engages in prolonged athletics (4–8 hours). `log1p(x) = ln(1 + x)` compresses long right tails, stabilizes feature variance, prevents $\ln(0)$ undefined errors for zero values, and makes linear relationships more symmetric for regression models.

### Q3: What is the difference between `OrdinalEncoder` and `OneHotEncoder`, and why did you use both?
> **Answer:**  
> - **OrdinalEncoder:** Used for `Stress_Level` (`Low` < `Medium` < `High`) because there is a natural, monotonic mathematical hierarchy ($0 < 1 < 2$). Imposing ordinal ranks preserves intrinsic ordering.
> - **OneHotEncoder:** Used for nominal features (`Country`, `Platform`) where categories have no inherent order. Using integer encoding here would mistakenly impose arbitrary mathematical distances. We also configure `handle_unknown='ignore'` so unseen categories at runtime default to all zeros rather than crashing the API.

### Q4: Why did Random Forest outperform Linear Regression?
> **Answer:**  
> Linear Regression assumes strictly linear, additive relationships between independent predictors and the wellness score. In human psychology and wellness, however, factors interact non-linearly (e.g., high screen time combined with high stress compounds negatively much more severely than each in isolation). Random Forest is an ensemble of decision trees that inherently models complex non-linear feature interactions and feature thresholds without requiring manual interaction terms.

### Q5: Why `RandomizedSearchCV` instead of `GridSearchCV`?
> **Answer:**  
> `GridSearchCV` evaluates every single combination across the Cartesian product of hyperparameters ($O(N^k)$), which is computationally wasteful when some hyperparameter dimensions (like `min_samples_leaf`) have higher sensitivity than others. `RandomizedSearchCV` samples fixed iterations ($N=30$ across 5 folds = 150 fits) randomly across the parameter space, finding near-optimal configurations in a fraction of the computational time and avoiding wasted grid points.

---

## 2. Software Architecture & System Design

### Q6: Why is the project built with a decoupled architecture (Separate FastAPI backend & Static Frontend)?
> **Answer:**  
> 1. **Separation of Concerns:** ML inference, preprocessing, and validation logic reside securely on the backend, while UI rendering and client-side interactions are managed by the browser.
> 2. **Independent Scalability:** Static assets are served with low latency via global CDNs/static hosts, while the compute-heavy FastAPI backend scales independently based on CPU/RAM inference demand.
> 3. **API Reusability:** The FastAPI backend can serve future mobile apps (Android/iOS), IoT wellness wearables, or external microservices via standard JSON REST endpoints without modifying frontend code.

### Q7: What is CORS and why is it configured strictly?
> **Answer:**  
> Cross-Origin Resource Sharing (CORS) is a browser security mechanism that restricts web pages from making AJAX requests to a different domain/port than the one that served the web page. In development, origins like `localhost:5500` or `127.0.0.1:8000` are permitted. In production on Render, CORS is constrained via the `ALLOWED_ORIGINS` environment variable to only permit requests from our authorized frontend (`https://mhsp-ui.onrender.com`), preventing unauthorized third-party domains from abusing our ML inference API.

### Q8: How does the system handle Render free-tier cold starts?
> **Answer:**  
> Free tier containers on Render sleep after 15 minutes of inactivity. When a user opens the frontend, `app.js` issues a non-blocking `GET /health` probe. If the server is spinning up, the UI displays an animated *"Connecting to server..."* alert banner and employs an exponential-backoff retry loop. Once `/health` responds with `200 OK`, the banner clears and the form enables smoothly.

---

## 3. Ethics & Responsible AI

### Q9: How are data privacy and ethics addressed in this application?
> **Answer:**  
> 1. **Zero Data Retention:** No user demographic or behavioral inputs are written to databases, local storage, or persistent logs.
> 2. **Input Privacy in Logging:** Structured application logs specifically sanitize payloads to never print user lifestyle details.
> 3. **Clear Non-Diagnostic Framing:** Visible, persistent disclaimers emphasize that this tool is strictly informational and provide verified crisis helpline numbers (e.g. Tele-MANAS `14416` in India) for anyone in distress.
