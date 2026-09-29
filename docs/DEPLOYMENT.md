# Deployment Guide — Render.com Blueprint

This guide provides click-by-click instructions for deploying the **Mental Health Score Predictor** application to Render.com using the included `render.yaml` Blueprint on the **Free Tier**.

---

## Architecture Overview on Render

```
                                  ┌───────────────────────────┐
                                  │   Static Site (Frontend)  │
                                  │   HTML / CSS / Vanilla JS │
                                  │   mhsp-ui.onrender.com    │
                                  └─────────────┬─────────────┘
                                                │ HTTPS / REST (CORS)
                                                ▼
                                  ┌───────────────────────────┐
                                  │   Web Service (FastAPI)   │
                                  │   Python 3.11 / Uvicorn   │
                                  │   mhsp-backend.onrender...│
                                  └─────────────┬─────────────┘
                                                │
                                                ▼
                                  ┌───────────────────────────┐
                                  │ Pre-trained Model Artifact│
                                  │ models/pipeline.joblib    │
                                  └───────────────────────────┘
```

---

## Step-by-Step Deployment (Click-by-Click)

### Step 1: Push the Repository to GitHub
Ensure all committed code, including the trained model artifact (`models/pipeline.joblib`), `render.yaml`, and documentation, is pushed to your GitHub repository:
```bash
git add .
git commit -m "feat: complete mental health score predictor production setup"
git push origin main
```

---

### Step 2: Create a Free Account on Render
1. Visit [https://render.com](https://render.com) and sign up using your GitHub account.

---

### Step 3: Deploy via Render Blueprint
1. In the Render Dashboard, click **New +** (top right) and select **Blueprint**.
2. Connect your GitHub repository: `mental-health-score-predictor`.
3. Render automatically discovers `render.yaml` at the root of your repo.
4. Render will create two services:
   - **`mhsp-backend`** (Web Service, Python 3.11)
   - **`mhsp-ui`** (Static Site)
5. Click **Apply**.

---

### Step 4: Configure Environment Variables & CORS
1. Once both services are provisioning, copy the URL of your static site (e.g., `https://mhsp-ui.onrender.com`).
2. Navigate to **mhsp-backend** > **Environment**.
3. Set or update the following environment variables:
   - `ALLOWED_ORIGINS`: `https://mhsp-ui.onrender.com,http://localhost:8000,http://127.0.0.1:5500`
   - `DEBUG`: `False`
4. Click **Save Changes** (this automatically redeploys the backend).

---

### Step 5: Update Frontend API Base URL
1. Open `frontend/config.js` in your repository.
2. Update the `API_BASE_URL` to point to your live Render backend:
   ```javascript
   // frontend/config.js
   const CONFIG = {
     API_BASE_URL: "https://mhsp-backend.onrender.com",
     REQUEST_TIMEOUT_MS: 30000,
   };
   ```
3. Commit and push the change to GitHub:
   ```bash
   git add frontend/config.js
   git commit -m "chore: set production backend URL"
   git push origin main
   ```
4. Render will automatically redeploy the static frontend within 30 seconds.

---

### Step 6: Verify Live Deployment
1. Visit `https://mhsp-backend.onrender.com/health` in your browser. You should receive:
   ```json
   {"status": "healthy", "model_loaded": true, "version": "1.0.0"}
   ```
2. Visit `https://mhsp-ui.onrender.com/` in your browser.
3. Submit a test prediction to verify end-to-end functionality.

---

## Troubleshooting & Common Pitfalls

### 1. Free Tier Cold Starts
- **Issue:** On the Render free tier, web services spin down after 15 minutes of inactivity. The first request may take 30–50 seconds.
- **Handled in Code:** The frontend has a built-in health check polling mechanism (`checkBackendHealth`) that displays a friendly *"Waking up server..."* animated banner with automatic retries before the user submits a form.

### 2. CORS Errors
- **Issue:** Browser console displays `Cross-Origin Request Blocked`.
- **Solution:** Verify that the frontend domain (e.g. `https://mhsp-ui.onrender.com` without trailing slash) is included in `ALLOWED_ORIGINS` in the backend environment settings on Render.

### 3. Scikit-learn Version Incompatibility
- **Issue:** `ValueError: Incompatible scikit-learn version`.
- **Solution:** Python and scikit-learn versions are strictly pinned in `backend/requirements.txt` (`scikit-learn==1.4.2` or matching). If retraining locally on a different scikit-learn version, always regenerate `models/pipeline.joblib` and update `requirements.txt`.
