/**
 * frontend/config.js
 * ------------------
 * Change API_BASE_URL to your Render backend URL before deploying.
 * For local development, leave it pointing to localhost.
 */

const CONFIG = {
  // Local development
  API_BASE_URL: "http://localhost:8000",

  // Production (uncomment after deploying to Render and replace with your URL)
  // API_BASE_URL: "https://your-backend.onrender.com",

  // Timeout for the initial health-check / cold-start retry (ms)
  COLD_START_TIMEOUT_MS: 60000,

  // How long to wait between cold-start retries (ms)
  RETRY_INTERVAL_MS: 3000,

  // Maximum number of cold-start retries before giving up
  MAX_RETRIES: 20,
};
