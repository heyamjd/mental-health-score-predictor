(() => {
  "use strict";

  // Connect to mental-health-score-predictor FastAPI backend (port 8000)
  const API_BASE = "http://127.0.0.1:8000";

  const form = document.getElementById("predict-form");
  const submitBtn = document.getElementById("submit-btn");
  const resetBtn = document.getElementById("reset-btn");
  const errorRetryBtn = document.getElementById("error-retry-btn");

  const stateIdle = document.getElementById("state-idle");
  const stateLoading = document.getElementById("state-loading");
  const stateResult = document.getElementById("state-result");
  const stateError = document.getElementById("state-error");

  const scoreNumberEl = document.getElementById("score-number");
  const scoreBandEl = document.getElementById("score-band");
  const scoreContextEl = document.getElementById("score-context");
  const gaugeFill = document.getElementById("gauge-fill");
  const errorCopyEl = document.getElementById("error-copy");

  const GAUGE_ARC_LENGTH = 314; // approx pi * r(100)

  // ---------------------------------------------------------
  // Draw tick marks on both gauges (0..10, every 2 units)
  // ---------------------------------------------------------
  function drawTicks() {
    document.querySelectorAll(".gauge-ticks").forEach((g) => {
      g.innerHTML = "";
      const cx = 120, cy = 140, rOuter = 100, rInner = 90;
      for (let i = 0; i <= 10; i += 2) {
        const angle = Math.PI - (i / 10) * Math.PI; // 180deg -> 0deg
        const x1 = cx + rOuter * Math.cos(angle);
        const y1 = cy - rOuter * Math.sin(angle);
        const x2 = cx + rInner * Math.cos(angle);
        const y2 = cy - rInner * Math.sin(angle);
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", x1.toFixed(1));
        line.setAttribute("y1", y1.toFixed(1));
        line.setAttribute("x2", x2.toFixed(1));
        line.setAttribute("y2", y2.toFixed(1));
        g.appendChild(line);
      }
    });
  }
  drawTicks();

  // ---------------------------------------------------------
  // Segmented control (stress_level) wiring
  // ---------------------------------------------------------
  const segGroup = document.getElementById("stress_level_group");
  const stressHiddenInput = document.getElementById("stress_level");
  if (segGroup && stressHiddenInput) {
    segGroup.querySelectorAll(".seg-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        segGroup.querySelectorAll(".seg-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        stressHiddenInput.value = btn.dataset.value;
        clearFieldError(stressHiddenInput);
      });
    });
  }

  // ---------------------------------------------------------
  // Field-level error helpers
  // ---------------------------------------------------------
  function fieldWrapper(input) {
    return input ? input.closest(".field") : null;
  }

  function setFieldError(input, message) {
    const wrap = fieldWrapper(input);
    if (!wrap) return;
    wrap.classList.add("field-error");
    const msgEl = wrap.querySelector(".error-msg");
    if (msgEl) msgEl.textContent = message;
  }

  function clearFieldError(input) {
    const wrap = fieldWrapper(input);
    if (!wrap) return;
    wrap.classList.remove("field-error");
    const msgEl = wrap.querySelector(".error-msg");
    if (msgEl) msgEl.textContent = "";
  }

  function clearAllErrors() {
    form.querySelectorAll(".field").forEach((f) => f.classList.remove("field-error"));
    form.querySelectorAll(".error-msg").forEach((m) => (m.textContent = ""));
  }

  // ---------------------------------------------------------
  // Client-side validation for the 6 core features
  // ---------------------------------------------------------
  function validate(payload) {
    const errors = [];

    const numericChecks = [
      ["sleep_hours", 0, 12, "Sleep must be 0–12 hrs."],
      ["physical_activity_hours", 0, 10, "Activity must be 0–10 hrs."],
      ["screen_time_hours", 0, 16, "Screen time must be 0–16 hrs."],
    ];

    numericChecks.forEach(([id, min, max, msg]) => {
      const input = document.getElementById(id);
      const val = parseFloat(input.value);
      if (input.value === "" || isNaN(val)) {
        errors.push([input, "This field is required."]);
      } else if (val < min || val > max) {
        errors.push([input, msg]);
      }
    });

    const countryInput = document.getElementById("country");
    if (!countryInput.value.trim()) {
      errors.push([countryInput, "Please enter your country."]);
    }

    if (!payload.Stress_Level) {
      errors.push([stressHiddenInput, "Pick a stress level."]);
    }

    return errors;
  }

  // ---------------------------------------------------------
  // Gather form data for the 6 features
  // ---------------------------------------------------------
  function collectPayload() {
    return {
      Physical_Activity_Hours: parseFloat(document.getElementById("physical_activity_hours").value) || 0,
      Sleep_Hours: parseFloat(document.getElementById("sleep_hours").value) || 0,
      Screen_Time_Hours: parseFloat(document.getElementById("screen_time_hours").value) || 0,
      Stress_Level: stressHiddenInput.value || "Medium",
      Country: (document.getElementById("country").value || "India").trim(),
      Platform: document.getElementById("platform").value || "Instagram",
    };
  }

  // ---------------------------------------------------------
  // UI state switching
  // ---------------------------------------------------------
  function showState(name) {
    [stateIdle, stateLoading, stateResult, stateError].forEach((el) => {
      if (el) el.hidden = true;
    });
    const target = { idle: stateIdle, loading: stateLoading, result: stateResult, error: stateError }[name];
    if (target) target.hidden = false;
  }

  function setSubmitting(isSubmitting) {
    submitBtn.disabled = isSubmitting;
    submitBtn.classList.toggle("loading", isSubmitting);
  }

  function renderResult(score, statusFlag, message) {
    const clamped = Math.max(0, Math.min(100, score));

    scoreNumberEl.textContent = score.toFixed(1);
    scoreBandEl.textContent = `Signal: ${statusFlag.toLowerCase()}`;
    scoreContextEl.textContent = message || "Your habits reflect a balanced daily rhythm.";

    // Animate arc fill
    gaugeFill.style.transition = "none";
    gaugeFill.style.strokeDashoffset = String(GAUGE_ARC_LENGTH);
    requestAnimationFrame(() => {
      gaugeFill.style.transition = "stroke-dashoffset 1.2s cubic-bezier(0.2, 0.9, 0.3, 1)";
      const offset = GAUGE_ARC_LENGTH * (1 - clamped / 100);
      gaugeFill.style.strokeDashoffset = String(offset);
    });

    showState("result");
  }

  function renderError(label, copy) {
    errorCopyEl.textContent = copy;
    showState("error");
  }

  // ---------------------------------------------------------
  // Submit handler
  // ---------------------------------------------------------
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    clearAllErrors();

    const payload = collectPayload();
    const clientErrors = validate(payload);

    if (clientErrors.length > 0) {
      clientErrors.forEach(([input, msg]) => input && setFieldError(input, msg));
      clientErrors[0][0]?.focus?.();
      return;
    }

    setSubmitting(true);
    showState("loading");

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => null);
        const detailMsg = (body && typeof body.detail === "string") ? body.detail : `Server error (${res.status})`;
        renderError("Prediction failed", detailMsg);
        return;
      }

      const data = await res.json();
      const score = typeof data.mental_health_score === "number" ? data.mental_health_score : 50;
      const status = data.status_flag || "Moderate";
      const msg = data.message || "Habit evaluation complete.";

      renderResult(score, status, msg);
    } catch (err) {
      renderError(
        "Can't reach the server",
        `Couldn't connect to ${API_BASE}. Make sure the backend is running.`
      );
    } finally {
      setSubmitting(false);
    }
  });

  // Live-clear errors
  form.querySelectorAll("input, select").forEach((el) => {
    el.addEventListener("input", () => clearFieldError(el));
    el.addEventListener("change", () => clearFieldError(el));
  });

  if (resetBtn) {
    resetBtn.addEventListener("click", () => {
      showState("idle");
    });
  }

  if (errorRetryBtn) {
    errorRetryBtn.addEventListener("click", () => {
      showState("idle");
    });
  }
})();
