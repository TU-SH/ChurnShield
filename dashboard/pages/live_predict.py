"""Live Predict page — submit a customer via the API and see the result."""
import requests
import streamlit as st

AU_STATES = ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"]

# Hardcoded — no dependency on .env or config.py
API_URL = "http://localhost:8000"


def check_api_status() -> tuple[bool, str]:
    """Check if API is reachable. Returns (is_up, message)."""
    try:
        resp = requests.get(f"{API_URL}/health", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            return True, f"Model: {data.get('model', 'unknown')} | Version: {data.get('version', '?')}"
        return False, f"API responded with status {resp.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "API is not running on localhost:8000"
    except Exception as e:
        return False, str(e)


def show():
    st.title("🤖 Live Predict")
    st.caption("Submit a customer record to the FastAPI endpoint and see real-time prediction")

    # ── API status indicator — always shown at top ────────────────────────────
    st.divider()
    is_up, status_msg = check_api_status()

    if is_up:
        st.success(f"✅ API is running — {status_msg}")
    else:
        st.error(
            f"❌ API is not reachable — {status_msg}\n\n"
            "**Fix:** Open a new terminal, activate your venv, and run:\n\n"
            "```\nuvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000\n```\n\n"
            "Keep that terminal open, then come back here and refresh the page."
        )
        st.stop()  # Don't show the form if API is down

    st.divider()

    # ── Input form ────────────────────────────────────────────────────────────
    with st.form("predict_form"):
        st.subheader("Customer details")
        col1, col2, col3 = st.columns(3)

        with col1:
            customer_id  = st.text_input("Customer ID", value="AU-TEST-999")
            state        = st.selectbox("State", AU_STATES, index=0)
            account_days = st.number_input("Account length (days)", 1, 3650, 365)
            intl_plan    = st.checkbox("International plan")
            vm_plan      = st.checkbox("Voicemail plan")
            vm_messages  = st.number_input("Voicemail messages", 0, 50, 0)
            cs_calls     = st.number_input("Customer service calls", 0, 15, 1)

        with col2:
            st.markdown("**Day usage**")
            day_mins   = st.number_input("Day minutes",    0.0, 500.0, 180.5)
            day_calls  = st.number_input("Day calls",      0,   300,   90)
            day_charge = st.number_input("Day charge ($)", 0.0, 100.0, 30.69)

            st.markdown("**Evening usage**")
            eve_mins   = st.number_input("Evening minutes",    0.0, 500.0, 200.1)
            eve_calls  = st.number_input("Evening calls",      0,   300,   90)
            eve_charge = st.number_input("Evening charge ($)", 0.0, 60.0,  17.01)

        with col3:
            st.markdown("**Night usage**")
            night_mins   = st.number_input("Night minutes",    0.0, 500.0, 201.4)
            night_calls  = st.number_input("Night calls",      0,   300,   90)
            night_charge = st.number_input("Night charge ($)", 0.0, 30.0,  9.06)

            st.markdown("**International usage**")
            intl_mins   = st.number_input("Intl minutes",    0.0, 100.0, 0.0)
            intl_calls  = st.number_input("Intl calls",      0,   50,    0)
            intl_charge = st.number_input("Intl charge ($)", 0.0, 30.0,  0.0)

        submitted = st.form_submit_button(
            "🔮 Predict churn risk", use_container_width=True, type="primary"
        )

    # ── Result ────────────────────────────────────────────────────────────────
    if submitted:
        payload = {
            "customer_id":            customer_id,
            "state":                  state,
            "account_length_days":    int(account_days),
            "international_plan":     intl_plan,
            "voicemail_plan":         vm_plan,
            "voicemail_messages":     int(vm_messages),
            "day_mins":               float(day_mins),
            "day_calls":              int(day_calls),
            "day_charge_aud":         float(day_charge),
            "evening_mins":           float(eve_mins),
            "evening_calls":          int(eve_calls),
            "evening_charge_aud":     float(eve_charge),
            "night_mins":             float(night_mins),
            "night_calls":            int(night_calls),
            "night_charge_aud":       float(night_charge),
            "intl_mins":              float(intl_mins),
            "intl_calls":             int(intl_calls),
            "intl_charge_aud":        float(intl_charge),
            "customer_service_calls": int(cs_calls),
        }

        with st.spinner("Calling API..."):
            try:
                resp = requests.post(
                    f"{API_URL}/predict",
                    json=payload,
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()

            except requests.exceptions.ConnectionError:
                st.error("Lost connection to API. Is the API terminal still running?")
                st.stop()
            except requests.exceptions.HTTPError as e:
                st.error(f"API returned an error: {e}")
                st.code(resp.text)
                st.stop()
            except Exception as e:
                st.error(f"Unexpected error: {e}")
                st.stop()

        # ── Display results ───────────────────────────────────────────────────
        st.divider()
        prob    = data["churn_probability"]
        risk    = data["risk_segment"]
        latency = data["latency_ms"]
        risk_emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(risk, "⚪")

        # Big result banner
        if risk == "HIGH":
            st.error(f"## {risk_emoji} HIGH RISK — {prob:.1%} churn probability")
        elif risk == "MEDIUM":
            st.warning(f"## {risk_emoji} MEDIUM RISK — {prob:.1%} churn probability")
        else:
            st.success(f"## {risk_emoji} LOW RISK — {prob:.1%} churn probability")

        # KPI metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Churn probability", f"{prob:.1%}")
        c2.metric("Risk segment",      f"{risk_emoji} {risk}")
        c3.metric("Predicted churn",   "Yes ⚠️" if data["churn_predicted"] else "No ✅")
        c4.metric("API latency",       f"{latency} ms")

        # SHAP risk factors
        st.subheader("Top 5 risk drivers")
        st.caption("Positive values push toward churn. Negative values reduce churn risk.")

        for i, (feat, val) in enumerate(data["top_risk_factors"], 1):
            direction  = "⬆️ increases risk" if val > 0 else "⬇️ reduces risk"
            bar_pct    = min(abs(val) * 500, 1.0)   # scale for progress bar
            col_l, col_r = st.columns([3, 1])
            col_l.markdown(f"**{i}. {feat}** — {direction}")
            col_r.markdown(f"`{val:+.4f}`")
            st.progress(bar_pct)

        # Raw response
        with st.expander("📄 Raw API response (JSON)"):
            st.json(data)
