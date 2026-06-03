"""
ChurnShield Dashboard — Streamlit entry point.
Run: streamlit run dashboard/app.py
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

st.set_page_config(
    page_title="ChurnShield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=60)
st.sidebar.title("ChurnShield")
st.sidebar.caption("AU Telco Churn Intelligence Platform")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    ["📊 Overview", "👥 Cohort Analysis", "🔍 Explain Customer", "🤖 Live Predict"],
)

st.sidebar.divider()
st.sidebar.caption("Built with XGBoost · MLflow · FastAPI · PostgreSQL")

# ── Route to pages ────────────────────────────────────────────────────────────────
if page == "📊 Overview":
    from dashboard.pages import overview
    overview.show()
elif page == "👥 Cohort Analysis":
    from dashboard.pages import cohorts
    cohorts.show()
elif page == "🔍 Explain Customer":
    from dashboard.pages import explain
    explain.show()
elif page == "🤖 Live Predict":
    from dashboard.pages import live_predict
    live_predict.show()
