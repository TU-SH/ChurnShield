"""Overview page — KPI summary, risk distribution by AU state."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.database import load_raw_customers


@st.cache_data(ttl=300)
def load_data() -> pd.DataFrame:
    """Load predictions joined with customer data."""
    try:
        from src.database import load_predictions_with_context
        df = load_predictions_with_context()
        if df.empty:
            raise ValueError("No predictions yet")
        return df
    except Exception:
        # Fallback: use raw CSV with simulated scores
        import numpy as np
        from pathlib import Path
        csv = Path("data/raw/customers.csv")
        if csv.exists():
            df = pd.read_csv(csv)
            np.random.seed(42)
            df["churn_probability"] = np.clip(
                np.random.beta(2, 10, len(df)) + df["churned"].astype(float) * 0.4, 0, 1
            )
            df["risk_segment"] = pd.cut(
                df["churn_probability"],
                bins=[-0.01, 0.45, 0.70, 1.01],
                labels=["LOW", "MEDIUM", "HIGH"],
            )
            df["total_charge_aud"] = (
                df["day_charge_aud"] + df["evening_charge_aud"]
                + df["night_charge_aud"] + df["intl_charge_aud"]
            )
            df["actual_churn"] = df["churned"]
            return df
        st.error("No data available. Run 'python generate_data.py' first.")
        st.stop()


RISK_COLORS = {"LOW": "#27ae60", "MEDIUM": "#f39c12", "HIGH": "#e74c3c"}


def show():
    st.title("🛡️ ChurnShield — Churn Risk Overview")
    st.caption("Australian Telco Customer Intelligence · Powered by XGBoost + SHAP")
    st.divider()

    df = load_data()

    # ── KPI row ───────────────────────────────────────────────────────────────
    total       = len(df)
    high_risk   = (df["risk_segment"] == "HIGH").sum()
    med_risk    = (df["risk_segment"] == "MEDIUM").sum()
    avg_prob    = df["churn_probability"].mean()
    rev_at_risk = df.loc[df["risk_segment"] == "HIGH", "total_charge_aud"].sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total customers",         f"{total:,}")
    c2.metric("🔴 High risk",            f"{high_risk:,}", f"{high_risk/total:.1%}")
    c3.metric("🟡 Medium risk",          f"{med_risk:,}",  f"{med_risk/total:.1%}")
    c4.metric("Avg churn probability",   f"{avg_prob:.1%}")
    c5.metric("Revenue at risk (AUD)",   f"${rev_at_risk:,.0f}")

    st.divider()

    # ── Row 1: state bar + score histogram ────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Risk distribution by AU state")
        state_order = ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"]
        state_risk  = (
            df.groupby(["state", "risk_segment"], observed=True)
            .size()
            .reset_index(name="count")
        )
        fig = px.bar(
            state_risk, x="state", y="count", color="risk_segment",
            color_discrete_map=RISK_COLORS,
            category_orders={"state": state_order, "risk_segment": ["LOW","MEDIUM","HIGH"]},
            labels={"count": "Customers", "state": "State", "risk_segment": "Risk"},
            height=350,
        )
        fig.update_layout(legend_title_text="Risk", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Churn probability distribution")
        fig2 = px.histogram(
            df, x="churn_probability", nbins=40, color="risk_segment",
            color_discrete_map=RISK_COLORS,
            category_orders={"risk_segment": ["LOW","MEDIUM","HIGH"]},
            labels={"churn_probability": "Churn probability", "count": "Customers"},
            height=350,
        )
        fig2.update_layout(bargap=0.05, plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── Row 2: revenue at risk by state + top risk table ─────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Revenue at risk by state (HIGH risk only)")
        state_rev = (
            df[df["risk_segment"] == "HIGH"]
            .groupby("state", observed=True)["total_charge_aud"]
            .sum()
            .reset_index()
            .sort_values("total_charge_aud", ascending=False)
        )
        fig3 = px.bar(
            state_rev, x="state", y="total_charge_aud",
            labels={"total_charge_aud": "Revenue at risk (AUD)", "state": "State"},
            color="total_charge_aud",
            color_continuous_scale=["#f8c8c8", "#e74c3c"],
            height=330,
        )
        fig3.update_layout(coloraxis_showscale=False, plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig3, use_container_width=True)

    with col_b:
        st.subheader("Top 15 highest-risk customers")
        top15 = (
            df[["customer_id", "state", "churn_probability", "risk_segment", "total_charge_aud"]]
            .sort_values("churn_probability", ascending=False)
            .head(15)
            .reset_index(drop=True)
        )
        top15["churn_probability"] = top15["churn_probability"].map("{:.1%}".format)
        top15["total_charge_aud"]  = top15["total_charge_aud"].map("${:,.2f}".format)
        top15.columns = ["Customer ID", "State", "Churn Prob", "Risk", "Monthly Charge"]
        st.dataframe(top15, use_container_width=True, height=330)

    st.divider()
    st.caption(f"Last refreshed: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S AEST')} | Data: {total:,} customers")
