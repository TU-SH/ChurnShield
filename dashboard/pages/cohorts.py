"""Cohort analysis page — retention heatmap, churn by plan type."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path


@st.cache_data(ttl=300)
def load_data() -> pd.DataFrame:
    csv = Path("data/raw/customers.csv")
    if csv.exists():
        df = pd.read_csv(csv)
        np.random.seed(99)
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
        # Tenure buckets
        df["tenure_bucket"] = pd.cut(
            df["account_length_days"],
            bins=[0, 90, 180, 365, 730, 1460, 9999],
            labels=["0–3 mo", "3–6 mo", "6–12 mo", "1–2 yr", "2–4 yr", "4+ yr"],
        )
        # CS calls bucket
        df["cs_calls_bucket"] = pd.cut(
            df["customer_service_calls"],
            bins=[-1, 0, 1, 2, 3, 100],
            labels=["0", "1", "2", "3", "4+"],
        )
        return df
    st.error("Run 'python generate_data.py' first.")
    st.stop()


def show():
    st.title("👥 Cohort Analysis")
    st.caption("Churn behaviour across tenure, plan type, and service call frequency")
    st.divider()

    df = load_data()

    # ── Row 1: Churn rate by tenure ───────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn rate by customer tenure")
        tenure_churn = (
            df.groupby("tenure_bucket", observed=True)["churned"]
            .agg(["sum", "count"])
            .reset_index()
        )
        tenure_churn["churn_rate"] = tenure_churn["sum"] / tenure_churn["count"]
        fig = px.bar(
            tenure_churn, x="tenure_bucket", y="churn_rate",
            labels={"tenure_bucket": "Tenure", "churn_rate": "Churn rate"},
            color="churn_rate", color_continuous_scale=["#27ae60", "#f39c12", "#e74c3c"],
            text=tenure_churn["churn_rate"].map("{:.1%}".format),
            height=340,
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(coloraxis_showscale=False, plot_bgcolor="rgba(0,0,0,0)",
                          yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Churn rate by customer service calls")
        cs_churn = (
            df.groupby("cs_calls_bucket", observed=True)["churned"]
            .agg(["sum", "count"])
            .reset_index()
        )
        cs_churn["churn_rate"] = cs_churn["sum"] / cs_churn["count"]
        fig2 = px.bar(
            cs_churn, x="cs_calls_bucket", y="churn_rate",
            labels={"cs_calls_bucket": "Customer service calls", "churn_rate": "Churn rate"},
            color="churn_rate", color_continuous_scale=["#27ae60", "#f39c12", "#e74c3c"],
            text=cs_churn["churn_rate"].map("{:.1%}".format),
            height=340,
        )
        fig2.update_traces(textposition="outside")
        fig2.update_layout(coloraxis_showscale=False, plot_bgcolor="rgba(0,0,0,0)",
                           yaxis_tickformat=".0%")
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── Row 2: Plan type breakdown ────────────────────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Churn rate by plan type")
        plan_labels = {
            (False, False): "No plans",
            (True,  False): "Intl only",
            (False, True):  "Voicemail only",
            (True,  True):  "Both plans",
        }
        df["plan_type"] = df.apply(
            lambda r: plan_labels[(r["international_plan"], r["voicemail_plan"])], axis=1
        )
        plan_churn = (
            df.groupby("plan_type")["churned"]
            .agg(["sum", "count"])
            .reset_index()
        )
        plan_churn["churn_rate"] = plan_churn["sum"] / plan_churn["count"]
        fig3 = px.bar(
            plan_churn.sort_values("churn_rate", ascending=False),
            x="plan_type", y="churn_rate",
            labels={"plan_type": "Plan type", "churn_rate": "Churn rate"},
            color="churn_rate", color_continuous_scale=["#27ae60", "#f39c12", "#e74c3c"],
            text=plan_churn.sort_values("churn_rate", ascending=False)["churn_rate"].map("{:.1%}".format),
            height=340,
        )
        fig3.update_traces(textposition="outside")
        fig3.update_layout(coloraxis_showscale=False, plot_bgcolor="rgba(0,0,0,0)",
                           yaxis_tickformat=".0%")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("Churn rate heatmap: tenure × CS calls")
        pivot = (
            df.groupby(["tenure_bucket", "cs_calls_bucket"], observed=True)["churned"]
            .mean()
            .reset_index()
            .pivot(index="tenure_bucket", columns="cs_calls_bucket", values="churned")
        )
        fig4 = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale=[[0,"#27ae60"],[0.5,"#f39c12"],[1,"#e74c3c"]],
            text=[[f"{v:.1%}" if not np.isnan(v) else "" for v in row] for row in pivot.values],
            texttemplate="%{text}",
            showscale=True,
            colorbar=dict(title="Churn rate", tickformat=".0%"),
        ))
        fig4.update_layout(
            height=340,
            xaxis_title="Customer service calls",
            yaxis_title="Tenure",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.divider()

    # ── Summary stats table ───────────────────────────────────────────────────
    st.subheader("Cohort summary statistics")
    summary = (
        df.groupby("risk_segment", observed=True)
        .agg(
            customers=("customer_id", "count"),
            avg_tenure_days=("account_length_days", "mean"),
            avg_charge_aud=("total_charge_aud", "mean"),
            avg_cs_calls=("customer_service_calls", "mean"),
            churn_rate=("churned", "mean"),
        )
        .reset_index()
    )
    summary["avg_tenure_days"] = summary["avg_tenure_days"].round(0).astype(int)
    summary["avg_charge_aud"]  = summary["avg_charge_aud"].round(2)
    summary["avg_cs_calls"]    = summary["avg_cs_calls"].round(2)
    summary["churn_rate"]      = summary["churn_rate"].map("{:.1%}".format)
    summary.columns = ["Risk segment","Customers","Avg tenure (days)","Avg charge (AUD)","Avg CS calls","Actual churn rate"]
    st.dataframe(summary, use_container_width=True)
