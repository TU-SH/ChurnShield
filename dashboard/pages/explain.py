"""Explain Customer page — SHAP waterfall plot for individual customers."""
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path


@st.cache_data(ttl=300)
def load_customers() -> pd.DataFrame:
    csv = Path("data/raw/customers.csv")
    if csv.exists():
        return pd.read_csv(csv)
    st.error("Run 'python generate_data.py' first.")
    st.stop()


def compute_shap_on_the_fly(customer_id: str, df: pd.DataFrame) -> dict:
    """Compute SHAP values for a customer using the loaded model."""
    from src.features.engineering import engineer_features_single, get_feature_matrix, FEATURE_COLUMNS
    from src.models.predict import load_artefact

    row = df[df["customer_id"] == customer_id].copy()
    feats = engineer_features_single(row)
    X = get_feature_matrix(feats)

    art = load_artefact()
    shap_vals = art["explainer"].shap_values(X)[0]
    return dict(zip(FEATURE_COLUMNS, [round(float(v), 4) for v in shap_vals]))


def shap_waterfall(shap_dict: dict, base_value: float, pred_prob: float, customer_id: str):
    """Render a SHAP waterfall chart using Plotly."""
    items = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    features = [i[0] for i in items]
    values   = [i[1] for i in items]

    colors = ["#e74c3c" if v > 0 else "#27ae60" for v in values]
    labels = [f"+{v:.3f}" if v > 0 else f"{v:.3f}" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation="h",
        marker_color=colors,
        text=labels,
        textposition="outside",
    ))
    fig.add_vline(x=0, line_width=1, line_color="gray")
    fig.update_layout(
        title=f"SHAP explanation for {customer_id} (predicted: {pred_prob:.1%})",
        xaxis_title="SHAP value (impact on churn probability)",
        yaxis_title="",
        height=420,
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(autorange="reversed"),
    )
    return fig


def show():
    st.title("🔍 Explain Customer")
    st.caption("Understand why a specific customer is at risk — powered by SHAP")
    st.divider()

    df = load_customers()
    customer_ids = df["customer_id"].tolist()

    col_sel, col_info = st.columns([1, 2])
    with col_sel:
        selected_id = st.selectbox("Select customer ID", customer_ids, index=0)
        st.caption(f"Showing {len(customer_ids):,} customers")

    row = df[df["customer_id"] == selected_id].iloc[0]

    with col_info:
        total_charge = row["day_charge_aud"] + row["evening_charge_aud"] + row["night_charge_aud"] + row["intl_charge_aud"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("State",           row["state"])
        c2.metric("Tenure (days)",   int(row["account_length_days"]))
        c3.metric("Monthly charge",  f"${total_charge:.2f}")
        c4.metric("CS calls",        int(row["customer_service_calls"]))

        intl  = "✅ Yes" if row["international_plan"] else "❌ No"
        vm    = "✅ Yes" if row["voicemail_plan"]     else "❌ No"
        churn = "🔴 Yes" if row["churned"]            else "🟢 No"
        st.write(f"**Intl plan:** {intl} &nbsp;&nbsp; **Voicemail plan:** {vm} &nbsp;&nbsp; **Actual churn:** {churn}")

    st.divider()

    # ── SHAP computation ──────────────────────────────────────────────────────
    try:
        with st.spinner("Computing SHAP values…"):
            shap_dict = compute_shap_on_the_fly(selected_id, df)

        from src.models.predict import predict_single
        from src.features.engineering import engineer_features_single, get_feature_matrix
        feats = engineer_features_single(df[df["customer_id"] == selected_id].copy())
        result = predict_single(feats)
        pred_prob = result["churn_probability"]
        risk      = result["risk_segment"]

        risk_color = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}[risk]
        st.subheader(f"Churn probability: {pred_prob:.1%}  {risk_color} {risk} risk")

        fig = shap_waterfall(shap_dict, base_value=0.14, pred_prob=pred_prob, customer_id=selected_id)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Top risk factors")
        top5 = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        for feat, val in top5:
            direction = "⬆️ increases" if val > 0 else "⬇️ decreases"
            st.write(f"**{feat}**: {direction} churn risk by `{abs(val):.4f}` SHAP units")

    except FileNotFoundError:
        st.warning("Model not trained yet. Run `python -m src.models.train` then restart the dashboard.")
        st.info("Showing raw customer data instead:")
        st.dataframe(df[df["customer_id"] == selected_id])
