"""
ui/app.py
──────────
Streamlit dashboard for the Credit Risk Factory.
Run with: streamlit run ui/app.py
"""

import sys
import os
import json
import tempfile
import traceback

import streamlit as st
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="Credit Risk Factory",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.metric-card {
    background: #1e1e2e;
    border: 1px solid #313244;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-label { color: #cdd6f4; font-size: 0.78rem; margin-bottom: 4px; }
.metric-value { color: #cba6f7; font-size: 1.8rem; font-weight: 700; }
.metric-sub   { color: #a6e3a1; font-size: 0.72rem; margin-top: 2px; }
.pass-badge   { background:#a6e3a1; color:#1e1e2e; border-radius:6px; padding:2px 10px; font-weight:700; }
.fail-badge   { background:#f38ba8; color:#1e1e2e; border-radius:6px; padding:2px 10px; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏦 Credit Risk Factory")
    st.markdown("---")
    uploaded = st.file_uploader("Upload CSV dataset", type=["csv"])
    trials   = st.slider("Optuna trials", 5, 100, 10, step=5)
    st.markdown("---")
    run_btn  = st.button("▶  Run Pipeline", type="primary", use_container_width=True)
    st.markdown("---")
    st.caption("LLM steps are skipped if no API key is set. All ML runs locally.")

# ── Session state ─────────────────────────────────────────────────────────────
if "pipeline_state" not in st.session_state:
    st.session_state.pipeline_state = None
if "error_msg" not in st.session_state:
    st.session_state.error_msg = None

# ── Run pipeline ──────────────────────────────────────────────────────────────
if run_btn:
    if uploaded is None:
        st.sidebar.error("Please upload a CSV first.")
    else:
        # Write upload to temp file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        tmp.write(uploaded.read())
        tmp.close()

        st.session_state.pipeline_state = None
        st.session_state.error_msg = None

        with st.spinner("Running pipeline… this takes 2–5 minutes. Please wait."):
            try:
                from orchestrator import CreditRiskOrchestrator
                orch = CreditRiskOrchestrator(
                    output_dir="outputs",
                    optuna_trials=trials,
                    auto_approve=True,
                    verbose=False,   # suppress colorama stdout issues
                )
                state = orch.run(
                    dataset_path=tmp.name,
                    dataset_name=uploaded.name,
                )
                st.session_state.pipeline_state = state
            except Exception as e:
                st.session_state.error_msg = traceback.format_exc()

        try:
            os.unlink(tmp.name)
        except Exception:
            pass

        st.rerun()

# ── Show error if pipeline failed ─────────────────────────────────────────────
if st.session_state.error_msg:
    st.error("Pipeline failed")
    with st.expander("Error details"):
        st.code(st.session_state.error_msg)

# ── Results ───────────────────────────────────────────────────────────────────
state = st.session_state.pipeline_state

if state is None and not st.session_state.error_msg:
    st.markdown("## Welcome to the Credit Risk Factory")
    st.markdown("""
Upload your Lending Club-style CSV in the sidebar and click **Run Pipeline**.

The pipeline will automatically:
- Define & validate the target variable
- Run a full Data Quality Review
- Engineer credit risk features
- Select the best predictors via IV & RF importance
- Train Logistic Regression, Random Forest, and XGBoost
- Compute SHAP feature importance
- Validate with AUC, KS, Gini, PSI, and calibration metrics
- Generate a model development report
""")
    col1, col2, col3 = st.columns(3)
    col1.metric("Models trained", "3")
    col2.metric("Auto checkpoints", "3")
    col3.metric("Output artefacts", "report + audit trail")

elif state is not None:
    vm    = state.validation_metrics or {}
    psi   = state.psi_results or {}
    auc   = vm.get("auc")
    gini  = vm.get("gini")
    ks    = vm.get("ks")
    brier = vm.get("brier_score")

    passed = state.validation_passed
    badge  = '<span class="pass-badge">PASS ✓</span>' if passed else '<span class="fail-badge">FAIL ✗</span>'

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(f"## 🏆 Champion: {state.champion_model_name or '—'}  &nbsp; {badge}",
                unsafe_allow_html=True)
    st.caption(f"Run ID: {state.run_id}  |  Dataset: {state.dataset_name}")
    st.markdown("---")

    # ── Key metrics ───────────────────────────────────────────────────────────
    def metric_card(col, label, value, sub=""):
        val_str = f"{value:.4f}" if isinstance(value, float) else (str(value) if value is not None else "—")
        col.markdown(f"""
<div class="metric-card">
  <div class="metric-label">{label}</div>
  <div class="metric-value">{val_str}</div>
  <div class="metric-sub">{sub}</div>
</div>""", unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    metric_card(c1, "AUC",         auc,   "area under ROC")
    metric_card(c2, "Gini",        gini,  "2×AUC − 1")
    metric_card(c3, "KS",          ks,    "max separation")
    metric_card(c4, "Brier Score", brier, "lower is better")
    psi_val = psi.get("psi_score")
    metric_card(c5, "PSI",         psi_val, psi.get("assessment", ""))

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Model Comparison",
        "🔍 Feature Importance",
        "📈 Score Deciles",
        "📄 Report",
        "🔬 Data Profile",
    ])

    # ── Tab 1: Model comparison ───────────────────────────────────────────────
    with tab1:
        challengers = vm.get("challenger_table", [])
        if challengers:
            import plotly.graph_objects as go
            df_ch = pd.DataFrame(challengers).rename(columns={
                "model": "Model", "auc_test": "AUC", "ks": "KS",
                "gini": "Gini", "overfit": "Overfit Δ", "champion": "Champion"
            })
            st.dataframe(
                df_ch.style
                    .highlight_max(subset=["AUC", "KS", "Gini"], color="#2d4a2d")
                    .format({"AUC": "{:.4f}", "KS": "{:.4f}", "Gini": "{:.4f}", "Overfit Δ": "{:.4f}"}),
                use_container_width=True,
            )
            names  = [r["model"] for r in challengers]
            aucs   = [r.get("auc_test", 0) for r in challengers]
            colors = ["#cba6f7" if n == state.champion_model_name else "#6c7086" for n in names]
            fig = go.Figure(go.Bar(
                x=names, y=aucs, marker_color=colors,
                text=[f"{a:.4f}" for a in aucs], textposition="outside"
            ))
            fig.update_layout(
                title="AUC by Model", yaxis_range=[0.5, max(aucs) * 1.1],
                plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
                font_color="#cdd6f4", height=320,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No model comparison data available.")

    # ── Tab 2: Feature importance ─────────────────────────────────────────────
    with tab2:
        fi = state.feature_importance
        if fi:
            import plotly.express as px
            df_fi = (pd.DataFrame(list(fi.items()), columns=["Feature", "Importance"])
                     .sort_values("Importance", ascending=True).tail(15))
            fig = px.bar(df_fi, x="Importance", y="Feature", orientation="h",
                         color="Importance", color_continuous_scale="purples",
                         title="SHAP / RF Feature Importance")
            fig.update_layout(
                plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
                font_color="#cdd6f4", height=420, coloraxis_showscale=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No feature importance data available.")

        st.markdown("**Selected Features**")
        if state.selected_features:
            st.write(", ".join(state.selected_features))

        iv = state.iv_table
        if iv is not None:
            try:
                df_iv = iv if isinstance(iv, pd.DataFrame) else pd.DataFrame(iv)
                if not df_iv.empty:
                    st.markdown("**Information Value (IV) Table — top 20**")
                    st.dataframe(df_iv.head(20), use_container_width=True)
            except Exception:
                pass

    # ── Tab 3: Score deciles ──────────────────────────────────────────────────
    with tab3:
        deciles = vm.get("decile_table", [])
        if deciles:
            import plotly.graph_objects as go
            df_dec = pd.DataFrame(deciles)
            st.dataframe(
                df_dec.style.format({"avg_prob": "{:.3f}", "bad_rate": "{:.2%}"}),
                use_container_width=True,
            )
            fig = go.Figure()
            fig.add_bar(x=df_dec["decile"].astype(str), y=df_dec["bad_rate"],
                        name="Bad Rate", marker_color="#f38ba8")
            fig.add_scatter(x=df_dec["decile"].astype(str), y=df_dec["avg_prob"],
                            name="Avg Predicted Prob", line_color="#cba6f7",
                            mode="lines+markers")
            fig.update_layout(
                title="Default Rate vs Predicted Score by Decile",
                plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
                font_color="#cdd6f4", height=360,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No decile data available.")

    # ── Tab 4: Report ─────────────────────────────────────────────────────────
    with tab4:
        rp = state.model_report_path
        if rp and os.path.exists(rp):
            with open(rp, "r", encoding="utf-8", errors="replace") as f:
                report_text = f.read()
            st.download_button("⬇  Download Report", report_text,
                               file_name=os.path.basename(rp), mime="text/plain")
            st.code(report_text, language=None)
        else:
            st.info("Report not generated yet.")

        audit_path = ""
        if rp:
            audit_path = rp.replace("model_report", "audit_trail").replace(".txt", ".json")
        if audit_path and os.path.exists(audit_path):
            with open(audit_path, "r", encoding="utf-8") as f:
                audit = json.load(f)
            with st.expander("📁 Audit Trail (JSON)"):
                st.json(audit)

        if state.errors:
            st.markdown("**Pipeline Errors**")
            for e in state.errors:
                st.error(f"[{e['agent']}] {e['message']}")

    # ── Tab 5: Data profile ───────────────────────────────────────────────────
    with tab5:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Dataset Summary**")
            raw = state.raw_df
            if raw is not None:
                st.write(f"Rows: {len(raw):,} | Columns: {raw.shape[1]}")
            st.write(f"**Target:** {state.target_definition}")

            st.markdown("**Leakage Columns Flagged**")
            if state.leakage_columns:
                st.write(state.leakage_columns)

            st.markdown("**High Missing (>40%)**")
            if state.high_missing_cols:
                st.write(state.high_missing_cols)

        with col_b:
            st.markdown("**DQR Flags**")
            if state.dqr_flags:
                for f in state.dqr_flags:
                    st.warning(f)
            else:
                st.success("No DQR flags raised.")

            ms = state.missing_summary
            if ms:
                st.markdown("**Missing Value Summary (top 10)**")
                df_ms = (pd.DataFrame.from_dict(ms, orient="index", columns=["missing_pct"])
                         .sort_values("missing_pct", ascending=False).head(10))
                st.dataframe(df_ms.style.format("{:.1%}"), use_container_width=True)
