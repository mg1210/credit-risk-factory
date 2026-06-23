"""
ui/app.py
──────────
Streamlit dashboard for the Credit Risk Factory.
Run with: streamlit run ui/app.py
"""

import sys
import os
import io
import json
import threading
import queue
import tempfile

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

# ── Styles ────────────────────────────────────────────────────────────────────
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
.phase-ok     { color: #a6e3a1; }
.phase-err    { color: #f38ba8; }
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
    st.caption("LLM steps are skipped if no API key is set. All ML steps run locally.")

# ── State init ────────────────────────────────────────────────────────────────
if "pipeline_state" not in st.session_state:
    st.session_state.pipeline_state = None
if "log_lines" not in st.session_state:
    st.session_state.log_lines = []
if "running" not in st.session_state:
    st.session_state.running = False

# ── Run pipeline ──────────────────────────────────────────────────────────────
def run_pipeline(csv_path: str, trials: int, log_q: queue.Queue):
    try:
        from orchestrator import CreditRiskOrchestrator
        orch = CreditRiskOrchestrator(
            output_dir="outputs",
            optuna_trials=trials,
            auto_approve=True,
            verbose=True,
        )
        # Redirect stdout into the queue
        class QueueWriter(io.TextIOBase):
            def write(self, s):
                if s.strip():
                    log_q.put(("log", s.rstrip()))
                return len(s)
            def flush(self): pass

        old_stdout = sys.stdout
        sys.stdout  = QueueWriter()
        try:
            state = orch.run(dataset_path=csv_path,
                             dataset_name=os.path.basename(csv_path))
        finally:
            sys.stdout = old_stdout

        log_q.put(("done", state))
    except Exception as e:
        log_q.put(("error", str(e)))


if run_btn:
    if uploaded is None:
        st.sidebar.error("Please upload a CSV first.")
    else:
        # Save upload to temp file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        tmp.write(uploaded.read())
        tmp.close()

        st.session_state.log_lines = []
        st.session_state.pipeline_state = None
        st.session_state.running = True
        st.session_state.csv_path = tmp.name

# ── If pipeline should run, do it synchronously (Streamlit threading workaround)
if st.session_state.running and st.session_state.pipeline_state is None:
    csv_path = st.session_state.csv_path
    log_box  = st.expander("📋 Pipeline Log", expanded=True)
    log_area = log_box.empty()
    progress = st.progress(0, text="Starting pipeline…")

    log_lines = []
    phase_map = {
        "Phase 1": 14, "Phase 2": 28, "Phase 3": 42,
        "Phase 4": 57, "Phase 5": 71, "Phase 6": 85, "Phase 7": 100,
    }

    try:
        from orchestrator import CreditRiskOrchestrator
        import io as _io

        class CapturingWriter(_io.TextIOBase):
            def __init__(self):
                self.lines = []
            def write(self, s):
                if s.strip():
                    self.lines.append(s.rstrip())
                return len(s)
            def flush(self): pass

        writer = CapturingWriter()
        old_stdout = sys.stdout
        sys.stdout = writer

        orch = CreditRiskOrchestrator(
            output_dir="outputs",
            optuna_trials=trials,
            auto_approve=True,
            verbose=True,
        )
        state = orch.run(
            dataset_path=csv_path,
            dataset_name=os.path.basename(csv_path),
        )
        sys.stdout = old_stdout
        log_lines = writer.lines

        pct = 0
        for line in log_lines:
            for phase, p in phase_map.items():
                if phase in line:
                    pct = p
            progress.progress(min(pct, 100), text=line[:80])

        progress.progress(100, text="Pipeline complete ✓")
        log_area.code("\n".join(log_lines), language=None)
        st.session_state.log_lines  = log_lines
        st.session_state.pipeline_state = state
        st.session_state.running = False
        st.rerun()

    except Exception as e:
        sys.stdout = old_stdout
        st.session_state.running = False
        st.error(f"Pipeline error: {e}")

# ── Results ───────────────────────────────────────────────────────────────────
state = st.session_state.pipeline_state

if state is None and not st.session_state.running:
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
    vm   = state.validation_metrics or {}
    psi  = state.psi_results or {}
    auc  = vm.get("auc")
    gini = vm.get("gini")
    ks   = vm.get("ks")
    brier = vm.get("brier_score")

    passed = state.validation_passed
    badge = '<span class="pass-badge">PASS ✓</span>' if passed else '<span class="fail-badge">FAIL ✗</span>'

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(f"## 🏆 Champion: {state.champion_model_name or '—'}  &nbsp; {badge}",
                unsafe_allow_html=True)
    st.caption(f"Run ID: {state.run_id}  |  Dataset: {state.dataset_name}")
    st.markdown("---")

    # ── Key metrics ───────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    def metric_card(col, label, value, sub=""):
        val_str = f"{value:.4f}" if isinstance(value, float) else (str(value) if value is not None else "—")
        col.markdown(f"""
<div class="metric-card">
  <div class="metric-label">{label}</div>
  <div class="metric-value">{val_str}</div>
  <div class="metric-sub">{sub}</div>
</div>""", unsafe_allow_html=True)

    metric_card(c1, "AUC",        auc,   "area under ROC")
    metric_card(c2, "Gini",       gini,  "2×AUC − 1")
    metric_card(c3, "KS",         ks,    "max separation")
    metric_card(c4, "Brier Score",brier, "lower is better")
    psi_val = psi.get("psi_score")
    metric_card(c5, "PSI",        psi_val, psi.get("assessment",""))

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Model Comparison",
        "🔍 Feature Importance",
        "📈 Score Deciles",
        "📋 Pipeline Log",
        "📄 Report",
        "🔬 Data Profile",
    ])

    # ── Tab 1: Model comparison ───────────────────────────────────────────────
    with tab1:
        challengers = vm.get("challenger_table", [])
        if challengers:
            df_ch = pd.DataFrame(challengers)
            df_ch = df_ch.rename(columns={
                "model":"Model","auc_test":"AUC","ks":"KS",
                "gini":"Gini","overfit":"Overfit Δ","champion":"Champion"
            })
            st.dataframe(
                df_ch.style
                    .highlight_max(subset=["AUC","KS","Gini"], color="#2d4a2d")
                    .highlight_min(subset=["Overfit Δ"], color="#2d4a2d")
                    .format({"AUC":"{:.4f}","KS":"{:.4f}","Gini":"{:.4f}","Overfit Δ":"{:.4f}"}),
                use_container_width=True,
            )
        else:
            st.info("No challenger data available.")

        # ROC-style AUC bar
        if challengers:
            import plotly.graph_objects as go
            names = [r["Model"] for r in challengers]
            aucs  = [r.get("auc_test", 0) for r in challengers]
            colors = ["#cba6f7" if n == state.champion_model_name else "#6c7086" for n in names]
            fig = go.Figure(go.Bar(x=names, y=aucs, marker_color=colors,
                                   text=[f"{a:.4f}" for a in aucs], textposition="outside"))
            fig.update_layout(
                title="AUC by Model", yaxis_range=[0.5, max(aucs)*1.1],
                plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
                font_color="#cdd6f4", height=320,
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2: Feature importance ─────────────────────────────────────────────
    with tab2:
        fi = state.feature_importance
        if fi:
            import plotly.express as px
            df_fi = pd.DataFrame(list(fi.items()), columns=["Feature","Importance"])
            df_fi = df_fi.sort_values("Importance", ascending=True).tail(15)
            fig = px.bar(df_fi, x="Importance", y="Feature", orientation="h",
                         color="Importance", color_continuous_scale="purples",
                         title="SHAP Feature Importance (mean |SHAP|)")
            fig.update_layout(plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
                               font_color="#cdd6f4", height=420,
                               coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

            # Also show IV table
            iv = state.iv_table
            if iv is not None:
                try:
                    df_iv = iv if isinstance(iv, pd.DataFrame) else pd.DataFrame(iv)
                    if not df_iv.empty:
                        st.markdown("**Information Value (IV) Table**")
                        st.dataframe(df_iv.head(20), use_container_width=True)
                except Exception:
                    pass
        else:
            st.info("No feature importance data — SHAP may have been skipped.")

        st.markdown("**Selected Features**")
        if state.selected_features:
            st.write(", ".join(state.selected_features))

    # ── Tab 3: Score deciles ──────────────────────────────────────────────────
    with tab3:
        deciles = vm.get("decile_table", [])
        if deciles:
            df_dec = pd.DataFrame(deciles)
            st.dataframe(
                df_dec.style.format({"avg_prob":"{:.3f}","bad_rate":"{:.2%}"}),
                use_container_width=True,
            )
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_bar(x=df_dec["decile"].astype(str), y=df_dec["bad_rate"],
                        name="Bad Rate", marker_color="#f38ba8")
            fig.add_scatter(x=df_dec["decile"].astype(str), y=df_dec["avg_prob"],
                            name="Avg Predicted Prob", line_color="#cba6f7", mode="lines+markers")
            fig.update_layout(
                title="Default Rate vs Predicted Score by Decile",
                plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
                font_color="#cdd6f4", height=360,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No decile data available.")

    # ── Tab 4: Pipeline log ───────────────────────────────────────────────────
    with tab4:
        log = st.session_state.log_lines
        if log:
            st.code("\n".join(log), language=None)
        else:
            st.info("No log captured.")

        if state.errors:
            st.markdown("**Errors**")
            for e in state.errors:
                st.error(f"[{e['agent']}] {e['message']}")
        if state.warnings:
            st.markdown("**Warnings**")
            for w in state.warnings:
                st.warning(f"[{w['agent']}] {w['message']}")

    # ── Tab 5: Report ─────────────────────────────────────────────────────────
    with tab5:
        rp = state.model_report_path
        if rp and os.path.exists(rp):
            with open(rp, "r", encoding="utf-8", errors="replace") as f:
                report_text = f.read()
            st.download_button("⬇  Download Report", report_text,
                               file_name=os.path.basename(rp), mime="text/plain")
            st.code(report_text, language=None)
        else:
            st.info("Report not yet generated.")

        audit_path = rp.replace("model_report", "audit_trail").replace(".txt", ".json") if rp else ""
        if audit_path and os.path.exists(audit_path):
            with open(audit_path, "r", encoding="utf-8") as f:
                audit = json.load(f)
            with st.expander("📁 Audit Trail (JSON)"):
                st.json(audit)

    # ── Tab 6: Data profile ───────────────────────────────────────────────────
    with tab6:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Dataset Summary**")
            raw = state.raw_df
            if raw is not None:
                st.write(f"Rows: {len(raw):,} | Columns: {raw.shape[1]}")
                st.write(f"Target default rate: {state.target_definition}")
            st.markdown("**Leakage Columns Flagged**")
            if state.leakage_columns:
                st.write(state.leakage_columns)
            st.markdown("**High Missing (>40%)**")
            hm = state.high_missing_cols
            if hm:
                st.write(hm)

        with col_b:
            st.markdown("**DQR Flags**")
            flags = state.dqr_flags
            if flags:
                for f in flags:
                    st.warning(f)
            else:
                st.success("No DQR flags raised.")

            ms = state.missing_summary
            if ms:
                st.markdown("**Missing Value Summary (top 10)**")
                df_ms = (pd.DataFrame.from_dict(ms, orient="index", columns=["missing_pct"])
                         .sort_values("missing_pct", ascending=False).head(10))
                st.dataframe(df_ms.style.format("{:.1%}"), use_container_width=True)
