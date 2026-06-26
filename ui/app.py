"""
ui/app.py
──────────
Credit Risk Factory — Streamlit Dashboard

On startup: loads the latest outputs/*_audit_trail.json automatically.
Also supports running the pipeline fresh from the UI.
"""

import sys
import os
import json
import glob
import tempfile
import traceback
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

# Ensure project root is on path regardless of where streamlit is launched from
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

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
    margin-bottom: 0.5rem;
}
.metric-label { color: #cdd6f4; font-size: 0.78rem; margin-bottom: 4px; }
.metric-value { color: #cba6f7; font-size: 1.8rem; font-weight: 700; }
.metric-sub   { color: #a6e3a1; font-size: 0.72rem; margin-top: 2px; }
.pass-badge { background:#a6e3a1; color:#1e1e2e; border-radius:6px;
              padding:3px 12px; font-weight:700; font-size:0.9rem; }
.fail-badge { background:#f38ba8; color:#1e1e2e; border-radius:6px;
              padding:3px 12px; font-weight:700; font-size:0.9rem; }
.section-head { color:#cba6f7; font-size:1rem; font-weight:600;
                border-bottom:1px solid #313244; padding-bottom:4px; margin-top:1rem; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_latest_audit() -> dict | None:
    """Return the most recently modified audit trail JSON, or None."""
    outputs_dir = ROOT / "outputs"
    pattern = str(outputs_dir / "*_audit_trail.json")
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not files:
        return None
    with open(files[0], "r", encoding="utf-8") as f:
        return json.load(f), files[0]


def find_latest_report() -> str | None:
    """Return text of the most recently modified model report, or None."""
    outputs_dir = ROOT / "outputs"
    pattern = str(outputs_dir / "*_model_report.txt")
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not files:
        return None
    with open(files[0], "r", encoding="utf-8", errors="replace") as f:
        return f.read(), files[0]


def metric_card(col, label, value, sub=""):
    if isinstance(value, float):
        val_str = f"{value:.4f}"
    elif value is None:
        val_str = "—"
    else:
        val_str = str(value)
    col.markdown(f"""
<div class="metric-card">
  <div class="metric-label">{label}</div>
  <div class="metric-value">{val_str}</div>
  <div class="metric-sub">{sub}</div>
</div>""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "pipeline_state" not in st.session_state:
    st.session_state.pipeline_state = None
if "running" not in st.session_state:
    st.session_state.running = False
if "error_msg" not in st.session_state:
    st.session_state.error_msg = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏦 Credit Risk Factory")
    st.markdown("---")

    st.markdown("### Run Pipeline")
    uploaded = st.file_uploader("Upload CSV dataset", type=["csv"])
    trials   = st.slider("Optuna trials", 5, 100, 10, step=5)
    run_btn  = st.button("▶  Run Pipeline", type="primary", use_container_width=True)

    st.markdown("---")
    refresh_btn = st.button("🔄  Reload Latest Results", use_container_width=True)
    st.markdown("---")
    st.caption("LLM steps are skipped without an API key. All ML runs locally.")

# ── Run pipeline ──────────────────────────────────────────────────────────────
if run_btn and not st.session_state.running:
    if uploaded is None:
        st.sidebar.error("Please upload a CSV first.")
    else:
        st.session_state.running = True
        st.session_state.error_msg = None
        st.session_state.pipeline_state = None

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        tmp.write(uploaded.read())
        tmp.close()

        with st.spinner("Running pipeline… this takes 2–5 minutes. Please wait."):
            try:
                from orchestrator import CreditRiskOrchestrator
                orch = CreditRiskOrchestrator(
                    output_dir=str(ROOT / "outputs"),
                    optuna_trials=trials,
                    auto_approve=True,
                    verbose=False,
                )
                state = orch.run(
                    dataset_path=tmp.name,
                    dataset_name=uploaded.name,
                )
                st.session_state.pipeline_state = state
            except Exception:
                st.session_state.error_msg = traceback.format_exc()
            finally:
                st.session_state.running = False
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass
        st.rerun()

# ── Load data: prefer live state, fall back to latest audit JSON ──────────────
live_state  = st.session_state.pipeline_state
audit_data  = None
audit_path  = None
report_text = None
report_path = None

result = find_latest_audit()
if result:
    audit_data, audit_path = result

result2 = find_latest_report()
if result2:
    report_text, report_path = result2

# ── Error display ─────────────────────────────────────────────────────────────
if st.session_state.error_msg:
    st.error("Pipeline failed — see details below")
    with st.expander("Error details"):
        st.code(st.session_state.error_msg)

# ── No data yet ───────────────────────────────────────────────────────────────
if live_state is None and audit_data is None:
    st.markdown("## Welcome to the Credit Risk Factory")
    st.info("Upload a CSV in the sidebar and click **Run Pipeline** — or drop a completed `outputs/` folder here and click **Reload Latest Results**.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Models trained", "3")
    col2.metric("Auto checkpoints", "3")
    col3.metric("Output artefacts", "report + audit trail")
    st.stop()

# ── Extract metrics (live state takes priority over saved JSON) ───────────────
if live_state is not None:
    mm        = live_state.model_metrics or {}
    vm        = live_state.validation_metrics or {}
    psi_res   = live_state.psi_results or {}
    champion  = live_state.champion_model_name or "—"
    features  = live_state.selected_features or []
    passed    = live_state.validation_passed
    run_id    = live_state.run_id
    dataset   = live_state.dataset_name
    chk       = {
        "target_confirmed"  : live_state.checkpoint_1_approved,
        "features_approved" : live_state.checkpoint_2_approved,
        "model_signed_off"  : live_state.checkpoint_3_approved,
    }
    errors    = live_state.errors or []
    warnings  = live_state.warnings or []
    audit_log = live_state.audit_log or []
    challengers = vm.get("challenger_table", [])
    deciles     = vm.get("decile_table", [])
    fi          = live_state.feature_importance or {}
    auc   = vm.get("auc")
    gini  = vm.get("gini")
    ks    = vm.get("ks")
    brier = vm.get("brier_score")
    psi_val   = psi_res.get("psi_score")
    psi_label = psi_res.get("assessment", "")
    source = "live run"
else:
    # Pull from saved audit JSON
    d         = audit_data
    mm        = d.get("model_metrics", {})
    champion  = d.get("champion_model_name", "—")
    features  = d.get("selected_features", [])
    passed    = d.get("validation_passed", False)
    run_id    = d.get("run_id", "—")
    dataset   = d.get("dataset_name", "—")
    chk       = d.get("checkpoints", {})
    errors    = d.get("errors", [])
    warnings  = d.get("warnings", [])
    audit_log = d.get("audit_log", [])

    # model_metrics nested per model
    auc = gini = ks = brier = psi_val = None
    psi_label = ""
    challengers = []
    for name, m in mm.items():
        if name == champion:
            auc   = m.get("auc_test")
            gini  = m.get("gini")
            ks    = m.get("ks")
        challengers.append({
            "model"    : name,
            "auc_test" : m.get("auc_test"),
            "ks"       : m.get("ks"),
            "gini"     : m.get("gini"),
            "overfit"  : m.get("overfit"),
            "champion" : "✓" if name == champion else "",
        })
    deciles = []
    fi      = {}
    source  = f"saved — {os.path.basename(audit_path)}"

# ── Header ────────────────────────────────────────────────────────────────────
badge = '<span class="pass-badge">PASS ✓</span>' if passed else '<span class="fail-badge">FAIL ✗</span>'
st.markdown(f"## 🏆 Champion: **{champion}** &nbsp; {badge}", unsafe_allow_html=True)
st.caption(f"Run: {run_id}  |  Dataset: {dataset}  |  Source: {source}")
st.markdown("---")

# ── Key metrics row ───────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
metric_card(c1, "AUC",         auc,   "area under ROC")
metric_card(c2, "Gini",        gini,  "2×AUC − 1")
metric_card(c3, "KS",          ks,    "max separation")
metric_card(c4, "Brier Score", brier, "lower is better")
metric_card(c5, "PSI",         psi_val, psi_label)

st.markdown("<br>", unsafe_allow_html=True)

# ── Checkpoints row ───────────────────────────────────────────────────────────
cc1, cc2, cc3 = st.columns(3)
for col, label, key in [
    (cc1, "Checkpoint 1 — Target Definition", "target_confirmed"),
    (cc2, "Checkpoint 2 — Feature Shortlist",  "features_approved"),
    (cc3, "Checkpoint 3 — Model Sign-Off",     "model_signed_off"),
]:
    ok = chk.get(key, False)
    col.markdown(
        f"{'✅' if ok else '❌'} **{label}**"
    )

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Model Comparison",
    "🔍 Features",
    "📈 Score Deciles",
    "📄 Report",
    "🔬 Audit Log",
    "⚠️ Errors & Warnings",
])

# ── Tab 1: Model comparison ───────────────────────────────────────────────────
with tab1:
    if challengers:
        import plotly.graph_objects as go
        df_ch = (pd.DataFrame(challengers)
                 .rename(columns={"model":"Model","auc_test":"AUC","ks":"KS",
                                  "gini":"Gini","overfit":"Overfit Δ","champion":"Champion"}))
        fmt = {"AUC":"{:.4f}","KS":"{:.4f}","Gini":"{:.4f}"}
        if "Overfit Δ" in df_ch.columns:
            fmt["Overfit Δ"] = "{:.4f}"
        st.dataframe(
            df_ch.style.highlight_max(
                subset=[c for c in ["AUC","KS","Gini"] if c in df_ch.columns],
                color="#2d4a2d"
            ).format(fmt, na_rep="—"),
            use_container_width=True,
        )
        names  = [r.get("model","") for r in challengers]
        aucs   = [r.get("auc_test") or 0 for r in challengers]
        colors = ["#cba6f7" if n == champion else "#6c7086" for n in names]
        fig = go.Figure(go.Bar(
            x=names, y=aucs, marker_color=colors,
            text=[f"{a:.4f}" for a in aucs], textposition="outside",
        ))
        fig.update_layout(
            title="AUC by Model",
            yaxis_range=[0.5, max(aucs) * 1.08] if aucs else [0, 1],
            plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
            font_color="#cdd6f4", height=320,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No model comparison data in this run.")

# ── Tab 2: Features ───────────────────────────────────────────────────────────
with tab2:
    st.markdown("**Selected Features**")
    if features:
        st.write(", ".join(features))
    else:
        st.info("No feature list available.")

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
        st.info("Feature importance chart available after a live run.")

# ── Tab 3: Score deciles ──────────────────────────────────────────────────────
with tab3:
    if deciles:
        import plotly.graph_objects as go
        df_dec = pd.DataFrame(deciles)
        st.dataframe(
            df_dec.style.format({"avg_prob":"{:.3f}","bad_rate":"{:.2%}"}, na_rep="—"),
            use_container_width=True,
        )
        fig = go.Figure()
        fig.add_bar(x=df_dec["decile"].astype(str), y=df_dec["bad_rate"],
                    name="Bad Rate", marker_color="#f38ba8")
        fig.add_scatter(x=df_dec["decile"].astype(str), y=df_dec["avg_prob"],
                        name="Avg Predicted Prob", line_color="#cba6f7",
                        mode="lines+markers")
        fig.update_layout(
            title="Default Rate vs Predicted Probability by Decile",
            plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
            font_color="#cdd6f4", height=360,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Score decile chart available after a live run.")

# ── Tab 4: Report ─────────────────────────────────────────────────────────────
with tab4:
    if report_text:
        st.download_button(
            "⬇  Download Model Report", report_text,
            file_name=os.path.basename(report_path), mime="text/plain",
        )
        st.code(report_text, language=None)
    else:
        st.info("No model report found in outputs/.")

# ── Tab 5: Audit log ──────────────────────────────────────────────────────────
with tab5:
    if audit_log:
        df_audit = pd.DataFrame(audit_log)
        st.dataframe(df_audit, use_container_width=True)
    else:
        st.info("No audit log entries.")

    if audit_data:
        with st.expander("Raw audit JSON"):
            st.json(audit_data)

# ── Tab 6: Errors & warnings ──────────────────────────────────────────────────
with tab6:
    if errors:
        st.markdown("**Errors**")
        for e in errors:
            st.error(f"[{e.get('agent','?')}] {e.get('message','')}")
    else:
        st.success("No errors recorded.")

    if warnings:
        st.markdown("**Warnings**")
        for w in warnings:
            st.warning(f"[{w.get('agent','?')}] {w.get('message','')}")
    else:
        st.success("No warnings recorded.")
