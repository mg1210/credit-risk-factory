"""
Credit Risk Factory — Multi-Page Streamlit UI
================================================
7 phase pages + 3 human-in-the-loop checkpoint pages.
All data read from outputs/*_audit_trail.json and model_report.txt.
"""

import os
import sys
import json
import glob
import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
OUTPUTS = os.path.join(ROOT, "outputs")
CHECKPOINT_FILE = os.path.join(OUTPUTS, "checkpoint_approvals.json")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Factory",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colour constants ──────────────────────────────────────────────────────────
PASS_COLOR  = "#00c853"
FAIL_COLOR  = "#d32f2f"
WARN_COLOR  = "#f9a825"
INFO_COLOR  = "#1565c0"

# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_audit(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_latest_json():
    files = glob.glob(os.path.join(OUTPUTS, "*_audit_trail.json"))
    return max(files, key=os.path.getmtime) if files else None


def find_report_for(run_id: str):
    path = os.path.join(OUTPUTS, f"{run_id}_model_report.txt")
    return path if os.path.exists(path) else None


def load_checkpoint_state() -> dict:
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_checkpoint_state(state: dict):
    os.makedirs(OUTPUTS, exist_ok=True)
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────

if "page" not in st.session_state:
    st.session_state["page"] = "home"
if "data" not in st.session_state:
    st.session_state["data"] = None
if "audit_path" not in st.session_state:
    st.session_state["audit_path"] = None
if "running" not in st.session_state:
    st.session_state["running"] = False
if "cp_state" not in st.session_state:
    st.session_state["cp_state"] = load_checkpoint_state()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def nav_button(label: str, key: str, icon: str = ""):
    active = st.session_state["page"] == key
    prefix = "▶" if active else "  "
    if st.sidebar.button(f"{prefix} {icon} {label}", key=f"nav_{key}", use_container_width=True):
        st.session_state["page"] = key
        st.rerun()


def build_sidebar():
    st.sidebar.markdown("## 🏦 Credit Risk Factory")
    st.sidebar.markdown("---")

    data = st.session_state.get("data")

    # ── Run control ───────────────────────────────────────────────
    st.sidebar.markdown("### Run Pipeline")
    uploaded = st.sidebar.file_uploader("Upload CSV dataset", type=["csv"],
                                        label_visibility="collapsed")

    col1, col2 = st.sidebar.columns(2)
    run_clicked  = col1.button("▶ Run",  use_container_width=True, type="primary",
                               disabled=st.session_state["running"])
    load_clicked = col2.button("📂 Load", use_container_width=True,
                               help="Load latest saved results")

    if load_clicked:
        p = find_latest_json()
        if p:
            load_audit.clear()
            st.session_state["data"]       = load_audit(p)
            st.session_state["audit_path"] = p
            st.session_state["cp_state"]   = load_checkpoint_state()
            st.session_state["page"]       = "p1"
            st.rerun()
        else:
            st.sidebar.error("No results found in outputs/")

    if run_clicked and not st.session_state["running"]:
        if uploaded is None:
            st.sidebar.error("Upload a CSV first.")
        else:
            _run_pipeline(uploaded)

    st.sidebar.markdown("---")

    # ── Phase navigation ──────────────────────────────────────────
    st.sidebar.markdown("### 📊 Pipeline Phases")
    nav_button("Data Understanding",   "p1", "1️⃣")
    nav_button("Data Quality Review",  "p2", "2️⃣")
    nav_button("Feature Engineering",  "p3", "3️⃣")
    nav_button("Variable Selection",   "p4", "4️⃣")
    nav_button("Model Development",    "p5", "5️⃣")
    nav_button("Explainability",       "p6", "6️⃣")
    nav_button("Validation",           "p7", "7️⃣")

    st.sidebar.markdown("---")

    # ── Checkpoint navigation ─────────────────────────────────────
    cp     = st.session_state.get("cp_state", {})
    run_id = (data or {}).get("run_id", "")

    def cp_icon(key):
        entry    = cp.get(run_id, {}).get(key, {})
        decision = entry.get("decision", "")
        if decision == "approve": return "✅"
        if decision == "reject":  return "❌"
        if decision == "changes": return "⚠️"
        return "🔲"

    st.sidebar.markdown("### 👤 Human Checkpoints")
    nav_button(f"Checkpoint 1 — Target {cp_icon('cp1')}",  "cp1", "")
    nav_button(f"Checkpoint 2 — Features {cp_icon('cp2')}", "cp2", "")
    nav_button(f"Checkpoint 3 — Sign-Off {cp_icon('cp3')}", "cp3", "")

    st.sidebar.markdown("---")

    # ── Run status summary ────────────────────────────────────────
    if data:
        champ   = data.get("champion_model_name", "—")
        vm      = data.get("validation_metrics", {})
        auc     = vm.get("auc", "—")
        passed  = data.get("validation_passed", False)
        status  = "✅ PASS" if passed else "⚠️ CONDITIONAL"
        st.sidebar.markdown(f"""
**Run:** `{data.get('run_id','')[:14]}`
**Champion:** {champ}
**AUC:** {auc}
**Status:** {status}
""")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_pipeline(uploaded):
    import tempfile
    st.session_state["running"] = True
    status = st.sidebar.empty()
    status.info("Pipeline running…")
    try:
        suffix = os.path.splitext(uploaded.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name

        devnull = open(os.devnull, "w")
        sys.stdout, sys.stderr = devnull, devnull
        try:
            from orchestrator import CreditRiskOrchestrator
            orc = CreditRiskOrchestrator(
                output_dir=OUTPUTS,
                optuna_trials=5,
                auto_approve=True,
                verbose=False,
            )
            orc.run(tmp_path, dataset_name=uploaded.name)
        finally:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            devnull.close()

        p = find_latest_json()
        if p:
            load_audit.clear()
            st.session_state["data"]       = load_audit(p)
            st.session_state["audit_path"] = p
            st.session_state["page"]       = "p1"
            status.success("Pipeline complete!")
        else:
            status.error("Pipeline finished but no output found.")
    except Exception as e:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        status.error(f"Error: {e}")
    finally:
        st.session_state["running"] = False
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Shared widget helpers
# ─────────────────────────────────────────────────────────────────────────────

def section(title: str):
    st.markdown(f"### {title}")
    st.markdown("---")


def na_box(msg="No data available for this section yet. Re-run the pipeline to populate."):
    st.info(f"ℹ️ {msg}")


def llm_box(text: str, label="LLM Narrative"):
    if text:
        with st.expander(f"🤖 {label}", expanded=True):
            st.markdown(text)
    else:
        na_box(f"{label} not available — requires LLM API key and pipeline re-run.")


# ─────────────────────────────────────────────────────────────────────────────
# Page 1 — Data Understanding
# ─────────────────────────────────────────────────────────────────────────────

def page_data_understanding(data: dict):
    st.title("1️⃣ Data Understanding")
    run_id = data.get("run_id", "")
    st.caption(f"Run: `{run_id}`  |  Dataset: `{data.get('dataset_name','')}`")

    schema   = data.get("schema_profile", {})
    leakage  = data.get("leakage_columns", [])
    num_cols = data.get("numeric_columns", [])
    cat_cols = data.get("categorical_columns", [])
    id_cols  = data.get("id_columns", [])
    dt_cols  = data.get("date_columns", [])

    # Row count from schema or audit log
    import re
    n_rows = None
    for v in schema.values():
        if isinstance(v, dict) and "n_rows" in v:
            n_rows = v["n_rows"]
            break
    if n_rows is None:
        for entry in data.get("audit_log", []):
            m = re.search(r"(\d[\d,]+)\s*rows", entry.get("detail",""))
            if m:
                n_rows = int(m.group(1).replace(",",""))
                break

    n_cols = len(schema) or len(num_cols) + len(cat_cols) + len(id_cols) + len(dt_cols) + len(leakage)

    # Default rate from audit log
    default_rate = None
    for entry in data.get("audit_log", []):
        m = re.search(r"default.rate[=:]\s*([\d.]+)", entry.get("detail",""), re.IGNORECASE)
        if m:
            default_rate = float(m.group(1))
            break

    # ── Dataset statistics ────────────────────────────────────────
    section("Dataset Statistics")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Rows",         f"{n_rows:,}" if n_rows else "—")
    c2.metric("Columns",      str(n_cols) if n_cols else "—")
    c3.metric("Target",       data.get("target_column","—"))
    c4.metric("Default Rate", f"{default_rate:.1%}" if default_rate else "—")
    c5.metric("Leakage Cols", str(len(leakage)))

    # ── Target definition ─────────────────────────────────────────
    section("Target Variable Definition")
    tdef   = data.get("target_definition","")
    target = data.get("target_column","—")
    if tdef:
        st.success(f"**`{target}`** — {tdef}")
    else:
        na_box("Target definition not saved. Re-run pipeline.")

    # ── Schema table ──────────────────────────────────────────────
    section("Column Schema")
    if schema:
        rows = []
        for col, meta in schema.items():
            if isinstance(meta, dict):
                rows.append({
                    "Column":    col,
                    "Dtype":     meta.get("dtype","—"),
                    "Missing %": meta.get("pct_missing","—"),
                    "Unique":    meta.get("n_unique","—"),
                    "Role":      meta.get("role","—"),
                    "Sample":    str(meta.get("sample",""))[:60],
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        # Fall back to column lists
        rows = (
            [{"Column": c, "Dtype": "numeric",     "Role": "feature"} for c in num_cols] +
            [{"Column": c, "Dtype": "categorical",  "Role": "feature"} for c in cat_cols] +
            [{"Column": c, "Dtype": "identifier",   "Role": "id"}     for c in id_cols] +
            [{"Column": c, "Dtype": "date",          "Role": "date"}   for c in dt_cols] +
            [{"Column": c, "Dtype": "—",             "Role": "leakage"} for c in leakage]
        )
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            na_box("Schema profile not available. Re-run pipeline for full schema.")

    # ── Column type breakdown ─────────────────────────────────────
    section("Column Type Breakdown")
    col_a, col_b = st.columns(2)
    with col_a:
        labels = ["Numeric","Categorical","Identifier","Date","Leakage"]
        values = [len(num_cols),len(cat_cols),len(id_cols),len(dt_cols),len(leakage)]
        if any(v > 0 for v in values):
            fig = px.pie(values=values, names=labels, hole=0.45,
                         title="Column Types",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(margin=dict(t=40,b=10), height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            na_box("Column breakdown not available.")

    with col_b:
        for label, cols_list in [("Numeric",num_cols),("Categorical",cat_cols),
                                  ("Identifier",id_cols),("Date",dt_cols)]:
            if cols_list:
                with st.expander(f"{label} columns ({len(cols_list)})", expanded=False):
                    st.write(", ".join(cols_list))

    # ── Leakage table ─────────────────────────────────────────────
    if leakage:
        section("Leakage Columns Removed")
        st.warning(f"{len(leakage)} columns removed to prevent data leakage:")
        st.dataframe(pd.DataFrame({"Column": leakage}), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Page 2 — Data Quality Review
# ─────────────────────────────────────────────────────────────────────────────

def page_dqr(data: dict):
    st.title("2️⃣ Data Quality Review")
    st.caption(f"Run: `{data.get('run_id','')}`")

    import re
    missing   = data.get("missing_summary", {})
    outliers  = data.get("outlier_summary", {})
    dqr_flags = data.get("dqr_flags", [])
    dqr_report= data.get("dqr_report", {})
    high_miss = data.get("high_missing_cols", [])

    # Duplicate count from audit log
    dup_count = None
    for entry in data.get("audit_log", []):
        m = re.search(r"(\d+)\s*duplicate", entry.get("detail",""), re.IGNORECASE)
        if m:
            dup_count = int(m.group(1))
            break

    # ── Summary metrics ───────────────────────────────────────────
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("High-Missing Cols (>40%)", str(len(high_miss)))
    c2.metric("DQR Flags",                str(len(dqr_flags)))
    c3.metric("Outlier Cols Flagged",     str(len(outliers)))
    c4.metric("Duplicate Rows",           str(dup_count) if dup_count is not None else "—")

    # ── Missing values ────────────────────────────────────────────
    section("Missing Value Analysis")
    if missing:
        rows = []
        for col, val in missing.items():
            if isinstance(val, dict):
                pct = float(val.get("pct_missing", val.get("pct", 0)))
                n   = val.get("n_missing","—")
            else:
                pct = float(val)
                n   = "—"
            pct_disp = pct * 100 if pct <= 1.0 else pct
            rows.append({"Column": col, "Missing %": round(pct_disp, 2), "N Missing": n})

        df_miss = pd.DataFrame(rows).sort_values("Missing %", ascending=False)
        top30   = df_miss.head(30)

        col_a, col_b = st.columns([2, 3])
        with col_a:
            st.dataframe(top30, use_container_width=True, hide_index=True)
        with col_b:
            fig = px.bar(top30, x="Missing %", y="Column", orientation="h",
                         title="Top Columns by Missing Rate (%)",
                         color="Missing %", color_continuous_scale="Reds")
            fig.update_layout(height=450, margin=dict(t=40,b=10))
            st.plotly_chart(fig, use_container_width=True)

        if high_miss:
            st.warning(f"**{len(high_miss)} columns** >40% missing — dropped: "
                       f"`{'`, `'.join(high_miss[:10])}`{'...' if len(high_miss)>10 else ''}")
    else:
        na_box("Missing value summary not available. Re-run pipeline.")

    # ── Outlier detection ─────────────────────────────────────────
    section("Outlier Detection")
    if outliers:
        rows = []
        for col, meta in outliers.items():
            if isinstance(meta, dict):
                rows.append({
                    "Column":        col,
                    "Method":        meta.get("method","IQR"),
                    "Outlier Count": meta.get("n_outliers","—"),
                    "Lower Bound":   meta.get("lower","—"),
                    "Upper Bound":   meta.get("upper","—"),
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.json(outliers)
    else:
        na_box("Outlier summary not available. Re-run pipeline.")

    # ── DQR flags ─────────────────────────────────────────────────
    section("DQR Flags")
    if dqr_flags:
        for flag in dqr_flags:
            st.warning(f"⚠️ {flag}")
    else:
        st.success("✅ No DQR flags raised.")

    # ── Distribution profiles ─────────────────────────────────────
    profiles = dqr_report.get("distribution_profiles", {})
    if profiles:
        section("Distribution Profiles (Top 10 Variables)")
        cols_show = list(profiles.keys())[:10]
        tabs = st.tabs(cols_show)
        for tab, col_name in zip(tabs, cols_show):
            with tab:
                st.json(profiles[col_name])
    else:
        section("Distribution Profiles")
        na_box("Distribution profiles not available. Re-run pipeline.")

    # ── LLM narrative ─────────────────────────────────────────────
    llm_box(dqr_report.get("llm_narrative",""), "DQR LLM Narrative")


# ─────────────────────────────────────────────────────────────────────────────
# Page 3 — Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

def page_feature_engineering(data: dict):
    st.title("3️⃣ Feature Engineering")
    st.caption(f"Run: `{data.get('run_id','')}`")

    import re
    feature_log = data.get("feature_log", [])
    dqr_report  = data.get("dqr_report", {})

    # Shape from audit log
    before_shape = after_shape = None
    for entry in data.get("audit_log", []):
        detail = entry.get("detail","")
        if "before" in detail.lower():
            m = re.search(r"\((\d+),\s*(\d+)\)", detail)
            if m: before_shape = (int(m.group(1)), int(m.group(2)))
        if "after" in detail.lower() or "output" in detail.lower():
            m = re.search(r"\((\d+),\s*(\d+)\)", detail)
            if m: after_shape  = (int(m.group(1)), int(m.group(2)))
        if re.search(r"\d+\s*rows.*\d+\s*cols", detail):
            m = re.search(r"(\d+)\s*rows.*?(\d+)\s*cols", detail)
            if m: after_shape  = (int(m.group(1)), int(m.group(2)))

    # ── Shape summary ─────────────────────────────────────────────
    section("Dataset Shape Before vs After Engineering")
    c1,c2,c3 = st.columns(3)
    c1.metric("Input Rows",    str(before_shape[0]) if before_shape else "—")
    c2.metric("Output Rows",   str(after_shape[0])  if after_shape  else "—")
    c3.metric("New Features",  str(len(feature_log)))

    # ── Feature log table ─────────────────────────────────────────
    section("Engineered Features Log")
    if feature_log:
        df_fl = pd.DataFrame(feature_log)
        for col in ["feature","source_columns","rationale"]:
            if col not in df_fl.columns:
                df_fl[col] = "—"
        st.dataframe(
            df_fl[["feature","source_columns","rationale"]].rename(columns={
                "feature":"Feature","source_columns":"Source Columns","rationale":"Business Rationale"
            }),
            use_container_width=True, hide_index=True
        )
    else:
        na_box("Feature engineering log not available. Re-run pipeline.")

    # ── Imputation summary ────────────────────────────────────────
    section("Imputation Summary")
    fe_summary = dqr_report.get("feature_engineering_summary","")
    if fe_summary:
        st.markdown(fe_summary)
    else:
        st.markdown("""
**Standard imputation strategy applied:**
| Type | Strategy |
|---|---|
| Numeric | Median imputation |
| Categorical | Mode imputation (→ "unknown" if empty) |
| Datetime | Converted to credit_age_months (days / 30.4375) |
""")

    # ── LLM narrative ─────────────────────────────────────────────
    llm_box(dqr_report.get("feature_engineering_narrative",""), "Feature Engineering LLM Summary")


# ─────────────────────────────────────────────────────────────────────────────
# Page 4 — Variable Selection
# ─────────────────────────────────────────────────────────────────────────────

def page_variable_selection(data: dict):
    st.title("4️⃣ Variable Selection")
    st.caption(f"Run: `{data.get('run_id','')}`")

    iv_records        = data.get("iv_table", [])
    selected_features = data.get("selected_features", [])
    rejected_features = data.get("rejected_features", {})
    fi                = data.get("feature_importance", {})
    dqr_report        = data.get("dqr_report", {})

    # ── Summary metrics ───────────────────────────────────────────
    c1,c2,c3 = st.columns(3)
    c1.metric("Features Selected", str(len(selected_features)))
    c2.metric("Features Rejected", str(len(rejected_features)))
    c3.metric("IV Records",        str(len(iv_records)))

    # ── IV table ──────────────────────────────────────────────────
    section("Information Value (IV) — All Features")
    if iv_records:
        df_iv = pd.DataFrame(iv_records)
        if "iv" in df_iv.columns:
            df_iv = df_iv.sort_values("iv", ascending=False)
        df_iv["Selected"] = df_iv["feature"].isin(selected_features).map({True:"✅",False:"❌"})

        col_a, col_b = st.columns([2, 3])
        with col_a:
            rename = {"feature":"Feature","iv":"IV","strength":"Strength","Selected":"Selected"}
            show_cols = [c for c in ["feature","iv","strength","Selected"] if c in df_iv.columns]
            st.dataframe(df_iv[show_cols].rename(columns=rename),
                         use_container_width=True, hide_index=True)
        with col_b:
            top25 = df_iv.head(25)
            if "feature" in top25.columns and "iv" in top25.columns:
                colors = [PASS_COLOR if f in selected_features else "#666"
                          for f in top25["feature"]]
                fig = go.Figure(go.Bar(
                    x=top25["iv"].tolist(), y=top25["feature"].tolist(),
                    orientation="h",
                    marker_color=colors,
                    text=top25.get("strength", pd.Series([""] * len(top25))).tolist(),
                    textposition="outside",
                ))
                fig.update_layout(title="Top 25 by IV (green = selected)",
                                  height=600, margin=dict(t=40,b=10),
                                  yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, use_container_width=True)
    else:
        na_box("IV table not available. Re-run pipeline to generate full IV analysis.")

    # ── Selected features ─────────────────────────────────────────
    section("Selected Features")
    if selected_features:
        if iv_records:
            df_iv  = pd.DataFrame(iv_records)
            df_sel = df_iv[df_iv["feature"].isin(selected_features)]
            if "iv" in df_sel.columns:
                df_sel = df_sel.sort_values("iv", ascending=False)
            show = [c for c in ["feature","iv","strength"] if c in df_sel.columns]
            st.dataframe(df_sel[show].rename(columns={"feature":"Feature","iv":"IV","strength":"Strength"}),
                         use_container_width=True, hide_index=True)
        else:
            st.dataframe(pd.DataFrame({"Feature": selected_features}), hide_index=True)
    else:
        na_box("Selected features list not available.")

    # ── RF / SHAP importance ──────────────────────────────────────
    if fi:
        section("Feature Importance (SHAP / RF)")
        df_fi = pd.DataFrame(list(fi.items()), columns=["Feature","Importance"]).sort_values("Importance")
        fig = px.bar(df_fi.tail(20), x="Importance", y="Feature", orientation="h",
                     title="Top 20 Features by Importance",
                     color="Importance", color_continuous_scale="Blues")
        fig.update_layout(height=480, margin=dict(t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)

    # ── Rejected features ─────────────────────────────────────────
    if rejected_features:
        section("Rejected Features with Reasons")
        df_rej = pd.DataFrame([{"Feature":k,"Rejection Reason":v} for k,v in rejected_features.items()])
        st.dataframe(df_rej, use_container_width=True, hide_index=True)

    # ── LLM rationale ─────────────────────────────────────────────
    llm_box(dqr_report.get("variable_selection_rationale",""), "Variable Selection LLM Rationale")


# ─────────────────────────────────────────────────────────────────────────────
# Page 5 — Model Development
# ─────────────────────────────────────────────────────────────────────────────

def page_model_development(data: dict):
    st.title("5️⃣ Model Development")
    st.caption(f"Run: `{data.get('run_id','')}`")

    model_metrics = data.get("model_metrics", {})
    champion_name = data.get("champion_model_name","—")
    rationale     = data.get("model_selection_rationale","")

    # ── Champion summary ──────────────────────────────────────────
    section("Champion Model")
    champion = model_metrics.get(champion_name, {})
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Champion",    champion_name)
    c2.metric("AUC (Test)",  str(champion.get("auc_test","—")))
    c3.metric("KS",          str(champion.get("ks","—")))
    c4.metric("Gini",        str(champion.get("gini","—")))
    c5.metric("Overfit Gap", str(champion.get("overfit","—")))

    # ── Model comparison table ────────────────────────────────────
    section("Model Comparison Table")
    if model_metrics:
        rows = []
        for name, m in model_metrics.items():
            rows.append({
                "Model":      name,
                "AUC Train":  m.get("auc_train","—"),
                "AUC Test":   m.get("auc_test","—"),
                "Gini":       m.get("gini","—"),
                "KS":         m.get("ks","—"),
                "Precision":  m.get("precision","—"),
                "Recall":     m.get("recall","—"),
                "F1":         m.get("f1","—"),
                "Overfit":    m.get("overfit","—"),
                "Champion":   "✅" if name == champion_name else "",
            })
        df_m = pd.DataFrame(rows)
        st.dataframe(df_m, use_container_width=True, hide_index=True)

        # Grouped bar comparison
        numeric_df = df_m[["Model","AUC Test","KS","Gini"]].copy()
        for col in ["AUC Test","KS","Gini"]:
            numeric_df[col] = pd.to_numeric(numeric_df[col], errors="coerce")
        df_melt = numeric_df.melt(id_vars="Model", var_name="Metric", value_name="Value")
        fig = px.bar(df_melt, x="Model", y="Value", color="Metric", barmode="group",
                     title="Model Performance Comparison",
                     color_discrete_sequence=px.colors.qualitative.Set1)
        fig.update_layout(height=350, margin=dict(t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        na_box("Model metrics not available. Re-run pipeline.")

    # ── Train vs Test overfit check ───────────────────────────────
    if model_metrics and any("auc_train" in m for m in model_metrics.values()):
        section("Training vs Test — Overfit Check")
        fig = go.Figure()
        colors_train = ["#42a5f5","#66bb6a","#ffa726"]
        colors_test  = ["#1565c0","#2e7d32","#e65100"]
        for i, (name, m) in enumerate(model_metrics.items()):
            if "auc_train" in m:
                fig.add_trace(go.Bar(name=f"{name} Train", x=[name],
                                     y=[float(m["auc_train"])], marker_color=colors_train[i % 3]))
                fig.add_trace(go.Bar(name=f"{name} Test",  x=[name],
                                     y=[float(m["auc_test"])],  marker_color=colors_test[i % 3]))
        fig.update_layout(barmode="group", title="AUC: Train vs Test",
                          height=300, margin=dict(t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)

    # ── LLM rationale ─────────────────────────────────────────────
    llm_box(rationale, "Model Selection LLM Rationale")


# ─────────────────────────────────────────────────────────────────────────────
# Page 6 — Explainability
# ─────────────────────────────────────────────────────────────────────────────

def page_explainability(data: dict):
    st.title("6️⃣ Explainability")
    st.caption(f"Run: `{data.get('run_id','')}`")

    fi            = data.get("feature_importance", {})
    adverse_codes = data.get("adverse_action_codes", {})
    shap_summary  = data.get("shap_summary","")
    champion_name = data.get("champion_model_name","—")

    # ── SHAP importance chart ─────────────────────────────────────
    section("SHAP Feature Importance")
    if fi:
        df_fi = pd.DataFrame(list(fi.items()), columns=["Feature","Mean |SHAP|"])
        df_fi = df_fi.sort_values("Mean |SHAP|").tail(20)
        fig = px.bar(df_fi, x="Mean |SHAP|", y="Feature", orientation="h",
                     title=f"Top 20 SHAP Drivers — {champion_name}",
                     color="Mean |SHAP|", color_continuous_scale="Oranges")
        fig.update_layout(height=520, margin=dict(t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)

        # Top 10 driver cards with business context
        section("Top 10 Risk Drivers — Business Interpretation")
        business_ctx = {
            "int_rate":           "Higher interest rate indicates higher-risk borrowers per underwriting guidelines",
            "dti":                "High debt-to-income ratio signals borrower is overextended",
            "revol_util":         "High revolving utilisation suggests reliance on credit",
            "loan_amnt":          "Larger loan amounts may exceed repayment capacity",
            "annual_inc":         "Lower income reduces ability to service debt",
            "credit_age_months":  "Shorter credit history indicates limited track record",
            "installment":        "Higher monthly obligation increases default risk",
            "open_acc":           "More open accounts may indicate credit-seeking behaviour",
            "total_acc":          "Total accounts reflects credit history breadth",
        }
        df_top = pd.DataFrame(list(fi.items()), columns=["Feature","Mean |SHAP|"]).head(10)
        for i, (_, row) in enumerate(df_top.iterrows(), 1):
            feat = row["Feature"]
            sv   = round(float(row["Mean |SHAP|"]), 5)
            ctx  = business_ctx.get(feat, f"Drives credit default risk prediction for {feat}")
            st.markdown(f"""
<div style="border-left:4px solid #ff7043;padding:8px 16px;margin:6px 0;
            background:#1e1e2e;border-radius:4px">
  <b>#{i} {feat}</b>&nbsp;
  <span style="color:#ff7043">mean |SHAP| = {sv}</span><br>
  <span style="font-size:0.85rem;color:#bdbdbd">{ctx}</span>
</div>""", unsafe_allow_html=True)
    else:
        na_box("SHAP feature importance not available. Re-run pipeline.")

    # ── Adverse action codes ──────────────────────────────────────
    section("Adverse Action Reason Codes")
    if adverse_codes:
        for sample_id, info in adverse_codes.items():
            prob    = info.get("predicted_prob","—")
            reasons = info.get("top_reasons",[])
            with st.expander(f"Sample `{sample_id}` — Predicted Default Prob: **{prob}**"):
                if reasons:
                    for r in reasons:
                        st.markdown(
                            f"- **{r.get('feature')}** = `{r.get('value')}`  \n"
                            f"  *{r.get('reason_code','—')}*  "
                            f"(SHAP: `{r.get('shap','—')}`)"
                        )
                else:
                    st.write("No reason codes available.")
    else:
        na_box("Adverse action codes not available. Re-run pipeline.")

    # ── LLM narrative ─────────────────────────────────────────────
    llm_box(shap_summary, "Model Explainability LLM Narrative")


# ─────────────────────────────────────────────────────────────────────────────
# Page 7 — Validation
# ─────────────────────────────────────────────────────────────────────────────

def page_validation(data: dict):
    st.title("7️⃣ Validation")
    st.caption(f"Run: `{data.get('run_id','')}`")

    vm            = data.get("validation_metrics", {})
    psi           = data.get("psi_results", {})
    val_summary   = data.get("validation_summary","")
    val_passed    = data.get("validation_passed", False)
    champion_name = data.get("champion_model_name","—")

    # ── PASS / FAIL banner ────────────────────────────────────────
    if val_passed:
        st.markdown(f"""
<div style="background:{PASS_COLOR};color:#fff;padding:18px 24px;border-radius:8px;
            font-size:1.4rem;font-weight:700;text-align:center;margin-bottom:16px">
  ✅ VALIDATION PASSED — {champion_name} approved for deployment consideration
</div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
<div style="background:{WARN_COLOR};color:#000;padding:18px 24px;border-radius:8px;
            font-size:1.4rem;font-weight:700;text-align:center;margin-bottom:16px">
  ⚠️ CONDITIONAL — Review required before deployment
</div>""", unsafe_allow_html=True)

    # ── Key metrics ───────────────────────────────────────────────
    section("Discrimination & Calibration Metrics")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("AUC",               str(vm.get("auc","—")))
    c2.metric("Gini",              str(vm.get("gini","—")))
    c3.metric("KS Statistic",      str(vm.get("ks","—")))
    c4.metric("Brier Score",       str(vm.get("brier_score","—")))
    c5.metric("Calibration Error", str(vm.get("calibration_error","—")))

    # ── PSI ───────────────────────────────────────────────────────
    section("Population Stability Index (PSI)")
    psi_score = psi.get("psi_score","—")
    psi_asmt  = psi.get("assessment","—")
    psi_label = psi.get("split_label","—")

    try:
        psi_float = float(psi_score)
        psi_color = PASS_COLOR if psi_float < 0.10 else WARN_COLOR if psi_float < 0.25 else FAIL_COLOR
    except Exception:
        psi_color = "#666"

    col_a, col_b = st.columns([1,2])
    with col_a:
        st.markdown(f"""
<div style="border:2px solid {psi_color};border-radius:8px;padding:20px;text-align:center">
  <div style="font-size:2.5rem;font-weight:700;color:{psi_color}">{psi_score}</div>
  <div style="font-size:1.1rem;font-weight:600">{psi_asmt}</div>
  <div style="font-size:0.8rem;color:#9e9e9e;margin-top:4px">{psi_label}</div>
</div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown("""
**PSI Thresholds (industry standard):**
| Range | Assessment |
|---|---|
| < 0.10 | ✅ Stable — no action required |
| 0.10 – 0.25 | ⚠️ Moderate shift — monitor closely |
| > 0.25 | ❌ Significant shift — recalibration recommended |
""")

    # ── Score decile table ────────────────────────────────────────
    decile_tbl = vm.get("decile_table",[])
    if decile_tbl:
        section("Score Decile Table")
        df_dec = pd.DataFrame(decile_tbl)
        if "bad_rate" in df_dec.columns:
            df_dec["bad_rate"] = df_dec["bad_rate"].apply(lambda x: f"{float(x):.2%}")
        if "avg_prob" in df_dec.columns:
            df_dec["avg_prob"] = df_dec["avg_prob"].apply(lambda x: f"{float(x):.4f}")

        col_a, col_b = st.columns([1,2])
        with col_a:
            st.dataframe(df_dec.rename(columns={
                "decile":"Decile","n":"N","n_bad":"N Bad",
                "avg_prob":"Avg Score","bad_rate":"Bad Rate"
            }), use_container_width=True, hide_index=True)
        with col_b:
            df_plot = pd.DataFrame(decile_tbl)
            if "bad_rate" in df_plot.columns and "decile" in df_plot.columns:
                df_plot["bad_rate"] = pd.to_numeric(df_plot["bad_rate"], errors="coerce")
                fig = px.bar(df_plot, x="decile", y="bad_rate",
                             title="Bad Rate by Score Decile",
                             color="bad_rate", color_continuous_scale="Reds",
                             labels={"decile":"Decile","bad_rate":"Bad Rate"})
                fig.update_layout(height=300, margin=dict(t=40,b=10))
                st.plotly_chart(fig, use_container_width=True)

    # ── Challenger comparison ─────────────────────────────────────
    challenger_tbl = vm.get("challenger_table",[])
    if challenger_tbl:
        section("Champion vs Challenger Comparison")
        st.dataframe(pd.DataFrame(challenger_tbl), use_container_width=True, hide_index=True)

    # ── LLM validation summary ────────────────────────────────────
    llm_box(val_summary, "Validation LLM Summary")

    # ── Full model report ─────────────────────────────────────────
    run_id   = data.get("run_id","")
    rpt_path = find_report_for(run_id)
    if rpt_path:
        section("Full Model Development Report")
        with open(rpt_path, encoding="utf-8") as f:
            report_txt = f.read()
        with st.expander("📄 View Full Report", expanded=False):
            st.text(report_txt)
        st.download_button("⬇️ Download Report (.txt)", data=report_txt,
                           file_name=f"{run_id}_model_report.txt", mime="text/plain")

    # ── Raw audit JSON download ───────────────────────────────────
    audit_path = st.session_state.get("audit_path")
    if audit_path and os.path.exists(audit_path):
        with open(audit_path, encoding="utf-8") as f:
            audit_txt = f.read()
        st.download_button("⬇️ Download Audit Trail (.json)", data=audit_txt,
                           file_name=os.path.basename(audit_path), mime="application/json")


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint 1 — Target Definition Review
# ─────────────────────────────────────────────────────────────────────────────

def page_checkpoint_1(data):
    st.title("👤 Checkpoint 1 — Target Definition Review")
    st.markdown("""
> **Analyst Action Required:** Review the target variable definition below.
> Approve to unlock Phase 2 (Data Quality Review). Reject to halt the pipeline.
""")

    if data is None:
        st.warning("No pipeline run loaded. Use **📂 Load** in the sidebar first.")
        return

    run_id   = data.get("run_id","")
    cp_all   = st.session_state.get("cp_state", {})
    existing = cp_all.get(run_id, {}).get("cp1", {})

    # ── Current definition ────────────────────────────────────────
    section("Current Target Definition")
    target_col = data.get("target_column","—")
    target_def = data.get("target_definition","")
    leakage    = data.get("leakage_columns",[])
    dqr_flags  = data.get("dqr_flags",[])

    st.info(f"**Target Column:** `{target_col}`")
    st.markdown(f"**Definition:** {target_def or '*(not available)*'}")

    if leakage:
        st.warning(f"**{len(leakage)} leakage columns** auto-excluded: "
                   f"`{'`, `'.join(leakage[:8])}`{'...' if len(leakage)>8 else ''}")

    if dqr_flags:
        with st.expander(f"DQR Flags ({len(dqr_flags)})", expanded=False):
            for flag in dqr_flags:
                st.write(f"⚠️ {flag}")

    st.markdown("---")

    # ── Analyst form ──────────────────────────────────────────────
    section("Analyst Review")
    override = st.text_area(
        "Override target definition (leave blank to keep current):",
        value=existing.get("override_definition",""), height=80)
    notes = st.text_area(
        "Review notes:", value=existing.get("notes",""), height=80)
    decision = st.radio(
        "Decision:",
        options=["approve","reject"],
        index=0 if existing.get("decision","approve") == "approve" else 1,
        format_func=lambda x: "✅ Approve — Proceed to Phase 2" if x == "approve"
                              else "❌ Reject — Halt Pipeline",
        horizontal=True,
    )

    if st.button("💾 Save Checkpoint 1", type="primary"):
        cp_all.setdefault(run_id, {})["cp1"] = {
            "decision":             decision,
            "override_definition":  override.strip(),
            "notes":                notes.strip(),
            "analyst_timestamp":    datetime.datetime.now().isoformat(),
        }
        save_checkpoint_state(cp_all)
        st.session_state["cp_state"] = cp_all
        if decision == "approve":
            st.success("✅ Checkpoint 1 approved. Proceed to Phase 2.")
        else:
            st.error("❌ Checkpoint 1 rejected. Pipeline halted.")

    if existing:
        st.markdown("---")
        st.markdown(f"""
**Recorded decision:** {existing.get('decision','—').upper()}
**Saved at:** {existing.get('analyst_timestamp','—')}
**Notes:** {existing.get('notes','—') or '—'}
""")


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint 2 — Feature Shortlist Approval
# ─────────────────────────────────────────────────────────────────────────────

def page_checkpoint_2(data):
    st.title("👤 Checkpoint 2 — Feature Shortlist Approval")
    st.markdown("""
> **Analyst Action Required:** Uncheck any features to remove them.
> Approve the shortlist to proceed to Model Development.
""")

    if data is None:
        st.warning("No pipeline run loaded. Use **📂 Load** in the sidebar first.")
        return

    run_id            = data.get("run_id","")
    cp_all            = st.session_state.get("cp_state", {})
    existing          = cp_all.get(run_id, {}).get("cp2", {})
    selected_features = data.get("selected_features",[])
    iv_records        = data.get("iv_table",[])
    rejected_features = data.get("rejected_features",{})

    iv_lookup = {}
    if iv_records:
        for rec in iv_records:
            iv_lookup[rec.get("feature","")] = (
                round(float(rec.get("iv",0)), 4), rec.get("strength","—"))

    section(f"Feature Shortlist — {len(selected_features)} features selected by pipeline")
    prev_removed = set(existing.get("removed_features",[]))

    checked_features = []
    removed_features = []
    cols = st.columns(2)
    for i, feat in enumerate(selected_features):
        iv_val, iv_str = iv_lookup.get(feat, ("—","—"))
        label = f"**{feat}** — IV: {iv_val} ({iv_str})"
        with cols[i % 2]:
            checked = st.checkbox(label, value=(feat not in prev_removed), key=f"cp2_{feat}")
            (checked_features if checked else removed_features).append(feat)

    st.markdown("---")

    if rejected_features:
        with st.expander(f"Pipeline-Rejected Features Reference ({len(rejected_features)})", expanded=False):
            df_rej = pd.DataFrame([{"Feature":k,"Reason":v} for k,v in rejected_features.items()])
            st.dataframe(df_rej, use_container_width=True, hide_index=True)

    notes = st.text_area("Review notes:", value=existing.get("notes",""), height=80)

    if st.button("✅ Approve Feature Shortlist", type="primary"):
        cp_all.setdefault(run_id, {})["cp2"] = {
            "decision":          "approve",
            "approved_features": checked_features,
            "removed_features":  removed_features,
            "notes":             notes.strip(),
            "analyst_timestamp": datetime.datetime.now().isoformat(),
        }
        save_checkpoint_state(cp_all)
        st.session_state["cp_state"] = cp_all
        st.success(f"✅ Approved {len(checked_features)} features"
                   + (f". Removed: {', '.join(removed_features)}" if removed_features else "."))

    if existing:
        st.markdown("---")
        st.markdown(f"""
**Status:** {existing.get('decision','—').upper()}
**Approved:** {len(existing.get('approved_features',[]))} features
**Removed:** {', '.join(existing.get('removed_features',[])) or 'none'}
**Saved at:** {existing.get('analyst_timestamp','—')}
""")


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint 3 — Model Sign-Off
# ─────────────────────────────────────────────────────────────────────────────

def page_checkpoint_3(data):
    st.title("👤 Checkpoint 3 — Model Sign-Off")
    st.markdown("""
> **Analyst Action Required:** Review the validation summary and sign off.
> Your digital signature will be recorded in the audit trail.
""")

    if data is None:
        st.warning("No pipeline run loaded. Use **📂 Load** in the sidebar first.")
        return

    run_id        = data.get("run_id","")
    cp_all        = st.session_state.get("cp_state", {})
    existing      = cp_all.get(run_id, {}).get("cp3", {})
    vm            = data.get("validation_metrics", {})
    psi           = data.get("psi_results", {})
    val_passed    = data.get("validation_passed", False)
    val_summary   = data.get("validation_summary","")
    champion_name = data.get("champion_model_name","—")

    # ── Validation snapshot ───────────────────────────────────────
    section("Validation Summary")
    color  = PASS_COLOR if val_passed else WARN_COLOR
    status = "PASSED" if val_passed else "CONDITIONAL"
    st.markdown(f"""
<div style="border:2px solid {color};border-radius:8px;padding:14px 18px;margin-bottom:12px">
  <b>Champion:</b> {champion_name} &nbsp;|&nbsp;
  <b>Outcome:</b> <span style="color:{color};font-weight:700">{status}</span>
</div>""", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("AUC",  str(vm.get("auc","—")))
    c2.metric("Gini", str(vm.get("gini","—")))
    c3.metric("KS",   str(vm.get("ks","—")))
    c4.metric("PSI",  f"{psi.get('psi_score','—')} ({psi.get('assessment','—')})")

    if val_summary:
        with st.expander("📋 LLM Validation Summary", expanded=True):
            st.markdown(val_summary)

    st.markdown("---")

    # ── Decision form ─────────────────────────────────────────────
    section("Sign-Off Decision")
    decision = st.radio(
        "Governance decision:",
        options=["approve","changes","reject"],
        index=["approve","changes","reject"].index(existing.get("decision","approve")),
        format_func=lambda x: {
            "approve": "✅ Approve for Deployment",
            "changes": "⚠️ Request Changes",
            "reject":  "❌ Reject Model",
        }[x],
    )
    notes = st.text_area(
        "Sign-off notes / conditions:",
        value=existing.get("notes",""), height=100,
        placeholder="e.g. 'Approved subject to quarterly PSI monitoring. SHAP analysis reviewed.'")

    st.markdown("---")
    st.markdown("**Digital Signature**")
    col_a, col_b = st.columns(2)
    analyst_name = col_a.text_input("Analyst Name:", value=existing.get("analyst_name",""))
    default_date = datetime.date.today()
    if existing.get("analyst_date"):
        try: default_date = datetime.date.fromisoformat(existing["analyst_date"])
        except Exception: pass
    analyst_date = col_b.date_input("Date:", value=default_date)

    if st.button("💾 Save Sign-Off", type="primary"):
        if not analyst_name.strip():
            st.error("Analyst name is required.")
        else:
            entry = {
                "decision":          decision,
                "notes":             notes.strip(),
                "analyst_name":      analyst_name.strip(),
                "analyst_date":      str(analyst_date),
                "run_id":            run_id,
                "champion_model":    champion_name,
                "auc":               vm.get("auc"),
                "gini":              vm.get("gini"),
                "ks":                vm.get("ks"),
                "psi":               psi.get("psi_score"),
                "validation_passed": val_passed,
                "analyst_timestamp": datetime.datetime.now().isoformat(),
            }
            cp_all.setdefault(run_id, {})["cp3"] = entry
            save_checkpoint_state(cp_all)
            st.session_state["cp_state"] = cp_all
            msgs = {
                "approve": f"✅ Model approved by **{analyst_name}** on {analyst_date}.",
                "changes": f"⚠️ Changes requested by **{analyst_name}**.",
                "reject":  f"❌ Model rejected by **{analyst_name}** on {analyst_date}.",
            }
            getattr(st, {"approve":"success","changes":"warning","reject":"error"}[decision])(
                msgs[decision])

    if existing:
        icons = {"approve":"✅","changes":"⚠️","reject":"❌"}
        dec   = existing.get("decision","—")
        st.markdown("---")
        st.markdown(f"""
### 📋 Recorded Sign-Off

| Field | Value |
|---|---|
| Decision | {icons.get(dec,"")} **{dec.upper()}** |
| Analyst | {existing.get('analyst_name','—')} |
| Date | {existing.get('analyst_date','—')} |
| Timestamp | {existing.get('analyst_timestamp','—')} |
| AUC | {existing.get('auc','—')} |
| Champion | {existing.get('champion_model','—')} |
| Notes | {existing.get('notes','—') or '—'} |
""")


# ─────────────────────────────────────────────────────────────────────────────
# Home screen
# ─────────────────────────────────────────────────────────────────────────────

def page_home():
    st.markdown("""
<div style="text-align:center;padding:60px 20px">
  <div style="font-size:4rem">🏦</div>
  <h1>Credit Risk Factory</h1>
  <p style="font-size:1.1rem;color:#9e9e9e">
    Agentic ML Pipeline for Credit Risk Modelling
  </p>
  <p style="color:#757575;max-width:600px;margin:0 auto">
    Upload a CSV dataset and click <b>▶ Run</b> to execute the full 7-phase pipeline,
    or click <b>📂 Load</b> to view results from a previous run.
  </p>
</div>
""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    items = [
        (c1, "🔍", "7 AI Agents",
         "Data Understanding → DQR → Feature Engineering → Variable Selection "
         "→ Modelling → Explainability → Validation"),
        (c2, "👤", "Human-in-Loop",
         "3 mandatory analyst checkpoints with audit trail and digital sign-off"),
        (c3, "📊", "Full Reports",
         "LLM narratives, SHAP explanations, PSI monitoring, and downloadable governance report"),
    ]
    for col, icon, title, desc in items:
        with col:
            st.markdown(f"""
<div style="border:1px solid #333;border-radius:8px;padding:20px;text-align:center;
            min-height:160px">
  <div style="font-size:2rem">{icon}</div>
  <b>{title}</b><br>
  <span style="font-size:0.82rem;color:#9e9e9e">{desc}</span>
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────

build_sidebar()

data = st.session_state.get("data")
page = st.session_state.get("page", "home")

PAGE_MAP = {
    "p1":  page_data_understanding,
    "p2":  page_dqr,
    "p3":  page_feature_engineering,
    "p4":  page_variable_selection,
    "p5":  page_model_development,
    "p6":  page_explainability,
    "p7":  page_validation,
    "cp1": page_checkpoint_1,
    "cp2": page_checkpoint_2,
    "cp3": page_checkpoint_3,
}

if page in PAGE_MAP and (data is not None or page in ("cp1","cp2","cp3")):
    PAGE_MAP[page](data)
else:
    page_home()
