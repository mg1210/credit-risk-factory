"""
agents/data_understanding_agent.py
────────────────────────────────────
Phase 1 — Data Understanding Agent

Responsibilities:
  • Load and profile the raw dataset
  • Identify column types (numeric, categorical, date, id, text)
  • Define the target variable from loan_status
  • Flag post-origination / data-leakage columns
  • Use LLM to generate business-meaning annotations for each field
"""

import pandas as pd
import numpy as np
from core.base_agent import BaseAgent
from core.state import PipelineState
from core.llm import ask_json, ask, CREDIT_RISK_SYSTEM


# Columns that are clearly post-origination and would cause target leakage
KNOWN_LEAKAGE = [
    "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "recoveries", "collection_recovery_fee", "last_pymnt_d",
    "last_pymnt_amnt", "next_pymnt_d", "last_credit_pull_d",
]

# Identifier / admin columns — not useful as features
ID_COLS = ["Record_No", "id", "member_id", "url", "desc", "title",
           "zip_code", "emp_title", "pymnt_plan", "policy_code"]

# Columns to always drop (constants, near-zero-info)
ALWAYS_DROP = ["application_type"]   # only INDIVIDUAL in dev set


class DataUnderstandingAgent(BaseAgent):

    def __init__(self, verbose: bool = True):
        super().__init__("DataUnderstandingAgent", verbose)

    # ─────────────────────────────────────────────────────────────
    def run(self, state: PipelineState) -> PipelineState:

        # 1. Load data
        self._log("Loading dataset …")
        df = pd.read_csv(state.dataset_path, low_memory=False)
        state.raw_df = df.copy()
        state.working_df = df.copy()
        self._info(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
        state.log_audit(self.name, "data_loaded",
                        f"{df.shape[0]} rows, {df.shape[1]} cols")

        # 2. Define target variable
        self._log("Defining target variable …")
        state.target_column = "loan_status"
        state = self._define_target(state)

        # 3. Classify columns
        self._log("Classifying columns …")
        state = self._classify_columns(state, df)

        # 4. Build schema profile
        self._log("Building schema profile …")
        state = self._build_schema_profile(state, df)

        # 5. LLM — annotate top features with business meaning
        self._log("Asking LLM to annotate field business meanings …")
        state = self._llm_annotate(state, df)

        return state

    # ─────────────────────────────────────────────────────────────
    def _define_target(self, state: PipelineState) -> PipelineState:
        df = state.working_df
        counts = df["loan_status"].value_counts().to_dict()
        self._info(f"loan_status distribution: {counts}")

        # Binary target: Charged Off = 1, Fully Paid = 0
        # Drop ambiguous statuses (Current, In Grace Period, Late, etc.)
        keep_statuses = {"Fully Paid", "Charged Off"}
        before = len(df)
        df = df[df["loan_status"].isin(keep_statuses)].copy()
        after = len(df)
        dropped = before - after
        self._info(f"Dropped {dropped:,} ambiguous rows (Current / In Grace Period / Late)")

        df["target"] = (df["loan_status"] == "Charged Off").astype(int)
        default_rate = df["target"].mean()
        self._info(f"Default rate: {default_rate:.2%}  ({df['target'].sum():,} defaults / {len(df):,} total)")

        state.working_df = df
        state.target_column = "target"
        state.target_definition = (
            f"Binary default flag: 1 = Charged Off, 0 = Fully Paid. "
            f"Derived from loan_status. Default rate: {default_rate:.2%}. "
            f"Ambiguous statuses (Current, In Grace Period, Late) excluded. "
            f"Final sample: {len(df):,} observations."
        )
        state.log_audit(self.name, "target_defined", state.target_definition)
        return state

    # ─────────────────────────────────────────────────────────────
    def _classify_columns(self, state: PipelineState, df: pd.DataFrame) -> PipelineState:
        state.leakage_columns = [c for c in KNOWN_LEAKAGE if c in df.columns]
        state.id_columns      = [c for c in ID_COLS if c in df.columns]
        state.drop_columns    = state.leakage_columns + state.id_columns + ALWAYS_DROP

        remaining = [c for c in df.columns
                     if c not in state.drop_columns
                     and c not in ("loan_status", "target")]

        date_keywords = ["_d", "earliest_cr", "issue_d", "payment_date"]
        state.date_columns = [c for c in remaining
                              if any(k in c for k in date_keywords)
                              or (df[c].dtype == object and
                                  df[c].dropna().head(20).astype(str)
                                  .str.match(r'^[A-Z][a-z]{2}-\d{2}$').mean() > 0.5)]

        state.categorical_columns = [
            c for c in remaining
            if c not in state.date_columns
            and (df[c].dtype == object or df[c].nunique() < 15)
        ]
        state.numeric_columns = [
            c for c in remaining
            if c not in state.date_columns
            and c not in state.categorical_columns
        ]

        self._info(f"Leakage cols flagged : {len(state.leakage_columns)}")
        self._info(f"ID / admin cols      : {len(state.id_columns)}")
        self._info(f"Date cols            : {len(state.date_columns)}")
        self._info(f"Categorical cols     : {len(state.categorical_columns)}")
        self._info(f"Numeric cols         : {len(state.numeric_columns)}")
        return state

    # ─────────────────────────────────────────────────────────────
    def _build_schema_profile(self, state: PipelineState, df: pd.DataFrame) -> PipelineState:
        profile = {}
        for col in df.columns:
            missing_pct = df[col].isna().mean()
            profile[col] = {
                "dtype"       : str(df[col].dtype),
                "missing_pct" : round(float(missing_pct), 4),
                "n_unique"    : int(df[col].nunique()),
                "sample_vals" : df[col].dropna().head(3).astype(str).tolist(),
                "role"        : (
                    "leakage"     if col in state.leakage_columns else
                    "identifier"  if col in state.id_columns else
                    "date"        if col in state.date_columns else
                    "categorical" if col in state.categorical_columns else
                    "numeric"     if col in state.numeric_columns else
                    "target"      if col in ("loan_status", "target") else
                    "other"
                ),
            }
        state.schema_profile = profile
        return state

    # ─────────────────────────────────────────────────────────────
    def _llm_annotate(self, state: PipelineState, df: pd.DataFrame) -> PipelineState:
        # Send top modelling columns to LLM for business annotation
        candidate_cols = state.numeric_columns[:20] + state.categorical_columns[:10]
        col_list = "\n".join(
            f"- {c} (dtype={df[c].dtype}, missing={df[c].isna().mean():.1%}, "
            f"sample={df[c].dropna().head(2).tolist()})"
            for c in candidate_cols if c in df.columns
        )
        prompt = f"""
You are reviewing a Lending Club consumer loan dataset for credit risk modelling.
Here are the candidate modelling columns:

{col_list}

For each column, provide a one-line business interpretation and classify it as one of:
application_info, credit_bureau, loan_terms, behavioural, or other.

Respond ONLY with a JSON object:
{{"column_name": {{"meaning": "...", "category": "..."}} }}
"""
        try:
            annotations = ask_json(prompt, system=CREDIT_RISK_SYSTEM, max_tokens=2000)
            for col, info in annotations.items():
                if col in state.schema_profile:
                    state.schema_profile[col]["business_meaning"] = info.get("meaning", "")
                    state.schema_profile[col]["category"] = info.get("category", "")
            self._info(f"LLM annotated {len(annotations)} columns")
        except Exception as e:
            state.log_warning(self.name, f"LLM annotation skipped: {e}")
            self._info(f"LLM annotation skipped — {e}")

        return state
