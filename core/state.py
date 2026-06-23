"""
core/state.py
─────────────
Shared pipeline state object passed between every agent.
Acts as the single source of truth for the entire factory run.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime
import json
import os


@dataclass
class PipelineState:
    # ── Run metadata ──────────────────────────────────────────────
    run_id: str = field(default_factory=lambda: datetime.now().strftime("RUN_%Y%m%d_%H%M%S"))
    dataset_name: str = ""
    dataset_path: str = ""
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # ── Raw data ──────────────────────────────────────────────────
    raw_df: Any = None                  # original DataFrame (never mutated)
    working_df: Any = None              # mutable working copy

    # ── Phase 1: Data understanding ───────────────────────────────
    schema_profile: dict = field(default_factory=dict)
    target_column: str = ""
    target_definition: str = ""
    leakage_columns: list = field(default_factory=list)
    drop_columns: list = field(default_factory=list)
    id_columns: list = field(default_factory=list)
    date_columns: list = field(default_factory=list)
    categorical_columns: list = field(default_factory=list)
    numeric_columns: list = field(default_factory=list)

    # ── Phase 2: DQR ─────────────────────────────────────────────
    dqr_report: dict = field(default_factory=dict)
    missing_summary: dict = field(default_factory=dict)
    outlier_summary: dict = field(default_factory=dict)
    high_missing_cols: list = field(default_factory=list)   # >40% missing
    dqr_flags: list = field(default_factory=list)           # human-readable warnings

    # ── Phase 3: Feature engineering ─────────────────────────────
    engineered_df: Any = None
    feature_log: list = field(default_factory=list)         # what was created and why
    woe_bins: dict = field(default_factory=dict)            # WOE encoding maps

    # ── Phase 4: Variable selection ───────────────────────────────
    iv_table: Any = None                # DataFrame with IV scores
    selected_features: list = field(default_factory=list)
    rejected_features: dict = field(default_factory=dict)  # feature -> reason
    correlation_matrix: Any = None

    # ── Phase 5: Model development ───────────────────────────────
    X_train: Any = None
    X_test: Any = None
    y_train: Any = None
    y_test: Any = None
    trained_models: dict = field(default_factory=dict)      # name -> model object
    model_metrics: dict = field(default_factory=dict)       # name -> metrics dict
    champion_model_name: str = ""
    champion_model: Any = None
    model_selection_rationale: str = ""

    # ── Phase 6: Explainability ───────────────────────────────────
    shap_values: Any = None
    feature_importance: dict = field(default_factory=dict)
    shap_summary: str = ""
    adverse_action_codes: dict = field(default_factory=dict)

    # ── Phase 7: Validation ───────────────────────────────────────
    validation_metrics: dict = field(default_factory=dict)
    psi_results: dict = field(default_factory=dict)
    validation_summary: str = ""
    validation_passed: bool = False

    # ── Phase 8: Documentation ────────────────────────────────────
    model_report_path: str = ""
    audit_log: list = field(default_factory=list)

    # ── Human checkpoint flags ────────────────────────────────────
    checkpoint_1_approved: bool = False   # target definition confirmed
    checkpoint_2_approved: bool = False   # feature shortlist approved
    checkpoint_3_approved: bool = False   # model sign-off

    # ── Errors and warnings ───────────────────────────────────────
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def log_audit(self, agent: str, action: str, detail: str = ""):
        self.audit_log.append({
            "timestamp": datetime.now().isoformat(),
            "agent": agent,
            "action": action,
            "detail": detail
        })

    def log_error(self, agent: str, msg: str):
        self.errors.append({"agent": agent, "message": msg, "timestamp": datetime.now().isoformat()})

    def log_warning(self, agent: str, msg: str):
        self.warnings.append({"agent": agent, "message": msg, "timestamp": datetime.now().isoformat()})

    def to_summary_dict(self) -> dict:
        """Return a JSON-serialisable summary (no DataFrames)."""
        return {
            "run_id": self.run_id,
            "dataset_name": self.dataset_name,
            "target_column": self.target_column,
            "target_definition": self.target_definition,
            "leakage_columns": self.leakage_columns,
            "selected_features": self.selected_features,
            "champion_model_name": self.champion_model_name,
            "model_metrics": self.model_metrics,
            "validation_passed": self.validation_passed,
            "checkpoints": {
                "target_confirmed": self.checkpoint_1_approved,
                "features_approved": self.checkpoint_2_approved,
                "model_signed_off": self.checkpoint_3_approved,
            },
            "warnings": self.warnings,
            "errors": self.errors,
            "audit_log": self.audit_log,
        }
