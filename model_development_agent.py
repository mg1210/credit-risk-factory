"""
agents/model_development_agent.py
───────────────────────────────────
Phase 5 — Model Development Agent

Responsibilities:
  • Train-test split (stratified, time-aware if vintage available)
  • Train 3 candidate models: Logistic Regression, Random Forest, XGBoost
  • Hyperparameter optimisation via Optuna (XGBoost)
  • Compute performance metrics: AUC, KS, Gini, Precision, Recall, F1
  • LLM-driven champion selection with written rationale
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, roc_curve,
                              precision_recall_fscore_support, confusion_matrix)
from sklearn.pipeline import Pipeline
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from core.base_agent import BaseAgent
from core.state import PipelineState
from core.llm import ask, CREDIT_RISK_SYSTEM


class ModelDevelopmentAgent(BaseAgent):

    def __init__(self, test_size: float = 0.25, optuna_trials: int = 30,
                 verbose: bool = True):
        super().__init__("ModelDevelopmentAgent", verbose)
        self.test_size     = test_size
        self.optuna_trials = optuna_trials

    # ─────────────────────────────────────────────────────────────
    def run(self, state: PipelineState) -> PipelineState:
        df = state.engineered_df.copy()
        feats = [f for f in state.selected_features if f in df.columns]
        self._info(f"Modelling on {len(feats)} selected features")

        X = df[feats].fillna(0)
        y = df["target"]

        self._log("Creating train / test split …")
        state = self._split(state, X, y)

        self._log("Training Logistic Regression …")
        state = self._train_logistic(state)

        self._log("Training Random Forest …")
        state = self._train_rf(state)

        self._log("Training XGBoost with Optuna tuning …")
        state = self._train_xgb(state)

        self._log("Selecting champion model …")
        state = self._select_champion(state)

        self._log("Asking LLM for model selection rationale …")
        state = self._llm_rationale(state)

        return state

    # ─────────────────────────────────────────────────────────────
    def _split(self, state: PipelineState, X, y) -> PipelineState:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=self.test_size, random_state=42, stratify=y
        )
        state.X_train, state.X_test = X_tr, X_te
        state.y_train, state.y_test = y_tr, y_te
        self._info(f"Train: {len(X_tr):,} | Test: {len(X_te):,} | "
                   f"Default rate train: {y_tr.mean():.2%} | test: {y_te.mean():.2%}")
        return state

    # ─────────────────────────────────────────────────────────────
    def _metrics(self, model, X_tr, X_te, y_tr, y_te, name: str) -> dict:
        y_prob_tr = model.predict_proba(X_tr)[:, 1]
        y_prob_te = model.predict_proba(X_te)[:, 1]
        y_pred_te = (y_prob_te >= 0.5).astype(int)

        auc_tr  = roc_auc_score(y_tr, y_prob_tr)
        auc_te  = roc_auc_score(y_te, y_prob_te)
        gini_te = 2 * auc_te - 1

        fpr, tpr, _ = roc_curve(y_te, y_prob_te)
        ks_te = float(np.max(tpr - fpr))

        prec, rec, f1, _ = precision_recall_fscore_support(
            y_te, y_pred_te, average="binary", zero_division=0
        )
        return {
            "model"     : name,
            "auc_train" : round(auc_tr, 4),
            "auc_test"  : round(auc_te, 4),
            "gini"      : round(gini_te, 4),
            "ks"        : round(ks_te, 4),
            "precision" : round(float(prec), 4),
            "recall"    : round(float(rec), 4),
            "f1"        : round(float(f1), 4),
            "overfit"   : round(auc_tr - auc_te, 4),
        }

    # ─────────────────────────────────────────────────────────────
    def _train_logistic(self, state: PipelineState) -> PipelineState:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=500, C=0.1,
                                      class_weight="balanced", random_state=42))
        ])
        pipe.fit(state.X_train, state.y_train)
        m = self._metrics(pipe, state.X_train, state.X_test,
                          state.y_train, state.y_test, "LogisticRegression")
        state.trained_models["LogisticRegression"] = pipe
        state.model_metrics["LogisticRegression"]  = m
        self._info(f"  AUC={m['auc_test']:.4f}  KS={m['ks']:.4f}  Gini={m['gini']:.4f}")
        return state

    # ─────────────────────────────────────────────────────────────
    def _train_rf(self, state: PipelineState) -> PipelineState:
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=50,
            class_weight="balanced", random_state=42, n_jobs=-1
        )
        rf.fit(state.X_train, state.y_train)
        m = self._metrics(rf, state.X_train, state.X_test,
                          state.y_train, state.y_test, "RandomForest")
        state.trained_models["RandomForest"] = rf
        state.model_metrics["RandomForest"]  = m
        self._info(f"  AUC={m['auc_test']:.4f}  KS={m['ks']:.4f}  Gini={m['gini']:.4f}")
        return state

    # ─────────────────────────────────────────────────────────────
    def _train_xgb(self, state: PipelineState) -> PipelineState:
        X_tr, y_tr = state.X_train, state.y_train
        X_te, y_te = state.X_test,  state.y_test
        scale_pos  = float((y_tr == 0).sum() / (y_tr == 1).sum())

        def objective(trial):
            params = {
                "n_estimators"     : trial.suggest_int("n_estimators", 100, 500),
                "max_depth"        : trial.suggest_int("max_depth", 3, 7),
                "learning_rate"    : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample"        : trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight" : trial.suggest_int("min_child_weight", 5, 50),
                "scale_pos_weight" : scale_pos,
                "random_state"     : 42,
                "eval_metric"      : "auc",
                "use_label_encoder": False,
            }
            model = xgb.XGBClassifier(**params, verbosity=0)
            cv_scores = cross_val_score(model, X_tr, y_tr, cv=3,
                                        scoring="roc_auc", n_jobs=-1)
            return cv_scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.optuna_trials, show_progress_bar=False)

        best = study.best_params
        best["scale_pos_weight"] = scale_pos
        best["random_state"]     = 42
        best["eval_metric"]      = "auc"

        xgb_model = xgb.XGBClassifier(**best, verbosity=0)
        xgb_model.fit(X_tr, y_tr,
                      eval_set=[(X_te, y_te)],
                      verbose=False)

        m = self._metrics(xgb_model, X_tr, X_te, y_tr, y_te, "XGBoost")
        m["best_params"] = best
        state.trained_models["XGBoost"] = xgb_model
        state.model_metrics["XGBoost"]  = m
        self._info(f"  AUC={m['auc_test']:.4f}  KS={m['ks']:.4f}  Gini={m['gini']:.4f}")
        self._info(f"  Best params: {best}")
        return state

    # ─────────────────────────────────────────────────────────────
    def _select_champion(self, state: PipelineState) -> PipelineState:
        """
        Champion = highest AUC on test set, with overfit penalty.
        If overfit > 0.03 AUC, penalise that model.
        """
        scores = {}
        for name, m in state.model_metrics.items():
            penalty = max(0, m["overfit"] - 0.03) * 2
            scores[name] = m["auc_test"] - penalty

        champion = max(scores, key=scores.get)
        state.champion_model_name = champion
        state.champion_model      = state.trained_models[champion]

        self._info(f"Champion: {champion}  (adj.score={scores[champion]:.4f})")
        for n, s in scores.items():
            self._info(f"  {n}: adj={s:.4f}  "
                       f"AUC={state.model_metrics[n]['auc_test']:.4f}  "
                       f"overfit={state.model_metrics[n]['overfit']:.4f}")
        return state

    # ─────────────────────────────────────────────────────────────
    def _llm_rationale(self, state: PipelineState) -> PipelineState:
        metrics_text = "\n".join(
            f"  {n}: AUC={m['auc_test']:.4f}, KS={m['ks']:.4f}, "
            f"Gini={m['gini']:.4f}, Overfit={m['overfit']:.4f}"
            for n, m in state.model_metrics.items()
        )
        prompt = f"""
You are a senior credit risk model validator reviewing candidate models for
a Lending Club binary default prediction scorecard.

MODEL COMPARISON:
{metrics_text}

SELECTED CHAMPION: {state.champion_model_name}

Write a Model Selection Rationale (max 200 words) for the model development report.
Address:
1. Why the champion was selected (performance, stability, interpretability)
2. Trade-offs vs the other candidates
3. Any governance or regulatory considerations for this choice
4. Recommended next steps for validation
"""
        try:
            rationale = ask(prompt, system=CREDIT_RISK_SYSTEM, max_tokens=500)
            state.model_selection_rationale = rationale
            self._info("LLM model selection rationale generated")
        except Exception as e:
            state.log_warning(self.name, f"LLM rationale skipped: {e}")
        return state
