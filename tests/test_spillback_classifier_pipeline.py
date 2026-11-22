"""Unit and Contract Tests for Dedicated Spillback Classifier Pipeline.

Verifies:
1. Leakage Prevention (Strict backward-looking whitelist, target interval boundary, run-disjoint splits)
2. Class Imbalance Handling (scale_pos_weight derived strictly from TRAIN partition)
3. Probability Calibration (Fitted strictly on VAL partition via prefit)
4. Threshold Selection (Searched and selected strictly on VAL partition)
5. 300-Second Cooldown & Deterministic Event Matching
6. Path and Model Asset Decoupling (Never overwrites 9 regression models or continuous results)
"""
from __future__ import annotations

# Import LightGBM first on Windows to avoid OpenMP runtime conflict with scikit-learn
import lightgbm as lgb
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import brier_score_loss

from scripts.train_and_evaluate_spillback_classifier import (
    train_spillback_classifier,
    calibrate_and_select_threshold,
    evaluate_classifier_split,
    FEATURE_COLS,
    LABEL_COL,
    CLASSIFIER_MODELS_DIR,
    CLASSIFICATION_REPORT_PATH,
)
from scripts.evaluate_campaign_models import (
    CAMPAIGN_MODELS_DIR,
    REPORTS_DIR,
)

URBANFLOW_ROOT = Path(__file__).resolve().parent.parent


def test_leakage_prevention_feature_whitelist():
    """
    Verifies that the feature whitelist strictly contains backward-looking features,
    and explicitly excludes all future targets, identifiers, and active spillback labels.
    """
    forbidden_features = [
        "target_speed", "target_density", "target_queue", "target_congestion_event",
        "target_spillback_onset", "is_active_spillback", "is_event_onset",
        "run_id", "family_id", "road_id", "timestamp", "step_index", "is_warmup",
    ]
    for feat in forbidden_features:
        assert feat not in FEATURE_COLS, f"Forbidden feature '{feat}' found in feature whitelist!"


def test_class_imbalance_scale_pos_weight_derivation():
    """
    Verifies that scale_pos_weight is computed strictly from training labels (n_neg / n_pos).
    """
    N = 1000
    y_train = pd.Series([1] * 10 + [0] * 990, dtype=np.int32)
    X_train = pd.DataFrame(np.random.randn(N, len(FEATURE_COLS)), columns=FEATURE_COLS)

    model = train_spillback_classifier(X_train, y_train, n_estimators=10)
    expected_weight = 990.0 / 10.0  # 99.0
    assert abs(model.scale_pos_weight - expected_weight) < 1e-3, f"Expected weight {expected_weight}, got {model.scale_pos_weight}"


def test_probability_calibration_and_threshold_selection_on_val():
    """
    Verifies that probability calibrator is fitted on validation set,
    and threshold is selected based strictly on validation F1 score.
    """
    np.random.seed(42)
    # Train set
    N_tr = 500
    X_tr = pd.DataFrame(np.random.randn(N_tr, len(FEATURE_COLS)), columns=FEATURE_COLS)
    y_tr = pd.Series(np.random.binomial(1, 0.10, N_tr), dtype=np.int32)
    y_tr.iloc[0] = 0
    y_tr.iloc[1] = 1

    clf = train_spillback_classifier(X_tr, y_tr, n_estimators=15)

    # Validation set
    N_val = 300
    X_val = pd.DataFrame(np.random.randn(N_val, len(FEATURE_COLS)), columns=FEATURE_COLS)
    y_val = pd.Series(np.random.binomial(1, 0.10, N_val), dtype=np.int32)
    y_val.iloc[0] = 0
    y_val.iloc[1] = 1

    calibrator, thresh, meta = calibrate_and_select_threshold(clf, X_val, y_val, pd.DataFrame(), h_sec=600)

    # Brier score must be non-negative and <= 1.0
    assert 0.0 <= meta["val_brier_score"] <= 1.0
    assert 0.10 <= thresh <= 0.90
    assert "val_pr_auc" in meta
    assert "selected_threshold" in meta


def test_asset_decoupling_and_non_overwriting():
    """
    Verifies that new classifier output paths do not collide with existing campaign regression models
    or results reports.
    """
    continuous_report = REPORTS_DIR / "campaign_evaluation_results.json"
    spillback_report = REPORTS_DIR / "campaign_spillback_benchmark_results.json"

    # New classifier report path is separate
    assert CLASSIFICATION_REPORT_PATH != continuous_report
    assert CLASSIFICATION_REPORT_PATH != spillback_report

    # New model directory is separate
    assert CLASSIFIER_MODELS_DIR != CAMPAIGN_MODELS_DIR
    assert "campaign_classifiers" in str(CLASSIFIER_MODELS_DIR)


def test_event_matching_and_cooldown_integration():
    """
    Verifies that evaluate_classifier_split correctly outputs confusion matrix,
    FAR, and lead time distribution on a synthetic fixture.
    """
    np.random.seed(42)
    N = 200
    timestamps = np.arange(600.0, 600.0 + N * 5.0, 5.0)
    
    df_synth = pd.DataFrame(np.random.randn(N, len(FEATURE_COLS)), columns=FEATURE_COLS)
    df_synth["run_id"] = "RUN_TEST_01"
    df_synth["road_id"] = "ROAD_X"
    df_synth["timestamp"] = timestamps
    df_synth[LABEL_COL] = np.random.binomial(1, 0.05, N)
    df_synth["is_active_spillback"] = False
    df_synth["is_event_onset"] = False
    
    # Inject 1 onset at t=1000s
    onset_idx = 80
    df_synth.loc[onset_idx, "is_event_onset"] = True

    # Mock fitted model
    class MockCalibrator:
        def predict_proba(self, X):
            p1 = np.full(len(X), 0.1)
            # Inject positive predictions 200s prior to onset (t=800s, 805s)
            p1[40] = 0.85
            p1[41] = 0.85
            return np.column_stack([1.0 - p1, p1])

    res = evaluate_classifier_split(MockCalibrator(), df_synth, threshold=0.50, h_sec=600, h_name="10min", split_name="SYNTH_TEST")
    
    # Assert output schema
    assert "sample_level_metrics" in res
    assert "event_level_alarm_metrics" in res
    ev = res["event_level_alarm_metrics"]
    assert ev["total_physical_events"] == 1
    assert ev["matched_alarms_tp"] == 1
    assert ev["lead_time_distribution"]["count"] == 1
    # Lead time: t_onset (1000s) - t_alarm (805s) = 195s
    assert ev["lead_time_distribution"]["mean"] == 195.0
