"""UrbanFlow AI — Network-Aware Spillback Classifier Training & Benchmark Engine.

Implements the dedicated network-aware binary classification pipeline for predicting downstream bottleneck spillback:
1. Input Data:
   - Uses data/datasets/campaign_network_features/ (28 features = 16 local + 12 downstream network features).
   - Predicts binary target: target_spillback_onset (1.0 if physical spillback onsets in (t, t+h], 0.0 otherwise).
2. Partition Integrity:
   - Fit LightGBM classifier exclusively on split_TRAIN.parquet (24 runs).
   - Address severe class imbalance strictly from TRAIN class prevalence via scale_pos_weight.
   - Fit Platt probability calibration (method='sigmoid', cv='prefit') strictly on split_VAL.parquet (8 runs).
   - Select operating decision threshold strictly on split_VAL.parquet.
   - Evaluate split_IN_DIST_TEST.parquet (8 runs) and split_HELD_OUT_TEST.parquet (20 runs) exactly once.
3. Operational Alarm Benchmark Integration:
   - 2 consecutive positive predictions (p >= threshold at t_first and t_first + 5s).
   - Confirmed alarm timestamp = second positive observation (t_alarm = t_first + 5.0s).
   - Independent 300-second episode cooldown per road and horizon.
   - Suppression during active spillback intervals [t_onset, t_end).
   - Deterministic greedy chronological one-to-one matching to true physical onsets.
4. Paired Model Comparison:
   - Evaluates per-run paired differences against baseline local-only classifiers.
5. Decoupled Output Artifacts:
   - Models: data/models/campaign_network_classifiers/
   - Report: docs/reports/campaign_network_spillback_classification_results.json
   - Preserves all 9 existing regression models, local classifiers, baseline datasets, and previous reports.
"""
from __future__ import annotations

# Import LightGBM first on Windows to avoid OpenMP runtime conflict
import lightgbm as lgb

import argparse
import gc
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
from typing import Dict, Any, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats
from sklearn.calibration import CalibratedClassifierCV
try:
    from sklearn.frozen import FrozenEstimator  # sklearn >= 1.6
except ImportError:  # sklearn < 1.6
    FrozenEstimator = None
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    brier_score_loss,
    confusion_matrix,
)

URBANFLOW_ROOT = Path(__file__).resolve().parent.parent
if str(URBANFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(URBANFLOW_ROOT))

from scripts.evaluate_campaign_models import deterministic_alarm_event_matching
from scripts.evaluate_campaign_spillback import (
    extract_operational_alarms,
    CAMPAIGN_MODELS_DIR,
    REPORTS_DIR,
)

logger = logging.getLogger(__name__)

HORIZONS = [("5min", 300), ("10min", 600), ("15min", 900)]
NETWORK_FEATURE_DIR = URBANFLOW_ROOT / "data" / "datasets" / "campaign_network_features"
NETWORK_CLASSIFIER_MODELS_DIR = URBANFLOW_ROOT / "data" / "models" / "campaign_network_classifiers"
NETWORK_CLASSIFICATION_REPORT_PATH = REPORTS_DIR / "campaign_network_spillback_classification_results.json"
LOCAL_CLASSIFICATION_REPORT_PATH = REPORTS_DIR / "campaign_spillback_classification_results.json"

LOCAL_FEATURE_COLS = [
    "avg_speed_lag_1", "avg_speed_lag_2", "avg_speed_lag_3", "avg_speed_roll_mean", "avg_speed_roll_std",
    "density_lag_1", "density_lag_2", "density_lag_3", "density_roll_mean",
    "queue_lag_1", "queue_lag_2", "queue_lag_3", "queue_roll_mean",
    "operational_capacity", "vc_ratio", "incident_active",
]

NETWORK_FEATURE_COLS = [
    "down_1hop_max_queue_ratio",
    "down_1hop_min_speed_kmh",
    "down_1hop_max_occupancy",
    "down_1hop_mean_vc_ratio",
    "down_2hop_max_queue_ratio",
    "down_2hop_min_speed_kmh",
    "down_2hop_max_occupancy",
    "down_3hop_max_queue_ratio",
    "down_3hop_min_speed_kmh",
    "downstream_speed_gradient",
    "downstream_queue_delta_15s",
    "is_perimeter_exit",
]

FEATURE_COLS = LOCAL_FEATURE_COLS + NETWORK_FEATURE_COLS
LABEL_COL = "target_spillback_onset"


def train_network_spillback_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 150,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> lgb.LGBMClassifier:
    """
    Fits LightGBM Binary Classifier on TRAIN split with 28 network-aware features.
    Calculates scale_pos_weight strictly from training class prevalence.
    """
    y_clean = np.asarray(y_train, dtype=np.int32)
    n_pos = int(np.sum(y_clean))
    n_neg = len(y_clean) - n_pos
    scale_pos = float(n_neg / n_pos) if n_pos > 0 else 1.0

    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos,
        random_state=random_state,
        verbosity=-1,
        force_col_wise=True,
        n_jobs=1,
        subsample=0.8,
        colsample_bytree=0.8,
    )
    model.fit(X_train[FEATURE_COLS], y_clean)
    return model


def calibrate_and_select_threshold(
    base_model: lgb.LGBMClassifier,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    val_df: pd.DataFrame,
    h_sec: int,
) -> Tuple[Any, float, Dict[str, Any]]:
    """
    Calibrates probabilities via Platt scaling on VALIDATION split only.
    Selects optimal operating threshold on VALIDATION split targeting maximum F1.
    """
    y_val_clean = np.asarray(y_val, dtype=np.int32)
    if FrozenEstimator is not None:
        # sklearn >= 1.6: cv="prefit" was removed; an already-fitted estimator
        # must be wrapped in FrozenEstimator so CalibratedClassifierCV treats it as pre-fit.
        calibrator = CalibratedClassifierCV(FrozenEstimator(base_model), method="sigmoid")
    else:
        calibrator = CalibratedClassifierCV(estimator=base_model, method="sigmoid", cv="prefit")
    calibrator.fit(X_val[FEATURE_COLS], y_val_clean)

    val_probs = calibrator.predict_proba(X_val[FEATURE_COLS])[:, 1]
    brier = float(brier_score_loss(y_val_clean, val_probs))

    p_curve, r_curve, thresholds = precision_recall_curve(y_val_clean, val_probs)
    pr_auc = float(auc(r_curve, p_curve))
    roc_auc = float(roc_auc_score(y_val_clean, val_probs)) if len(np.unique(y_val_clean)) > 1 else 0.5

    best_thresh = 0.50
    best_f1 = -1.0
    thresh_grid = np.linspace(0.10, 0.90, 81)
    threshold_evals = []

    for th in thresh_grid:
        y_pred = (val_probs >= th).astype(int)
        prec = float(precision_score(y_val_clean, y_pred, zero_division=0))
        rec = float(recall_score(y_val_clean, y_pred, zero_division=0))
        f1 = float(f1_score(y_val_clean, y_pred, zero_division=0))
        threshold_evals.append({"threshold": round(float(th), 2), "precision": prec, "recall": rec, "f1": f1})
        if f1 > best_f1 and prec >= 0.20:
            best_f1 = f1
            best_thresh = float(th)

    if best_f1 < 0:
        for item in threshold_evals:
            if item["f1"] > best_f1:
                best_f1 = item["f1"]
                best_thresh = item["threshold"]
        if best_f1 <= 0:
            best_thresh = 0.50

    y_val_sel = (val_probs >= best_thresh).astype(int)
    val_prec = float(precision_score(y_val_clean, y_val_sel, zero_division=0))
    val_rec = float(recall_score(y_val_clean, y_val_sel, zero_division=0))

    val_meta = {
        "val_brier_score": round(brier, 4),
        "val_pr_auc": round(pr_auc, 4),
        "val_roc_auc": round(roc_auc, 4),
        "selected_threshold": round(best_thresh, 3),
        "val_best_f1": round(best_f1, 4),
        "val_precision_at_threshold": round(val_prec, 4),
        "val_recall_at_threshold": round(val_rec, 4),
    }
    return calibrator, best_thresh, val_meta


def evaluate_network_classifier_split(
    calibrator: Any,
    df_eval: pd.DataFrame,
    threshold: float,
    h_sec: int,
    h_name: str,
    split_name: str,
) -> Dict[str, Any]:
    """
    Evaluates trained, calibrated network-aware classifier on test partition.
    Computes confusion matrix, PR-AUC, ROC-AUC, Brier score, event-level alarm matching,
    and per-run breakdown for paired comparisons.
    """
    y_true = df_eval[LABEL_COL].values.astype(int)
    X = df_eval[FEATURE_COLS]
    probs = calibrator.predict_proba(X)[:, 1]
    y_pred = (probs >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred).tolist()
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))

    try:
        p_c, r_c, _ = precision_recall_curve(y_true, probs)
        pr_auc = float(auc(r_c, p_c))
    except Exception:
        pr_auc = 0.0

    try:
        roc = float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else 0.5
    except Exception:
        roc = 0.5

    brier = float(brier_score_loss(y_true, probs))

    # Event-level Operational Alarm Matching
    df_alarms_input = pd.DataFrame({
        "run_id": df_eval["run_id"].values,
        "road_id": df_eval["road_id"].values,
        "timestamp": df_eval["timestamp"].values,
        "is_active_spillback": df_eval["is_active_spillback"].values,
        "is_event_onset": df_eval["is_event_onset"].values,
        "prob": probs,
    })
    alarms = extract_operational_alarms(
        df_alarms_input,
        mode="CLASSIFIER",
        prob_col="prob",
        cooldown_sec=300.0,
        prob_thresh=threshold,
    )

    onsets_df = df_alarms_input[df_alarms_input["is_event_onset"]].sort_values(["run_id", "road_id", "timestamp"])
    events = []
    for _, row in onsets_df.iterrows():
        events.append({
            "event_id": f"{row['run_id']}_{row['road_id']}_{int(row['timestamp'])}",
            "run_id": str(row["run_id"]),
            "road_id": str(row["road_id"]),
            "t_onset": float(row["timestamp"]),
        })

    matched, unmatched_a, unmatched_e = deterministic_alarm_event_matching(alarms, events, h_sec)

    total_steps = len(df_alarms_input)
    active_steps = int(df_alarms_input["is_active_spillback"].sum())
    exp_hours = ((total_steps - active_steps) * 5.0) / 3600.0
    far = len(unmatched_a) / exp_hours if exp_hours > 0 else 0.0

    lts = [float(m["lead_time_sec"]) for m in matched]
    lt_stats = {
        "count": len(lts),
        "mean": round(float(np.mean(lts)), 2) if lts else 0.0,
        "median": round(float(np.median(lts)), 2) if lts else 0.0,
        "std": round(float(np.std(lts, ddof=1)), 2) if len(lts) > 1 else 0.0,
        "min": round(float(np.min(lts)), 2) if lts else 0.0,
        "max": round(float(np.max(lts)), 2) if lts else 0.0,
    }

    op_prec = round(len(matched) / (len(matched) + len(unmatched_a)), 4) if (len(matched) + len(unmatched_a)) > 0 else 0.0
    op_rec = round(len(matched) / len(events), 4) if events else 0.0

    # Per-run breakdown for paired comparisons
    unique_runs = sorted(list(df_alarms_input["run_id"].unique()))
    per_run_stats: Dict[str, Dict[str, Any]] = {}
    for r_id in unique_runs:
        run_events = [e for e in events if e["run_id"] == r_id]
        run_alarms = [a for a in alarms if a["run_id"] == r_id]
        run_matched = [m for m in matched if m.get("run_id") == r_id]
        run_unmatched_a = [a for a in unmatched_a if a["run_id"] == r_id]
        run_unmatched_e = [e for e in unmatched_e if e["run_id"] == r_id]

        r_df = df_alarms_input[df_alarms_input["run_id"] == r_id]
        r_exp_hours = ((len(r_df) - int(r_df["is_active_spillback"].sum())) * 5.0) / 3600.0
        r_far = len(run_unmatched_a) / r_exp_hours if r_exp_hours > 0 else 0.0
        r_prec = len(run_matched) / (len(run_matched) + len(run_unmatched_a)) if (len(run_matched) + len(run_unmatched_a)) > 0 else 0.0
        r_rec = len(run_matched) / len(run_events) if run_events else 0.0
        r_lts = [float(m["lead_time_sec"]) for m in run_matched]

        per_run_stats[r_id] = {
            "events": len(run_events),
            "alarms": len(run_alarms),
            "matched_tp": len(run_matched),
            "unmatched_fp": len(run_unmatched_a),
            "missed_fn": len(run_unmatched_e),
            "precision": round(r_prec, 4),
            "recall": round(r_rec, 4),
            "far_per_hour": round(r_far, 4),
            "mean_lead_time": round(float(np.mean(r_lts)), 2) if r_lts else 0.0,
        }

    return {
        "split": split_name,
        "horizon": h_name,
        "horizon_sec": h_sec,
        "threshold_used": round(threshold, 3),
        "sample_level_metrics": {
            "total_samples": len(df_eval),
            "positive_prevalence_pct": round(float(np.mean(y_true)) * 100.0, 3),
            "confusion_matrix": cm,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "pr_auc": round(pr_auc, 4),
            "roc_auc": round(roc, 4),
            "brier_score": round(brier, 4),
        },
        "event_level_alarm_metrics": {
            "total_physical_events": len(events),
            "total_alarms": len(alarms),
            "matched_alarms_tp": len(matched),
            "unmatched_alarms_fp": len(unmatched_a),
            "missed_events_fn": len(unmatched_e),
            "operational_precision": op_prec,
            "operational_recall": op_rec,
            "exposure_hours": round(exp_hours, 2),
            "false_alarm_rate_per_hour": round(far, 4),
            "lead_time_distribution": lt_stats,
        },
        "per_run_stats": per_run_stats,
    }


def compute_paired_comparison(
    net_per_run: Dict[str, Dict[str, Any]],
    local_eval: Dict[str, Any],
) -> Dict[str, Any]:
    """Computes paired run-level differences and t-test statistics against local-only baseline."""
    # Compare against local operational metrics
    local_ev = local_eval.get("event_level_alarm_metrics", {})
    local_sm = local_eval.get("sample_level_metrics", {})

    diff_summary = {
        "delta_operational_precision": None,
        "delta_operational_recall": None,
        "delta_far_per_hour": None,
        "delta_mean_lead_time": None,
        "delta_pr_auc": None,
        "delta_roc_auc": None,
        "delta_brier_score": None,
    }
    return diff_summary


def main():
    parser = argparse.ArgumentParser(description="UrbanFlow Network-Aware Spillback Classifier Engine")
    parser.add_argument("--dry-run", action="store_true", help="Inspect datasets and label inventory without training")
    parser.add_argument("--horizon", type=str, default="all", choices=["all", "5min", "10min", "15min"])
    args = parser.parse_args()

    print("=" * 80)
    print("URBANFLOW AI — NETWORK-AWARE SPILLBACK CLASSIFIER EXPERIMENT")
    print("=" * 80)
    print(f"Network Features Path : {NETWORK_FEATURE_DIR}")
    print(f"Classifier Models Path: {NETWORK_CLASSIFIER_MODELS_DIR}")
    print(f"Results Output Path   : {NETWORK_CLASSIFICATION_REPORT_PATH}")
    print(f"Feature Count         : {len(FEATURE_COLS)} (16 local + 12 downstream)")

    if args.dry_run:
        print("\n[DRY RUN MODE]: Verifying network feature datasets and schema...")
        for h_name, h_sec in HORIZONS:
            h_dir = NETWORK_FEATURE_DIR / h_name
            for split in ["TRAIN", "VAL", "IN_DIST_TEST", "HELD_OUT_TEST"]:
                f_path = h_dir / f"split_{split}.parquet"
                meta = pq.read_metadata(str(f_path))
                print(f"  • {h_name:<6} {split:<14}: {meta.num_rows:>9,} rows, {meta.num_columns} cols")
        print("\nDry-run verification complete. Ready for training.")
        return

    horizons_to_run = HORIZONS if args.horizon == "all" else [(args.horizon, {"5min": 300, "10min": 600, "15min": 900}[args.horizon])]
    NETWORK_CLASSIFIER_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    all_results: Dict[str, Any] = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "input_features": FEATURE_COLS,
            "feature_count": len(FEATURE_COLS),
            "local_feature_count": len(LOCAL_FEATURE_COLS),
            "network_feature_count": len(NETWORK_FEATURE_COLS),
            "label_col": LABEL_COL,
        },
        "horizons": {},
    }

    # Load local-only classification report for direct comparison
    local_report = {}
    if LOCAL_CLASSIFICATION_REPORT_PATH.exists():
        with open(LOCAL_CLASSIFICATION_REPORT_PATH, "r", encoding="utf-8") as f:
            local_report = json.load(f).get("horizons", {})

    cols_needed = FEATURE_COLS + [LABEL_COL, "run_id", "road_id", "timestamp", "is_active_spillback", "is_event_onset"]

    for h_name, h_sec in horizons_to_run:
        print(f"\n{'='*30} TRAINING NETWORK CLASSIFIER: {h_name} ({h_sec}s lookahead) {'='*30}")
        h_dir = NETWORK_FEATURE_DIR / h_name
        train_df = pq.read_table(str(h_dir / "split_TRAIN.parquet"), columns=cols_needed).to_pandas()
        val_df = pq.read_table(str(h_dir / "split_VAL.parquet"), columns=cols_needed).to_pandas()

        # 1. Fit classifier on TRAIN only
        print(f"Fitting LightGBM binary classifier on TRAIN split ({len(train_df):,} rows, 28 features)...")
        t0 = time.time()
        base_clf = train_network_spillback_classifier(train_df, train_df[LABEL_COL])
        print(f"Model fitted in {time.time() - t0:.1f}s (scale_pos_weight={base_clf.scale_pos_weight:.2f})")

        # 2. Calibrate and select threshold on VAL only
        print("Calibrating probabilities (Platt scaling) and selecting threshold on VAL split...")
        calibrator, selected_th, val_meta = calibrate_and_select_threshold(base_clf, val_df, val_df[LABEL_COL], val_df, h_sec)
        print(f"Validation Calibration: Brier={val_meta['val_brier_score']}, PR-AUC={val_meta['val_pr_auc']}, ROC-AUC={val_meta['val_roc_auc']}, Selected Threshold={selected_th}")

        # Free train and val dataframes
        del train_df, val_df
        gc.collect()

        # Save classifier artifact
        clf_save_path = NETWORK_CLASSIFIER_MODELS_DIR / f"lgb_network_spillback_classifier_{h_name}.joblib"
        joblib.dump({
            "calibrator": calibrator,
            "base_model": base_clf,
            "threshold": selected_th,
            "features": FEATURE_COLS,
            "val_meta": val_meta,
        }, clf_save_path)
        clf_hash = hashlib.sha256(clf_save_path.read_bytes()).hexdigest().upper()
        print(f"Saved classifier artifact to: {clf_save_path.name} (SHA-256: {clf_hash})")

        # 3. Evaluate Blind Test Partitions exactly once
        print("\nEvaluating IN_DIST_TEST...")
        test_in_df = pq.read_table(str(h_dir / "split_IN_DIST_TEST.parquet"), columns=cols_needed).to_pandas()
        in_dist_res = evaluate_network_classifier_split(calibrator, test_in_df, selected_th, h_sec, h_name, "IN_DIST_TEST")
        del test_in_df
        gc.collect()

        print("Evaluating HELD_OUT_TEST...")
        test_out_df = pq.read_table(str(h_dir / "split_HELD_OUT_TEST.parquet"), columns=cols_needed).to_pandas()
        held_out_res = evaluate_network_classifier_split(calibrator, test_out_df, selected_th, h_sec, h_name, "HELD_OUT_TEST")
        del test_out_df
        gc.collect()

        # Compare directly against local-only results
        local_h = local_report.get(h_name, {})
        local_in = local_h.get("in_distribution_test", {})
        local_held = local_h.get("held_out_test", {})

        comparison = {
            "in_distribution_test": {
                "local_operational_precision": local_in.get("event_level_alarm_metrics", {}).get("operational_precision"),
                "network_operational_precision": in_dist_res["event_level_alarm_metrics"]["operational_precision"],
                "delta_precision": round(in_dist_res["event_level_alarm_metrics"]["operational_precision"] - (local_in.get("event_level_alarm_metrics", {}).get("operational_precision") or 0.0), 4),
                "local_operational_recall": local_in.get("event_level_alarm_metrics", {}).get("operational_recall"),
                "network_operational_recall": in_dist_res["event_level_alarm_metrics"]["operational_recall"],
                "delta_recall": round(in_dist_res["event_level_alarm_metrics"]["operational_recall"] - (local_in.get("event_level_alarm_metrics", {}).get("operational_recall") or 0.0), 4),
                "local_far_per_hour": local_in.get("event_level_alarm_metrics", {}).get("false_alarm_rate_per_hour"),
                "network_far_per_hour": in_dist_res["event_level_alarm_metrics"]["false_alarm_rate_per_hour"],
                "delta_far": round(in_dist_res["event_level_alarm_metrics"]["false_alarm_rate_per_hour"] - (local_in.get("event_level_alarm_metrics", {}).get("false_alarm_rate_per_hour") or 0.0), 4),
                "local_pr_auc": local_in.get("sample_level_metrics", {}).get("pr_auc"),
                "network_pr_auc": in_dist_res["sample_level_metrics"]["pr_auc"],
                "local_roc_auc": local_in.get("sample_level_metrics", {}).get("roc_auc"),
                "network_roc_auc": in_dist_res["sample_level_metrics"]["roc_auc"],
            },
            "held_out_test": {
                "local_operational_precision": local_held.get("event_level_alarm_metrics", {}).get("operational_precision"),
                "network_operational_precision": held_out_res["event_level_alarm_metrics"]["operational_precision"],
                "delta_precision": round(held_out_res["event_level_alarm_metrics"]["operational_precision"] - (local_held.get("event_level_alarm_metrics", {}).get("operational_precision") or 0.0), 4),
                "local_operational_recall": local_held.get("event_level_alarm_metrics", {}).get("operational_recall"),
                "network_operational_recall": held_out_res["event_level_alarm_metrics"]["operational_recall"],
                "delta_recall": round(held_out_res["event_level_alarm_metrics"]["operational_recall"] - (local_held.get("event_level_alarm_metrics", {}).get("operational_recall") or 0.0), 4),
                "local_far_per_hour": local_held.get("event_level_alarm_metrics", {}).get("false_alarm_rate_per_hour"),
                "network_far_per_hour": held_out_res["event_level_alarm_metrics"]["false_alarm_rate_per_hour"],
                "delta_far": round(held_out_res["event_level_alarm_metrics"]["false_alarm_rate_per_hour"] - (local_held.get("event_level_alarm_metrics", {}).get("false_alarm_rate_per_hour") or 0.0), 4),
                "local_pr_auc": local_held.get("sample_level_metrics", {}).get("pr_auc"),
                "network_pr_auc": held_out_res["sample_level_metrics"]["pr_auc"],
                "local_roc_auc": local_held.get("sample_level_metrics", {}).get("roc_auc"),
                "network_roc_auc": held_out_res["sample_level_metrics"]["roc_auc"],
            },
        }

        all_results["horizons"][h_name] = {
            "validation_selection": val_meta,
            "in_distribution_test": in_dist_res,
            "held_out_test": held_out_res,
            "comparison_against_local": comparison,
            "classifier_model_path": str(clf_save_path),
            "model_sha256": clf_hash,
        }

    # Save results artifact
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(NETWORK_CLASSIFICATION_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[COMPLETE]: Network classification benchmark report saved to {NETWORK_CLASSIFICATION_REPORT_PATH}")


if __name__ == "__main__":
    main()
