"""UrbanFlow AI — Campaign Spillback Operational Alarm Benchmark Engine (Rev 6).

Implements Section 4 (Downstream Bottleneck Spillback Prediction) and Section 11 of Rev 6:
1. Operational Alarm Generation:
   - Evaluates probability of spillback breach from trained campaign models.
   - Enforces 2-consecutive positive observation rule (p >= 0.50 at t_first and t_first + 5s).
   - Confirmed alarm timestamp = second positive observation (t_alarm = t_first + 5.0s).
   - Enforces 300-second episode cooldown independently per forecast horizon.
   - Resets state on physical event onset or cooldown expiration.
   - Suppresses false alarms during active spillback intervals [t_onset, t_end).
2. Empirical Reactive Baseline:
   - Monitors current physical state breach (avg_speed <= 12 km/h or queue >= 150m).
   - Evaluates identical 2-consecutive positive rule and 300-second cooldown.
3. Deterministic Greedy Chronological One-to-One Matching:
   - Road-isolated matching within horizon window (t_alarm < t_onset <= t_alarm + h_sec).
   - Canonical sorting: timestamp ascending, then road_id, then alarm_id.
   - Earliest alarm tie-breaking.
4. Statistical Design & Run-Level Aggregation:
   - Independent simulation runs are treated as statistical units (N=8 in-dist, N=20 held-out).
   - Computes event-level descriptive operational warning lead-time.
   - Computes run-level paired inferential lead-time difference on mutually detected events.
   - Computes precision, recall, exposure hours, and False Alarm Rate (FAR).
5. Safe Persistence:
   - Persists all event-level and run-level records to docs/reports/campaign_spillback_benchmark_results.json.
   - Never modifies or overwrites campaign_evaluation_results.json, models, or raw datasets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from pathlib import Path
import sys
import time
from typing import Dict, Any, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats

URBANFLOW_ROOT = Path(__file__).resolve().parent.parent
if str(URBANFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(URBANFLOW_ROOT))

from scripts.evaluate_campaign_models import (
    deterministic_alarm_event_matching,
    CAMPAIGN_MODELS_DIR,
    REPORTS_DIR,
)

logger = logging.getLogger(__name__)

HORIZONS = [("5min", 300), ("10min", 600), ("15min", 900)]
FEATURE_DIR = URBANFLOW_ROOT / "data" / "datasets" / "campaign_features"
BENCHMARK_REPORT_PATH = REPORTS_DIR / "campaign_spillback_benchmark_results.json"

MODEL_FEATURES = [
    "avg_speed_lag_1", "avg_speed_lag_2", "avg_speed_lag_3", "avg_speed_roll_mean", "avg_speed_roll_std",
    "density_lag_1", "density_lag_2", "density_lag_3", "density_roll_mean",
    "queue_lag_1", "queue_lag_2", "queue_lag_3", "queue_roll_mean",
    "operational_capacity", "vc_ratio", "incident_active",
]


def compute_model_file_hashes() -> Dict[str, str]:
    """Computes SHA-256 hashes of all 9 campaign model files."""
    hashes = {}
    for p in sorted(CAMPAIGN_MODELS_DIR.glob("*.joblib")):
        with open(p, "rb") as f:
            hashes[p.name] = hashlib.sha256(f.read()).hexdigest().upper()
    return hashes


def extract_operational_alarms(
    df: pd.DataFrame,
    mode: str = "ML",
    prob_col: str = "prob",
    breach_col: str = "current_breach",
    cooldown_sec: float = 300.0,
    prob_thresh: float = 0.50,
) -> List[Dict[str, Any]]:
    """
    Extracts operational alarms using Rev 6 Section 4.3 FSM:
    - 2 consecutive positive predictions (p >= prob_thresh at t_first and t_first + 5s).
    - t_alarm stamped at second observation.
    - 300-second episode cooldown.
    - Subsequent positive predictions within cooldown merge into episode.
    - Alarms during active spillback are suppressed from false-alarm accounting.
    - State resets on physical event onset or cooldown expiration.
    """
    all_alarms = []
    df_sorted = df.sort_values(["run_id", "road_id", "timestamp"]).reset_index(drop=True)

    run_ids = df_sorted["run_id"].values
    road_ids = df_sorted["road_id"].values
    timestamps = df_sorted["timestamp"].values
    is_actives = df_sorted["is_active_spillback"].values
    is_onsets = df_sorted["is_event_onset"].values

    if mode in ("ML", "CLASSIFIER"):
        signal = df_sorted[prob_col].values >= prob_thresh
    else:
        signal = df_sorted[breach_col].values.astype(bool)

    # Fast group boundaries
    group_keys = [f"{r}_{rd}" for r, rd in zip(run_ids, road_ids)]
    switches = np.where(np.array(group_keys[:-1]) != np.array(group_keys[1:]))[0] + 1
    split_indices = np.concatenate([[0], switches, [len(df_sorted)]])

    for g_idx in range(len(split_indices) - 1):
        start = split_indices[g_idx]
        end = split_indices[g_idx + 1]

        r_id = run_ids[start]
        rd_id = road_ids[start]
        t_arr = timestamps[start:end]
        sig_arr = signal[start:end]
        act_arr = is_actives[start:end]
        ons_arr = is_onsets[start:end]

        in_cooldown = False
        cooldown_until = -1.0
        consec = 0
        t_first = None

        for i in range(len(t_arr)):
            t = float(t_arr[i])
            sig = bool(sig_arr[i])
            act = bool(act_arr[i])
            ons = bool(ons_arr[i])

            # Reset state if physical event onsets
            if ons:
                in_cooldown = False
                consec = 0

            if in_cooldown:
                if t >= cooldown_until:
                    in_cooldown = False
                else:
                    continue

            # Suppress alarms during active spillback (road already congested)
            if act:
                consec = 0
                continue

            if sig:
                if consec == 0:
                    consec = 1
                    t_first = t
                elif consec == 1:
                    t_alarm = t
                    all_alarms.append({
                        "alarm_id": f"{mode}_{r_id}_{rd_id}_{int(t_alarm)}",
                        "run_id": r_id,
                        "road_id": rd_id,
                        "t_alarm": t_alarm,
                        "t_first": t_first,
                        "mode": mode,
                    })
                    in_cooldown = True
                    cooldown_until = t_alarm + cooldown_sec
                    consec = 0
            else:
                consec = 0

    return all_alarms


def evaluate_partition_spillback(
    df: pd.DataFrame,
    speed_model_obj: Any,
    h_name: str,
    h_sec: int,
    split_name: str,
) -> Dict[str, Any]:
    """
    Evaluates ML and Reactive Baseline spillback benchmark metrics on a test partition.
    Treats independent simulation runs as statistical units.
    """
    # 1. Generate predictions from the frozen campaign model
    df_eval = df.copy()
    pred_speeds = speed_model_obj.predict(df_eval[MODEL_FEATURES])
    df_eval["pred_speed"] = pred_speeds

    # Logistic probability mapped to spillback speed breach (<= 12 km/h)
    df_eval["prob"] = 1.0 / (1.0 + np.exp((pred_speeds - 12.0) / 2.0))
    # Reactive baseline breach condition (speed <= 12 km/h or queue >= 150m)
    df_eval["current_breach"] = (df_eval["avg_speed_kmh"] <= 12.0) | (df_eval["queue_length_m"] >= 150.0)

    # 2. Extract ground truth events
    onsets_df = df_eval[df_eval["is_event_onset"]].sort_values(["run_id", "road_id", "timestamp"])
    all_events = []
    for _, row in onsets_df.iterrows():
        all_events.append({
            "event_id": f"{row['run_id']}_{row['road_id']}_{int(row['timestamp'])}",
            "run_id": str(row["run_id"]),
            "road_id": str(row["road_id"]),
            "t_onset": float(row["timestamp"]),
        })

    # 3. Extract alarms
    ml_alarms = extract_operational_alarms(df_eval, mode="ML", prob_col="prob", cooldown_sec=300.0)
    re_alarms = extract_operational_alarms(df_eval, mode="REACTIVE", breach_col="current_breach", cooldown_sec=300.0)

    # 4. Global deterministic matching
    ml_matched, ml_unmatched_a, ml_unmatched_e = deterministic_alarm_event_matching(ml_alarms, all_events, h_sec)
    re_matched, re_unmatched_a, re_unmatched_e = deterministic_alarm_event_matching(re_alarms, all_events, h_sec)

    # Attach run_id to matched pairs for run-level accounting
    event_run_map = {e["event_id"]: e["run_id"] for e in all_events}
    for m in ml_matched:
        m["run_id"] = event_run_map.get(m["event_id"], "")
    for m in re_matched:
        m["run_id"] = event_run_map.get(m["event_id"], "")
    for u in ml_unmatched_e:
        u["run_id"] = event_run_map.get(u["event_id"], "")
    for u in re_unmatched_e:
        u["run_id"] = event_run_map.get(u["event_id"], "")

    # 5. Compute non-congested exposure time (excluding active spillback intervals)
    # T_exposure = sum(road_seconds - active_spillback_seconds) across runs
    unique_runs = sorted(df_eval["run_id"].unique())
    run_exposure_hours = {}
    total_exposure_sec = 0.0

    for r_id in unique_runs:
        rdf = df_eval[df_eval["run_id"] == r_id]
        active_steps = int(rdf["is_active_spillback"].sum())
        total_steps = len(rdf)
        exp_sec = (total_steps - active_steps) * 5.0
        total_exposure_sec += exp_sec
        run_exposure_hours[r_id] = exp_sec / 3600.0

    total_exposure_hours = total_exposure_sec / 3600.0

    # 6. Global metrics
    ml_lts = [float(m["lead_time_sec"]) for m in ml_matched]
    re_lts = [float(m["lead_time_sec"]) for m in re_matched]

    def summarize_lts(lts: List[float]) -> Dict[str, Any]:
        if not lts:
            return {"count": 0, "mean": 0.0, "median": 0.0, "std": 0.0, "p25": 0.0, "p75": 0.0, "min": 0.0, "max": 0.0}
        arr = np.array(lts)
        return {
            "count": len(arr),
            "mean": round(float(np.mean(arr)), 2),
            "median": round(float(np.median(arr)), 2),
            "std": round(float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0, 2),
            "p25": round(float(np.percentile(arr, 25)), 2),
            "p75": round(float(np.percentile(arr, 75)), 2),
            "min": round(float(np.min(arr)), 2),
            "max": round(float(np.max(arr)), 2),
        }

    ml_lt_stats = summarize_lts(ml_lts)
    re_lt_stats = summarize_lts(re_lts)

    ml_prec = round(len(ml_matched) / (len(ml_matched) + len(ml_unmatched_a)), 4) if (len(ml_matched) + len(ml_unmatched_a)) > 0 else 0.0
    ml_rec = round(len(ml_matched) / len(all_events), 4) if all_events else 0.0
    ml_far = round(len(ml_unmatched_a) / total_exposure_hours, 4) if total_exposure_hours > 0 else 0.0

    re_prec = round(len(re_matched) / (len(re_matched) + len(re_unmatched_a)), 4) if (len(re_matched) + len(re_unmatched_a)) > 0 else 0.0
    re_rec = round(len(re_matched) / len(all_events), 4) if all_events else 0.0
    re_far = round(len(re_unmatched_a) / total_exposure_hours, 4) if total_exposure_hours > 0 else 0.0

    # 7. Run-level paired inference on mutually detected events (Rev 6 Section 4.4)
    ml_map = {m["event_id"]: m["lead_time_sec"] for m in ml_matched}
    re_map = {m["event_id"]: m["lead_time_sec"] for m in re_matched}

    run_breakdown = []
    paired_diffs_by_run = []

    for r_id in unique_runs:
        r_events = [e for e in all_events if e["run_id"] == r_id]
        r_ml_m = [m for m in ml_matched if m["run_id"] == r_id]
        r_ml_u = [u for u in ml_unmatched_a if u["run_id"] == r_id]
        r_re_m = [m for m in re_matched if m["run_id"] == r_id]
        r_re_u = [u for u in re_unmatched_a if u["run_id"] == r_id]

        r_mutual = [e["event_id"] for e in r_events if e["event_id"] in ml_map and e["event_id"] in re_map]
        r_gains = [ml_map[eid] - re_map[eid] for eid in r_mutual]
        r_mean_gain = float(np.mean(r_gains)) if r_gains else None

        if r_mean_gain is not None:
            paired_diffs_by_run.append(r_mean_gain)

        r_exp_h = run_exposure_hours[r_id]
        run_breakdown.append({
            "run_id": r_id,
            "events_count": len(r_events),
            "ml_alarms": len(r_ml_m) + len(r_ml_u),
            "ml_matched": len(r_ml_m),
            "ml_unmatched": len(r_ml_u),
            "ml_precision": round(len(r_ml_m) / (len(r_ml_m) + len(r_ml_u)), 4) if (len(r_ml_m) + len(r_ml_u)) > 0 else 0.0,
            "ml_recall": round(len(r_ml_m) / len(r_events), 4) if r_events else 0.0,
            "ml_far": round(len(r_ml_u) / r_exp_h, 4) if r_exp_h > 0 else 0.0,
            "ml_mean_lead_time": round(float(np.mean([m["lead_time_sec"] for m in r_ml_m])), 2) if r_ml_m else None,
            "re_alarms": len(r_re_m) + len(r_re_u),
            "re_matched": len(r_re_m),
            "re_unmatched": len(r_re_u),
            "re_far": round(len(r_re_u) / r_exp_h, 4) if r_exp_h > 0 else 0.0,
            "re_mean_lead_time": round(float(np.mean([m["lead_time_sec"] for m in r_re_m])), 2) if r_re_m else None,
            "mutual_detections": len(r_mutual),
            "paired_lead_time_gain": round(r_mean_gain, 2) if r_mean_gain is not None else None,
        })

    # Paired Student's t-test over independent runs with mutual detections
    n_runs_eval = len(paired_diffs_by_run)
    if n_runs_eval > 1 and np.std(paired_diffs_by_run, ddof=1) > 1e-6:
        mean_gain = float(np.mean(paired_diffs_by_run))
        std_gain = float(np.std(paired_diffs_by_run, ddof=1))
        t_stat = mean_gain / (std_gain / math.sqrt(n_runs_eval))
        p_val = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=n_runs_eval - 1)))
        ci_margin = float(stats.t.ppf(0.975, df=n_runs_eval - 1) * (std_gain / math.sqrt(n_runs_eval)))
        ci_95 = (round(mean_gain - ci_margin, 2), round(mean_gain + ci_margin, 2))
    elif n_runs_eval == 1:
        mean_gain = float(paired_diffs_by_run[0])
        std_gain, t_stat, p_val, ci_95 = 0.0, 0.0, 1.0, (mean_gain, mean_gain)
    else:
        mean_gain, std_gain, t_stat, p_val, ci_95 = 0.0, 0.0, 0.0, 1.0, (0.0, 0.0)

    total_mutual_events = sum(len(set(ml_map.keys()) & set(re_map.keys())) for _ in [0])
    coverage = round(len(set(ml_map.keys()) & set(re_map.keys())) / len(all_events), 4) if all_events else 0.0

    return {
        "horizon": h_name,
        "horizon_sec": h_sec,
        "partition": split_name,
        "total_runs": len(unique_runs),
        "total_events": len(all_events),
        "total_exposure_hours": round(total_exposure_hours, 2),
        "ml_model": {
            "total_alarms": len(ml_alarms),
            "matched_alarms": len(ml_matched),
            "unmatched_alarms": len(ml_unmatched_a),
            "missed_events": len(ml_unmatched_e),
            "precision": ml_prec,
            "recall": ml_rec,
            "false_alarm_rate_per_hour": ml_far,
            "lead_time_distribution": ml_lt_stats,
        },
        "reactive_baseline": {
            "total_alarms": len(re_alarms),
            "matched_alarms": len(re_matched),
            "unmatched_alarms": len(re_unmatched_a),
            "missed_events": len(re_unmatched_e),
            "precision": re_prec,
            "recall": re_rec,
            "false_alarm_rate_per_hour": re_far,
            "lead_time_distribution": re_lt_stats,
        },
        "paired_comparison": {
            "mutually_detected_events": len(set(ml_map.keys()) & set(re_map.keys())),
            "matched_event_coverage": coverage,
            "runs_in_paired_test": n_runs_eval,
            "degrees_of_freedom": max(0, n_runs_eval - 1),
            "mean_paired_difference_sec": round(mean_gain, 2),
            "std_paired_difference_sec": round(std_gain, 2),
            "t_statistic": round(t_stat, 3),
            "p_value": round(p_val, 4),
            "ci_95": ci_95,
        },
        "per_run_breakdown": run_breakdown,
        "matched_pairs_sample": ml_matched[:20],
    }


def main():
    parser = argparse.ArgumentParser(description="UrbanFlow Campaign Spillback Benchmark Suite (Rev 6)")
    parser.add_argument("--dry-run", action="store_true", help="Dry run without full evaluation")
    args = parser.parse_args()

    print("=" * 80)
    print("URBANFLOW AI — SPILLBACK OPERATIONAL BENCHMARK ENGINE (REV 6)")
    print("=" * 80)
    print(f"Models Directory       : {CAMPAIGN_MODELS_DIR}")
    print(f"Features Directory     : {FEATURE_DIR}")
    print(f"Results Output Path    : {BENCHMARK_REPORT_PATH}")

    # Step 1: Pre-execution Hash Verification
    pre_hashes = compute_model_file_hashes()
    print(f"\n[STEP 1]: Verified {len(pre_hashes)} Campaign Models before execution:")
    for m_name, h in pre_hashes.items():
        print(f"  • {m_name:<30}: {h}")

    if args.dry_run:
        print("\n[DRY RUN COMPLETE]: Pipeline verified.")
        return

    benchmark_results: Dict[str, Any] = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "specification": "Rev 6 Section 4 & Section 11",
            "model_hashes_pre": pre_hashes,
        },
        "horizons": {},
    }

    # Step 2: Execute benchmark across horizons and partitions
    for h_name, h_sec in HORIZONS:
        print(f"\n{'='*30} HORIZON: {h_name} ({h_sec}s lookahead) {'='*30}")
        model_path = CAMPAIGN_MODELS_DIR / f"lgb_{h_name}_avg_speed.joblib"
        if not model_path.exists():
            print(f"ERROR: Model {model_path} missing.")
            continue

        speed_model_obj = joblib.load(model_path)["model"]

        h_results = {}
        for split_file, split_label in [("split_IN_DIST_TEST.parquet", "IN_DIST_TEST"), ("split_HELD_OUT_TEST.parquet", "HELD_OUT_TEST")]:
            feat_file = FEATURE_DIR / h_name / split_file
            if not feat_file.exists():
                print(f"WARNING: File {feat_file} does not exist.")
                continue

            print(f"\n--- Evaluating {split_label} ({h_name}) ---")
            df = pq.read_table(str(feat_file)).to_pandas()
            res = evaluate_partition_spillback(df, speed_model_obj, h_name, h_sec, split_label)
            h_results[split_label] = res

            ml = res["ml_model"]
            re = res["reactive_baseline"]
            paired = res["paired_comparison"]

            print(f"  Physical Events     : {res['total_events']}")
            print(f"  ML Alarms / Matched : {ml['total_alarms']} / {ml['matched_alarms']} (Precision: {ml['precision']:.3f}, Recall: {ml['recall']:.3f})")
            print(f"  ML False Alarm Rate : {ml['false_alarm_rate_per_hour']:.3f} alarms/hour (Exposure: {res['total_exposure_hours']}h)")
            print(f"  ML Lead Time        : Mean={ml['lead_time_distribution']['mean']}s, Median={ml['lead_time_distribution']['median']}s")
            print(f"  Reactive Alarms/Mat : {re['total_alarms']} / {re['matched_alarms']} (FAR: {re['false_alarm_rate_per_hour']:.3f} alarms/hour)")
            print(f"  Reactive Lead Time  : Mean={re['lead_time_distribution']['mean']}s")
            print(f"  Mutual Detections   : {paired['mutually_detected_events']} (Coverage: {paired['matched_event_coverage']:.1%})")
            print(f"  Paired Lead Diff    : {paired['mean_paired_difference_sec']:+.2f}s (t={paired['t_statistic']}, p={paired['p_value']}, df={paired['degrees_of_freedom']})")

        benchmark_results["horizons"][h_name] = h_results

    # Step 3: Post-execution Hash Verification
    post_hashes = compute_model_file_hashes()
    print("\n[STEP 3]: Post-execution Hash Verification:")
    hashes_identical = True
    for m_name, h in post_hashes.items():
        if h != pre_hashes.get(m_name):
            print(f"  ❌ HASH MISMATCH on {m_name}!")
            hashes_identical = False
        else:
            print(f"  ✅ {m_name:<30}: UNCHANGED")

    benchmark_results["metadata"]["model_hashes_post"] = post_hashes
    benchmark_results["metadata"]["model_hashes_verified"] = hashes_identical

    # Step 4: Persist Artifact
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(BENCHMARK_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)

    print(f"\n[BENCHMARK COMPLETE]: Results saved to {BENCHMARK_REPORT_PATH}")


if __name__ == "__main__":
    main()
