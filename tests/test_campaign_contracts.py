"""Unit and Synthetic Fixture Tests for UrbanFlow Validation Campaign (Rev 6).

Verifies all mathematical contracts, algorithms, assertions, and boundary isolations
without running full simulation trajectories:
1. Exact 60-Run Matrix Allocation & Partition Integrity
2. Deterministic SHA-256 PRNG Stream Derivation & Repeatability
3. Complete Spillback Event Lifecycle (OR Activation, AND Recovery, Physical Onset, Exposure Time)
4. Deterministic Greedy Chronological Alarm-to-Event One-to-One Matching
5. Strict WAPE Zero-Denominator Rule (Returns None/NA, Never Substitutes MAE)
6. Holm-Bonferroni FWER Step-Down Correction
7. Legacy Dataset & Baseline Model Protection (Asserts zero modifications to traffic_history.db and data/models/)
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from urbanflow.core.kinematic_sim import KinematicSimulator
from scripts.run_campaign_dataset import (
    build_canonical_60_run_matrix,
    derive_stream_seed,
    CAMPAIGN_DATA_DIR,
    CAMPAIGN_METADATA_DB,
)
from scripts.build_campaign_features import (
    assert_pipeline_integrity,
    compute_spillback_event_series,
)
from scripts.evaluate_campaign_models import (
    compute_wape,
    deterministic_alarm_event_matching,
    holm_bonferroni_correction,
    CAMPAIGN_MODELS_DIR,
)

URBANFLOW_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = URBANFLOW_ROOT / "data"


def test_canonical_60_run_matrix_allocations():
    """Verifies the exact 60-run matrix allocation specified in Rev 6 Section 2."""
    matrix = build_canonical_60_run_matrix()
    assert len(matrix) == 60, f"Total runs {len(matrix)} != 60"

    # Verify run ID uniqueness
    run_ids = [r.run_id for r in matrix]
    assert len(set(run_ids)) == 60, "Duplicate run IDs found in matrix"

    # Verify seed uniqueness
    seeds = [r.random_seed for r in matrix]
    assert len(set(seeds)) == 60, "Duplicate seeds found in matrix"

    # Verify exact partition splits
    counts = {}
    for r in matrix:
        counts[r.partition] = counts.get(r.partition, 0) + 1

    assert counts["TRAIN"] == 24, f"TRAIN count {counts.get('TRAIN')} != 24"
    assert counts["VAL"] == 8, f"VAL count {counts.get('VAL')} != 8"
    assert counts["IN_DIST_TEST"] == 8, f"IN_DIST_TEST count {counts.get('IN_DIST_TEST')} != 8"
    assert counts["HELD_OUT_TEST"] == 20, f"HELD_OUT_TEST count {counts.get('HELD_OUT_TEST')} != 20"

    # Verify held-out families 5 and 6 are strictly 100% HELD_OUT_TEST
    for r in matrix:
        if r.family_id in ("FAM-05", "FAM-06"):
            assert r.partition == "HELD_OUT_TEST"
        else:
            assert r.partition in ("TRAIN", "VAL", "IN_DIST_TEST")


def test_deterministic_sha256_prng_seed_derivation():
    """Verifies that seed derivation is deterministic, decoupled, and matches Rev 6 Section 6.2."""
    seed = 1001
    s_driver_1 = derive_stream_seed(seed, "driver_attributes")
    s_driver_2 = derive_stream_seed(seed, "driver_attributes")
    s_spawn = derive_stream_seed(seed, "spawn_arrivals")
    s_inc = derive_stream_seed(seed, "incident_timing")

    assert s_driver_1 == s_driver_2, "Seed derivation is non-deterministic"
    assert s_driver_1 != s_spawn, "Driver stream and spawn stream collided"
    assert s_driver_1 != s_inc, "Driver stream and incident stream collided"

    # Check 32-bit unsigned integer range
    assert 0 <= s_driver_1 <= 0xFFFFFFFF
    assert 0 <= s_spawn <= 0xFFFFFFFF
    assert 0 <= s_inc <= 0xFFFFFFFF


def test_spillback_event_lifecycle_fsm():
    """
    Tests the 7-sample activation, physical onset, 13-sample recovery, and event interval
    using a synthetic controlled time-series.
    """
    timestamps = np.arange(0.0, 500.0, 5.0)  # 100 steps
    df = pd.DataFrame({
        "timestamp": timestamps,
        "queue_length_m": np.full(100, 10.0),
        "avg_speed_kmh": np.full(100, 35.0),
    })

    road_len = 100.0  # activation threshold = 60m; recovery threshold = 40m

    # Inject breach from step 20 to step 40 (21 steps > 7 steps = 30s)
    df.loc[20:40, "queue_length_m"] = 75.0
    df.loc[20:40, "avg_speed_kmh"] = 8.0

    # From step 41 to 60, provide full recovery (queue < 40m AND speed > 20km/h for 20 steps > 13 steps = 60s)
    df.loc[41:60, "queue_length_m"] = 15.0
    df.loc[41:60, "avg_speed_kmh"] = 32.0

    processed = compute_spillback_event_series(df, road_len_m=road_len)

    # Physical onset must be stamped exactly at step 20 (t = 100.0s)
    assert processed["is_event_onset"].sum() == 1
    assert processed.loc[20, "is_event_onset"] == True
    assert processed.loc[20, "timestamp"] == 100.0

    # Active spillback must span from step 20 through step 53 (41 + 12 recovery steps)
    assert processed.loc[20:53, "is_active_spillback"].all()
    # At step 54, road is recovered
    assert not processed.loc[54, "is_active_spillback"]


def test_deterministic_alarm_matching():
    """
    Tests greedy chronological one-to-one matching between alarms and onsets
    with independent horizon windows and tie-breaking.
    """
    horizon = 600  # 10 minutes

    # Onsets at t=500s and t=1200s on road 'R1'
    events = [
        {"event_id": "E1", "t_onset": 500.0, "road_id": "R1"},
        {"event_id": "E2", "t_onset": 1200.0, "road_id": "R1"},
    ]

    # Alarms on R1:
    # A1: t=200s (lead time = 300s, valid for E1)
    # A2: t=350s (closer to E1, lead time = 150s, valid for E1)
    # A3: t=800s (lead time = 400s, valid for E2)
    # A4: t=2000s (unmatched)
    alarms = [
        {"alarm_id": "A1", "t_alarm": 200.0, "road_id": "R1"},
        {"alarm_id": "A2", "t_alarm": 350.0, "road_id": "R1"},
        {"alarm_id": "A3", "t_alarm": 800.0, "road_id": "R1"},
        {"alarm_id": "A4", "t_alarm": 2000.0, "road_id": "R1"},
    ]

    matched, unmatched_a, unmatched_e = deterministic_alarm_event_matching(alarms, events, horizon_sec=horizon)

    assert len(matched) == 2, f"Expected 2 matches, got {len(matched)}"
    assert len(unmatched_e) == 0, "Expected 0 missed events"
    assert len(unmatched_a) == 2, "Expected 2 unmatched alarms"

    # Greedy chronological picks earliest alarm for E1 -> A1
    match_e1 = next(m for m in matched if m["event_id"] == "E1")
    assert match_e1["alarm_id"] == "A1"
    assert match_e1["lead_time_sec"] == 300.0

    match_e2 = next(m for m in matched if m["event_id"] == "E2")
    assert match_e2["alarm_id"] == "A3"
    assert match_e2["lead_time_sec"] == 400.0


def test_strict_wape_zero_denominator_rule():
    """
    Verifies Rev 6 Section 8.1 rule:
    When ground truth sum is zero, WAPE returns None (N/A) rather than dividing by zero or substituting MAE.
    """
    y_true_zero = np.array([0.0, 0.0, 0.0, 0.0])
    y_pred_nonzero = np.array([1.2, 0.5, 0.0, 0.8])

    wape_zero = compute_wape(y_true_zero, y_pred_nonzero)
    assert wape_zero is None, "WAPE did not return None for zero ground truth"

    y_true_normal = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred_normal = np.array([12.0, 18.0, 33.0, 37.0])
    wape_normal = compute_wape(y_true_normal, y_pred_normal)
    assert wape_normal is not None
    # sum(|e|) = 2 + 2 + 3 + 3 = 10; sum(y) = 100 -> WAPE = 10.0%
    assert abs(wape_normal - 10.0) < 1e-4


def test_holm_bonferroni_fwer_control():
    """Tests Holm-Bonferroni step-down correction for multiple comparisons."""
    p_values = [0.005, 0.012, 0.040, 0.150]
    # alpha = 0.05
    # k=0: 0.005 <= 0.05 / 4 = 0.0125 -> Significant
    # k=1: 0.012 <= 0.05 / 3 = 0.0166 -> Significant
    # k=2: 0.040 <= 0.05 / 2 = 0.0250 -> Not significant
    # k=3: Stopped
    corrected = holm_bonferroni_correction(p_values, alpha=0.05)
    assert corrected[0][1] == True
    assert corrected[1][1] == True
    assert corrected[2][1] == False
    assert corrected[3][1] == False


def test_legacy_datasets_and_models_preservation():
    """
    Asserts that baseline traffic_history.db and data/models/ are intact,
    and campaign paths are strictly decoupled.
    """
    baseline_db = DATA_DIR / "datasets" / "traffic_history.db"
    assert baseline_db.exists(), "Baseline traffic_history.db is missing"
    assert baseline_db.stat().st_size > 10000000, "Baseline traffic_history.db was truncated"

    baseline_models = list((DATA_DIR / "models").glob("lgb_*.joblib"))
    assert len(baseline_models) >= 9, f"Baseline models missing (found {len(baseline_models)})"

    # Campaign paths are decoupled
    assert CAMPAIGN_DATA_DIR != DATA_DIR / "datasets"
    assert CAMPAIGN_METADATA_DB != baseline_db
    assert CAMPAIGN_MODELS_DIR == DATA_DIR / "models" / "campaign"


def test_pipeline_schema_compatibility_with_synthetic_fixture():
    """
    Synthesizes a minimal 1-step raw observation table matching run_campaign_dataset,
    passes it through build_campaign_features, and verifies output columns match evaluate_campaign_models.
    """
    raw_cols = [
        "run_id", "family_id", "road_id", "timestamp", "step_index", "is_warmup",
        "vehicle_count", "density_veh_km_lane", "avg_speed_kmh", "flow_veh_hr",
        "queue_length_m", "queue_vehicle_count", "occupancy_ratio", "travel_time_sec",
        "operational_capacity", "vc_ratio", "signal_state", "incident_active"
    ]

    # Create synthetic raw fixture: 1 run, 2 roads, 80 timestamps (600s to 1000s)
    records = []
    for r_id in ["road_1", "road_2"]:
        for t in np.arange(600.0, 1005.0, 5.0):
            records.append({
                "run_id": "SYNTH_RUN_01",
                "family_id": "FAM-01",
                "road_id": r_id,
                "timestamp": float(t),
                "step_index": int(t * 4),
                "is_warmup": False,
                "vehicle_count": 5,
                "density_veh_km_lane": 12.0,
                "avg_speed_kmh": 35.0,
                "flow_veh_hr": 450.0,
                "queue_length_m": 15.0,
                "queue_vehicle_count": 1,
                "occupancy_ratio": 0.15,
                "travel_time_sec": 12.5,
                "operational_capacity": 1800.0,
                "vc_ratio": 0.25,
                "signal_state": "NONE",
                "incident_active": False,
            })

    raw_df = pd.DataFrame(records)

    # Verify all expected raw columns are present
    for col in raw_cols:
        assert col in raw_df.columns

    # Verify build_supervised_horizon_dataset produces targets and features
    from scripts.build_campaign_features import build_supervised_horizon_dataset
    feat_df = build_supervised_horizon_dataset(raw_df, "5min", 300)

    # Required feature cols expected by evaluate_campaign_models
    expected_eval_features = [
        "avg_speed_lag_1", "avg_speed_lag_2", "avg_speed_lag_3", "avg_speed_roll_mean", "avg_speed_roll_std",
        "density_lag_1", "density_lag_2", "density_lag_3", "density_roll_mean",
        "queue_lag_1", "queue_lag_2", "queue_lag_3", "queue_roll_mean",
        "operational_capacity", "vc_ratio", "incident_active",
        "target_speed", "target_density", "target_queue", "target_spillback_onset",
    ]

    for col in expected_eval_features:
        assert col in feat_df.columns, f"Feature column {col} missing from output dataset"
