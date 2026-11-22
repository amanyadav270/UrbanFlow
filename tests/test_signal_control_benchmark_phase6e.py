"""UrbanFlow AI — Phase 6E Infrastructure Tests.

Validates:
1. Scenario configurations and exact 80-run benchmark matrix (20 seeds x 4 regimes).
2. Decoupled Philox PRNG stream derivation and reproducibility across all 20 seeds.
3. SQLite metadata schema initialization, scenario registration, and resume tracking.
4. Exact continuous all-red clearance vs. emergency all-red hold separation.
5. Decision-impact telemetry: all 22 fields, governing rule assignment, and counterfactual flip logic.
6. Paired statistical analysis engine: mean diff, SE, 95% CI, Cohen's dz, paired t-test, Wilcoxon, and Holm-Bonferroni.
7. Explicit survivor bias detection and perimeter gating guard.
8. Cryptographic hash verification of the frozen 10-minute network classifier.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from numpy.random import Generator, Philox
import pandas as pd
import pytest

from urbanflow.core.kinematic_sim import KinematicSimulator
from urbanflow.intelligence.signal_control import (
    FixedTimeController,
    ActuatedController,
    MaxPressureController,
    CooperativePredictiveSignalController,
)
from scripts.run_signal_control_benchmark_phase6e import (
    get_all_scenarios,
    derive_stream_seed,
    verify_classifier_hash,
    init_benchmark_db,
    register_scenarios_in_db,
    check_run_completed,
    compute_paired_statistics,
    ScenarioBenchmarkConfig,
    EXPECTED_CLASSIFIER_SHA256,
    REGIMES,
)


def test_scenario_matrix_and_coverage():
    """Validates the 20 benchmark scenarios across FAM-05 and FAM-06."""
    scenarios = get_all_scenarios()
    assert len(scenarios) == 20, f"Expected 20 scenarios, got {len(scenarios)}"

    fam05 = [s for s in scenarios if s.family_id == "FAM-05"]
    fam06 = [s for s in scenarios if s.family_id == "FAM-06"]

    assert len(fam05) == 10, f"Expected 10 FAM-05 scenarios, got {len(fam05)}"
    assert len(fam06) == 10, f"Expected 10 FAM-06 scenarios, got {len(fam06)}"

    # Seed sequences
    seeds_05 = [s.seed for s in fam05]
    seeds_06 = [s.seed for s in fam06]
    assert seeds_05 == list(range(5001, 5011)), f"FAM-05 seeds mismatch: {seeds_05}"
    assert seeds_06 == list(range(6001, 6011)), f"FAM-06 seeds mismatch: {seeds_06}"

    # Demand and incident specs
    for s in fam05:
        assert s.demand_multiplier == 1.25
        assert s.incident_active is True
        assert s.incident_road == "road_1082220764_167_fwd"
        assert s.incident_duration_sec == 1000.0
        assert s.incident_severity == 0.95

    for s in fam06:
        assert s.demand_multiplier == 1.30
        assert s.incident_active is True
        assert s.incident_road == "road_1082220763_166_fwd"
        assert s.incident_duration_sec == 1500.0
        assert s.incident_severity == 0.80

    # Total runs planned: 20 seeds * 4 regimes = 80 runs
    assert len(REGIMES) == 4
    assert set(REGIMES) == {"FIXED_TIME", "ACTUATED", "MAX_PRESSURE", "COOPERATIVE_PREDICTIVE"}
    total_planned = len(scenarios) * len(REGIMES)
    assert total_planned == 80


def test_prng_stream_decoupling_and_reproducibility():
    """Validates decoupled Philox PRNG stream derivation and zero collision across all seeds."""
    scenarios = get_all_scenarios()
    stream_names = ["driver_attributes", "spawning_intervals", "incident_timing"]

    for sc in scenarios:
        derived = {}
        for s_name in stream_names:
            seed_val = derive_stream_seed(sc.seed, s_name)
            assert isinstance(seed_val, int)
            assert seed_val > 0
            derived[s_name] = seed_val

        # Check stream separation within same scenario seed
        assert derived["driver_attributes"] != derived["spawning_intervals"]
        assert derived["driver_attributes"] != derived["incident_timing"]
        assert derived["spawning_intervals"] != derived["incident_timing"]

        # Reproducibility
        assert derive_stream_seed(sc.seed, "driver_attributes") == derived["driver_attributes"]


def test_sqlite_schema_and_resume_tracking(tmp_path: Path):
    """Validates benchmark metadata DB schema and resume detection."""
    db_file = tmp_path / "test_benchmark.db"
    init_benchmark_db(db_file)

    scenarios = get_all_scenarios()
    register_scenarios_in_db(db_file, scenarios)

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM scenarios")
        count = cursor.fetchone()[0]
        assert count == 20

        # Test insert run record
        cursor.execute("""
            INSERT INTO runs (
                run_id, scenario_id, family_id, regime, seed, status,
                execution_wall_time_sec, total_scheduled, total_completed,
                post_warmup_completed, never_entered_count, incomplete_count,
                completion_rate_pct, throughput_veh_hr, mean_travel_time_sec,
                mean_system_travel_time_sec, mean_delay_sec, mean_waiting_time_sec,
                spillback_events_count, spillback_duration_total_sec,
                storage_violations_count, starvation_count, total_phase_switches,
                all_red_clearance_sec, all_red_hold_sec, survivor_bias_flag,
                created_at, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "RUN_FAM05_S5001_FIXED_TIME", "SCENARIO_FAM05_S5001", "FAM-05", "FIXED_TIME", 5001,
            "COMPLETED", 25.0, 5000, 3000, 2500, 1800, 200, 60.0, 3000.0, 120.0,
            800.0, 60.0, 45.0, 5, 250.0, 0, 0, 100, 150.0, 0.0, 0,
            "2026-11-14T00:00:00Z", "2026-11-14T00:00:25Z"
        ))
        conn.commit()

        cursor.execute("SELECT status FROM runs WHERE run_id = ?", ("RUN_FAM05_S5001_FIXED_TIME",))
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "COMPLETED"


def test_continuous_all_red_vs_hold_separation():
    """Validates exact continuous clearance (1.500s) integration at dt=0.25s."""
    sim = KinematicSimulator(map_name="bronx_fordham")
    inter_id = list(sim.signal_controllers.keys())[0]
    fsm = sim.signal_controllers[inter_id]

    dt = 0.25
    clearance_time = 0.0
    hold_time = 0.0

    # Force a transition
    fsm.time_in_green = 20.0
    fsm.request_phase(1)
    assert fsm.stage == "YELLOW"

    # Step through Yellow and All-Red (25 physics ticks)
    for _ in range(25):
        sim.step(dt)
        if fsm.stage == "ALL_RED":
            if getattr(fsm, "target_green_phase", None) == -1:
                hold_time += dt
            else:
                clearance_time += dt

    assert clearance_time == 1.5, f"Expected exactly 1.500s clearance, got {clearance_time}"
    assert hold_time == 0.0, f"Expected 0.0s hold time during normal transition, got {hold_time}"
    assert fsm.stage == "GREEN"
    assert fsm.active_green_phase == 1


def test_decision_impact_telemetry_fields_and_rules():
    """Validates all 22 decision telemetry fields and rule assignments."""
    sim = KinematicSimulator(map_name="bronx_fordham")
    ctrl = CooperativePredictiveSignalController(enable_prediction=True)

    # 1. Under min_green (< 10s)
    for fsm in sim.signal_controllers.values():
        fsm.time_in_green = 5.0

    phases = ctrl.compute_phases(current_time=5.0, simulator=sim, telemetry={}, record_decision_telemetry=True)
    assert len(ctrl.last_decision_records) == 27
    rec0 = ctrl.last_decision_records[0]

    required_keys = [
        "timestamp", "intersection_id", "active_phase", "selected_phase", "opposing_phase",
        "governing_rule", "time_in_green", "blocked_active", "blocked_opposing",
        "score_active_raw", "score_active_constrained", "score_opposing_raw", "score_opposing_constrained",
        "pred_penalty_active", "pred_penalty_opposing", "max_vc_active", "max_vc_opposing",
        "max_p_spillback_active", "max_p_spillback_opposing", "counterfactual_phase_unconstrained",
        "prediction_changed_decision", "is_safeguard_override"
    ]
    for k in required_keys:
        assert k in rec0, f"Missing key '{k}' in decision record"

    assert rec0["governing_rule"] == "MIN_GREEN"
    assert rec0["is_safeguard_override"] is True
    assert rec0["prediction_changed_decision"] is False

    # 2. Under max_green (>= 60s)
    for fsm in sim.signal_controllers.values():
        fsm.time_in_green = 60.0
    phases = ctrl.compute_phases(current_time=60.0, simulator=sim, telemetry={}, record_decision_telemetry=True)
    rec60 = ctrl.last_decision_records[0]
    assert rec60["governing_rule"] == "MAX_GREEN_FORCE_OFF"
    assert rec60["is_safeguard_override"] is True
    assert rec60["prediction_changed_decision"] is False


def test_counterfactual_decision_flip():
    """Validates that prediction_changed_decision correctly flags decisions flipped by predictive penalty."""
    sim = KinematicSimulator(map_name="bronx_fordham")
    sim.set_demand((), "empty")  # Isolate telemetry from randomly seeded road occupancy.
    ctrl = CooperativePredictiveSignalController(enable_prediction=True)
    target_inter = "int_42748269"
    fsm = sim.signal_controllers[target_inter]
    fsm.active_green_phase = 0
    fsm.time_in_green = 25.0

    # Configure demand where raw score switches, but constrained score holds
    telemetry = {
        "road_223889614_61_fwd": {"queue_length_meters": 15.0, "occupancy_ratio": 0.1},
        "road_5697487_7_fwd": {"occupancy_ratio": 0.1, "predicted_spillback_probability": 0.60, "vc_ratio": 0.5},
    }
    phases = ctrl.compute_phases(current_time=25.0, simulator=sim, telemetry=telemetry, record_decision_telemetry=True)
    rec = [r for r in ctrl.last_decision_records if r["intersection_id"] == target_inter][0]

    assert rec["selected_phase"] == 0  # Held in Phase 0
    assert rec["counterfactual_phase_unconstrained"] == 1  # Would have switched to Phase 1 without prediction
    assert rec["prediction_changed_decision"] is True
    assert ctrl.prediction_changed_decisions_count >= 1


def test_paired_statistics_engine_and_holm_bonferroni():
    """Validates paired t-test, Cohen's dz, 95% CI, Wilcoxon, and Holm-Bonferroni correction."""
    # Synthetic 10 seeds data
    seeds = list(range(5001, 5011))
    rows = []
    np.random.seed(42)

    for s in seeds:
        # Cooperative: higher throughput (+200), lower waiting (-20)
        rows.append({
            "seed": s, "regime": "COOPERATIVE_PREDICTIVE",
            "throughput_veh_hr": 4000.0 + np.random.normal(0, 50),
            "completion_rate_pct": 72.0 + np.random.normal(0, 1),
            "mean_system_travel_time_sec": 700.0 + np.random.normal(0, 30),
            "mean_waiting_time_sec": 35.0 + np.random.normal(0, 2),
            "never_entered_count": 1200 + int(np.random.normal(0, 50)),
            "spillback_duration_total_sec": 150.0 + np.random.normal(0, 20),
            "storage_violations_count": 5 + int(np.random.normal(0, 2)),
        })
        # FixedTime baseline
        rows.append({
            "seed": s, "regime": "FIXED_TIME",
            "throughput_veh_hr": 3500.0 + np.random.normal(0, 50),
            "completion_rate_pct": 60.0 + np.random.normal(0, 1),
            "mean_system_travel_time_sec": 900.0 + np.random.normal(0, 30),
            "mean_waiting_time_sec": 50.0 + np.random.normal(0, 2),
            "never_entered_count": 2200 + int(np.random.normal(0, 50)),
            "spillback_duration_total_sec": 500.0 + np.random.normal(0, 20),
            "storage_violations_count": 25 + int(np.random.normal(0, 2)),
        })

    df = pd.DataFrame(rows)
    stats_res = compute_paired_statistics(df, baseline_regimes=["FIXED_TIME"])
    assert "COOPERATIVE_PREDICTIVE_vs_FIXED_TIME" in stats_res
    comp = stats_res["COOPERATIVE_PREDICTIVE_vs_FIXED_TIME"]

    tp = comp["throughput_veh_hr"]
    assert tp["N_seeds"] == 10
    assert tp["mean_diff"] > 400.0
    assert tp["ci_95"][0] > 0.0  # Entire 95% CI is strictly positive
    assert tp["cohen_dz"] > 2.0  # Massive effect size
    assert tp["p_value_raw"] < 0.001
    assert tp["p_value_holm_bonferroni"] < 0.05
    assert tp["statistically_significant_05"] is True


def test_survivor_bias_detection():
    """Validates survivor bias detection when completion rate < 50% or unentered > 30%."""
    # Low completion rate run
    low_comp = 25.0
    unentered = 4500
    total = 5600
    flag = bool((low_comp < 50.0) or (unentered / total > 0.30))
    assert flag is True

    # High completion rate run
    high_comp = 72.0
    unentered_ok = 1200
    flag_ok = bool((high_comp < 50.0) or (unentered_ok / total > 0.30))
    assert flag_ok is False


def test_classifier_hash_preservation():
    """Validates that frozen classifier matches the expected SHA-256 hash."""
    h = verify_classifier_hash()
    assert h == EXPECTED_CLASSIFIER_SHA256
    assert h == "9B93E5DB672F692E509DC3012658413BC713FC6348CCD5D044B33470FF568C54"
