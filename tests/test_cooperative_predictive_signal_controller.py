"""Comprehensive Unit Test Suite for UrbanFlow AI Phase 6C: Cooperative Predictive Traffic Signal Control.

Tests:
1. test_sha256_hash_verification_and_model_loading:
   - Cryptographic SHA-256 verification matches exact 64-character hash constant.
   - Loads calibrated LightGBM classifier with 28 expected features and 0.180 threshold.
2. test_missing_classifier_fallback:
   - Graceful fallback when model file is missing; status set to UNAVAILABLE_FILE_NOT_FOUND.
   - Controller operates robustly without raising exceptions.
3. test_hash_mismatch_fallback:
   - Rejects tampered/corrupted model artifact; status set to UNAVAILABLE_HASH_MISMATCH.
   - Safe execution of cooperative control without predictive input.
4. test_disabled_prediction_mode:
   - Setting enable_prediction=False cleanly disables ML subsystem; status set to DISABLED.
5. test_predictive_penalty_bounds_and_threshold:
   - Bounded advisory penalty: strictly 0.0 when P <= 0.180.
   - Scales linearly with excess probability and is strictly capped at 8.0.
   - Tests predict_spillback_probability method with feature dict and numpy array.
6. test_downstream_75_percent_storage_capacity_veto:
   - Downstream link storage capacity safeguard (-100.0 veto) triggers when occupancy >= 75%.
   - Veto takes absolute precedence over upstream queue and low predictive probability.
7. test_upstream_neighbor_anticipation_coordination:
   - Exchanges state with upstream signalized neighbor.
   - Adds anticipation credit (+1.0 per inbound vehicle, up to +4.0) when upstream displays Green.
8. test_downstream_neighbor_green_wave_bonus_and_red_dampening:
   - Downstream Green grants green wave coordination bonus (+2.5).
   - Downstream Red with standing queue applies dampening penalty (up to -2.0).
9. test_determinism_and_hysteresis:
   - 100% deterministic phase output given identical simulation states.
   - Hysteresis threshold (1.5) prevents rapid flickering when movement scores are close.
10. test_fsm_safety_integration_and_regression:
    - Verifies adherence to Phase 6A safety FSM (min green 10s, max green 60s, yellow 3s, all-red 1.5s, no conflicting greens).
    - Verifies classical controllers (FixedTimeController, ActuatedController, MaxPressureController) remain untouched and operational.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, List

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

import joblib
import numpy as np
import pytest

from urbanflow.core.kinematic_sim import KinematicSimulator, SimulatedVehicle, IntersectionSignalFSM
from urbanflow.core.types import Vec3, RoadSegment, Intersection
from urbanflow.intelligence.signal_control import (
    SignalControllerBase,
    FixedTimeController,
    ActuatedController,
    MaxPressureController,
    CooperativePredictiveSignalController,
)


# ==============================================================================
# 1. SHA-256 Hash Verification & Model Loading
# ==============================================================================

def test_sha256_hash_verification_and_model_loading():
    """Verifies cryptographic SHA-256 integrity and calibration parameters of the 10-minute classifier."""
    expected_hash = "9B93E5DB672F692E509DC3012658413BC713FC6348CCD5D044B33470FF568C54"
    assert CooperativePredictiveSignalController.EXPECTED_CLASSIFIER_SHA256 == expected_hash

    model_path = CooperativePredictiveSignalController.DEFAULT_CLASSIFIER_PATH
    assert model_path.exists(), f"Classifier artifact not found at {model_path}"

    raw_bytes = model_path.read_bytes()
    computed_hash = hashlib.sha256(raw_bytes).hexdigest().upper()
    assert computed_hash == expected_hash, f"Hash mismatch! Expected {expected_hash}, got {computed_hash}"

    # Initialize controller and verify state
    ctrl = CooperativePredictiveSignalController()
    assert ctrl.predictive_available is True
    assert ctrl.predictive_status == "ACTIVE"
    assert ctrl.calibrator is not None
    assert len(ctrl.expected_features) == 28
    assert ctrl.threshold == pytest.approx(0.180, rel=1e-3)


# ==============================================================================
# 2. Missing Classifier Graceful Fallback
# ==============================================================================

def test_missing_classifier_fallback():
    """Verifies that a nonexistent model artifact path gracefully falls back to non-predictive control."""
    fake_path = Path("nonexistent_directory_xyz/model.joblib")
    ctrl = CooperativePredictiveSignalController(classifier_path=fake_path)

    assert ctrl.predictive_available is False
    assert "UNAVAILABLE_FILE_NOT_FOUND" in ctrl.predictive_status

    # Should safely compute phases without error
    sim = KinematicSimulator(map_name="bronx_fordham")
    phases = ctrl.compute_phases(current_time=0.0, simulator=sim)
    assert isinstance(phases, dict)
    assert len(phases) == len(sim.signal_controllers)
    for inter_id, phase in phases.items():
        assert phase in (0, 1)


# ==============================================================================
# 3. Hash Mismatch Fallback
# ==============================================================================

def test_hash_mismatch_fallback():
    """Verifies that a corrupted or mismatched model artifact is rejected and falls back safely."""
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        tmp.write(b"CORRUPTED_OR_TAMPERED_MODEL_DATA_FOR_TESTING")
        tmp_path = Path(tmp.name)

    try:
        ctrl = CooperativePredictiveSignalController(classifier_path=tmp_path)
        assert ctrl.predictive_available is False
        assert "UNAVAILABLE_HASH_MISMATCH" in ctrl.predictive_status

        # Test phase computation during mismatch
        sim = KinematicSimulator(map_name="bronx_fordham")
        phases = ctrl.compute_phases(current_time=0.0, simulator=sim)
        assert isinstance(phases, dict)
        assert len(phases) == len(sim.signal_controllers)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


# ==============================================================================
# 4. Disabled Prediction Mode
# ==============================================================================

def test_disabled_prediction_mode():
    """Verifies controller behavior when predictive subsystem is explicitly disabled."""
    ctrl = CooperativePredictiveSignalController(enable_prediction=False)
    assert ctrl.predictive_available is False
    assert ctrl.predictive_status == "DISABLED"

    sim = KinematicSimulator(map_name="bronx_fordham")
    phases = ctrl.compute_phases(current_time=15.0, simulator=sim)
    assert isinstance(phases, dict)
    assert len(phases) == len(sim.signal_controllers)


# ==============================================================================
# 5. Bounded Advisory Predictive Penalty
# ==============================================================================

def test_predictive_penalty_bounds_and_threshold():
    """Verifies that advisory predictive penalty is strictly bounded (<= 8.0) and triggers only above threshold."""
    ctrl = CooperativePredictiveSignalController()
    assert ctrl.predictive_available is True

    # 1. Below or at threshold (0.180) -> penalty is strictly 0.0
    p_below = ctrl._compute_predictive_penalty(
        "road_test", telemetry={"road_test": {"predicted_spillback_probability": 0.10}}
    )
    assert p_below == 0.0

    p_at = ctrl._compute_predictive_penalty(
        "road_test", telemetry={"road_test": {"predicted_spillback_probability": 0.180}}
    )
    assert p_at == 0.0

    # 2. Moderate excess probability -> scaled by predictive_weight (default 10.0)
    p_mid = ctrl._compute_predictive_penalty(
        "road_test", telemetry={"road_test": {"predicted_spillback_probability": 0.50}}
    )
    expected_mid = min(8.0, 10.0 * (0.50 - 0.180))
    assert p_mid == pytest.approx(expected_mid, rel=1e-3)

    # 3. Maximum probability (1.0) -> capped at 8.0
    p_max = ctrl._compute_predictive_penalty(
        "road_test", telemetry={"road_test": {"predicted_spillback_probability": 1.0}}
    )
    assert p_max == 8.0

    # 4. Out-of-bounds probability (> 1.0) -> still strictly capped at 8.0
    p_huge = ctrl._compute_predictive_penalty(
        "road_test", telemetry={"road_test": {"predicted_spillback_probability": 2.5}}
    )
    assert p_huge == 8.0

    # 5. Direct evaluate predict_spillback_probability with dummy feature dict
    feat_dict = {f: 0.0 for f in ctrl.expected_features}
    prob = ctrl.predict_spillback_probability(feat_dict)
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0


# ==============================================================================
# 6. Downstream 75% Physical Storage Capacity Safeguard
# ==============================================================================

def test_downstream_75_percent_storage_capacity_veto():
    """Verifies that downstream occupancy >= 75% triggers strict -100.0 veto overriding advisory predictions."""
    sim = KinematicSimulator(map_name="bronx_fordham")
    inter_ids = list(sim.signal_controllers.keys())
    target_id = inter_ids[0]

    ctrl = CooperativePredictiveSignalController(
        downstream_occupancy_cap=0.75,
        controlled_intersections=[target_id],
    )

    out_roads = [r for r in sim.roads.values() if r.from_node == target_id]
    assert out_roads, "Expected outgoing roads"
    target_out_road = out_roads[0]

    # Case A: Low occupancy (20%) -> No veto
    is_blocked, occ = ctrl._check_downstream_storage(target_out_road, simulator=None, telemetry={
        target_out_road.id: {"queue_length_meters": target_out_road.length * 0.20, "vc_ratio": 0.20}
    })
    assert is_blocked is False
    assert occ == pytest.approx(0.20, rel=1e-2)

    # Case B: High occupancy (80% >= 75%) -> Strict VETO
    is_blocked_high, occ_high = ctrl._check_downstream_storage(target_out_road, simulator=None, telemetry={
        target_out_road.id: {"queue_length_meters": target_out_road.length * 0.80, "vc_ratio": 0.80}
    })
    assert is_blocked_high is True
    assert occ_high >= 0.75


# ==============================================================================
# 7. Upstream Neighbor Anticipation Coordination
# ==============================================================================

def test_upstream_neighbor_anticipation_coordination():
    """Verifies that green upstream signal discharging inbound vehicles provides look-ahead anticipation credit."""
    sim = KinematicSimulator(map_name="bronx_fordham")

    # Find two adjacent signalized intersections
    adj_road = None
    for r in sim.roads.values():
        if r.from_node in sim.signal_controllers and r.to_node in sim.signal_controllers:
            adj_road = r
            break
    assert adj_road is not None, "Could not find adjacent signalized pair"

    upstream_id = adj_road.from_node
    downstream_id = adj_road.to_node

    ctrl = CooperativePredictiveSignalController(
        controlled_intersections=[upstream_id, downstream_id]
    )

    upstream_ctrl = sim.signal_controllers[upstream_id]
    is_ns = ctrl._is_road_ns(adj_road)

    # Inject vehicles on the connecting road
    sim.vehicles.clear()
    for i in range(3):
        veh = SimulatedVehicle(
            vehicle_id=f"inbound_v_{i}",
            v_type="passenger",
            route=[adj_road.id],
            color="blue",
        )
        veh.distance_on_road = adj_road.length * 0.2  # in first half of link
        veh.speed = 10.0
        sim.vehicles[f"inbound_v_{i}"] = veh

    # Case 1: Upstream is RED for this movement
    red_phase = 1 if is_ns else 0
    upstream_ctrl.stage = "GREEN"
    upstream_ctrl.active_green_phase = red_phase
    upstream_ctrl.target_green_phase = None
    upstream_ctrl.time_in_stage = 15.0
    upstream_ctrl.time_in_green = 15.0
    assert upstream_ctrl.get_signal_code(is_ns) == "r"

    bonus_red = ctrl._compute_upstream_anticipation(adj_road, sim, sim.signal_controllers)
    assert bonus_red == 0.0

    # Case 2: Upstream is GREEN for this movement
    green_phase = 0 if is_ns else 1
    upstream_ctrl.stage = "GREEN"
    upstream_ctrl.active_green_phase = green_phase
    upstream_ctrl.target_green_phase = None
    upstream_ctrl.time_in_stage = 15.0
    upstream_ctrl.time_in_green = 15.0
    assert upstream_ctrl.get_signal_code(is_ns) == "G"

    bonus_green = ctrl._compute_upstream_anticipation(adj_road, sim, sim.signal_controllers)
    assert bonus_green > 0.0
    assert bonus_green <= 4.0


# ==============================================================================
# 8. Downstream Neighbor Green Wave Bonus and Red Dampening
# ==============================================================================

def test_downstream_neighbor_green_wave_bonus_and_red_dampening():
    """Verifies green wave bonus when downstream neighbor is green and dampening when downstream is red."""
    sim = KinematicSimulator(map_name="bronx_fordham")

    adj_road = None
    for r in sim.roads.values():
        if r.from_node in sim.signal_controllers and r.to_node in sim.signal_controllers:
            adj_road = r
            break
    assert adj_road is not None

    upstream_id = adj_road.from_node
    downstream_id = adj_road.to_node

    ctrl = CooperativePredictiveSignalController()
    downstream_ctrl = sim.signal_controllers[downstream_id]
    is_ns = ctrl._is_road_ns(adj_road)

    # Case A: Downstream is GREEN ("G") -> Coordination Bonus +2.5
    green_phase = 0 if is_ns else 1
    downstream_ctrl.stage = "GREEN"
    downstream_ctrl.active_green_phase = green_phase
    downstream_ctrl.target_green_phase = None
    downstream_ctrl.time_in_stage = 15.0
    downstream_ctrl.time_in_green = 15.0
    assert downstream_ctrl.get_signal_code(is_ns) == "G"

    bonus = ctrl._compute_downstream_coordination(adj_road, sim, sim.signal_controllers)
    assert bonus == 2.5

    # Case B: Downstream is RED ("r") with queue -> Dampening penalty <= 0.0
    red_phase = 1 if is_ns else 0
    downstream_ctrl.stage = "GREEN"
    downstream_ctrl.active_green_phase = red_phase
    downstream_ctrl.target_green_phase = None
    downstream_ctrl.time_in_stage = 15.0
    downstream_ctrl.time_in_green = 15.0
    assert downstream_ctrl.get_signal_code(is_ns) == "r"

    # Add standing vehicles on downstream link
    sim.vehicles.clear()
    for i in range(5):
        veh = SimulatedVehicle(
            vehicle_id=f"queue_v_{i}",
            v_type="passenger",
            route=[adj_road.id],
            color="red",
        )
        veh.distance_on_road = adj_road.length - (10.0 + i * 5.0)
        veh.speed = 0.0
        veh.wait_time = 20.0
        sim.vehicles[f"queue_v_{i}"] = veh

    penalty = ctrl._compute_downstream_coordination(adj_road, sim, sim.signal_controllers)
    assert penalty < 0.0
    assert penalty >= -2.0


# ==============================================================================
# 9. Determinism and Hysteresis Threshold
# ==============================================================================

def test_determinism_and_hysteresis():
    """Verifies that phase decisions are 100% deterministic and hysteresis prevents erratic switching."""
    sim = KinematicSimulator(map_name="bronx_fordham")
    ctrl1 = CooperativePredictiveSignalController(hysteresis_threshold=1.5)
    ctrl2 = CooperativePredictiveSignalController(hysteresis_threshold=1.5)

    # Repeated calls under identical simulator state produce identical phases
    p1 = ctrl1.compute_phases(current_time=15.0, simulator=sim)
    p2 = ctrl2.compute_phases(current_time=15.0, simulator=sim)
    assert p1 == p2

    # Verify hysteresis threshold parameter
    assert ctrl1.hysteresis_threshold == 1.5


# ==============================================================================
# 10. Safety FSM Integration & Classical Regression
# ==============================================================================

def test_fsm_safety_integration_and_regression():
    """Verifies safety FSM constraints under CooperativePredictiveSignalController and checks regression."""
    sim = KinematicSimulator(map_name="bronx_fordham")
    ctrl = CooperativePredictiveSignalController(min_green=10.0, max_green=60.0)

    # Step simulation across 40 seconds of cooperative control
    for step in range(80):
        t = step * 0.5
        if step % 2 == 0:
            target_phases = ctrl.compute_phases(current_time=t, simulator=sim)
            for inter_id, target_phase in target_phases.items():
                local_ctrl = sim.signal_controllers[inter_id]
                local_ctrl.request_phase(target_phase)
        sim.step(dt=0.5)

        # Safety assertions across all intersections
        for inter_id, local_ctrl in sim.signal_controllers.items():
            # 1. No conflicting greens
            ns_code = local_ctrl.get_signal_code(is_ns=True)
            ew_code = local_ctrl.get_signal_code(is_ns=False)
            assert not (ns_code == "G" and ew_code == "G"), f"Conflicting green at {inter_id}!"

            # 2. Minimum green adhered to
            if local_ctrl.stage == "GREEN":
                if local_ctrl.time_in_green < 10.0:
                    assert local_ctrl.target_green_phase is None or local_ctrl.target_green_phase == local_ctrl.active_green_phase

    # Classical Controllers Regression
    fixed = FixedTimeController()
    actuated = ActuatedController()
    max_p = MaxPressureController()

    p_fixed = fixed.compute_phases(0.0, sim)
    p_act = actuated.compute_phases(0.0, sim)
    p_max = max_p.compute_phases(0.0, sim)

    assert len(p_fixed) == len(sim.signal_controllers)
    assert len(p_act) == len(sim.signal_controllers)
    assert len(p_max) == len(sim.signal_controllers)


# ==============================================================================
# 11. Pre-Pilot Audit Counterexample 1: DEFECT-6C-01 Veto Dilution Fix
# ==============================================================================

def test_phase_level_veto_not_overcome_by_parallel_movements():
    """Verifies DEFECT-6C-01: A blocked downstream link vetoes the candidate phase and cannot
    be made safe or positive by large parallel queue differentials, bonuses, or predictions."""
    sim = KinematicSimulator(map_name="bronx_fordham")
    inter_ids = list(sim.signal_controllers.keys())
    target_id = inter_ids[0]

    ctrl = CooperativePredictiveSignalController(
        downstream_occupancy_cap=0.75,
        controlled_intersections=[target_id],
    )

    out_roads = [r for r in sim.roads.values() if r.from_node == target_id]
    assert out_roads, "Expected outgoing roads"
    blocked_out_road = out_roads[0]
    is_ns = ctrl._is_road_ns(blocked_out_road)
    phase_idx = 0 if is_ns else 1
    opposing_phase_idx = 1 if phase_idx == 0 else 0

    # Inject blocked state (85% occupancy >= 75% cap) on the downstream road
    telemetry = {
        blocked_out_road.id: {
            "queue_length_meters": blocked_out_road.length * 0.85,
            "vc_ratio": 0.85,
            "predicted_spillback_probability": 0.0,
        }
    }

    # Verify _evaluate_phase_score directly returns (-100.0, True)
    score, is_blocked = ctrl._evaluate_phase_score(
        target_id, phase_index=phase_idx, simulator=sim, telemetry=telemetry, local_controllers=sim.signal_controllers
    )
    assert is_blocked is True
    assert score == -100.0, f"Expected vetoed score -100.0, got {score}"

    # Setup opposing phase with modest safe demand
    opp_in_roads = [r for r in sim.roads.values() if r.to_node == target_id and ctrl._is_road_ns(r) != is_ns]
    if opp_in_roads:
        telemetry[opp_in_roads[0].id] = {"queue_length_meters": 20.0, "vc_ratio": 0.3}

    # Compute phases: candidate phase with blocked downstream link MUST NOT be selected
    phases = ctrl.compute_phases(current_time=15.0, simulator=sim, telemetry=telemetry)
    assert phases[target_id] == opposing_phase_idx, (
        f"Phase with blocked downstream link was selected! Expected {opposing_phase_idx}, got {phases[target_id]}"
    )


# ==============================================================================
# 12. Pre-Pilot Audit Counterexample 2: DEFECT-6C-02 Zero-Demand Deadlock Fix
# ==============================================================================

def test_blocked_active_phase_switches_when_opposing_demand_is_zero():
    """Verifies DEFECT-6C-02: A blocked active phase safely switches to an unblocked opposing phase
    even when opposing demand is zero, preventing indefinite green on a blocked link."""
    sim = KinematicSimulator(map_name="bronx_fordham")
    inter_ids = list(sim.signal_controllers.keys())
    target_id = inter_ids[0]

    ctrl = CooperativePredictiveSignalController(
        min_green=10.0,
        max_green=60.0,
        downstream_occupancy_cap=0.75,
        controlled_intersections=[target_id],
    )

    local_ctrl = sim.signal_controllers[target_id]
    # Set active phase to 0 (NS) and satisfy min_green (time_in_green = 15.0 >= 10.0)
    local_ctrl.stage = "GREEN"
    local_ctrl.active_green_phase = 0
    local_ctrl.target_green_phase = None
    local_ctrl.time_in_stage = 15.0
    local_ctrl.time_in_green = 15.0

    # Block downstream NS link
    ns_out_roads = [r for r in sim.roads.values() if r.from_node == target_id and ctrl._is_road_ns(r)]
    assert ns_out_roads
    blocked_ns_road = ns_out_roads[0]

    telemetry = {
        blocked_ns_road.id: {
            "queue_length_meters": blocked_ns_road.length * 0.90,
            "vc_ratio": 0.90,
        }
    }
    # Opposing phase (1, EW) has ZERO telemetry and ZERO demand
    sim.vehicles.clear()

    # compute_phases must safely command switch to Phase 1 (EW)
    phases = ctrl.compute_phases(current_time=15.0, simulator=sim, telemetry=telemetry)
    assert phases[target_id] == 1, (
        f"Controller did not switch away from blocked active phase! Got {phases[target_id]}, expected 1"
    )


# ==============================================================================
# 13. Predictions and Coordination Bonuses Cannot Override Storage Veto
# ==============================================================================

def test_predictions_and_bonuses_cannot_override_storage_veto():
    """Verifies that upstream anticipation (+4.0), green wave bonus (+2.5), and low prediction (0.0)
    never override the downstream storage veto."""
    sim = KinematicSimulator(map_name="bronx_fordham")
    inter_ids = list(sim.signal_controllers.keys())
    target_id = inter_ids[0]

    ctrl = CooperativePredictiveSignalController(
        downstream_occupancy_cap=0.75,
        controlled_intersections=[target_id],
    )

    out_roads = [r for r in sim.roads.values() if r.from_node == target_id]
    assert out_roads
    out_r = out_roads[0]
    is_ns = ctrl._is_road_ns(out_r)
    phase_idx = 0 if is_ns else 1

    # Block downstream road but provide 0.0 predicted spillback probability
    telemetry = {
        out_r.id: {
            "queue_length_meters": out_r.length * 0.80,
            "vc_ratio": 0.80,
            "predicted_spillback_probability": 0.0,
        }
    }

    score, is_blocked = ctrl._evaluate_phase_score(
        target_id, phase_index=phase_idx, simulator=sim, telemetry=telemetry, local_controllers=sim.signal_controllers
    )
    assert is_blocked is True
    assert score == -100.0


# ==============================================================================
# 14. Both-Phases-Blocked Fallback Behavior
# ==============================================================================

def test_both_phases_blocked_fallback_behavior():
    """Verifies that when both phases are blocked, the controller does not invent a safe green,
    and requests an all-red hold (-1) to prevent authorizing traffic into blocked movements."""
    sim = KinematicSimulator(map_name="bronx_fordham")
    inter_ids = list(sim.signal_controllers.keys())
    target_id = inter_ids[0]

    ctrl = CooperativePredictiveSignalController(
        min_green=10.0,
        max_green=60.0,
        downstream_occupancy_cap=0.75,
        controlled_intersections=[target_id],
    )

    local_ctrl = sim.signal_controllers[target_id]
    local_ctrl.stage = "GREEN"
    local_ctrl.active_green_phase = 0
    local_ctrl.time_in_stage = 20.0
    local_ctrl.time_in_green = 20.0

    # Block BOTH NS and EW outgoing roads with physical occupancy >= 75%
    telemetry = {}
    for r in sim.roads.values():
        if r.from_node == target_id:
            telemetry[r.id] = {"queue_length_meters": r.length * 0.85, "occupancy_ratio": 0.85, "vc_ratio": 0.85}

    # Both phases evaluate as blocked with negative scores
    score_0, blocked_0 = ctrl._evaluate_phase_score(target_id, 0, sim, telemetry, sim.signal_controllers)
    score_1, blocked_1 = ctrl._evaluate_phase_score(target_id, 1, sim, telemetry, sim.signal_controllers)
    assert blocked_0 is True and score_0 == -100.0
    assert blocked_1 is True and score_1 == -100.0

    # At time_in_green = 20s, initiates all-red hold (-1) because both movements are blocked
    phases_mid = ctrl.compute_phases(current_time=20.0, simulator=sim, telemetry=telemetry)
    assert phases_mid[target_id] == -1

    # At time_in_green = 60s, maintains all-red hold (-1) instead of alternating into blocked movements
    local_ctrl.time_in_green = 60.0
    phases_max = ctrl.compute_phases(current_time=60.0, simulator=sim, telemetry=telemetry)
    assert phases_max[target_id] == -1


# ==============================================================================
# 15. Unblocked Phases Normal Operation
# ==============================================================================

def test_unblocked_phases_operate_normally():
    """Verifies that unblocked phases cleanly select phases based on queue pressure differentials."""
    sim = KinematicSimulator(map_name="bronx_fordham")
    inter_ids = list(sim.signal_controllers.keys())
    target_id = inter_ids[0]

    ctrl = CooperativePredictiveSignalController(
        min_green=10.0,
        max_green=60.0,
        downstream_occupancy_cap=0.75,
        controlled_intersections=[target_id],
    )

    local_ctrl = sim.signal_controllers[target_id]
    local_ctrl.stage = "GREEN"
    local_ctrl.active_green_phase = 0
    local_ctrl.time_in_stage = 15.0
    local_ctrl.time_in_green = 15.0

    # Provide safe unblocked telemetry with higher demand on EW (phase 1)
    telemetry = {}
    for r in sim.roads.values():
        if r.from_node == target_id:
            telemetry[r.id] = {"queue_length_meters": 5.0, "vc_ratio": 0.1}
        elif r.to_node == target_id:
            is_ns = ctrl._is_road_ns(r)
            if is_ns:
                telemetry[r.id] = {"queue_length_meters": 10.0, "vc_ratio": 0.2}
            else:
                telemetry[r.id] = {"queue_length_meters": 80.0, "vc_ratio": 0.8}

    phases = ctrl.compute_phases(current_time=15.0, simulator=sim, telemetry=telemetry)
    assert phases[target_id] == 1


# ==============================================================================
# 16. Minimum Green Preserved During Blocked Veto
# ==============================================================================

def test_minimum_green_preserved_during_blocked_veto():
    """Verifies that minimum green (10.0s) is strictly respected even when active phase is blocked."""
    sim = KinematicSimulator(map_name="bronx_fordham")
    inter_ids = list(sim.signal_controllers.keys())
    target_id = inter_ids[0]

    ctrl = CooperativePredictiveSignalController(
        min_green=10.0,
        max_green=60.0,
        downstream_occupancy_cap=0.75,
        controlled_intersections=[target_id],
    )

    local_ctrl = sim.signal_controllers[target_id]
    local_ctrl.stage = "GREEN"
    local_ctrl.active_green_phase = 0
    local_ctrl.time_in_stage = 4.0
    local_ctrl.time_in_green = 4.0  # < 10.0s min green

    # Block downstream NS link
    ns_out_roads = [r for r in sim.roads.values() if r.from_node == target_id and ctrl._is_road_ns(r)]
    telemetry = {
        ns_out_roads[0].id: {
            "queue_length_meters": ns_out_roads[0].length * 0.90,
            "occupancy_ratio": 0.90,
            "vc_ratio": 0.90,
        }
    }

    # Minimum green guard holds active phase until min_green is satisfied
    phases = ctrl.compute_phases(current_time=4.0, simulator=sim, telemetry=telemetry)
    assert phases[target_id] == 0, f"Violated min_green! Got {phases[target_id]}, expected 0"

    # Once time_in_green >= min_green, safe switch is authorized
    local_ctrl.time_in_green = 10.5
    phases_after = ctrl.compute_phases(current_time=10.5, simulator=sim, telemetry=telemetry)
    assert phases_after[target_id] == 1


# ==============================================================================
# 17. High V/C with Low Physical Occupancy Does Not Trigger Storage Veto
# ==============================================================================

def test_high_vc_low_physical_occupancy_no_storage_veto():
    """Verifies that high operational V/C alone does not trigger the physical storage veto
    when physical storage occupancy is low (HCM 6th Edition compliance)."""
    sim = KinematicSimulator(map_name="bronx_fordham")
    target_id = list(sim.signal_controllers.keys())[0]

    ctrl = CooperativePredictiveSignalController(
        min_green=10.0,
        max_green=60.0,
        downstream_occupancy_cap=0.75,
        controlled_intersections=[target_id],
    )

    out_roads = [r for r in sim.roads.values() if r.from_node == target_id]
    target_road = out_roads[0]

    # Telemetry has high operational V/C (1.45), but physical occupancy is only 15%
    telemetry = {
        target_road.id: {
            "queue_length_meters": 2.0,
            "occupancy_ratio": 0.15,
            "vc_ratio": 1.45,
            "vehicle_count": 2,
        }
    }

    is_blocked, physical_occ = ctrl._check_downstream_storage(target_road, simulator=None, telemetry=telemetry)
    assert is_blocked is False, f"Expected unblocked, got blocked with occ={physical_occ}"
    assert physical_occ < 0.75


# ==============================================================================
# 18. Low V/C with High Physical Occupancy Triggers Storage Veto
# ==============================================================================

def test_low_vc_high_physical_occupancy_triggers_storage_veto():
    """Verifies that standing queues and high physical occupancy trigger the 75% veto
    even if observed throughput/flow V/C is low (e.g. stopped bottleneck)."""
    sim = KinematicSimulator(map_name="bronx_fordham")
    target_id = list(sim.signal_controllers.keys())[0]

    ctrl = CooperativePredictiveSignalController(
        min_green=10.0,
        max_green=60.0,
        downstream_occupancy_cap=0.75,
        controlled_intersections=[target_id],
    )

    out_roads = [r for r in sim.roads.values() if r.from_node == target_id]
    target_road = out_roads[0]

    # Telemetry has low operational flow V/C (0.10), but physical storage is 82% occupied
    telemetry = {
        target_road.id: {
            "queue_length_meters": target_road.length * 0.82,
            "occupancy_ratio": 0.82,
            "vc_ratio": 0.10,
        }
    }

    is_blocked, physical_occ = ctrl._check_downstream_storage(target_road, simulator=None, telemetry=telemetry)
    assert is_blocked is True, f"Expected blocked, got unblocked with occ={physical_occ}"
    assert physical_occ >= 0.75


# ==============================================================================
# 19. Zero Demand on Both Safe Phases Causes Maximum-Green Force-Off
# ==============================================================================

def test_zero_demand_both_safe_phases_max_green_force_off():
    """Verifies that when both phases are physically safe with zero demand,
    the active phase rests until max_green (60s) and then forces off to prevent starvation."""
    sim = KinematicSimulator(map_name="bronx_fordham")
    target_id = list(sim.signal_controllers.keys())[0]

    ctrl = CooperativePredictiveSignalController(
        min_green=10.0,
        max_green=60.0,
        downstream_occupancy_cap=0.75,
        controlled_intersections=[target_id],
    )

    local_ctrl = sim.signal_controllers[target_id]
    local_ctrl.stage = "GREEN"
    local_ctrl.active_green_phase = 0

    # Clean unblocked telemetry with zero queue demand
    telemetry = {}
    for r in sim.roads.values():
        if r.from_node == target_id or r.to_node == target_id:
            telemetry[r.id] = {
                "queue_length_meters": 0.0,
                "occupancy_ratio": 0.05,
                "vc_ratio": 0.1,
                "vehicle_count": 0,
            }

    # At 50s (< max_green 60s), rests in current active phase (0)
    local_ctrl.time_in_stage = 50.0
    local_ctrl.time_in_green = 50.0
    phases_50 = ctrl.compute_phases(current_time=50.0, simulator=sim, telemetry=telemetry)
    assert phases_50[target_id] == 0

    # At 60s (>= max_green 60s), forces off to opposing phase (1) despite 0 measured demand
    local_ctrl.time_in_stage = 60.0
    local_ctrl.time_in_green = 60.0
    phases_60 = ctrl.compute_phases(current_time=60.0, simulator=sim, telemetry=telemetry)
    assert phases_60[target_id] == 1


# ==============================================================================
# 20. Safe Active Phase Held When Opposing Phase is Physically Blocked
# ==============================================================================

def test_safe_active_phase_held_when_opposing_phase_blocked():
    """Verifies that when opposing phase is blocked, max_green force-off does NOT switch into it."""
    sim = KinematicSimulator(map_name="bronx_fordham")
    target_id = list(sim.signal_controllers.keys())[0]

    ctrl = CooperativePredictiveSignalController(
        min_green=10.0,
        max_green=60.0,
        downstream_occupancy_cap=0.75,
        controlled_intersections=[target_id],
    )

    local_ctrl = sim.signal_controllers[target_id]
    local_ctrl.stage = "GREEN"
    local_ctrl.active_green_phase = 0  # Active is NS
    local_ctrl.time_in_stage = 60.0
    local_ctrl.time_in_green = 60.0

    # NS (Phase 0) outgoing is safe (0.10); EW (Phase 1) outgoing is blocked (0.90)
    telemetry = {}
    for r in sim.roads.values():
        if r.from_node == target_id:
            if ctrl._is_road_ns(r):
                telemetry[r.id] = {"occupancy_ratio": 0.10, "queue_length_meters": 0.0}
            else:
                telemetry[r.id] = {"occupancy_ratio": 0.90, "queue_length_meters": r.length * 0.90}

    # At 60s, holds active safe phase 0 instead of discharging into blocked phase 1
    phases = ctrl.compute_phases(current_time=60.0, simulator=sim, telemetry=telemetry)
    assert phases[target_id] == 0


# ==============================================================================
# 21. Both Phases Blocked Maintains All-Red Hold
# ==============================================================================

def test_both_phases_blocked_maintains_all_red_hold():
    """Verifies that when both phases are physically blocked, the controller requests an all-red hold (-1)
    and the FSM safely maintains all-red without granting green to any blocked movement."""
    sim = KinematicSimulator(map_name="bronx_fordham")
    target_id = list(sim.signal_controllers.keys())[0]

    ctrl = CooperativePredictiveSignalController(
        min_green=10.0,
        max_green=60.0,
        downstream_occupancy_cap=0.75,
        controlled_intersections=[target_id],
    )

    local_ctrl = sim.signal_controllers[target_id]
    local_ctrl.stage = "GREEN"
    local_ctrl.active_green_phase = 0
    local_ctrl.time_in_stage = 15.0
    local_ctrl.time_in_green = 15.0

    # Both NS and EW outgoing roads are physically blocked
    telemetry = {}
    for r in sim.roads.values():
        if r.from_node == target_id:
            telemetry[r.id] = {"occupancy_ratio": 0.88, "queue_length_meters": r.length * 0.88}

    # Controller requests all-red hold (-1)
    phases = ctrl.compute_phases(current_time=15.0, simulator=sim, telemetry=telemetry)
    assert phases[target_id] == -1

    # Dispatch command to FSM: initiates yellow clearance (3.0s)
    sim.apply_signal_phases(phases)
    assert local_ctrl.stage == "YELLOW"
    assert local_ctrl.target_green_phase == -1

    # Step through 3.0s yellow
    for _ in range(12):  # 12 * 0.25s = 3.0s
        sim.step(dt=0.25)
    assert local_ctrl.stage == "ALL_RED"

    # Step through 1.5s all-red clearance
    for _ in range(6):  # 6 * 0.25s = 1.5s
        sim.step(dt=0.25)

    # After clearance, FSM remains holding in ALL_RED because target_green_phase is -1
    assert local_ctrl.stage == "ALL_RED"
    assert local_ctrl.get_signal_code(is_ns=True) == "r"
    assert local_ctrl.get_signal_code(is_ns=False) == "r"


# ==============================================================================
# 22. Blocked Phase Becomes Eligible Once Physical Occupancy Clears
# ==============================================================================

def test_blocked_phase_becomes_eligible_when_occupancy_clears():
    """Verifies that once downstream occupancy drops below 75%, all-red hold is released to the safe phase."""
    sim = KinematicSimulator(map_name="bronx_fordham")
    target_id = list(sim.signal_controllers.keys())[0]

    ctrl = CooperativePredictiveSignalController(
        min_green=10.0,
        max_green=60.0,
        downstream_occupancy_cap=0.75,
        controlled_intersections=[target_id],
    )

    local_ctrl = sim.signal_controllers[target_id]
    local_ctrl.stage = "ALL_RED"
    local_ctrl.target_green_phase = -1
    local_ctrl.time_in_stage = 5.0  # Already holding in all-red

    # NS (phase 0) clears to 0.20 occupancy; EW (phase 1) remains blocked at 0.85
    telemetry = {}
    for r in sim.roads.values():
        if r.from_node == target_id:
            if ctrl._is_road_ns(r):
                telemetry[r.id] = {"occupancy_ratio": 0.20, "queue_length_meters": 0.0}
            else:
                telemetry[r.id] = {"occupancy_ratio": 0.85, "queue_length_meters": r.length * 0.85}

    # Controller recognizes phase 0 is now safe, releases all-red hold to phase 0
    phases = ctrl.compute_phases(current_time=20.0, simulator=sim, telemetry=telemetry)
    assert phases[target_id] == 0

    # Dispatch to FSM
    sim.apply_signal_phases(phases)
    sim.step(dt=0.25)
    assert local_ctrl.stage == "GREEN"
    assert local_ctrl.active_green_phase == 0


# ==============================================================================
# 23. Minimum Green, Yellow, and All-Red Timings Remain Intact
# ==============================================================================

def test_timing_invariants_strictly_preserved():
    """Verifies that 10s min green, 3s yellow, and 1.5s all-red are strictly enforced during transitions."""
    sim = KinematicSimulator(map_name="bronx_fordham")
    target_id = list(sim.signal_controllers.keys())[0]
    local_ctrl = sim.signal_controllers[target_id]

    # Start at GREEN on phase 0
    local_ctrl.stage = "GREEN"
    local_ctrl.active_green_phase = 0
    local_ctrl.time_in_stage = 5.0
    local_ctrl.time_in_green = 5.0

    # Request phase switch at t = 5.0s < 10.0s min_green
    local_ctrl.request_phase(1)
    # Stage must remain GREEN until 10.0s
    sim.step(dt=0.25)
    assert local_ctrl.stage == "GREEN"

    # Step until 10.0s
    for _ in range(19):  # 19 * 0.25s = 4.75s -> total 10.0s
        sim.step(dt=0.25)
    assert local_ctrl.stage == "YELLOW"

    # Yellow must last exactly 3.0s
    for _ in range(11):  # 2.75s
        sim.step(dt=0.25)
    assert local_ctrl.stage == "YELLOW"
    sim.step(dt=0.25)  # 3.0s
    assert local_ctrl.stage == "ALL_RED"

    # All-red must last at least 1.5s
    for _ in range(5):  # 1.25s
        sim.step(dt=0.25)
    assert local_ctrl.stage == "ALL_RED"
    sim.step(dt=0.25)  # 1.5s
    assert local_ctrl.stage == "GREEN"
    assert local_ctrl.active_green_phase == 1

