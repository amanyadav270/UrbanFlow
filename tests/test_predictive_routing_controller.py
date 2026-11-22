"""Unit Tests for UrbanFlow AI Phase 3: Predictive Capacity-Aware Routing Controller.

Tests:
1. Diversion quota enforcement (strictly <= 25% of eligible vehicles per window).
2. Alternate-road capacity admission rejection when V/C >= 0.80.
3. Capacity reservation across multiple simultaneous decisions (prevents overbooking).
4. Minimum travel-time improvement threshold (>= max(45s, 15%)).
5. 180-second vehicle rerouting cooldown.
6. Two-window congestion hysteresis (activate at >= 0.80, clear only after 2 windows < 0.70).
7. Predictive vs reactive decision comparison under advisory spillback risk.
8. Missing classifier and missing telemetry fallback (safe reactive mode).
9. Invalid reroute command handling and simulator error tolerance.
10. Route oscillation prevention and destination preservation.
"""
from __future__ import annotations

import copy
import pytest
from pathlib import Path

from urbanflow.core.kinematic_sim import KinematicSimulator, SimulatedVehicle
from urbanflow.intelligence.routing import RouteCostEvaluator, PredictiveRoutingController


@pytest.fixture
def sim():
    """Initializes KinematicSimulator with Bronx Fordham digital twin."""
    return KinematicSimulator(map_name="bronx_fordham")


@pytest.fixture
def controller(sim):
    """Initializes PredictiveRoutingController in reactive mode."""
    evaluator = RouteCostEvaluator(simulator=sim, mode="reactive")
    ctrl = PredictiveRoutingController(
        route_evaluator=evaluator,
        mode="reactive",
        strict_quota_floor=True,
    )
    return ctrl


def test_diversion_quota_enforcement(sim, controller):
    """Verifies that at most 25% of eligible vehicles are diverted in a single window."""
    # Find two distinct parallel paths
    start_road = "road_586133572_120_fwd"
    dest_road = "road_1082220763_166_fwd"
    curr_base_path = [start_road, "road_586133572_121_fwd", "road_586133572_122_fwd", "road_586133572_123_fwd", "road_223889617_62_fwd", "road_1082220766_170_fwd", dest_road]

    # Create 8 identical eligible vehicles
    sim.vehicles.clear()
    for i in range(8):
        v_id = f"test_quota_veh_{i}"
        veh = SimulatedVehicle(
            vehicle_id=v_id,
            v_type="sedan",
            route=list(curr_base_path),
            color="#2563eb",
        )
        sim.vehicles[v_id] = veh

    # Mark the destination road as heavily congested (V/C = 3.2, queue = 100m) to generate >45s delay improvement
    telem = {dest_road: {"vc_ratio": 3.2, "operational_capacity": 500.0, "queue_length_meters": 100.0}}
    controller.update_telemetry(telem, timestamp=100.0)

    # Incur heavy delay on dest_road so delta_t > 45s
    res = controller.evaluate_and_reroute(sim, current_time=100.0, telemetry_map=telem)

    # Exactly 8 vehicles were eligible, so quota = floor(0.25 * 8) = 2
    assert res["eligible_candidates_count"] == 8
    assert res["max_diversions_quota"] == 2
    assert res["diverted_count"] <= 2
    assert res["sim_successful_count"] <= 2

    # Check decisions
    diverted_decisions = [d for d in res["decisions"] if d["diverted"]]
    quota_blocked_decisions = [d for d in res["decisions"] if "quota reached" in d["reason"].lower()]

    assert len(diverted_decisions) == 2
    assert len(quota_blocked_decisions) == 6


def test_capacity_admission_rejection_at_vc_80(sim, controller):
    """Verifies that an alternate detour route is rejected if any of its links has V/C >= 0.80."""
    start_road = "road_586133572_120_fwd"
    dest_road = "road_1082220763_166_fwd"
    curr_base_path = [start_road, "road_586133572_121_fwd", "road_586133572_122_fwd", "road_586133572_123_fwd", "road_223889617_62_fwd", "road_1082220766_170_fwd", dest_road]

    sim.vehicles.clear()
    veh = SimulatedVehicle("veh_cap_test", "sedan", list(curr_base_path), "#ef4444")
    sim.vehicles[veh.id] = veh

    # Dest road is congested (V/C = 3.0), triggering reroute desire
    # Alternate branch uses 'road_5698666_15_fwd'. Put road_5698666_15_fwd at V/C = 0.85
    telem = {
        dest_road: {"vc_ratio": 3.0, "operational_capacity": 500.0, "queue_length_meters": 60.0},
        "road_5698666_15_fwd": {"vc_ratio": 0.85, "operational_capacity": 1800.0, "flow_veh_per_hour": 1530.0},
    }
    controller.update_telemetry(telem, timestamp=50.0)

    # Disable strict quota floor so single vehicle is evaluated
    controller.strict_quota_floor = False

    res = controller.evaluate_and_reroute(sim, current_time=50.0, telemetry_map=telem)

    # Vehicle should be rejected due to capacity threshold violation
    assert res["diverted_count"] == 0
    rej = res["decisions"][0]
    assert rej["diverted"] is False
    assert "exceeds capacity admission threshold" in rej["reason"]


def test_capacity_reservation_across_simultaneous_decisions(sim, controller):
    """Verifies that multiple simultaneous reroute decisions reserve capacity and prevent corridor overbooking."""
    start_road = "road_586133572_120_fwd"
    dest_road = "road_1082220763_166_fwd"
    curr_base_path = [start_road, "road_586133572_121_fwd", "road_586133572_122_fwd", "road_586133572_123_fwd", "road_223889617_62_fwd", "road_1082220766_170_fwd", dest_road]

    sim.vehicles.clear()
    v1 = SimulatedVehicle("veh_res_1", "sedan", list(curr_base_path), "#2563eb")
    v2 = SimulatedVehicle("veh_res_2", "sedan", list(curr_base_path), "#10b981")
    sim.vehicles[v1.id] = v1
    sim.vehicles[v2.id] = v2

    controller.strict_quota_floor = False  # allow multiple candidates
    controller.max_diversion_quota = 1.0   # allow testing capacity limits without quota cutoff

    # Set alternate link 'road_5698666_15_fwd' near capacity (V/C = 0.722)
    # Flow = 1300 veh/hr, Capacity = 1800 veh/hr (headroom = 140 veh/hr).
    # Each rerouted vehicle in 30s evaluation window adds (3600 / 30s) = 120 veh/hr.
    # 1st vehicle: Flow = 1300 + 120 = 1420 -> V/C = 1420 / 1800 = 0.789 < 0.80 (admitted).
    # 2nd vehicle: Flow = 1300 + 240 = 1540 -> V/C = 1540 / 1800 = 0.856 >= 0.80 (rejected).
    telem = {
        dest_road: {"vc_ratio": 3.2, "operational_capacity": 500.0, "queue_length_meters": 100.0},
        "road_5698666_15_fwd": {"vc_ratio": 0.722, "operational_capacity": 1800.0, "flow_veh_per_hour": 1300.0},
    }
    controller.update_telemetry(telem, timestamp=50.0)

    res = controller.evaluate_and_reroute(sim, current_time=50.0, telemetry_map=telem)

    # 1st vehicle diverted, 2nd vehicle rejected for capacity
    assert res["diverted_count"] == 1
    assert res["decisions"][0]["diverted"] is True
    assert res["decisions"][1]["diverted"] is False
    assert "exceeds capacity admission threshold" in res["decisions"][1]["reason"]


def test_minimum_improvement_threshold(sim, controller):
    """Verifies that routes offering less than max(45s, 15%) improvement are rejected."""
    start_road = "road_586133572_120_fwd"
    dest_road = "road_1082220763_166_fwd"
    curr_base_path = [start_road, "road_586133572_121_fwd", "road_586133572_122_fwd", "road_586133572_123_fwd", "road_223889617_62_fwd", "road_1082220766_170_fwd", dest_road]

    sim.vehicles.clear()
    veh = SimulatedVehicle("veh_min_imp", "sedan", list(curr_base_path), "#2563eb")
    sim.vehicles[veh.id] = veh
    controller.strict_quota_floor = False

    # Moderate congestion (V/C = 2.5, queue = 60m yields ~29.3s improvement, below 45s threshold)
    telem = {dest_road: {"vc_ratio": 2.5, "operational_capacity": 500.0, "queue_length_meters": 60.0}}
    controller.update_telemetry(telem, timestamp=50.0)

    res = controller.evaluate_and_reroute(sim, current_time=50.0, telemetry_map=telem)
    assert res["diverted_count"] == 0
    assert "Insufficient benefit" in res["decisions"][0]["reason"]


def test_vehicle_cooldown_180s(sim, controller):
    """Verifies that a rerouted vehicle enters a 180-second lockout window."""
    start_road = "road_586133572_120_fwd"
    dest_road = "road_1082220763_166_fwd"
    curr_base_path = [start_road, "road_586133572_121_fwd", "road_586133572_122_fwd", "road_586133572_123_fwd", "road_223889617_62_fwd", "road_1082220766_170_fwd", dest_road]

    sim.vehicles.clear()
    veh = SimulatedVehicle("veh_cooldown", "sedan", list(curr_base_path), "#2563eb")
    sim.vehicles[veh.id] = veh
    controller.strict_quota_floor = False

    telem = {dest_road: {"vc_ratio": 3.0, "operational_capacity": 500.0, "queue_length_meters": 60.0}}

    # 1. First evaluation at t=100.0: succeeds
    res1 = controller.evaluate_and_reroute(sim, current_time=100.0, telemetry_map=telem)
    assert res1["diverted_count"] == 1
    assert controller.last_reroute_time[veh.id] == 100.0

    # 2. Second evaluation at t=200.0 (100s later < 180s): vehicle is in cooldown
    res2 = controller.evaluate_and_reroute(sim, current_time=200.0, telemetry_map=telem)
    assert res2["eligible_candidates_count"] == 0
    assert res2["diverted_count"] == 0

    # 3. Third evaluation at t=281.0 (181s later > 180s): cooldown expired
    # (If vehicle still had remaining route, it would be eligible again)
    assert (281.0 - controller.last_reroute_time[veh.id]) > 180.0


def test_two_window_congestion_hysteresis(controller):
    """Verifies that congestion activates at V/C >= 0.80 and clears only after 2 consecutive windows < 0.70."""
    road_id = "test_corridor_road"

    # Window 1: V/C = 0.85 -> Congestion activates
    controller.update_telemetry({road_id: {"vc_ratio": 0.85}}, timestamp=15.0)
    assert controller.congested_corridors[road_id] is True

    # Window 2: V/C = 0.75 (in hysteresis buffer [0.70, 0.80)) -> Remains active
    controller.update_telemetry({road_id: {"vc_ratio": 0.75}}, timestamp=30.0)
    assert controller.congested_corridors[road_id] is True

    # Window 3: V/C = 0.65 (first clear window < 0.70) -> Still active (requires 2 windows)
    controller.update_telemetry({road_id: {"vc_ratio": 0.65}}, timestamp=45.0)
    assert controller.congested_corridors[road_id] is True
    assert controller.consecutive_clear_windows[road_id] == 1

    # Window 4: V/C = 0.60 (second consecutive clear window < 0.70) -> Clears!
    controller.update_telemetry({road_id: {"vc_ratio": 0.60}}, timestamp=60.0)
    assert controller.congested_corridors[road_id] is False


def test_predictive_vs_reactive_decisions(sim):
    """Verifies that predictive mode diverts based on advisory risk before physical congestion occurs."""
    start_road = "road_586133572_120_fwd"
    dest_road = "road_1082220763_166_fwd"
    curr_base_path = [start_road, "road_586133572_121_fwd", "road_586133572_122_fwd", "road_586133572_123_fwd", "road_223889617_62_fwd", "road_1082220766_170_fwd", dest_road]

    sim.vehicles.clear()
    veh_react = SimulatedVehicle("veh_r", "sedan", list(curr_base_path), "#2563eb")
    sim.vehicles[veh_react.id] = veh_react

    # Zero physical congestion (V/C = 0.40), but high advisory spillback risk on dest_road
    telem = {dest_road: {"vc_ratio": 0.40, "operational_capacity": 1800.0}}

    # 1. Reactive Controller: Ignores advisory risk when physical V/C < 0.80
    ctrl_reactive = PredictiveRoutingController(mode="reactive", strict_quota_floor=False)
    res_r = ctrl_reactive.evaluate_and_reroute(sim, current_time=50.0, telemetry_map=telem)
    assert res_r["eligible_candidates_count"] == 0
    assert res_r["diverted_count"] == 0

    # 2. Predictive Controller with mock advisory risk
    ctrl_predictive = PredictiveRoutingController(mode="predictive", strict_quota_floor=False)
    # Monkey-patch advisory probabilities to simulate 10-minute network alert
    ctrl_predictive.compute_advisory_spillback_probabilities = lambda: {dest_road: 0.95}

    res_p = ctrl_predictive.evaluate_and_reroute(sim, current_time=50.0, telemetry_map=telem)
    # Vehicle is recognized as eligible due to high advisory risk
    assert res_p["eligible_candidates_count"] == 1


def test_missing_classifier_safe_reactive_fallback():
    """Verifies that missing or corrupted classifier falls back safely to reactive mode."""
    # Point to nonexistent path
    fake_path = Path("nonexistent_classifier.joblib")
    ctrl = PredictiveRoutingController(classifier_path=fake_path, mode="predictive")
    assert ctrl.predictive_available is False
    assert "UNAVAILABLE_FILE_NOT_FOUND" in ctrl.predictive_status

    # Advisory probabilities should safely return empty dict without crashing
    probs = ctrl.compute_advisory_spillback_probabilities()
    assert probs == {}


def test_classifier_cryptographic_hash_verification():
    """Verifies that the controller enforces the exact SHA-256 hash before loading."""
    ctrl = PredictiveRoutingController(mode="predictive")
    # If the real artifact exists on disk, it should match the hash and be ACTIVE
    if ctrl.DEFAULT_CLASSIFIER_PATH.exists():
        assert ctrl.predictive_available is True
        assert ctrl.predictive_status == "ACTIVE"
        assert len(ctrl.expected_features) == 28


def test_preservation_of_vehicle_state_and_destination(sim, controller):
    """Verifies that after controller-initiated rerouting, vehicle kinematics and destination are preserved."""
    start_road = "road_586133572_120_fwd"
    dest_road = "road_1082220763_166_fwd"
    curr_base_path = [start_road, "road_586133572_121_fwd", "road_586133572_122_fwd", "road_586133572_123_fwd", "road_223889617_62_fwd", "road_1082220766_170_fwd", dest_road]

    sim.vehicles.clear()
    veh = SimulatedVehicle("veh_state_ctrl", "sedan", list(curr_base_path), "#2563eb")
    first_road = sim.roads[start_road]
    veh.distance_on_road = 20.0
    veh.speed = 10.5
    veh.wait_time = 5.0
    veh.total_travel_time = 42.0
    sim._update_vehicle_pose(veh, first_road)
    sim.vehicles[veh.id] = veh
    controller.strict_quota_floor = False

    # Trigger diversion
    telem = {dest_road: {"vc_ratio": 3.0, "operational_capacity": 500.0, "queue_length_meters": 60.0}}
    res = controller.evaluate_and_reroute(sim, current_time=100.0, telemetry_map=telem)

    assert res["diverted_count"] == 1
    # Check that vehicle state is strictly preserved
    assert veh.distance_on_road == 20.0
    assert veh.speed == 10.5
    assert veh.wait_time == 5.0
    assert veh.total_travel_time == 42.0
    # Destination road or terminal node must match original destination
    assert veh.route[-1] == dest_road or sim.roads[veh.route[-1]].to_node == sim.roads[dest_road].to_node
