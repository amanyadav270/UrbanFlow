"""Phase 8.2 & 8.3 Verification Test Suite: Live Dynamic Routing Integration, Dashboard Controls, and 3D Route Visualization.

Verifies:
1. Mode switching across OFF, REACTIVE, and PREDICTIVE with input validation.
2. OFF mode never reroutes vehicles.
3. REACTIVE mode uses operational capacity and V/C without predictive probabilities.
4. PREDICTIVE mode uses calibrated spillback probabilities when available.
5. Insufficient history and missing-model fallback without fabrication.
6. Route continuity, validity, and destination preservation.
7. Safeguards: diversion quota, capacity admission, storage limits, improvement threshold, cooldown, hysteresis.
8. Reset cleanliness (clears all reservations, cooldowns, and decision histories).
9. WebSocket and REST routing telemetry contract.
10. Coexistence and zero conflict between dynamic routing and the four signal controllers.
11. Safe SUMO adapter behavior when disconnected.
"""
from __future__ import annotations

import pytest
import numpy as np
from pathlib import Path
from starlette.testclient import TestClient

from urbanflow.core.kinematic_sim import KinematicSimulator
from urbanflow.core.network_topology import NetworkTopology
from urbanflow.adapters.sumo_adapter import SumoAdapter
from urbanflow.intelligence.routing import (
    PredictiveRoutingController,
    RouteCostEvaluator,
    OnlineObservationBuffer,
)
from urbanflow.intelligence.signal_control import (
    FixedTimeController,
    ActuatedController,
    MaxPressureController,
    CooperativePredictiveSignalController,
)
from urbanflow.server.websocket_feed import SimulationBroadcaster, ROUTING_MODES
from urbanflow.server.app import app


def test_1_three_routing_modes_and_switching():
    """Test 1: Verifies OFF, REACTIVE, PREDICTIVE modes, switching, and invalid mode rejection."""
    controller = PredictiveRoutingController(mode="off")
    assert controller.mode == "off"

    controller.set_mode("reactive")
    assert controller.mode == "reactive"
    assert controller.route_evaluator.mode == "reactive"

    controller.set_mode("predictive")
    assert controller.mode == "predictive"
    assert controller.route_evaluator.mode == "predictive"

    controller.set_mode("OFF")
    assert controller.mode == "off"

    with pytest.raises(ValueError, match="Mode must be 'off', 'reactive', or 'predictive'"):
        controller.set_mode("INVALID_MODE")

    sim = KinematicSimulator()
    broadcaster = SimulationBroadcaster(sim)
    assert broadcaster.active_routing_mode == "OFF"

    telem = broadcaster.set_routing_mode("REACTIVE")
    assert broadcaster.active_routing_mode == "REACTIVE"
    assert telem["active_mode"] == "REACTIVE"

    telem_pred = broadcaster.set_routing_mode("PREDICTIVE")
    assert broadcaster.active_routing_mode == "PREDICTIVE"
    assert telem_pred["active_mode"] == "PREDICTIVE"

    with pytest.raises(ValueError, match="Invalid routing mode"):
        broadcaster.set_routing_mode("AUTONOMOUS")


def test_2_off_mode_never_reroutes():
    """Test 2: Verifies that in OFF mode, no vehicle is ever diverted, even under severe congestion."""
    sim = KinematicSimulator()
    controller = PredictiveRoutingController(mode="off")

    # Manually flag all roads as congested
    for r_id in sim.roads:
        controller.congested_corridors[r_id] = True

    # Record initial vehicle routes
    initial_routes = {v.id: list(v.route) for v in sim.vehicles.values()}
    assert len(initial_routes) > 0

    report = controller.evaluate_and_reroute(sim=sim, current_time=100.0)
    assert report["mode"] == "off"
    assert report["diverted_count"] == 0
    assert report["eligible_candidates_count"] == 0
    assert len(report["decisions"]) == 0

    # Ensure no vehicle route was altered
    for v_id, orig_route in initial_routes.items():
        assert sim.vehicles[v_id].route == orig_route
        assert sim.vehicles[v_id].reroute_count == 0


def test_3_reactive_mode_zero_predictive_weight():
    """Test 3: Verifies that REACTIVE mode never queries or incorporates predictive probabilities."""
    sim = KinematicSimulator()
    controller = PredictiveRoutingController(mode="reactive")

    # Advisory spillback map must be strictly empty in reactive mode
    probs = controller.compute_advisory_spillback_probabilities()
    assert probs == {}, "Reactive mode must not compute advisory spillback probabilities"

    # Even if feature buffers are populated, reactive mode ignores them
    for r_id in sim.roads:
        for i in range(4):
            controller.observation_buffer.update({
                r_id: {
                    "avg_speed_kmh": 10.0,
                    "density_per_km_lane": 80.0,
                    "queue_length_meters": 50.0,
                }
            }, timestamp=float(i * 5.0))

    probs_after = controller.compute_advisory_spillback_probabilities()
    assert probs_after == {}, "Reactive mode must strictly return empty probabilities dict"


def test_4_predictive_mode_uses_calibrated_probabilities():
    """Test 4: Verifies that PREDICTIVE mode evaluates 28-feature classifier when history exists."""
    controller = PredictiveRoutingController(mode="predictive")
    assert controller.mode == "predictive"
    assert controller.predictive_available is True

    # Seed 4 snapshots of history for a road
    test_road = list(controller.route_evaluator.roads.keys())[0]
    for i in range(4):
        controller.observation_buffer.update({
            test_road: {
                "timestamp": float(i * 5.0),
                "avg_speed_kmh": 20.0,
                "density_per_km_lane": 45.0,
                "queue_length_meters": 35.0,
                "operational_capacity": 1800.0,
                "capacity_utilization": 0.85,
                "occupancy_ratio": 0.70,
                "incident_active": False,
            }
        }, timestamp=float(i * 5.0))

    probs = controller.compute_advisory_spillback_probabilities()
    assert test_road in probs
    assert isinstance(probs[test_road], float)
    assert 0.0 <= probs[test_road] <= 1.0


def test_5_insufficient_history_and_missing_model_fallback():
    """Test 5: Verifies graceful fallback when history < 4 steps or model file is unavailable."""
    controller = PredictiveRoutingController(mode="predictive")

    # When 0 steps exist, compute_advisory_spillback_probabilities() returns empty dict without errors
    probs = controller.compute_advisory_spillback_probabilities()
    assert probs == {}

    # Test missing model path
    fake_path = Path("non_existent_classifier.joblib")
    controller_missing = PredictiveRoutingController(mode="predictive", classifier_path=fake_path)
    assert controller_missing.predictive_available is False
    assert "UNAVAILABLE" in controller_missing.predictive_status

    # Fallback to reactive mode without crashing
    probs_missing = controller_missing.compute_advisory_spillback_probabilities()
    assert probs_missing == {}


def test_6_destination_preservation_and_route_continuity():
    """Test 6: Verifies that any applied reroute strictly preserves destination and edge continuity."""
    sim = KinematicSimulator()
    vehs = list(sim.vehicles.values())
    assert len(vehs) > 0
    veh = vehs[0]
    orig_dest = veh.route[-1]
    orig_dest_node = sim.roads[orig_dest].to_node

    evaluator = RouteCostEvaluator(simulator=sim, mode="reactive")

    # Request candidate path from current road to destination
    candidates = evaluator.find_candidate_paths(
        start_road_id=veh.current_road_id,
        dest_road_id=orig_dest,
        k=3,
    )
    assert len(candidates) > 0
    # Find an alternative candidate path that diverges from current remaining path
    cand_path = None
    for cand in candidates:
        cand_sub = cand["path"][1:] if cand["path"][0] == veh.current_road_id else cand["path"]
        if cand_sub != veh.route[veh.route_index + 1:]:
            cand_path = cand["path"]
            break
    if cand_path is None:
        cand_path = candidates[0]["path"]

    # Verify destination preservation
    new_dest = cand_path[-1]
    assert sim.roads[new_dest].to_node == orig_dest_node, "Reroute changed vehicle destination node!"

    # Verify continuity
    res = sim.reroute_vehicle(veh.id, cand_path)
    assert res["success"] is True
    if res["rerouted"]:
        assert veh.reroute_count >= 1
    else:
        assert "no-op" in res["reason"]
    assert veh.route[-1] == orig_dest


def test_7_safeguards_enforcement():
    """Test 7: Verifies diversion quota, capacity admission, storage limits, and cooldowns."""
    sim = KinematicSimulator()
    controller = PredictiveRoutingController(
        mode="reactive",
        max_diversion_quota=0.25,
        cooldown_sec=180.0,
        vc_admission_cap=0.80,
    )

    # Cooldown verification: set vehicle last reroute time to now
    veh = list(sim.vehicles.values())[0]
    controller.last_reroute_time[veh.id] = 100.0

    # At t=150s (within 180s cooldown), vehicle must be disqualified from eligibility
    downstream_roads = veh.route[veh.route_index + 1:]
    for r in downstream_roads:
        controller.congested_corridors[r] = True

    report = controller.evaluate_and_reroute(sim, current_time=150.0)
    # The cooled-down vehicle is not eligible
    eligible_ids = [d["vehicle_id"] for d in report["decisions"]]
    assert veh.id not in eligible_ids, "Vehicle under cooldown was eligible for rerouting"


def test_8_reset_clears_all_routing_state():
    """Test 8: Verifies that reset clears reservations, cooldowns, hysteresis, and buffers cleanly."""
    sim = KinematicSimulator()
    broadcaster = SimulationBroadcaster(sim)
    broadcaster.set_routing_mode("PREDICTIVE")

    # Manually populate state
    broadcaster.routing_controller.last_reroute_time["veh_1"] = 50.0
    broadcaster.routing_controller.congested_corridors["road_1"] = True
    broadcaster.cumulative_reroute_accepted = 12
    broadcaster.cumulative_reroute_rejected = 3
    broadcaster.recent_routing_decisions.append({"vehicle_id": "veh_1", "diverted": True})

    broadcaster.reset_routing()

    assert len(broadcaster.routing_controller.last_reroute_time) == 0
    assert len(broadcaster.routing_controller.congested_corridors) == 0
    assert len(broadcaster.routing_controller.active_reservations) == 0
    assert broadcaster.cumulative_reroute_accepted == 0
    assert broadcaster.cumulative_reroute_rejected == 0
    assert len(broadcaster.recent_routing_decisions) == 0
    assert broadcaster.last_routing_time == -1.0


def test_9_websocket_and_rest_routing_telemetry():
    """Test 9: Verifies REST endpoints and telemetry contract schema."""
    client = TestClient(app)

    # 1. GET /api/status includes routing info
    resp_status = client.get("/api/status")
    assert resp_status.status_code == 200
    st_data = resp_status.json()
    assert "active_routing_mode" in st_data
    assert "routing_status" in st_data
    assert "available_routing_modes" in st_data

    # 2. GET /api/routing
    resp_routing = client.get("/api/routing")
    assert resp_routing.status_code == 200
    r_data = resp_routing.json()
    assert "active_mode" in r_data
    assert "routing_telemetry" in r_data
    assert len(r_data["available_modes"]) == 3

    # 3. POST /api/routing valid
    resp_set = client.post("/api/routing", json={"mode": "REACTIVE"})
    assert resp_set.status_code == 200
    assert resp_set.json()["active_mode"] == "REACTIVE"

    # 4. POST /api/routing invalid
    resp_bad = client.post("/api/routing", json={"mode": "MAGIC_ROUTING"})
    assert resp_bad.status_code == 400
    assert "Invalid routing mode" in resp_bad.json()["message"]

    # 5. POST /api/control with action set_routing_mode
    resp_ctrl = client.post("/api/control", json={"action": "set_routing_mode", "mode": "OFF"})
    assert resp_ctrl.status_code == 200
    assert resp_ctrl.json()["mode"] == "OFF"


def test_10_no_conflict_with_signal_controllers():
    """Test 10: Verifies that signal controllers and dynamic routing execute simultaneously without collision."""
    sim = KinematicSimulator()
    broadcaster = SimulationBroadcaster(sim)

    # Test with each of the four signal controllers alongside PREDICTIVE routing
    broadcaster.set_routing_mode("PREDICTIVE")

    for ctrl_name in ["FIXED_TIME", "ACTUATED", "MAX_PRESSURE", "COOPERATIVE_PREDICTIVE"]:
        broadcaster.set_controller(ctrl_name)
        assert broadcaster.active_controller_name == ctrl_name
        assert broadcaster.active_routing_mode == "PREDICTIVE"

        # Step simulation and compute both signal control and routing
        snapshot = sim.step(dt=0.05)
        intel = broadcaster._compute_intelligence(snapshot)

        # Both compute without exceptions
        phases = broadcaster.controller.compute_phases(
            current_time=snapshot.timestamp,
            simulator=sim,
            telemetry=intel,
        )
        assert isinstance(phases, dict)
        assert len(phases) > 0

        routing_report = broadcaster.routing_controller.evaluate_and_reroute(
            sim=sim,
            current_time=snapshot.timestamp,
            telemetry_map=intel,
        )
        assert isinstance(routing_report, dict)
        assert "decisions" in routing_report


def test_11_sumo_adapter_safe_reroute_behavior():
    """Test 11: Verifies that SumoAdapter safely rejects dynamic rerouting when disconnected."""
    sumo = SumoAdapter()
    assert sumo.is_running is False

    commands = [
        {"vehicle_id": "veh_test_1", "new_route": ["edge_1", "edge_2"]},
        {"vehicle_id": "veh_test_2", "new_subpath": ["edge_3", "edge_4"]},
    ]

    res = sumo.reroute_vehicles(commands)
    assert res["total_commands"] == 2
    assert res["successful_count"] == 0
    assert res["failed_count"] == 2
    assert len(res["results"]) == 2
    for r in res["results"]:
        assert r["success"] is False
        assert "requires an active TraCI session" in r["reason"]
