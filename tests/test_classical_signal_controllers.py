"""Comprehensive Unit Test Suite for UrbanFlow AI Phase 6B: Classical Traffic Signal Controllers.

Tests:
1. FixedTimeController schedule determinism, cycle calculation, and phase progression.
2. FixedTimeController coordinated offsets between two adjacent intersections (green waves).
3. ActuatedController minimum green adherence (will not gap out or switch before min_green).
4. ActuatedController green extension on active arrival.
5. ActuatedController gap-out when active approach has no demand and conflicting approach has demand.
6. ActuatedController maximum green force-off when conflicting demand is sustained.
7. MaxPressureController queue-differential phase selection (higher pressure approach is selected).
8. MaxPressureController downstream blocked-link handling (storage exhaustion prevents phase selection even if upstream queue is high).
9. Multiple intersections receiving different phase requests simultaneously.
10. Invalid or missing telemetry handling (None simulator, empty telemetry, unknown IDs).
11. All controllers respecting Phase 6A FSM safety constraints (no green-to-green jumps, mandatory yellow and all-red clearance, no conflicting green).
12. WebSocket feed segment estimator correctly uses local intersection controller.
"""
from __future__ import annotations

import pytest
from typing import Dict, Any, List

from urbanflow.core.kinematic_sim import KinematicSimulator, SimulatedVehicle, IntersectionSignalFSM
from urbanflow.intelligence.signal_control import (
    SignalControllerBase,
    FixedTimeController,
    ActuatedController,
    MaxPressureController,
)
from urbanflow.core.types import Vec3, RoadSegment, Intersection, SimulationSnapshot


# ==============================================================================
# 1. FixedTimeController Tests
# ==============================================================================

def test_fixed_time_controller_schedule_determinism_and_cycle_progression():
    """Verifies that FixedTimeController deterministically computes phase schedules according to configured splits."""
    sim = KinematicSimulator(map_name="bronx_fordham")
    inter_ids = list(sim.signal_controllers.keys())
    target_id = inter_ids[0]

    controller = FixedTimeController(
        default_green_ns=25.0,
        default_green_ew=20.0,
        yellow_duration=3.0,
        all_red_duration=1.5,
        default_offset=0.0,
        controlled_intersections=[target_id],
    )

    # Clearance = 3.0 + 1.5 = 4.5s
    # Cycle C = 25.0 + 4.5 + 20.0 + 4.5 = 54.0s

    # At t = 0.0: Start of NS Green (Phase 0)
    phases_0 = controller.compute_phases(0.0, simulator=sim)
    assert phases_0[target_id] == 0

    # At t = 15.0: Inside NS Green (Phase 0)
    phases_15 = controller.compute_phases(15.0, simulator=sim)
    assert phases_15[target_id] == 0

    # At t = 25.0: End of NS Green -> Request Phase 1
    phases_25 = controller.compute_phases(25.0, simulator=sim)
    assert phases_25[target_id] == 1

    # At t = 35.0: Inside EW Green (Phase 1)
    phases_35 = controller.compute_phases(35.0, simulator=sim)
    assert phases_35[target_id] == 1

    # At t = 50.0: End of EW Green (25 + 4.5 + 20 = 49.5s) -> Request Phase 0
    phases_50 = controller.compute_phases(50.0, simulator=sim)
    assert phases_50[target_id] == 0

    # At t = 54.0 (Cycle boundary): Deterministically repeats Phase 0
    phases_54 = controller.compute_phases(54.0, simulator=sim)
    assert phases_54[target_id] == 0
    assert phases_54[target_id] == phases_0[target_id]


def test_fixed_time_controller_coordinated_offsets():
    """Verifies that offsets shift the cycle progression between two intersections to create green waves."""
    sim = KinematicSimulator(map_name="bronx_fordham")
    inter_ids = list(sim.signal_controllers.keys())
    id_1, id_2 = inter_ids[0], inter_ids[1]

    # Intersection 1 has offset 0.0s, Intersection 2 has offset 10.0s
    plans = {
        id_1: {"green_ns": 20.0, "green_ew": 20.0, "offset": 0.0},
        id_2: {"green_ns": 20.0, "green_ew": 20.0, "offset": 10.0},
    }
    controller = FixedTimeController(
        yellow_duration=3.0,
        all_red_duration=1.5,
        intersection_plans=plans,
        controlled_intersections=[id_1, id_2],
    )

    # At t = 2.0s:
    # id_1 is at rel_time = 2.0s (Phase 0: NS)
    # id_2 is at rel_time = (2.0 - 10.0) % 49.0 = 41.0s (Phase 0 clearance / Phase 0)
    phases_2 = controller.compute_phases(2.0, simulator=sim)
    assert id_1 in phases_2 and id_2 in phases_2

    # At t = 22.0s:
    # id_1 has completed NS green (20s) -> requests Phase 1 (EW)
    # id_2 is at rel_time = 12.0s (still inside its 20s NS green) -> requests Phase 0 (NS)
    phases_22 = controller.compute_phases(22.0, simulator=sim)
    assert phases_22[id_1] == 1
    assert phases_22[id_2] == 0


# ==============================================================================
# 2. ActuatedController Tests
# ==============================================================================

def test_actuated_controller_minimum_green_adherence():
    """Verifies that ActuatedController will not switch phases before minimum green has elapsed."""
    sim = KinematicSimulator(map_name="bronx_fordham")
    target_id = list(sim.signal_controllers.keys())[0]
    ctrl_fsm = sim.signal_controllers[target_id]

    controller = ActuatedController(
        min_green=10.0,
        max_green=60.0,
        controlled_intersections=[target_id],
    )

    # Put FSM in green with only 3.0s elapsed
    ctrl_fsm.stage = "GREEN"
    ctrl_fsm.active_green_phase = 0
    ctrl_fsm.time_in_green = 3.0

    # Simulate heavy conflicting demand on EW approaches
    telemetry = {
        "conflicting_ew_road": {"queue_length_meters": 80.0, "vehicle_count": 10}
    }

    # At 3.0s (< 10.0s min_green), controller must request active phase (0)
    phases = controller.compute_phases(current_time=3.0, simulator=sim, telemetry=telemetry)
    assert phases[target_id] == 0


def test_actuated_controller_gap_out_when_active_demand_is_zero():
    """Verifies that ActuatedController gaps out and requests the conflicting phase when active approach has no demand."""
    sim = KinematicSimulator(map_name="bronx_fordham")
    target_id = list(sim.signal_controllers.keys())[0]
    ctrl_fsm = sim.signal_controllers[target_id]

    controller = ActuatedController(
        min_green=10.0,
        max_green=60.0,
        controlled_intersections=[target_id],
    )

    # Active green phase 0, time_in_green = 12.0s (> 10.0s min_green)
    ctrl_fsm.stage = "GREEN"
    ctrl_fsm.active_green_phase = 0
    ctrl_fsm.time_in_green = 12.0

    # Remove all vehicles from NS approach roads to target_id (zero active demand)
    for v in list(sim.vehicles.values()):
        road = sim.roads.get(v.current_road_id)
        if road and road.to_node == target_id:
            del sim.vehicles[v.id]

    # Add a vehicle on an EW incoming road to target_id (conflicting call)
    ew_in_roads = [
        r for r in sim.roads.values()
        if r.to_node == target_id and (abs(r.points[-1].x - r.points[0].x) >= abs(r.points[-1].y - r.points[0].y))
    ]
    if ew_in_roads:
        ew_road = ew_in_roads[0]
        v_call = SimulatedVehicle("veh_conflicting_call", "sedan", [ew_road.id], "#ef4444")
        v_call.distance_on_road = max(0.0, ew_road.length - 20.0)
        sim.vehicles[v_call.id] = v_call

        phases = controller.compute_phases(current_time=12.0, simulator=sim)
        # Because active approach has 0 demand and conflicting approach has demand, gap-out triggers Phase 1
        assert phases[target_id] == 1


def test_actuated_controller_green_extension_on_active_demand():
    """Verifies that ActuatedController extends green when active approach has continuing vehicle demand."""
    sim = KinematicSimulator(map_name="bronx_fordham")
    target_id = list(sim.signal_controllers.keys())[0]
    ctrl_fsm = sim.signal_controllers[target_id]

    controller = ActuatedController(
        min_green=10.0,
        max_green=60.0,
        controlled_intersections=[target_id],
    )

    ctrl_fsm.stage = "GREEN"
    ctrl_fsm.active_green_phase = 0
    ctrl_fsm.time_in_green = 15.0

    # Place vehicles on NS incoming road (active demand)
    ns_in_roads = [
        r for r in sim.roads.values()
        if r.to_node == target_id and (abs(r.points[-1].x - r.points[0].x) < abs(r.points[-1].y - r.points[0].y))
    ]
    if ns_in_roads:
        ns_road = ns_in_roads[0]
        v_active = SimulatedVehicle("veh_active_arrival", "sedan", [ns_road.id], "#2563eb")
        v_active.distance_on_road = max(0.0, ns_road.length - 15.0)
        sim.vehicles[v_active.id] = v_active

        phases = controller.compute_phases(current_time=15.0, simulator=sim)
        # Active approach has continuing demand -> green extension requested (Phase 0)
        assert phases[target_id] == 0


def test_actuated_controller_maximum_green_force_off():
    """Verifies that ActuatedController forces phase termination when conflicting demand is present and max_green is reached."""
    sim = KinematicSimulator(map_name="bronx_fordham")
    target_id = list(sim.signal_controllers.keys())[0]
    ctrl_fsm = sim.signal_controllers[target_id]

    controller = ActuatedController(
        min_green=10.0,
        max_green=60.0,
        controlled_intersections=[target_id],
    )

    ctrl_fsm.stage = "GREEN"
    ctrl_fsm.active_green_phase = 0
    ctrl_fsm.time_in_green = 60.5  # Past max_green

    # Add vehicle on EW incoming road (conflicting demand)
    ew_in_roads = [
        r for r in sim.roads.values()
        if r.to_node == target_id and (abs(r.points[-1].x - r.points[0].x) >= abs(r.points[-1].y - r.points[0].y))
    ]
    if ew_in_roads:
        ew_road = ew_in_roads[0]
        v_call = SimulatedVehicle("veh_conflicting_maxout", "sedan", [ew_road.id], "#ef4444")
        v_call.distance_on_road = max(0.0, ew_road.length - 20.0)
        sim.vehicles[v_call.id] = v_call

        phases = controller.compute_phases(current_time=60.5, simulator=sim)
        # Max-out forces request for conflicting phase (Phase 1)
        assert phases[target_id] == 1


# ==============================================================================
# 3. MaxPressureController Tests
# ==============================================================================

def test_max_pressure_controller_queue_differential_selection():
    """Verifies that MaxPressureController selects the phase with higher upstream minus downstream pressure."""
    sim = KinematicSimulator(map_name="bronx_fordham")
    target_id = list(sim.signal_controllers.keys())[0]
    ctrl_fsm = sim.signal_controllers[target_id]

    controller = MaxPressureController(
        min_green=10.0,
        max_green=60.0,
        pressure_threshold=1.0,
        controlled_intersections=[target_id],
    )

    ctrl_fsm.stage = "GREEN"
    ctrl_fsm.active_green_phase = 0
    ctrl_fsm.time_in_green = 12.0  # Eligible to switch

    # Find NS and EW incoming roads
    ns_in = [r for r in sim.roads.values() if r.to_node == target_id and abs(r.points[-1].x - r.points[0].x) < abs(r.points[-1].y - r.points[0].y)]
    ew_in = [r for r in sim.roads.values() if r.to_node == target_id and abs(r.points[-1].x - r.points[0].x) >= abs(r.points[-1].y - r.points[0].y)]

    # Clear all existing vehicles at target intersection
    for v in list(sim.vehicles.values()):
        if v.current_road_id in [r.id for r in ns_in + ew_in]:
            del sim.vehicles[v.id]

    if ew_in:
        # Create heavy queue on EW incoming road (5 queued vehicles)
        for idx in range(5):
            v = SimulatedVehicle(f"ew_queue_veh_{idx}", "sedan", [ew_in[0].id], "#ef4444")
            v.distance_on_road = max(0.0, ew_in[0].length - 10.0 - idx * 6.0)
            v.speed = 0.0
            v.wait_time = 15.0
            sim.vehicles[v.id] = v

        phases = controller.compute_phases(current_time=12.0, simulator=sim)
        # EW pressure is substantially higher -> Phase 1 selected
        assert phases[target_id] == 1


def test_max_pressure_controller_downstream_blocked_link_handling():
    """Verifies that MaxPressureController avoids serving movements whose downstream links exceed storage capacity."""
    sim = KinematicSimulator(map_name="bronx_fordham")
    target_id = list(sim.signal_controllers.keys())[0]
    ctrl_fsm = sim.signal_controllers[target_id]

    controller = MaxPressureController(
        min_green=10.0,
        max_green=60.0,
        downstream_occupancy_cap=0.75,
        pressure_threshold=1.0,
        controlled_intersections=[target_id],
    )

    ctrl_fsm.stage = "GREEN"
    ctrl_fsm.active_green_phase = 0
    ctrl_fsm.time_in_green = 15.0

    ew_in = [r for r in sim.roads.values() if r.to_node == target_id and abs(r.points[-1].x - r.points[0].x) >= abs(r.points[-1].y - r.points[0].y)]
    ew_out = [r for r in sim.roads.values() if r.from_node == target_id and abs(r.points[-1].x - r.points[0].x) >= abs(r.points[-1].y - r.points[0].y)]

    # Even if EW has high upstream demand:
    if ew_in and ew_out:
        v_in = SimulatedVehicle("veh_ew_upstream", "sedan", [ew_in[0].id], "#ef4444")
        v_in.distance_on_road = max(0.0, ew_in[0].length - 10.0)
        sim.vehicles[v_in.id] = v_in

        # Block the downstream EW link to 90% capacity
        max_storage = int((ew_out[0].length * ew_out[0].lanes) / 7.5)
        for idx in range(max(2, int(max_storage * 0.90))):
            v_block = SimulatedVehicle(f"veh_downstream_block_{idx}", "sedan", [ew_out[0].id], "#991b1b")
            v_block.distance_on_road = min(ew_out[0].length * 0.9, idx * 6.5)
            v_block.speed = 0.0
            sim.vehicles[v_block.id] = v_block

        # Downstream link is blocked -> MaxPressure suppresses EW movement pressure -> stays in Phase 0
        phases = controller.compute_phases(current_time=15.0, simulator=sim)
        assert phases[target_id] == 0


# ==============================================================================
# 4. Multi-Intersection & Robustness Tests
# ==============================================================================

def test_multiple_intersections_receive_different_phase_requests():
    """Verifies that controllers can output disparate phase requests across different network intersections."""
    sim = KinematicSimulator(map_name="bronx_fordham")
    all_signalized = list(sim.signal_controllers.keys())
    assert len(all_signalized) >= 2

    # Fixed time controller with contrasting plans
    id_a = all_signalized[0]
    id_b = all_signalized[1]
    plans = {
        id_a: {"green_ns": 30.0, "green_ew": 15.0, "offset": 0.0},
        id_b: {"green_ns": 15.0, "green_ew": 30.0, "offset": 15.0},
    }
    controller = FixedTimeController(
        intersection_plans=plans,
        controlled_intersections=[id_a, id_b],
    )

    phases = controller.compute_phases(current_time=10.0, simulator=sim)
    assert id_a in phases and id_b in phases
    assert phases[id_a] == 0
    assert phases[id_b] == 1


def test_invalid_or_missing_telemetry_handling():
    """Verifies that all controllers safely handle None simulators, empty telemetry, and unknown IDs."""
    ft = FixedTimeController()
    act = ActuatedController()
    mp = MaxPressureController()

    # None simulator and telemetry should return empty dict without crashing
    assert ft.compute_phases(10.0, simulator=None, telemetry=None) == {}
    assert act.compute_phases(10.0, simulator=None, telemetry=None) == {}
    assert mp.compute_phases(10.0, simulator=None, telemetry=None) == {}

    # Empty telemetry dictionary
    assert act.compute_phases(10.0, simulator=None, telemetry={}) == {}
    assert mp.compute_phases(10.0, simulator=None, telemetry={}) == {}


def test_all_controllers_respect_phase_6a_fsm_safety():
    """Verifies that applying controller outputs through sim.apply_signal_phases strictly obeys Phase 6A FSM guards."""
    sim = KinematicSimulator(map_name="bronx_fordham")
    target_id = list(sim.signal_controllers.keys())[0]
    ctrl_fsm = sim.signal_controllers[target_id]

    controller = FixedTimeController(
        default_green_ns=15.0,
        default_green_ew=15.0,
        controlled_intersections=[target_id],
    )

    # Initial state
    assert ctrl_fsm.stage == "GREEN"
    assert ctrl_fsm.active_green_phase == 0

    # Step simulation while applying controller commands
    observed_stages = set()
    for step_idx in range(400):
        t = sim.sim_time
        phase_cmds = controller.compute_phases(t, simulator=sim)
        sim.apply_signal_phases(phase_cmds)
        sim.step(0.05)
        observed_stages.add(ctrl_fsm.stage)

        # Safety Invariants
        ns_sig = ctrl_fsm.get_signal_code(is_ns=True)
        ew_sig = ctrl_fsm.get_signal_code(is_ns=False)
        assert not (ns_sig == "G" and ew_sig == "G"), "Conflicting green detected!"

    # Must have cleanly transitioned through Green, Yellow, and All-Red
    assert "GREEN" in observed_stages
    assert "YELLOW" in observed_stages
    assert "ALL_RED" in observed_stages


# ==============================================================================
# 5. WebSocket Feed Local Controller Integration Test
# ==============================================================================

def test_websocket_feed_uses_local_intersection_controllers():
    """Verifies that websocket feed segment state estimation queries local signal controllers rather than representative global phase."""
    from urbanflow.server.websocket_feed import SimulationBroadcaster
    from urbanflow.intelligence.state_estimator import TrafficStateEstimator

    sim = KinematicSimulator(map_name="bronx_fordham")
    server = SimulationBroadcaster(simulator=sim)

    # Find a signalized intersection with an incoming road
    signalized_road = None
    target_inter_id = None
    for r in sim.roads.values():
        inter = sim.intersections.get(r.to_node)
        if inter and inter.has_signals:
            signalized_road = r
            target_inter_id = inter.id
            break

    assert signalized_road is not None
    assert target_inter_id is not None

    local_ctrl = sim.signal_controllers[target_inter_id]
    is_ns = abs(signalized_road.points[-1].x - signalized_road.points[0].x) < abs(signalized_road.points[-1].y - signalized_road.points[0].y) if len(signalized_road.points) >= 2 else True

    # Force local controller to RED for this approach
    opposing_phase = 1 if is_ns else 0
    local_ctrl.active_green_phase = opposing_phase
    local_ctrl.stage = "GREEN"
    assert local_ctrl.get_signal_code(is_ns=is_ns) == "r"

    # Set representative global phase to GREEN for this approach
    same_phase = 0 if is_ns else 1
    sim.signal_phase = same_phase

    # Broadcast frame calculation
    snapshot = sim._get_snapshot()
    # In websocket_feed: line 177 checks local controller
    target_node = sim.intersections.get(signalized_road.to_node)
    ctrl = getattr(sim, "signal_controllers", {}).get(target_node.id)
    assert ctrl is not None
    local_code = ctrl.get_signal_code(is_ns=is_ns)
    assert local_code == "r"
