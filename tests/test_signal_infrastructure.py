"""Comprehensive Unit Test Suite for UrbanFlow AI Phase 6A: Independent Per-Intersection Traffic Signal Infrastructure.

Verifies:
1. All 27 signalized intersections in the Bronx network initialize with independent state machines.
2. Distinct intersections can hold different phases simultaneously.
3. Mandatory clearance transitions (Yellow 3.0s -> All-Red 1.5s -> Next Green) with no green-to-green jumps.
4. Minimum green (10.0s) enforcement prevents premature phase switching.
5. Maximum green (60.0s) enforcement terminates green under competing demand.
6. Conflicting greens are strictly prohibited across all states.
7. Vehicles respond to their specific local downstream intersection signal.
8. Unknown intersection IDs and invalid phase commands are handled gracefully without crashing.
9. Snapshot telemetry correctly reflects per-intersection signal states.
10. Reset returns all 27 controllers to clean initial states.
"""
from __future__ import annotations

import pytest
from typing import Dict, Any, List

from urbanflow.core.kinematic_sim import KinematicSimulator, IntersectionSignalFSM
from urbanflow.core.types import Vec3, RoadSegment, Intersection, SignalState


def test_all_27_signalized_intersections_initialized_independently():
    """Verifies that all 27 signalized intersections are initialized with independent state machines."""
    sim = KinematicSimulator(map_name="bronx_fordham")

    # Bronx digital twin has exactly 27 signalized intersections
    signalized_inter_ids = [
        inter_id for inter_id, inter in sim.intersections.items() if inter.has_signals
    ]
    assert len(signalized_inter_ids) == 27
    assert len(sim.signal_controllers) == 27

    # Verify each controller is an independent IntersectionSignalFSM instance
    seen_ctrl_ids = set()
    for inter_id in signalized_inter_ids:
        assert inter_id in sim.signal_controllers
        ctrl = sim.signal_controllers[inter_id]
        assert isinstance(ctrl, IntersectionSignalFSM)
        assert ctrl.intersection_id == inter_id
        assert ctrl.min_green == 10.0
        assert ctrl.max_green == 60.0
        assert ctrl.yellow_duration == 3.0
        assert ctrl.all_red_duration == 1.5
        assert ctrl.stage == "GREEN"
        assert ctrl.active_green_phase == 0

        # Memory address uniqueness
        assert id(ctrl) not in seen_ctrl_ids
        seen_ctrl_ids.add(id(ctrl))


def test_two_intersections_hold_different_phases_simultaneously():
    """Verifies that two distinct intersections can maintain different signal phases simultaneously."""
    sim = KinematicSimulator(map_name="bronx_fordham")
    inter_ids = list(sim.signal_controllers.keys())
    assert len(inter_ids) >= 2

    id_a = inter_ids[0]
    id_b = inter_ids[1]

    # Command intersection A to Phase 0 (NS Green) and intersection B to Phase 1 (EW Green)
    # Give intersection B enough time for min_green to allow switch, plus clearance
    sim.signal_controllers[id_b].time_in_green = 12.0
    sim.apply_signal_phases({id_a: 0, id_b: 1})

    # Intersection B should transition through Yellow (3s) and All-Red (1.5s)
    # Step 50 ticks of 0.1s = 5.0s
    for _ in range(50):
        sim.step(0.1)

    ctrl_a = sim.signal_controllers[id_a]
    ctrl_b = sim.signal_controllers[id_b]

    # Intersection A is in NS Green (Phase 0)
    assert ctrl_a.active_green_phase == 0
    assert ctrl_a.stage == "GREEN"
    assert ctrl_a.get_signal_code(is_ns=True) == "G"
    assert ctrl_a.get_signal_code(is_ns=False) == "r"
    assert ctrl_a.get_phase_index() == 0

    # Intersection B is in EW Green (Phase 1 / phase index 2)
    assert ctrl_b.active_green_phase == 1
    assert ctrl_b.stage == "GREEN"
    assert ctrl_b.get_signal_code(is_ns=True) == "r"
    assert ctrl_b.get_signal_code(is_ns=False) == "G"
    assert ctrl_b.get_phase_index() == 2

    # Verify they have genuinely different signal states
    assert ctrl_a.get_phase_index() != ctrl_b.get_phase_index()
    assert ctrl_a.get_signal_code(is_ns=True) != ctrl_b.get_signal_code(is_ns=True)


def test_clearance_transitions_enforced_on_phase_change():
    """Verifies exact clearance sequence: Green -> Yellow (3.0s) -> All-Red (1.5s) -> Next Green."""
    fsm = IntersectionSignalFSM(
        intersection_id="test_node",
        min_green=10.0,
        max_green=60.0,
        yellow_duration=3.0,
        all_red_duration=1.5,
    )

    # Initially in Green (Phase 0: NS)
    assert fsm.stage == "GREEN"
    assert fsm.active_green_phase == 0
    assert fsm.get_phase_index() == 0
    assert fsm.get_signal_code(is_ns=True) == "G"
    assert fsm.get_signal_code(is_ns=False) == "r"

    # Satisfy min_green
    fsm.time_in_green = 10.0

    # Request phase 1 (EW)
    success = fsm.request_phase(1)
    assert success is True

    # 1. Immediate entry into YELLOW
    assert fsm.stage == "YELLOW"
    assert fsm.time_in_stage == 0.0
    assert fsm.get_phase_index() == 1  # NS Yellow
    assert fsm.get_signal_code(is_ns=True) == "y"
    assert fsm.get_signal_code(is_ns=False) == "r"  # Conflicting approach remains red

    # Advance by 2.0s (< 3.0s yellow duration)
    fsm.step(2.0)
    assert fsm.stage == "YELLOW"
    assert fsm.get_phase_index() == 1

    # Advance by 1.0s (total 3.0s) -> Transitions to ALL_RED
    fsm.step(1.0)
    assert fsm.stage == "ALL_RED"
    assert fsm.time_in_stage == 0.0
    assert fsm.get_phase_index() == 4  # All-Red Clearance
    assert fsm.get_signal_code(is_ns=True) == "r"
    assert fsm.get_signal_code(is_ns=False) == "r"  # Neither direction has green

    # Advance by 1.0s (< 1.5s all-red duration)
    fsm.step(1.0)
    assert fsm.stage == "ALL_RED"
    assert fsm.get_phase_index() == 4

    # Advance by 0.5s (total 1.5s) -> Transitions to next GREEN (Phase 1: EW)
    fsm.step(0.5)
    assert fsm.stage == "GREEN"
    assert fsm.active_green_phase == 1
    assert fsm.time_in_stage == 0.0
    assert fsm.time_in_green == 0.0
    assert fsm.get_phase_index() == 2  # EW Green
    assert fsm.get_signal_code(is_ns=True) == "r"
    assert fsm.get_signal_code(is_ns=False) == "G"


def test_minimum_green_prevents_premature_phase_switching():
    """Verifies that a phase change request cannot terminate green before min_green (10s) has elapsed."""
    fsm = IntersectionSignalFSM("test_node", min_green=10.0)

    # In green for only 4.0 seconds (< 10.0s)
    fsm.time_in_green = 4.0
    fsm.time_in_stage = 4.0

    # Request phase switch
    fsm.request_phase(1)

    # Must NOT switch to yellow immediately
    assert fsm.stage == "GREEN"
    assert fsm.active_green_phase == 0
    assert fsm.target_green_phase == 1

    # Advance by 5.0s (total green 9.0s < 10.0s)
    fsm.step(5.0)
    assert fsm.stage == "GREEN"
    assert fsm.target_green_phase == 1

    # Advance by 1.0s (total green 10.0s == 10.0s min_green)
    fsm.step(1.0)
    # Now it transitions to YELLOW
    assert fsm.stage == "YELLOW"
    assert fsm.time_in_stage == 0.0


def test_maximum_green_forces_phase_switch_under_competing_demand():
    """Verifies that green terminates at max_green (60s) when competing demand is present."""
    fsm = IntersectionSignalFSM("test_node", min_green=10.0, max_green=60.0)

    # External controller commands phase 0 (attempting to hold green indefinitely)
    fsm.request_phase(0)
    fsm.competing_demand_detected = True

    # Advance 59 seconds
    fsm.step(59.0)
    assert fsm.stage == "GREEN"
    assert fsm.time_in_green == 59.0

    # Advance past 60.0s threshold
    fsm.step(1.5)
    assert fsm.stage == "YELLOW"
    assert fsm.target_green_phase == 1


def test_no_competing_demand_dwells_in_green():
    """Verifies that when no competing demand exists, an intersection can dwell in green beyond max_green."""
    fsm = IntersectionSignalFSM("test_node", min_green=10.0, max_green=60.0)
    fsm.request_phase(0)
    fsm.competing_demand_detected = False

    # Advance 90 seconds without competing demand
    fsm.step(90.0)
    assert fsm.stage == "GREEN"
    assert fsm.active_green_phase == 0


def test_conflicting_green_strictly_prohibited():
    """Verifies that no signal state ever produces simultaneous green for conflicting approaches."""
    fsm = IntersectionSignalFSM("test_node")

    # Test all possible discrete states
    for stage in ["GREEN", "YELLOW", "ALL_RED"]:
        for active_phase in [0, 1]:
            fsm.stage = stage
            fsm.active_green_phase = active_phase

            ns_code = fsm.get_signal_code(is_ns=True)
            ew_code = fsm.get_signal_code(is_ns=False)

            # Conflicting green invariant
            assert not (ns_code == "G" and ew_code == "G"), (
                f"Conflicting green detected in stage {stage}, active_phase {active_phase}: NS={ns_code}, EW={ew_code}"
            )


def test_vehicles_respond_to_local_signal_states():
    """Verifies vehicles approaching signalized intersections obey red/yellow stop lines and proceed on green."""
    sim = KinematicSimulator(map_name="bronx_fordham")

    # Find a signalized intersection with incoming roads
    target_inter_id = None
    target_road = None
    for r in sim.roads.values():
        inter = sim.intersections.get(r.to_node)
        if inter and inter.has_signals and r.length > 30.0:
            target_inter_id = inter.id
            target_road = r
            break

    assert target_inter_id is not None
    assert target_road is not None

    ctrl = sim.signal_controllers[target_inter_id]
    is_ns = (
        abs(target_road.points[-1].x - target_road.points[0].x)
        < abs(target_road.points[-1].y - target_road.points[0].y)
        if len(target_road.points) >= 2
        else True
    )

    # Force signal to RED for this approach
    opposing_phase = 1 if is_ns else 0
    ctrl.active_green_phase = opposing_phase
    ctrl.stage = "GREEN"
    ctrl.time_in_green = 15.0
    assert ctrl.get_signal_code(is_ns=is_ns) == "r"

    # Spawn test vehicle near stop line
    sim.vehicle_counter += 1
    v_id = f"test_stop_veh_{sim.vehicle_counter}"
    from urbanflow.core.kinematic_sim import SimulatedVehicle
    veh = SimulatedVehicle(
        vehicle_id=v_id,
        v_type="sedan",
        route=[target_road.id],
        color="#2563eb",
        desired_speed=11.2,
    )
    veh.distance_on_road = target_road.length - 15.0  # 15m before stop line
    veh.speed = 10.0
    sim.vehicles[v_id] = veh
    sim._update_vehicle_pose(veh, target_road)

    # Step simulation while facing red (allowing IDM vehicle to decelerate to standstill)
    for _ in range(70):
        sim.step(0.05)

    # Vehicle must have decelerated and stopped before the stop line
    assert v_id in sim.vehicles
    assert veh.speed < 1.0
    assert veh.distance_on_road < target_road.length

    # Now turn signal to GREEN for this approach
    same_phase = 0 if is_ns else 1
    ctrl.active_green_phase = same_phase
    ctrl.stage = "GREEN"
    ctrl.time_in_green = 0.0
    assert ctrl.get_signal_code(is_ns=is_ns) == "G"

    # Step simulation facing green
    initial_dist = veh.distance_on_road
    for _ in range(40):
        sim.step(0.05)

    # Vehicle must accelerate and move forward on green
    assert veh.distance_on_road > initial_dist or v_id not in sim.vehicles  # Traversed or moved


def test_apply_signal_phases_error_handling_and_robustness():
    """Verifies graceful handling of unknown intersection IDs, non-dict payloads, and invalid phases."""
    sim = KinematicSimulator(map_name="bronx_fordham")

    # 1. Unknown intersection ID should not crash
    sim.apply_signal_phases({"NON_EXISTENT_INTERSECTION_9999": 1})

    # 2. Invalid phase numbers should be safely ignored
    first_id = next(iter(sim.signal_controllers.keys()))
    ctrl = sim.signal_controllers[first_id]
    original_phase = ctrl.active_green_phase

    sim.apply_signal_phases({first_id: 999})
    assert ctrl.active_green_phase == original_phase

    sim.apply_signal_phases({first_id: -5})
    assert ctrl.active_green_phase == original_phase

    # 3. None and empty dictionary
    sim.apply_signal_phases({})
    sim.apply_signal_phases(None)  # type: ignore

    # 4. Non-string keys
    sim.apply_signal_phases({12345: 0})  # type: ignore


def test_snapshot_telemetry_contains_per_intersection_state():
    """Verifies that SimulationSnapshot contains structured, per-intersection SignalState entries."""
    sim = KinematicSimulator(map_name="bronx_fordham")
    snapshot = sim._get_snapshot()

    assert len(snapshot.signals) == 27

    seen_ids = set()
    for sig in snapshot.signals:
        assert isinstance(sig, SignalState)
        assert sig.intersection_id in sim.signal_controllers
        assert sig.stage in ("GREEN", "YELLOW", "ALL_RED")
        assert sig.phase_index in (0, 1, 2, 3, 4)
        assert isinstance(sig.time_in_phase, float)
        assert isinstance(sig.duration, float)
        assert "NS:" in sig.state_string and "EW:" in sig.state_string
        seen_ids.add(sig.intersection_id)

    assert len(seen_ids) == 27

    # Verify serialization
    snap_dict = snapshot.to_dict()
    assert len(snap_dict["signals"]) == 27
    s0 = snap_dict["signals"][0]
    assert "stage" in s0
    assert "is_yellow" in s0
    assert "is_all_red" in s0


def test_reset_restores_all_controllers_to_initial_state():
    """Verifies that simulator reset restores all 27 controllers to initial green and zero timers."""
    sim = KinematicSimulator(map_name="bronx_fordham")

    # Modify state of several controllers
    for idx, ctrl in enumerate(sim.signal_controllers.values()):
        if idx % 2 == 0:
            ctrl.stage = "YELLOW"
            ctrl.time_in_stage = 2.5
            ctrl.active_green_phase = 1

    # Reset
    sim.reset()

    # All controllers must be reset
    for ctrl in sim.signal_controllers.values():
        assert ctrl.stage == "GREEN"
        assert ctrl.active_green_phase == 0
        assert ctrl.target_green_phase is None
        assert ctrl.time_in_stage == 0.0
        assert ctrl.time_in_green == 0.0
