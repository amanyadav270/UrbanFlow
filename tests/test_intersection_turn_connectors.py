"""
Regression tests for the intersection turn-connector fix.

Root cause (found by static trace + confirmed numerically below): a vehicle's
route is a flat list of real road ids. When `_update_vehicle_pose` snapped a
vehicle from the lane-specific end point of its current road straight onto the
lane-specific start point of the next road in its route, those two points are
only coincident for a collinear straight-through movement. For any genuine
turn (e.g. eastbound Street North -> northbound Avenue West at INT_NW), the
end of the incoming road and the start of the outgoing road are several
meters apart diagonally, so the vehicle teleported instantly -- matching the
reported "vehicle leaving lanes" / "jumps sideways" / "incorrect turning"
symptoms.

The fix inserts a tangent-matched Bezier connector segment into the route
whenever those two points are not already coincident, so the vehicle drives a
continuous curved path through the intersection instead.
"""
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from urbanflow.core.kinematic_sim import KinematicSimulator
from urbanflow.core.types import Vec3


def _max_consecutive_point_gap(points):
    return max(
        math.hypot(points[i + 1].x - points[i].x, points[i + 1].y - points[i].y)
        for i in range(len(points) - 1)
    )


def _heading_at(points, i):
    dx = points[i + 1].x - points[i].x
    dy = points[i + 1].y - points[i].y
    return (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0


def _angle_diff(a, b):
    d = (a - b) % 360.0
    return d - 360.0 if d > 180.0 else d


def test_turn_connector_is_built_for_a_real_right_turn_at_int_nw():
    """
    road_w_in_nw (eastbound Street North, ends at INT_NW) turning right onto
    road_nw_out_n (northbound Avenue West, starts at INT_NW) is a genuine
    turning movement: its endpoints are NOT already coincident, so a connector
    must be built.
    """
    sim = KinematicSimulator(map_name="classroom_grid")
    prev_road = sim.roads["road_w_in_nw"]
    next_road = sim.roads["road_nw_out_n"]

    # Confirm the premise: these endpoints really are apart (this is the bug).
    p_start, _ = sim._get_lane_point_and_tangent(prev_road, 0, at_end=True)
    p_end, _ = sim._get_lane_point_and_tangent(next_road, 0, at_end=False)
    raw_gap = math.hypot(p_end[0] - p_start[0], p_end[1] - p_start[1])
    assert raw_gap > 1.0, "test premise invalid: endpoints are already coincident"

    connector = sim._get_or_build_turn_connector(prev_road, 0, next_road, 0)
    assert connector is not None, "no connector was built for a genuine turning movement"

    # Connector must start exactly where the incoming lane ends...
    assert math.isclose(connector.points[0].x, p_start[0], abs_tol=1e-6)
    assert math.isclose(connector.points[0].y, p_start[1], abs_tol=1e-6)
    # ...and end exactly where the outgoing lane begins.
    assert math.isclose(connector.points[-1].x, p_end[0], abs_tol=1e-6)
    assert math.isclose(connector.points[-1].y, p_end[1], abs_tol=1e-6)

    # No single hop along the sampled connector may exceed a plausible
    # per-frame travel distance -- i.e. no teleport anywhere along the path.
    assert _max_consecutive_point_gap(connector.points) < 3.0


def test_straight_connector_spans_the_junction_between_matching_lanes():
    sim = KinematicSimulator(map_name="classroom_grid")
    incoming = sim.roads["road_w_in_nw"]
    outgoing = sim.roads["road_nw_ne"]
    connector = sim._get_or_build_turn_connector(incoming, 0, outgoing, 0)
    assert connector is not None
    assert math.isclose(connector.length, 16.0)
    assert all(math.isclose(p.y, connector.points[0].y) for p in connector.points)


def test_connector_heading_is_continuous_no_sudden_flips():
    """
    Along the turn connector, consecutive-segment headings must vary
    smoothly (bounded per-sample change) -- not jump by more than a lane
    change's worth of angle in one sample, which would indicate a kink
    or a reversed tangent.
    """
    sim = KinematicSimulator(map_name="classroom_grid")
    prev_road = sim.roads["road_w_in_nw"]
    next_road = sim.roads["road_nw_out_n"]
    connector = sim._get_or_build_turn_connector(prev_road, 0, next_road, 0)
    assert connector is not None

    headings = [_heading_at(connector.points, i) for i in range(len(connector.points) - 1)]
    for a, b in zip(headings, headings[1:]):
        assert abs(_angle_diff(b, a)) < 45.0, (a, b)


def test_vehicle_driven_through_a_turn_never_teleports():
    """
    End-to-end: drive a single vehicle along a route that turns at INT_NW
    (road_w_in_nw -> road_nw_out_n) using the real simulator tick loop, and
    assert its world position never jumps more than a physically plausible
    per-tick distance (speed-bounded), confirming the connector is actually
    used by the live transition logic in `step`, not just reachable in
    isolation.
    """
    sim = KinematicSimulator(map_name="classroom_grid")
    sim.vehicles.clear()
    sim.pending_spawns.clear()

    from urbanflow.core.kinematic_sim import SimulatedVehicle

    veh = SimulatedVehicle(
        vehicle_id="turn_test_veh",
        v_type="car",
        route=["road_w_in_nw", "road_nw_out_n"],
        color="#ffffff",
        desired_speed=10.0,
        length=4.5,
        scheduled_spawn_time=0.0,
    )
    veh.distance_on_road = sim.roads[veh.current_road_id].length - 8.0
    veh.lane_index = 0
    veh.speed = 10.0
    sim.vehicles[veh.id] = veh
    sim._update_vehicle_pose(veh, sim.roads["road_w_in_nw"])

    # Force the signal controlling INT_NW green so the vehicle is not held at
    # the stop line -- this test is about geometry continuity, not signal
    # timing, which is exercised separately.
    ctrl = sim.signal_controllers.get("INT_NW")
    if ctrl is not None:
        ctrl.get_signal_code = lambda is_ns=True: "G"

    dt = 0.05
    last_x, last_y = veh.x, veh.y
    max_gap_seen = 0.0
    for _ in range(400):
        if veh.id not in sim.vehicles:
            break  # vehicle completed its route
        sim.step(dt)
        if veh.id in sim.vehicles:
            gap = math.hypot(veh.x - last_x, veh.y - last_y)
            # Generous bound: even at full speed (10 m/s) * dt (0.05s) = 0.5m
            # per tick; allow slack for lane-offset changes but a turn jump
            # of several meters (the original bug) must never occur.
            max_gap_seen = max(max_gap_seen, gap)
            last_x, last_y = veh.x, veh.y

    assert max_gap_seen < 1.5, f"vehicle jumped {max_gap_seen:.2f}m in a single tick"


def test_no_chained_or_nested_connectors_during_simulation():
    """Confirms that multi-step simulation never chains connectors into CONN__CONN__."""
    sim = KinematicSimulator(map_name="classroom_grid")
    for _ in range(200):
        sim.step(0.25)
        for v in sim.vehicles.values():
            assert not v.current_road_id.startswith("CONN__CONN__"), (
                f"Vehicle {v.id} entered chained connector {v.current_road_id}"
            )
