"""Focused Unit Tests for UrbanFlow AI Phase 1: Safe Dynamic Rerouting Foundation.

Tests:
1. Successful in-flight vehicle dynamic rerouting.
2. Invariant preservation of vehicle kinematics, position, and trip metrics.
3. Rejection of invalid candidate paths (missing vehicle, unknown roads, empty subpath).
4. Rejection of route discontinuities (non-adjacent directed edges).
5. Rejection of destination mismatches (violates destination preservation).
6. Detection and rejection of unintended U-turns (180-degree reversals).
7. Safe no-op handling for identical candidate routes.
8. Batch `reroute_vehicles` contract and accounting.
9. `SumoAdapter` explicit safe unsupported response.
10. `SimulatorAdapter` base contract default fallback.
11. End-to-end multi-step simulation progress along dynamically updated route.
"""
from __future__ import annotations

import copy
import pytest
from pathlib import Path

from urbanflow.core.kinematic_sim import KinematicSimulator, SimulatedVehicle
from urbanflow.core.types import is_ns_approach
from urbanflow.core.simulator_base import SimulatorAdapter
from urbanflow.adapters.sumo_adapter import SumoAdapter


@pytest.fixture
def sim():
    """Initializes a KinematicSimulator on the Bronx Fordham digital twin."""
    s = KinematicSimulator(map_name="bronx_fordham")
    return s


def test_successful_dynamic_reroute(sim):
    """Verifies that an active vehicle can be successfully rerouted to a valid alternative path."""
    # Find or spawn a vehicle with at least 3 roads in its route
    target_veh = None
    for v in sim.vehicles.values():
        if len(v.route) >= 3 and v.route_index < len(v.route) - 2:
            target_veh = v
            break

    if target_veh is None:
        # Manually create and register an active vehicle on a known multi-hop route
        # Find a route from route_pool with >= 4 roads
        candidate_route = None
        for r in sim.route_pool:
            if len(r) >= 4:
                candidate_route = r
                break
        assert candidate_route is not None, "Route pool has no routes of length >= 4"
        target_veh = SimulatedVehicle(
            vehicle_id="test_veh_01",
            v_type="sedan",
            route=candidate_route,
            color="#2563eb",
            desired_speed=11.2,
        )
        first_road = sim.roads[candidate_route[0]]
        target_veh.distance_on_road = 15.0
        sim._update_vehicle_pose(target_veh, first_road)
        sim.vehicles[target_veh.id] = target_veh

    # Record pre-reroute state
    v_id = target_veh.id
    curr_road = target_veh.current_road_id
    curr_idx = target_veh.route_index
    dest_road = target_veh.route[-1]
    dest_node = sim.roads[dest_road].to_node

    pre_dist = target_veh.distance_on_road
    pre_speed = target_veh.speed
    pre_accel = target_veh.accel
    pre_wait = target_veh.wait_time
    pre_total_time = target_veh.total_travel_time
    pre_lane = target_veh.lane_index
    pre_x = target_veh.x
    pre_y = target_veh.y
    pre_heading = target_veh.heading
    orig_route = list(target_veh.route)

    # Find an alternative valid path from current road's to_node to destination node
    import networkx as nx
    curr_to_node = sim.roads[curr_road].to_node
    
    # Extract simple paths or alternate shortest path
    all_paths = []
    try:
        # Find paths between curr_to_node and dest_node
        paths = list(nx.all_shortest_paths(sim.graph, curr_to_node, dest_node, weight="length"))
        # Map node path to road path
        edge_to_road = {(r.from_node, r.to_node): r.id for r in sim.roads.values()}
        for np in paths:
            rp = []
            valid = True
            for i in range(len(np) - 1):
                edge = (np[i], np[i + 1])
                if edge in edge_to_road:
                    rp.append(edge_to_road[edge])
                else:
                    valid = False
                    break
            if valid and rp:
                all_paths.append(rp)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        pass

    # If no alternate path in graph, use the current remaining path prefixed by current road
    if all_paths:
        candidate_subpath = all_paths[0]
    else:
        candidate_subpath = orig_route[curr_idx + 1:]

    # Test rerouting with candidate_subpath
    res = sim.reroute_vehicle(v_id, candidate_subpath)
    assert res["success"] is True
    assert res["vehicle_id"] == v_id

    # Verify physical and trip state are strictly preserved
    assert target_veh.distance_on_road == pre_dist
    assert target_veh.speed == pre_speed
    assert target_veh.accel == pre_accel
    assert target_veh.wait_time == pre_wait
    assert target_veh.total_travel_time == pre_total_time
    assert target_veh.lane_index == pre_lane
    assert target_veh.x == pre_x
    assert target_veh.y == pre_y
    assert target_veh.heading == pre_heading
    assert target_veh.current_road_id == curr_road
    assert target_veh.route_index == curr_idx

    # Verify route history
    assert len(target_veh.route_history) >= 1
    assert target_veh.route_history[0] == orig_route


def test_vehicle_state_preservation_comprehensive(sim):
    """Rigorous verification that all physical kinematic attributes remain untouched."""
    # Find a multi-hop route
    candidate_route = None
    for r in sim.route_pool:
        if len(r) >= 4:
            candidate_route = r
            break
    assert candidate_route is not None

    veh = SimulatedVehicle(
        vehicle_id="veh_preserve_test",
        v_type="suv",
        route=candidate_route,
        color="#10b981",
        desired_speed=12.5,
    )
    first_road = sim.roads[candidate_route[0]]
    veh.distance_on_road = 37.42
    veh.speed = 8.91
    veh.accel = -1.25
    veh.wait_time = 14.5
    veh.total_travel_time = 82.3
    veh.lane_index = min(1, first_road.lanes - 1)
    sim._update_vehicle_pose(veh, first_road)
    sim.vehicles[veh.id] = veh

    snapshot_before = {
        "distance_on_road": veh.distance_on_road,
        "speed": veh.speed,
        "accel": veh.accel,
        "wait_time": veh.wait_time,
        "total_travel_time": veh.total_travel_time,
        "lane_index": veh.lane_index,
        "x": veh.x,
        "y": veh.y,
        "z": veh.z,
        "heading": veh.heading,
        "route_index": veh.route_index,
        "current_road_id": veh.current_road_id,
    }

    # Pass remaining route prefixed by current road
    res = sim.reroute_vehicle(veh.id, candidate_route)
    assert res["success"] is True

    for k, v in snapshot_before.items():
        assert getattr(veh, k) == v, f"Field '{k}' was unexpectedly modified during rerouting!"


def test_reject_missing_vehicle(sim):
    """Verifies safe rejection when vehicle ID does not exist."""
    res = sim.reroute_vehicle("non_existent_vehicle_9999", ["any_road"])
    assert res["success"] is False
    assert res["rerouted"] is False
    assert "not found in active fleet" in res["reason"]


def test_reject_empty_or_invalid_subpath(sim):
    """Verifies safe rejection for empty or non-list subpath."""
    # Spawn a vehicle
    v = next(iter(sim.vehicles.values()))
    res_empty = sim.reroute_vehicle(v.id, [])
    assert res_empty["success"] is False
    assert "non-empty" in res_empty["reason"]

    res_none = sim.reroute_vehicle(v.id, None)
    assert res_none["success"] is False
    assert "non-empty" in res_none["reason"]


def test_reject_unknown_road_id(sim):
    """Verifies safe rejection when candidate path contains a nonexistent road ID."""
    v = next(iter(sim.vehicles.values()))
    res = sim.reroute_vehicle(v.id, ["UNKNOWN_GHOST_ROAD_XYZ"])
    assert res["success"] is False
    assert "not found in network topology" in res["reason"]


def test_reject_route_discontinuity(sim):
    """Verifies rejection when candidate path contains non-adjacent directed links."""
    # Pick two distinct roads that do NOT connect
    all_roads = list(sim.roads.values())
    disconnected_pair = None
    for r1 in all_roads:
        for r2 in all_roads:
            if r1.id != r2.id and r1.to_node != r2.from_node:
                disconnected_pair = (r1, r2)
                break
        if disconnected_pair:
            break
    assert disconnected_pair is not None

    r1, r2 = disconnected_pair
    # Set up vehicle on r1
    veh = SimulatedVehicle(
        vehicle_id="veh_discont_test",
        v_type="sedan",
        route=[r1.id, r1.id],  # dummy
        color="#f59e0b",
    )
    sim.vehicles[veh.id] = veh

    # Attempt to route directly to r2 (which does not connect to r1.to_node)
    res = sim.reroute_vehicle(veh.id, [r2.id])
    assert res["success"] is False
    assert "Route discontinuity" in res["reason"]


def test_reject_destination_mismatch(sim):
    """Verifies rejection when candidate route terminates at a different destination."""
    # Find two routes sharing the same start road but terminating at different destination nodes
    route_a = None
    route_b = None
    for r1_path in sim.route_pool:
        for r2_path in sim.route_pool:
            if r1_path[0] == r2_path[0] and r1_path[-1] != r2_path[-1]:
                d1 = sim.roads[r1_path[-1]].to_node
                d2 = sim.roads[r2_path[-1]].to_node
                if d1 != d2 and len(r1_path) >= 3 and len(r2_path) >= 3:
                    route_a = r1_path
                    route_b = r2_path
                    break
        if route_a:
            break
    assert route_a is not None and route_b is not None

    veh = SimulatedVehicle(
        vehicle_id="veh_dest_test",
        v_type="sedan",
        route=route_a,
        color="#dc2626",
    )
    sim.vehicles[veh.id] = veh

    # Attempt to replace downstream path with route_b's downstream path (different destination)
    cand_path = route_b[1:]
    res = sim.reroute_vehicle(veh.id, cand_path)
    assert res["success"] is False
    assert "Destination mismatch" in res["reason"]


def test_reject_unintended_u_turn(sim):
    """Verifies that immediate 180-degree backtracking U-turns are detected and rejected."""
    # Find a pair of opposing two-way roads (r1: A -> B, r2: B -> A)
    all_roads = list(sim.roads.values())
    opposing_pair = None
    for r1 in all_roads:
        for r2 in all_roads:
            if r1.id != r2.id and r1.from_node == r2.to_node and r1.to_node == r2.from_node:
                if r1.from_node != r1.to_node:  # non-loop
                    opposing_pair = (r1, r2)
                    break
        if opposing_pair:
            break

    if opposing_pair:
        r1, r2 = opposing_pair
        veh = SimulatedVehicle(
            vehicle_id="veh_uturn_test",
            v_type="sedan",
            route=[r1.id, "dummy_road"],
            color="#9333ea",
        )
        sim.vehicles[veh.id] = veh

        # Attempt to reverse directly back onto r2
        res = sim.reroute_vehicle(veh.id, [r2.id])
        assert res["success"] is False
        assert "Unintended U-turn" in res["reason"]


def test_batch_reroute_vehicles_contract(sim):
    """Verifies the batch reroute_vehicles interface and aggregate accounting."""
    commands = [
        {"vehicle_id": "missing_1", "new_route": ["some_road"]},
        {"vehicle_id": "missing_2", "new_subpath": ["some_road"]},
    ]
    batch_res = sim.reroute_vehicles(commands)
    assert batch_res["total_commands"] == 2
    assert batch_res["successful_count"] == 0
    assert batch_res["failed_count"] == 2
    assert batch_res["rerouted_count"] == 0
    assert len(batch_res["results"]) == 2
    assert batch_res["results"][0]["vehicle_id"] == "missing_1"
    assert batch_res["results"][0]["success"] is False


def test_sumo_adapter_safe_unsupported(sim):
    """Verifies SumoAdapter safely returns explicit unsupported response when TraCI is inactive."""
    sumo = SumoAdapter()
    assert sumo.is_running is False

    commands = [{"vehicle_id": "sumo_veh_01", "new_route": ["road_1", "road_2"]}]
    res = sumo.reroute_vehicles(commands)

    assert res["total_commands"] == 1
    assert res["successful_count"] == 0
    assert res["failed_count"] == 1
    assert res["results"][0]["success"] is False
    assert "requires an active TraCI session" in res["results"][0]["reason"]


def test_simulator_adapter_base_default():
    """Verifies that the base SimulatorAdapter provides a safe default fallback."""
    class MinimalTestAdapter(SimulatorAdapter):
        def initialize(self, config): pass
        def step(self, dt): pass
        def reset(self): pass
        def get_network_topology(self): return {}
        def apply_signal_phases(self, phase_commands): pass
        def inject_incident(self, road_id, severity, duration_sec): return True
        def trigger_congestion_surge(self, origin_road_id, vehicle_count): return True
        def close(self): pass

    adapter = MinimalTestAdapter()
    res = adapter.reroute_vehicles([{"vehicle_id": "test_v"}])
    assert res["total_commands"] == 1
    assert res["successful_count"] == 0
    assert res["failed_count"] == 1
    assert "not implemented" in res["results"][0]["reason"]


def test_end_to_end_stepping_along_rerouted_path(sim):
    """Verifies that after rerouting, vehicle advances along the newly assigned route correctly."""
    # Isolate the probe from the randomly pre-seeded Bronx fleet.
    sim.set_demand((), "rerouting test")
    # Find a route with >= 3 roads
    candidate_route = None
    for r in sim.route_pool:
        if len(r) >= 3:
            candidate_route = r
            break
    assert candidate_route is not None

    veh = SimulatedVehicle(
        vehicle_id="veh_e2e_reroute",
        v_type="sedan",
        route=candidate_route,
        color="#06b6d4",
        desired_speed=15.0,
    )
    first_road = sim.roads[candidate_route[0]]
    # Place vehicle near the end of first road
    signal = sim.signal_controllers.get(first_road.to_node)
    if signal:
        signal.active_green_phase = int(not is_ns_approach(first_road))
    veh.distance_on_road = first_road.length - 1.0
    veh.speed = 10.0
    sim._update_vehicle_pose(veh, first_road)
    sim.vehicles[veh.id] = veh

    # Reroute to same downstream path (or alternative)
    res = sim.reroute_vehicle(veh.id, candidate_route[1:])
    assert res["success"] is True

    # Advance 1 physics step (dt=0.25s at 10 m/s = 2.5m advance, which crosses the boundary)
    sim.step(dt=0.25)

    # Vehicle should have transitioned to the next road in its route.
    #
    # NOTE: as of the intersection-turn-connector fix (see
    # tests/test_intersection_turn_connectors.py), a turning movement no
    # longer jumps straight onto candidate_route[1]: a tangent-matched
    # connector segment is spliced into the route first so the vehicle drives
    # a continuous path through the intersection instead of teleporting.
    # A straight-through movement still transitions directly, with no
    # connector needed. Either is correct; assert the vehicle is on one or
    # the other rather than assuming the old direct-jump behavior.
    if veh.id in sim.vehicles:
        assert veh.route_index == 1
        current = veh.current_road_id
        is_direct = current == candidate_route[1]
        is_via_connector = current.startswith("CONN__") and current.endswith(f"__{candidate_route[1]}__0")
        assert is_direct or is_via_connector, (
            f"expected direct transition to {candidate_route[1]!r} or a connector "
            f"leading to it, got {current!r}"
        )
        assert veh.distance_on_road >= 0.0
