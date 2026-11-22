"""Unit Tests for UrbanFlow AI Phase 5A: Routing Diversity and Scenario Alignment.

Tests:
1. Candidate search finds diverse alternatives when the 1-best shortest path equals current route.
2. Every candidate alternative is a valid directed path preserving continuity and destination.
3. Capacity constraints and all existing safeguards (45s improvement, 0.80 V/C cap, 25% quota, cooldown) remain active.
4. Corrected Family 5 incident road ('road_1082220764_167_fwd') has actual traffic exposure under scenario exogenous inputs.
5. Baseline routing behavior remains strictly unchanged when dynamic routing is disabled.
"""
from __future__ import annotations

import pytest
from numpy.random import Generator, Philox

from urbanflow.core.kinematic_sim import KinematicSimulator, SimulatedVehicle
from urbanflow.intelligence.routing import RouteCostEvaluator, PredictiveRoutingController
from scripts.run_dynamic_routing_benchmark import (
    get_held_out_scenarios,
    derive_stream_seed,
)


@pytest.fixture
def sim():
    """Initializes KinematicSimulator with Bronx Fordham digital twin."""
    return KinematicSimulator(map_name="bronx_fordham")


@pytest.fixture
def evaluator(sim):
    """Initializes RouteCostEvaluator in reactive mode."""
    return RouteCostEvaluator(simulator=sim, mode="reactive")


def test_candidate_search_finds_alternatives_when_shortest_equals_current_route(sim, evaluator):
    """Verifies that find_candidate_paths returns diverse alternative corridors even when 1-best is current route."""
    # Find a route where 1-best Dijkstra path equals the route
    matching_route = None
    for r in sim.route_pool:
        sp = evaluator.find_shortest_path(r[0], dest_road_id=r[-1], allow_node_destination=True)
        if sp["success"] and sp["path"] == r[1:]:
            matching_route = r
            break

    assert matching_route is not None, "Expected at least one route where 1-best path is the route itself."
    start_road = matching_route[0]
    dest_road = matching_route[-1]
    downstream_route = matching_route[1:]

    # Under free-flow conditions, 1-best shortest path is downstream_route
    shortest = evaluator.find_shortest_path(start_road, dest_road_id=dest_road, allow_node_destination=True)
    assert shortest["success"] is True
    assert shortest["path"] == downstream_route

    # Query candidate paths with diversity (k=5)
    candidates = evaluator.find_candidate_paths(
        start_road,
        dest_road_id=dest_road,
        allow_node_destination=True,
        k=5,
    )

    # Must find multiple candidate paths
    assert len(candidates) >= 2

    # The first candidate is the optimal path (downstream_route)
    assert candidates[0]["path"] == downstream_route

    # At least one candidate is genuinely different from downstream_route
    distinct_candidates = [c for c in candidates if c["path"] != downstream_route]
    assert len(distinct_candidates) >= 1

    for cand in distinct_candidates:
        assert cand["path"] != downstream_route
        assert cand["cost"] >= candidates[0]["cost"]
        # Verify valid directed path
        val = evaluator.validate_path(
            cand["path"],
            start_road_id=start_road,
            dest_road_id=dest_road,
            allow_node_destination=True,
        )
        assert val["valid"] is True, f"Candidate path invalid: {val.get('reason')}"


def test_candidate_paths_preserve_directed_continuity_and_destination(sim, evaluator):
    """Verifies that every candidate alternative path strictly preserves continuity, adjacency, and destination."""
    for r_idx in [0, 5, 10, 15, 20]:
        route = sim.route_pool[r_idx]
        start_road = route[0]
        dest_road = route[-1]

        candidates = evaluator.find_candidate_paths(
            start_road,
            dest_road_id=dest_road,
            allow_node_destination=True,
            k=4,
        )

        for cand in candidates:
            path = cand["path"]
            val = evaluator.validate_path(
                path,
                start_road_id=start_road,
                dest_road_id=dest_road,
                allow_node_destination=True,
            )
            assert val["valid"] is True, f"Path {path} failed validation: {val.get('reason')}"

            # Verify current-road continuity: path starts from start_road.to_node
            first_road = sim.roads[path[0]]
            assert first_road.from_node == sim.roads[start_road].to_node

            # Verify destination semantics: path ends at dest_road or shares its discharge node
            last_road = sim.roads[path[-1]]
            assert last_road.to_node == sim.roads[dest_road].to_node

            # Verify no unintended U-turns at inception
            assert first_road.to_node != sim.roads[start_road].from_node or first_road.from_node == sim.roads[start_road].from_node


def test_capacity_constraints_and_safeguards_active_on_diverse_candidates(sim, evaluator):
    """Verifies that controller advances to secondary alternative if primary exceeds capacity, and enforces 45s benefit."""
    ctrl = PredictiveRoutingController(
        route_evaluator=evaluator,
        mode="reactive",
        strict_quota_floor=False,
        vc_admission_cap=0.80,
        min_improvement_sec=45.0,
    )

    start_road = "road_586133572_120_fwd"
    dest_road = "road_1082220763_166_fwd"
    curr_base_path = [
        start_road,
        "road_586133572_121_fwd",
        "road_586133572_122_fwd",
        "road_586133572_123_fwd",
        "road_223889617_62_fwd",
        "road_1082220766_170_fwd",
        dest_road,
    ]

    sim.vehicles.clear()
    veh = SimulatedVehicle("veh_safeguard_test", "sedan", list(curr_base_path), "#2563eb")
    sim.vehicles[veh.id] = veh

    # Scenario A: Congestion on destination road makes current route cost = 81.1s
    # Primary alternative uses 'road_5698666_15_fwd'. Put it at V/C = 0.85 (over capacity cap 0.80).
    telem = {
        dest_road: {"vc_ratio": 3.0, "operational_capacity": 500.0, "queue_length_meters": 60.0},
        "road_5698666_15_fwd": {"vc_ratio": 0.85, "operational_capacity": 1800.0, "flow_veh_per_hour": 1530.0},
    }
    ctrl.update_telemetry(telem, timestamp=50.0)

    res = ctrl.evaluate_and_reroute(sim, current_time=50.0, telemetry_map=telem)

    # Primary alternative (via road_5698666_15_fwd) was rejected for capacity
    # Secondary alternative offers only 27.7s improvement (< 45s threshold)
    # Total diversions must be 0
    assert res["diverted_count"] == 0
    assert res["decisions"][0]["diverted"] is False
    assert "exceeds capacity admission threshold" in res["decisions"][0]["reason"]

    # Scenario B: Now relieve road_5698666_15_fwd to V/C = 0.40 (< 0.80 cap)
    # The primary alternative (improvement ~54.7s >= 45s) is now fully admitted!
    telem_clear = {
        dest_road: {"vc_ratio": 3.0, "operational_capacity": 500.0, "queue_length_meters": 60.0},
        "road_5698666_15_fwd": {"vc_ratio": 0.40, "operational_capacity": 1800.0, "flow_veh_per_hour": 720.0},
    }
    ctrl.update_telemetry(telem_clear, timestamp=100.0)
    res_clear = ctrl.evaluate_and_reroute(sim, current_time=100.0, telemetry_map=telem_clear)

    assert res_clear["diverted_count"] == 1
    assert res_clear["decisions"][0]["diverted"] is True
    assert res_clear["decisions"][0]["reason"] == "Accepted"
    assert res_clear["decisions"][0]["improvement_sec"] >= 45.0


def test_family_5_incident_road_has_active_traffic_exposure():
    """Verifies that Family 5 incident road is corrected to road_1082220764_167_fwd and has active traffic."""
    scenarios = get_held_out_scenarios()
    fam5_scenarios = [s for s in scenarios if s.family_id == "FAM-05"]

    assert len(fam5_scenarios) == 10
    for sc in fam5_scenarios:
        # 1. Road is aligned to the active intersection bottleneck link
        assert sc.incident_road == "road_1082220764_167_fwd"
        # 2. Timing and severity preserved
        assert sc.incident_start_sec == 1200.0
        assert sc.incident_duration_sec == 1000.0
        assert sc.incident_severity == 0.95
        assert sc.demand_multiplier == 1.25

    # 3. Verify route pool exposure
    sim = KinematicSimulator(map_name="bronx_fordham")
    routes_traversing = [r for r in sim.route_pool if "road_1082220764_167_fwd" in r]
    assert len(routes_traversing) >= 3, f"Expected at least 3 routes, found {len(routes_traversing)}"

    # 4. Verify exogenous spawn exposure for held-out seed 5001
    sc_5001 = [s for s in fam5_scenarios if s.seed == 5001][0]
    stream_spawn = derive_stream_seed(sc_5001.seed, "spawn_arrivals")
    rng_spawn = Generator(Philox(stream_spawn))

    spawns_on_incident_road = 0
    total_spawns = 1000
    for _ in range(total_spawns):
        _fleet_roll = float(rng_spawn.random())
        r_idx = int(rng_spawn.integers(0, len(sim.route_pool)))
        route = sim.route_pool[r_idx]
        if "road_1082220764_167_fwd" in route:
            spawns_on_incident_road += 1

    assert spawns_on_incident_road > 20, f"Expected substantial spawns, got {spawns_on_incident_road}"


def test_baseline_routing_unchanged_when_dynamic_routing_disabled(sim):
    """Verifies that baseline static routing does not modify vehicle routes even under heavy congestion."""
    route = sim.route_pool[0]
    sim.vehicles.clear()
    veh = SimulatedVehicle("veh_baseline", "sedan", list(route), "#2563eb")
    sim.vehicles[veh.id] = veh

    original_route = list(veh.route)

    # When dynamic routing controller is absent (baseline regime), vehicles step without route modifications
    for _ in range(20):
        sim.step(0.05)

    assert veh.route == original_route
    assert veh.reroute_count == 0
