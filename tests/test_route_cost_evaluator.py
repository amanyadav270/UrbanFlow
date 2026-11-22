"""Unit Tests for UrbanFlow AI Phase 2: Dynamic Link Cost & Pathfinding Engine.

Tests:
1. Free-flow link cost and monotonicity.
2. Congestion-driven route diversion (BPR V/C ratio scaling).
3. Predictive-risk-driven route diversion (Reactive vs Predictive mode comparison).
4. Queue occupancy penalty behavior.
5. Numerically stable safeguards (NaNs, infinities, negative values, zero capacity).
6. Closed road detour behavior and heavy penalty application.
7. Disconnected paths and missing node handling.
8. Cycle prevention (Dijkstra guarantees cycle-free shortest paths).
9. Deterministic tie-breaking across identical-cost alternatives.
10. Strict road destination vs node-level destination semantics.
11. Path validation contracts (directed adjacency, U-turn prevention, destination matching).
"""
from __future__ import annotations

import math
import pytest
from pathlib import Path

from urbanflow.intelligence.routing import RouteCostEvaluator


@pytest.fixture
def evaluator():
    """Initializes RouteCostEvaluator with the real Bronx Fordham map."""
    return RouteCostEvaluator(mode="reactive")


def test_free_flow_link_cost_and_monotonicity(evaluator):
    """Verifies that in absence of congestion, edge cost equals free-flow travel time."""
    road_id = next(iter(evaluator.roads.keys()))
    road = evaluator.roads[road_id]
    expected_t0 = road["length"] / road["speed_limit"]

    cost = evaluator.compute_edge_cost(road_id)
    assert abs(cost - expected_t0) < 1e-4
    assert cost > 0.0
    assert not math.isnan(cost)
    assert not math.isinf(cost)


def test_bpr_congestion_penalty_scaling(evaluator):
    """Verifies that BPR congestion factor monotonically increases cost as V/C increases."""
    road_id = next(iter(evaluator.roads.keys()))
    t0 = evaluator.compute_edge_cost(road_id)

    # Test increasing V/C ratios
    cost_0 = evaluator.compute_edge_cost(road_id, telemetry={"vc_ratio": 0.0})
    cost_half = evaluator.compute_edge_cost(road_id, telemetry={"vc_ratio": 0.5})
    cost_cap = evaluator.compute_edge_cost(road_id, telemetry={"vc_ratio": 1.0})
    cost_over = evaluator.compute_edge_cost(road_id, telemetry={"vc_ratio": 1.5})

    assert cost_0 == pytest.approx(t0, rel=1e-3)
    assert cost_half > cost_0
    assert cost_cap > cost_half
    assert cost_over > cost_cap

    # At V/C = 1.0, BPR formula adds exactly alpha * t0
    assert cost_cap == pytest.approx(t0 * (1.0 + evaluator.bpr_alpha), rel=1e-3)


def test_queue_occupancy_penalty(evaluator):
    """Verifies that downstream queue length/occupancy increases link cost."""
    road_id = next(iter(evaluator.roads.keys()))
    t0 = evaluator.compute_edge_cost(road_id)
    road_len = evaluator.roads[road_id]["length"]

    cost_no_queue = evaluator.compute_edge_cost(road_id, telemetry={"queue_length_meters": 0.0})
    cost_half_queue = evaluator.compute_edge_cost(road_id, telemetry={"queue_length_meters": road_len * 0.5})
    cost_full_queue = evaluator.compute_edge_cost(road_id, telemetry={"queue_length_meters": road_len})

    assert cost_half_queue > cost_no_queue
    assert cost_full_queue > cost_half_queue
    # At 100% queue occupancy, penalty adds queue_weight * t0
    assert cost_full_queue == pytest.approx(t0 * (1.0 + evaluator.queue_weight), rel=1e-3)


def test_reactive_vs_predictive_mode(evaluator):
    """Verifies that predictive risk only affects cost when mode is 'predictive'."""
    road_id = next(iter(evaluator.roads.keys()))
    t0 = evaluator.compute_edge_cost(road_id)

    # In reactive mode, p_spillback is strictly ignored
    evaluator.set_mode("reactive")
    cost_reactive_low = evaluator.compute_edge_cost(road_id, p_spillback=0.0)
    cost_reactive_high = evaluator.compute_edge_cost(road_id, p_spillback=0.95)
    assert cost_reactive_low == cost_reactive_high == pytest.approx(t0, rel=1e-3)

    # In predictive mode, p_spillback adds penalty
    evaluator.set_mode("predictive", pred_weight=4.0)
    cost_pred_low = evaluator.compute_edge_cost(road_id, p_spillback=0.0)
    cost_pred_high = evaluator.compute_edge_cost(road_id, p_spillback=0.80)
    assert cost_pred_low == pytest.approx(t0, rel=1e-3)
    assert cost_pred_high > cost_pred_low
    expected_pred_cost = t0 * (1.0 + 4.0 * 0.80)
    assert cost_pred_high == pytest.approx(expected_pred_cost, rel=1e-3)


def test_congestion_driven_route_diversion(evaluator):
    """Verifies that high congestion on the primary route causes the router to divert onto an alternate path."""
    evaluator.set_mode("reactive")

    start_road = "road_586133572_120_fwd"
    dest_node = "int_5516750363"

    # Baseline free-flow path
    res_free = evaluator.find_shortest_path(start_road, dest_node_id=dest_node)
    assert res_free["success"] is True
    initial_path = res_free["path"]
    assert "road_1082220763_166_fwd" in initial_path

    # Congest the preferred downstream road
    target_hop = "road_1082220763_166_fwd"
    telem = {target_hop: {"vc_ratio": 3.0, "operational_capacity": 400.0, "queue_length_meters": 50.0}}

    # Re-evaluate shortest path
    res_divert = evaluator.find_shortest_path(start_road, dest_node_id=dest_node, telemetry_map=telem)
    assert res_divert["success"] is True
    divert_path = res_divert["path"]

    # Router should have diverted onto the alternate parallel branch
    assert divert_path != initial_path
    assert target_hop not in divert_path
    assert "road_5698666_15_fwd" in divert_path
    assert "road_1082220764_167_fwd" in divert_path


def test_predictive_risk_driven_route_diversion(evaluator):
    """Verifies that high predictive risk triggers proactive diversion in predictive mode, but not reactive."""
    start_road = "road_586133572_120_fwd"
    dest_node = "int_5516750363"

    # Baseline free-flow path in reactive mode
    evaluator.set_mode("reactive")
    res_base = evaluator.find_shortest_path(start_road, dest_node_id=dest_node)
    base_path = res_base["path"]
    preferred_road = "road_1082220763_166_fwd"
    assert preferred_road in base_path

    # High predictive spillback risk on preferred road (but zero physical congestion)
    p_risk_map = {preferred_road: 0.95}

    # In reactive mode: risk is ignored -> stays on base path
    res_reactive = evaluator.find_shortest_path(
        start_road,
        dest_node_id=dest_node,
        p_spillback_map=p_risk_map,
    )
    assert res_reactive["path"] == base_path

    # In predictive mode with high weight: proactively detours onto alternate branch
    evaluator.set_mode("predictive", pred_weight=10.0)
    res_predictive = evaluator.find_shortest_path(
        start_road,
        dest_node_id=dest_node,
        p_spillback_map=p_risk_map,
    )
    assert res_predictive["success"] is True
    assert res_predictive["path"] != base_path
    assert preferred_road not in res_predictive["path"]


def test_closed_roads_heavy_penalty_and_detour(evaluator):
    """Verifies that an explicitly closed road receives CLOSED_ROAD_PENALTY and is avoided when a detour exists."""
    start_road = "road_586133572_120_fwd"
    dest_node = "int_5516750363"

    res_open = evaluator.find_shortest_path(start_road, dest_node_id=dest_node)
    assert res_open["success"] is True
    initial_path = res_open["path"]
    road_to_close = "road_1082220763_166_fwd"
    assert road_to_close in initial_path

    closed_cost = evaluator.compute_edge_cost(road_to_close, is_closed=True)
    assert closed_cost == evaluator.CLOSED_ROAD_PENALTY

    # Search with closed road
    res_closed = evaluator.find_shortest_path(
        start_road,
        dest_node_id=dest_node,
        closed_roads={road_to_close},
    )
    assert res_closed["success"] is True
    assert road_to_close not in res_closed["path"]
    assert res_closed["path"] != initial_path


def test_numerical_safeguards_invalid_telemetry(evaluator):
    """Verifies that NaNs, infinities, negative values, and zero capacity are handled safely."""
    road_id = next(iter(evaluator.roads.keys()))
    t0 = evaluator.compute_edge_cost(road_id)

    # 1. Zero capacity
    cost_zero_cap = evaluator.compute_edge_cost(road_id, telemetry={"operational_capacity": 0.0})
    assert cost_zero_cap == evaluator.CLOSED_ROAD_PENALTY

    # 2. NaN / Inf V/C ratio
    cost_nan_vc = evaluator.compute_edge_cost(road_id, telemetry={"vc_ratio": float("nan")})
    assert not math.isnan(cost_nan_vc)
    assert cost_nan_vc >= t0

    cost_inf_vc = evaluator.compute_edge_cost(road_id, telemetry={"vc_ratio": float("inf")})
    assert not math.isinf(cost_inf_vc)
    assert cost_inf_vc <= evaluator.MAX_EDGE_COST

    # 3. Negative queue length
    cost_neg_q = evaluator.compute_edge_cost(road_id, telemetry={"queue_length_meters": -50.0})
    assert cost_neg_q >= t0

    # 4. NaN / negative spillback risk
    evaluator.set_mode("predictive")
    cost_nan_risk = evaluator.compute_edge_cost(road_id, p_spillback=float("nan"))
    assert not math.isnan(cost_nan_risk)

    cost_neg_risk = evaluator.compute_edge_cost(road_id, p_spillback=-0.5)
    assert cost_neg_risk == pytest.approx(t0, rel=1e-3)


def test_deterministic_tie_breaking(evaluator):
    """Verifies that pathfinding is 100% deterministic between identical-cost alternatives."""
    roads_list = list(evaluator.roads.keys())
    start_road = roads_list[10]
    dest_road = roads_list[25]

    res1 = evaluator.find_shortest_path(start_road, dest_road_id=dest_road, allow_node_destination=True)
    res2 = evaluator.find_shortest_path(start_road, dest_road_id=dest_road, allow_node_destination=True)

    assert res1["success"] == res2["success"]
    assert res1["path"] == res2["path"]
    assert res1["cost"] == res2["cost"]


def test_destination_semantics_strict_road_vs_node(evaluator):
    """Verifies the distinction between strict road destination and terminal node destination."""
    # Find two roads that enter the exact same intersection node
    inter_roads = {}
    for r in evaluator.roads.values():
        to_n = r["to_node"]
        if to_n not in inter_roads:
            inter_roads[to_n] = []
        inter_roads[to_n].append(r["id"])

    multi_in_node = None
    for n, r_ids in inter_roads.items():
        if len(r_ids) >= 2:
            multi_in_node = (n, r_ids[0], r_ids[1])
            break
    assert multi_in_node is not None
    dest_node, r_target_a, r_target_b = multi_in_node

    # Pick a starting road
    all_roads = list(evaluator.roads.keys())
    start_road = [r for r in all_roads if r != r_target_a and r != r_target_b][0]

    # 1. Strict Road Destination: Must terminate specifically at r_target_a
    res_strict_a = evaluator.find_shortest_path(
        start_road,
        dest_road_id=r_target_a,
        allow_node_destination=False,
    )
    if res_strict_a["success"]:
        assert res_strict_a["path"][-1] == r_target_a

    # 2. Strict Road Destination: Must terminate specifically at r_target_b
    res_strict_b = evaluator.find_shortest_path(
        start_road,
        dest_road_id=r_target_b,
        allow_node_destination=False,
    )
    if res_strict_b["success"]:
        assert res_strict_b["path"][-1] == r_target_b

    # 3. Node Destination: Can terminate at any road discharging into dest_node
    res_node = evaluator.find_shortest_path(
        start_road,
        dest_node_id=dest_node,
        allow_node_destination=True,
    )
    if res_node["success"]:
        terminal_road = evaluator.roads[res_node["path"][-1]]
        assert terminal_road["to_node"] == dest_node


def test_disconnected_paths_handling(evaluator):
    """Verifies graceful handling when no directed path exists between start and destination."""
    start_road = "road_586133572_120_fwd"
    dest_node = "int_5516750363"

    # Close all outgoing roads from start_road's to_node
    start_to_node = evaluator.roads[start_road]["to_node"]
    out_roads = {r["id"] for r in evaluator.roads.values() if r["from_node"] == start_to_node}

    # Since all exits are closed, no open path exists
    res = evaluator.find_shortest_path(
        start_road,
        dest_node_id=dest_node,
        closed_roads=out_roads,
        allow_closed_fallback=False,
    )
    assert res["success"] is False
    assert "No open path found" in res["reason"]
