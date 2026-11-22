"""Capacity Estimation and Capacity-Aware Dynamic Routing Engine for UrbanFlow AI.

Phase 2: Dynamic Link Cost & Pathfinding Engine.

Provides:
- RouteCostEvaluator: Multi-attribute dynamic link impedance evaluator supporting:
  * Free-flow traversal time (length / speed_limit)
  * Numerically stable BPR congestion delay function (V/C ratio)
  * Downstream physical queue occupancy penalty
  * Advisory predictive spillback risk penalty (configurable weight)
  * Strict safeguards: closed roads, missing telemetry, zero capacity, non-negative, finite costs
  * Reactive Mode (lambda_pred = 0.0) vs Predictive Mode (lambda_pred > 0.0)
- Dynamic Directed Pathfinding:
  * NetworkX Dijkstra path calculation on directed line-graph
  * Deterministic tie-breaking on identical costs
  * Validates every returned path's adjacency, continuity, and destination
  * Supports both strict road-level destination semantics and node-level arterial gateway semantics
- Static Baseline Route Generation:
  * Preserves free-flow / distance-only shortest path for A/B benchmarking.
"""
from __future__ import annotations

# Import LightGBM first on Windows to avoid OpenMP runtime conflicts
try:
    import lightgbm as lgb
except ImportError:
    lgb = None

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple

import joblib
import networkx as nx
import numpy as np
import pandas as pd

from urbanflow.core.network_topology import NetworkTopology

logger = logging.getLogger(__name__)

DEFAULT_MAP_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "maps" / "bronx_fordham.json"


class RouteCostEvaluator:
    """Computes dynamic link impedance and finds optimal capacity-aware routes.

    Destination Semantics:
    1. Strict Road Destination (allow_node_destination=False, Default):
       - Target destination is a specific directed road segment R_dest.
       - A route is valid if and only if its terminal road is R_dest (route[-1] == R_dest).
       - Matches vehicle trip completion where trip terminates after traversing R_dest.
    2. Node Destination (allow_node_destination=True):
       - Target destination is an intersection/arterial gateway node N_dest.
       - A route is valid if its terminal road discharges into N_dest (to_node == N_dest).
    """

    DEFAULT_BPR_ALPHA: float = 0.15
    DEFAULT_BPR_BETA: float = 4.0
    DEFAULT_QUEUE_WEIGHT: float = 2.0  # scales queue occupancy by free-flow time
    DEFAULT_PRED_WEIGHT: float = 5.0   # scales advisory spillback probability by free-flow time
    CLOSED_ROAD_PENALTY: float = 100_000.0  # seconds
    MAX_VC_RATIO: float = 4.0          # cap to prevent (V/C)^beta numerical overflow
    MAX_EDGE_COST: float = 1_000_000.0 # upper ceiling in seconds

    def __init__(
        self,
        roads: Optional[Dict[str, Any]] = None,
        graph: Optional[nx.DiGraph] = None,
        map_path: Optional[Path] = None,
        mode: str = "reactive",  # "reactive" or "predictive"
        bpr_alpha: float = DEFAULT_BPR_ALPHA,
        bpr_beta: float = DEFAULT_BPR_BETA,
        queue_weight: float = DEFAULT_QUEUE_WEIGHT,
        pred_weight: float = DEFAULT_PRED_WEIGHT,
        simulator: Optional[Any] = None,
    ):
        self.mode = mode.lower()
        if self.mode not in ("reactive", "predictive"):
            raise ValueError(f"Mode must be 'reactive' or 'predictive', got '{mode}'")

        self.bpr_alpha = float(bpr_alpha)
        self.bpr_beta = float(bpr_beta)
        self.queue_weight = float(queue_weight)
        self.pred_weight = float(pred_weight) if self.mode == "predictive" else 0.0

        self.roads: Dict[str, Any] = {}
        self.graph: nx.DiGraph = nx.DiGraph()
        self.edge_to_road: Dict[Tuple[str, str], str] = {}
        self.road_to_edge: Dict[str, Tuple[str, str]] = {}

        if simulator is not None:
            self._init_from_simulator(simulator)
        elif roads is not None and graph is not None:
            self._init_from_components(roads, graph)
        else:
            self._load_from_json(map_path or DEFAULT_MAP_PATH)

    def _init_from_simulator(self, sim: Any):
        """Initializes topology from an active KinematicSimulator instance."""
        self.roads = dict(sim.roads)
        self.graph = nx.DiGraph(sim.graph)
        for r_id, r in self.roads.items():
            u, v = r.from_node, r.to_node
            self.edge_to_road[(u, v)] = r_id
            self.road_to_edge[r_id] = (u, v)

    def _init_from_components(self, roads: Dict[str, Any], graph: nx.DiGraph):
        """Initializes from existing roads dictionary and NetworkX DiGraph."""
        self.roads = dict(roads)
        self.graph = nx.DiGraph(graph)
        for r_id, r in self.roads.items():
            u = getattr(r, "from_node", None) or r.get("from_node")
            v = getattr(r, "to_node", None) or r.get("to_node")
            if u and v:
                self.edge_to_road[(u, v)] = r_id
                self.road_to_edge[r_id] = (u, v)

    def _load_from_json(self, json_path: Path):
        """Loads road network definitions from geographic JSON map."""
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"Map file not found: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for r in data.get("roads", []):
            r_id = r["id"]
            u = r["from_node"]
            v = r["to_node"]
            length = float(r.get("length", 100.0))
            speed = float(r.get("speed_limit", 11.2))
            lanes = int(r.get("lanes", 1))
            capacity = float(r.get("capacity", 1800.0 * lanes))

            self.roads[r_id] = {
                "id": r_id,
                "from_node": u,
                "to_node": v,
                "length": length,
                "speed_limit": speed,
                "lanes": lanes,
                "capacity": capacity,
            }
            self.edge_to_road[(u, v)] = r_id
            self.road_to_edge[r_id] = (u, v)

            t0 = max(0.1, length / max(0.5, speed))
            self.graph.add_edge(u, v, road_id=r_id, length=length, free_flow_time=t0)

    def set_mode(self, mode: str, pred_weight: Optional[float] = None):
        """Switches router between 'reactive' (zero predictive weight) and 'predictive'."""
        mode_clean = mode.lower()
        if mode_clean not in ("reactive", "predictive"):
            raise ValueError(f"Mode must be 'reactive' or 'predictive', got '{mode}'")
        self.mode = mode_clean
        if pred_weight is not None:
            self.pred_weight = float(pred_weight)
        elif self.mode == "reactive":
            self.pred_weight = 0.0
        elif self.mode == "predictive" and self.pred_weight == 0.0:
            self.pred_weight = self.DEFAULT_PRED_WEIGHT

    def compute_edge_cost(
        self,
        road_id: str,
        telemetry: Optional[Dict[str, Any]] = None,
        p_spillback: Optional[float] = None,
        is_closed: bool = False,
    ) -> float:
        """Calculates generalized link impedance for a single directed road.

        Formula:
            cost(e) = t_0(e) * [ 1.0 + alpha * (V/C_safe)^beta ]
                    + queue_weight * t_0(e) * q_occ(e)
                    + lambda_pred * t_0(e) * P_spillback(e)

        Guarantees:
            - Strictly positive: cost >= t_0(e) > 0.
            - Bounded: cost <= MAX_EDGE_COST.
            - Numerically stable: No NaNs, infinities, or negative costs.
        """
        road = self.roads.get(road_id)
        if not road:
            return self.CLOSED_ROAD_PENALTY

        length_m = getattr(road, "length", None) or (road.get("length") if isinstance(road, dict) else 100.0)
        length_m = getattr(road, "routing_length", None) or length_m
        speed_mps = getattr(road, "speed_limit", None) or (road.get("speed_limit") if isinstance(road, dict) else 11.2)

        try:
            length_m = float(length_m)
            speed_mps = float(speed_mps)
        except (TypeError, ValueError):
            length_m, speed_mps = 100.0, 11.2

        if math.isnan(length_m) or math.isinf(length_m) or length_m <= 0:
            length_m = 100.0
        if math.isnan(speed_mps) or math.isinf(speed_mps) or speed_mps <= 0.5:
            speed_mps = 11.2

        t_0 = max(0.1, length_m / speed_mps)

        # Immediate closed road penalty
        if is_closed:
            return self.CLOSED_ROAD_PENALTY

        vc_safe = 0.0
        q_occ = 0.0

        if telemetry is not None:
            if hasattr(telemetry, "to_dict"):
                telemetry = telemetry.to_dict()

            if isinstance(telemetry, dict):
                if telemetry.get("is_closed", False):
                    return self.CLOSED_ROAD_PENALTY

                op_cap = telemetry.get("operational_capacity")
                try:
                    op_cap = float(op_cap) if op_cap is not None else 1800.0
                except (TypeError, ValueError):
                    op_cap = 1800.0

                if op_cap <= 0.0:
                    return self.CLOSED_ROAD_PENALTY

                vc_val = telemetry.get("capacity_utilization")
                if vc_val is None:
                    vc_val = telemetry.get("vc_ratio")
                if vc_val is None:
                    flow = telemetry.get("flow_veh_per_hour") or telemetry.get("flow_veh_hr") or 0.0
                    try:
                        flow = float(flow)
                    except (TypeError, ValueError):
                        flow = 0.0
                    vc_val = flow / max(1.0, op_cap)
                else:
                    try:
                        vc_val = float(vc_val)
                    except (TypeError, ValueError):
                        vc_val = 0.0

                if math.isnan(vc_val) or math.isinf(vc_val):
                    vc_val = 0.0
                vc_safe = min(self.MAX_VC_RATIO, max(0.0, vc_val))

                q_len = telemetry.get("queue_length_meters")
                if q_len is None:
                    q_len = telemetry.get("queue_length_m") or 0.0
                try:
                    q_len = float(q_len)
                except (TypeError, ValueError):
                    q_len = 0.0

                if math.isnan(q_len) or math.isinf(q_len) or q_len < 0:
                    q_len = 0.0
                q_occ = min(1.0, max(0.0, q_len / max(10.0, length_m)))

        # 1. BPR Congestion delay
        bpr_cost = t_0 * (1.0 + self.bpr_alpha * (vc_safe ** self.bpr_beta))

        # 2. Downstream queue occupancy penalty
        queue_penalty = self.queue_weight * t_0 * q_occ

        # 3. Advisory predictive spillback risk penalty (predictive mode only)
        pred_penalty = 0.0
        if self.mode == "predictive" and self.pred_weight > 0.0 and p_spillback is not None:
            try:
                risk = float(p_spillback)
            except (TypeError, ValueError):
                risk = 0.0
            if math.isnan(risk) or math.isinf(risk):
                risk = 0.0
            risk = min(1.0, max(0.0, risk))
            pred_penalty = self.pred_weight * t_0 * risk

        total_cost = bpr_cost + queue_penalty + pred_penalty

        if math.isnan(total_cost) or math.isinf(total_cost) or total_cost < t_0:
            total_cost = t_0

        return min(self.MAX_EDGE_COST, total_cost)

    def compute_all_edge_costs(
        self,
        telemetry_map: Optional[Dict[str, Any]] = None,
        p_spillback_map: Optional[Dict[str, float]] = None,
        closed_roads: Optional[Set[str]] = None,
    ) -> Dict[str, float]:
        """Calculates dynamic link costs across all roads in the network."""
        costs = {}
        for r_id in self.roads:
            telem = telemetry_map.get(r_id) if telemetry_map else None
            p_risk = p_spillback_map.get(r_id) if p_spillback_map else None
            is_closed = (closed_roads is not None and r_id in closed_roads)
            costs[r_id] = self.compute_edge_cost(r_id, telem, p_risk, is_closed)
        return costs

    def find_shortest_path(
        self,
        start_road_id: str,
        dest_road_id: Optional[str] = None,
        dest_node_id: Optional[str] = None,
        telemetry_map: Optional[Dict[str, Any]] = None,
        p_spillback_map: Optional[Dict[str, float]] = None,
        closed_roads: Optional[Set[str]] = None,
        allow_node_destination: bool = False,
        allow_closed_fallback: bool = False,
        avoid_roads: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """Computes dynamic minimum-cost directed path from start_road to destination.

        Deterministic Tie-Breaking:
            Augments edge weights with a micro-offset derived from MD5(road_id) modulo 10,000 * 1e-9.
            Ensures deterministic total ordering between identical-cost alternatives.
        """
        if start_road_id not in self.roads:
            return {
                "success": False,
                "reason": f"Start road '{start_road_id}' not found in network topology.",
                "path": [],
                "cost": 0.0,
            }

        start_road = self.roads[start_road_id]
        origin_node = getattr(start_road, "to_node", None) or start_road["to_node"]
        start_from = getattr(start_road, "from_node", None) or start_road["from_node"]

        # Validate destination target
        if dest_road_id is not None:
            if dest_road_id not in self.roads:
                return {
                    "success": False,
                    "reason": f"Destination road '{dest_road_id}' not found in network topology.",
                    "path": [],
                    "cost": 0.0,
                }
            dest_road = self.roads[dest_road_id]
            target_dest_node = getattr(dest_road, "to_node", None) or dest_road["to_node"]
            target_dest_from = getattr(dest_road, "from_node", None) or dest_road["from_node"]
        elif dest_node_id is not None:
            if dest_node_id not in self.graph:
                return {
                    "success": False,
                    "reason": f"Destination node '{dest_node_id}' not found in graph.",
                    "path": [],
                    "cost": 0.0,
                }
            target_dest_node = dest_node_id
            target_dest_from = None
            allow_node_destination = True
        else:
            return {
                "success": False,
                "reason": "Either dest_road_id or dest_node_id must be provided.",
                "path": [],
                "cost": 0.0,
            }

        # Check trivial arrival (already on destination road)
        if dest_road_id is not None and start_road_id == dest_road_id:
            c = self.compute_edge_cost(dest_road_id, telemetry_map.get(dest_road_id) if telemetry_map else None)
            return {
                "success": True,
                "reason": "Origin is already destination road.",
                "path": [dest_road_id],
                "cost": c,
            }

        # Compute dynamic edge costs
        edge_costs = self.compute_all_edge_costs(telemetry_map, p_spillback_map, closed_roads)

        # Dynamic Dijkstra weight function with deterministic tie-breaking and U-turn avoidance
        def get_weight(u: str, v: str, data: Dict[str, Any]) -> float:
            # Heavily penalize immediate U-turn reversal from start road
            if u == origin_node and v == start_from and origin_node != start_from:
                return RouteCostEvaluator.CLOSED_ROAD_PENALTY * 10.0

            r_id = self.edge_to_road.get((u, v))
            if not r_id:
                return RouteCostEvaluator.CLOSED_ROAD_PENALTY
            if avoid_roads and r_id in avoid_roads:
                return RouteCostEvaluator.CLOSED_ROAD_PENALTY * 2.0
            base_cost = edge_costs.get(r_id, data.get("free_flow_time", 10.0))
            h_int = int(hashlib.md5(r_id.encode("utf-8")).hexdigest()[:6], 16)
            tie_breaker = (h_int % 10000) * 1e-9
            return base_cost + tie_breaker

        # Pathfinding Strategy based on destination semantics
        try:
            if not allow_node_destination and dest_road_id is not None:
                # Strict Road Destination: Path must enter dest_road.from_node and traverse dest_road
                if origin_node == target_dest_from:
                    road_path = [dest_road_id]
                else:
                    node_path = nx.dijkstra_path(self.graph, origin_node, target_dest_from, weight=get_weight)
                    road_path = []
                    for i in range(len(node_path) - 1):
                        edge = (node_path[i], node_path[i + 1])
                        road_path.append(self.edge_to_road[edge])
                    road_path.append(dest_road_id)
            else:
                # Node Destination: Path terminates at target_dest_node
                node_path = nx.dijkstra_path(self.graph, origin_node, target_dest_node, weight=get_weight)
                road_path = []
                for i in range(len(node_path) - 1):
                    edge = (node_path[i], node_path[i + 1])
                    road_path.append(self.edge_to_road[edge])

        except (nx.NetworkXNoPath, nx.NodeNotFound) as exc:
            return {
                "success": False,
                "reason": f"No directed path found from node '{origin_node}' to destination: {exc}",
                "path": [],
                "cost": float("inf"),
            }

        # Validate path
        val_res = self.validate_path(
            road_path,
            start_road_id=start_road_id,
            dest_road_id=dest_road_id,
            dest_node_id=dest_node_id,
            allow_node_destination=allow_node_destination,
        )
        if not val_res["valid"]:
            return {
                "success": False,
                "reason": val_res["reason"],
                "path": road_path,
                "cost": float("inf"),
            }

        # Check if path traverses closed roads
        if closed_roads and not allow_closed_fallback:
            contained_closed = [r for r in road_path if r in closed_roads]
            if contained_closed:
                return {
                    "success": False,
                    "reason": f"No open path found; all available paths require traversing closed road(s): {contained_closed}",
                    "path": road_path,
                    "cost": float("inf"),
                }

        total_cost = sum(edge_costs.get(r, 0.0) for r in road_path)
        return {
            "success": True,
            "reason": "Optimal path computed successfully.",
            "path": road_path,
            "cost": round(total_cost, 4),
            "hop_count": len(road_path),
            "mode": self.mode,
            "allow_node_destination": allow_node_destination,
        }

    def find_candidate_paths(
        self,
        start_road_id: str,
        dest_road_id: Optional[str] = None,
        dest_node_id: Optional[str] = None,
        telemetry_map: Optional[Dict[str, Any]] = None,
        p_spillback_map: Optional[Dict[str, float]] = None,
        closed_roads: Optional[Set[str]] = None,
        allow_node_destination: bool = False,
        allow_closed_fallback: bool = False,
        k: int = 5,
        avoid_roads: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Discovers up to k directed candidate alternative paths using K-shortest simple paths and corridor avoidance.

        Guarantees:
            - Valid directed continuity, destination preservation, and no-U-turn constraints.
            - Deterministic tie-breaking and numerical stability.
            - Candidates returned sorted in ascending order of generalized dynamic impedance.
        """
        if start_road_id not in self.roads or start_road_id.startswith("CONN__"):
            return []

        start_road = self.roads[start_road_id]
        origin_node = getattr(start_road, "to_node", None) or start_road["to_node"]
        start_from = getattr(start_road, "from_node", None) or start_road["from_node"]

        # Validate destination target
        if dest_road_id is not None:
            if dest_road_id not in self.roads:
                return []
            dest_road = self.roads[dest_road_id]
            target_dest_node = getattr(dest_road, "to_node", None) or dest_road["to_node"]
            target_dest_from = getattr(dest_road, "from_node", None) or dest_road["from_node"]
        elif dest_node_id is not None:
            if dest_node_id not in self.graph:
                return []
            target_dest_node = dest_node_id
            target_dest_from = None
            allow_node_destination = True
        else:
            return []

        # Trivial arrival check
        if dest_road_id is not None and start_road_id == dest_road_id:
            c = self.compute_edge_cost(dest_road_id, telemetry_map.get(dest_road_id) if telemetry_map else None)
            return [{
                "success": True,
                "reason": "Origin is already destination road.",
                "path": [dest_road_id],
                "cost": round(c, 4),
                "hop_count": 1,
                "mode": self.mode,
                "allow_node_destination": allow_node_destination,
            }]

        edge_costs = self.compute_all_edge_costs(telemetry_map, p_spillback_map, closed_roads)

        def get_weight(u: str, v: str, data: Dict[str, Any]) -> float:
            if u == origin_node and v == start_from and origin_node != start_from:
                return RouteCostEvaluator.CLOSED_ROAD_PENALTY * 10.0
            r_id = self.edge_to_road.get((u, v))
            if not r_id:
                return RouteCostEvaluator.CLOSED_ROAD_PENALTY
            base_cost = edge_costs.get(r_id, data.get("free_flow_time", 10.0))
            h_int = int(hashlib.md5(r_id.encode("utf-8")).hexdigest()[:6], 16)
            tie_breaker = (h_int % 10000) * 1e-9
            return base_cost + tie_breaker

        if not allow_node_destination and dest_road_id is not None:
            target_node = target_dest_from
        else:
            target_node = target_dest_node

        raw_node_paths = []
        if not allow_node_destination and dest_road_id is not None and origin_node == target_dest_from:
            raw_node_paths.append([origin_node, target_dest_node])
        else:
            try:
                path_gen = nx.shortest_simple_paths(self.graph, origin_node, target_node, weight=get_weight)
                for _ in range(max(1, k)):
                    try:
                        raw_node_paths.append(next(path_gen))
                    except StopIteration:
                        break
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass

        # Corridor avoidance search: explicitly penalize avoid_roads to discover distinct corridor
        if avoid_roads:
            def get_avoid_weight(u: str, v: str, data: Dict[str, Any]) -> float:
                r_id = self.edge_to_road.get((u, v))
                if r_id in avoid_roads:
                    return RouteCostEvaluator.CLOSED_ROAD_PENALTY * 2.0
                return get_weight(u, v, data)

            try:
                if not (not allow_node_destination and dest_road_id is not None and origin_node == target_dest_from):
                    avoid_node_path = nx.dijkstra_path(self.graph, origin_node, target_node, weight=get_avoid_weight)
                    if avoid_node_path not in raw_node_paths:
                        raw_node_paths.append(avoid_node_path)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass

        candidates: List[Dict[str, Any]] = []
        seen_paths: Set[Tuple[str, ...]] = set()

        for node_path in raw_node_paths:
            road_path: List[str] = []
            valid_conversion = True
            for i in range(len(node_path) - 1):
                edge = (node_path[i], node_path[i + 1])
                if edge in self.edge_to_road:
                    road_path.append(self.edge_to_road[edge])
                else:
                    valid_conversion = False
                    break
            if not valid_conversion or not road_path:
                continue

            if not allow_node_destination and dest_road_id is not None and origin_node != target_dest_from:
                road_path.append(dest_road_id)

            path_tuple = tuple(road_path)
            if path_tuple in seen_paths:
                continue
            seen_paths.add(path_tuple)

            val_res = self.validate_path(
                road_path,
                start_road_id=start_road_id,
                dest_road_id=dest_road_id,
                dest_node_id=dest_node_id,
                allow_node_destination=allow_node_destination,
            )
            if not val_res["valid"]:
                continue

            if closed_roads and not allow_closed_fallback:
                if any(r in closed_roads for r in road_path):
                    continue

            total_cost = sum(edge_costs.get(r, 0.0) for r in road_path)
            candidates.append({
                "success": True,
                "reason": "Candidate path valid.",
                "path": road_path,
                "cost": round(total_cost, 4),
                "hop_count": len(road_path),
                "mode": self.mode,
                "allow_node_destination": allow_node_destination,
            })

        candidates.sort(key=lambda c: c["cost"])
        return candidates

    def find_static_shortest_path(
        self,
        start_road_id: str,
        dest_road_id: Optional[str] = None,
        dest_node_id: Optional[str] = None,
        allow_node_destination: bool = False,
    ) -> Dict[str, Any]:
        """Computes baseline shortest path using purely static free-flow travel time (no dynamic telemetry)."""
        # Save current mode
        old_mode = self.mode
        self.mode = "reactive"
        res = self.find_shortest_path(
            start_road_id=start_road_id,
            dest_road_id=dest_road_id,
            dest_node_id=dest_node_id,
            telemetry_map=None,
            p_spillback_map=None,
            closed_roads=None,
            allow_node_destination=allow_node_destination,
        )
        self.mode = old_mode
        return res

    def validate_path(
        self,
        path: List[str],
        start_road_id: Optional[str] = None,
        dest_road_id: Optional[str] = None,
        dest_node_id: Optional[str] = None,
        allow_node_destination: bool = False,
    ) -> Dict[str, Any]:
        """Validates that a road path is continuous, cycle-free, and reaches the target destination."""
        if not path:
            return {"valid": False, "reason": "Path is empty."}

        # 1. Road existence
        for r_id in path:
            if r_id not in self.roads:
                return {"valid": False, "reason": f"Road ID '{r_id}' not found in network."}

        # 2. Connection from start_road if provided
        if start_road_id is not None and start_road_id in self.roads:
            start_r = self.roads[start_road_id]
            first_r = self.roads[path[0]]
            s_to = getattr(start_r, "to_node", None) or start_r["to_node"]
            f_from = getattr(first_r, "from_node", None) or first_r["from_node"]
            s_from = getattr(start_r, "from_node", None) or start_r["from_node"]
            f_to = getattr(first_r, "to_node", None) or first_r["to_node"]

            if s_to != f_from:
                return {
                    "valid": False,
                    "reason": f"Discontinuity: start road '{start_road_id}' (to_node: {s_to}) does not connect to first road '{path[0]}' (from_node: {f_from}).",
                }
            if s_from != s_to and f_to == s_from:
                return {
                    "valid": False,
                    "reason": f"Unintended U-turn: first road '{path[0]}' reverses back to node '{s_from}'.",
                }

        # 3. Sequential link continuity & U-turn checks
        for i in range(len(path) - 1):
            r_curr = self.roads[path[i]]
            r_next = self.roads[path[i + 1]]
            c_to = getattr(r_curr, "to_node", None) or r_curr["to_node"]
            n_from = getattr(r_next, "from_node", None) or r_next["from_node"]
            c_from = getattr(r_curr, "from_node", None) or r_curr["from_node"]
            n_to = getattr(r_next, "to_node", None) or r_next["to_node"]

            if c_to != n_from:
                return {
                    "valid": False,
                    "reason": f"Discontinuity: road '{path[i]}' (to_node: {c_to}) does not connect to '{path[i+1]}' (from_node: {n_from}).",
                }
            if c_from != c_to and n_to == c_from:
                return {
                    "valid": False,
                    "reason": f"Unintended U-turn: road '{path[i+1]}' reverses back to node '{c_from}'.",
                }

        # 4. Destination verification
        last_r = self.roads[path[-1]]
        last_to = getattr(last_r, "to_node", None) or last_r["to_node"]

        if dest_road_id is not None:
            dest_r = self.roads[dest_road_id]
            d_to = getattr(dest_r, "to_node", None) or dest_r["to_node"]

            if not allow_node_destination:
                if path[-1] != dest_road_id:
                    return {
                        "valid": False,
                        "reason": f"Destination road mismatch: path terminates at '{path[-1]}' instead of '{dest_road_id}'.",
                    }
            else:
                if last_to != d_to:
                    return {
                        "valid": False,
                        "reason": f"Destination node mismatch: path terminates at node '{last_to}' instead of node '{d_to}'.",
                    }
        elif dest_node_id is not None:
            if last_to != dest_node_id:
                return {
                    "valid": False,
                    "reason": f"Destination node mismatch: path terminates at node '{last_to}' instead of node '{dest_node_id}'.",
                }

        return {"valid": True, "reason": "Path is valid."}


class CapacityAwareRouter:
    """Estimates dynamic road capacities and computes capacity-aware route diversions."""

    def __init__(self):
        self.incidents: Dict[str, float] = {}

    def estimate_effective_capacity(self, road_id: str, base_capacity: float = 1800.0) -> float:
        """Computes current real-time road capacity accounting for weather and active incidents."""
        incident_penalty = self.incidents.get(road_id, 0.0)
        return max(300.0, base_capacity * (1.0 - incident_penalty))

    def evaluate_diversion_recommendation(
        self,
        current_route: List[str],
        congestion_predictions: Dict[str, float],
    ) -> Dict[str, Any]:
        """Calculates alternate diversion routes when downstream roads approach capacity."""
        should_divert = False
        congested_edge = None

        for edge in current_route:
            risk = congestion_predictions.get(edge, 0.0)
            if risk > 0.65:
                should_divert = True
                congested_edge = edge
                break

        return {
            "should_divert": should_divert,
            "triggering_edge": congested_edge,
            "recommended_action": "DIVERT_VIA_PERIMETER_AVENUE" if should_divert else "MAINTAIN_ROUTE",
        }


class OnlineObservationBuffer:
    """Maintains a rolling historical telemetry buffer across road segments.

    Required for deriving the exact 28 features (16 local lags/roll stats + 12 downstream network features)
    used by the 10-minute network-aware spillback classifier.
    """

    MAX_HISTORY: int = 4  # 4 snapshots at 5s sampling = [t-15s, t-10s, t-5s, t]

    def __init__(self, network_topology: Optional[NetworkTopology] = None):
        self.topology = network_topology or NetworkTopology.get_instance()
        self.buffer: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.last_update_time: float = -1.0

    def reset(self) -> None:
        """Clears all historical snapshots and resets update timestamp."""
        self.buffer.clear()
        self.last_update_time = -1.0

    def update(self, telemetry_map: Dict[str, Any], timestamp: float):
        """Appends a new macroscopic telemetry snapshot for each road."""
        self.last_update_time = float(timestamp)
        if isinstance(telemetry_map, dict) and "road_heatmap" in telemetry_map and isinstance(telemetry_map["road_heatmap"], dict):
            telemetry_map = telemetry_map["road_heatmap"]
        for road_id, telem in telemetry_map.items():
            if hasattr(telem, "to_dict"):
                telem = telem.to_dict()
            if not isinstance(telem, dict):
                continue
            snap = {
                "timestamp": float(telem.get("timestamp", timestamp)),
                "avg_speed_kmh": float(telem.get("average_speed_kmh") if telem.get("average_speed_kmh") is not None else telem.get("avg_speed_kmh", 40.0)),
                "density_per_km_lane": float(telem.get("density_per_km_lane") if telem.get("density_per_km_lane") is not None else telem.get("density_veh_km_lane", 0.0)),
                "queue_length_meters": float(telem.get("queue_length_meters") if telem.get("queue_length_meters") is not None else telem.get("queue_length_m", 0.0)),
                "operational_capacity": float(telem.get("operational_capacity", 1800.0)),
                "capacity_utilization": float(telem.get("capacity_utilization") if telem.get("capacity_utilization") is not None else telem.get("vc_ratio", 0.0)),
                "occupancy_ratio": float(telem.get("occupancy_ratio", 0.0)),
                "incident_active": 1.0 if telem.get("incident_active", False) else 0.0,
            }
            hist = self.buffer[road_id]
            hist.append(snap)
            if len(hist) > self.MAX_HISTORY:
                hist.pop(0)

    def is_road_ready(self, road_id: str) -> bool:
        """Returns True if the road has sufficient temporal history (4 snapshots)."""
        return len(self.buffer.get(road_id, [])) >= self.MAX_HISTORY

    def build_feature_dict(self, road_id: str) -> Optional[Dict[str, float]]:
        """Constructs the 28 features matching the training feature schema."""
        hist = self.buffer.get(road_id)
        if not hist or len(hist) < self.MAX_HISTORY:
            return None

        obs_3, obs_2, obs_1, obs_t = hist[-4], hist[-3], hist[-2], hist[-1]

        # 1. Local Speeds
        s1 = obs_1["avg_speed_kmh"]
        s2 = obs_2["avg_speed_kmh"]
        s3 = obs_3["avg_speed_kmh"]
        speeds_4 = [obs_3["avg_speed_kmh"], obs_2["avg_speed_kmh"], obs_1["avg_speed_kmh"], obs_t["avg_speed_kmh"]]
        s_mean = float(np.mean(speeds_4))
        s_std = float(np.std(speeds_4, ddof=1)) if len(speeds_4) > 1 else 0.0

        # 2. Local Densities
        d_mean = float(np.mean([obs_3["density_per_km_lane"], obs_2["density_per_km_lane"], obs_1["density_per_km_lane"], obs_t["density_per_km_lane"]]))

        # 3. Local Queues
        q_mean = float(np.mean([obs_3["queue_length_meters"], obs_2["queue_length_meters"], obs_1["queue_length_meters"], obs_t["queue_length_meters"]]))

        # 4. Multi-hop Downstream Line Graph Sets
        d1 = self.topology.get_downstream_roads(road_id, 1)
        d2 = self.topology.get_downstream_roads(road_id, 2)
        d3 = self.topology.get_downstream_roads(road_id, 3)
        local_speed_limit = self.topology.get_road_speed_limit(road_id)  # Unscaled m/s matching offline campaign training feature convention

        def get_curr(r: str) -> Dict[str, Any]:
            if r in self.buffer and self.buffer[r]:
                return self.buffer[r][-1]
            return {
                "avg_speed_kmh": local_speed_limit,
                "density_per_km_lane": 0.0,
                "queue_length_meters": 0.0,
                "operational_capacity": 1800.0,
                "capacity_utilization": 0.0,
                "occupancy_ratio": 0.0,
                "incident_active": 0.0,
            }

        # 1-Hop Features
        if not d1:
            is_perimeter = 1.0
            d1_max_q_ratio = 0.0
            d1_min_v = local_speed_limit
            d1_max_occ = 0.0
            d1_mean_vc = 0.0
            speed_gradient = 0.0
            queue_delta_15s = 0.0
        else:
            is_perimeter = 0.0
            curr_d1 = [get_curr(r) for r in d1]
            q_ratios = [c["queue_length_meters"] / max(1.0, self.topology.get_road_length(r)) for c, r in zip(curr_d1, d1)]
            v_vals = [c["avg_speed_kmh"] for c in curr_d1]
            occ_vals = [c["occupancy_ratio"] for c in curr_d1]
            vc_vals = [c["capacity_utilization"] for c in curr_d1]

            d1_max_q_ratio = float(np.max(q_ratios))
            d1_min_v = float(np.min(v_vals))
            d1_max_occ = float(np.max(occ_vals))
            d1_mean_vc = float(np.mean(vc_vals))
            speed_gradient = float(obs_t["avg_speed_kmh"] - d1_min_v)

            # 15s downstream queue delta: max_q1(t) - max_q1(t-3)
            max_q1_t = max(c["queue_length_meters"] for c in curr_d1)
            q1_t3 = []
            for r in d1:
                if r in self.buffer and len(self.buffer[r]) >= 4:
                    q1_t3.append(self.buffer[r][-4]["queue_length_meters"])
                elif r in self.buffer and self.buffer[r]:
                    q1_t3.append(self.buffer[r][0]["queue_length_meters"])
                else:
                    q1_t3.append(0.0)
            max_q1_t3 = max(q1_t3) if q1_t3 else 0.0
            queue_delta_15s = float(max_q1_t - max_q1_t3)

        # 2-Hop Features
        if not d2:
            d2_max_q_ratio = 0.0
            d2_min_v = local_speed_limit
            d2_max_occ = 0.0
        else:
            curr_d2 = [get_curr(r) for r in d2]
            q_ratios_2 = [c["queue_length_meters"] / max(1.0, self.topology.get_road_length(r)) for c, r in zip(curr_d2, d2)]
            d2_max_q_ratio = float(np.max(q_ratios_2))
            d2_min_v = float(np.min([c["avg_speed_kmh"] for c in curr_d2]))
            d2_max_occ = float(np.max([c["occupancy_ratio"] for c in curr_d2]))

        # 3-Hop Features
        if not d3:
            d3_max_q_ratio = 0.0
            d3_min_v = local_speed_limit
        else:
            curr_d3 = [get_curr(r) for r in d3]
            q_ratios_3 = [c["queue_length_meters"] / max(1.0, self.topology.get_road_length(r)) for c, r in zip(curr_d3, d3)]
            d3_max_q_ratio = float(np.max(q_ratios_3))
            d3_min_v = float(np.min([c["avg_speed_kmh"] for c in curr_d3]))

        feat_dict = {
            "avg_speed_lag_1": float(s1),
            "avg_speed_lag_2": float(s2),
            "avg_speed_lag_3": float(s3),
            "avg_speed_roll_mean": float(s_mean),
            "avg_speed_roll_std": float(s_std),
            "density_lag_1": float(obs_1["density_per_km_lane"]),
            "density_lag_2": float(obs_2["density_per_km_lane"]),
            "density_lag_3": float(obs_3["density_per_km_lane"]),
            "density_roll_mean": float(d_mean),
            "queue_lag_1": float(obs_1["queue_length_meters"]),
            "queue_lag_2": float(obs_2["queue_length_meters"]),
            "queue_lag_3": float(obs_3["queue_length_meters"]),
            "queue_roll_mean": float(q_mean),
            "operational_capacity": float(obs_t["operational_capacity"]),
            "vc_ratio": float(obs_t["capacity_utilization"]),
            "incident_active": float(obs_t["incident_active"]),
            "down_1hop_max_queue_ratio": float(d1_max_q_ratio),
            "down_1hop_min_speed_kmh": float(d1_min_v),
            "down_1hop_max_occupancy": float(d1_max_occ),
            "down_1hop_mean_vc_ratio": float(d1_mean_vc),
            "down_2hop_max_queue_ratio": float(d2_max_q_ratio),
            "down_2hop_min_speed_kmh": float(d2_min_v),
            "down_2hop_max_occupancy": float(d2_max_occ),
            "down_3hop_max_queue_ratio": float(d3_max_q_ratio),
            "down_3hop_min_speed_kmh": float(d3_min_v),
            "downstream_speed_gradient": float(speed_gradient),
            "downstream_queue_delta_15s": float(queue_delta_15s),
            "is_perimeter_exit": float(is_perimeter),
        }
        return feat_dict


@dataclass
class LinkReservation:
    """Represents a committed physical and flow capacity reservation on a directed road link."""

    vehicle_id: str
    road_id: str
    reserved_time: float
    expected_arrival_time: float
    vehicle_length_m: float = 7.5  # standard vehicle footprint (length 5m + safety gap 2.5m)


class PredictiveRoutingController:
    """Predictive, capacity-aware dynamic routing controller for UrbanFlow AI."""

    DEFAULT_COOLDOWN_SEC: float = 180.0
    DEFAULT_MIN_IMPROVEMENT_SEC: float = 45.0
    DEFAULT_MIN_IMPROVEMENT_RATIO: float = 0.15
    DEFAULT_VC_CONGESTION_THRESHOLD: float = 0.80
    DEFAULT_VC_RECOVERY_THRESHOLD: float = 0.70
    DEFAULT_REQUIRED_RECOVERY_WINDOWS: int = 2
    DEFAULT_MAX_DIVERSION_QUOTA: float = 0.25
    DEFAULT_EVALUATION_INTERVAL_SEC: float = 15.0

    DEFAULT_CLASSIFIER_PATH: Path = (
        Path(__file__).resolve().parent.parent.parent
        / "data"
        / "models"
        / "campaign_network_classifiers"
        / "lgb_network_spillback_classifier_10min.joblib"
    )
    EXPECTED_CLASSIFIER_SHA256: str = (
        "9B93E5DB672F692E509DC3012658413BC713FC6348CCD5D044B33470FF568C54"
    )

    def __init__(
        self,
        route_evaluator: Optional[RouteCostEvaluator] = None,
        mode: str = "reactive",
        classifier_path: Optional[Path] = None,
        max_diversion_quota: float = DEFAULT_MAX_DIVERSION_QUOTA,
        cooldown_sec: float = DEFAULT_COOLDOWN_SEC,
        min_improvement_sec: float = DEFAULT_MIN_IMPROVEMENT_SEC,
        min_improvement_ratio: float = DEFAULT_MIN_IMPROVEMENT_RATIO,
        vc_admission_cap: float = 0.80,
        evaluation_interval_sec: float = DEFAULT_EVALUATION_INTERVAL_SEC,
        strict_quota_floor: bool = True,
        allow_node_destination: bool = True,
    ):
        self.mode = mode.lower()
        if self.mode not in ("off", "reactive", "predictive"):
            raise ValueError(f"Mode must be 'off', 'reactive', or 'predictive', got '{mode}'")

        eval_mode = "reactive" if self.mode == "off" else self.mode
        self.route_evaluator = route_evaluator or RouteCostEvaluator(mode=eval_mode)
        self.route_evaluator.set_mode(eval_mode)

        self.max_diversion_quota = float(max_diversion_quota)
        self.cooldown_sec = float(cooldown_sec)
        self.min_improvement_sec = float(min_improvement_sec)
        self.min_improvement_ratio = float(min_improvement_ratio)
        self.vc_admission_cap = float(vc_admission_cap)
        self.evaluation_interval_sec = float(evaluation_interval_sec)
        self.strict_quota_floor = bool(strict_quota_floor)
        self.allow_node_destination = bool(allow_node_destination)

        self.observation_buffer = OnlineObservationBuffer(NetworkTopology.get_instance())

        # Persistent capacity reservations: road_id -> {vehicle_id: LinkReservation}
        self.active_reservations: Dict[str, Dict[str, LinkReservation]] = defaultdict(dict)
        self.storage_safety_limit: float = 0.75  # Capped at 75% of link physical storage capacity

        # Hysteresis state
        self.congested_corridors: Dict[str, bool] = {}
        self.consecutive_clear_windows: Dict[str, int] = {}

        # Cooldown state: vehicle_id -> last reroute timestamp
        self.last_reroute_time: Dict[str, float] = {}

        # Classifier state
        self.calibrator = None
        self.expected_features: List[str] = []
        self.threshold: float = 0.180
        self.predictive_available: bool = False
        self.predictive_status: str = "INITIALIZING"

        self._init_classifier(classifier_path or self.DEFAULT_CLASSIFIER_PATH)

    def set_mode(self, mode: str) -> None:
        """Sets the routing mode to 'off', 'reactive', or 'predictive'."""
        clean_mode = str(mode).strip().lower()
        if clean_mode not in ("off", "reactive", "predictive"):
            raise ValueError(f"Mode must be 'off', 'reactive', or 'predictive', got '{mode}'")
        self.mode = clean_mode
        eval_mode = "reactive" if self.mode == "off" else self.mode
        self.route_evaluator.set_mode(eval_mode)

    def reset(self) -> None:
        """Clears all dynamic routing history, cooldowns, reservations, and buffers."""
        self.active_reservations.clear()
        self.congested_corridors.clear()
        self.consecutive_clear_windows.clear()
        self.last_reroute_time.clear()
        self.observation_buffer.reset()

    def _init_classifier(self, path: Path):
        """Loads the 10-minute network classifier read-only with cryptographic hash validation."""
        path = Path(path)
        if not path.exists():
            self.predictive_available = False
            self.predictive_status = f"UNAVAILABLE_FILE_NOT_FOUND: {path.name}"
            logger.warning(self.predictive_status)
            return

        # Cryptographic SHA-256 verification before loading
        h = hashlib.sha256(path.read_bytes()).hexdigest().upper()
        if h != self.EXPECTED_CLASSIFIER_SHA256:
            self.predictive_available = False
            self.predictive_status = f"UNAVAILABLE_HASH_MISMATCH: expected {self.EXPECTED_CLASSIFIER_SHA256[:8]}, got {h[:8]}"
            logger.error(self.predictive_status)
            return

        try:
            artifact = joblib.load(path)
            self.calibrator = artifact["calibrator"]
            self.expected_features = list(artifact["features"])
            self.threshold = float(artifact.get("threshold", 0.180))
            self.predictive_available = True
            self.predictive_status = "ACTIVE"
        except Exception as exc:
            self.predictive_available = False
            self.predictive_status = f"UNAVAILABLE_LOAD_ERROR: {exc}"
            logger.error(self.predictive_status)

    def clean_reservations(self, sim: Any, current_time: float):
        """Cleans stale reservations for vehicles that completed trips, passed the road, or rerouted."""
        active_veh_dict = getattr(sim, "vehicles", {})
        for road_id in list(self.active_reservations.keys()):
            res_dict = self.active_reservations[road_id]
            for veh_id in list(res_dict.keys()):
                res = res_dict[veh_id]
                veh = active_veh_dict.get(veh_id)
                if veh is None:
                    # Vehicle completed trip or never entered/removed
                    res_dict.pop(veh_id, None)
                    continue
                if road_id not in veh.route:
                    # Vehicle rerouted away from this road
                    res_dict.pop(veh_id, None)
                    continue
                try:
                    idx = veh.route.index(road_id)
                except ValueError:
                    idx = -1
                if idx <= veh.route_index:
                    # Vehicle already reached or traversed this road; now captured in physical telemetry
                    res_dict.pop(veh_id, None)
                else:
                    # Vehicle still upstream. Keep reservation and handle upstream queue delay conservatively
                    if current_time > res.expected_arrival_time:
                        res.expected_arrival_time = current_time

    def get_active_reservations(self, road_id: str, exclude_vehicle_id: Optional[str] = None) -> List[LinkReservation]:
        """Returns active reservations for a given road segment, optionally excluding a specific vehicle."""
        res_dict = self.active_reservations.get(road_id, {})
        return [r for v_id, r in res_dict.items() if v_id != exclude_vehicle_id]

    def add_route_reservations(self, vehicle: Any, path: List[str], current_time: float):
        """Registers persistent capacity reservations along the vehicle's new route."""
        # Release any old reservations for this vehicle that are not on the new path
        for r_id in list(self.active_reservations.keys()):
            if vehicle.id in self.active_reservations[r_id] and r_id not in path:
                self.active_reservations[r_id].pop(vehicle.id, None)

        cum_time = current_time
        veh_len = getattr(vehicle, "length", 5.0) + 2.5
        for r_id in path:
            road_obj = self.route_evaluator.roads.get(r_id, {})
            ff_time = getattr(road_obj, "free_flow_time", None) or (road_obj.get("free_flow_time") if isinstance(road_obj, dict) else 10.0)
            try:
                ff_time = float(ff_time)
            except (TypeError, ValueError):
                ff_time = 10.0

            self.active_reservations[r_id][vehicle.id] = LinkReservation(
                vehicle_id=vehicle.id,
                road_id=r_id,
                reserved_time=current_time,
                expected_arrival_time=cum_time,
                vehicle_length_m=veh_len,
            )
            cum_time += max(1.0, ff_time)

    def release_vehicle_reservations(self, vehicle_id: str):
        """Releases all reservations held by a given vehicle."""
        for r_id in list(self.active_reservations.keys()):
            self.active_reservations[r_id].pop(vehicle_id, None)

    def update_telemetry(self, telemetry_map: Dict[str, Any], timestamp: float):
        """Updates rolling observation buffer and manages two-window congestion hysteresis."""
        if isinstance(telemetry_map, dict) and "road_heatmap" in telemetry_map and isinstance(telemetry_map["road_heatmap"], dict):
            telemetry_map = telemetry_map["road_heatmap"]
        self.observation_buffer.update(telemetry_map, timestamp)

        # Update Congestion Hysteresis per road
        for road_id, telem in telemetry_map.items():
            if hasattr(telem, "to_dict"):
                telem = telem.to_dict()
            if not isinstance(telem, dict):
                continue

            vc = telem.get("capacity_utilization")
            if vc is None:
                vc = telem.get("vc_ratio", 0.0)
            try:
                vc = float(vc)
            except (TypeError, ValueError):
                vc = 0.0

            is_currently_congested = self.congested_corridors.get(road_id, False)

            if vc >= self.DEFAULT_VC_CONGESTION_THRESHOLD:
                # Immediate activation at V/C >= 0.80
                self.congested_corridors[road_id] = True
                self.consecutive_clear_windows[road_id] = 0
            elif is_currently_congested:
                # In recovery mode: requires 2 consecutive windows below 0.70
                if vc < self.DEFAULT_VC_RECOVERY_THRESHOLD:
                    clears = self.consecutive_clear_windows.get(road_id, 0) + 1
                    self.consecutive_clear_windows[road_id] = clears
                    if clears >= self.DEFAULT_REQUIRED_RECOVERY_WINDOWS:
                        self.congested_corridors[road_id] = False
                        self.consecutive_clear_windows[road_id] = 0
                else:
                    # Stalled in hysteresis buffer [0.70, 0.80)
                    self.consecutive_clear_windows[road_id] = 0

    def compute_advisory_spillback_probabilities(self) -> Dict[str, float]:
        """Calculates advisory spillback probability across roads with ready feature buffers."""
        probs = {}
        if not self.predictive_available or self.mode != "predictive" or self.calibrator is None:
            return probs

        for road_id in self.route_evaluator.roads:
            feat_dict = self.observation_buffer.build_feature_dict(road_id)
            if feat_dict is None:
                continue
            try:
                row = np.array([[feat_dict[f] for f in self.expected_features]], dtype=np.float32)
                names = getattr(self.calibrator, "feature_names_in_", None)
                inputs = pd.DataFrame(row, columns=names) if names is not None else row
                p = float(self.calibrator.predict_proba(inputs)[0, 1])
                probs[road_id] = p
            except Exception:
                probs[road_id] = 0.0

        return probs

    def evaluate_and_reroute(
        self,
        sim: Any,
        current_time: float,
        telemetry_map: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Periodically screens eligible vehicles and applies capacity-safe diversions."""
        if self.mode == "off":
            if telemetry_map:
                self.update_telemetry(telemetry_map, current_time)
            self.clean_reservations(sim, current_time)
            return {
                "current_time": current_time,
                "mode": "off",
                "predictive_status": "DISABLED",
                "eligible_candidates_count": 0,
                "max_diversions_quota": 0,
                "diverted_count": 0,
                "sim_successful_count": 0,
                "sim_failed_count": 0,
                "decisions": [],
            }

        if isinstance(telemetry_map, dict) and "road_heatmap" in telemetry_map and isinstance(telemetry_map["road_heatmap"], dict):
            telemetry_map = telemetry_map["road_heatmap"]

        if telemetry_map:
            self.update_telemetry(telemetry_map, current_time)

        # 0. Clean stale persistent reservations across evaluation windows
        self.clean_reservations(sim, current_time)

        # 1. Compute advisory spillback probabilities (if predictive mode active)
        p_spillback_map = self.compute_advisory_spillback_probabilities()

        # 2. Identify eligible candidate vehicles
        eligible_vehicles = []
        for veh in sorted(sim.vehicles.values(), key=lambda v: v.id):
            if veh.route_index >= len(veh.route) - 1 or veh.current_road_id.startswith("CONN__"):
                continue

            last_t = self.last_reroute_time.get(veh.id, -999.0)
            if current_time - last_t < self.cooldown_sec:
                continue

            downstream_roads = veh.route[veh.route_index + 1:]
            is_route_congested = any(self.congested_corridors.get(r, False) for r in downstream_roads)
            is_predictive_alert = any(p_spillback_map.get(r, 0.0) >= self.threshold for r in downstream_roads)

            if is_route_congested or (self.mode == "predictive" and is_predictive_alert):
                eligible_vehicles.append(veh)

        # 3. Diversion Quota Calculation (25%)
        n_eligible = len(eligible_vehicles)
        if self.strict_quota_floor:
            max_diversions = int(math.floor(self.max_diversion_quota * n_eligible))
        else:
            max_diversions = max(1, int(math.floor(self.max_diversion_quota * n_eligible))) if n_eligible > 0 else 0

        commands_to_apply = []
        decisions = []

        for veh in eligible_vehicles:
            if len(commands_to_apply) >= max_diversions:
                decisions.append({
                    "vehicle_id": veh.id,
                    "diverted": False,
                    "reason": "Diversion quota reached for this evaluation window.",
                    "original_route": list(veh.route),
                    "cost_curr": 0.0,
                    "cost_before": 0.0,
                    "cost_after": 0.0,
                    "timestamp": current_time,
                })
                continue

            curr_road = veh.current_road_id
            dest_road = veh.route[-1]
            downstream_route = veh.route[veh.route_index + 1:]

            cost_curr = sum(
                self.route_evaluator.compute_edge_cost(
                    r,
                    telemetry_map.get(r) if telemetry_map else None,
                    p_spillback_map.get(r),
                )
                for r in downstream_route
            )

            # Identify congested or predictive alert roads along downstream route
            avoid_corridor = {
                r for r in downstream_route
                if self.congested_corridors.get(r, False)
                or (self.mode == "predictive" and p_spillback_map.get(r, 0.0) >= self.threshold)
            }

            # Query candidate alternative paths (K-shortest and corridor-avoiding)
            candidates = self.route_evaluator.find_candidate_paths(
                curr_road,
                dest_road_id=dest_road,
                telemetry_map=telemetry_map,
                p_spillback_map=p_spillback_map,
                allow_node_destination=self.allow_node_destination,
                k=5,
                avoid_roads=avoid_corridor if avoid_corridor else None,
            )

            if not candidates:
                decisions.append({
                    "vehicle_id": veh.id,
                    "diverted": False,
                    "reason": "No valid alternative path found.",
                })
                continue

            # Filter candidates that are genuinely different from current remaining route
            alternative_candidates = [c for c in candidates if c["path"] != downstream_route]

            # If the shortest path is identical to downstream route and no alternative was in top-k,
            # continue searching for genuinely different alternatives by avoiding the immediate next link:
            if not alternative_candidates and len(downstream_route) > 0:
                detour_cands = self.route_evaluator.find_candidate_paths(
                    curr_road,
                    dest_road_id=dest_road,
                    telemetry_map=telemetry_map,
                    p_spillback_map=p_spillback_map,
                    allow_node_destination=self.allow_node_destination,
                    k=3,
                    avoid_roads={downstream_route[0]},
                )
                alternative_candidates = [c for c in detour_cands if c["path"] != downstream_route]

            if not alternative_candidates:
                decisions.append({
                    "vehicle_id": veh.id,
                    "diverted": False,
                    "reason": "Candidate path is identical to existing remaining route.",
                })
                continue

            # Evaluate candidate alternatives against improvement threshold and capacity admission
            admissible_candidates = []
            capacity_rejection_reason = None
            benefit_rejection_reason = None

            for cand in alternative_candidates:
                cand_path = cand["path"]
                cost_cand = cand["cost"]
                delta_t = cost_curr - cost_cand
                min_req = max(self.min_improvement_sec, self.min_improvement_ratio * cost_curr)

                if delta_t < min_req:
                    if not benefit_rejection_reason:
                        benefit_rejection_reason = f"Insufficient benefit: improvement {delta_t:.1f}s < required {min_req:.1f}s."
                    break

                # Capacity Admission & Persistent Reservation across ALL roads on candidate route
                admitted = True
                rejection_reason = ""

                for r in cand_path:
                    r_road = self.route_evaluator.roads.get(r, {})
                    r_cap = getattr(r_road, "capacity", None) or (r_road.get("capacity") if isinstance(r_road, dict) else 1800.0)
                    try:
                        r_cap = float(r_cap)
                    except (TypeError, ValueError):
                        r_cap = 1800.0

                    r_telem = telemetry_map.get(r, {}) if telemetry_map else {}
                    r_flow = r_telem.get("flow_veh_per_hour") or r_telem.get("flow_veh_hr") or 0.0
                    try:
                        r_flow = float(r_flow)
                    except (TypeError, ValueError):
                        r_flow = 0.0

                    r_vc = r_telem.get("vc_ratio") or r_telem.get("capacity_utilization")
                    if r_vc is not None:
                        try:
                            r_flow = max(r_flow, float(r_vc) * r_cap)
                        except (TypeError, ValueError):
                            pass

                    # 1. Operational Flow / V/C check including persistent reservations
                    active_res = self.get_active_reservations(r, exclude_vehicle_id=veh.id)
                    n_res = len(active_res)
                    effective_flow = r_flow + (n_res + 1) * (3600.0 / self.evaluation_interval_sec)
                    effective_vc = effective_flow / max(1.0, r_cap)

                    if effective_vc >= self.vc_admission_cap:
                        admitted = False
                        rejection_reason = f"Alternate road '{r}' exceeds capacity admission threshold (effective V/C: {effective_vc:.3f} >= {self.vc_admission_cap})."
                        break

                    # 2. Physical Storage Capacity Check (75% threshold for storage links >= 25m)
                    r_len = getattr(r_road, "length", None) or (r_road.get("length") if isinstance(r_road, dict) else 100.0)
                    r_lanes = getattr(r_road, "lanes", None) or (r_road.get("lanes") if isinstance(r_road, dict) else 1)
                    try:
                        storage_cap_m = float(r_len) * float(r_lanes)
                    except (TypeError, ValueError):
                        storage_cap_m = 100.0

                    if storage_cap_m >= 25.0:
                        # Number of physical vehicles on road
                        phys_vehs = [v for v in getattr(sim, "vehicles", {}).values() if getattr(v, "current_road_id", None) == r]
                        n_phys = len(phys_vehs)
                        q_len = float(r_telem.get("queue_length_meters") or r_telem.get("queue_length_m") or 0.0)
                        if q_len > 0:
                            n_phys = max(n_phys, int(math.ceil(q_len / 7.5)))

                        # Max safe vehicle storage capacity (75% of physical storage)
                        safe_veh_storage = max(1, int(math.floor(self.storage_safety_limit * (storage_cap_m / 7.5))))
                        projected_veh_count = n_phys + n_res + 1

                        if projected_veh_count > safe_veh_storage:
                            admitted = False
                            rejection_reason = f"Alternate road '{r}' exceeds storage capacity threshold (projected {projected_veh_count} vehs > {safe_veh_storage} safe limit)."
                            break

                if not admitted:
                    if not capacity_rejection_reason:
                        capacity_rejection_reason = rejection_reason
                    continue

                # Corridor burden across all roads on candidate path
                corridor_burden = sum(len(self.get_active_reservations(r, exclude_vehicle_id=veh.id)) for r in cand_path)
                admissible_candidates.append({
                    "cand": cand,
                    "delta_t": delta_t,
                    "burden": corridor_burden,
                    "cost": cost_cand,
                })

            if not admissible_candidates:
                final_reason = capacity_rejection_reason or benefit_rejection_reason or "No candidate alternative met acceptance criteria."
                decisions.append({
                    "vehicle_id": veh.id,
                    "diverted": False,
                    "reason": final_reason,
                    "original_route": list(veh.route),
                    "cost_curr": round(cost_curr, 2),
                    "cost_before": round(cost_curr, 2),
                    "cost_after": round(cost_curr, 2),
                    "best_candidate_improvement_sec": round(max([cost_curr - c["cost"] for c in alternative_candidates], default=0.0), 2),
                    "candidate_paths_count": len(alternative_candidates),
                    "max_advisory_spillback": round(max([p_spillback_map.get(r, 0.0) for r in downstream_route], default=0.0), 4) if self.mode == "predictive" else 0.0,
                    "timestamp": current_time,
                })
                continue

            # Corridor Load Balancing / Dispersion:
            # Select candidate with lowest corridor reservation burden; break ties by lowest cost (highest improvement)
            best_choice = min(admissible_candidates, key=lambda x: (x["burden"], x["cost"]))
            accepted_candidate = best_choice["cand"]
            accepted_delta_t = best_choice["delta_t"]
            cand_path = accepted_candidate["path"]

            # Register persistent capacity reservations immediately
            self.add_route_reservations(veh, cand_path, current_time)

            commands_to_apply.append({
                "vehicle_id": veh.id,
                "new_subpath": cand_path,
            })
            orig_route = list(veh.route)
            full_new_route = list(veh.route[:veh.route_index + 1]) + list(cand_path)
            decisions.append({
                "vehicle_id": veh.id,
                "diverted": True,
                "reason": "Accepted",
                "improvement_sec": round(accepted_delta_t, 2),
                "estimated_improvement_sec": round(accepted_delta_t, 2),
                "new_path": cand_path,
                "original_route": orig_route,
                "new_route": full_new_route,
                "cost_curr": round(cost_curr, 2),
                "cost_before": round(cost_curr, 2),
                "cost_new": round(accepted_candidate["cost"], 2),
                "cost_after": round(accepted_candidate["cost"], 2),
                "max_advisory_spillback": round(max([p_spillback_map.get(r, 0.0) for r in downstream_route], default=0.0), 4) if self.mode == "predictive" else 0.0,
                "timestamp": current_time,
            })

        # 5. Submit validated commands to simulator interface
        sim_res = sim.reroute_vehicles(commands_to_apply) if commands_to_apply else {"successful_count": 0, "results": []}

        for res in sim_res.get("results", []):
            veh_id = res.get("vehicle_id")
            if res.get("success", False) and res.get("rerouted", False):
                self.last_reroute_time[veh_id] = current_time
            else:
                self.release_vehicle_reservations(veh_id)

        return {
            "current_time": current_time,
            "mode": self.mode,
            "predictive_status": self.predictive_status,
            "eligible_candidates_count": n_eligible,
            "max_diversions_quota": max_diversions,
            "diverted_count": len(commands_to_apply),
            "sim_successful_count": sim_res.get("successful_count", 0),
            "sim_failed_count": sim_res.get("failed_count", 0),
            "decisions": decisions,
        }

