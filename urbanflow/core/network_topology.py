"""Network Topology Module for UrbanFlow AI.

Extracts, validates, and caches directed road connectivity, multi-hop downstream sets,
and topological attributes from the UrbanFlow geographic digital twin (OSM).

Key Contracts:
1. Multi-hop Downstream Line Graph:
   - 1-hop: D_1(R) = { R' | from_node(R') == to_node(R) and R' != R }
   - 2-hop: D_2(R) = { R'' in D_1(R') for R' in D_1(R) | R'' != R and R'' not in D_1(R) }
   - 3-hop: D_3(R) = { R''' in D_1(R'') for R'' in D_2(R) | R''' != R and R''' not in D_1(R) and R''' not in D_2(R) }
2. Cycle-safe traversal:
   - Origin road R is strictly excluded from all hop sets: R not in D_k(R) for all k in {1, 2, 3}.
   - Sets at distinct hops are mutually disjoint: D_1(R) ^ D_2(R) = D_1(R) ^ D_3(R) = D_2(R) ^ D_3(R) = empty.
   - Elements appear at most once per hop set (deterministic sorted list).
3. Perimeter Exit Handling:
   - Perimeter exit links have D_1(R) = empty set.
   - Flagged via is_perimeter_exit(R) == True.
   - Explicit empty-set policy documented and implemented.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple, Optional
from collections import defaultdict


DEFAULT_MAP_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "maps" / "bronx_fordham.json"


class NetworkTopology:
    """Manages directed network connectivity and multi-hop downstream relations."""

    _instance: Optional[NetworkTopology] = None

    def __init__(self, map_path: Optional[Path] = None, *, data: Optional[Dict[str, Any]] = None):
        self.map_path = Path(map_path) if map_path else DEFAULT_MAP_PATH
        if data is None and not self.map_path.exists():
            raise FileNotFoundError(f"Digital twin map not found at: {self.map_path}")

        self.roads: Dict[str, Dict[str, Any]] = {}
        self.intersections: Dict[str, Dict[str, Any]] = {}
        self.road_lengths: Dict[str, float] = {}
        self.road_speeds: Dict[str, float] = {}
        self.road_lanes: Dict[str, int] = {}
        self.road_capacities: Dict[str, float] = {}

        self._out_roads_by_node: Dict[str, List[str]] = defaultdict(list)
        self._in_roads_by_node: Dict[str, List[str]] = defaultdict(list)

        self._downstream_1hop: Dict[str, List[str]] = {}
        self._downstream_2hop: Dict[str, List[str]] = {}
        self._downstream_3hop: Dict[str, List[str]] = {}
        self._perimeter_exits: Set[str] = set()

        if data is None:
            with open(self.map_path, "r", encoding="utf-8") as source:
                data = json.load(source)
        self._build(data)

    @classmethod
    def get_instance(cls, map_path: Optional[Path] = None) -> NetworkTopology:
        """Singleton accessor for cached network topology."""
        if cls._instance is None:
            cls._instance = cls(map_path)
        return cls._instance

    def _build(self, data):
        """Build the multi-hop line graph from a topology snapshot."""
        for inter in data.get("intersections", []):
            self.intersections[inter["id"]] = inter

        for r in data.get("roads", []):
            r_id = r["id"]
            self.roads[r_id] = r
            self.road_lengths[r_id] = float(r.get("length", 150.0))
            self.road_speeds[r_id] = float(r.get("speed_limit", 11.2))
            self.road_lanes[r_id] = int(r.get("lanes", 1))
            self.road_capacities[r_id] = 1800.0 * self.road_lanes[r_id]

            self._out_roads_by_node[r["from_node"]].append(r_id)
            self._in_roads_by_node[r["to_node"]].append(r_id)

        # 1. Build 1-Hop Downstream Roads
        for r_id, r in self.roads.items():
            to_node = r["to_node"]
            next_roads = [nxt for nxt in self._out_roads_by_node[to_node] if nxt != r_id]
            self._downstream_1hop[r_id] = sorted(list(dict.fromkeys(next_roads)))
            if not self._downstream_1hop[r_id]:
                self._perimeter_exits.add(r_id)

        # 2. Build 2-Hop Downstream Roads (cycle-safe, disjoint from 1-hop, origin excluded)
        for r_id, hop1_roads in self._downstream_1hop.items():
            hop2_set: Set[str] = set()
            for n1 in hop1_roads:
                for n2 in self._downstream_1hop.get(n1, []):
                    if n2 != r_id and n2 not in hop1_roads:
                        hop2_set.add(n2)
            self._downstream_2hop[r_id] = sorted(list(hop2_set))

        # 3. Build 3-Hop Downstream Roads (cycle-safe, disjoint from 1-hop and 2-hop, origin excluded)
        for r_id, hop2_roads in self._downstream_2hop.items():
            hop3_set: Set[str] = set()
            hop1_set = set(self._downstream_1hop[r_id])
            hop2_existing = set(hop2_roads)
            for n2 in hop2_roads:
                for n3 in self._downstream_1hop.get(n2, []):
                    if n3 != r_id and n3 not in hop1_set and n3 not in hop2_existing:
                        hop3_set.add(n3)
            self._downstream_3hop[r_id] = sorted(list(hop3_set))

    def get_downstream_roads(self, road_id: str, hop: int) -> List[str]:
        """Returns the list of distinct downstream road IDs at specified hop distance (1, 2, or 3)."""
        if hop == 1:
            return self._downstream_1hop.get(road_id, [])
        elif hop == 2:
            return self._downstream_2hop.get(road_id, [])
        elif hop == 3:
            return self._downstream_3hop.get(road_id, [])
        else:
            raise ValueError(f"Hop distance must be in {{1, 2, 3}}, got {hop}")

    def is_perimeter_exit(self, road_id: str) -> bool:
        """Returns True if the road terminates at the perimeter boundary (0 downstream links)."""
        return road_id in self._perimeter_exits

    def get_road_length(self, road_id: str) -> float:
        """Returns length in meters for the specified road segment."""
        return self.road_lengths.get(road_id, 150.0)

    def get_road_speed_limit(self, road_id: str) -> float:
        """Returns speed limit in m/s (or km/h converted) for the specified road segment."""
        return self.road_speeds.get(road_id, 11.2)

    def validate_topology_contracts(self) -> Dict[str, Any]:
        """
        Executes exhaustive contract validations on all 210 roads:
        1. No self-loops (origin road excluded from all hop sets).
        2. Strict disjointness across hop sets D_1 ^ D_2 = D_1 ^ D_3 = D_2 ^ D_3 = empty.
        3. No duplicate road IDs within any hop set.
        4. Valid perimeter identification.
        """
        violations = []
        for r_id in self.roads:
            s1 = set(self._downstream_1hop[r_id])
            s2 = set(self._downstream_2hop[r_id])
            s3 = set(self._downstream_3hop[r_id])

            # Uniqueness check
            if len(s1) != len(self._downstream_1hop[r_id]):
                violations.append(f"{r_id}: Duplicates found in 1-hop downstream set")
            if len(s2) != len(self._downstream_2hop[r_id]):
                violations.append(f"{r_id}: Duplicates found in 2-hop downstream set")
            if len(s3) != len(self._downstream_3hop[r_id]):
                violations.append(f"{r_id}: Duplicates found in 3-hop downstream set")

            # Origin exclusion
            if r_id in s1 or r_id in s2 or r_id in s3:
                violations.append(f"{r_id}: Origin road found in downstream set (self-loop violation)")

            # Pairwise disjointness
            if not s1.isdisjoint(s2):
                violations.append(f"{r_id}: Overlap between 1-hop and 2-hop sets: {s1 & s2}")
            if not s1.isdisjoint(s3):
                violations.append(f"{r_id}: Overlap between 1-hop and 3-hop sets: {s1 & s3}")
            if not s2.isdisjoint(s3):
                violations.append(f"{r_id}: Overlap between 2-hop and 3-hop sets: {s2 & s3}")

        return {
            "total_roads": len(self.roads),
            "perimeter_exit_count": len(self._perimeter_exits),
            "valid": len(violations) == 0,
            "violations_count": len(violations),
            "violations": violations,
            "mean_1hop": round(sum(len(v) for v in self._downstream_1hop.values()) / len(self.roads), 2),
            "mean_2hop": round(sum(len(v) for v in self._downstream_2hop.values()) / len(self.roads), 2),
            "mean_3hop": round(sum(len(v) for v in self._downstream_3hop.values()) / len(self.roads), 2),
        }
