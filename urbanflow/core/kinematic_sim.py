"""High-Performance Kinematic Traffic Micro-Simulator with Real Geographic OSM Map Support.

Supports both:
1. Real geographic urban digital twin (e.g. Fordham & Grand Concourse, Bronx, NY).
2. Synthetic/procedural 4-way corridor baseline.

Features Intelligent Driver Model (IDM) car-following along polyline geometry,
NetworkX graph-based multi-hop routing, intersection lane queuing, and multi-phase traffic signals.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import networkx as nx

from urbanflow.core.demand import Departure, FLEET, TRAFFIC_LEVELS, scale_departures
import numpy as np

from urbanflow.core.types import (
    Vec3,
    RoadSegment,
    Intersection,
    VehicleState,
    SignalState,
    NetworkMetrics,
    SimulationSnapshot,
    is_ns_approach,
)
from urbanflow.core.districts import CITIES, CITY_STOP_LINE_SETBACK, DISTRICTS, build_district
from urbanflow.core.observed_demand import OBSERVED_WINDOWS
from urbanflow.core.network_geometry import VehicleOccupancy, centerline_length, offset_point, trim_start
from urbanflow.core.simulator_base import SimulatorAdapter


class SimulatedVehicle:
    """Microscopic vehicle agent with IDM car-following dynamics along polyline roads."""

    def __init__(
        self,
        vehicle_id: str,
        v_type: str,
        route: List[str],  # sequence of connected road_ids
        color: str,
        desired_speed: float = 11.2,  # m/s (~25 mph NYC speed limit)
        length: float = 4.5,
        scheduled_spawn_time: float = 0.0,
        actual_entry_time: Optional[float] = None,
    ):
        self.id = vehicle_id
        self.type = v_type
        self.route = list(route)
        self.route_index = 0
        self.color = color
        self.desired_speed = desired_speed
        self.length = length

        # Counterfactual spawn and entry accounting
        self.scheduled_spawn_time = float(scheduled_spawn_time)
        self.actual_entry_time = float(actual_entry_time) if actual_entry_time is not None else None
        self.entry_delay = 0.0
        self.blocked_entry_time = 0.0

        # Kinematic state
        self.distance_on_road = 0.0  # meters from road start
        self.speed = desired_speed * 0.7
        self.accel = 0.0
        self.wait_time = 0.0
        self.total_travel_time = 0.0
        self.lane_index = 0

        # Current 3D pose
        self.x = 0.0
        self.y = 0.0
        self.z = 0.45
        self.heading = 0.0

        # Dynamic routing history
        self.reroute_count = 0
        self.last_reroute_time = 0.0
        self.route_history = [list(self.route)]
        self.spawn_time = float(scheduled_spawn_time)
        self.distance_traveled = 0.0

    @property
    def system_travel_time(self) -> float:
        """Total time from scheduled spawn time to current time."""
        return self.total_travel_time + self.entry_delay

    @property
    def current_road_id(self) -> str:
        if self.route_index < len(self.route):
            return self.route[self.route_index]
        return ""


class IntersectionSignalFSM:
    """Independent Finite State Machine for an individual signalized intersection.

    Enforces NEMA-compliant safety transitions:
    Green -> Yellow (3.0s) -> All-Red (1.5s) -> Next Green
    Guarantees:
    - Minimum green enforcement (default 10.0s)
    - Maximum green enforcement under competing demand (default 60.0s)
    - Mandatory yellow clearance (default 3.0s)
    - Mandatory all-red clearance (default 1.5s)
    - Conflicting green elimination (never simultaneous NS and EW green)
    """

    def __init__(
        self,
        intersection_id: str,
        min_green: float = 10.0,
        max_green: float = 60.0,
        yellow_duration: float = 3.0,
        all_red_duration: float = 1.5,
        default_green: float = 18.0,
        initial_green_phase: int = 0,
    ):
        self.intersection_id = intersection_id
        self.min_green = float(min_green)
        self.max_green = float(max_green)
        self.yellow_duration = float(yellow_duration)
        self.all_red_duration = float(all_red_duration)
        self.default_green = float(default_green)

        # FSM State
        self.stage: str = "GREEN"  # "GREEN", "YELLOW", "ALL_RED"
        self.active_green_phase: int = 0 if initial_green_phase in (0, "0", "NS") else 1  # 0: NS, 1: EW
        self.target_green_phase: Optional[int] = None
        self.time_in_stage: float = 0.0
        self.time_in_green: float = 0.0
        self.control_mode: str = "fixed_time"  # "fixed_time" or "command"
        self.competing_demand_detected: bool = False

    def reset(self) -> None:
        """Resets FSM state to initial green."""
        self.stage = "GREEN"
        self.active_green_phase = 0
        self.target_green_phase = None
        self.time_in_stage = 0.0
        self.time_in_green = 0.0
        self.control_mode = "fixed_time"
        self.competing_demand_detected = False

    @property
    def current_duration(self) -> float:
        if self.stage == "GREEN":
            return self.default_green
        elif self.stage == "YELLOW":
            return self.yellow_duration
        else:
            return self.all_red_duration

    def request_phase(self, target_phase: Union[int, str]) -> bool:
        """Requests a target green phase (0 for NS, 1 or 2 for EW, -1 or 4 for All-Red hold).

        Enforces minimum green before initiating yellow transition.
        Prohibits direct green-to-green transitions.
        """
        # Normalize target
        if target_phase in (0, "0", "NS", "ns"):
            desired = 0
        elif target_phase in (1, 2, "1", "2", "EW", "ew"):
            desired = 1
        elif target_phase in (-1, "-1", 4, "4", "ALL_RED", "all_red", "RED", "red"):
            desired = -1
        else:
            return False

        self.control_mode = "command"

        if self.stage == "GREEN":
            if desired == self.active_green_phase:
                # Holding current green phase; clear pending switches
                self.target_green_phase = None
                return True

            # Requesting switch to opposite phase or all-red hold
            self.target_green_phase = desired
            if self.time_in_green >= self.min_green:
                # Can begin yellow transition immediately
                self.stage = "YELLOW"
                self.time_in_stage = 0.0
            # Else: target_green_phase remains set; step() will transition once min_green reached
            return True

        elif self.stage in ("YELLOW", "ALL_RED"):
            # Clearance already in progress; queue the desired green phase or all-red hold
            self.target_green_phase = desired
            return True

        return False

    def step(self, dt: float, sim: Optional[Any] = None) -> None:
        """Advances state machine by dt seconds with guard checks."""
        self.time_in_stage += dt

        if self.stage == "GREEN":
            self.time_in_green += dt

            should_transition = False

            if self.target_green_phase is not None and self.target_green_phase != self.active_green_phase:
                # External switch requested; enforce min_green
                if self.time_in_green >= self.min_green:
                    should_transition = True
            elif self.time_in_green + 1e-8 >= self.max_green and (
                getattr(sim, "managed_junctions", None) or self._check_competing_demand(sim)
            ):
                # Managed networks enforce max green without scanning the fleet.
                should_transition = True
                self.target_green_phase = 1 if self.active_green_phase == 0 else 0
            elif self.control_mode == "fixed_time" and self.target_green_phase is None:
                # Autonomous cycle expiration
                if self.time_in_green >= self.default_green:
                    should_transition = True
                    self.target_green_phase = 1 if self.active_green_phase == 0 else 0

            if should_transition and self.time_in_green >= self.min_green:
                self.stage = "YELLOW"
                self.time_in_stage = 0.0

        elif self.stage == "YELLOW":
            if self.time_in_stage + 1e-8 >= self.yellow_duration:
                self.stage = "ALL_RED"
                self.time_in_stage = 0.0

        elif self.stage == "ALL_RED":
            if self.target_green_phase == -1:
                # The classroom admits each vehicle against real lane storage.
                # Retry an axis after a short hold when a controller vetoes both forecasts.
                if getattr(sim, "managed_junctions", None) and self.time_in_stage >= 3.0:
                    self.target_green_phase = 1 - self.active_green_phase
                return
            if (self.time_in_stage + 1e-8 >= self.all_red_duration
                  and self.intersection_id not in getattr(sim, "junction_reservations", {})):
                self.stage = "GREEN"
                self.time_in_stage = 0.0
                self.time_in_green = 0.0
                if self.target_green_phase is not None:
                    self.active_green_phase = self.target_green_phase
                    self.target_green_phase = None
                else:
                    self.active_green_phase = 1 if self.active_green_phase == 0 else 0

    def get_signal_code(self, is_ns: bool) -> str:
        """Returns signal code ('G', 'y', 'r') for approach orientation."""
        if self.stage == "GREEN":
            if self.active_green_phase == 0:
                return "G" if is_ns else "r"
            else:
                return "r" if is_ns else "G"
        elif self.stage == "YELLOW":
            if self.active_green_phase == 0:
                return "y" if is_ns else "r"
            else:
                return "r" if is_ns else "y"
        else:  # ALL_RED
            return "r"

    def get_phase_index(self) -> int:
        """Returns compatible integer phase index."""
        if self.stage == "GREEN":
            return 0 if self.active_green_phase == 0 else 2
        elif self.stage == "YELLOW":
            return 1 if self.active_green_phase == 0 else 3
        else:
            return 4

    def get_phase_name(self) -> str:
        """Returns descriptive phase name."""
        names = {
            0: "North-South Green",
            1: "North-South Yellow",
            2: "East-West Green",
            3: "East-West Yellow",
            4: "All-Red Clearance",
        }
        return names.get(self.get_phase_index(), "Unknown Phase")

    def get_signal_state(self) -> SignalState:
        """Constructs a SignalState snapshot dataclass."""
        ns_code = self.get_signal_code(is_ns=True)
        ew_code = self.get_signal_code(is_ns=False)
        return SignalState(
            intersection_id=self.intersection_id,
            phase_index=self.get_phase_index(),
            phase_name=self.get_phase_name(),
            time_in_phase=round(self.time_in_stage, 4),
            duration=round(self.current_duration, 4),
            state_string=f"NS:{ns_code}|EW:{ew_code}",
            stage=self.stage,
            is_yellow=(self.stage == "YELLOW"),
            is_all_red=(self.stage == "ALL_RED"),
        )

    def _check_competing_demand(self, sim: Optional[Any]) -> bool:
        """Inspects simulator vehicles approaching on conflicting approaches."""
        if self.competing_demand_detected:
            return True
        if sim is None or not hasattr(sim, "vehicles") or not hasattr(sim, "roads"):
            return False

        # If currently serving NS (phase 0), competing is EW (not is_ns)
        # If currently serving EW (phase 1), competing is NS (is_ns)
        competing_is_ns = (self.active_green_phase == 1)

        for veh in sim.vehicles.values():
            road = sim.roads.get(veh.current_road_id)
            if road and road.signal_controlled and road.to_node == self.intersection_id:
                road_is_ns = is_ns_approach(road)
                if road_is_ns == competing_is_ns:
                    return True
        return False


class KinematicSimulator(SimulatorAdapter):
    """Zero-dependency discrete-time kinematic traffic simulator for UrbanFlow."""

    def __init__(
        self,
        map_name: str = "classroom_grid",
        rng_spawn: Optional[Any] = None,
        rng_driver: Optional[Any] = None,
    ):
        self.departures: Optional[tuple[Departure, ...]] = None
        self.base_departures = None
        self.traffic_factor = 1
        self.departure_index = 0
        self.junction_reservations: Dict[str, Dict[str, tuple[str, str, int]]] = {}
        self._movement_bounds = {}
        self._lane_geometry = {}
        self._straight_pose = {}
        self.junction_movements: Dict[tuple[str, str], str] = {}
        self.classroom_movements = self.junction_movements  # Existing scenario/test callers.
        self.managed_junctions = set()
        self.junction_radii = {}
        self.demand_name = ""
        self.restriction = None
        self.incident_limits = {}
        self._connector_cache = {}
        self.map_name = map_name
        self.step_count = 0
        self.sim_time = 0.0
        self.vehicles: Dict[str, SimulatedVehicle] = {}
        self.pending_spawns: List[SimulatedVehicle] = []
        self.all_scheduled_spawns: List[Dict[str, Any]] = []
        self.next_scheduled_spawn_time = 0.0
        self.total_arrived = 0
        self.completed_trips: List[Dict[str, Any]] = []
        self.spawn_interval = 0.8  # seconds between spawns
        self.last_spawn_time = 0.0
        self.vehicle_counter = 0

        # Optional decoupled PRNG streams (defaults to None -> legacy random)
        self.rng_spawn = rng_spawn
        self.rng_driver = rng_driver

        # Topology containers
        self.roads: Dict[str, RoadSegment] = {}
        self.intersections: Dict[str, Intersection] = {}
        self.buildings: List[Dict[str, Any]] = []
        self.map_metadata: Dict[str, Any] = {}
        self.graph = nx.DiGraph()
        self.route_pool: List[List[str]] = []

        # Traffic Signal controllers & phases
        self.signal_controllers: Dict[str, IntersectionSignalFSM] = {}
        self.signal_phase = 0
        self.signal_timer = 0.0
        self.phases = [
            {"name": "North-South Green", "duration": 18.0, "ns": "G", "ew": "r"},
            {"name": "North-South Yellow", "duration": 3.0, "ns": "y", "ew": "r"},
            {"name": "East-West Green", "duration": 18.0, "ns": "r", "ew": "G"},
            {"name": "East-West Yellow", "duration": 3.0, "ns": "r", "ew": "y"},
            {"name": "All-Red Clearance", "duration": 1.5, "ns": "r", "ew": "r"},
        ]

        # Load target map
        self.load_map(map_name)

    def load_map(self, map_name: str):
        """Loads either the real Bronx OSM digital twin or the procedural corridor."""
        from urbanflow.core.districts import AVAILABLE_NETWORKS
        if map_name not in AVAILABLE_NETWORKS:
            raise ValueError(f"Unknown road network: {map_name}")
        self.map_name = map_name
        self.map_metadata = {}
        self.vehicles.clear()
        self.pending_spawns.clear()
        self.all_scheduled_spawns.clear()
        self.next_scheduled_spawn_time = 0.0
        self.departures = None
        self.base_departures = None
        self.traffic_factor = 1
        self.demand_name = ""
        self.restriction = None
        self.incident_limits.clear()
        self._connector_cache.clear()
        self.junction_reservations.clear()
        self.junction_movements.clear()
        self.managed_junctions.clear()
        self.junction_radii.clear()
        self._movement_bounds.clear()
        self._lane_geometry.clear()
        self._straight_pose.clear()
        self.roads.clear()
        self.intersections.clear()
        self.signal_controllers.clear()
        self.buildings.clear()
        self.graph.clear()
        self.route_pool.clear()
        self.scheduled_routes = []
        self.step_count = 0
        self.sim_time = 0.0
        self.total_arrived = 0
        self.completed_trips.clear()

        if map_name == "bronx_fordham":
            json_path = Path(__file__).resolve().parent.parent.parent / "data" / "maps" / "bronx_fordham.json"
            if json_path.exists():
                self._load_from_osm_json(json_path)
            else:
                self._build_classroom_network()
        elif map_name in DISTRICTS:
            build_district(self, IntersectionSignalFSM, map_name)
        elif map_name == "procedural":
            self._build_procedural_corridor()
            self._prepare_junctions()
        else:
            self._build_classroom_network()

        self._index_junction_movements()

        # Pre-seed initial active traffic fleet so digital twin is immediately active
        seed_count = 32 if map_name in ("classroom_grid", "classroom", "classroom_demo") else (25 if map_name == "bronx_fordham" else 12)
        for _ in range(seed_count):
            self._spawn_vehicle(initial_stagger=True)

    def _load_from_osm_json(self, json_path: Path):
        """Builds simulation topology and routing graph from processed OSM JSON data."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.map_metadata = data.get("metadata", {})
        self.buildings = data.get("buildings", [])

        # 1. Load Intersections
        for item in data.get("intersections", []):
            i_id = item["id"]
            pos = Vec3(item["position"]["x"], item["position"]["y"], 0.0)
            has_signals = item.get("has_signals", False)
            inter = Intersection(
                id=i_id,
                position=pos,
                connected_roads=item.get("connected_roads", []),
                has_signals=has_signals,
                current_phase=0,
                phase_names=[p["name"] for p in self.phases],
                name=item.get("name"),
            )
            self.intersections[i_id] = inter
            if has_signals:
                self.signal_controllers[i_id] = IntersectionSignalFSM(
                    intersection_id=i_id,
                    min_green=inter.min_green,
                    max_green=inter.max_green,
                    yellow_duration=inter.yellow_duration,
                    all_red_duration=inter.all_red_duration,
                )

        # 2. Load Road Segments
        for item in data.get("roads", []):
            r_id = item["id"]
            pts = [Vec3(p["x"], p["y"], 0.0) for p in item.get("points", [])]
            source_lanes = item.get("lanes", 1)
            road = RoadSegment(
                id=r_id,
                from_node=item["from_node"],
                to_node=item["to_node"],
                lanes=source_lanes,
                visual_lanes=source_lanes,
                length=item.get("length", 50.0),
                speed_limit=item.get("speed_limit", 11.2),
                capacity=1800.0 * item.get("lanes", 1),
                routing_length=item.get("length"),
                name=item.get("name"),
                points=pts,
                signal_controlled=item.get("signal_controlled", True),
            )
            self.roads[r_id] = road

            # Add to routing graph
            self.graph.add_edge(
                road.from_node,
                road.to_node,
                id=r_id,
                length=road.length,
                lanes=road.lanes,
            )

        # OSM extracts do not always carry junction labels.  Build a useful
        # display label from the named roads that meet at each junction so the
        # route planner never exposes opaque node IDs as its primary label.
        connected_names = {node_id: [] for node_id in self.intersections}
        for road in self.roads.values():
            if road.name:
                for node_id in (road.from_node, road.to_node):
                    if node_id in connected_names and road.name not in connected_names[node_id]:
                        connected_names[node_id].append(road.name)
        for node_id, intersection in self.intersections.items():
            if not intersection.name and connected_names[node_id]:
                intersection.name = " / ".join(connected_names[node_id][:2])

        self._prepare_junctions(separate_directions=True)

        # 3. Pre-generate realistic multi-hop routes across connected road components
        self._build_route_pool()

    def _prepare_junctions(self, *, separate_directions=False):
        if separate_directions:
            point_keys = {
                (road.from_node, road.to_node, tuple((p.x, p.y) for p in road.points))
                for road in self.roads.values()
            }
            for road in self.roads.values():
                reverse = (road.to_node, road.from_node, tuple((p.x, p.y) for p in reversed(road.points)))
                if reverse in point_keys:
                    # The OSM input duplicates the entire road centerline and
                    # lane count for each direction. Split the count and place
                    # each carriageway on its own side of the shared centerline.
                    road.visual_lanes = max(1, (road.lanes + 1) // 2)
                    side = 1 if self.map_metadata.get("traffic_side") == "left" else -1
                    road.points = [offset_point(road.points, i, side * road.visual_lanes * 1.75)
                                   for i in range(len(road.points))]
                else:
                    road.visual_lanes = max(1, road.lanes)
                road.lanes = 1
                # Preserve the mapped capacity for route costs. Admission still
                # uses modeled lane count and measured downstream storage.
                road.capacity = road.visual_lanes * 1800.0

        adjacent = {node: {} for node in self.intersections}
        for road in self.roads.values():
            if road.from_node in adjacent:
                adjacent[road.from_node][road.to_node] = road.lanes
            if road.to_node in adjacent:
                adjacent[road.to_node][road.from_node] = road.lanes
        radii = {
            node: max(8.0, min(16.0, max(neighbors.values(), default=1) * 3.5 + 1))
            for node, neighbors in adjacent.items() if node in self.signal_controllers or len(neighbors) >= 3
        }
        self.managed_junctions.update(radii)
        self.junction_radii = radii
        self.graph.clear()
        for road in self.roads.values():
            length = centerline_length(road.points)
            start_trim = min(radii.get(road.from_node, 0), max(0.0, (length - 8.0) / 2))
            end_trim = min(radii.get(road.to_node, 0), max(0.0, (length - 8.0) / 2))
            if start_trim:
                road.points = trim_start(road.points, start_trim)
            if end_trim:
                road.points = list(reversed(trim_start(list(reversed(road.points)), end_trim)))
            road.length = centerline_length(road.points)
            # Very short links remain visible. Exclude them from trip routing when
            # the rear cannot clear one junction before the front reaches the next.
            if road.length > 0:
                self.graph.add_edge(road.from_node, road.to_node, id=road.id,
                                    length=road.routing_length or road.length, lanes=road.lanes)

    def _index_junction_movements(self):
        self.managed_junctions.update(self.signal_controllers)
        outgoing_roads = {}
        for road in self.roads.values():
            outgoing_roads.setdefault(road.from_node, []).append(road)
        for incoming in self.roads.values():
            if incoming.to_node not in self.managed_junctions:
                continue
            _, (ax, ay) = self._get_lane_point_and_tangent(incoming, 0, at_end=True)
            for outgoing in outgoing_roads.get(incoming.to_node, []):
                if incoming.from_node == outgoing.to_node:
                    continue
                _, (bx, by) = self._get_lane_point_and_tangent(outgoing, 0, at_end=False)
                cross = ax * by - ay * bx
                dot = ax * bx + ay * by
                kind = "straight" if abs(cross) < 0.16 and dot > 0 else "left" if cross > 0 else "right"
                self.junction_movements[incoming.id, outgoing.id] = kind

    def _junction_hold_offset(self, road, vehicle_length):
        """Extra upstream stopping distance when a nearby OSM movement owns a junction."""
        if self.map_name in CITIES and road.to_node in self.managed_junctions:
            return CITY_STOP_LINE_SETBACK  # Leave room for the rear of a turning bus.
        if self.map_name == "procedural" and road.to_node in self.managed_junctions:
            return 2.0
        if self.map_name != "bronx_fordham" or road.to_node not in self.managed_junctions:
            return 0.0
        node = self.intersections.get(road.to_node)
        if node is None or not road.points:
            return 0.0
        endpoint_gap = math.hypot(road.points[-1].x - node.position.x,
                                  road.points[-1].y - node.position.y)
        clearance = min(self.junction_radii.get(road.to_node, 8.0), 6.0)
        # Keep the vehicle's front outside the junction clearance radius even
        # when a short link could not be trimmed back to that full radius.
        return max(0.0, clearance - endpoint_gap - 0.6)

    def _build_route_pool(self):
        """Discovers valid multi-hop origin-destination paths through the road graph."""
        if self.graph.number_of_nodes() == 0:
            return

        graph = self.graph
        sccs = list(nx.strongly_connected_components(graph))
        if not sccs:
            return

        largest_scc = max(sccs, key=len)
        scc_nodes = sorted(largest_scc)
        # Build mapping from (u, v) edge to road_id
        edge_to_road = {}
        for u, v, edge in graph.edges(data=True):
            edge_to_road[(u, v)] = edge["id"]

        # Generate paths between distinct peripheral/arterial nodes
        routes = []
        rng = random.Random(42)
        attempts = 0

        while len(routes) < 80 and attempts < 600:
            attempts += 1
            src = rng.choice(scc_nodes)
            dst = rng.choice(scc_nodes)
            if src == dst:
                continue

            try:
                node_path = nx.shortest_path(graph, src, dst, weight="length")
                if 3 <= len(node_path) <= 18:
                    road_path = []
                    valid = True
                    for k in range(len(node_path) - 1):
                        u, v = node_path[k], node_path[k + 1]
                        if (u, v) in edge_to_road:
                            road_path.append(edge_to_road[(u, v)])
                        else:
                            valid = False
                            break
                    if valid and road_path:
                        routes.append(road_path)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass

        self.route_pool = routes
        demo_routes = self.map_metadata.get("demo_routes", [])
        self.scheduled_routes = routes + [route for route in demo_routes
                                         if self.route_fits_vehicle(route, FLEET["sedan"][0])]
        if self.map_name == "bronx_fordham":
            longer_graph = nx.subgraph_view(self.graph, filter_edge=lambda u, v:
                                           self.roads[self.graph[u][v]["id"]].length >= 12.0)
            safe_routes = []
            for route in routes:
                start = self.roads[route[0]].from_node
                end = self.roads[route[-1]].to_node
                try:
                    path = nx.shortest_path(longer_graph, start, end, weight="length")
                except nx.NetworkXNoPath:
                    continue
                if len(path) >= 3:
                    safe_routes.append([edge_to_road[u, v] for u, v in zip(path, path[1:])])
            if safe_routes:
                self.scheduled_routes = safe_routes

    def _build_classroom_network(self):
        """Constructs the high-fidelity 4-intersection rectangular demonstration grid."""
        self.map_metadata = {
            "name": "Classroom 4-Way Arterial Grid",
            "map_name": "classroom_grid",
            "intersections_count": 4,
            "roads_count": 24,
            "grid_dim": "2x2",
            "lane_width": 3.5,
            "junction_half_width": 8.0,
        }

        # 1. Four Signalized Intersections in 2x2 Grid
        inter_coords = {
            "INT_NW": (-60.0, 60.0),
            "INT_NE": (60.0, 60.0),
            "INT_SW": (-60.0, -60.0),
            "INT_SE": (60.0, -60.0),
        }

        for i_id, (ix, iy) in inter_coords.items():
            self.intersections[i_id] = Intersection(
                id=i_id,
                position=Vec3(ix, iy, 0.0),
                connected_roads=[],
                has_signals=True,
                current_phase=0,
                phase_names=[p["name"] for p in self.phases],
                min_green=10.0,
                max_green=60.0,
                yellow_duration=3.0,
                all_red_duration=1.5,
            )
            self.signal_controllers[i_id] = IntersectionSignalFSM(
                intersection_id=i_id,
                min_green=10.0,
                max_green=60.0,
                yellow_duration=3.0,
                all_red_duration=1.5,
                default_green=16.0,
            )

        # 2. Road Segments (24 directed segments: 12 EW + 12 NS)
        # Lane width = 3.5m, 2 lanes per direction = 7.0m roadway width
        # Centerline offset = 3.5m from road median
        roads_def = [
            # Street North (Y_median = 60.0)
            # Eastbound (Y = 56.5)
            ("road_w_in_nw", "NODE_W_N", "INT_NW", [(-160.0, 56.5), (-60.0, 56.5)], 100.0),
            ("road_nw_ne", "INT_NW", "INT_NE", [(-60.0, 56.5), (60.0, 56.5)], 120.0),
            ("road_ne_out_e", "INT_NE", "NODE_E_N", [(60.0, 56.5), (160.0, 56.5)], 100.0),
            # Westbound (Y = 63.5)
            ("road_e_in_ne", "NODE_E_N", "INT_NE", [(160.0, 63.5), (60.0, 63.5)], 100.0),
            ("road_ne_nw", "INT_NE", "INT_NW", [(60.0, 63.5), (-60.0, 63.5)], 120.0),
            ("road_nw_out_w", "INT_NW", "NODE_W_N", [(-60.0, 63.5), (-160.0, 63.5)], 100.0),

            # Street South (Y_median = -60.0)
            # Eastbound (Y = -63.5)
            ("road_w_in_sw", "NODE_W_S", "INT_SW", [(-160.0, -63.5), (-60.0, -63.5)], 100.0),
            ("road_sw_se", "INT_SW", "INT_SE", [(-60.0, -63.5), (60.0, -63.5)], 120.0),
            ("road_se_out_e", "INT_SE", "NODE_E_S", [(60.0, -63.5), (160.0, -63.5)], 100.0),
            # Westbound (Y = -56.5)
            ("road_e_in_se", "NODE_E_S", "INT_SE", [(160.0, -56.5), (60.0, -56.5)], 100.0),
            ("road_se_sw", "INT_SE", "INT_SW", [(60.0, -56.5), (-60.0, -56.5)], 120.0),
            ("road_sw_out_w", "INT_SW", "NODE_W_S", [(-60.0, -56.5), (-160.0, -56.5)], 100.0),

            # Avenue West (X_median = -60.0)
            # Southbound (X = -63.5)
            ("road_n_in_nw", "NODE_N_W", "INT_NW", [(-63.5, 160.0), (-63.5, 60.0)], 100.0),
            ("road_nw_sw", "INT_NW", "INT_SW", [(-63.5, 60.0), (-63.5, -60.0)], 120.0),
            ("road_sw_out_s", "INT_SW", "NODE_S_W", [(-63.5, -60.0), (-63.5, -160.0)], 100.0),
            # Northbound (X = -56.5)
            ("road_s_in_sw", "NODE_S_W", "INT_SW", [(-56.5, -160.0), (-56.5, -60.0)], 100.0),
            ("road_sw_nw", "INT_SW", "INT_NW", [(-56.5, -60.0), (-56.5, 60.0)], 120.0),
            ("road_nw_out_n", "INT_NW", "NODE_N_W", [(-56.5, 60.0), (-56.5, 160.0)], 100.0),

            # Avenue East (X_median = 60.0)
            # Southbound (X = 56.5)
            ("road_n_in_ne", "NODE_N_E", "INT_NE", [(56.5, 160.0), (56.5, 60.0)], 100.0),
            ("road_ne_se", "INT_NE", "INT_SE", [(56.5, 60.0), (56.5, -60.0)], 120.0),
            ("road_se_out_s", "INT_SE", "NODE_S_E", [(56.5, -60.0), (56.5, -160.0)], 100.0),
            # Northbound (X = 63.5)
            ("road_s_in_se", "NODE_S_E", "INT_SE", [(63.5, -160.0), (63.5, -60.0)], 100.0),
            ("road_se_ne", "INT_SE", "INT_NE", [(63.5, -60.0), (63.5, 60.0)], 120.0),
            ("road_ne_out_n", "INT_NE", "NODE_N_E", [(63.5, 60.0), (63.5, 160.0)], 100.0),
        ]

        for r_id, u, v, pts_raw, length in roads_def:
            pts = [Vec3(p[0], p[1], 0.0) for p in pts_raw]
            # Stop lines bound the 16 m junction footprint shared with the renderer.
            tx = (pts[1].x - pts[0].x) / length
            ty = (pts[1].y - pts[0].y) / length
            inset = self.map_metadata["junction_half_width"]
            if u in self.intersections:
                pts[0] = Vec3(pts[0].x + tx * inset, pts[0].y + ty * inset, 0.0)
                length -= inset
            if v in self.intersections:
                pts[1] = Vec3(pts[1].x - tx * inset, pts[1].y - ty * inset, 0.0)
                length -= inset
            road = RoadSegment(
                id=r_id,
                from_node=u,
                to_node=v,
                lanes=2,
                length=length,
                speed_limit=13.89,
                capacity=3600.0,
                points=pts,
            )
            self.roads[r_id] = road
            self.graph.add_edge(u, v, id=r_id, length=length, lanes=2)
            if u in self.intersections:
                self.intersections[u].connected_roads.append(r_id)
            if v in self.intersections:
                self.intersections[v].connected_roads.append(r_id)

        # 3. Clean Buildings strictly outside road corridors (setbacks >= 15m)
        bldg_defs = [
            # Center block between the 4 intersections (X in [-40, 40], Y in [-40, 40])
            ("bldg_center_1", 38.0, [(-36.0, 10.0), (-10.0, 10.0), (-10.0, 36.0), (-36.0, 36.0)]),
            ("bldg_center_2", 44.0, [(10.0, 10.0), (36.0, 10.0), (36.0, 36.0), (10.0, 36.0)]),
            ("bldg_center_3", 32.0, [(-36.0, -36.0), (-10.0, -36.0), (-10.0, -10.0), (-36.0, -10.0)]),
            ("bldg_center_4", 48.0, [(10.0, -36.0), (36.0, -36.0), (36.0, -10.0), (10.0, -10.0)]),
            # NW Quadrant (X <= -85, Y >= 85)
            ("bldg_nw_1", 30.0, [(-150.0, 85.0), (-85.0, 85.0), (-85.0, 150.0), (-150.0, 150.0)]),
            # NE Quadrant (X >= 85, Y >= 85)
            ("bldg_ne_1", 34.0, [(85.0, 85.0), (150.0, 85.0), (150.0, 150.0), (85.0, 150.0)]),
            # SW Quadrant (X <= -85, Y <= -85)
            ("bldg_sw_1", 28.0, [(-150.0, -150.0), (-85.0, -150.0), (-85.0, -85.0), (-150.0, -85.0)]),
            # SE Quadrant (X >= 85, Y <= -85)
            ("bldg_se_1", 40.0, [(85.0, -150.0), (150.0, -150.0), (150.0, -85.0), (85.0, -85.0)]),
            # North Mid (X in [-35, 35], Y >= 85)
            ("bldg_n_mid", 26.0, [(-35.0, 85.0), (35.0, 85.0), (35.0, 140.0), (-35.0, 140.0)]),
            # South Mid (X in [-35, 35], Y <= -85)
            ("bldg_s_mid", 26.0, [(-35.0, -140.0), (35.0, -140.0), (35.0, -85.0), (-35.0, -85.0)]),
            # West Mid (X <= -85, Y in [-35, 35])
            ("bldg_w_mid", 30.0, [(-150.0, -35.0), (-85.0, -35.0), (-85.0, 35.0), (-150.0, 35.0)]),
            # East Mid (X >= 85, Y in [-35, 35])
            ("bldg_e_mid", 32.0, [(85.0, -35.0), (150.0, -35.0), (150.0, 35.0), (85.0, 35.0)]),
        ]

        self.buildings = [
            {
                "id": b_id,
                "name": b_id.replace("_", " ").title(),
                "type": "commercial",
                "height": h,
                "polygon": [{"x": p[0], "y": p[1]} for p in poly],
            }
            for b_id, h, poly in bldg_defs
        ]

        # 4. Multi-hop Route Pool across all 4 intersections (32 routes)
        self.route_pool = [
            # Through Horizontal North
            ["road_w_in_nw", "road_nw_ne", "road_ne_out_e"],
            ["road_e_in_ne", "road_ne_nw", "road_nw_out_w"],
            # Through Horizontal South
            ["road_w_in_sw", "road_sw_se", "road_se_out_e"],
            ["road_e_in_se", "road_se_sw", "road_sw_out_w"],
            # Through Vertical West
            ["road_n_in_nw", "road_nw_sw", "road_sw_out_s"],
            ["road_s_in_sw", "road_sw_nw", "road_nw_out_n"],
            # Through Vertical East
            ["road_n_in_ne", "road_ne_se", "road_se_out_s"],
            ["road_s_in_se", "road_se_ne", "road_ne_out_n"],
            # Right Turns
            ["road_w_in_nw", "road_nw_sw", "road_sw_out_s"],
            ["road_n_in_nw", "road_nw_out_w"],
            ["road_e_in_ne", "road_ne_out_n"],
            ["road_s_in_se", "road_se_out_e"],
            ["road_w_in_sw", "road_sw_out_s"],
            ["road_e_in_se", "road_se_sw", "road_sw_out_w"],
            ["road_s_in_sw", "road_sw_se", "road_se_out_e"],
            ["road_n_in_ne", "road_ne_nw", "road_nw_out_w"],
            # Left Turns
            ["road_w_in_nw", "road_nw_out_n"],
            ["road_n_in_nw", "road_nw_ne", "road_ne_out_e"],
            ["road_e_in_ne", "road_ne_se", "road_se_out_s"],
            ["road_s_in_se", "road_se_ne", "road_ne_nw", "road_nw_out_w"],
            ["road_w_in_sw", "road_sw_nw", "road_nw_ne", "road_ne_out_e"],
            ["road_e_in_se", "road_se_out_s"],
            ["road_s_in_sw", "road_sw_out_w"],
            ["road_n_in_ne", "road_ne_out_e"],
            # Multi-Intersection Diagonal & L-shape Routes
            ["road_w_in_nw", "road_nw_sw", "road_sw_se", "road_se_out_e"],
            ["road_w_in_sw", "road_sw_nw", "road_nw_ne", "road_ne_se", "road_se_out_s"],
            ["road_e_in_ne", "road_ne_nw", "road_nw_sw", "road_sw_out_s"],
            ["road_e_in_se", "road_se_ne", "road_ne_nw", "road_nw_out_w"],
            ["road_n_in_nw", "road_nw_ne", "road_ne_se", "road_se_out_s"],
            ["road_s_in_sw", "road_sw_se", "road_se_ne", "road_ne_out_n"],
            ["road_n_in_ne", "road_ne_nw", "road_nw_sw", "road_sw_se", "road_se_out_e"],
            ["road_s_in_se", "road_se_sw", "road_sw_nw", "road_nw_out_n"],
        ]

    def _build_procedural_corridor(self):
        """Constructs the baseline 4-way synthetic intersection corridor."""
        R_LEN = 160.0
        L_WIDTH = 3.5

        self.intersections["INT_CENTER"] = Intersection(
            id="INT_CENTER",
            position=Vec3(0, 0, 0),
            connected_roads=["west_in", "west_out", "east_in", "east_out", "north_in", "north_out", "south_in", "south_out"],
            has_signals=True,
            current_phase=0,
            phase_names=[p["name"] for p in self.phases],
        )
        self.signal_controllers["INT_CENTER"] = IntersectionSignalFSM(
            intersection_id="INT_CENTER",
            min_green=10.0,
            max_green=60.0,
            yellow_duration=3.0,
            all_red_duration=1.5,
        )

        self.roads["west_in"] = RoadSegment(
            id="west_in", from_node="NODE_W", to_node="INT_CENTER", length=R_LEN,
            points=[Vec3(-R_LEN, -L_WIDTH, 0), Vec3(0, -L_WIDTH, 0)], lanes=2
        )
        self.roads["west_out"] = RoadSegment(
            id="west_out", from_node="INT_CENTER", to_node="NODE_W", length=R_LEN,
            points=[Vec3(0, L_WIDTH, 0), Vec3(-R_LEN, L_WIDTH, 0)], lanes=2
        )
        self.roads["east_in"] = RoadSegment(
            id="east_in", from_node="NODE_E", to_node="INT_CENTER", length=R_LEN,
            points=[Vec3(R_LEN, L_WIDTH, 0), Vec3(0, L_WIDTH, 0)], lanes=2
        )
        self.roads["east_out"] = RoadSegment(
            id="east_out", from_node="INT_CENTER", to_node="NODE_E", length=R_LEN,
            points=[Vec3(0, -L_WIDTH, 0), Vec3(R_LEN, -L_WIDTH, 0)], lanes=2
        )
        self.roads["south_in"] = RoadSegment(
            id="south_in", from_node="NODE_S", to_node="INT_CENTER", length=R_LEN,
            points=[Vec3(L_WIDTH, -R_LEN, 0), Vec3(L_WIDTH, 0, 0)], lanes=2
        )
        self.roads["south_out"] = RoadSegment(
            id="south_out", from_node="INT_CENTER", to_node="NODE_S", length=R_LEN,
            points=[Vec3(-L_WIDTH, 0, 0), Vec3(-L_WIDTH, -R_LEN, 0)], lanes=2
        )
        self.roads["north_in"] = RoadSegment(
            id="north_in", from_node="NODE_N", to_node="INT_CENTER", length=R_LEN,
            points=[Vec3(-L_WIDTH, R_LEN, 0), Vec3(-L_WIDTH, 0, 0)], lanes=2
        )
        self.roads["north_out"] = RoadSegment(
            id="north_out", from_node="INT_CENTER", to_node="NODE_N", length=R_LEN,
            points=[Vec3(L_WIDTH, 0, 0), Vec3(L_WIDTH, R_LEN, 0)], lanes=2
        )

        # Procedural route pool
        self.route_pool = [
            ["west_in", "east_out"],
            ["east_in", "west_out"],
            ["south_in", "north_out"],
            ["north_in", "south_out"],
            ["west_in", "south_out"],
            ["east_in", "north_out"],
            ["south_in", "east_out"],
            ["north_in", "west_out"],
        ]

    def initialize(self, network_config: Optional[Dict[str, Any]] = None) -> None:
        self.reset()

    def set_demand(self, departures, name):
        self.base_departures = self.departures = tuple(departures)
        self.traffic_factor = 1
        self.demand_name = name
        self.restriction = {
            "lane_closure": ("road_nw_ne", 0, 20.0, 100.0, 0.0),
            "incident": ("road_nw_ne", None, 20.0, 110.0, 3.0),
            "gridlock": ("road_ne_out_e", None, 20.0, 140.0, 0.0),
        }.get(name)
        return self.reset()

    def set_traffic_factor(self, factor):
        if factor not in TRAFFIC_LEVELS or not self.base_departures:
            raise ValueError("Choose a traffic level for a loaded demand schedule.")
        self.departures = scale_departures(self, self.base_departures, factor)
        self.traffic_factor = factor
        return self.reset()

    def route_fits_vehicle(self, route, length):
        margin = 0.7 if self.map_name == "bronx_fordham" else 1.2
        for road_id in route:
            road = self.roads[road_id]
            if road.from_node in self.managed_junctions:
                space = length + margin if road.to_node in self.managed_junctions else length / 2 + 0.6
                if road.length < space:
                    return False
        return True

    def _speed_limit(self, road, lane):
        if self.restriction:
            target, closed_lane, start, end, limit = self.restriction
            if target == road.id and start <= self.sim_time < end and closed_lane in (None, lane):
                return limit
        incident = self.incident_limits.get(road.id)
        if incident and self.sim_time < incident[1]:
            return max(2.0, road.speed_limit * (1 - incident[0]))
        return road.speed_limit

    def _schedule_departures(self):
        while self.departure_index < len(self.departures):
            record = self.departures[self.departure_index]
            if record.time > self.sim_time + 1e-8:
                break
            length, _, color = FLEET[record.kind]
            vehicle = SimulatedVehicle(record.id, record.kind, list(record.route), color, record.speed,
                                       length, scheduled_spawn_time=record.time)
            vehicle.lane_index = record.lane
            self.pending_spawns.append(vehicle)
            self.all_scheduled_spawns.append({"id": record.id, "scheduled_time": record.time})
            self.departure_index += 1

    def demand_counts(self):
        due = len(self.all_scheduled_spawns)
        pending = len(self.pending_spawns)
        observed = OBSERVED_WINDOWS.get(self.demand_name)
        return {
            "dataset": self.demand_name,
            "traffic_factor": self.traffic_factor,
            "count_window": f"2013-01-{observed[0]:02} at {observed[1]:02}:00" if observed else None,
            "scheduled": len(self.departures) if self.departures is not None else due,
            "due": due,
            "pending": pending,
            "entered": len(self.vehicles) + self.total_arrived,
            "active": len(self.vehicles),
            "completed": self.total_arrived,
            "incomplete": len(self.vehicles),
            "never_entered": pending,
        }

    def reset(self) -> SimulationSnapshot:
        self.junction_reservations.clear()
        self.incident_limits.clear()
        self.departure_index = 0
        self.step_count = 0
        self.sim_time = 0.0
        self.vehicles.clear()
        self.pending_spawns.clear()
        self.all_scheduled_spawns.clear()
        self.next_scheduled_spawn_time = 0.0
        self.total_arrived = 0
        self.completed_trips.clear()
        self.last_spawn_time = 0.0
        self.vehicle_counter = 0
        for ctrl in self.signal_controllers.values():
            ctrl.reset()
        if self.signal_controllers:
            rep_ctrl = (
                self.signal_controllers.get("INT_CENTER")
                or next(iter(self.signal_controllers.values()))
            )
            self.signal_phase = rep_ctrl.get_phase_index()
            self.signal_timer = rep_ctrl.time_in_stage
        else:
            self.signal_phase = 0
            self.signal_timer = 0.0
        return self._get_snapshot()

    def _generate_scheduled_vehicle(self, scheduled_time: float) -> SimulatedVehicle:
        """Deterministically generates a scheduled vehicle according to PRNG streams."""
        if not self.route_pool:
            raise RuntimeError("Cannot spawn vehicles without route_pool loaded.")

        self.vehicle_counter += 1
        v_id = f"veh_{self.vehicle_counter}"

        # Fleet distribution sampling using rng_spawn if available, else random
        if self.rng_spawn is not None:
            fleet_roll = float(self.rng_spawn.random())
        else:
            fleet_roll = random.random()

        if fleet_roll < 0.70:
            v_type = "sedan"
            color = "#2563eb" if self.rng_spawn is not None else random.choice(["#2563eb", "#3b82f6", "#e2e8f0", "#475569", "#dc2626", "#0f172a"])
            desired_speed = 12.0 if self.rng_driver is not None else random.uniform(11.0, 13.5)
            length = 4.2
        elif fleet_roll < 0.85:
            v_type = "van"
            color = "#f59e0b" if self.rng_spawn is not None else random.choice(["#f59e0b", "#64748b", "#cbd5e1"])
            desired_speed = 11.0 if self.rng_driver is not None else random.uniform(10.0, 12.0)
            length = 5.4
        elif fleet_roll < 0.95:
            v_type = "bus"
            color = "#059669"
            desired_speed = 9.5 if self.rng_driver is not None else random.uniform(8.5, 10.5)
            length = 10.5
        else:
            v_type = "emergency"
            color = "#ef4444"
            desired_speed = 15.0 if self.rng_driver is not None else random.uniform(14.0, 16.5)
            length = 5.0

        # Apply stochastic driver speed multiplier if rng_driver is supplied
        if self.rng_driver is not None:
            mult = float(np.clip(self.rng_driver.normal(1.0, 0.10), 0.80, 1.25))
            desired_speed *= mult

        # Route selection
        if self.rng_spawn is not None:
            r_idx = int(self.rng_spawn.integers(0, len(self.route_pool)))
            route = self.route_pool[r_idx]
        else:
            route = random.choice(self.route_pool)

        if not self.route_fits_vehicle(route, length):
            v_type = "sedan"
            length, desired_speed, color = FLEET[v_type]
        first_road = self.roads.get(route[0])
        max_lane = max(0, (first_road.lanes if first_road else 1) - 1)

        if self.rng_spawn is not None:
            lane_idx = int(self.rng_spawn.integers(0, max_lane + 1))
        else:
            lane_idx = random.randint(0, max_lane)

        veh = SimulatedVehicle(
            vehicle_id=v_id,
            v_type=v_type,
            route=route,
            color=color,
            desired_speed=desired_speed,
            length=length,
            scheduled_spawn_time=round(scheduled_time, 4),
        )
        veh.lane_index = lane_idx

        self.all_scheduled_spawns.append({
            "id": veh.id,
            "scheduled_time": veh.scheduled_spawn_time,
            "scheduled_spawn_time": veh.scheduled_spawn_time,
            "type": veh.type,
            "initial_route": list(veh.route),
            "route": list(veh.route),
            "desired_speed": round(veh.desired_speed, 4),
            "length": veh.length,
        })

        return veh

    def _spawn_vehicle(self, initial_stagger: bool = False):
        """Spawns a vehicle on a valid pre-computed road route."""
        if not self.route_pool:
            return

        veh = self._generate_scheduled_vehicle(scheduled_time=self.sim_time)
        first_road = self.roads.get(veh.route[0])

        if initial_stagger and first_road and first_road.length > 20.0:
            existing = [v for v in self.vehicles.values() if v.current_road_id == first_road.id and v.lane_index == veh.lane_index]
            placed = False
            for _ in range(5):
                cand_dist = float(self.rng_spawn.uniform(8.0, first_road.length * 0.75)) if self.rng_spawn is not None else random.uniform(8.0, first_road.length * 0.75)
                if not any(abs(v.distance_on_road - cand_dist) < (veh.length + v.length) / 2 + 3.0 for v in existing):
                    veh.distance_on_road = cand_dist
                    veh.actual_entry_time = 0.0
                    veh.entry_delay = 0.0
                    self._update_vehicle_pose(veh, first_road)
                    self.vehicles[veh.id] = veh
                    placed = True
                    break
            if not placed:
                self.pending_spawns.append(veh)
        else:
            # Check entry space
            road_vehs = [v for v in self.vehicles.values() if v.current_road_id == veh.route[0] and v.lane_index == veh.lane_index]
            if any(v.distance_on_road < (veh.length + 4.0) for v in road_vehs):
                # Cannot enter immediately; queue in pending spawns
                self.pending_spawns.append(veh)
                return
            if first_road:
                self._update_vehicle_pose(veh, first_road)
            veh.actual_entry_time = round(self.sim_time, 4)
            veh.entry_delay = 0.0
            self.vehicles[veh.id] = veh

    def step(self, dt: float = 0.05, *, snapshot: bool = True) -> Optional[SimulationSnapshot]:
        """Simulates 1 discrete tick using IDM dynamics, signals, and polyline navigation."""
        self.step_count += 1
        self.sim_time += dt

        # 1. Update Traffic Signals
        for ctrl in self.signal_controllers.values():
            ctrl.step(dt, sim=self)
            if ctrl.intersection_id in self.intersections:
                self.intersections[ctrl.intersection_id].current_phase = ctrl.get_phase_index()

        if self.signal_controllers:
            rep_ctrl = (
                self.signal_controllers.get("INT_CENTER")
                or next(iter(self.signal_controllers.values()))
            )
            self.signal_phase = rep_ctrl.get_phase_index()
            self.signal_timer = rep_ctrl.time_in_stage

        # 2. Vehicle Scheduling & Physical Entry Processing
        default_limit = 220 if self.map_name in CITIES else 120 if self.map_name == "bronx_fordham" else 80
        max_vehicles = getattr(self, "max_vehicles", default_limit)

        # A. Schedule arrivals deterministically
        if self.departures is not None:
            self._schedule_departures()
        else:
            if self.next_scheduled_spawn_time <= 0.0 and self.step_count == 1:
                self.next_scheduled_spawn_time = round(self.sim_time + self.spawn_interval, 4)

            while self.sim_time >= self.next_scheduled_spawn_time:
                sched_veh = self._generate_scheduled_vehicle(self.next_scheduled_spawn_time)
                self.pending_spawns.append(sched_veh)
                self.next_scheduled_spawn_time = round(self.next_scheduled_spawn_time + self.spawn_interval, 4)
                self.last_spawn_time = self.sim_time

        occupancy = (VehicleOccupancy(self.vehicles.values())
                     if self.map_name == "bronx_fordham" and self.departures is not None else None)
        # B. Attempt entry for pending queued vehicles in FIFO order
        unadmitted = []
        for p_veh in self.pending_spawns:
            if len(self.vehicles) >= max_vehicles:
                unadmitted.append(p_veh)
                continue

            first_road = self.roads.get(p_veh.route[0])
            road_vehs = [v for v in self.vehicles.values() if v.current_road_id == p_veh.route[0] and v.lane_index == p_veh.lane_index]
            # Keep the entrance position close to the mapped edge. A larger
            # offset consumes most of the very short Fordham OSM links and
            # creates artificial pending demand even when the network has room.
            spawn_distance = (p_veh.length / 2 + 0.6
                              if self.departures is not None and first_road.from_node in self.managed_junctions else 0.0)
            upstream_node = first_road.from_node
            node_position = self.intersections.get(upstream_node)
            if self.map_name == "bronx_fordham" and self.departures is not None and node_position is not None:
                upstream_blocked = False
                for other in self.vehicles.values():
                    current = self.roads.get(other.current_road_id)
                    if current is None:
                        continue
                    # A short OSM link cannot hold a new entrance vehicle while
                    # its predecessor is still turning off the far end.
                    if (current.id.startswith("CONN__") and other.route_index > 0
                            and other.route[other.route_index - 1] == first_road.id):
                        gap_to_leader = first_road.length - spawn_distance + other.distance_on_road
                        if gap_to_leader < (other.length + p_veh.length) / 2 + 2.0:
                            upstream_blocked = True
                            break
                    if current.to_node != upstream_node:
                        continue
                    end_gap = math.hypot(current.points[-1].x - node_position.position.x,
                                         current.points[-1].y - node_position.position.y)
                    gap_to_node = max(0.0, current.length - other.distance_on_road) + end_gap
                    if gap_to_node < spawn_distance + (other.length + p_veh.length) / 2 + 1.0:
                        upstream_blocked = True
                        break
            else:
                upstream_blocked = False
            if (first_road.from_node in self.junction_reservations
                    or upstream_blocked
                    or any(v.distance_on_road < spawn_distance + (p_veh.length + v.length) / 2 + 3.0
                           for v in road_vehs)):
                # Entrance headway blocked; remain in pending queue
                unadmitted.append(p_veh)
                continue

            # Safely admitted into physical network
            p_veh.distance_on_road = spawn_distance
            self._update_vehicle_pose(p_veh, first_road)
            if occupancy and occupancy.blocked(p_veh):
                unadmitted.append(p_veh)
                continue
            if occupancy:
                occupancy.update(p_veh)
            p_veh.actual_entry_time = round(self.sim_time, 4)
            p_veh.entry_delay = round(max(0.0, self.sim_time - p_veh.scheduled_spawn_time), 4)
            p_veh.blocked_entry_time = p_veh.entry_delay
            self.vehicles[p_veh.id] = p_veh

        self.pending_spawns = unadmitted

        # 3. Sort vehicles per road & lane to compute car-following gaps
        road_lanes: Dict[str, Dict[int, List[SimulatedVehicle]]] = {}
        vehicles_by_road: Dict[str, Dict[str, SimulatedVehicle]] = {}
        for veh in self.vehicles.values():
            r_id = veh.current_road_id
            if not r_id:
                continue
            if r_id not in road_lanes:
                road_lanes[r_id] = {}
            if veh.lane_index not in road_lanes[r_id]:
                road_lanes[r_id][veh.lane_index] = []
            road_lanes[r_id][veh.lane_index].append(veh)
            vehicles_by_road.setdefault(r_id, {})[veh.id] = veh

        for r_id in road_lanes:
            for lane_idx in road_lanes[r_id]:
                road_lanes[r_id][lane_idx].sort(key=lambda v: v.distance_on_road, reverse=True)

        # 4. Advance each vehicle using Intelligent Driver Model (IDM)
        arrived_ids = []
        rear_margin = 0.05 if self.map_name == "bronx_fordham" else 0.5
        for node, reservations in list(self.junction_reservations.items()):
            for vehicle_id in list(reservations):
                vehicle = self.vehicles.get(vehicle_id)
                if vehicle is None:
                    del reservations[vehicle_id]
                    continue
                current = self.roads[vehicle.current_road_id]
                if (current.from_node == node and not current.id.startswith("CONN__")
                        and vehicle.distance_on_road >= vehicle.length / 2 + rear_margin):
                    del reservations[vehicle_id]
            if not reservations:
                del self.junction_reservations[node]
        ordered_vehicles = [vehicle for lanes in road_lanes.values()
                            for lane in lanes.values() for vehicle in lane]
        def junction_order(vehicle):
            road = self.roads.get(vehicle.current_road_id)
            if road is None or road.to_node not in self.managed_junctions:
                return (1, "", 1, 1, 0.0, vehicle.id)
            node = road.to_node
            held = vehicle.id in self.junction_reservations.get(node, {})
            distance_to_line = max(0.0, road.length - vehicle.distance_on_road)
            green_priority = 1
            target = self.intersections.get(node)
            if target and target.has_signals and road.signal_controlled:
                is_ns = is_ns_approach(road)
                signal = self.signal_controllers.get(node)
                code = signal.get_signal_code(is_ns=is_ns) if signal else "r"
                green_priority = 0 if code == "G" else 1
            return (0, node, 0 if held else 1, green_priority, distance_to_line, vehicle.id)

        if self.map_name == "bronx_fordham":
            ordered_vehicles.sort(key=junction_order)
        probe = getattr(self, 'route_probe', None)
        for veh in ordered_vehicles:
            observed = probe is not None and veh.id == probe.vehicle_id
            wait_before = veh.wait_time if observed else 0
            veh.total_travel_time += dt
            r_id = veh.current_road_id
            road = self.roads.get(r_id)
            if not road:
                arrived_ids.append(veh.id)
                continue

            reservations = self.junction_reservations.get(road.to_node, {})
            committed = veh.id in reservations
            next_id = veh.route[veh.route_index + 1] if veh.route_index + 1 < len(veh.route) else ""
            exit_lane = int(r_id.rsplit("__", 1)[1]) if r_id.startswith("CONN__") else veh.lane_index
            next_road_for_lane = self.roads.get(next_id)
            approach_managed = road.to_node in self.managed_junctions
            if approach_managed and next_road_for_lane:
                exit_lane = min(exit_lane, next_road_for_lane.lanes - 1)
            movement = (r_id, next_id, veh.lane_index)
            junction_blocked = not committed and any(
                self._movements_conflict(movement, veh.length, other, self.vehicles[owner].length)
                for owner, other in reservations.items()
            )

            # Find leader in the same lane
            lane_vehicles = road_lanes.get(r_id, {}).get(veh.lane_index, [])
            veh_ahead: Optional[SimulatedVehicle] = None
            try:
                idx = lane_vehicles.index(veh)
                if idx > 0:
                    veh_ahead = lane_vehicles[idx - 1]
            except ValueError:
                pass

            gap = 999.0
            leader_speed = veh.desired_speed
            downstream_leader = None

            if veh_ahead and veh_ahead.current_road_id == r_id:
                raw_gap = veh_ahead.distance_on_road - veh.distance_on_road - (veh_ahead.length + veh.length) / 2
                if raw_gap < 0.5:
                    gap = 0.5
                    leader_speed = max(0.0, veh_ahead.speed)
                else:
                    gap = raw_gap
                    leader_speed = veh_ahead.speed
            else:
                # Check for signalized intersection stop line at end of road
                target_node = self.intersections.get(road.to_node)
                dist_to_stop = road.length - veh.distance_on_road
                if target_node and target_node.has_signals and road.signal_controlled and dist_to_stop > 0:
                    is_ns = is_ns_approach(road)
                    ctrl = self.signal_controllers.get(target_node.id)
                    if ctrl is not None:
                        sig_code = ctrl.get_signal_code(is_ns=is_ns)
                    else:
                        phase_info = self.phases[min(self.signal_phase, len(self.phases) - 1)]
                        sig_code = phase_info["ns"] if is_ns else phase_info["ew"]

                    if not committed and (sig_code in ("r", "y") or junction_blocked):
                        hold_offset = self._junction_hold_offset(road, veh.length)
                        gap = max(0.5, dist_to_stop - hold_offset - veh.length / 2)
                        leader_speed = 0.0

            # Look ahead across the next link even when there is a same-road
            # leader: a vehicle that has entered a connector is still the leader
            # for traffic behind it on the approach.
            if next_id and road.length - veh.distance_on_road < 35.0 and (self.map_name == "bronx_fordham" or gap > 80.0):
                next_r_id = veh.route[veh.route_index + 1]
                next_road = self.roads.get(next_r_id)
                if next_road:
                    candidates = (self.vehicles.values() if self.map_name == "bronx_fordham"
                                  else vehicles_by_road.get(next_r_id, {}).values())
                    next_vehs = [
                        other for other in candidates
                        if other.lane_index == min(exit_lane, next_road.lanes - 1)
                        and (other.current_road_id == next_r_id
                             or (self.map_name == "bronx_fordham" and other.route_index > 0
                                 and other.route[other.route_index - 1] == r_id
                                 and other.route_index + 1 < len(other.route)
                                 and other.route[other.route_index + 1] == next_r_id))
                    ]
                    if next_vehs:
                        closest_next = min(next_vehs, key=lambda v: v.distance_on_road)
                        dist_to_leader = (road.length - veh.distance_on_road) + closest_next.distance_on_road
                        if dist_to_leader < 35.0:
                            downstream_gap = max(0.5, dist_to_leader - (closest_next.length + veh.length) / 2)
                            if downstream_gap < gap:
                                gap = downstream_gap
                                leader_speed = closest_next.speed
                            if self.map_name == "bronx_fordham":
                                downstream_leader = closest_next

            # IDM Acceleration Calculation
            a_max = 2.0
            b_comf = 2.5
            s_0 = 2.5
            T = 1.4

            effective_v0 = min(veh.desired_speed, self._speed_limit(road, veh.lane_index))
            if self.junction_movements.get((r_id, next_id)) in ("left", "right"):
                turn_approach_speed = math.sqrt(6.0 ** 2 + 2 * b_comf * max(0.0, road.length - veh.distance_on_road))
                effective_v0 = min(effective_v0, turn_approach_speed)
            delta_v = veh.speed - leader_speed
            s_star = s_0 + max(0.0, veh.speed * T + (veh.speed * delta_v) / (2.0 * math.sqrt(a_max * b_comf)))
            accel = a_max * (1.0 - (veh.speed / max(0.1, effective_v0)) ** 4 - (s_star / max(0.5, gap)) ** 2)
            veh.accel = max(-7.0, min(a_max, accel))

            # Anti-deadlock creep: if stopped and gap opened up in front, allow creep
            if veh.speed < 0.2 and gap > 2.5 and leader_speed >= 0.0:
                veh.accel = max(0.6, veh.accel)

            veh.speed = max(0.0, veh.speed + veh.accel * dt)
            start_distance = veh.distance_on_road
            proposed = start_distance + veh.speed * dt
            lane_safety = 3.0 if self.map_name == "bronx_fordham" else 1.0
            if veh_ahead and veh_ahead.current_road_id == r_id:
                limit = veh_ahead.distance_on_road - (veh_ahead.length + veh.length) / 2 - lane_safety
                proposed = min(proposed, max(start_distance, limit))
            if self.map_name == "bronx_fordham" and downstream_leader is not None:
                limit = road.length + downstream_leader.distance_on_road - (downstream_leader.length + veh.length) / 2 - lane_safety
                proposed = min(proposed, max(start_distance, limit))
            veh.distance_on_road = proposed
            veh.speed = (proposed - start_distance) / dt

            if occupancy and veh.speed > 0:
                self._update_vehicle_pose(veh, road)
                if occupancy.blocked(veh):
                    # Do not roll the whole vehicle back when a proposed step
                    # would overlap a nearby body. A hard rollback makes two
                    # vehicles on intersecting approaches deadlock nose-to-nose.
                    # Find the furthest collision-free point in this short step
                    # so the vehicles can brake and creep around tight geometry.
                    low, high = start_distance, proposed
                    for _ in range(12):
                        candidate = (low + high) / 2.0
                        veh.distance_on_road = candidate
                        self._update_vehicle_pose(veh, road)
                        if occupancy.blocked(veh):
                            high = candidate
                        else:
                            low = candidate
                    veh.distance_on_road = low
                    veh.speed = max(0.0, (low - start_distance) / dt)
                    veh.accel = min(veh.accel, (veh.speed - (proposed - start_distance) / dt) / dt)

            if veh.speed < 0.5:
                veh.wait_time += dt

            # Transition to next road in route or hold before red/blocked intersection
            target_node = self.intersections.get(road.to_node)
            stop_distance = max(0.0, road.length - veh.length / 2 - 0.6
                                - self._junction_hold_offset(road, veh.length))
            is_approach = approach_managed and bool(next_id)
            if veh.distance_on_road >= (stop_distance if is_approach else road.length):
                target_node = self.intersections.get(road.to_node)
                is_red = False
                if target_node and target_node.has_signals and road.signal_controlled:
                    is_ns = is_ns_approach(road)
                    ctrl = self.signal_controllers.get(target_node.id)
                    sig_code = ctrl.get_signal_code(is_ns=is_ns) if ctrl is not None else "G"
                    if sig_code in ("r", "y"):
                        is_red = True

                if not committed and (is_red or (is_approach and junction_blocked)):
                    veh.distance_on_road = max(start_distance, stop_distance)
                    veh.speed = 0.0
                    veh.accel = 0.0
                elif veh.route_index + 1 >= len(veh.route):
                    arrived_ids.append(veh.id)
                else:
                    next_r_id = veh.route[veh.route_index + 1]
                    next_road = self.roads.get(next_r_id)
                    if not next_road:
                        arrived_ids.append(veh.id)
                    else:
                        next_vehs = vehicles_by_road.get(next_r_id, {})
                        max_storage = max(2, int((next_road.length * next_road.lanes) / 7.0))
                        min_dist_per_lane = {l: 999.0 for l in range(next_road.lanes)}
                        lane_counts = {l: 0 for l in range(next_road.lanes)}
                        for other in next_vehs.values():
                            l = min(other.lane_index, next_road.lanes - 1)
                            lane_counts[l] = lane_counts.get(l, 0) + 1
                            min_dist_per_lane[l] = min(min_dist_per_lane[l], other.distance_on_road - other.length / 2)

                        next_lane = min(exit_lane, next_road.lanes - 1)
                        # Reserve space for the entire body beyond the junction. A
                        # front-bumper-only check can leave the tail in the conflict zone.
                        required_space = veh.length + 3.5 if is_approach else veh.length / 2 + 3.0
                        available = [
                            lane for lane in ([next_lane] if approach_managed or self.map_name == "bronx_fordham" or road.id.startswith("CONN__") else range(next_road.lanes))
                            if min_dist_per_lane[lane] >= required_space
                            and (road.id.startswith("CONN__") or self._speed_limit(next_road, lane) > 0)
                        ]
                        road_overloaded = not (is_approach or road.id.startswith("CONN__")) and len(next_vehs) >= max_storage
                        if not available or road_overloaded:
                            limit = stop_distance if is_approach and not committed else road.length - 0.01
                            veh.distance_on_road = max(start_distance, limit)
                            veh.speed = 0.0
                            veh.accel = 0.0
                        else:
                            if is_approach:
                                if (road.id, next_road.id) not in self.junction_movements:
                                    raise RuntimeError(f"Invalid movement: {road.id} to {next_road.id}")
                                self.junction_reservations.setdefault(road.to_node, {})[veh.id] = movement
                            if veh.distance_on_road < road.length:
                                veh.distance_traveled += max(0.0, veh.distance_on_road - start_distance)
                                self._update_vehicle_pose(veh, road)
                                if occupancy:
                                    occupancy.update(veh)
                                continue
                            best_lane = max(available, key=lambda l: (min_dist_per_lane[l], -lane_counts[l]))
                            overflow = max(0.0, veh.distance_on_road - road.length)

                            # Root-cause fix: a vehicle whose route jumps from this
                            # road directly onto `next_road` (a turn) would otherwise
                            # teleport from this road's lane-specific end point to
                            # next_road's lane-specific start point in a single tick,
                            # since those points are not coincident for any turning
                            # movement (only collinear straight-throughs line up).
                            # Splice in a tangent-matched connector segment so the
                            # vehicle instead drives a continuous curved path through
                            # the intersection. Only bridge between real roads (never
                            # chain a connector to another connector).
                            connector = None
                            if not road.id.startswith("CONN__") and not next_road.id.startswith("CONN__"):
                                connector = self._get_or_build_turn_connector(
                                    road, veh.lane_index, next_road, best_lane
                                )
                            if connector is not None:
                                veh.route.insert(veh.route_index + 1, connector.id)
                                veh.route_index += 1
                                veh.lane_index = 0  # connectors are single-lane
                                veh.distance_on_road = min(overflow, connector.length)
                                road = connector
                            else:
                                veh.route_index += 1
                                veh.lane_index = best_lane
                                veh.distance_on_road = min(overflow, max(0.0, min_dist_per_lane[best_lane] - veh.length / 2 - 1.0))
                                veh.distance_on_road = max(0.0, veh.distance_on_road)
                                road = next_road

            if road.id == r_id:
                veh.distance_traveled += max(0.0, veh.distance_on_road - start_distance)
            else:
                vehicles_by_road[r_id].pop(veh.id)
                vehicles_by_road.setdefault(road.id, {})[veh.id] = veh
                veh.distance_traveled += max(0.0, self.roads[r_id].length - start_distance) + veh.distance_on_road

            if observed:
                leader = veh_ahead or downstream_leader
                probe.observe(self, veh, self.roads[r_id], veh.wait_time - wait_before,
                              leader.id if leader and gap <= 12 else None,
                              junction_blocked, committed)

            # Update spatial 3D pose along road polyline.
            self._update_vehicle_pose(veh, road)
            if occupancy:
                occupancy.update(veh)

        # Cleanup arrived vehicles
        for a_id in arrived_ids:
            if a_id in self.vehicles:
                veh = self.vehicles[a_id]
                entry_t = veh.actual_entry_time if veh.actual_entry_time is not None else veh.scheduled_spawn_time
                e_delay = veh.entry_delay
                self.completed_trips.append({
                    "vehicle_id": veh.id,
                    "type": veh.type,
                    "scheduled_spawn_time": round(veh.scheduled_spawn_time, 4),
                    "actual_entry_time": round(entry_t, 4),
                    "spawn_time": round(entry_t, 4),  # backward compatibility with legacy analyzers
                    "entry_delay_sec": round(e_delay, 4),
                    "arrival_time": round(self.sim_time, 4),
                    "network_travel_time_sec": round(veh.total_travel_time, 4),
                    "travel_time_sec": round(veh.total_travel_time, 4),  # backward compatibility
                    "system_travel_time_sec": round(self.sim_time - veh.scheduled_spawn_time, 4),
                    "wait_time_sec": round(veh.wait_time, 4),
                    "total_system_wait_time_sec": round(veh.wait_time + e_delay, 4),
                    "distance_traveled_m": round(veh.distance_traveled, 4),
                    "reroute_count": veh.reroute_count,
                    "initial_route": list(veh.route_history[0]),
                    "final_route": list(veh.route),
                })
                del self.vehicles[a_id]
                self.total_arrived += 1

        return self._get_snapshot() if snapshot else None

    def get_never_entered_vehicles(self) -> List[Dict[str, Any]]:
        """Returns records of vehicles that were scheduled by the exogenous demand stream but never entered."""
        return [
            {
                "vehicle_id": v.id,
                "type": v.type,
                "scheduled_spawn_time": round(v.scheduled_spawn_time, 4),
                "actual_entry_time": None,
                "entry_delay_sec": round(max(0.0, self.sim_time - v.scheduled_spawn_time), 4),
                "status": "NEVER_ENTERED",
                "initial_route": list(v.route),
                "desired_speed": round(v.desired_speed, 4),
                "length": v.length,
            }
            for v in self.pending_spawns
        ]

    def _lane_points(self, road, lane):
        offset = (lane - (road.lanes - 1) / 2.0) * 3.5
        if not offset:
            return road.points
        if not hasattr(self, "_lane_geometry"):
            self._lane_geometry = {}
        key = (road.id, lane)
        if key not in self._lane_geometry:
            self._lane_geometry[key] = [offset_point(road.points, i, offset) for i in range(len(road.points))]
        return self._lane_geometry[key]

    def _get_lane_point_and_tangent(self, road: RoadSegment, lane_index: int, at_end: bool):
        pts = self._lane_points(road, lane_index)
        p0, p1 = (pts[-2], pts[-1]) if at_end else (pts[0], pts[1])
        dx, dy = p1.x - p0.x, p1.y - p0.y
        length = max(0.001, math.hypot(dx, dy))
        base = p1 if at_end else p0
        return (base.x, base.y), (dx / length, dy / length)

    def _movements_conflict(self, movement, length, other, other_length):
        # Use exclusive clearance on irregular OSM junctions. Classroom and
        # corridor approaches have separated, axis-aligned paths for concurrency.
        if self.map_name == "bronx_fordham":
            if movement[:2] == other[:2] and movement[2] == other[2]:
                # Successive vehicles in the same lane and movement are handled
                # by longitudinal following, not by intersection arbitration.
                return False
            return True
        entry, exit_id, lane = movement
        other_entry, other_exit, other_lane = other
        if lane == other_lane and (entry == other_entry or exit_id == other_exit):
            return True
        kind = self.classroom_movements.get((entry, exit_id))
        other_kind = self.classroom_movements.get((other_entry, other_exit))
        if kind not in ("straight", "right") or other_kind not in ("straight", "right"):
            return True
        a = self._movement_envelope(movement, length, kind)
        b = self._movement_envelope(other, other_length, other_kind)
        return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]

    def _movement_envelope(self, movement, length, kind):
        """Conservative footprint from the stop line to rear clearance."""
        key = (movement, length)
        if key not in self._movement_bounds:
            entry, exit_id, lane = movement
            incoming, outgoing = self.roads[entry], self.roads[exit_id]
            start, approach = self._get_lane_point_and_tangent(incoming, lane, at_end=True)
            end, departure = self._get_lane_point_and_tangent(outgoing, lane, at_end=False)
            extension = length / 2 + 0.6
            approach_extension = extension + self._junction_hold_offset(incoming, length)
            xs = [start[0] - approach[0] * approach_extension, end[0] + departure[0] * extension]
            ys = [start[1] - approach[1] * approach_extension, end[1] + departure[1] * extension]
            half_width = 1.4  # The widest fleet model is 2.6 m.
            if kind == "straight":
                longitudinal = length / 2 + 0.1
                pad_x = abs(approach[0]) * longitudinal + abs(approach[1]) * half_width
                pad_y = abs(approach[1]) * longitudinal + abs(approach[0]) * half_width
            else:
                connector = self._get_or_build_turn_connector(incoming, lane, outgoing, lane)
                xs.extend(point.x for point in connector.points)
                ys.extend(point.y for point in connector.points)
                pad_x = pad_y = math.hypot(length / 2, half_width)
            self._movement_bounds[key] = (min(xs) - pad_x, min(ys) - pad_y,
                                          max(xs) + pad_x, max(ys) + pad_y)
        return self._movement_bounds[key]

    def _get_or_build_turn_connector(
        self, prev_road: RoadSegment, prev_lane: int, next_road: RoadSegment, next_lane: int
    ) -> Optional[RoadSegment]:
        """Builds (and caches) a short connector RoadSegment bridging the exact
        lane-specific end point of `prev_road` to the exact lane-specific start
        point of `next_road`, used to drive vehicles through an intersection
        turn with a continuous, tangent-matched trajectory instead of an
        instantaneous jump.

        Returns None when the two points are already coincident (a straight
        continuation, e.g. two collinear through-roads) and no connector is
        needed.
        """
        if prev_road.id.startswith("CONN__") or next_road.id.startswith("CONN__"):
            return None

        if not hasattr(self, "_connector_cache"):
            self._connector_cache: Dict[Any, RoadSegment] = {}

        cache_key = (prev_road.id, prev_lane, next_road.id, next_lane)
        cached = self._connector_cache.get(cache_key)
        if cached is not None:
            return cached if cached.length > 0 else None

        if len(prev_road.points) < 2 or len(next_road.points) < 2:
            self._connector_cache[cache_key] = RoadSegment(
                id=f"__nogeom__{cache_key}", from_node="", to_node="", length=0.0, points=[]
            )
            return None

        p_start, t_start = self._get_lane_point_and_tangent(prev_road, prev_lane, at_end=True)
        p_end, t_end = self._get_lane_point_and_tangent(next_road, next_lane, at_end=False)

        gap = math.hypot(p_end[0] - p_start[0], p_end[1] - p_start[1])
        if gap < 0.05:
            # Already continuous (straight-through movement) -- no connector needed.
            self._connector_cache[cache_key] = RoadSegment(
                id=f"__nogeom__{cache_key}", from_node="", to_node="", length=0.0, points=[]
            )
            return None

        # Cubic Bezier with control points extended along each end's actual
        # tangent, so heading is continuous (no corner kink) at both the
        # hand-off from prev_road and the hand-off onto next_road.
        ctrl_dist = max(1.0, gap * 0.5)
        c1 = (p_start[0] + t_start[0] * ctrl_dist, p_start[1] + t_start[1] * ctrl_dist)
        c2 = (p_end[0] - t_end[0] * ctrl_dist, p_end[1] - t_end[1] * ctrl_dist)

        n_samples = max(12, math.ceil(gap / 1.0))
        pts: List[Vec3] = []
        for i in range(n_samples + 1):
            s = i / n_samples
            mt = 1.0 - s
            x = (mt ** 3) * p_start[0] + 3 * (mt ** 2) * s * c1[0] + 3 * mt * (s ** 2) * c2[0] + (s ** 3) * p_end[0]
            y = (mt ** 3) * p_start[1] + 3 * (mt ** 2) * s * c1[1] + 3 * mt * (s ** 2) * c2[1] + (s ** 3) * p_end[1]
            pts.append(Vec3(x, y, 0.0))

        arc_length = sum(
            math.hypot(pts[i + 1].x - pts[i].x, pts[i + 1].y - pts[i].y) for i in range(len(pts) - 1)
        )

        connector = RoadSegment(
            id=f"CONN__{prev_road.id}__{prev_lane}__{next_road.id}__{next_lane}",
            from_node=prev_road.to_node,
            # Deliberately NOT next_road.from_node (the real intersection id):
            # the signal gate was already satisfied to enter this connector, so
            # to_node is left as a non-intersection sentinel to avoid a second,
            # redundant signal check stranding the vehicle mid-turn if the
            # phase flips during the brief connector traversal.
            to_node="",
            lanes=1,
            length=max(0.5, arc_length),
            speed_limit=min(prev_road.speed_limit, next_road.speed_limit)
            if self.junction_movements.get((prev_road.id, next_road.id)) == "straight"
            else min(prev_road.speed_limit, next_road.speed_limit, 6.0),
            capacity=min(prev_road.capacity, next_road.capacity),
            points=pts,
        )
        self.roads[connector.id] = connector
        self._connector_cache[cache_key] = connector
        return connector

    def _update_vehicle_pose(self, veh: SimulatedVehicle, road: RoadSegment):
        """Calculates 3D world coordinates and heading along arbitrary polyline road geometry."""
        if not road.points or len(road.points) < 2:
            return

        points = road.points
        if len(points) == 2:
            if not hasattr(self, "_straight_pose"):
                self._straight_pose = {}
            key = (road.id, veh.lane_index)
            pose = self._straight_pose.get(key)
            if pose is None:
                left, right = self._lane_points(road, veh.lane_index)
                dx, dy = right.x - left.x, right.y - left.y
                a, b = points
                length = max(0.001, math.hypot(b.x - a.x, b.y - a.y))
                pose = (left.x, left.y, dx, dy, length, math.degrees(math.atan2(dx, dy)) % 360)
                self._straight_pose[key] = pose
            x, y, dx, dy, length, heading = pose
            ratio = min(1.0, max(0.0, veh.distance_on_road / length))
            veh.x, veh.y, veh.z, veh.heading = x + dx * ratio, y + dy * ratio, 0.45, heading
            return
        target_dist = veh.distance_on_road
        accum = 0.0

        p0, p1 = points[0], points[1]
        dx = p1.x - p0.x
        dy = p1.y - p0.y
        seg_len = max(0.001, math.hypot(dx, dy))

        for i in range(len(points) - 1):
            curr_p0 = points[i]
            curr_p1 = points[i + 1]
            cur_dx = curr_p1.x - curr_p0.x
            cur_dy = curr_p1.y - curr_p0.y
            cur_len = max(0.001, math.hypot(cur_dx, cur_dy))

            if accum + cur_len >= target_dist or i == len(points) - 2:
                p0, p1 = curr_p0, curr_p1
                dx, dy = cur_dx, cur_dy
                seg_len = cur_len
                target_dist -= accum
                break
            accum += cur_len

        progress = min(1.0, max(0.0, target_dist / seg_len))

        lane_points = self._lane_points(road, veh.lane_index)
        left, right = lane_points[i], lane_points[i + 1]
        dx, dy = right.x - left.x, right.y - left.y
        veh.x = left.x + progress * dx
        veh.y = left.y + progress * dy
        veh.z = 0.45

        angle_rad = math.atan2(dx, dy)
        veh.heading = (math.degrees(angle_rad) + 360.0) % 360.0

    def _get_snapshot(self) -> SimulationSnapshot:
        """Constructs an immutable telemetry snapshot."""
        v_states = [
            VehicleState(
                id=v.id,
                type=v.type,
                x=v.x,
                y=v.y,
                z=v.z,
                speed=v.speed,
                heading=v.heading,
                road_id=v.current_road_id,
                lane_index=v.lane_index,
                accel=v.accel,
                wait_time=v.wait_time,
                color=v.color,
                route=list(v.route),
                route_index=v.route_index,
                reroute_count=getattr(v, "reroute_count", 0),
                original_route=list(v.route_history[0]) if getattr(v, "route_history", None) else list(v.route),
            )
            for v in self.vehicles.values()
        ]

        sig_states = [
            ctrl.get_signal_state()
            for ctrl in self.signal_controllers.values()
        ]

        active_count = len(self.vehicles)
        if active_count > 0:
            avg_spd_kmh = (sum(v.speed for v in self.vehicles.values()) / active_count) * 3.6
            avg_delay = sum(v.wait_time for v in self.vehicles.values()) / active_count
        else:
            avg_spd_kmh = 0.0
            avg_delay = 0.0

        throughput_hr = (self.total_arrived / max(1.0, self.sim_time)) * 3600.0
        congestion = max(0.0, min(1.0, 1.0 - (avg_spd_kmh / 45.0))) if active_count else 0.0

        metrics = NetworkMetrics(
            active_vehicles=active_count,
            total_arrived=self.total_arrived,
            throughput_per_hour=throughput_hr,
            avg_speed_kmh=avg_spd_kmh,
            avg_delay_sec=avg_delay,
            congestion_index=congestion,
        )

        return SimulationSnapshot(
            timestamp=self.sim_time,
            step=self.step_count,
            vehicles=v_states,
            signals=sig_states,
            metrics=metrics,
        )

    def _road_visual_data(self, road):
        result = road.to_dict()
        if self.map_name == "bronx_fordham":
            width = (road.visual_lanes or road.lanes) * 1.75
            result["left_edge"] = [offset_point(road.points, i, width).to_dict()
                                   for i in range(len(road.points))]
            result["right_edge"] = [offset_point(road.points, i, -width).to_dict()
                                    for i in range(len(road.points))]
        return result

    def get_network_topology(self) -> Dict[str, Any]:
        """Returns geographic definitions of roads, intersections, and 3D buildings."""
        route = next((route for route in self.scheduled_routes or self.route_pool
                      if self.route_fits_vehicle(route, FLEET["sedan"][0])), None)
        return {
            "map_name": self.map_name,
            "metadata": self.map_metadata,
            "intersections": [i.to_dict() for i in self.intersections.values()],
            "roads": [self._road_visual_data(r) for r in self.roads.values() if not r.id.startswith("CONN__")],
            "buildings": self.buildings,
            "managed_junctions": sorted(self.managed_junctions),
            "route_defaults": {"from": self.roads[route[0]].from_node, "to": self.roads[route[-1]].to_node}
            if route else None,
            "movements": [
                {"from_road": incoming, "to_road": outgoing, "turn": turn}
                for (incoming, outgoing), turn in self.junction_movements.items()
            ],
        }

    def apply_signal_phases(self, phase_commands: Dict[str, int]) -> None:
        """Applies phase commands to signalized intersections independently.

        Args:
            phase_commands: Mapping of intersection_id to requested target phase.
                            e.g. {"INT_CENTER": 1, "node_123": 0}
        """
        if not phase_commands or not isinstance(phase_commands, dict):
            return

        for inter_id, target_phase in phase_commands.items():
            if not isinstance(inter_id, str):
                continue
            ctrl = self.signal_controllers.get(inter_id)
            if ctrl is not None:
                if (inter_id in self.managed_junctions and ctrl.stage != "GREEN"
                        and ctrl.target_green_phase in (0, 1)):
                    # Preserve the phase that earned service through its clearance,
                    # including a forced switch when maximum green expires.
                    continue
                try:
                    ctrl.request_phase(int(target_phase))
                except (ValueError, TypeError):
                    pass

        # Maintain representative legacy attributes for backward compatibility
        if self.signal_controllers:
            rep_ctrl = (
                self.signal_controllers.get("INT_CENTER")
                or next(iter(self.signal_controllers.values()))
            )
            self.signal_phase = rep_ctrl.get_phase_index()
            self.signal_timer = rep_ctrl.time_in_stage

    def inject_incident(self, road_id: str, severity: float, duration_sec: float) -> bool:
        if road_id in self.roads:
            self.incident_limits[road_id] = (severity, self.sim_time + duration_sec)
            return True
        return False

    def trigger_congestion_surge(self, origin_road_id: str = "", vehicle_count: int = 15) -> bool:
        added = 0
        target_route = [origin_road_id] if origin_road_id in self.roads else None
        for i in range(vehicle_count):
            self.vehicle_counter += 1
            v_id = f"surge_{self.vehicle_counter}"
            route = target_route if target_route else (random.choice(self.route_pool) if self.route_pool else [])
            if not route:
                continue
            r_id = route[0]
            road = self.roads.get(r_id)
            if not road:
                continue
            veh = SimulatedVehicle(
                vehicle_id=v_id,
                v_type="sedan",
                route=route,
                color="#dc2626",
                desired_speed=11.2,
            )
            num_lanes = max(1, road.lanes)
            veh.lane_index = i % num_lanes
            stagger_spacing = veh.length + 4.0
            stagger_pos = (i // num_lanes + 1) * stagger_spacing
            veh.distance_on_road = max(2.0, min(max(2.0, road.length * 0.85), stagger_pos))
            self._update_vehicle_pose(veh, road)
            self.vehicles[v_id] = veh
            added += 1
        return added > 0

    def reroute_vehicle(self, vehicle_id: str, new_subpath: List[str]) -> Dict[str, Any]:
        """Dynamically reroutes an in-flight vehicle to an alternative valid downstream path.

        Preserves:
            - Current road segment and vehicle progression (distance_on_road)
            - Exact spatial coordinates (x, y, z) and heading
            - Kinematic properties (speed, acceleration, desired_speed)
            - Accumulated trip metrics (wait_time, total_travel_time)
            - Assigned lane index
            - Prior route history

        Validation Invariants:
            1. Vehicle existence: Vehicle must be currently active in the simulator fleet.
            2. Active transit: Vehicle must not have already reached the end of its route.
            3. Non-empty subpath: Candidate subpath must contain at least one road.
            4. Valid network roads: All road IDs in candidate subpath must exist in self.roads.
            5. Candidate normalization: If new_subpath[0] matches current_road_id, downstream
               sequence is new_subpath[1:]; otherwise new_subpath is treated as the downstream sequence.
            6. Downstream existence: At least one downstream road must remain unless already at destination.
            7. Directed adjacency:
               - The first downstream road must originate at the current road's to_node.
               - Each subsequent road must originate at the prior road's to_node.
            8. Destination preservation:
               - The final road of the new route must terminate at the original route's final destination node
                 (or match the original destination road ID).
            9. No unintended U-turn:
               - The first downstream road must not be an immediate 180-degree reversal returning to current road's from_node
                 (unless an explicit one-way turnaround or self-connecting topology).
               - Subsequent adjacent pairs in the new path must not reverse back to the immediate predecessor's from_node.

        Args:
            vehicle_id: ID of the simulated vehicle.
            new_subpath: Sequence of road IDs for the remaining journey.

        Returns:
            Dict containing:
                - success (bool): True if reroute succeeded or was a valid no-op.
                - reason (str): Human-readable explanation.
                - vehicle_id (str): ID of the vehicle.
                - rerouted (bool): True if route was altered; False if no-op or failed.
                - old_route (List[str], optional): Route before reroute.
                - new_route (List[str], optional): Route after reroute.
                - current_road_id (str, optional): Current road vehicle is on.
                - route_index (int, optional): Preserved route index.
        """
        # 1. Vehicle existence check
        if vehicle_id not in self.vehicles:
            return {
                "success": False,
                "rerouted": False,
                "vehicle_id": vehicle_id,
                "reason": f"Vehicle '{vehicle_id}' not found in active fleet.",
            }

        veh = self.vehicles[vehicle_id]
        if any(vehicle_id in reservations for reservations in self.junction_reservations.values()):
            return {"success": False, "rerouted": False, "vehicle_id": vehicle_id,
                    "reason": "Vehicle is clearing a reserved junction."}

        # 2. Active transit check
        if veh.route_index >= len(veh.route) or not veh.current_road_id:
            return {
                "success": False,
                "rerouted": False,
                "vehicle_id": vehicle_id,
                "reason": f"Vehicle '{vehicle_id}' has already arrived or has an invalid route.",
            }

        current_road_id = veh.current_road_id
        if current_road_id not in self.roads:
            return {
                "success": False,
                "rerouted": False,
                "vehicle_id": vehicle_id,
                "reason": f"Vehicle's current road '{current_road_id}' not found in network topology.",
            }
        current_road = self.roads[current_road_id]

        # 3. Non-empty subpath check
        if not new_subpath or not isinstance(new_subpath, (list, tuple)):
            return {
                "success": False,
                "rerouted": False,
                "vehicle_id": vehicle_id,
                "reason": "New subpath must be a non-empty sequence of road IDs.",
            }

        # 4. Valid network roads check
        for r_id in new_subpath:
            if not isinstance(r_id, str) or r_id not in self.roads:
                return {
                    "success": False,
                    "rerouted": False,
                    "vehicle_id": vehicle_id,
                    "reason": f"Road ID '{r_id}' not found in network topology.",
                }

        # 5. Candidate normalization
        if new_subpath[0] == current_road_id:
            future_roads = list(new_subpath[1:])
        else:
            future_roads = list(new_subpath)

        # 6. Downstream existence check
        if not future_roads:
            if veh.route_index == len(veh.route) - 1:
                return {
                    "success": True,
                    "rerouted": False,
                    "vehicle_id": vehicle_id,
                    "reason": "Vehicle is already on its final destination road; no downstream roads to alter.",
                    "current_road_id": current_road_id,
                    "route_index": veh.route_index,
                    "new_route": list(veh.route),
                }
            return {
                "success": False,
                "rerouted": False,
                "vehicle_id": vehicle_id,
                "reason": "New subpath provides no remaining downstream roads.",
            }

        # Check for no-op (identical to current downstream path)
        current_remaining = veh.route[veh.route_index + 1:]
        if future_roads == current_remaining:
            return {
                "success": True,
                "rerouted": False,
                "vehicle_id": vehicle_id,
                "reason": "Candidate route is identical to existing remaining route (no-op).",
                "current_road_id": current_road_id,
                "route_index": veh.route_index,
                "new_route": list(veh.route),
            }

        # 7. Directed adjacency: current road -> first future road
        first_future_road = self.roads[future_roads[0]]
        if current_road.to_node != first_future_road.from_node:
            return {
                "success": False,
                "rerouted": False,
                "vehicle_id": vehicle_id,
                "reason": (
                    f"Route discontinuity: current road '{current_road_id}' (to_node: '{current_road.to_node}') "
                    f"does not connect to first candidate road '{first_future_road.id}' (from_node: '{first_future_road.from_node}')."
                ),
            }

        # 8. Unintended U-turn check on initial divergence
        if (
            current_road.from_node != current_road.to_node
            and first_future_road.to_node == current_road.from_node
        ):
            return {
                "success": False,
                "rerouted": False,
                "vehicle_id": vehicle_id,
                "reason": (
                    f"Unintended U-turn detected: candidate road '{first_future_road.id}' "
                    f"reverses directly back to node '{current_road.from_node}'."
                ),
            }

        # Sequential connectivity & U-turn check along future_roads
        for k in range(len(future_roads) - 1):
            r_k = self.roads[future_roads[k]]
            r_next = self.roads[future_roads[k + 1]]
            if r_k.to_node != r_next.from_node:
                return {
                    "success": False,
                    "rerouted": False,
                    "vehicle_id": vehicle_id,
                    "reason": (
                        f"Route discontinuity: road '{r_k.id}' (to_node: '{r_k.to_node}') "
                        f"does not connect to road '{r_next.id}' (from_node: '{r_next.from_node}')."
                    ),
                }
            if r_k.from_node != r_k.to_node and r_next.to_node == r_k.from_node:
                return {
                    "success": False,
                    "rerouted": False,
                    "vehicle_id": vehicle_id,
                    "reason": (
                        f"Unintended U-turn detected: road '{r_next.id}' "
                        f"reverses directly back to node '{r_k.from_node}'."
                    ),
                }

        if not self.route_fits_vehicle(new_subpath, veh.length):
            return {"success": False, "rerouted": False, "vehicle_id": vehicle_id,
                    "reason": "The vehicle cannot clear a short junction link on the alternate route."}

        # 9. Destination preservation
        orig_dest_road_id = veh.route[-1]
        orig_dest_road = self.roads[orig_dest_road_id]
        orig_dest_node = orig_dest_road.to_node

        new_dest_road_id = future_roads[-1]
        new_dest_road = self.roads[new_dest_road_id]
        new_dest_node = new_dest_road.to_node

        if new_dest_road_id != orig_dest_road_id and new_dest_node != orig_dest_node:
            return {
                "success": False,
                "rerouted": False,
                "vehicle_id": vehicle_id,
                "reason": (
                    f"Destination mismatch: original destination road '{orig_dest_road_id}' "
                    f"(terminating at node '{orig_dest_node}') does not match new destination road "
                    f"'{new_dest_road_id}' (terminating at node '{new_dest_node}')."
                ),
            }

        # Atomic state update
        old_route = list(veh.route)
        past_and_curr = veh.route[:veh.route_index + 1]
        new_full_route = past_and_curr + future_roads

        veh.route = new_full_route
        veh.reroute_count += 1
        veh.last_reroute_time = self.sim_time
        veh.route_history.append(list(new_full_route))

        return {
            "success": True,
            "rerouted": True,
            "vehicle_id": vehicle_id,
            "reason": "Route successfully updated.",
            "old_route": old_route,
            "new_route": new_full_route,
            "current_road_id": current_road_id,
            "route_index": veh.route_index,
            "remaining_roads_count": len(future_roads),
        }

    def reroute_vehicles(self, reroute_commands: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Applies dynamic route modifications to in-flight vehicles.

        Args:
            reroute_commands: List of command dictionaries, each containing:
                - vehicle_id (str): ID of the vehicle to reroute
                - new_route or new_subpath (List[str]): Sequence of road IDs representing the new remaining path

        Returns:
            Dictionary summarizing results per command and aggregate counts.
        """
        results = []
        for cmd in reroute_commands:
            v_id = str(cmd.get("vehicle_id", ""))
            subpath = cmd.get("new_route") if "new_route" in cmd else cmd.get("new_subpath", [])
            res = self.reroute_vehicle(v_id, subpath)
            results.append(res)

        return {
            "total_commands": len(reroute_commands),
            "successful_count": sum(1 for r in results if r["success"]),
            "failed_count": sum(1 for r in results if not r["success"]),
            "rerouted_count": sum(1 for r in results if r.get("rerouted", False)),
            "results": results,
        }

    def close(self) -> None:
        self.vehicles.clear()
