"""Core domain models and telemetry types for UrbanFlow AI."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class Vec3:
    x: float
    y: float
    z: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {"x": round(self.x, 3), "y": round(self.y, 3), "z": round(self.z, 3)}


@dataclass
class GeoPoint:
    lat: float
    lon: float
    alt: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {"lat": self.lat, "lon": self.lon, "alt": self.alt}


@dataclass
class Lane:
    index: int
    length: float
    speed_limit: float = 13.89  # 50 km/h in m/s
    width: float = 3.5  # standard lane width in meters


def is_ns_approach(road) -> bool:
    aligned = getattr(road, "approach_ns", None)
    if aligned is not None:
        return aligned
    if len(road.points) < 2:
        return True
    start, end = road.points[0], road.points[-1]
    return abs(end.x - start.x) < abs(end.y - start.y)


@dataclass
class RoadSegment:
    id: str
    from_node: str
    to_node: str
    lanes: int = 2
    length: float = 150.0
    speed_limit: float = 13.89
    capacity: float = 1800.0  # veh/hr/lane
    points: List[Vec3] = field(default_factory=list)

    # Travel distance includes approach geometry removed from the lane ribbon.
    routing_length: Optional[float] = None
    # Physical lane count used by the renderer when traffic uses fewer lanes.
    visual_lanes: Optional[int] = None
    name: Optional[str] = None
    approach_ns: Optional[bool] = None
    signal_controlled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "from_node": self.from_node,
            "to_node": self.to_node,
            "lanes": self.visual_lanes or self.lanes,
            "length": self.length,
            "speed_limit": self.speed_limit,
            "capacity": self.capacity,
            "name": self.name,
            "points": [p.to_dict() for p in self.points],
        }


@dataclass
class Intersection:
    id: str
    position: Vec3
    connected_roads: List[str] = field(default_factory=list)
    has_signals: bool = True
    current_phase: int = 0
    phase_names: List[str] = field(default_factory=list)
    min_green: float = 10.0
    max_green: float = 60.0
    yellow_duration: float = 3.0
    all_red_duration: float = 1.5
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "position": self.position.to_dict(),
            "name": self.name,
            "connected_roads": self.connected_roads,
            "has_signals": self.has_signals,
            "current_phase": self.current_phase,
            "phase_names": self.phase_names,
            "min_green": self.min_green,
            "max_green": self.max_green,
            "yellow_duration": self.yellow_duration,
            "all_red_duration": self.all_red_duration,
        }


@dataclass
class VehicleState:
    id: str
    type: str  # "sedan", "van", "bus", "emergency"
    x: float
    y: float
    z: float
    speed: float  # m/s
    heading: float  # degrees (0 = North/Y+, 90 = East/X+)
    road_id: str
    lane_index: int
    accel: float = 0.0
    wait_time: float = 0.0
    color: str = "#2563eb"  # hex color
    route: List[str] = field(default_factory=list)
    route_index: int = 0
    reroute_count: int = 0
    original_route: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "z": round(self.z, 2),
            "speed": round(self.speed, 2),
            "heading": round(self.heading, 1),
            "road_id": self.road_id,
            "lane_index": self.lane_index,
            "accel": round(self.accel, 2),
            "wait_time": round(self.wait_time, 1),
            "color": self.color,
            "route": list(self.route),
            "route_index": self.route_index,
            "reroute_count": self.reroute_count,
            "original_route": list(self.original_route),
        }


@dataclass
class SignalState:
    intersection_id: str
    phase_index: int
    phase_name: str
    time_in_phase: float
    duration: float
    state_string: str  # "GGgrrr", etc.
    stage: str = "GREEN"  # "GREEN", "YELLOW", "ALL_RED"
    is_yellow: bool = False
    is_all_red: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intersection_id": self.intersection_id,
            "phase_index": self.phase_index,
            "phase_name": self.phase_name,
            "time_in_phase": round(self.time_in_phase, 1),
            "duration": round(self.duration, 1),
            "state_string": self.state_string,
            "stage": self.stage,
            "is_yellow": self.is_yellow,
            "is_all_red": self.is_all_red,
        }


@dataclass
class NetworkMetrics:
    active_vehicles: int
    total_arrived: int
    throughput_per_hour: float
    avg_speed_kmh: float
    avg_delay_sec: float
    congestion_index: float  # 0.0 (free flow) to 1.0 (gridlock)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_vehicles": self.active_vehicles,
            "total_arrived": self.total_arrived,
            "throughput_per_hour": round(self.throughput_per_hour, 1),
            "avg_speed_kmh": round(self.avg_speed_kmh, 1),
            "avg_delay_sec": round(self.avg_delay_sec, 1),
            "congestion_index": round(self.congestion_index, 3),
        }


@dataclass
class SimulationSnapshot:
    timestamp: float
    step: int
    vehicles: List[VehicleState]
    signals: List[SignalState]
    metrics: NetworkMetrics

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": round(self.timestamp, 3),
            "step": self.step,
            "vehicles": [v.to_dict() for v in self.vehicles],
            "signals": [s.to_dict() for s in self.signals],
            "metrics": self.metrics.to_dict(),
        }
