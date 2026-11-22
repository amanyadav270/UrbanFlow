"""Phase 1: Real-Time Traffic State Estimation Engine for UrbanFlow AI.

Computes microscopic-to-macroscopic aggregated traffic states for every road segment:
- Vehicle count & density (veh/km/lane)
- Time mean speed and space mean speed (km/h)
- Observed flow rate (veh/hr)
- Physical queue length (meters) and occupancy ratio (0.0 - 1.0)
- Estimated operational capacity (veh/hr) and travel time (sec)
- Downstream intersection signal constraint state
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple

from urbanflow.intelligence.capacity_estimator import RoadCapacityEstimator


@dataclass
class SegmentTrafficState:
    """Standardized traffic observation snapshot for a single directed road segment."""
    road_id: str
    timestamp: float                    # Simulation time in seconds
    vehicle_count: int                  # Instantaneous number of vehicles on segment
    density_per_km_lane: float          # Vehicles / kilometer / lane
    average_speed_kmh: float            # Space-mean speed in km/h
    flow_veh_per_hour: float            # Observed throughput flow rate (veh/hr)
    queue_length_meters: float          # Longest continuous queue of stopped/crawling vehicles (m)
    queue_vehicle_count: int            # Number of vehicles queued (speed < 1.0 m/s)
    occupancy_ratio: float              # Physical road surface occupancy (0.0 to 1.0)
    travel_time_sec: float              # Estimated segment traversal time (sec)
    theoretical_capacity: float         # HCM theoretical capacity (veh/hr)
    operational_capacity: float         # Dynamic operational capacity (veh/hr)
    capacity_utilization: float         # Flow to Capacity ratio (V/C)
    signal_state: str                   # Downstream signal state: 'G', 'y', 'r', or 'NONE'
    incident_active: bool               # Whether an active incident is reducing capacity
    data_source: str = "simulation_synthetic"  # Explicitly distinguished from real sensor data

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TrafficStateEstimator:
    """Collects vehicle kinematics and computes macroscopic segment traffic states."""

    def __init__(self, sample_interval_sec: float = 1.0):
        self.sample_interval = sample_interval_sec
        # Historical vehicle exit logs per road for rolling flow rate estimation: road_id -> list of timestamps
        self._exit_events: Dict[str, List[float]] = {}
        # Rolling flow window (60 seconds)
        self.flow_window_sec = 60.0
        self._capacity_estimator = RoadCapacityEstimator()

    def record_vehicle_exit(self, road_id: str, timestamp: float):
        """Records a vehicle exiting a road segment to measure empirical flow."""
        if road_id not in self._exit_events:
            self._exit_events[road_id] = []
        self._exit_events[road_id].append(timestamp)

    def estimate_segment_state(
        self,
        road_id: str,
        timestamp: float,
        road_length: float,
        lanes: int,
        speed_limit_mps: float,
        vehicles_on_road: List[Any],  # List of SimulatedVehicle
        downstream_signal_state: str = "NONE",
        incident_severity: float = 0.0,
    ) -> SegmentTrafficState:
        """Computes comprehensive macroscopic traffic state for a single road segment."""
        lanes = max(1, lanes)
        road_len_km = max(0.01, road_length / 1000.0)
        v_count = len(vehicles_on_road)

        # 1. Density (vehicles per km per lane)
        density = v_count / (road_len_km * lanes)

        # 2. Average Speed (Harmonic space-mean speed preferred for traffic theory)
        if v_count > 0:
            speeds_mps = [max(0.1, v.speed) for v in vehicles_on_road]
            # Harmonic mean: N / sum(1 / v)
            harmonic_speed_mps = len(speeds_mps) / sum(1.0 / s for s in speeds_mps)
            avg_speed_kmh = harmonic_speed_mps * 3.6
        else:
            avg_speed_kmh = speed_limit_mps * 3.6

        # 3. Flow (Vehicles per hour)
        # Prune old exit events
        cutoff = timestamp - self.flow_window_sec
        exits = self._exit_events.get(road_id, [])
        exits = [t for t in exits if t >= cutoff]
        self._exit_events[road_id] = exits

        if len(exits) > 0 and self.flow_window_sec > 0:
            flow_veh_hr = (len(exits) / self.flow_window_sec) * 3600.0
        else:
            # Fundamental relationship fallback: q = k * v (in compatible units)
            # density (veh/km) * speed (km/h) = flow (veh/h)
            flow_veh_hr = (density * lanes) * avg_speed_kmh

        # 4. Queue Length (vehicles with speed < 1.0 m/s = ~2.2 mph)
        queued_vehs = [v for v in vehicles_on_road if v.speed < 1.0]
        queue_count = len(queued_vehs)
        if queue_count > 0:
            # Physical queue length along road
            avg_veh_length = sum(v.length for v in queued_vehs) / queue_count
            queue_length_m = min(road_length, queue_count * (avg_veh_length + 2.5) / lanes)
        else:
            queue_length_m = 0.0

        # 5. Occupancy Ratio (physical road surface space taken by vehicles)
        total_veh_length = sum(v.length + 2.0 for v in vehicles_on_road)
        total_lane_capacity_length = road_length * lanes
        occupancy = min(1.0, max(0.0, total_veh_length / max(1.0, total_lane_capacity_length)))

        # 6. Travel Time (seconds)
        eff_speed_mps = max(0.5, avg_speed_kmh / 3.6)
        travel_time = road_length / eff_speed_mps

        # 7. Theoretical vs Operational Capacity via RoadCapacityEstimator
        cap_report = self._capacity_estimator.estimate_capacity(
            road_id=road_id,
            road_length_meters=road_length,
            lanes=lanes,
            speed_limit_kmh=speed_limit_mps * 3.6,
            observed_flow_veh_hr=flow_veh_hr,
            density_veh_km_lane=density,
            queue_length_meters=queue_length_m,
            downstream_signal_state=downstream_signal_state,
            incident_severity=incident_severity,
        )
        theoretical_cap = cap_report.theoretical_capacity_veh_hr
        operational_cap = cap_report.operational_capacity_veh_hr
        vc_ratio = cap_report.vc_ratio

        return SegmentTrafficState(
            road_id=road_id,
            timestamp=round(timestamp, 2),
            vehicle_count=v_count,
            density_per_km_lane=round(density, 2),
            average_speed_kmh=round(avg_speed_kmh, 2),
            flow_veh_per_hour=round(flow_veh_hr, 1),
            queue_length_meters=round(queue_length_m, 2),
            queue_vehicle_count=queue_count,
            occupancy_ratio=round(occupancy, 3),
            travel_time_sec=round(travel_time, 2),
            theoretical_capacity=round(theoretical_cap, 1),
            operational_capacity=round(operational_cap, 1),
            capacity_utilization=round(vc_ratio, 3),
            signal_state=downstream_signal_state,
            incident_active=(incident_severity > 0.0),
            data_source="simulation_synthetic",
        )
