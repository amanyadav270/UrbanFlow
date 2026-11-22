"""Phase 4: Road Capacity Estimation Engine for UrbanFlow AI.

Methodology Note:
This module implements an HCM-inspired operational V/C heuristic.
In formal Highway Capacity Manual (HCM 6th Edition) methodology:
- Signalized intersection Level of Service (LOS) is evaluated by Control Delay (sec/veh) [HCM Chapter 19].
- Urban street / arterial LOS is evaluated by Travel Speed as a % of Base Free-Flow Speed [HCM Chapter 16].
The Volume-to-Capacity (V/C) classification below is an engineering operational proxy
designed for real-time digital twin monitoring and bottleneck detection, not a formally
validated HCM control-delay audit.

Volume-to-Capacity (V/C) Level of Service (LOS) Bands:
- LOS A: [0.00, 0.35)  — Free Flow / Uncongested
- LOS B: [0.35, 0.55)  — Reasonably Free Flow / Stable
- LOS C: [0.55, 0.75)  — Stable Flow / Moderate Density
- LOS D: [0.75, 0.90)  — Approaching Capacity / High Delay
- LOS E: [0.90, 1.00]  — At Operational Capacity / Critical
- LOS F: (1.00, inf)   — Oversaturated Breakdown / Forced Flow
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple


@dataclass
class CapacityReport:
    """Detailed capacity breakdown for a specific road segment."""
    road_id: str
    theoretical_capacity_veh_hr: float
    operational_capacity_veh_hr: float
    observed_flow_veh_hr: float
    vc_ratio: float
    level_of_service: str             # 'A', 'B', 'C', 'D', 'E', 'F'
    bottleneck_risk: str              # 'LOW', 'MODERATE', 'HIGH', 'CRITICAL'
    limiting_factor: str              # 'INCIDENT', 'QUEUE_SPILLBACK', 'SIGNAL_RED', 'NORMAL'
    green_split: float                # Green ratio used (g/C)
    green_split_source: str           # 'actual_timing', 'assumed_default_0.48', or 'unsignalized'
    methodology: str = "HCM-inspired operational V/C heuristic"
    data_source: str = "simulation_synthetic"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RoadCapacityEstimator:
    """Estimates theoretical saturation capacity and real-time operational capacity."""

    BASE_CAPACITY_PER_LANE = 1850.0  # passenger cars / hour / lane (urban arterial baseline)
    DEFAULT_SIGNAL_GREEN_SPLIT = 0.48  # standard assumed 2-phase urban green split

    # Explicit Level of Service V/C ratio boundaries: (lower_inclusive, upper_exclusive, los, risk, color)
    LOS_BANDS: Tuple[Tuple[float, float, str, str, str], ...] = (
        (0.00, 0.35, "A", "LOW", "#10b981"),
        (0.35, 0.55, "B", "LOW", "#34d399"),
        (0.55, 0.75, "C", "MODERATE", "#f59e0b"),
        (0.75, 0.90, "D", "HIGH", "#f97316"),
        (0.90, 1.00, "E", "CRITICAL", "#ef4444"),
        (1.00, float("inf"), "F", "CRITICAL", "#991b1b"),
    )

    @classmethod
    def classify_los(cls, vc_ratio: float) -> Tuple[str, str, str]:
        """
        Classifies V/C ratio into continuous LOS band (A through F).
        Guarantees every non-negative float is classified.
        Returns: (los_letter, risk_level, hex_color)
        """
        vc = max(0.0, float(vc_ratio))
        for lower, upper, los, risk, color in cls.LOS_BANDS:
            if lower <= vc < upper or (upper == 1.00 and vc <= 1.00):
                return los, risk, color
        # Upper bound fallback for extreme oversaturation
        return "F", "CRITICAL", "#991b1b"

    def estimate_capacity(
        self,
        road_id: str,
        road_length_meters: float,
        lanes: int,
        speed_limit_kmh: float,
        observed_flow_veh_hr: float,
        density_veh_km_lane: float,
        queue_length_meters: float,
        downstream_signal_state: str = "NONE",
        incident_severity: float = 0.0,
        actual_green_ratio: Optional[float] = None,
    ) -> CapacityReport:
        """
        Computes theoretical baseline capacity and current operational capacity.

        Resolves previous signal double-counting:
        - Theoretical capacity accounts for cycle-averaged green split (g/C).
        - Operational capacity degrades theoretical capacity based on physical incidents,
          queue spillback, and transient phase throttling without compounding secondary
          multiplicative cycle penalties.
        """
        lanes = max(1, lanes)
        road_length = max(10.0, road_length_meters)

        # 1. Theoretical Approach Capacity (Cycle-Averaged HCM Baseline)
        has_signal = downstream_signal_state in ("G", "y", "r")
        if actual_green_ratio is not None and has_signal:
            signal_green_split = max(0.05, min(1.0, float(actual_green_ratio)))
            green_source = "actual_timing"
        elif has_signal:
            signal_green_split = self.DEFAULT_SIGNAL_GREEN_SPLIT
            green_source = "assumed_default_0.48"
        else:
            signal_green_split = 1.0
            green_source = "unsignalized"

        # Speed adjustment factor (speeds below 50 km/h reduce capacity due to headway friction)
        speed_factor = min(1.0, max(0.75, speed_limit_kmh / 50.0))
        theoretical_cap = lanes * self.BASE_CAPACITY_PER_LANE * signal_green_split * speed_factor

        # 2. Operational Capacity Degradation Factors
        # A. Incident penalty (physical lane blockage)
        incident_penalty = max(0.0, min(0.95, incident_severity))

        # B. Queue spillback penalty (downstream blockage consuming approach storage)
        queue_fraction = min(1.0, max(0.0, queue_length_meters / road_length))
        queue_penalty = queue_fraction * 0.70

        # C. Transient signal phase bottleneck
        # Note: Theoretical capacity already factored the cycle green split.
        # This transient modifier reflects instantaneous clearance/hold restriction
        # without duplicating the full cycle-length reduction.
        signal_penalty = 0.0
        if downstream_signal_state == "r":
            signal_penalty = 0.25  # temporary phase hold resistance
        elif downstream_signal_state == "y":
            signal_penalty = 0.10  # phase clearance hesitation

        # Combined degradation (capped at 90% loss to preserve minimal physical throughput)
        total_loss = min(0.90, incident_penalty + queue_penalty + signal_penalty)
        operational_cap = max(150.0, theoretical_cap * (1.0 - total_loss))

        # 3. Volume-to-Capacity (V/C) Ratio
        vc_ratio = observed_flow_veh_hr / operational_cap

        # 4. Level of Service (LOS) Determination across all A–F bands
        los, risk, _ = self.classify_los(vc_ratio)

        # 5. Determine Primary Limiting Factor
        if incident_penalty > 0.30:
            limiting = "INCIDENT"
        elif queue_fraction > 0.50:
            limiting = "QUEUE_SPILLBACK"
        elif downstream_signal_state == "r" and vc_ratio > 0.70:
            limiting = "SIGNAL_RED"
        else:
            limiting = "NORMAL"

        return CapacityReport(
            road_id=road_id,
            theoretical_capacity_veh_hr=round(theoretical_cap, 1),
            operational_capacity_veh_hr=round(operational_cap, 1),
            observed_flow_veh_hr=round(observed_flow_veh_hr, 1),
            vc_ratio=round(vc_ratio, 3),
            level_of_service=los,
            bottleneck_risk=risk,
            limiting_factor=limiting,
            green_split=round(signal_green_split, 2),
            green_split_source=green_source,
            methodology="HCM-inspired operational V/C heuristic",
            data_source="simulation_synthetic",
        )
