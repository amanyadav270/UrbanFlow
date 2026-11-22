"""Downstream Congestion Prediction Engine for UrbanFlow AI."""
from __future__ import annotations

from typing import Dict, Any, List
from urbanflow.core.types import SimulationSnapshot


class CongestionPredictor:
    """Predicts downstream corridor congestion bottlenecks 5-15 minutes into the future."""

    def __init__(self, horizon_minutes: int = 10):
        self.horizon_minutes = horizon_minutes
        self.history: List[Dict[str, Any]] = []

    def update(self, snapshot: SimulationSnapshot) -> Dict[str, Any]:
        """Analyzes spatial density and deceleration trends to forecast bottlenecks."""
        predictions: Dict[str, float] = {}

        # Aggregate vehicles per road
        road_counts: Dict[str, int] = {}
        for v in snapshot.vehicles:
            road_counts[v.road_id] = road_counts.get(v.road_id, 0) + 1

        # High density detection
        for r_id, count in road_counts.items():
            if count > 12:
                # Projected bottleneck probability
                predictions[r_id] = min(1.0, 0.4 + (count - 12) * 0.08)
            else:
                predictions[r_id] = max(0.0, count * 0.03)

        return {
            "horizon_minutes": self.horizon_minutes,
            "predicted_bottlenecks": predictions,
            "network_risk_level": "ELEVATED" if snapshot.metrics.congestion_index > 0.4 else "NORMAL",
        }
