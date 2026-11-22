"""Abstract base class for all UrbanFlow simulation engine adapters."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from urbanflow.core.types import SimulationSnapshot, RoadSegment, Intersection


class SimulatorAdapter(ABC):
    """Abstract contract for interchangeable microscopic/kinematic traffic simulators."""

    @abstractmethod
    def initialize(self, network_config: Dict[str, Any]) -> None:
        """Initializes the simulation with a road network topology and scenario parameters."""
        pass

    @abstractmethod
    def step(self, dt: float) -> SimulationSnapshot:
        """Advances the simulation by dt seconds and returns the new world state snapshot."""
        pass

    @abstractmethod
    def reset(self) -> SimulationSnapshot:
        """Resets the simulation to the initial scenario state."""
        pass

    @abstractmethod
    def get_network_topology(self) -> Dict[str, Any]:
        """Returns geographic definitions of roads, lanes, and intersections."""
        pass

    @abstractmethod
    def apply_signal_phases(self, phase_commands: Dict[str, int]) -> None:
        """Overrides or sets target phase index for specified intersections."""
        pass

    @abstractmethod
    def inject_incident(self, road_id: str, severity: float, duration_sec: float) -> bool:
        """Injects a dynamic incident or lane closure on the specified road."""
        pass

    @abstractmethod
    def trigger_congestion_surge(self, origin_road_id: str, vehicle_count: int) -> bool:
        """Injects an impromptu influx of vehicles to test dynamic capacity and routing."""
        pass

    def reroute_vehicles(self, reroute_commands: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Applies dynamic route modifications to in-flight vehicles.

        Args:
            reroute_commands: List of command dictionaries, each containing:
                - vehicle_id (str): ID of the vehicle to reroute
                - new_route or new_subpath (List[str]): Sequence of road IDs representing the new remaining path

        Returns:
            Dictionary summarizing results:
                - total_commands (int): Number of commands received
                - successful_count (int): Number of successfully executed reroutes
                - failed_count (int): Number of rejected reroutes
                - results (List[Dict[str, Any]]): Detailed outcome per command
        """
        return {
            "total_commands": len(reroute_commands),
            "successful_count": 0,
            "failed_count": len(reroute_commands),
            "results": [
                {
                    "vehicle_id": cmd.get("vehicle_id", ""),
                    "success": False,
                    "reason": "Dynamic rerouting is not implemented in this simulator adapter.",
                }
                for cmd in reroute_commands
            ],
        }

    @abstractmethod
    def close(self) -> None:
        """Performs cleanup of native handles, sockets, and subprocesses."""
        pass
