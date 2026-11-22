"""Traffic Signal Controllers for UrbanFlow AI.

Phase 6B: Classical Signal-Control Baselines.
Phase 6C: Cooperative Predictive Traffic Signal Control.

Provides:
1. SignalControllerBase:
   - Common abstract interface for traffic signal controllers.
   - Separation of control policy from simulator safety enforcement (IntersectionSignalFSM).
2. FixedTimeController:
   - Deterministic per-intersection cycle schedules with configurable green splits and offsets.
   - Respects minimum green, maximum green, and clearance timing bounds.
   - Reproducible baseline for fixed-time coordinated arterial control.
3. ActuatedController:
   - Standard NEMA-inspired actuated local controller using approach vehicle demand and queues.
   - Supports passage time / unit extension, gap-out, minimum green, and maximum green force-off.
   - Avoids extending green when the active approach has no demand and conflicting call is present.
4. MaxPressureController:
   - Decentralized network queue-differential pressure controller (Tassiulas-Ephremides / Varaiya).
   - Upstream minus downstream pressure formulation without using ML classifiers.
   - Downstream link storage capacity safeguard: penalizes/disables movements whose downstream
     links exceed physical storage capacity, preventing spillback gridlock.
5. CooperativePredictiveSignalController:
   - Network-aware cooperative controller coordinating across neighboring signalized intersections.
   - Upstream platoon anticipation and downstream green wave coordination bonus / red dampening.
   - Strict 75% downstream physical storage capacity safeguard (-100.0 veto).
   - Advisory 10-minute network spillback classifier with cryptographic SHA-256 verification,
     bounded influence (<= 8.0), and graceful non-predictive fallback.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import logging
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple, Union

from urbanflow.core.types import is_ns_approach

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "SignalControllerBase",
    "FixedTimeController",
    "ActuatedController",
    "MaxPressureController",
    "CooperativePredictiveSignalController",
]


class SignalControllerBase(ABC):
    """Abstract base contract for traffic signal controllers in UrbanFlow.

    Controllers evaluate simulation and telemetry states to propose target phase requests
    for signalized intersections. Simulator-level safety constraints (clearance intervals,
    conflicting green prevention, minimum/maximum green bounds) are strictly enforced by
    the underlying IntersectionSignalFSM.
    """

    @abstractmethod
    def compute_phases(
        self,
        current_time: float,
        simulator: Optional[Any] = None,
        telemetry: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, int]:
        """Computes and returns a mapping of intersection_id to requested target phase.

        Args:
            current_time: Current simulation timestamp in seconds.
            simulator: Optional reference to the traffic simulator (e.g. KinematicSimulator).
            telemetry: Optional real-time segment telemetry dictionary.

        Returns:
            Dict[str, int] mapping intersection_id -> target phase index (0 for NS, 1 for EW).
        """
        pass

    def reset(self) -> None:
        """Resets controller internal states (timers, memory, call history)."""
        pass


class FixedTimeController(SignalControllerBase):
    """Deterministic fixed-time traffic signal controller with configurable splits and offsets.

    Operates on a repetitive cycle:
    Cycle C = Green_NS + Yellow + AllRed + Green_EW + Yellow + AllRed.
    Offsets shift the cycle reference time: t_rel = (current_time - offset) mod C.
    """

    def __init__(
        self,
        default_green_ns: float = 25.0,
        default_green_ew: float = 20.0,
        yellow_duration: float = 3.0,
        all_red_duration: float = 1.5,
        default_offset: float = 0.0,
        intersection_plans: Optional[Dict[str, Dict[str, float]]] = None,
        controlled_intersections: Optional[List[str]] = None,
    ):
        self.default_green_ns = float(default_green_ns)
        self.default_green_ew = float(default_green_ew)
        self.yellow_duration = float(yellow_duration)
        self.all_red_duration = float(all_red_duration)
        self.default_offset = float(default_offset)
        self.intersection_plans = intersection_plans or {}
        self.controlled_intersections = controlled_intersections

    def compute_phases(
        self,
        current_time: float,
        simulator: Optional[Any] = None,
        telemetry: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, int]:
        """Computes scheduled target phases based on deterministic cyclic time-of-day progression."""
        target_phases: Dict[str, int] = {}

        # Determine target intersections to control
        inter_ids = self._get_controlled_intersection_ids(simulator)
        if not inter_ids:
            return target_phases

        clearance = self.yellow_duration + self.all_red_duration

        for inter_id in inter_ids:
            plan = self.intersection_plans.get(inter_id, {})
            green_ns = float(plan.get("green_ns", self.default_green_ns))
            green_ew = float(plan.get("green_ew", self.default_green_ew))
            offset = float(plan.get("offset", self.default_offset))

            # Enforce safety bounds (minimum green >= 10.0s, maximum green <= 60.0s)
            green_ns = max(10.0, min(60.0, green_ns))
            green_ew = max(10.0, min(60.0, green_ew))

            cycle_length = green_ns + clearance + green_ew + clearance
            if cycle_length <= 0.0:
                target_phases[inter_id] = 0
                continue

            # Normalized relative time in cycle
            t_rel = (current_time - offset) % cycle_length
            if t_rel < 0.0:
                t_rel += cycle_length

            # Phase scheduling logic:
            # [0, green_ns): Serving Phase 0 (NS Green)
            # [green_ns, green_ns + clearance): Clearance interval to Phase 1 (request Phase 1)
            # [green_ns + clearance, green_ns + clearance + green_ew): Serving Phase 1 (EW Green)
            # [green_ns + clearance + green_ew, cycle_length): Clearance interval to Phase 0 (request Phase 0)
            if t_rel < green_ns:
                target_phases[inter_id] = 0
            elif t_rel < green_ns + clearance + green_ew:
                target_phases[inter_id] = 1
            else:
                target_phases[inter_id] = 0

        return target_phases

    def _get_controlled_intersection_ids(self, simulator: Optional[Any]) -> List[str]:
        if self.controlled_intersections is not None:
            return list(self.controlled_intersections)
        if simulator is not None and hasattr(simulator, "signal_controllers"):
            return list(simulator.signal_controllers.keys())
        if simulator is not None and hasattr(simulator, "intersections"):
            return [i_id for i_id, inter in simulator.intersections.items() if inter.has_signals]
        return list(self.intersection_plans.keys())


class ActuatedController(SignalControllerBase):
    """NEMA-inspired vehicle-actuated traffic signal controller.

    Monitors approach vehicle presence and queues to execute demand-responsive phase allocation:
    - Minimum green enforcement (green held for min_green seconds).
    - Unit extension / passage time: approaching active vehicles extend green.
    - Gap-out: green terminates early if active approach has no demand and conflicting call is pending.
    - Maximum green force-off: green terminates at max_green when conflicting demand is sustained.
    - Rest in green: when no conflicting demand exists, rests in active phase without switching.
    """

    def __init__(
        self,
        min_green: float = 10.0,
        max_green: float = 60.0,
        passage_time: float = 3.0,
        detection_distance_m: float = 50.0,
        controlled_intersections: Optional[List[str]] = None,
    ):
        self.min_green = float(min_green)
        self.max_green = float(max_green)
        self.passage_time = float(passage_time)
        self.detection_distance_m = float(detection_distance_m)
        self.controlled_intersections = controlled_intersections

        # Track internal state per intersection: target phase request
        self._last_requested_phase: Dict[str, int] = {}

    def reset(self) -> None:
        self._last_requested_phase.clear()

    def compute_phases(
        self,
        current_time: float,
        simulator: Optional[Any] = None,
        telemetry: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, int]:
        """Evaluates approach demand and returns actuated phase commands."""
        target_phases: Dict[str, int] = {}
        if simulator is None and telemetry is None:
            return target_phases

        inter_ids = self._get_controlled_intersection_ids(simulator)
        if not inter_ids:
            return target_phases

        for inter_id in inter_ids:
            ctrl = getattr(simulator, "signal_controllers", {}).get(inter_id) if simulator else None
            active_phase = ctrl.active_green_phase if ctrl else self._last_requested_phase.get(inter_id, 0)
            if ctrl and ctrl.stage != "GREEN" and ctrl.target_green_phase in (0, 1):
                target_phases[inter_id] = ctrl.target_green_phase
                continue
            time_in_green = ctrl.time_in_green if ctrl else 0.0

            # Measure vehicle demand on NS and EW approaches
            ns_demand, ns_queued = self._measure_approach_demand(inter_id, is_ns=True, simulator=simulator, telemetry=telemetry)
            ew_demand, ew_queued = self._measure_approach_demand(inter_id, is_ns=False, simulator=simulator, telemetry=telemetry)

            active_demand = ns_demand if active_phase == 0 else ew_demand
            conflicting_demand = ew_demand if active_phase == 0 else ns_demand
            conflicting_phase = 1 if active_phase == 0 else 0

            # 1. Minimum Green Guard: Must not terminate green before min_green
            if time_in_green < self.min_green:
                target_phases[inter_id] = active_phase
                self._last_requested_phase[inter_id] = active_phase
                continue

            # 2. Maximum Green Guard: Force-off if conflicting call and time_in_green >= max_green
            if time_in_green >= self.max_green and conflicting_demand > 0:
                target_phases[inter_id] = conflicting_phase
                self._last_requested_phase[inter_id] = conflicting_phase
                continue

            # 3. Demand-driven Gap-Out & Extension (min_green <= time_in_green < max_green)
            if conflicting_demand > 0:
                if active_demand == 0:
                    # Active approach has gapped out / empty; serve conflicting approach
                    target_phases[inter_id] = conflicting_phase
                else:
                    # Active approach has continuing traffic; extend green
                    target_phases[inter_id] = active_phase
            else:
                # No conflicting demand: dwell/rest in active green
                target_phases[inter_id] = active_phase

            self._last_requested_phase[inter_id] = target_phases[inter_id]

        return target_phases

    def _measure_approach_demand(
        self,
        inter_id: str,
        is_ns: bool,
        simulator: Optional[Any],
        telemetry: Optional[Dict[str, Any]],
    ) -> Tuple[int, int]:
        """Measures approaching vehicle count and queued vehicle count for an intersection approach."""
        demand_count = 0
        queued_count = 0

        # Method A: Direct simulator inspection
        if simulator is not None and hasattr(simulator, "roads") and hasattr(simulator, "vehicles"):
            for road in simulator.roads.values():
                if road.to_node == inter_id and getattr(road, "signal_controlled", True):
                    road_is_ns = is_ns_approach(road)
                    if road_is_ns == is_ns:
                        for veh in simulator.vehicles.values():
                            if veh.current_road_id == road.id:
                                dist_to_stop = max(0.0, road.length - veh.distance_on_road)
                                if dist_to_stop <= self.detection_distance_m or road.length <= self.detection_distance_m:
                                    demand_count += 1
                                    if veh.speed < 1.0 or veh.wait_time > 0.0:
                                        queued_count += 1
            return demand_count, queued_count

        # Method B: Telemetry-based estimation
        if telemetry is not None:
            for road_id, data in telemetry.items():
                if isinstance(data, dict):
                    q_len = data.get("queue_length_meters", 0.0)
                    v_cnt = data.get("vehicle_count", int(q_len / 7.0) if q_len > 0 else 0)
                    if v_cnt > 0 or q_len > 0:
                        demand_count += max(1, v_cnt)
                        if q_len > 0:
                            queued_count += max(1, int(q_len / 7.0))

        return demand_count, queued_count

    def _get_controlled_intersection_ids(self, simulator: Optional[Any]) -> List[str]:
        if self.controlled_intersections is not None:
            return list(self.controlled_intersections)
        if simulator is not None and hasattr(simulator, "signal_controllers"):
            return list(simulator.signal_controllers.keys())
        if simulator is not None and hasattr(simulator, "intersections"):
            return [i_id for i_id, inter in simulator.intersections.items() if inter.has_signals]
        return []


class MaxPressureController(SignalControllerBase):
    """Decentralized Max-Pressure traffic signal controller with downstream storage safeguards.

    Calculates movement pressure differentials:
        Pressure(u -> v) = Queue_u - Queue_v
    Selects the phase maximizing aggregate movement pressure:
        Phase* = argmax_p sum_{(u,v) in M_p} max(0, Queue_u - Queue_v)

    Storage Capacity Safeguard:
    If a downstream link's physical occupancy exceeds `downstream_occupancy_cap` (default 75%),
    its movement pressure is penalized/suppressed, preventing discharge into blocked links.
    """

    def __init__(
        self,
        min_green: float = 10.0,
        max_green: float = 60.0,
        downstream_occupancy_cap: float = 0.75,
        pressure_threshold: float = 1.0,
        controlled_intersections: Optional[List[str]] = None,
    ):
        self.min_green = float(min_green)
        self.max_green = float(max_green)
        self.downstream_occupancy_cap = float(downstream_occupancy_cap)
        self.pressure_threshold = float(pressure_threshold)
        self.controlled_intersections = controlled_intersections

        self._last_requested_phase: Dict[str, int] = {}

    def reset(self) -> None:
        self._last_requested_phase.clear()

    def compute_phases(
        self,
        current_time: float,
        simulator: Optional[Any] = None,
        telemetry: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, int]:
        """Calculates queue differentials across approaches and selects maximum pressure phases."""
        target_phases: Dict[str, int] = {}
        if simulator is None and telemetry is None:
            return target_phases

        inter_ids = self._get_controlled_intersection_ids(simulator)
        if not inter_ids:
            return target_phases

        for inter_id in inter_ids:
            ctrl = getattr(simulator, "signal_controllers", {}).get(inter_id) if simulator else None
            active_phase = ctrl.active_green_phase if ctrl else self._last_requested_phase.get(inter_id, 0)
            time_in_green = ctrl.time_in_green if ctrl else 0.0

            # 1. Calculate pressure for Phase 0 (North-South) and Phase 1 (East-West)
            press_0 = self._calculate_phase_pressure(inter_id, phase_index=0, simulator=simulator, telemetry=telemetry)
            press_1 = self._calculate_phase_pressure(inter_id, phase_index=1, simulator=simulator, telemetry=telemetry)

            # 2. Minimum Green Guard
            if time_in_green < self.min_green:
                target_phases[inter_id] = active_phase
                self._last_requested_phase[inter_id] = active_phase
                continue

            # 3. Maximum Green Guard: Force-off if opposing phase has pressure and time_in_green >= max_green
            opposing_phase = 1 if active_phase == 0 else 0
            opposing_press = press_1 if active_phase == 0 else press_0
            if time_in_green >= self.max_green and opposing_press > 0.0:
                target_phases[inter_id] = opposing_phase
                self._last_requested_phase[inter_id] = opposing_phase
                continue

            # 4. Pressure-Differential Selection with Hysteresis Threshold
            if active_phase == 0:
                if press_1 > press_0 + self.pressure_threshold:
                    chosen_phase = 1
                else:
                    chosen_phase = 0
            else:
                if press_0 > press_1 + self.pressure_threshold:
                    chosen_phase = 0
                else:
                    chosen_phase = 1

            target_phases[inter_id] = chosen_phase
            self._last_requested_phase[inter_id] = chosen_phase

        return target_phases

    def _calculate_phase_pressure(
        self,
        inter_id: str,
        phase_index: int,
        simulator: Optional[Any],
        telemetry: Optional[Dict[str, Any]],
    ) -> float:
        """Calculates aggregate movement pressure for the given phase, penalizing blocked downstream links."""
        is_ns_phase = (phase_index == 0)
        total_pressure = 0.0

        if simulator is not None and hasattr(simulator, "roads"):
            # Discover incoming and outgoing roads
            in_roads = [
                r for r in simulator.roads.values()
                if r.to_node == inter_id and getattr(r, "signal_controlled", True) and self._is_road_ns(r) == is_ns_phase
            ]
            out_roads = [
                r for r in simulator.roads.values()
                if r.from_node == inter_id and self._is_road_ns(r) == is_ns_phase
            ]

            for in_r in in_roads:
                q_u = self._get_road_queue(in_r.id, simulator=simulator, telemetry=telemetry)
                if q_u <= 0.0:
                    continue

                if not out_roads:
                    # Boundary discharge: downstream queue is zero
                    total_pressure += q_u
                else:
                    for out_r in out_roads:
                        # Storage Safeguard Check on downstream link
                        is_blocked = self._is_downstream_blocked(out_r, simulator=simulator, telemetry=telemetry)
                        if is_blocked:
                            # Downstream link is congested/blocked: suppress/penalize movement pressure
                            movement_press = -50.0
                        else:
                            q_v = self._get_road_queue(out_r.id, simulator=simulator, telemetry=telemetry)
                            movement_press = max(0.0, q_u - q_v)

                        total_pressure += movement_press

            return max(0.0, total_pressure)

        # Telemetry-only fallback
        if telemetry is not None:
            # Aggregate queue lengths matching phase
            for road_id, data in telemetry.items():
                if isinstance(data, dict):
                    q = data.get("queue_length_meters", 0.0)
                    total_pressure += max(0.0, q)
            return total_pressure

        return 0.0

    def _is_downstream_blocked(
        self,
        road: Any,
        simulator: Optional[Any],
        telemetry: Optional[Dict[str, Any]],
    ) -> bool:
        """Determines if a downstream link has exceeded its physical storage capacity threshold."""
        # 1. Direct simulator inspection
        if simulator is not None and hasattr(simulator, "vehicles"):
            vehs_on_road = [v for v in simulator.vehicles.values() if v.current_road_id == road.id]
            max_storage = max(1.0, (road.length * road.lanes) / 7.5)
            occupancy = len(vehs_on_road) / max_storage
            if occupancy >= self.downstream_occupancy_cap:
                return True

            # Also check physical queue length
            queued_len = sum(v.length + 2.5 for v in vehs_on_road if v.speed < 1.0)
            if queued_len >= self.downstream_occupancy_cap * road.length:
                return True

        # 2. Telemetry inspection
        if telemetry is not None and road.id in telemetry:
            t_data = telemetry[road.id]
            if isinstance(t_data, dict):
                q_m = t_data.get("queue_length_meters", 0.0)
                vc = t_data.get("vc_ratio", 0.0)
                if q_m >= self.downstream_occupancy_cap * road.length or vc >= 1.0:
                    return True

        return False

    def _get_road_queue(
        self,
        road_id: str,
        simulator: Optional[Any],
        telemetry: Optional[Dict[str, Any]],
    ) -> float:
        """Returns the queue count or queue length for a given road segment."""
        if simulator is not None and hasattr(simulator, "vehicles"):
            q_cnt = 0
            for v in simulator.vehicles.values():
                if v.current_road_id == road_id and (v.speed < 1.0 or v.wait_time > 0.0):
                    q_cnt += 1
            if q_cnt > 0:
                return float(q_cnt)
            # Count in-flight vehicles if none are completely stopped
            all_v = sum(1 for v in simulator.vehicles.values() if v.current_road_id == road_id)
            return float(all_v)

        if telemetry is not None and road_id in telemetry:
            data = telemetry[road_id]
            if isinstance(data, dict):
                q_m = data.get("queue_length_meters", 0.0)
                return max(0.0, q_m / 7.0)

        return 0.0

    def _is_road_ns(self, road: Any) -> bool:
        return is_ns_approach(road)

    def _get_controlled_intersection_ids(self, simulator: Optional[Any]) -> List[str]:
        if self.controlled_intersections is not None:
            return list(self.controlled_intersections)
        if simulator is not None and hasattr(simulator, "signal_controllers"):
            return list(simulator.signal_controllers.keys())
        if simulator is not None and hasattr(simulator, "intersections"):
            return [i_id for i_id, inter in simulator.intersections.items() if inter.has_signals]
        return []


class CooperativePredictiveSignalController(SignalControllerBase):
    """Cooperative, network-aware traffic signal controller with predictive spillback intelligence.

    Phase 6C: Cooperative Predictive Signal Control.

    Architecture & Coordination Mechanics:
    1. Local Demand: Evaluates approaching queues on the intersection's incoming approach links.
    2. Upstream Anticipation: Exchanges state with upstream signalized neighbors to anticipate
       inbound vehicle platoons discharged during upstream green phases.
    3. Downstream Receiving Capacity: Models physical link storage capacity (link_length * lanes / 7.5m).
       If a downstream link's occupancy exceeds downstream_occupancy_cap (75%), a strict storage
       safeguard veto (-100.0) suppresses green allocation, preventing spillback gridlock.
    4. Downstream Neighbor Coordination: Shares signal phase with downstream signalized neighbors.
       Allocates a coordination bonus (+2.5) if downstream is currently green (green wave discharge),
       or a dampening penalty if downstream is red and queuing.
    5. Advisory Predictive Intelligence: Loads the 10-minute network spillback classifier after
       verifying its cryptographic SHA-256 hash (9B93E5DB672F692E509DC3012658413BC713FC6348CCD5D044B33470FF568C54).
       Applies a bounded advisory penalty (capped at 8.0) when predicted spillback probability exceeds
       the calibrated threshold (0.180).
    6. Safety Hierarchy: Physical capacity limits and FSM clearance guards strictly take precedence
       over advisory predictions. Missing or hash-mismatched models gracefully fall back to pure
       cooperative control without crashing or interrupting simulation.
    """

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
        min_green: float = 10.0,
        max_green: float = 60.0,
        downstream_occupancy_cap: float = 0.75,
        hysteresis_threshold: float = 1.5,
        predictive_weight: float = 10.0,
        classifier_path: Optional[Union[Path, str]] = None,
        enable_prediction: bool = True,
        controlled_intersections: Optional[List[str]] = None,
    ):
        self.min_green = float(min_green)
        self.max_green = float(max_green)
        self.downstream_occupancy_cap = float(downstream_occupancy_cap)
        self.hysteresis_threshold = float(hysteresis_threshold)
        self.predictive_weight = float(predictive_weight)
        self.enable_prediction = bool(enable_prediction)
        self.controlled_intersections = controlled_intersections

        # Internal state
        self._last_requested_phase: Dict[str, int] = {}
        self._last_downstream_occupancy: Dict[str, float] = {}
        self.calibrator = None
        self._booster: Optional[Any] = None
        self._sigmoid_calibrator: Optional[Any] = None
        self.expected_features: List[str] = []
        self.threshold: float = 0.180
        self.predictive_available: bool = False
        self.predictive_status: str = "INITIALIZING"

        # Diagnostic tracking
        self.last_decision_records: List[Dict[str, Any]] = []
        self.prediction_changed_decisions_count: int = 0
        self.nonzero_penalties_count: int = 0

        if self.enable_prediction:
            self._init_classifier(classifier_path or self.DEFAULT_CLASSIFIER_PATH)
        else:
            self.predictive_available = False
            self.predictive_status = "DISABLED"

    def _init_classifier(self, path: Union[Path, str]) -> None:
        """Loads the 10-minute network spillback classifier with cryptographic SHA-256 verification."""
        path = Path(path)
        if not path.exists():
            self.predictive_available = False
            self.predictive_status = f"UNAVAILABLE_FILE_NOT_FOUND: {path.name}"
            logger.warning(self.predictive_status)
            return

        # Cryptographic SHA-256 verification before loading
        try:
            content = path.read_bytes()
            digest = hashlib.sha256(content).hexdigest().upper()
        except Exception as exc:
            self.predictive_available = False
            self.predictive_status = f"UNAVAILABLE_READ_ERROR: {exc}"
            logger.error(self.predictive_status)
            return

        if digest != self.EXPECTED_CLASSIFIER_SHA256:
            self.predictive_available = False
            self.predictive_status = (
                f"UNAVAILABLE_HASH_MISMATCH: expected {self.EXPECTED_CLASSIFIER_SHA256[:8]}, got {digest[:8]}"
            )
            logger.error(self.predictive_status)
            return

        try:
            artifact = joblib.load(path)
            self.calibrator = artifact["calibrator"]
            self.expected_features = list(artifact["features"])
            self.threshold = float(artifact.get("threshold", 0.180))

            # Initialize robust native LightGBM predictor and sigmoid calibrator
            try:
                import lightgbm as lgb
                base_m = artifact.get("base_model")
                if base_m is not None and hasattr(base_m, "booster_"):
                    model_str = base_m.booster_.model_to_string()
                    self._booster = lgb.Booster(model_str=model_str)
                elif hasattr(self.calibrator, "calibrated_classifiers_"):
                    first_cc = self.calibrator.calibrated_classifiers_[0]
                    if hasattr(first_cc.estimator, "booster_"):
                        model_str = first_cc.estimator.booster_.model_to_string()
                        self._booster = lgb.Booster(model_str=model_str)

                if hasattr(self.calibrator, "calibrated_classifiers_") and self.calibrator.calibrated_classifiers_:
                    first_cc = self.calibrator.calibrated_classifiers_[0]
                    if hasattr(first_cc, "calibrators") and first_cc.calibrators:
                        self._sigmoid_calibrator = first_cc.calibrators[0]
            except Exception as exc:
                logger.warning(f"Could not initialize native LightGBM booster: {exc}")

            self.predictive_available = True
            self.predictive_status = "ACTIVE"
        except Exception as exc:
            self.predictive_available = False
            self.predictive_status = f"UNAVAILABLE_LOAD_ERROR: {exc}"
            logger.error(self.predictive_status)

    def reset(self) -> None:
        """Resets controller internal memory and call records."""
        self._last_requested_phase.clear()
        self._last_downstream_occupancy.clear()
        self.last_decision_records.clear()
        self.prediction_changed_decisions_count = 0
        self.nonzero_penalties_count = 0

    def compute_phases(
        self,
        current_time: float,
        simulator: Optional[Any] = None,
        telemetry: Optional[Dict[str, Any]] = None,
        record_decision_telemetry: bool = False,
    ) -> Dict[str, int]:
        """Calculates cooperative, network-aware phase requests for all controlled signalized intersections."""
        target_phases: Dict[str, int] = {}
        if simulator is None and telemetry is None:
            return target_phases

        inter_ids = self._get_controlled_intersection_ids(simulator)
        if not inter_ids:
            return target_phases

        local_controllers: Dict[str, Any] = getattr(simulator, "signal_controllers", {}) if simulator else {}
        if record_decision_telemetry:
            self.last_decision_records.clear()

        for inter_id in inter_ids:
            ctrl = local_controllers.get(inter_id)
            active_phase = ctrl.active_green_phase if ctrl else self._last_requested_phase.get(inter_id, 0)
            time_in_green = ctrl.time_in_green if ctrl else 0.0
            is_all_red_hold = (
                (ctrl.stage == "ALL_RED" and getattr(ctrl, "target_green_phase", None) == -1)
                if ctrl
                else (self._last_requested_phase.get(inter_id) == -1)
            )

            # Evaluate phase scores and blocked status
            if record_decision_telemetry:
                score_0, blocked_0, diag_0 = self._evaluate_phase_score(
                    inter_id, phase_index=0, simulator=simulator, telemetry=telemetry,
                    local_controllers=local_controllers, return_diagnostics=True
                )
                score_1, blocked_1, diag_1 = self._evaluate_phase_score(
                    inter_id, phase_index=1, simulator=simulator, telemetry=telemetry,
                    local_controllers=local_controllers, return_diagnostics=True
                )
            else:
                score_0, blocked_0 = self._evaluate_phase_score(
                    inter_id, phase_index=0, simulator=simulator, telemetry=telemetry,
                    local_controllers=local_controllers, return_diagnostics=False
                )
                score_1, blocked_1 = self._evaluate_phase_score(
                    inter_id, phase_index=1, simulator=simulator, telemetry=telemetry,
                    local_controllers=local_controllers, return_diagnostics=False
                )
                diag_0, diag_1 = {}, {}

            # Case A: If currently in ALL-RED hold
            if is_all_red_hold:
                if blocked_0 and blocked_1:
                    # Still both blocked: hold all-red safely
                    governing_rule = "ALL_RED_HOLD_STILL_BLOCKED"
                    chosen = -1
                    chosen_unconstrained = -1
                elif not blocked_0 and not blocked_1:
                    # Both phases are safe now: pick highest score, or phase 0 if tied
                    governing_rule = "ALL_RED_RELEASE_BOTH_SAFE"
                    chosen = 1 if score_1 > score_0 else 0
                    score_0_raw = diag_0.get("raw_score", score_0)
                    score_1_raw = diag_1.get("raw_score", score_1)
                    chosen_unconstrained = 1 if score_1_raw > score_0_raw else 0
                elif not blocked_0:
                    # Phase 0 became safe: release all-red to Phase 0
                    governing_rule = "ALL_RED_RELEASE_PHASE_0"
                    chosen = 0
                    chosen_unconstrained = 0
                else:
                    # Phase 1 became safe: release all-red to Phase 1
                    governing_rule = "ALL_RED_RELEASE_PHASE_1"
                    chosen = 1
                    chosen_unconstrained = 1

                target_phases[inter_id] = chosen
                self._last_requested_phase[inter_id] = chosen

                if record_decision_telemetry:
                    pred_changed = (chosen != chosen_unconstrained)
                    diag_active = diag_0 if active_phase == 0 else diag_1
                    diag_opposing = diag_1 if active_phase == 0 else diag_0
                    self.last_decision_records.append({
                        "timestamp": current_time,
                        "intersection_id": inter_id,
                        "active_phase": active_phase,
                        "selected_phase": chosen,
                        "opposing_phase": 1 if active_phase == 0 else 0,
                        "governing_rule": governing_rule,
                        "time_in_green": round(time_in_green, 2),
                        "blocked_active": blocked_0 if active_phase == 0 else blocked_1,
                        "blocked_opposing": blocked_1 if active_phase == 0 else blocked_0,
                        "score_active_raw": round(diag_active.get("raw_score", 0.0), 4),
                        "score_active_constrained": round(score_0 if active_phase == 0 else score_1, 4),
                        "score_opposing_raw": round(diag_opposing.get("raw_score", 0.0), 4),
                        "score_opposing_constrained": round(score_1 if active_phase == 0 else score_0, 4),
                        "pred_penalty_active": round(diag_active.get("pred_penalty", 0.0), 4),
                        "pred_penalty_opposing": round(diag_opposing.get("pred_penalty", 0.0), 4),
                        "max_vc_active": round(diag_active.get("max_vc", 0.0), 4),
                        "max_vc_opposing": round(diag_opposing.get("max_vc", 0.0), 4),
                        "max_p_spillback_active": round(diag_active.get("max_p_spillback", 0.0), 4),
                        "max_p_spillback_opposing": round(diag_opposing.get("max_p_spillback", 0.0), 4),
                        "counterfactual_phase_unconstrained": chosen_unconstrained,
                        "prediction_changed_decision": pred_changed,
                        "is_safeguard_override": True,
                    })
                    if pred_changed:
                        self.prediction_changed_decisions_count += 1
                    if (diag_0.get("pred_penalty", 0.0) > 0.0) or (diag_1.get("pred_penalty", 0.0) > 0.0):
                        self.nonzero_penalties_count += 1
                continue

            # Currently in GREEN (or transitioning through YELLOW)
            active_is_blocked = blocked_0 if active_phase == 0 else blocked_1
            opposing_is_blocked = blocked_1 if active_phase == 0 else blocked_0
            opposing_phase = 1 if active_phase == 0 else 0
            opposing_score = score_1 if active_phase == 0 else score_0
            active_score = score_0 if active_phase == 0 else score_1

            diag_active = diag_0 if active_phase == 0 else diag_1
            diag_opposing = diag_1 if active_phase == 0 else diag_0
            active_raw = diag_active.get("raw_score", active_score)
            opposing_raw = diag_opposing.get("raw_score", opposing_score)

            # 1. Minimum Green Safety Guard: Must not switch before min_green
            if time_in_green < self.min_green:
                governing_rule = "MIN_GREEN"
                chosen = active_phase
                chosen_unconstrained = active_phase

            # 2. Both Phases Blocked: Safe all-red hold
            elif active_is_blocked and opposing_is_blocked:
                governing_rule = "BOTH_BLOCKED_ALL_RED"
                chosen = -1
                chosen_unconstrained = -1

            # 3. Active Phase Blocked, Opposing Phase Safe: Safe switch away from blocked active phase
            elif active_is_blocked and not opposing_is_blocked:
                governing_rule = "ACTIVE_BLOCKED_SWITCH"
                chosen = opposing_phase
                chosen_unconstrained = opposing_phase

            # 4. Opposing Phase Blocked, Active Phase Safe: Keep active phase
            elif not active_is_blocked and opposing_is_blocked:
                governing_rule = "OPPOSING_BLOCKED_HOLD"
                chosen = active_phase
                chosen_unconstrained = active_phase

            # 5. Neither Phase Blocked (Both Phases Safe)
            elif time_in_green >= self.max_green:
                governing_rule = "MAX_GREEN_FORCE_OFF"
                chosen = opposing_phase
                chosen_unconstrained = opposing_phase

            # Before max_green: when both phases have zero demand, allow current green to rest
            elif score_0 <= 0.0 and score_1 <= 0.0:
                governing_rule = "ZERO_DEMAND_REST"
                chosen = active_phase
                score_0_raw = diag_0.get("raw_score", score_0)
                score_1_raw = diag_1.get("raw_score", score_1)
                if active_phase == 0:
                    chosen_unconstrained = 1 if (score_1_raw > score_0_raw + self.hysteresis_threshold and score_1_raw > 0.0) else 0
                else:
                    chosen_unconstrained = 0 if (score_0_raw > score_1_raw + self.hysteresis_threshold and score_0_raw > 0.0) else 1

            # Demand exists: standard cooperative pressure selection with hysteresis
            else:
                governing_rule = "PRESSURE_SELECTION"
                if active_phase == 0:
                    chosen = 1 if (score_1 > score_0 + self.hysteresis_threshold) else 0
                    chosen_unconstrained = 1 if (diag_1.get("raw_score", score_1) > diag_0.get("raw_score", score_0) + self.hysteresis_threshold) else 0
                else:
                    chosen = 0 if (score_0 > score_1 + self.hysteresis_threshold) else 1
                    chosen_unconstrained = 0 if (diag_0.get("raw_score", score_0) > diag_1.get("raw_score", score_1) + self.hysteresis_threshold) else 1

            target_phases[inter_id] = chosen
            self._last_requested_phase[inter_id] = chosen

            if record_decision_telemetry:
                pred_changed = (chosen != chosen_unconstrained)
                self.last_decision_records.append({
                    "timestamp": current_time,
                    "intersection_id": inter_id,
                    "active_phase": active_phase,
                    "selected_phase": chosen,
                    "opposing_phase": opposing_phase,
                    "governing_rule": governing_rule,
                    "time_in_green": round(time_in_green, 2),
                    "blocked_active": active_is_blocked,
                    "blocked_opposing": opposing_is_blocked,
                    "score_active_raw": round(active_raw, 4),
                    "score_active_constrained": round(active_score, 4),
                    "score_opposing_raw": round(opposing_raw, 4),
                    "score_opposing_constrained": round(opposing_score, 4),
                    "pred_penalty_active": round(diag_active.get("pred_penalty", 0.0), 4),
                    "pred_penalty_opposing": round(diag_opposing.get("pred_penalty", 0.0), 4),
                    "max_vc_active": round(diag_active.get("max_vc", 0.0), 4),
                    "max_vc_opposing": round(diag_opposing.get("max_vc", 0.0), 4),
                    "max_p_spillback_active": round(diag_active.get("max_p_spillback", 0.0), 4),
                    "max_p_spillback_opposing": round(diag_opposing.get("max_p_spillback", 0.0), 4),
                    "counterfactual_phase_unconstrained": chosen_unconstrained,
                    "prediction_changed_decision": pred_changed,
                    "is_safeguard_override": (governing_rule in (
                        "MIN_GREEN", "MAX_GREEN_FORCE_OFF", "BOTH_BLOCKED_ALL_RED",
                        "ACTIVE_BLOCKED_SWITCH", "OPPOSING_BLOCKED_HOLD"
                    )),
                })
                if pred_changed:
                    self.prediction_changed_decisions_count += 1
                if (diag_0.get("pred_penalty", 0.0) > 0.0) or (diag_1.get("pred_penalty", 0.0) > 0.0):
                    self.nonzero_penalties_count += 1

        return target_phases

    def _evaluate_phase_score(
        self,
        inter_id: str,
        phase_index: int,
        simulator: Optional[Any],
        telemetry: Optional[Dict[str, Any]],
        local_controllers: Dict[str, Any],
        return_diagnostics: bool = False,
    ) -> Union[Tuple[float, bool], Tuple[float, bool, Dict[str, Any]]]:
        """Computes aggregate cooperative score and blocked status for a candidate phase across served movements."""
        is_ns_phase = (phase_index == 0)
        total_score = 0.0
        phase_is_blocked = False
        total_pred_penalty = 0.0
        total_vc_penalty = 0.0
        max_vc = 0.0
        max_p_spillback = 0.0

        if simulator is not None and hasattr(simulator, "roads"):
            in_roads = [
                r for r in simulator.roads.values()
                if r.to_node == inter_id and getattr(r, "signal_controlled", True) and self._is_road_ns(r) == is_ns_phase
            ]
            out_roads = [
                r for r in simulator.roads.values()
                if r.from_node == inter_id and self._is_road_ns(r) == is_ns_phase
            ]

            # 1. Strict Phase-Level Capacity Safeguard Check on all downstream receiving links (DEFECT-6C-01 Fix)
            # If any downstream link served by this phase exceeds physical storage capacity threshold,
            # the entire phase is vetoed. It cannot be made safe by parallel unblocked movements or bonuses.
            for out_r in out_roads:
                is_blocked, _ = self._check_downstream_storage(
                    out_r, simulator=simulator, telemetry=telemetry
                )
                t_out = None
                if telemetry is not None:
                    if out_r.id in telemetry:
                        t_out = telemetry[out_r.id]
                    elif "road_heatmap" in telemetry and isinstance(telemetry["road_heatmap"], dict) and out_r.id in telemetry["road_heatmap"]:
                        t_out = telemetry["road_heatmap"][out_r.id]

                if t_out is not None and isinstance(t_out, dict):
                    max_vc = max(max_vc, float(t_out.get("vc_ratio", 0.0) or 0.0))
                    p_raw = t_out.get("predicted_spillback_probability")
                    if p_raw is not None:
                        try:
                            max_p_spillback = max(max_p_spillback, float(p_raw))
                        except (ValueError, TypeError):
                            pass
                if is_blocked:
                    phase_is_blocked = True
                    break

            if phase_is_blocked:
                # Absolute phase-level veto: strictly negative score; cannot authorize traffic into blocked link
                diag = {
                    "raw_score": -100.0,
                    "constrained_score": -100.0,
                    "pred_penalty": 0.0,
                    "vc_penalty": 0.0,
                    "max_vc": max_vc,
                    "max_p_spillback": max_p_spillback,
                    "is_blocked": True,
                }
                if return_diagnostics:
                    return -100.0, True, diag
                return -100.0, True

            # 2. Evaluate unblocked movement pressure, coordination, and bounded predictive penalties
            for in_r in in_roads:
                q_u = self._get_road_queue(in_r.id, simulator=simulator, telemetry=telemetry)
                if q_u <= 0.0:
                    continue

                # Upstream neighbor look-ahead anticipation
                upstream_bonus = self._compute_upstream_anticipation(
                    in_r, simulator=simulator, local_controllers=local_controllers
                )

                if not out_roads:
                    # Perimeter exit: full discharge credit
                    total_score += (q_u + upstream_bonus)
                else:
                    for out_r in out_roads:
                        q_v = self._get_road_queue(out_r.id, simulator=simulator, telemetry=telemetry)
                        base_pressure = max(0.0, q_u - q_v)

                        _, occupancy = self._check_downstream_storage(
                            out_r, simulator=simulator, telemetry=telemetry
                        )
                        # Smooth receiving capacity attenuation as occupancy approaches threshold
                        receiving_factor = max(0.0, 1.0 - (occupancy / self.downstream_occupancy_cap))

                        # Downstream neighbor coordination (green wave / clearance bonus)
                        coord_factor = self._compute_downstream_coordination(
                            out_r, simulator=simulator, local_controllers=local_controllers
                        )

                        # Advisory predictive spillback penalty (bounded <= 8.0)
                        pred_penalty = self._compute_predictive_penalty(out_r.id, telemetry=telemetry)
                        total_pred_penalty += pred_penalty

                        # Operational V/C congestion penalty: separate bounded soft penalty
                        # Activates when downstream V/C exceeds 0.85, bounded to at most 3.0
                        # Does NOT declare a road physically full or trigger the storage veto
                        vc_penalty = 0.0
                        t_out = None
                        if telemetry is not None:
                            if out_r.id in telemetry:
                                t_out = telemetry[out_r.id]
                            elif "road_heatmap" in telemetry and isinstance(telemetry["road_heatmap"], dict) and out_r.id in telemetry["road_heatmap"]:
                                t_out = telemetry["road_heatmap"][out_r.id]

                        if t_out is not None and isinstance(t_out, dict):
                            vc_val = float(t_out.get("vc_ratio", 0.0) or 0.0)
                            max_vc = max(max_vc, vc_val)
                            p_raw = t_out.get("predicted_spillback_probability")
                            if p_raw is not None:
                                try:
                                    max_p_spillback = max(max_p_spillback, float(p_raw))
                                except (ValueError, TypeError):
                                    pass
                            if vc_val > 0.85:
                                vc_penalty = min(3.0, (vc_val - 0.85) * 5.0)
                        total_vc_penalty += vc_penalty

                        movement_score = (
                            (base_pressure * receiving_factor)
                            + upstream_bonus
                            + coord_factor
                            - pred_penalty
                            - vc_penalty
                        )
                        total_score += movement_score

            diag = {
                "raw_score": max(0.0, total_score + total_pred_penalty),
                "constrained_score": max(0.0, total_score),
                "pred_penalty": total_pred_penalty,
                "vc_penalty": total_vc_penalty,
                "max_vc": max_vc,
                "max_p_spillback": max_p_spillback,
                "is_blocked": False,
            }
            if return_diagnostics:
                return max(0.0, total_score), False, diag
            return max(0.0, total_score), False

        # Telemetry-only fallback
        if telemetry is not None:
            for road_id, data in telemetry.items():
                if isinstance(data, dict):
                    q = data.get("queue_length_meters", 0.0)
                    occ = data.get("occupancy_ratio", 0.0)
                    if occ >= self.downstream_occupancy_cap:
                        phase_is_blocked = True
                    total_score += max(0.0, q)
            diag = {
                "raw_score": max(0.0, total_score),
                "constrained_score": max(0.0, total_score),
                "pred_penalty": 0.0,
                "vc_penalty": 0.0,
                "max_vc": 0.0,
                "max_p_spillback": 0.0,
                "is_blocked": phase_is_blocked,
            }
            if phase_is_blocked:
                if return_diagnostics:
                    return -100.0, True, diag
                return -100.0, True
            if return_diagnostics:
                return max(0.0, total_score), False, diag
            return max(0.0, total_score), False

        default_diag = {
            "raw_score": 0.0,
            "constrained_score": 0.0,
            "pred_penalty": 0.0,
            "vc_penalty": 0.0,
            "max_vc": 0.0,
            "max_p_spillback": 0.0,
            "is_blocked": False,
        }
        if return_diagnostics:
            return 0.0, False, default_diag
        return 0.0, False

    def _compute_downstream_coordination(
        self,
        out_road: Any,
        simulator: Any,
        local_controllers: Dict[str, Any],
    ) -> float:
        """Shares traffic information with downstream neighboring signal controller."""
        target_node = simulator.intersections.get(out_road.to_node)
        if not target_node or not target_node.has_signals:
            return 0.0

        downstream_ctrl = local_controllers.get(target_node.id)
        if downstream_ctrl is None:
            return 0.0

        # Check what signal code downstream neighbor displays for this road's arrival
        is_ns = self._is_road_ns(out_road)
        downstream_code = downstream_ctrl.get_signal_code(is_ns=is_ns)

        if downstream_code == "G":
            # Downstream intersection is green: active discharge bonus (green wave coordination)
            return 2.5
        elif downstream_code == "r":
            # Downstream intersection is red: slight coordination dampening if link has standing queue
            q_v = self._get_road_queue(out_road.id, simulator=simulator, telemetry=None)
            max_storage = max(1.0, (out_road.length * out_road.lanes) / 7.5)
            queue_ratio = min(1.0, q_v / max_storage)
            return -2.0 * queue_ratio
        return 0.0

    def _compute_upstream_anticipation(
        self,
        in_road: Any,
        simulator: Any,
        local_controllers: Dict[str, Any],
    ) -> float:
        """Shares traffic information with upstream neighboring signal controller to anticipate inbound arrivals."""
        from_node = simulator.intersections.get(in_road.from_node)
        if not from_node or not from_node.has_signals:
            return 0.0

        upstream_ctrl = local_controllers.get(from_node.id)
        if upstream_ctrl is None:
            return 0.0

        is_ns = self._is_road_ns(in_road)
        upstream_code = upstream_ctrl.get_signal_code(is_ns=is_ns)

        if upstream_code == "G":
            # Upstream intersection is actively feeding this incoming road
            inbound_vehs = sum(
                1
                for v in simulator.vehicles.values()
                if v.current_road_id == in_road.id and v.distance_on_road < in_road.length * 0.5
            )
            return min(4.0, 1.0 * inbound_vehs)
        return 0.0

    def _check_downstream_storage(
        self,
        road: Any,
        simulator: Optional[Any],
        telemetry: Optional[Dict[str, Any]],
    ) -> Tuple[bool, float]:
        """Checks if downstream link exceeds physical storage capacity threshold (75%).

        Physical storage occupancy evaluates the fraction of physical road space occupied
        by vehicles (stopped or moving). Operational V/C is strictly excluded from this check.
        """
        lanes = max(1, getattr(road, "lanes", 1) if not isinstance(road, dict) else road.get("lanes", 1))
        road_length = max(1.0, float(getattr(road, "length", 100.0) if not isinstance(road, dict) else road.get("length", 100.0)))
        road_id = getattr(road, "id", "") if not isinstance(road, dict) else road.get("id", "")
        max_storage_vehicles = max(1.0, (road_length * lanes) / 7.5)

        sim_occ: Optional[float] = None
        # 1. Direct simulator inspection (ground truth physical space)
        if simulator is not None and hasattr(simulator, "vehicles"):
            vehs_on_road = [v for v in simulator.vehicles.values() if v.current_road_id == road_id]
            veh_count_occ = len(vehs_on_road) / max_storage_vehicles
            queued_len = sum(v.length + 2.5 for v in vehs_on_road if v.speed < 1.0)
            total_lane_length = road_length * lanes
            q_ratio = queued_len / total_lane_length
            sim_occ = max(veh_count_occ, q_ratio)

        # 2. Macroscopic telemetry inspection (physical occupancy only)
        tel_occ: Optional[float] = None
        t_data = None
        if telemetry is not None:
            if road_id in telemetry:
                t_data = telemetry[road_id]
            elif "road_heatmap" in telemetry and isinstance(telemetry["road_heatmap"], dict) and road_id in telemetry["road_heatmap"]:
                t_data = telemetry["road_heatmap"][road_id]

        if t_data is not None and isinstance(t_data, dict):
            # Macroscopic occupancy ratio directly from estimator
                raw_occ = t_data.get("occupancy_ratio")
                if raw_occ is not None:
                    try:
                        raw_occ_f = float(raw_occ)
                        if not np.isnan(raw_occ_f):
                            tel_occ = max(0.0, min(1.0, raw_occ_f))
                    except (ValueError, TypeError):
                        pass

                # Also inspect physical queue length and vehicle count if present
                q_m = t_data.get("queue_length_meters")
                if q_m is not None:
                    try:
                        q_m_f = float(q_m)
                        if not np.isnan(q_m_f):
                            q_occ = max(0.0, min(1.0, q_m_f / road_length))
                            tel_occ = max(tel_occ or 0.0, q_occ)
                    except (ValueError, TypeError):
                        pass

                v_cnt = t_data.get("vehicle_count")
                if v_cnt is not None:
                    try:
                        v_cnt_f = float(v_cnt)
                        if not np.isnan(v_cnt_f):
                            cnt_occ = max(0.0, min(1.0, v_cnt_f / max_storage_vehicles))
                            tel_occ = max(tel_occ or 0.0, cnt_occ)
                    except (ValueError, TypeError):
                        pass

        # 3. Safe reconciliation across sources
        if sim_occ is not None and tel_occ is not None:
            physical_occ = max(sim_occ, tel_occ)
        elif sim_occ is not None:
            physical_occ = sim_occ
        elif tel_occ is not None:
            physical_occ = tel_occ
        else:
            # Fall back to cached last-observed occupancy if available
            cached_occ = self._last_downstream_occupancy.get(road_id)
            if cached_occ is not None:
                physical_occ = cached_occ
            else:
                physical_occ = 0.0

        self._last_downstream_occupancy[road_id] = physical_occ
        is_blocked = (physical_occ >= self.downstream_occupancy_cap)
        return is_blocked, physical_occ

    def predict_spillback_probability(
        self,
        features: Union[Dict[str, float], np.ndarray, List[float]],
    ) -> float:
        """Evaluates calibrated 10-minute spillback probability for a given feature dictionary or vector."""
        if not self.predictive_available:
            return 0.0

        try:
            if isinstance(features, dict):
                row = np.array([[float(features.get(f, 0.0)) for f in self.expected_features]], dtype=np.float32)
            elif isinstance(features, np.ndarray):
                row = features.reshape(1, -1) if features.ndim == 1 else features
                row = row.astype(np.float32)
            else:
                row = np.array([features], dtype=np.float32)

            if self._booster is not None:
                raw = self._booster.predict(row, raw_score=True)
                if self._sigmoid_calibrator is not None:
                    prob = float(self._sigmoid_calibrator.predict(raw)[0])
                else:
                    prob = float(1.0 / (1.0 + np.exp(-raw[0])))
                return float(np.clip(prob, 0.0, 1.0))

            if self.calibrator is not None:
                names = getattr(self.calibrator, "feature_names_in_", None)
                inputs = pd.DataFrame(row, columns=names) if names is not None else row
                prob = float(self.calibrator.predict_proba(inputs)[0, 1])
                return float(np.clip(prob, 0.0, 1.0))

            return 0.0
        except Exception as exc:
            logger.warning(f"Failed to evaluate spillback prediction: {exc}")
            return 0.0

    def _compute_predictive_penalty(
        self,
        road_id: str,
        telemetry: Optional[Dict[str, Any]],
    ) -> float:
        """Computes bounded advisory penalty from 10-minute predicted spillback probability."""
        if not self.predictive_available:
            return 0.0

        p_spillback = 0.0
        t_data = None
        if telemetry is not None:
            if road_id in telemetry:
                t_data = telemetry[road_id]
            elif "road_heatmap" in telemetry and isinstance(telemetry["road_heatmap"], dict) and road_id in telemetry["road_heatmap"]:
                t_data = telemetry["road_heatmap"][road_id]

        if t_data and isinstance(t_data, dict):
            p_status = t_data.get("prediction_status")
            if p_status in ("INSUFFICIENT_HISTORY", "UNAVAILABLE"):
                return 0.0

            p_val = t_data.get("predicted_spillback_probability")
            if p_val is not None:
                try:
                    p_spillback = float(p_val)
                except (ValueError, TypeError):
                    p_spillback = 0.0
            elif "spillback_risk" in t_data and t_data["spillback_risk"] is not None:
                try:
                    p_spillback = float(t_data["spillback_risk"])
                except (ValueError, TypeError):
                    p_spillback = 0.0
            elif "features" in t_data and t_data["features"] is not None:
                p_spillback = self.predict_spillback_probability(t_data["features"])

        # Bounded advisory penalty: applies only when probability exceeds threshold (0.180)
        if p_spillback > self.threshold:
            excess = min(1.0, p_spillback) - self.threshold
            # Scaled by predictive_weight, strictly capped at 8.0 to ensure bounded influence
            return min(8.0, self.predictive_weight * excess)

        return 0.0

    def _get_road_queue(
        self,
        road_id: str,
        simulator: Optional[Any],
        telemetry: Optional[Dict[str, Any]],
    ) -> float:
        """Returns the queue count or queue length for a given road segment."""
        q_sim = 0.0
        if simulator is not None and hasattr(simulator, "vehicles"):
            q_cnt = 0
            for v in simulator.vehicles.values():
                if v.current_road_id == road_id and (v.speed < 1.0 or v.wait_time > 0.0):
                    q_cnt += 1
            if q_cnt > 0:
                q_sim = float(q_cnt)
            else:
                all_v = sum(1 for v in simulator.vehicles.values() if v.current_road_id == road_id)
                q_sim = float(all_v)

        q_tel = 0.0
        t_data = None
        if telemetry is not None:
            if road_id in telemetry:
                t_data = telemetry[road_id]
            elif "road_heatmap" in telemetry and isinstance(telemetry["road_heatmap"], dict) and road_id in telemetry["road_heatmap"]:
                t_data = telemetry["road_heatmap"][road_id]

        if t_data is not None and isinstance(t_data, dict):
            q_m = t_data.get("queue_length_meters", 0.0)
            v_c = t_data.get("vehicle_count", 0.0)
            q_tel = max(float(v_c), max(0.0, float(q_m) / 7.0))

        return max(q_sim, q_tel)

    def _is_road_ns(self, road: Any) -> bool:
        return is_ns_approach(road)

    def _get_controlled_intersection_ids(self, simulator: Optional[Any]) -> List[str]:
        if self.controlled_intersections is not None:
            return list(self.controlled_intersections)
        if simulator is not None and hasattr(simulator, "signal_controllers"):
            return list(simulator.signal_controllers.keys())
        if simulator is not None and hasattr(simulator, "intersections"):
            return [i_id for i_id, inter in simulator.intersections.items() if inter.has_signals]
        return []

