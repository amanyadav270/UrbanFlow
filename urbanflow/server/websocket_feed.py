"""Real-time simulation execution and WebSocket broadcast pipeline."""
from __future__ import annotations

import asyncio
from collections import deque, defaultdict
import json
import logging
from pathlib import Path
from typing import Set, Dict, Any, List, Optional
import numpy as np
from starlette.websockets import WebSocket, WebSocketDisconnect

from urbanflow.core.kinematic_sim import KinematicSimulator
from urbanflow.core.districts import AVAILABLE_NETWORKS, CITIES
from urbanflow.core.demand import STRESS_SCENARIOS, classroom_demand
from urbanflow.core.observed_demand import OBSERVED_WINDOWS
from urbanflow.core.classroom_evaluation import classroom_controller
from urbanflow.core.simulator_base import SimulatorAdapter
from urbanflow.core.types import is_ns_approach
from urbanflow.intelligence.state_estimator import TrafficStateEstimator, SegmentTrafficState
from urbanflow.intelligence.capacity_estimator import RoadCapacityEstimator
from urbanflow.intelligence.congestion_predictor import ModularCongestionPredictor
from urbanflow.intelligence.routing import (
    OnlineObservationBuffer,
    PredictiveRoutingController,
    RouteCostEvaluator,
)
from urbanflow.core.network_topology import NetworkTopology
from urbanflow.rl.live import POLICIES, PPOController
from urbanflow.intelligence.signal_control import (
    SignalControllerBase,
    FixedTimeController,
    ActuatedController,
    MaxPressureController,
    CooperativePredictiveSignalController,
)

logger = logging.getLogger(__name__)

ROUTING_MODES: Dict[str, Dict[str, Any]] = {
    "OFF": {
        "name": "Off",
        "description": "Keeps each vehicle on its original route.",
    },
    "REACTIVE": {
        "name": "Reactive",
        "description": "Looks for a better route using queues and available road space right now.",
    },
    "PREDICTIVE": {
        "name": "Predictive",
        "description": "Uses live traffic and forecast queue spillback to weigh alternate routes.",
    },
}

CONTROLLER_REGIMES: Dict[str, Dict[str, Any]] = {
    "FIXED_TIME": {
        "name": "Fixed-Time",
        "description": "Uses a repeating timetable for each signal.",
        "class": FixedTimeController,
    },
    "ACTUATED": {
        "name": "Actuated",
        "description": "Extends green for approaching traffic and switches when demand eases.",
        "class": ActuatedController,
    },
    "MAX_PRESSURE": {
        "name": "MaxPressure",
        "description": "Favors longer queues when the next road has room.",
        "class": MaxPressureController,
    },
    "COOPERATIVE_PREDICTIVE": {
        "name": "CooperativePredictive",
        "description": "Coordinates nearby junctions and estimates where queues may spill back.",
        "class": CooperativePredictiveSignalController,
    },
}

LEARNED_REGIMES = {
    key: {"name": f"PPO ({variant})", "description": "Learned policy matched to the selected network."}
    for key, variant in POLICIES.items()
}
CLASSROOM_MODEL = "PPO_NEIGHBOR"
CLASSROOM_MODEL_NAME = "Multi-Agent RL Model"
CLASSROOM_MODEL_DESCRIPTION = (
    "Four-intersection multi-agent reinforcement learning controller using a shared learned "
    "policy and neighboring intersection observations."
)



class SimulationBroadcaster:
    """Manages simulator execution loop and broadcasts snapshots to connected 3D twin clients."""

    def __init__(self, simulator: SimulatorAdapter):
        self.simulator = simulator
        self.control_error = None
        self.policy_cycles = 0
        self.clients: Set[WebSocket] = set()
        self.is_running = True
        self.is_paused = False
        self.sim_speed_multiplier = 1.0
        self.effective_speed = 0.0
        self.state_estimator = TrafficStateEstimator()
        self.capacity_estimator = RoadCapacityEstimator()
        self.predictor = ModularCongestionPredictor()
        
        # Load trained model artifacts ONCE upon broadcaster initialization
        project_root = Path(__file__).resolve().parent.parent.parent
        self.models_dir = project_root / "data" / "models"
        self.is_predictor_loaded = False
        if self.models_dir.exists():
            self.is_predictor_loaded = self.predictor.load_models(self.models_dir)
            if self.is_predictor_loaded:
                logger.info("ModularCongestionPredictor loaded models from %s", self.models_dir)
            else:
                logger.warning("No valid model artifacts found in %s; operating with baselines/fallback", self.models_dir)
        else:
            logger.info("Model directory %s does not exist; predictor operating in missing-model mode", self.models_dir)

        # Sliding state history per road for online rolling lag features (stores last 8 seconds of state)
        self._road_history: Dict[str, deque] = {}

        # Dedicated 28-feature online observation buffer sampled strictly at 5s intervals
        self.observation_buffer = OnlineObservationBuffer(NetworkTopology(data=simulator.get_network_topology()))
        self.last_obs_sample_time = -1.0
        self.obs_sample_interval_sec = 5.0
        # Dedicated spillback predictor instance to ensure predictive spillback evaluation
        # is available across all controllers/modes
        self.spillback_predictor = CooperativePredictiveSignalController(enable_prediction=True)

        self.active_horizon = "10min"
        self.last_intel_time = -1.0
        self.current_intel: Dict[str, Any] = {}
        self._task: asyncio.Task | None = None

        # Traffic Signal Controller Integration
        self.control_interval_sec = 1.0
        self.demand_revision = 0
        self.last_control_time = -1.0
        self.signal_decisions = deque(maxlen=48)
        self.signal_transitions = deque(maxlen=48)
        self.step_pending = False
        self.active_controller_name = "COOPERATIVE_PREDICTIVE"
        self.controller: SignalControllerBase = self._create_controller(self.active_controller_name)
        self.controller_telemetry: Dict[str, Any] = {}
        self._update_controller_telemetry()

        # Dynamic Routing Controller Integration
        self.routing_modes = ROUTING_MODES
        self.active_routing_mode = "OFF"
        self.routing_controller = PredictiveRoutingController(
            route_evaluator=RouteCostEvaluator(simulator=simulator),
            mode="off",
            evaluation_interval_sec=15.0,
        )
        self.routing_interval_sec = 15.0
        self.last_routing_time = -1.0
        self.cumulative_reroute_accepted = 0
        self.cumulative_reroute_rejected = 0
        self.rejection_reasons_tally: Dict[str, int] = defaultdict(int)
        self.recent_routing_decisions: deque = deque(maxlen=30)
        self.routing_telemetry: Dict[str, Any] = {}
        self.active_incidents: Dict[str, Dict[str, float]] = {}
        self._update_routing_telemetry()

    def _create_controller(self, name: str) -> SignalControllerBase:
        """Instantiates the requested signal controller with verified parameters."""
        key = str(name).strip().upper()
        if key in POLICIES:
            return PPOController(self.simulator, key)
        if key not in CONTROLLER_REGIMES:
            valid = list(CONTROLLER_REGIMES.keys())
            raise ValueError(f"Invalid signal controller '{name}'. Valid options: {valid}")

        if key in ("FIXED_TIME", "ACTUATED"):
            return classroom_controller(key)
        if key == "MAX_PRESSURE":
            return MaxPressureController(
                min_green=10.0,
                max_green=60.0,
                downstream_occupancy_cap=0.75,
                pressure_threshold=1.0,
            )
        elif key == "COOPERATIVE_PREDICTIVE":
            return CooperativePredictiveSignalController(
                min_green=10.0,
                max_green=60.0,
                downstream_occupancy_cap=0.75,
                hysteresis_threshold=1.5,
                predictive_weight=10.0,
                enable_prediction=True,
            )
        raise ValueError(f"Unhandled controller regime '{key}'")

    def set_controller(self, name: str) -> Dict[str, Any]:
        """Switches the active signal controller, resetting internal state safely."""
        key = str(name).strip().upper()
        new_ctrl = self._create_controller(key)
        self.active_controller_name = key
        self.controller = new_ctrl
        self.control_error = None
        self.control_interval_sec = getattr(new_ctrl, "decision_interval", 1.0)
        self.last_control_time = -1.0
        self._update_controller_telemetry()
        logger.info("Signal controller switched to %s", key)
        return self.controller_telemetry

    def activate_classroom_model(self):
        self.is_paused = True
        try:
            self.set_controller(CLASSROOM_MODEL)
        except ValueError as error:
            self.control_error = str(error)
            self._update_controller_telemetry()
            raise
        self.simulator.reset()
        self.reset_run()
        self.demand_revision += 1

    def prepare_classroom_model(self):
        try:
            self.activate_classroom_model()
        except ValueError as error:
            logger.warning("Classroom model unavailable: %s", error)
        self.set_routing_mode("OFF")

    def check_playback(self):
        if self.control_error:
            raise ValueError(self.control_error)

    def _update_controller_telemetry(self, phases: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        """Builds real-time diagnostic telemetry for the active signal controller."""
        regime_info = (CONTROLLER_REGIMES | LEARNED_REGIMES).get(self.active_controller_name, {})
        controlled_count = len(getattr(self.simulator, "signal_controllers", {}))

        # Count downstream blocked links (occupancy >= 0.75) if simulator is loaded
        blocked_storage = 0
        if hasattr(self.simulator, "roads") and self.simulator.roads:
            counts = defaultdict(int)
            for vehicle in self.simulator.vehicles.values():
                counts[vehicle.current_road_id] += 1
            for road in self.simulator.roads.values():
                cap = max(1.0, (road.length * road.lanes) / 7.5)
                if (counts[road.id] / cap) >= 0.75:
                    blocked_storage += 1

        if self.active_controller_name == CLASSROOM_MODEL and self.simulator.map_name == "classroom_grid":
            regime_info = {"name": CLASSROOM_MODEL_NAME, "description": CLASSROOM_MODEL_DESCRIPTION}
        self.controller_telemetry = {
            "error": self.control_error,
            "policy_cycles": self.policy_cycles,
            "checkpoint": getattr(self.controller, "checkpoint", None),
            "active_controller": self.active_controller_name,
            "controller_name": regime_info.get("name", self.active_controller_name),
            "description": regime_info.get("description", ""),
            "status": "ERROR" if self.control_error else "OPERATIONAL",
            "predictive_status": getattr(self.controller, "predictive_status", "N/A"),
            "control_interval_sec": self.control_interval_sec,
            "last_control_time": round(self.last_control_time, 2) if self.last_control_time >= 0 else None,
            "last_decisions_count": len(phases) if phases is not None else len(getattr(self.controller, "_last_requested_phase", {})),
            "controlled_intersections_count": controlled_count,
            "switches_count": getattr(self.controller, "prediction_changed_decisions_count", 0),
            "nonzero_penalties_count": getattr(self.controller, "nonzero_penalties_count", 0),
            "blocked_storage_count": blocked_storage,
            "last_phases": phases if phases is not None else getattr(self.controller, "_last_requested_phase", {}),
        }
        return self.controller_telemetry

    def set_routing_mode(self, mode: str) -> Dict[str, Any]:
        """Switches dynamic routing mode ('OFF', 'REACTIVE', 'PREDICTIVE')."""
        key = str(mode).strip().upper()
        if key not in ROUTING_MODES:
            valid = list(ROUTING_MODES.keys())
            raise ValueError(f"Invalid routing mode '{mode}'. Valid options: {valid}")

        self.active_routing_mode = key
        self.routing_controller.set_mode(key.lower())
        self.last_routing_time = -1.0
        self._update_routing_telemetry()
        logger.info("Routing mode switched to %s (%s)", key, ROUTING_MODES[key]["name"])
        return self.routing_telemetry

    def reset_routing(self) -> None:
        """Resets all routing state, reservations, cooldowns, and decision history."""
        self.routing_controller.reset()
        self.last_routing_time = -1.0
        self.cumulative_reroute_accepted = 0
        self.cumulative_reroute_rejected = 0
        self.rejection_reasons_tally.clear()
        self.recent_routing_decisions.clear()
        self.active_incidents.clear()
        self._update_routing_telemetry()

    def _update_routing_telemetry(self, last_report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Builds structured real-time diagnostic telemetry for dynamic routing."""
        mode_upper = self.active_routing_mode.upper()
        mode_meta = ROUTING_MODES.get(mode_upper, {})

        active_res_count = 0
        storage_warnings = []
        if hasattr(self.routing_controller, "active_reservations"):
            for r_id, res_dict in self.routing_controller.active_reservations.items():
                if res_dict:
                    active_res_count += len(res_dict)
                    storage_warnings.append({
                        "road_id": r_id,
                        "reserved_vehicles": len(res_dict),
                    })

        affected_vehicles = []
        if hasattr(self.simulator, "vehicles"):
            for v in sorted(self.simulator.vehicles.values(), key=lambda veh: veh.id):
                if getattr(v, "reroute_count", 0) > 0:
                    affected_vehicles.append({
                        "id": v.id,
                        "reroute_count": v.reroute_count,
                        "current_road": getattr(v, "current_road_id", None),
                        "route_index": v.route_index,
                        "total_roads": len(v.route),
                        "route": list(v.route),
                        "original_route": list(v.route_history[0]) if getattr(v, "route_history", None) else list(v.route),
                    })

        telemetry = {
            "active_mode": mode_upper,
            "mode_name": mode_meta.get("name", mode_upper),
            "mode_description": mode_meta.get("description", ""),
            "status": "ACTIVE" if mode_upper != "OFF" else "OFF",
            "predictive_available": getattr(self.routing_controller, "predictive_available", False),
            "predictive_status": getattr(self.routing_controller, "predictive_status", "OFF" if mode_upper == "OFF" else "N/A"),
            "last_evaluation_time": round(last_report.get("current_time", self.last_routing_time), 2) if (last_report or self.last_routing_time >= 0) else None,
            "evaluation_interval_sec": self.routing_interval_sec,
            "accepted_count": self.cumulative_reroute_accepted,
            "rejected_count": self.cumulative_reroute_rejected,
            "rejection_reasons": dict(self.rejection_reasons_tally),
            "recent_decisions": list(self.recent_routing_decisions),
            "active_reservations_count": active_res_count,
            "storage_warnings": storage_warnings,
            "affected_vehicles_count": len(affected_vehicles),
            "affected_vehicles": affected_vehicles[:20],
            "safeguards": {
                "max_diversion_quota": self.routing_controller.max_diversion_quota,
                "cooldown_sec": self.routing_controller.cooldown_sec,
                "min_improvement_sec": self.routing_controller.min_improvement_sec,
                "min_improvement_ratio": self.routing_controller.min_improvement_ratio,
                "vc_admission_cap": self.routing_controller.vc_admission_cap,
                "storage_safety_limit": self.routing_controller.storage_safety_limit,
            },
        }
        self.routing_telemetry = telemetry
        return telemetry

    async def register(self, websocket: WebSocket):
        await websocket.accept()
        self.clients.add(websocket)
        # Send initial topology immediately upon connection
        topology = self.simulator.get_network_topology()
        await websocket.send_json({"type": "topology", "data": topology})
        await websocket.send_text(self.snapshot_message(self.simulator._get_snapshot()))

    def unregister(self, websocket: WebSocket):
        self.clients.discard(websocket)

    def demand_finished(self):
        sim = self.simulator
        return (bool(sim.departures) and sim.departure_index >= len(sim.departures)
                and not sim.pending_spawns and not sim.vehicles)

    async def handle_client_message(self, message: Dict[str, Any]):
        """Processes control actions from client operations HUD."""
        action = message.get("action")
        if action == "pause":
            self.is_paused = True
        elif action == "step_decision":
            self.check_playback()
            if not self.is_paused or self.demand_finished():
                raise ValueError("Pause an unfinished run before stepping.")
            if self.step_pending:
                raise ValueError("A decision step is already pending.")
            self.step_pending = True
        elif action == "resume":
            self.check_playback()
            if self.demand_finished():
                self.simulator.reset()
                self.reset_run()
            self.is_paused = False
        elif action == "toggle_pause":
            if self.is_paused:
                self.check_playback()
            self.is_paused = not self.is_paused
        elif action == "set_speed":
            self.sim_speed_multiplier = max(0.2, min(20.0, float(message.get("multiplier", 1.0))))
            self.effective_speed = 0.0
        elif action == "set_traffic_factor":
            self.simulator.set_traffic_factor(float(message["factor"]))
            self.demand_revision += 1
            self.reset_run()
        elif action == "reset":
            self.simulator.reset()
            self.reset_run()
        elif action == "start_routing_demo":
            self.check_playback()
            from urbanflow.core.route_planner import routing_demo

            self.simulator.reset()
            self.reset_run()
            demo = routing_demo(self.simulator)
            self.demand_revision += 1
            self.set_routing_mode("PREDICTIVE")
            self.simulator.inject_incident(demo["road"], demo["severity"], demo["end_time"])
            self.active_incidents[demo["road"]] = {"severity": demo["severity"], "end_time": demo["end_time"]}
            self.is_paused = False
            return demo
        elif action == "load_map":
            map_name = message.get("map_name", "bronx_fordham")
            if map_name not in AVAILABLE_NETWORKS:
                raise ValueError("Unknown road network.")
            if hasattr(self.simulator, "load_map"):
                self.simulator.load_map(map_name)
            self.control_error = None
            if self.active_controller_name in POLICIES and map_name != "classroom_grid":
                try:
                    self.set_controller(self.active_controller_name)
                except ValueError:
                    self.set_controller("FIXED_TIME")
            scenario = "rush_hour" if map_name == "classroom_grid" else "normal"
            if map_name in CITIES:
                scenario = "city_shift"
            self.simulator.set_demand(classroom_demand(self.simulator, scenario), scenario)
            self.demand_revision += 1
            topology = self.simulator.get_network_topology()
            self.observation_buffer = OnlineObservationBuffer(NetworkTopology(data=topology))
            evaluation_mode = "predictive" if self.active_routing_mode == "PREDICTIVE" else "reactive"
            self.routing_controller = PredictiveRoutingController(
                route_evaluator=RouteCostEvaluator(simulator=self.simulator, mode=evaluation_mode),
                mode=self.active_routing_mode.lower(),
                evaluation_interval_sec=self.routing_interval_sec,
            )
            self.reset_run()
            if map_name == "classroom_grid":
                self.simulator.set_traffic_factor(1)
                self.prepare_classroom_model()
            elif imported is not None:
                self.is_paused = True
            topo_payload = json.dumps({"type": "topology", "data": topology})
            for client in list(self.clients):
                try:
                    await client.send_text(topo_payload)
                except Exception:
                    pass
        elif action == "trigger_surge":
            surge_road = "west_in" if "west_in" in getattr(self.simulator, "roads", {}) else ""
            v_count = int(message.get("vehicle_count", 25))
            self.simulator.trigger_congestion_surge(surge_road, vehicle_count=v_count)
            logger.info("[SURGE] Injected congestion surge of %d vehicles", v_count)
        elif action == "inject_incident":
            road_id = message.get("road_id")
            if not road_id or road_id not in getattr(self.simulator, "roads", {}):
                if "road_1082220763_166_fwd" in getattr(self.simulator, "roads", {}):
                    road_id = "road_1082220763_166_fwd"
                elif "west_in" in getattr(self.simulator, "roads", {}):
                    road_id = "west_in"
                elif getattr(self.simulator, "roads", {}):
                    road_id = next(iter(self.simulator.roads))
            if road_id:
                severity = float(message.get("severity", 0.75))
                duration = float(message.get("duration", 60.0))
                self.simulator.inject_incident(road_id, severity=severity, duration_sec=duration)
                self.active_incidents[road_id] = {
                    "severity": severity,
                    "end_time": self.simulator.sim_time + duration,
                }
                logger.info("[INCIDENT] Injected incident on road '%s' (severity: %.2f, duration: %.1fs)", road_id, severity, duration)
        elif action == "load_scenario":
            scen = str(message.get("scenario", "normal")).lower()
            scen = "surge" if scen == "peak_hour" else scen
            if scen in ("normal", "rush_hour", "surge", *OBSERVED_WINDOWS, *STRESS_SCENARIOS) or self.simulator.map_name == "classroom_grid":
                self.simulator.set_demand(classroom_demand(self.simulator, scen), scen)
                self.demand_revision += 1
                self.reset_run()
                return
            if scen == "normal":
                self.active_incidents.clear()
                self.reset_routing()
                logger.info("[SCENARIO] Loaded Normal Traffic Flow")
            elif scen in ("peak_hour", "surge"):
                self.simulator.trigger_congestion_surge(vehicle_count=30)
                logger.info("[SCENARIO] Injected Peak-Hour Congestion Surge (30 vehicles)")
            elif scen == "incident":
                target_road = "road_1082220763_166_fwd" if "road_1082220763_166_fwd" in getattr(self.simulator, "roads", {}) else ("west_in" if "west_in" in getattr(self.simulator, "roads", {}) else (next(iter(self.simulator.roads)) if self.simulator.roads else ""))
                if target_road:
                    self.simulator.inject_incident(target_road, severity=0.8, duration_sec=90.0)
                    self.active_incidents[target_road] = {
                        "severity": 0.8,
                        "end_time": self.simulator.sim_time + 90.0,
                    }
                    logger.info("[SCENARIO] Injected Arterial Incident on '%s'", target_road)
            elif scen == "gridlock":
                self.simulator.trigger_congestion_surge(vehicle_count=35)
                target_road = "road_1082220763_166_fwd" if "road_1082220763_166_fwd" in getattr(self.simulator, "roads", {}) else ("west_in" if "west_in" in getattr(self.simulator, "roads", {}) else "")
                if target_road:
                    self.simulator.inject_incident(target_road, severity=0.9, duration_sec=120.0)
                    self.active_incidents[target_road] = {
                        "severity": 0.9,
                        "end_time": self.simulator.sim_time + 120.0,
                    }
                logger.info("[SCENARIO] Injected Downstream Gridlock & Storage Stress")
        elif action == "set_phase":
            phase_idx = int(message.get("phase", 0))
            self.simulator.apply_signal_phases({"INT_CENTER": phase_idx})
        elif action == "set_controller":
            ctrl_name = message.get("controller", "")
            self.set_controller(ctrl_name)
        elif action == "set_routing_mode":
            mode_name = message.get("mode", "")
            self.set_routing_mode(mode_name)
        elif action == "set_horizon":
            new_horizon = message.get("horizon", "10min")
            if new_horizon in self.predictor.HORIZONS:
                self.active_horizon = new_horizon
                # Invalidate last intelligence cache time to force re-evaluation on next frame
                self.last_intel_time = -1.0

    def reset_run(self):
        self.policy_cycles = 0
        self.signal_decisions.clear()
        self.signal_transitions.clear()
        self.step_pending = False
        self.effective_speed = 0.0
        self._road_history.clear()
        self.observation_buffer.reset()
        self.last_obs_sample_time = -1.0
        self.last_intel_time = -1.0
        self.current_intel = {}
        self.controller.reset()
        self.last_control_time = -1.0
        self._update_controller_telemetry()
        self.reset_routing()

    def _build_road_online_features(self, state: SegmentTrafficState) -> Dict[str, Any]:
        """Constructs lag and rolling features from historical buffer for real-time model inference."""
        r_id = state.road_id
        hist = list(self._road_history.get(r_id, []))

        # Lag extraction (1s step interval in live broadcast)
        speed_lag_1 = hist[-2].average_speed_kmh if len(hist) >= 2 else state.average_speed_kmh
        speed_lag_2 = hist[-3].average_speed_kmh if len(hist) >= 3 else speed_lag_1
        speed_lag_3 = hist[-4].average_speed_kmh if len(hist) >= 4 else speed_lag_2

        dens_lag_1 = hist[-2].density_per_km_lane if len(hist) >= 2 else state.density_per_km_lane
        dens_lag_2 = hist[-3].density_per_km_lane if len(hist) >= 3 else dens_lag_1
        dens_lag_3 = hist[-4].density_per_km_lane if len(hist) >= 4 else dens_lag_2

        q_lag_1 = hist[-2].queue_length_meters if len(hist) >= 2 else state.queue_length_meters
        q_lag_2 = hist[-3].queue_length_meters if len(hist) >= 3 else q_lag_1
        q_lag_3 = hist[-4].queue_length_meters if len(hist) >= 4 else q_lag_2

        recent_speeds = [h.average_speed_kmh for h in hist[-4:]] if hist else [state.average_speed_kmh]
        recent_dens = [h.density_per_km_lane for h in hist[-4:]] if hist else [state.density_per_km_lane]
        recent_q = [h.queue_length_meters for h in hist[-4:]] if hist else [state.queue_length_meters]

        speed_roll_mean = float(np.mean(recent_speeds))
        speed_roll_std = float(np.std(recent_speeds)) if len(recent_speeds) > 1 else 0.0

        return {
            "current_val": state.average_speed_kmh,
            "avg_speed": state.average_speed_kmh,
            "density": state.density_per_km_lane,
            "flow": state.flow_veh_per_hour,
            "queue_length": state.queue_length_meters,
            "occupancy": state.occupancy_ratio,
            "capacity_utilization": state.capacity_utilization,
            "signal_is_red": 1 if state.signal_state == "r" else 0,
            "incident_active": 1 if state.incident_active else 0,
            "avg_speed_roll_mean": speed_roll_mean,
            "avg_speed_roll_std": speed_roll_std,
            "avg_speed_lag_1": speed_lag_1,
            "avg_speed_lag_2": speed_lag_2,
            "avg_speed_lag_3": speed_lag_3,
            "density_roll_mean": float(np.mean(recent_dens)),
            "density_lag_1": dens_lag_1,
            "density_lag_2": dens_lag_2,
            "density_lag_3": dens_lag_3,
            "queue_roll_mean": float(np.mean(recent_q)),
            "queue_lag_1": q_lag_1,
            "queue_lag_2": q_lag_2,
            "queue_lag_3": q_lag_3,
        }

    def _compute_intelligence(self, snapshot, *, fast_forward=False) -> Dict[str, Any]:
        """Calculates road-by-road state estimates, operational capacity, 28-feature observations, and downstream bottlenecks."""
        if not hasattr(self.simulator, "roads"):
            return {}

        sim = self.simulator
        vehs_by_road: Dict[str, list] = {}
        for v in sim.vehicles.values():
            vehs_by_road.setdefault(v.current_road_id, []).append(v)

        road_states: List[SegmentTrafficState] = []
        road_states_map: Dict[str, Any] = {}

        for r_id, road in sim.roads.items():
            if r_id.startswith("CONN__"):
                continue
            vehs = vehs_by_road.get(r_id, [])
            target_node = sim.intersections.get(road.to_node)
            downstream_sig = "NONE"
            if target_node and target_node.has_signals and road.signal_controlled:
                is_ns = is_ns_approach(road)
                ctrl = getattr(sim, "signal_controllers", {}).get(target_node.id)
                if ctrl is not None:
                    downstream_sig = ctrl.get_signal_code(is_ns=is_ns)
                else:
                    phase_info = sim.phases[min(getattr(sim, "signal_phase", 0), len(sim.phases) - 1)]
                    downstream_sig = phase_info["ns"] if is_ns else phase_info["ew"]

            inc_info = self.active_incidents.get(r_id)
            if inc_info and snapshot.timestamp > inc_info.get("end_time", 0.0):
                self.active_incidents.pop(r_id, None)
                inc_info = None
            inc_sev = inc_info["severity"] if inc_info else 0.0

            state = self.state_estimator.estimate_segment_state(
                road_id=r_id,
                timestamp=snapshot.timestamp,
                road_length=road.length,
                lanes=road.lanes,
                speed_limit_mps=road.speed_limit,
                vehicles_on_road=vehs,
                downstream_signal_state=downstream_sig,
                incident_severity=inc_sev,
            )
            road_states.append(state)
            road_states_map[r_id] = state.to_dict()

            # Maintain historical state window for online feature lag extraction
            if r_id not in self._road_history:
                self._road_history[r_id] = deque(maxlen=8)
            self._road_history[r_id].append(state)

        # Sample into OnlineObservationBuffer at strictly 5-second simulation intervals
        if self.last_obs_sample_time < 0 or (snapshot.timestamp - self.last_obs_sample_time) >= self.obs_sample_interval_sec:
            self.observation_buffer.update(road_states_map, snapshot.timestamp)
            self.last_obs_sample_time = snapshot.timestamp

        heatmap: Dict[str, Any] = {}
        spillback_needed = (not fast_forward or self.active_controller_name == "COOPERATIVE_PREDICTIVE"
                            or self.active_routing_mode == "PREDICTIVE")
        for state in road_states:
            r_id = state.road_id
            los, risk, color = self.capacity_estimator.classify_los(state.capacity_utilization)

            # Build 28 features and evaluate calibrated spillback probability
            if spillback_needed and self.observation_buffer.is_road_ready(r_id):
                feat_28 = self.observation_buffer.build_feature_dict(r_id)
                if self.spillback_predictor.predictive_available and feat_28 is not None:
                    p_spillback = self.spillback_predictor.predict_spillback_probability(feat_28)
                    pred_status = "AVAILABLE"
                else:
                    p_spillback = None
                    pred_status = "MODEL_UNAVAILABLE"
            else:
                feat_28 = None
                p_spillback = None
                pred_status = "INSUFFICIENT_HISTORY"

            road_info = {
                "road_id": r_id,
                "color": color,
                "los": los,
                "risk": risk,
                "vc_ratio": state.capacity_utilization,
                "capacity_utilization": state.capacity_utilization,
                "speed": state.average_speed_kmh,
                "average_speed_kmh": state.average_speed_kmh,
                "density": state.density_per_km_lane,
                "density_per_km_lane": state.density_per_km_lane,
                "queue_m": state.queue_length_meters,
                "queue_length_meters": state.queue_length_meters,
                "queue_vehicle_count": state.queue_vehicle_count,
                "vehicle_count": state.vehicle_count,
                "flow_veh_per_hour": state.flow_veh_per_hour,
                "occupancy_ratio": state.occupancy_ratio,
                "operational_capacity": state.operational_capacity,
                "theoretical_capacity": state.theoretical_capacity,
                "signal_state": state.signal_state,
                "incident_active": state.incident_active,
                "features": feat_28,
                "predicted_spillback_probability": p_spillback,
                "prediction_status": pred_status,
            }
            heatmap[r_id] = road_info

        # Sort bottlenecks by V/C and queue length
        sorted_bottlenecks = sorted(road_states, key=lambda s: (s.capacity_utilization, s.queue_length_meters), reverse=True)
        top_bottlenecks = []

        for s in sorted_bottlenecks[:5]:
            # Build real supervised feature vector from state and historical window
            feat = self._build_road_online_features(s)

            # Query real multi-horizon prediction engine (NO heuristic multipliers)
            pred_out = self.predictor.predict_online(
                current_state_features=feat,
                horizon=self.active_horizon,
                timestamp=snapshot.timestamp,
            ) if not fast_forward else {}

            # Extract model prediction values and model identities
            predicted_speed = pred_out.get("predicted_avg_speed")
            predicted_density = pred_out.get("predicted_density")
            predicted_queue = pred_out.get("predicted_queue_length")
            congestion_prob = pred_out.get("congestion_probability")
            models_used = pred_out.get("models_used", {})
            inf_status = pred_out.get("inference_status", "unavailable")

            top_bottlenecks.append({
                "road_id": s.road_id,
                "name": getattr(sim.roads.get(s.road_id), "name", s.road_id) or s.road_id,
                "current_speed": s.average_speed_kmh,
                "predicted_speed": predicted_speed,
                "predicted_density": predicted_density,
                "predicted_queue_m": predicted_queue,
                "congestion_probability": congestion_prob,
                "predicted_spillback_probability": heatmap[s.road_id].get("predicted_spillback_probability"),
                "prediction_status": heatmap[s.road_id].get("prediction_status"),
                "model_used": models_used.get("avg_speed"),
                "models_used": models_used,
                "inference_status": inf_status,
                "horizon": self.active_horizon,
                "prediction_timestamp": snapshot.timestamp,
                "queue_m": s.queue_length_meters,
                "vc_ratio": s.capacity_utilization,
                "color": heatmap[s.road_id]["color"],
                "los": heatmap[s.road_id]["los"],
            })

        total_theo = sum(s.theoretical_capacity for s in road_states)
        total_oper = sum(s.operational_capacity for s in road_states)
        avg_vc = (sum(s.capacity_utilization for s in road_states) / len(road_states)) if road_states else 0.0

        intel = {
            "horizon": self.active_horizon,
            "horizon_seconds": self.predictor.HORIZONS.get(self.active_horizon, 600),
            "prediction_timestamp": round(snapshot.timestamp, 2),
            "active_models": self.predictor.BEST_MODELS.get(self.active_horizon, {}),
            "predictor_loaded": self.is_predictor_loaded,
            "bottlenecks": top_bottlenecks,
            "road_heatmap": heatmap,
            "capacity_summary": {
                "theoretical_cap_total": round(total_theo, 0),
                "operational_cap_total": round(total_oper, 0),
                "avg_vc_ratio": round(avg_vc, 3),
                "methodology": "HCM-inspired operational V/C heuristic",
            },
            "network_risk_level": "CRITICAL" if any(b["vc_ratio"] >= 0.90 for b in top_bottlenecks) else ("ELEVATED" if avg_vc > 0.50 else "NORMAL"),
            "data_source": "simulation_synthetic",
        }
        # Populate per-road keys at root level for direct telemetry[road_id] access
        for r_id, r_info in heatmap.items():
            intel[r_id] = r_info

        return intel

    def snapshot_message(self, snapshot, *, include_intel=True):
        data = snapshot.to_dict()
        data["demand"] = self.simulator.demand_counts()
        data["map_name"] = self.simulator.map_name
        data["demand_revision"] = self.demand_revision
        data["signal_decisions"] = list(self.signal_decisions)
        data["signal_transitions"] = list(self.signal_transitions)
        if self.active_controller_name in POLICIES and self.simulator.map_name == "classroom_grid":
            data["agent_signals"] = {}
            for node, signal in self.simulator.signal_controllers.items():
                target = signal.target_green_phase
                active = signal.active_green_phase
                remaining = None
                if signal.stage == "GREEN" and target is not None and target != active:
                    remaining = max(0.0, signal.min_green - signal.time_in_green)
                elif signal.stage != "GREEN":
                    remaining = max(0.0, signal.current_duration - signal.time_in_stage)
                data["agent_signals"][node] = {
                    "stage": signal.stage, "target": target, "active": active,
                    "earliest_stage_end": round(snapshot.timestamp + remaining, 2) if remaining is not None else None,
                    "reservation": node in self.simulator.junction_reservations,
                }
        data["paused"] = self.is_paused
        data["finished"] = self.demand_finished()
        data["speed_multiplier"] = self.sim_speed_multiplier
        data["effective_speed"] = round(self.effective_speed, 1) if not self.is_paused else 0.0
        data["incidents"] = [{"road_id": road, **entry} for road, entry in self.active_incidents.items()
                             if entry["end_time"] > snapshot.timestamp]
        return json.dumps({
            "type": "snapshot",
            "data": data,
            "intelligence": self.current_intel if include_intel else None,
            "signal_controller": self.controller_telemetry,
            "routing_controller": self.routing_telemetry,
        })

    async def broadcast_snapshot(self, snapshot, *, include_intel=True):
        message = self.snapshot_message(snapshot, include_intel=include_intel)
        for client in list(self.clients):
            try:
                await asyncio.wait_for(client.send_text(message), timeout=0.5)
            except Exception:
                self.unregister(client)

    async def run_loop(self):
        """Asynchronous fixed-rate tick loop decoupled from WebGL client rendering."""
        base_dt = 0.05  # 20 Hz baseline
        while self.is_running:
            start_time = asyncio.get_event_loop().time()
            interval = 0.2

            stepping = self.step_pending
            if not self.is_paused or stepping:
                # Batch fixed physics ticks while sending only one visual frame.
                speed = self.sim_speed_multiplier
                fast = speed >= 10
                batches = 1
                if fast:
                    map_name = self.simulator.map_name
                    if map_name in CITIES:
                        batches = 3
                    elif map_name == "bronx_fordham":
                        batches = 2
                steps = max(1, round(speed * batches))
                if stepping:
                    steps = round(self.control_interval_sec / base_dt)
                    fast = False
                interval = base_dt * steps / speed
                sim_start = self.simulator.sim_time
                intel_updated = False
                intel_interval = 4.0 if fast and self.active_controller_name in ("FIXED_TIME", "ACTUATED") and self.active_routing_mode == "OFF" else 1.0
                try:
                    for _ in range(steps):
                        track = self.active_controller_name in POLICIES
                        before_tick = {node: (signal.stage, signal.active_green_phase, signal.target_green_phase)
                                       for node, signal in self.simulator.signal_controllers.items()} if track else {}
                        self.simulator.step(dt=base_dt, snapshot=False)
                        timestamp = self.simulator.sim_time
                        for node, previous in before_tick.items():
                            signal = self.simulator.signal_controllers[node]
                            if previous[:2] != (signal.stage, signal.active_green_phase):
                                reason = ("maximum green reached" if signal.stage == "YELLOW" and previous[2] is None
                                          and signal.time_in_green + 1e-8 >= signal.max_green
                                          else "deferred request entered clearance" if signal.stage == "YELLOW"
                                          and previous[2] is not None else None)
                                self.signal_transitions.append({"junction": node, "time": round(timestamp, 2),
                                                                "stage": signal.stage, "green": signal.active_green_phase,
                                                                "reason": reason})
                        if self.demand_finished():
                            self.is_paused = True
                            self.step_pending = False
                            break

                        # Recompute intelligence every 1.0 simulation second
                        if timestamp - self.last_intel_time >= intel_interval or not self.current_intel:
                            self.last_intel_time = timestamp
                            self.current_intel = self._compute_intelligence(self.simulator._get_snapshot(), fast_forward=fast)
                            intel_updated = True

                        # Recompute signal control decisions every control_interval_sec simulation seconds
                        tick = self.simulator.step_count
                        cadence = round(self.control_interval_sec / base_dt)
                        due = (tick - 1) % cadence == 0 if self.active_controller_name in POLICIES else tick % cadence == 0
                        if due or self.last_control_time < 0:
                            self.last_control_time = timestamp
                            try:
                                phases = self.controller.compute_phases(
                                    current_time=timestamp,
                                    simulator=self.simulator,
                                    telemetry=self.current_intel,
                                )
                                signals = self.simulator.signal_controllers
                                before = {node: (signals[node].stage, signals[node].target_green_phase)
                                          for node in phases if node in signals}
                                self.simulator.apply_signal_phases(phases)
                                for node, (stage, previous_target) in before.items():
                                    signal = signals[node]
                                    requested = phases[node]
                                    desired = {0: 0, 1: 1, 2: 1, -1: -1, 4: -1}.get(requested)
                                    applied = signal.stage == "GREEN" and signal.active_green_phase == desired
                                    preserved = (node in self.simulator.managed_junctions and stage != "GREEN"
                                                 and previous_target in (0, 1) and previous_target != desired)
                                    outcome = ("invalid request" if desired is None else "green active" if applied
                                               else "clearance target preserved" if preserved
                                               else "waiting for minimum green" if signal.stage == "GREEN"
                                               else "yellow clearance started" if stage == "GREEN" and signal.stage == "YELLOW"
                                               else "clearance in progress")
                                    safety_remaining = 0.0
                                    if outcome == "waiting for minimum green":
                                        safety_remaining = max(0.0, signal.min_green - signal.time_in_green)
                                    elif signal.stage != "GREEN":
                                        safety_remaining = max(0.0, signal.current_duration - signal.time_in_stage)
                                    if track and stage == "GREEN" and signal.stage == "YELLOW":
                                        self.signal_transitions.append({"junction": node, "time": round(timestamp, 2),
                                                                        "stage": "YELLOW", "green": signal.active_green_phase,
                                                                        "reason": "policy request started clearance"})
                                    self.signal_decisions.append({"junction": node, "time": round(timestamp, 2),
                                                                  "safety_stage_remaining": round(safety_remaining, 2),
                                                                  "requested": requested, "stage": signal.stage,
                                                                  "active_green": signal.active_green_phase,
                                                                  "observation": getattr(self.controller, "observations", {}).get(node),
                                                                  "outcome": outcome})
                                if self.active_controller_name in POLICIES:
                                    self.policy_cycles += 1
                                self._update_controller_telemetry(phases)
                                logger.debug(
                                    "[SIGNAL_CONTROL_TICK] Controller: %s | sim_time: %.2fs | decisions: %d intersections",
                                    self.active_controller_name,
                                    timestamp,
                                    len(phases),
                                )
                            except Exception as e:
                                logger.exception("Error evaluating signal controller %s", self.active_controller_name)
                                if self.active_controller_name in POLICIES:
                                    self.control_error = f"Learned signal inference failed: {e}"
                                    self.is_paused = True
                                    self.last_control_time = self.signal_decisions[-1]["time"] if self.signal_decisions else -1
                                    self._update_controller_telemetry()
                                    self.step_pending = False
                                    break
                            if stepping:
                                self.step_pending = False
                                break

                        # Recompute dynamic routing decisions every routing_interval_sec simulation seconds
                        if (timestamp - self.last_routing_time) >= self.routing_interval_sec or self.last_routing_time < 0:
                            self.last_routing_time = timestamp
                            try:
                                routing_report = self.routing_controller.evaluate_and_reroute(
                                    sim=self.simulator,
                                    current_time=timestamp,
                                    telemetry_map=self.current_intel,
                                )
                                for d in routing_report.get("decisions", []):
                                    if d.get("diverted"):
                                        self.cumulative_reroute_accepted += 1
                                    else:
                                        self.cumulative_reroute_rejected += 1
                                        r_reason = d.get("reason", "Rejected")
                                        self.rejection_reasons_tally[r_reason] += 1
                                    self.recent_routing_decisions.appendleft(d)

                                self._update_routing_telemetry(last_report=routing_report)
                                if routing_report.get("diverted_count", 0) > 0:
                                    logger.info(
                                        "[DYNAMIC_ROUTING_TICK] Mode: %s | sim_time: %.2fs | rerouted: %d/%d eligible vehicles",
                                        self.active_routing_mode,
                                        timestamp,
                                        routing_report.get("diverted_count", 0),
                                        routing_report.get("eligible_candidates_count", 0),
                                    )
                            except Exception as e:
                                logger.error("Error evaluating routing controller %s: %s", self.active_routing_mode, e)

                    if self.clients:
                        elapsed = asyncio.get_event_loop().time() - start_time
                        actual = (self.simulator.sim_time - sim_start) / max(interval, elapsed)
                        self.effective_speed = 0.7 * self.effective_speed + 0.3 * actual if self.effective_speed else actual
                        snapshot = self.simulator._get_snapshot()
                        if not fast or intel_updated:
                            await self.broadcast_snapshot(snapshot)
                        else:
                            await self.broadcast_snapshot(snapshot, include_intel=False)
                except Exception as e:
                    logger.error("Unexpected error in broadcast tick: %s", e, exc_info=True)

            elif self.clients:
                await self.broadcast_snapshot(self.simulator._get_snapshot())

            elapsed = asyncio.get_event_loop().time() - start_time
            sleep_time = max(0.001, interval - elapsed)
            await asyncio.sleep(sleep_time)

    def start(self):
        self.is_running = True
        self._task = asyncio.create_task(self.run_loop())

    def stop(self):
        self.is_running = False
        if self._task:
            self._task.cancel()
