"""Signal and vehicle telemetry for an isolated matched route test."""
from collections import Counter

from urbanflow.core.types import is_ns_approach


class RouteProbe:
    def __init__(self, vehicle_id, roads, detailed=False):
        self.vehicle_id = vehicle_id
        self.approaches = {road.id for road in roads if road.signal_controlled}
        self.nodes = {road.to_node for road in roads if road.id in self.approaches}
        self.visits = []
        self.current = None
        self.entry = None
        self.decisions = []
        self.ticks = [] if detailed else None

    def observe(self, sim, car, road, stopped, leader, blocked, committed):
        time = sim.sim_time
        if self.entry is None:
            self.entry = car.actual_entry_time
        if road.id not in self.approaches:
            self.current = None
            return
        signal = sim.signal_controllers.get(road.to_node)
        if signal is None:
            return
        code = signal.get_signal_code(is_ns_approach(road))
        if self.current is None or self.current["road"] != road.id:
            self.current = {"intersection": road.to_node, "road": road.id,
                            "approach_entry": time, "entry_signal": code, "stop_started": None,
                            "first_green_after_stop": None, "first_permission_after_stop": None,
                            "crossed": None, "seconds": Counter(), "blockers": Counter()}
            self.visits.append(self.current)
        visit = self.current
        if stopped > 0:
            if visit["stop_started"] is None:
                visit["stop_started"] = time
            if not committed and code.lower() != "g":
                reason = "clearance" if signal.stage != "GREEN" else "opposing_green"
            elif blocked:
                reason = "junction_conflict"
            elif leader is not None:
                reason = "nearby_leader"
            else:
                reason = "other"
            visit["seconds"][reason] += stopped
            if leader is not None:
                visit["blockers"][leader] += stopped
        if visit["stop_started"] is not None and code.lower() == "g":
            if visit["first_green_after_stop"] is None:
                visit["first_green_after_stop"] = time
            if not blocked and visit["first_permission_after_stop"] is None:
                visit["first_permission_after_stop"] = time
        if car.current_road_id != road.id:
            visit["crossed"] = time
            self.current = None
        if self.ticks is not None:
            self.ticks.append([time, road.id, code, signal.stage, round(car.speed, 3),
                               round(stopped, 2), leader, blocked, committed])

    def record_decisions(self, sim, phases, before, observations):
        for node in sorted(self.nodes.intersection(phases)):
            signal = sim.signal_controllers[node]
            stage, old_target = before[node]
            requested = phases[node]
            target = {0: 0, 1: 1, 2: 1, 4: -1, -1: -1}.get(requested)
            if signal.stage == "GREEN" and signal.active_green_phase == target:
                outcome = "active"
            elif stage == "GREEN" and signal.stage == "YELLOW":
                outcome = "clearance_started"
            elif signal.stage == "GREEN":
                outcome = "minimum_green"
            elif old_target != target and signal.target_green_phase == old_target:
                outcome = "clearance_target_preserved"
            else:
                outcome = "clearance"
            obs = observations.get(node, {})
            self.decisions.append({"time": sim.sim_time, "intersection": node,
                                   "requested": requested, "outcome": outcome,
                                   "stage": signal.stage, "target": signal.target_green_phase,
                                   "action": obs.get("action"), "probabilities": obs.get("probabilities")})

    def report(self):
        visits = []
        for visit in self.visits:
            row = dict(visit)
            row["seconds"] = {key: round(value, 2) for key, value in visit["seconds"].items()}
            row["blockers"] = {key: round(value, 2) for key, value in visit["blockers"].items()}
            row["stopped"] = round(sum(row["seconds"].values()), 2)
            visits.append(row)
        return {"actual_entry": self.entry, "intersections": visits, "decisions": self.decisions,
                **({"ticks": self.ticks} if self.ticks is not None else {})}
