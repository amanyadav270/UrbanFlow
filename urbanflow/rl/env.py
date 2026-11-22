"""Classroom observations shared by training and live inference."""
import hashlib
import json

import numpy as np

from urbanflow.core.demand import classroom_demand
from urbanflow.core.kinematic_sim import KinematicSimulator

DT = .05
DECISION_SECONDS = 5
FEATURES = ("ns_queue", "ew_queue", "ns_out_load", "ew_out_load",
            "ns_green", "ew_green", "elapsed_green", "ns_in_present",
            "ew_in_present", "ns_out_present", "ew_out_present")
NEIGHBOR_FEATURES = ("present", "ns_queue", "ew_queue", "ns_green", "ew_green")
SCHEMA = FEATURES + tuple(f"neighbor_{i}_{name}" for i in range(4) for name in NEIGHBOR_FEATURES)
RESTRICTION = {"road_id": "road_nw_ne", "start": 20, "end": 110, "speed_mps": 3}


def fingerprint(sim):
    topology = sim.get_network_topology()
    topology.pop("route_defaults", None)
    for junction in topology["intersections"]:
        junction.pop("current_phase", None)
    return hashlib.sha256(json.dumps(topology, sort_keys=True).encode()).hexdigest()


def observe(sim, neighbor_aware, *, queue_counts=None):
    """Normalize queues/occupancy by 7.5 m storage; sort neighbors by junction ID."""
    agents = sorted(sim.signal_controllers)
    vehicles = {}
    for vehicle in sim.vehicles.values():
        vehicles.setdefault(vehicle.current_road_id, []).append(vehicle)
    local, adjacent = {}, {}
    approaches = {node: [] for node in agents}
    for road in sim.roads.values():
        if road.id.startswith("CONN__") or len(road.points) < 2:
            continue
        for node in {road.from_node, road.to_node}:
            if node in approaches:
                approaches[node].append(road)
    for node in agents:
        queues, loads = [0., 0.], [0., 0.]
        stopped = [0, 0]
        inbound, outbound = [0, 0], [0, 0]
        adjacent[node] = set()
        for road in approaches[node]:
            first, last = road.points[0], road.points[-1]
            axis = int(abs(last.x - first.x) >= abs(last.y - first.y))
            traffic = vehicles.get(road.id, ())
            storage = max(1, road.length * road.lanes / 7.5)
            if road.to_node == node and road.signal_controlled:
                approach = (int(not road.approach_ns) if road.approach_ns is not None else axis)
                inbound[approach] = 1
                count = sum(v.speed < .5 for v in traffic)
                stopped[approach] += count
                queues[approach] = max(queues[approach], count / storage)
            if road.from_node == node:
                outbound[axis] = 1
                loads[axis] = max(loads[axis], len(traffic) / storage)
                if road.to_node in approaches:
                    adjacent[node].add(road.to_node)
        if queue_counts is not None:
            queue_counts[node] = {"ns": stopped[0], "ew": stopped[1]}
        signal = sim.signal_controllers[node]
        local[node] = queues + loads + [
            signal.stage == "GREEN" and signal.active_green_phase == phase for phase in (0, 1)
        ] + [signal.time_in_green / signal.max_green] + inbound + outbound
    rows = []
    for node in agents:
        neighbor_values = []
        if neighbor_aware:
            for other in sorted(adjacent[node])[:4]:
                neighbor_values.extend((1, *local[other][:2], *local[other][4:6]))
        rows.append(local[node] + neighbor_values + [0] * (20 - len(neighbor_values)))
    return agents, np.clip(np.asarray(rows, dtype=np.float32), 0, 1)


def requests(sim, agents, actions):
    """Action 0 requests the current green; action 1 requests the opposite green."""
    return {node: sim.signal_controllers[node].active_green_phase ^ int(action)
            for node, action in zip(agents, actions)}


class ClassroomEnv:
    def reset(self, seed, scenario, neighbor_aware, duration, map_name="classroom_grid"):
        self.sim = KinematicSimulator(map_name)
        self.sim.set_demand(classroom_demand(self.sim, scenario, seed), scenario)
        self.neighbor_aware, self.duration = neighbor_aware, duration
        self.sim.step(DT, snapshot=False)
        self.agents, obs = observe(self.sim, neighbor_aware)
        return obs

    def step(self, actions):
        self.sim.apply_signal_phases(requests(self.sim, self.agents, actions))
        for _ in range(round(DECISION_SECONDS / DT)):
            self.sim.step(DT, snapshot=False)
        _, obs = observe(self.sim, self.neighbor_aware)
        reward = -obs[:, :2].mean(axis=1) - .2 * (obs[:, 2:4].max(axis=1) >= .75) - .02 * np.asarray(actions)
        return obs, reward, self.sim.sim_time >= self.duration
