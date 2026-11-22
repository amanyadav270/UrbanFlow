"""Estimate travel time using observed motion, queues and signal phases."""
from collections import defaultdict
from heapq import heappop, heappush
from statistics import mean

import networkx as nx

from urbanflow.core.demand import FLEET
from urbanflow.core.types import is_ns_approach


def signal_wait(signal, phase, arrival):
    """Estimate the next green window from the current phase and nominal cycle."""
    green = max(signal.min_green, signal.default_green)
    clearance = signal.yellow_duration + signal.all_red_duration
    cycle = 2 * (green + clearance)
    if signal.stage == "GREEN":
        pending = signal.target_green_phase not in (None, signal.active_green_phase)
        limit = signal.min_green if pending else green
        remaining = max(0.0, limit - signal.time_in_green)
        if signal.active_green_phase == phase:
            if arrival < remaining:
                return 0.0
            opening = remaining + green + 2 * clearance
        else:
            opening = remaining + clearance
    else:
        next_phase = signal.target_green_phase
        if next_phase not in (0, 1):
            next_phase = 1 - signal.active_green_phase
        if signal.stage == "YELLOW":
            remaining = max(0.0, signal.yellow_duration - signal.time_in_stage) + signal.all_red_duration
        else:
            remaining = max(0.0, signal.all_red_duration - signal.time_in_stage)
        opening = remaining + (green + clearance if phase != next_phase else 0.0)
    if arrival < opening:
        return opening - arrival
    offset = (arrival - opening) % cycle
    return 0.0 if offset < green else cycle - offset


def plan_route(sim, source, destination, incidents):
    if not isinstance(source, str) or not isinstance(destination, str):
        raise ValueError("Choose an origin and destination.")
    if source not in sim.graph or destination not in sim.graph or source == destination:
        raise ValueError("Choose two different locations on the selected network.")
    graph = nx.subgraph_view(sim.graph, filter_edge=lambda a, b:
                             sim.route_fits_vehicle((sim.graph[a][b]["id"],), FLEET["sedan"][0]))

    speeds, queues = defaultdict(list), defaultdict(int)
    for vehicle in sim.vehicles.values():
        road = sim.roads[vehicle.current_road_id]
        if vehicle.speed >= 1:
            speeds[road.id].append(vehicle.speed)
        elif road.length - vehicle.distance_on_road < 55:
            queues[road.id] += 1

    link_times = {}

    def edge_cost(edge, elapsed):
        road = sim.roads[edge["id"]]
        if road.id not in link_times:
            observed = mean(speeds[road.id]) if speeds[road.id] else road.speed_limit
            incident = incidents.get(road.id, {})
            severity = incident.get("severity", 0) if incident.get("end_time", 0) > sim.sim_time else 0
            limit = max(2.0, road.speed_limit * (1 - severity))
            link_times[road.id] = edge["length"] / max(1.0, min(observed, limit))
        drive = link_times[road.id]
        signal = sim.signal_controllers.get(road.to_node) if road.to_node != destination else None
        wait = 0.0
        if signal and road.signal_controlled:
            phase = int(not is_ns_approach(road))
            wait = signal_wait(signal, phase, elapsed + drive)
        queue = 2.0 * queues[road.id] / max(1, road.lanes)
        return drive, wait, queue

    try:
        shortest = nx.shortest_path(graph, source, destination, weight="length")
    except nx.NetworkXNoPath as exc:
        raise ValueError("No drivable route connects these locations.") from exc

    # Each approach's predicted service time is nondecreasing with arrival time.
    times, previous = {source: 0.0}, {}
    frontier = [(0.0, source)]
    while frontier:
        elapsed, node = heappop(frontier)
        if elapsed != times[node]:
            continue
        if node == destination:
            break
        for neighbor, edge in graph[node].items():
            eta = elapsed + sum(edge_cost(edge, elapsed))
            if eta < times.get(neighbor, float("inf")):
                times[neighbor], previous[neighbor] = eta, node
                heappush(frontier, (eta, neighbor))

    fastest = [destination]
    while fastest[-1] != source:
        fastest.append(previous[fastest[-1]])

    def summarize(nodes):
        elapsed = distance = signal_seconds = queue_seconds = 0.0
        roads = []
        for a, b in zip(nodes, nodes[1:]):
            edge = sim.graph[a][b]
            drive, wait, queue = edge_cost(edge, elapsed)
            elapsed += drive + wait + queue
            distance += edge["length"]
            signal_seconds += wait
            queue_seconds += queue
            roads.append(edge["id"])
        return {"road_ids": roads, "distance_m": round(distance), "estimated_seconds": round(elapsed),
                "signal_wait_seconds": round(signal_seconds), "queue_delay_seconds": round(queue_seconds),
                "average_speed_kmh": round(3.6 * distance / elapsed, 1)}

    return {"map_name": sim.map_name, **summarize(list(reversed(fastest))), "shortest": summarize(shortest)}


def routing_demo(sim):
    """Find a reproducible example on the current network, preferring a faster detour."""
    fallback = None
    tested = set()
    for route in (sim.scheduled_routes or sim.route_pool)[:12]:
        source, destination = sim.roads[route[0]].from_node, sim.roads[route[-1]].to_node
        endpoints = source, destination
        if endpoints in tested:
            continue
        tested.add(endpoints)
        try:
            baseline = plan_route(sim, source, destination, {})
        except ValueError:
            continue
        road = baseline["shortest"]["road_ids"][len(baseline["shortest"]["road_ids"]) // 2]
        incident = {"severity": .88, "end_time": sim.sim_time + 180.0}
        result = plan_route(sim, source, destination, {road: incident})
        demo = {"map_name": sim.map_name, "from": source, "to": destination, "road": road, **incident}
        if road not in result["road_ids"] and result["estimated_seconds"] < result["shortest"]["estimated_seconds"]:
            return demo
        if fallback is None:
            fallback = demo
    if fallback is None:
        raise ValueError("This network has no drivable route for a routing demo.")
    return fallback
