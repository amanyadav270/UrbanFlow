"""Matched controller replays for a dedicated sedan on a fixed route."""
from collections import deque

from urbanflow.core.demand import Departure, FLEET
from urbanflow.core.kinematic_sim import KinematicSimulator
from urbanflow.core.route_probe import RouteProbe

PROBE = "route-experiment-car"
DT = .05
DEPARTURE = 0
MIN_TRIP_LIMIT = 300


def validate_route(sim, roads, source, destination):
    if not isinstance(roads, list) or not roads or any(
        not isinstance(road, str) or road not in sim.roads or road.startswith("CONN__") for road in roads
    ):
        raise ValueError("Find a route before starting the test drive.")
    links = [sim.roads[road] for road in roads]
    if (links[0].from_node != source or links[-1].to_node != destination
            or any(a.to_node != b.from_node for a, b in zip(links, links[1:]))):
        raise ValueError("The route no longer matches the selected endpoints. Find it again.")
    if len(roads) > len(sim.graph) or not sim.route_fits_vehicle(roads, FLEET["sedan"][0]):
        raise ValueError("Choose a route with enough space for the test car.")


def run_route_experiment(map_name, scenario, demand, roads, reference, incidents, on_reference=None, cancelled=None,
                         on_samples=None, controllers=None, departure_seconds=DEPARTURE, detailed=False, network_horizon=None):
    from urbanflow.rl.live import POLICIES
    from urbanflow.server.websocket_feed import (
        CONTROLLER_REGIMES, LEARNED_REGIMES, SimulationBroadcaster,
    )

    sim = KinematicSimulator(map_name)
    replay_baselines = ("FIXED_TIME", "ACTUATED", "MAX_PRESSURE", "COOPERATIVE_PREDICTIVE")
    free_flow = sum(sim.roads[road].length / min(sim.roads[road].speed_limit, FLEET["sedan"][1]) for road in roads)
    limit = max(MIN_TRIP_LIMIT, min(900, round(3 * free_flow + 180)))
    variants = dict(CONTROLLER_REGIMES)
    for key in POLICIES:
        info = LEARNED_REGIMES[key]
        variants[key] = ({**info, "name": f"{info['name']} (experimental)"}
                         if map_name != "classroom_grid" else info)
    if reference not in variants:
        raise ValueError("The selected reference controller is unavailable.")
    probe_id = PROBE
    occupied_ids = {record.id for record in demand}
    while probe_id in occupied_ids:
        probe_id += "-test"
    probe = Departure(probe_id, departure_seconds, tuple(roads), 0, "sedan", FLEET["sedan"][1])
    schedule = sorted((probe, *demand), key=lambda record: record.time)
    replay = SimulationBroadcaster(sim)
    results, trace, traffic = [], [], []
    sent_trace = sent_traffic = 0

    def publish_samples():
        nonlocal sent_trace, sent_traffic
        if on_samples and (len(trace) > sent_trace or len(traffic) > sent_traffic):
            on_samples({"trace": trace[sent_trace:], "traffic": traffic[sent_traffic:]})
            sent_trace, sent_traffic = len(trace), len(traffic)

    origin = sim.roads[roads[0]].points[0]
    for key in controllers or dict.fromkeys((reference, "PPO_NEIGHBOR", *replay_baselines)):
        recording = key == reference
        sim.set_demand(schedule, scenario)
        try:
            replay.set_controller(key)
        except ValueError:
            if key == reference or controllers:
                raise
            continue
        monitor = sim.route_probe = RouteProbe(probe_id, [sim.roads[road] for road in roads], detailed)
        name = variants[key]["name"]
        checkpoint = getattr(replay.controller, "checkpoint", None)
        if key == "PPO_NEIGHBOR" and map_name == "classroom_grid":
            if checkpoint == "data/classroom_ppo_directional_v2/neighbor/weights.pt":
                name = "Multi-Agent RL Model · Directional v2"
            elif checkpoint == "data/classroom_ppo/neighbor/weights.pt":
                name = "Multi-Agent RL Model · Original neighbor"
        if key == "COOPERATIVE_PREDICTIVE":
            name = "Cooperative Predictive"
        replay.reset_run()
        for road, restriction in incidents.items():
            sim.inject_incident(road, restriction["severity"], restriction["end_time"])
            replay.active_incidents[road] = restriction
        cadence = round(replay.control_interval_sec / DT)
        arrival = None
        checked_arrivals = 0
        network = None
        last_position = deque(maxlen=1)
        for tick in range(1, round((departure_seconds + limit) / DT) + 1):
            if cancelled is not None and cancelled():
                return None
            control_due = tick == 1 or (tick - 1 if key in POLICIES else tick) % cadence == 0
            snapshot = sim.step(DT, snapshot=control_due and key in ("MAX_PRESSURE", "COOPERATIVE_PREDICTIVE"))
            if control_due:
                telemetry = replay._compute_intelligence(snapshot, fast_forward=True) if snapshot else None
                phases = replay.controller.compute_phases(sim.sim_time, sim, telemetry)
                before = {node: (sim.signal_controllers[node].stage, sim.signal_controllers[node].target_green_phase)
                          for node in monitor.nodes.intersection(phases)}
                sim.apply_signal_phases(phases)
                if arrival is None:
                    monitor.record_decisions(sim, phases, before, getattr(replay.controller, "observations", {}))
            car = sim.vehicles.get(probe_id)
            if network_horizon is not None and network is None and sim.sim_time >= network_horizon:
                trips = [trip for trip in sim.completed_trips if trip["vehicle_id"] != probe_id]
                vehicles = [v for v in sim.vehicles.values() if v.id != probe_id]
                network = {"horizon": network_horizon, "completed": len(trips),
                           "stopped_vehicle_seconds": round(sum(t["wait_time_sec"] for t in trips)
                                                            + sum(v.wait_time for v in vehicles), 2)}
            if recording and arrival is None and sim.sim_time >= departure_seconds and tick % 4 == 0:
                x, y = (car.x, car.y) if car is not None else (origin.x, origin.y)
                nearby = [(v, (v.x - x) ** 2 + (v.y - y) ** 2)
                          for v in sim.vehicles.values() if v.id != probe_id]
                closest = sorted((item for item in nearby if item[1] <= 140 ** 2),
                                 key=lambda pair: pair[1])[:36]
                traffic.append([round(sim.sim_time - departure_seconds, 2),
                                [[v.id, round(v.x, 2), round(v.y, 2), round(v.heading, 1),
                                  round(v.speed, 2), v.type, v.color, v.current_road_id] for v, _ in closest],
                                [[node, f"NS:{signal.get_signal_code(True)} EW:{signal.get_signal_code(False)}"]
                                 for node, signal in sim.signal_controllers.items()]])
            if car is not None and recording:
                position = [round(sim.sim_time - departure_seconds, 2), round(car.x, 2),
                            round(car.y, 2), round(car.heading, 1)]
                if not last_position or tick % 4 == 0:
                    trace.append(position)
                last_position.append(position)
            elif car is None and len(sim.completed_trips) > checked_arrivals:
                arrival = arrival or next((trip for trip in sim.completed_trips[checked_arrivals:]
                                if trip["vehicle_id"] == probe_id), None)
                checked_arrivals = len(sim.completed_trips)
            if arrival is not None and (network_horizon is None or network is not None):
                break
            if key == reference and (tick == 1 or tick % 16 == 0):
                publish_samples()
        if recording and trace and last_position[0] != trace[-1]:
            trace.append(last_position[0])
        if key == reference:
            publish_samples()
        results.append({"controller": name, "id": key, "checkpoint": checkpoint,
                        "action_selection": "argmax" if checkpoint else None,
                        "departure": departure_seconds, "arrival": arrival["arrival_time"] if arrival else None,
                        "network": network, **monitor.report(),
                        "seconds": round(arrival["system_travel_time_sec"], 2) if arrival else None,
                        "entry_wait": round(arrival["entry_delay_sec"], 2) if arrival else None,
                        "stopped": round(arrival["wait_time_sec"], 2) if arrival else None})
        if key == reference and on_reference is not None:
            on_reference({"trace": trace, "traffic": traffic, "results": results[:1],
                          "timeout_seconds": limit, "background_trips": len(demand)})
    return {"trace": trace, "traffic": traffic, "results": results, "timeout_seconds": limit,
            "departure_seconds": departure_seconds, "background_trips": len(demand),
            "scenario": scenario, "map_name": map_name, "route": list(roads), "incidents": incidents}
