"""Matched classroom runs using the live simulator and signal controllers."""
import math
import hashlib
from statistics import mean

from urbanflow.core.demand import classroom_demand, write_demand_csv
from urbanflow.core.kinematic_sim import KinematicSimulator
from urbanflow.core.observed_demand import OBSERVED_WINDOWS
from urbanflow.intelligence.signal_control import ActuatedController, FixedTimeController


def classroom_controller(name):
    if name == "FIXED_TIME":
        return FixedTimeController(default_green_ns=25.0, default_green_ew=20.0,
                                   yellow_duration=3.0, all_red_duration=1.5)
    return ActuatedController(min_green=10.0, max_green=60.0, passage_time=3.0,
                              detection_distance_m=50.0)


def compare_classroom(demand=None, scenario="rush_hour", demand_revision=None, on_progress=None,
                      *, map_name="classroom_grid", include_ppo=False, candidate_directory=None):
    return _compare(demand, scenario, demand_revision, on_progress, map_name=map_name,
                    include_ppo=include_ppo, candidate_directory=candidate_directory)


def compare_routing(demand, scenario, demand_revision, on_progress, controller, *, map_name="classroom_grid"):
    return _compare(demand, scenario, demand_revision, on_progress, controller=controller, map_name=map_name)


def _compare(demand, scenario, demand_revision, on_progress, *, controller=None, map_name="classroom_grid",
             include_ppo=False, candidate_directory=None):
    """Hold demand fixed, varying either signals or routing across replays."""
    # The live broadcaster imports the shared controller defaults above.
    from urbanflow.server.websocket_feed import SimulationBroadcaster, CONTROLLER_REGIMES, ROUTING_MODES, LEARNED_REGIMES
    from urbanflow.rl.live import PPOController, load

    sim = KinematicSimulator(map_name=map_name)
    generated = demand is None
    if generated:
        demand = classroom_demand(sim, scenario)
    cutoff, clearance = (86700, 300) if scenario == "uploaded" else (4200, 300) if scenario in OBSERVED_WINDOWS else (1800, 120)
    duration = max(240, min(cutoff, math.ceil(demand[-1].time + clearance) if demand else 240))
    dt = 0.05
    results = {}
    replay = SimulationBroadcaster(sim)
    variants = ROUTING_MODES if controller else CONTROLLER_REGIMES
    candidate = None
    if candidate_directory is not None and controller is None:
        candidate = PPOController(sim, "PPO_NEIGHBOR", directory=candidate_directory)
        variants = {**variants, "PPO_NEIGHBOR": LEARNED_REGIMES["PPO_NEIGHBOR"]}
    if include_ppo and not controller:
        variants = variants.copy()
        for key, info in LEARNED_REGIMES.items():
            try:
                load(sim, key)
            except ValueError:
                continue
            variants[key] = ({**info, "name": f"{info['name']} (experimental)"}
                             if map_name != "classroom_grid" else info)
    for number, (name, info) in enumerate(variants.items(), 1):
        sim.set_demand(demand, scenario)
        if candidate is not None and name == "PPO_NEIGHBOR":
            replay.controller = candidate
            replay.active_controller_name = name
        else:
            replay.set_controller(controller or name)
        replay.set_routing_mode(name if controller else "OFF")
        replay.reset_run()
        sampled_speeds, queues = [], []
        upstream_queue = []
        spillback_seconds = 0
        phase_changes = 0
        reroutes = 0
        last_routing_time = -1.0
        previous_signals = None
        total_ticks = round(duration / dt)
        for tick in range(1, total_ticks + 1):
            if on_progress and (tick == 1 or tick % 200 == 0):
                on_progress({"demand_revision": demand_revision, "controller": info["name"],
                             "controller_number": number, "controller_count": len(variants),
                             "fraction": (number - 1 + tick / total_ticks) / len(variants)})
            needs_signal = ((tick - 1) % round(replay.control_interval_sec / dt) == 0
                            if replay.active_controller_name in LEARNED_REGIMES
                            else tick == 1 or tick % 20 == 0)
            needs_routing = controller and name != "OFF" and (
                last_routing_time < 0 or sim.sim_time + dt - last_routing_time >= replay.routing_interval_sec
            )
            snapshot = sim.step(dt, snapshot=needs_signal or needs_routing or tick % 20 == 0)
            needs_telemetry = needs_routing or (needs_signal and replay.active_controller_name in
                                                ("MAX_PRESSURE", "COOPERATIVE_PREDICTIVE"))
            telemetry = replay._compute_intelligence(snapshot, fast_forward=True) if needs_telemetry else None
            if needs_signal:
                sim.apply_signal_phases(replay.controller.compute_phases(snapshot.timestamp,
                                                                         simulator=sim, telemetry=telemetry))
            if needs_routing:
                last_routing_time = snapshot.timestamp
                routing_report = replay.routing_controller.evaluate_and_reroute(
                    sim=sim, current_time=snapshot.timestamp, telemetry_map=telemetry or {}
                )
                reroutes += routing_report.get("diverted_count", 0)
            if tick % 20:
                continue
            phase_indexes = [signal.phase_index for signal in snapshot.signals]
            if previous_signals is not None:
                phase_changes += sum(a != b for a, b in zip(previous_signals, phase_indexes))
            previous_signals = phase_indexes
            sampled_speeds.extend(v.speed for v in sim.vehicles.values())
            stopped = [v for v in sim.vehicles.values() if v.speed < 0.5
                       and sim.roads[v.current_road_id].to_node in sim.intersections]
            queues.append(len(stopped))
            upstream_queue.append(sum(sim.roads[v.current_road_id].to_node == "INT_NW" for v in stopped))
            if any(v.distance_on_road < 10 and sim.roads[v.current_road_id].from_node in sim.intersections
                   for v in stopped):
                spillback_seconds += 1
        trips = sim.completed_trips
        scheduled_by_id = {record.id: record for record in demand}
        delays = []
        for trip in trips:
            record = scheduled_by_id[trip["vehicle_id"]]
            free_flow = sum(sim.roads[road].length / min(record.speed, sim.roads[road].speed_limit)
                            for road in record.route)
            delays.append(max(0, trip["system_travel_time_sec"] - free_flow))
        delays.sort()
        counts = sim.demand_counts()
        results[name] = {
            **counts,
            "future_departures": counts["scheduled"] - counts["due"],
            "removed_trips": counts["entered"] - counts["active"] - counts["completed"],
            "completed_delay_mean_sec": round(mean(delays), 2) if delays else None,
            "completed_delay_p95_sec": round(delays[math.ceil(.95 * len(delays)) - 1], 2) if delays else None,
            "name": info["name"],
            "description": info["description"],
            "predictive_status": (getattr(replay.routing_controller, "predictive_status", None)
                                  if controller and name == "PREDICTIVE"
                                  else getattr(replay.controller, "predictive_status", None) if not controller else None),
            "reroutes": reroutes,
            "speed_kmh": round(mean(sampled_speeds) * 3.6, 2) if sampled_speeds else None,
            "travel_time_sec": round(mean(t["network_travel_time_sec"] for t in trips), 2) if trips else None,
            "wait_sec": round(mean(t["wait_time_sec"] for t in trips), 2) if trips else None,
            "mean_queue": round(mean(queues), 2),
            "peak_queue": max(queues),
            "queue_trace": queues,
            "upstream_queue_trace": upstream_queue,
            "throughput_per_hour": round(len(trips) * 3600 / duration),
            "spillback_seconds": spillback_seconds,
            "signal_stage_changes": phase_changes,
        }
    ordered = sorted(results, key=lambda name: (
        -results[name]["completed"], results[name]["mean_queue"],
        results[name]["wait_sec"] if results[name]["wait_sec"] is not None else math.inf,
    ))
    first = results[ordered[0]]
    scores = lambda result: (result["completed"], result["mean_queue"], result["wait_sec"])
    leaders = [name for name in ordered if scores(results[name]) == scores(first)]
    return {
        "engine": "UrbanFlow kinematic simulator",
        "demand_sha256": hashlib.sha256(write_demand_csv(demand).encode("utf-8")).hexdigest(),
        "queue_sample_seconds": 1,
        "routing": "varied" if controller else "OFF",
        "restriction": {"road": "road_nw_ne", "start": 20, "end": 110, "speed_mps": 3}
        if scenario == "incident" and map_name == "classroom_grid" else None,
        "map_name": map_name,
        "kind": "routing" if controller else "controller",
        "controller": controller,
        "controller_name": CONTROLLER_REGIMES[controller]["name"] if controller else None,
        "duration_seconds": duration,
        "seed": 42 if generated or scenario != "uploaded" else None,
        "scenario": scenario,
        "demand_revision": demand_revision,
        "scheduled": len(demand),
        "evaluated_departures": sum(record.time <= duration for record in demand),
        "recommended": leaders[0] if len(leaders) == 1 and first["completed"] else None,
        "ranked": ordered,
        "results": results,
    }
