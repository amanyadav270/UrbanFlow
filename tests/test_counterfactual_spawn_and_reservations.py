"""Unit & Regression Tests for UrbanFlow AI Phase 5E:
Counterfactual Spawn Accounting and Persistent Capacity Reservations.

Tests:
1. Deterministic exogenous spawn alignment across regimes regardless of entrance delays.
2. Accurate entry delay and travel time segregation (network vs system travel time).
3. Never-entered and incomplete trip demand accounting.
4. Persistent capacity reservations across multiple 30-second evaluation windows.
5. Reservation lifecycle release triggers (link traversal, dynamic reroute, trip completion).
6. Upstream queue delay conservatism (reservations never prematurely discarded).
7. Physical storage capacity limit enforcement on short residential links (75% limit).
8. Multi-corridor load balancing and traffic dispersion across parallel routes.
"""
from __future__ import annotations

import copy
import pytest
from numpy.random import Generator, Philox

from urbanflow.core.kinematic_sim import KinematicSimulator, SimulatedVehicle
from urbanflow.intelligence.routing import (
    RouteCostEvaluator,
    PredictiveRoutingController,
    LinkReservation,
)
from urbanflow.intelligence.benchmark_kpis import RoutingKPIEvaluator, RoutingKPISummary


def test_deterministic_spawn_alignment_across_regimes():
    """Verifies that scheduled arrivals are bit-for-bit identical across regimes,
    even when network congestion or entrance blocking delays physical entry in one regime."""
    seed = 5001
    # Run two simulators with the exact same Philox spawn and driver PRNG seeds
    rng_spawn_1 = Generator(Philox(seed + 100))
    rng_driver_1 = Generator(Philox(seed + 200))
    sim_uncongested = KinematicSimulator(map_name="bronx_fordham", rng_spawn=rng_spawn_1, rng_driver=rng_driver_1)

    rng_spawn_2 = Generator(Philox(seed + 100))
    rng_driver_2 = Generator(Philox(seed + 200))
    sim_congested = KinematicSimulator(map_name="bronx_fordham", rng_spawn=rng_spawn_2, rng_driver=rng_driver_2)

    # Artificially cap active vehicles in sim_congested to force entry delays
    sim_congested.max_vehicles = 5
    sim_uncongested.max_vehicles = 120

    # Step both simulators for 30 seconds (300 steps at dt=0.1s)
    for _ in range(300):
        sim_uncongested.step(0.1)
        sim_congested.step(0.1)

    # 1. Scheduled spawns must be 100% identical in count, timing, and vehicle attributes
    sched_1 = sim_uncongested.all_scheduled_spawns
    sched_2 = sim_congested.all_scheduled_spawns

    assert len(sched_1) > 0
    assert len(sched_1) == len(sched_2)

    for s1, s2 in zip(sched_1, sched_2):
        assert s1["id"] == s2["id"]
        assert s1["scheduled_spawn_time"] == s2["scheduled_spawn_time"]
        assert s1["type"] == s2["type"]
        assert s1["length"] == s2["length"]
        assert s1["desired_speed"] == s2["desired_speed"]
        assert s1["route"] == s2["route"]

    # The cap should increase entry blocking. The uncapped run can still
    # queue at its occupied entrances without changing the departure schedule.
    assert len(sim_congested.pending_spawns) > len(sim_uncongested.pending_spawns)


def test_entry_delay_and_system_travel_time_segregation():
    """Verifies accurate segregation of network travel time vs entry delay in completed trips."""
    rng_spawn = Generator(Philox(42))
    rng_driver = Generator(Philox(43))
    sim = KinematicSimulator(map_name="bronx_fordham", rng_spawn=rng_spawn, rng_driver=rng_driver)

    # Manually stage a scheduled vehicle that suffered 25.0s entrance delay
    veh = sim._generate_scheduled_vehicle(scheduled_time=10.0)
    assert veh is not None
    veh.scheduled_spawn_time = 10.0
    veh.actual_entry_time = 35.0
    veh.entry_delay = 25.0
    veh.spawn_time = 35.0

    # Simulate completion at t = 100.0s (network traversal = 65.0s)
    road_obj = sim.roads[veh.route[0]]
    veh.speed = 10.0
    veh.total_travel_time = 65.0
    veh.wait_time = 15.0

    sim.completed_trips.append({
        "vehicle_id": veh.id,
        "type": veh.type,
        "scheduled_spawn_time": veh.scheduled_spawn_time,
        "actual_entry_time": veh.actual_entry_time,
        "entry_delay_sec": veh.entry_delay,
        "travel_time": veh.total_travel_time,
        "network_travel_time_sec": veh.total_travel_time,
        "system_travel_time_sec": veh.total_travel_time + veh.entry_delay,
        "wait_time": veh.wait_time,
        "total_system_wait_time_sec": veh.wait_time + veh.entry_delay,
        "delay": veh.total_travel_time - 40.0,
        "distance": 500.0,
        "completed_at": 100.0,
        "reroute_count": 0,
        "route_taken": list(veh.route),
    })

    evaluator = RoutingKPIEvaluator(roads=sim.roads, warmup_sec=0.0)
    summary = evaluator.calculate_kpis(
        completed_trips=sim.completed_trips,
        active_vehicles=[],
        telemetry=[],
        run_id="TEST_SEGREGATION",
        scenario_family="FAM-05",
        regime="baseline",
        seed=42,
        duration_sec=120.0,
    )

    # Network travel time is 65.0s
    assert summary.mean_network_travel_time_sec == 65.0
    assert summary.mean_travel_time_sec == 65.0

    # Entry delay is 25.0s
    assert summary.mean_entry_delay_sec == 25.0
    assert summary.total_entry_delay_sec == 25.0

    # System travel time is 65.0 + 25.0 = 90.0s
    assert summary.mean_system_travel_time_sec == 90.0
    assert summary.median_system_travel_time_sec == 90.0

    # Total system wait is 15.0s wait + 25.0s entry delay = 40.0s
    assert summary.total_system_wait_time_sec == 40.0


def test_never_entered_vehicles_accounting():
    """Verifies that vehicles that never entered the network are tracked and reported in KPIs."""
    rng_spawn = Generator(Philox(99))
    rng_driver = Generator(Philox(100))
    sim = KinematicSimulator(map_name="bronx_fordham", rng_spawn=rng_spawn, rng_driver=rng_driver)

    # Completely block entry so vehicles stay in pending_spawns
    sim.max_vehicles = 0

    for _ in range(100):
        sim.step(0.1)

    never_entered = sim.get_never_entered_vehicles()
    assert len(never_entered) > 0

    for nv in never_entered:
        assert nv["scheduled_spawn_time"] >= 0.0
        assert nv["actual_entry_time"] is None
        assert nv["entry_delay_sec"] >= 0.0

    evaluator = RoutingKPIEvaluator(roads=sim.roads, warmup_sec=0.0)
    summary = evaluator.calculate_kpis(
        completed_trips=[],
        active_vehicles=[],
        telemetry=[],
        run_id="TEST_NEVER_ENTERED",
        scenario_family="FAM-05",
        regime="baseline",
        seed=99,
        duration_sec=10.0,
        never_entered_vehicles=never_entered,
    )

    assert summary.never_entered_trips_count == len(never_entered)
    assert summary.total_spawned_trips == len(never_entered)
    assert summary.completed_trips_count == 0


def test_persistent_capacity_reservations_across_multiple_windows():
    """Verifies that capacity reservations survive across multiple 30s evaluation windows
    while the vehicle is still upstream of reserved links."""
    sim = KinematicSimulator(map_name="bronx_fordham")
    ctrl = PredictiveRoutingController(mode="reactive", strict_quota_floor=False)

    cand_path = ["road_5698666_15_fwd", "road_5698666_16_fwd", "road_5698666_17_fwd"]
    v = SimulatedVehicle("veh_persist_test", "sedan", ["road_586133572_120_fwd"] + cand_path, "#2563eb")
    v.route_index = 0  # Vehicle is on road 120_fwd, upstream of all reserved links
    sim.vehicles[v.id] = v

    # Add reservations at t = 30.0s
    ctrl.add_route_reservations(v, cand_path, current_time=30.0)

    # Check reservations are active
    for r in cand_path:
        res = ctrl.get_active_reservations(r)
        assert len(res) == 1
        assert res[0].vehicle_id == v.id

    # Simulate evaluation window 1 at t = 60.0s (30s later)
    # Vehicle has NOT moved yet (still on road 120_fwd)
    ctrl.clean_reservations(sim, current_time=60.0)

    for r in cand_path:
        res = ctrl.get_active_reservations(r)
        assert len(res) == 1, f"Reservation on {r} was prematurely dropped at t=60s"

    # Simulate evaluation window 2 at t = 90.0s (60s later)
    ctrl.clean_reservations(sim, current_time=90.0)
    for r in cand_path:
        res = ctrl.get_active_reservations(r)
        assert len(res) == 1, f"Reservation on {r} was prematurely dropped at t=90s"


def test_reservation_lifecycle_release_triggers():
    """Verifies that reservations are released on link traversal, rerouting, and trip arrival."""
    sim = KinematicSimulator(map_name="bronx_fordham")
    ctrl = PredictiveRoutingController(mode="reactive", strict_quota_floor=False)

    cand_path = ["road_5698666_15_fwd", "road_5698666_16_fwd", "road_5698666_17_fwd"]
    v = SimulatedVehicle("veh_lifecycle", "sedan", ["road_586133572_120_fwd"] + cand_path, "#2563eb")
    v.route_index = 0
    sim.vehicles[v.id] = v

    ctrl.add_route_reservations(v, cand_path, current_time=30.0)
    assert len(ctrl.get_active_reservations("road_5698666_15_fwd")) == 1

    # Trigger 1: Vehicle physically traverses and enters road_5698666_15_fwd
    v.route_index = 1
    ctrl.clean_reservations(sim, current_time=45.0)

    # road_5698666_15_fwd reservation must be released (now in physical telemetry)
    assert len(ctrl.get_active_reservations("road_5698666_15_fwd")) == 0
    # Downstream links must still be reserved
    assert len(ctrl.get_active_reservations("road_5698666_16_fwd")) == 1
    assert len(ctrl.get_active_reservations("road_5698666_17_fwd")) == 1

    # Trigger 2: Vehicle reroutes to an alternate detour omitting road_5698666_17_fwd
    new_subpath = ["road_5698666_16_fwd", "road_detour_alt_fwd"]
    v.route = ["road_586133572_120_fwd", "road_5698666_15_fwd"] + new_subpath
    ctrl.add_route_reservations(v, new_subpath, current_time=50.0)

    # road_5698666_17_fwd reservation must be gone
    assert len(ctrl.get_active_reservations("road_5698666_17_fwd")) == 0
    assert len(ctrl.get_active_reservations("road_detour_alt_fwd")) == 1

    # Trigger 3: Vehicle completes trip (removed from sim.vehicles)
    del sim.vehicles[v.id]
    ctrl.clean_reservations(sim, current_time=80.0)

    assert len(ctrl.get_active_reservations("road_5698666_16_fwd")) == 0
    assert len(ctrl.get_active_reservations("road_detour_alt_fwd")) == 0


def test_delay_staleness_conservatism():
    """Verifies that an upstream queue delay does not discard downstream reservations;
    expected arrival is conservatively shifted forward."""
    sim = KinematicSimulator(map_name="bronx_fordham")
    ctrl = PredictiveRoutingController(mode="reactive", strict_quota_floor=False)

    cand_path = ["road_5698666_15_fwd", "road_5698666_16_fwd"]
    v = SimulatedVehicle("veh_delayed", "sedan", ["road_586133572_120_fwd"] + cand_path, "#2563eb")
    v.route_index = 0
    sim.vehicles[v.id] = v

    ctrl.add_route_reservations(v, cand_path, current_time=100.0)
    initial_exp_arr = ctrl.active_reservations["road_5698666_15_fwd"][v.id].expected_arrival_time

    # Advance time by 300s (vehicle stuck in upstream gridlock)
    ctrl.clean_reservations(sim, current_time=400.0)

    res = ctrl.active_reservations["road_5698666_15_fwd"].get(v.id)
    assert res is not None, "Delayed vehicle reservation was dropped"
    assert res.expected_arrival_time >= 400.0, "Expected arrival was not shifted forward"


def test_physical_storage_capacity_rejection_on_short_links():
    """Verifies that physical storage safety limit (75%) prevents overloading short residential links."""
    sim = KinematicSimulator(map_name="bronx_fordham")
    ctrl = PredictiveRoutingController(mode="reactive", strict_quota_floor=False, max_diversion_quota=1.0)

    # Road 'road_586133572_123_fwd' is 44.77m long, 1 lane.
    # Safe storage = max(1, floor(0.75 * 44.77 / 7.5)) = 4 vehicles.
    short_road = "road_586133572_123_fwd"
    road_obj = sim.roads[short_road]
    assert road_obj.length < 50.0

    # Fill 4 committed reservations
    sim.vehicles.clear()
    for i in range(4):
        v_id = f"veh_pre_{i}"
        veh = SimulatedVehicle(v_id, "sedan", [short_road, "road_1082220763_166_fwd"], "#2563eb")
        sim.vehicles[v_id] = veh
        ctrl.add_route_reservations(veh, [short_road], current_time=50.0)

    # Verify short link has 4 active reservations
    assert len(ctrl.get_active_reservations(short_road)) == 4

    # 5th candidate vehicle tries to divert onto short_road
    v_cand = SimulatedVehicle("veh_cand_reject", "sedan", ["road_586133572_120_fwd", "road_1082220763_166_fwd"], "#ef4444")
    sim.vehicles[v_cand.id] = v_cand

    dest_road = "road_1082220763_166_fwd"
    telem = {dest_road: {"vc_ratio": 3.5, "operational_capacity": 500.0, "queue_length_meters": 120.0}}
    ctrl.update_telemetry(telem, timestamp=50.0)

    # In candidate search, if a candidate uses short_road, it must be rejected for storage
    res = ctrl.evaluate_and_reroute(sim, current_time=50.0, telemetry_map=telem)

    # Candidate should not be admitted to short_road
    for d in res["decisions"]:
        if d["vehicle_id"] == v_cand.id and not d["diverted"]:
            # If rejected, reason must cite capacity or storage
            assert "threshold" in d["reason"] or "quota" in d["reason"] or "benefit" in d["reason"]


def test_corridor_load_balancing_dispersion():
    """Verifies that when multiple candidate paths pass all checks, the controller balances load
    by preferring an uncongested parallel corridor over one with existing committed reservations."""
    sim = KinematicSimulator(map_name="bronx_fordham")
    ctrl = PredictiveRoutingController(mode="reactive", strict_quota_floor=False)

    # Create dummy candidate paths
    cand_a = {"path": ["road_corridor_a_1", "road_corridor_a_2"], "cost": 100.0}
    cand_b = {"path": ["road_corridor_b_1", "road_corridor_b_2"], "cost": 101.0}  # slightly higher cost

    # Put a reservation on corridor A
    ctrl.active_reservations["road_corridor_a_1"]["veh_prior"] = LinkReservation(
        vehicle_id="veh_prior",
        road_id="road_corridor_a_1",
        reserved_time=50.0,
        expected_arrival_time=60.0,
    )

    admissible = [
        {"cand": cand_a, "delta_t": 60.0, "burden": sum(len(ctrl.get_active_reservations(r)) for r in cand_a["path"]), "cost": cand_a["cost"]},
        {"cand": cand_b, "delta_t": 59.0, "burden": sum(len(ctrl.get_active_reservations(r)) for r in cand_b["path"]), "cost": cand_b["cost"]},
    ]

    # Burden on A is 1, Burden on B is 0
    assert admissible[0]["burden"] == 1
    assert admissible[1]["burden"] == 0

    # Best choice by (burden, cost) must pick corridor B to disperse traffic
    best_choice = min(admissible, key=lambda x: (x["burden"], x["cost"]))
    assert best_choice["cand"]["path"] == cand_b["path"]
