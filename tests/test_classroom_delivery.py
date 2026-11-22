import json
from collections import defaultdict

import pytest

from urbanflow.core.classroom_evaluation import classroom_controller
from urbanflow.core.demand import classroom_demand, parse_demand
from urbanflow.core.kinematic_sim import KinematicSimulator, SimulatedVehicle


@pytest.mark.parametrize("controller_name", ["FIXED_TIME", "ACTUATED"])
def test_rush_hour_conflict_clearance_and_trip_accounting(controller_name):
    sim = KinematicSimulator()
    demand = classroom_demand(sim, "rush_hour")
    sim.set_demand(demand, "rush_hour")
    controller = classroom_controller(controller_name)
    assert len(sim.get_network_topology()["roads"]) == 24
    assert len(sim.intersections) == 4
    prior_reservations = {}
    saw_parallel_green_release = False
    max_moving = 0
    saw_stopped_at_red = False
    for tick in range(2000):
        sim.step(0.05)
        if tick == 0 or (tick + 1) % 20 == 0:
            sim.apply_signal_phases(controller.compute_phases(sim.sim_time, simulator=sim))
        max_moving = max(max_moving, sum(v.speed > 0.5 for v in sim.vehicles.values()))
        lane_vehicles = defaultdict(list)
        for vehicle in sim.vehicles.values():
            road = sim.roads[vehicle.current_road_id]
            lane_vehicles[(road.id, vehicle.lane_index)].append(vehicle)
            if road.id.startswith("CONN__"):
                assert vehicle.id in sim.junction_reservations[road.from_node]
            signal = sim.signal_controllers.get(road.to_node)
            if signal and vehicle.id not in sim.junction_reservations.get(road.to_node, {}):
                ns = abs(road.points[-1].x - road.points[0].x) < abs(road.points[-1].y - road.points[0].y)
                if signal.get_signal_code(ns) != "G":
                    assert vehicle.distance_on_road + vehicle.length / 2 <= road.length
                    if vehicle.speed < 0.5 and vehicle.distance_on_road > road.length - 10:
                        saw_stopped_at_red = True
        for vehicles in lane_vehicles.values():
            vehicles.sort(key=lambda vehicle: vehicle.distance_on_road)
            for rear, front in zip(vehicles, vehicles[1:]):
                assert front.distance_on_road - rear.distance_on_road >= (front.length + rear.length) / 2 - 1e-7
        for node, reservations in sim.junction_reservations.items():
            if reservations.keys() - prior_reservations.get(node, set()):
                assert sim.signal_controllers[node].stage in ("GREEN", "YELLOW")
            if len(reservations) > 1:
                saw_parallel_green_release = True
                movements = list(reservations.items())
                for index, (owner, movement) in enumerate(movements):
                    for other_owner, other in movements[index + 1:]:
                        assert not sim._movements_conflict(
                            movement, sim.vehicles[owner].length, other, sim.vehicles[other_owner].length)
        prior_reservations = {node: set(owners) for node, owners in sim.junction_reservations.items()}
    counts = sim.demand_counts()
    assert max_moving >= 30
    assert saw_stopped_at_red
    assert saw_parallel_green_release
    assert counts["completed"] > 0
    assert counts["due"] == counts["entered"] + counts["pending"]
    assert counts["entered"] == counts["completed"] + counts["active"]


def test_upload_validation_and_replay():
    sim = KinematicSimulator()
    row = dict(vehicle_id="vehicle-1", departure_time=1.0, entry_road="road_w_in_nw", entry_lane=1,
               destination="road_ne_out_e", vehicle_type="bus", desired_speed=9)
    departures = parse_demand(sim, json.dumps([row]), "json")
    sim.set_demand(departures, "uploaded")
    for _ in range(2):
        for _ in range(19):
            sim.step(0.05)
        assert not sim.vehicles
        sim.step(0.05)
        vehicle = sim.vehicles["vehicle-1"]
        assert vehicle.lane_index == 1
        assert vehicle.route[-1] == "road_ne_out_e"
        sim.reset()
    with pytest.raises(ValueError, match="unique"):
        parse_demand(sim, json.dumps([row, row]), "json")
    with pytest.raises(ValueError, match="departure"):
        parse_demand(sim, json.dumps([dict(row, departure_time=float("nan"))]), "json")
    assert len(classroom_demand(sim, "normal")) < len(classroom_demand(sim, "rush_hour"))


def test_green_releases_parallel_lanes_on_the_same_tick():
    sim = KinematicSimulator()
    sim.set_demand((), "test")
    signal = sim.signal_controllers["INT_NW"]
    signal.active_green_phase = 1
    incoming = sim.roads["road_w_in_nw"]
    outgoing = sim.roads["road_nw_ne"]
    for lane in (0, 1):
        vehicle = SimulatedVehicle(f"queued-{lane}", "sedan", [incoming.id, outgoing.id], "#000000")
        vehicle.lane_index = lane
        vehicle.speed = 0
        vehicle.distance_on_road = incoming.length - vehicle.length / 2 - 0.6
        sim.vehicles[vehicle.id] = vehicle

    sim.step(0.05)

    assert set(sim.junction_reservations["INT_NW"]) == {"queued-0", "queued-1"}
    assert all(vehicle.speed > 0 for vehicle in sim.vehicles.values())


def test_connector_looks_at_its_exit_lane_instead_of_a_blocked_adjacent_lane():
    sim = KinematicSimulator()
    sim.set_demand((), "test")
    incoming = sim.roads["road_w_in_sw"]
    outgoing = sim.roads["road_sw_nw"]
    connector = sim._get_or_build_turn_connector(incoming, 1, outgoing, 1)
    vehicle = SimulatedVehicle("turning", "sedan", [incoming.id, connector.id, outgoing.id], "#000000")
    vehicle.route_index = 1
    vehicle.speed = 0
    vehicle.distance_on_road = connector.length - 0.1
    sim.vehicles[vehicle.id] = vehicle
    sim.junction_reservations["INT_SW"] = {vehicle.id: (incoming.id, outgoing.id, 1)}

    bus = SimulatedVehicle("stopped-bus", "bus", [outgoing.id, "road_nw_out_n"],
                           "#000000", desired_speed=0, length=10.5)
    bus.distance_on_road = 7
    sim.vehicles[bus.id] = bus
    sim.restriction = (outgoing.id, 0, 0, 20, 0)

    for _ in range(20):
        sim.step(0.05)

    assert vehicle.current_road_id == outgoing.id
    assert vehicle.lane_index == 1
    assert bus.lane_index == 0


def test_controller_cannot_cancel_a_committed_green_during_clearance():
    sim = KinematicSimulator()
    sim.set_demand((), "test")
    controller = sim.signal_controllers["INT_NW"]
    controller.time_in_green = controller.min_green
    sim.apply_signal_phases({"INT_NW": 1})
    assert controller.stage == "YELLOW"

    for _ in range(90):
        sim.apply_signal_phases({"INT_NW": 0})
        sim.step(0.05)

    assert controller.stage == "GREEN"
    assert controller.active_green_phase == 1


def test_opposite_right_turns_release_together_when_their_paths_are_clear():
    sim = KinematicSimulator()
    sim.set_demand((), "test")
    sim.signal_controllers["INT_NW"].active_green_phase = 1
    routes = [("road_w_in_nw", "road_nw_sw"), ("road_ne_nw", "road_nw_out_n")]
    for index, (entry, exit_road) in enumerate(routes):
        assert sim.classroom_movements[entry, exit_road] == "right"
        car = SimulatedVehicle(str(index), "sedan", [entry, exit_road], "#000000")
        car.speed = 0
        car.distance_on_road = sim.roads[entry].length - car.length / 2 - 0.6
        sim.vehicles[car.id] = car

    sim.step(0.05)

    assert set(sim.junction_reservations["INT_NW"]) == {"0", "1"}


def test_resume_while_paused_frame_is_in_flight_keeps_tick_loop_running():
    import asyncio

    from urbanflow.server.websocket_feed import SimulationBroadcaster

    sim = KinematicSimulator()
    sim.set_demand((), "test")
    broadcaster = SimulationBroadcaster(sim)
    broadcaster.is_paused = True
    broadcaster.sim_speed_multiplier = 20
    broadcaster.clients.add(object())
    broadcaster._compute_intelligence = lambda snapshot, **kwargs: {}
    observed = []

    async def broadcast(snapshot):
        observed.append(snapshot.timestamp)
        if len(observed) == 1:
            broadcaster.is_paused = False
        else:
            broadcaster.is_running = False

    broadcaster.broadcast_snapshot = broadcast

    async def run():
        await asyncio.wait_for(broadcaster.run_loop(), timeout=2)

    asyncio.run(run())
    assert observed == [0, pytest.approx(1)]


def test_clearance_reservation_measures_space_behind_the_bumper():
    sim = KinematicSimulator()
    sim.set_demand((), "test")
    sim.signal_controllers["INT_NW"].active_green_phase = 1
    entry = sim.roads["road_w_in_nw"]
    exit_road = sim.roads["road_nw_ne"]
    car = SimulatedVehicle("turning-car", "sedan", [entry.id, exit_road.id], "#000000")
    car.distance_on_road = entry.length - car.length / 2 - 0.6
    car.speed = 0
    bus = SimulatedVehicle("queued-bus", "bus", [exit_road.id, "road_ne_out_e"],
                           "#000000", desired_speed=0, length=10.5)
    bus.distance_on_road = 8.5
    sim.vehicles = {car.id: car, bus.id: bus}

    sim.step(0.05)
    assert "INT_NW" not in sim.junction_reservations

    bus.distance_on_road = 16
    sim.step(0.05)
    assert car.id in sim.junction_reservations["INT_NW"]
