import pytest
from urbanflow.core.classroom_evaluation import classroom_controller
from urbanflow.core.demand import classroom_demand, parse_demand, write_demand_csv
from urbanflow.core.districts import DISTRICTS
from urbanflow.core.kinematic_sim import KinematicSimulator, SimulatedVehicle
from urbanflow.core.network_geometry import VehicleOccupancy
from urbanflow.core.observed_demand import COUNTS, DIRECTION, OBSERVED_WINDOWS, measured_slots
from urbanflow.core.rl_benchmarks import read_benchmarks


@pytest.mark.parametrize("map_name", DISTRICTS)
def test_district_normal_demand_drains(map_name):
    simulator = KinematicSimulator(map_name)
    departures = classroom_demand(simulator, "normal")
    simulator.set_demand(departures, "normal")
    controller = classroom_controller("ACTUATED")
    assert len(simulator.intersections) >= 6
    assert all(len(simulator.roads[road].points) == 2
               for departure in departures for road in departure.route)
    for tick in range(10_000):
        snapshot = simulator.step(0.05)
        if tick % 20 == 0:
            simulator.apply_signal_phases(controller.compute_phases(snapshot.timestamp, simulator=simulator))
        if simulator.total_arrived == len(departures):
            break
    assert simulator.total_arrived == len(departures) == 75
    assert not simulator.pending_spawns
    assert not simulator.junction_reservations or not simulator.vehicles


def test_archived_rl_scores_report_both_wins_and_losses():
    studies = [study for study in read_benchmarks()["studies"] if study["context"] == "Archived SUMO benchmark"]
    assert len(studies) == 9
    assert all(study["seed_count"] == 5 for study in studies)
    assert sum(study["results"][0]["is_rl"] for study in studies) == 6
    assert next(study for study in studies if study["source"] == "grid_test_sym.csv")["results"][0]["controller"] == "sumo-actuated"


@pytest.mark.parametrize("window", OBSERVED_WINDOWS)
def test_city_demand_preserves_measured_direction_per_interval(window):
    from collections import Counter

    sim = KinematicSimulator("harbor")
    assert len(sim.signal_controllers) == 48
    direction = {side: heading for heading, side in DIRECTION.items()}
    departures = classroom_demand(sim, window)
    simulated_counts = Counter((int(d.time // 900) * 15, direction[sim.roads[d.route[0]].from_node.split("_")[0]])
                               for d in departures)
    assert simulated_counts == measured_slots(window)
    assert departures == classroom_demand(sim, window)


def test_city_stop_line_clears_bus_tail_beside_stopped_van():
    sim = KinematicSimulator("metro")
    incoming = sim.roads["int_5_4__to__int_6_4"]
    outgoing = sim.roads["int_6_4__to__int_6_3"]
    turn = sim._get_or_build_turn_connector(incoming, 0, outgoing, 0)
    bus = SimulatedVehicle("bus", "bus", [turn.id], "#059669", length=10.5)
    bus.distance_on_road = 1.435
    sim._update_vehicle_pose(bus, turn)
    occupancy = VehicleOccupancy([bus])

    van = SimulatedVehicle("van", "van", [incoming.id], "#f59e0b", length=5.4)
    van.lane_index = 1
    van.distance_on_road = incoming.length - van.length / 2 - 0.6
    sim._update_vehicle_pose(van, incoming)
    assert occupancy.blocked(van)
    van.distance_on_road -= sim._junction_hold_offset(incoming, van.length)
    sim._update_vehicle_pose(van, incoming)
    assert not occupancy.blocked(van)


def test_measured_benchmarks_include_original_baselines_and_paired_seeds():
    measured = [study for study in read_benchmarks()["studies"] if study["source_url"]]
    assert len(measured) == len(OBSERVED_WINDOWS)
    for study in measured:
        assert study["seed_count"] == 5
        assert len(study["results"]) == 4
        assert {r["controller"] for r in study["results"]} >= {"sumo-actuated", "PPO(real_mix_s0)"}


def test_downloaded_counts_and_city_schedule_can_be_uploaded_and_scaled():
    sim = KinematicSimulator("harbor")
    departures = parse_demand(sim, COUNTS.read_text(encoding="utf-8"), "csv")
    assert len(departures) == 573 + 347 + 468
    assert 7200 < departures[-1].time < 10800
    assert parse_demand(sim, write_demand_csv(departures), "csv") == departures
    sim.set_demand(departures, "uploaded")
    sim.set_traffic_factor(4)
    assert len({vehicle.id for vehicle in sim.departures}) == 4 * len(departures)
    scaled = sim.departures
    sim.set_traffic_factor(4)
    assert sim.departures == scaled
    sim.set_traffic_factor(1)
    assert sim.departures == departures


def test_score_csv_gets_explanatory_upload_error():
    with pytest.raises(ValueError, match="Controller result CSVs contain scores only"):
        parse_demand(KinematicSimulator(), "controller,seed,avg_time_loss\nactuated,4101,7.8", "csv")
