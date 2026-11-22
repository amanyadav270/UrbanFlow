import numpy as np
import pytest

from urbanflow.core.kinematic_sim import KinematicSimulator
from urbanflow.rl.env import DT, observe, requests


def test_switch_maps_both_directions_and_respects_clearance():
    sim = KinematicSimulator("classroom_grid")
    nodes = sorted(sim.signal_controllers)
    sim.apply_signal_phases(requests(sim, nodes, [1] * 4))
    for signal in sim.signal_controllers.values():
        assert signal.stage == "GREEN"
        assert signal.target_green_phase == 1
    for _ in range(500):
        sim.step(DT, snapshot=False)
        if all(signal.active_green_phase == 1 for signal in sim.signal_controllers.values()):
            break
    assert all(signal.stage == "GREEN" and signal.active_green_phase == 1
               for signal in sim.signal_controllers.values())
    assert requests(sim, nodes, [1] * 4) == dict.fromkeys(nodes, 0)
    assert requests(sim, nodes, [0] * 4) == dict.fromkeys(nodes, 1)


def test_queue_features_match_stopped_vehicles_and_use_training_schema():
    pytest.importorskip("torch")
    from urbanflow.rl.diagnose import schedule
    from urbanflow.rl.env import SCHEMA

    sim = KinematicSimulator("classroom_grid")
    sim.set_demand(schedule(sim, "east_west", 2501), "east_west")
    for _ in range(900):
        sim.step(DT, snapshot=False)
    totals = {}
    agents, obs = observe(sim, True, queue_counts=totals)
    assert obs.shape == (4, len(SCHEMA)) and obs.dtype == np.float32
    assert np.isfinite(obs).all() and (obs >= 0).all() and (obs <= 1).all()
    assert np.unique(obs, axis=0).shape[0] > 1
    assert sum(row["ew"] for row in totals.values()) > 0
    for i, node in enumerate(agents):
        for axis, key in enumerate(("ns", "ew")):
            links = [road for road in sim.roads.values()
                     if road.to_node == node and road.signal_controlled
                     and not road.id.startswith("CONN__") and len(road.points) >= 2
                     and bool(road.approach_ns if road.approach_ns is not None else
                              abs(road.points[-1].y - road.points[0].y) >
                              abs(road.points[-1].x - road.points[0].x)) == (axis == 0)]
            stopped = [sum(v.current_road_id == road.id and v.speed < .5 for v in sim.vehicles.values())
                       for road in links]
            expected = max((count / max(1, road.length * road.lanes / 7.5)
                            for road, count in zip(links, stopped)), default=0)
            assert totals[node][key] == sum(stopped)
            assert obs[i, axis] == pytest.approx(min(1, expected))


def test_vehicle_waits_for_switch_clearance_and_proceeds_on_green():
    from urbanflow.core.demand import Departure

    sim = KinematicSimulator("classroom_grid")
    route = next(route for route in sim.route_pool
                 if route[0].startswith(("road_w_", "road_e_")))
    road = sim.roads[route[0]]
    node = road.to_node
    sim.set_demand([Departure("probe", 0, tuple(route), 0, "sedan", 12)], "probe")
    for _ in range(800):
        sim.step(DT, snapshot=False)
        car = sim.vehicles.get("probe")
        if car and car.current_road_id == road.id and car.speed < .1 and car.distance_on_road > road.length / 2:
            break
    else:
        pytest.fail("The east-west probe did not stop at the red signal.")
    signal = sim.signal_controllers[node]
    assert signal.get_signal_code(False) == "r"
    sim.apply_signal_phases(requests(sim, [node], [1]))
    seen = set()
    for _ in range(500):
        seen.add(signal.stage)
        sim.step(DT, snapshot=False)
        if sim.vehicles["probe"].current_road_id != road.id:
            assert signal.get_signal_code(False) == "G"
            break
    else:
        pytest.fail("The probe did not advance after the green signal.")
    assert {"YELLOW", "ALL_RED"} <= seen
