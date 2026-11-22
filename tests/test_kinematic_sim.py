"""Unit tests for the UrbanFlow kinematic micro-simulation engine.
Tests both procedural baseline and real geographic (Bronx, NY) modes.
"""
import pytest
from urbanflow.core.kinematic_sim import KinematicSimulator
from urbanflow.core.demand import classroom_demand
from urbanflow.core.network_geometry import VehicleOccupancy
from urbanflow.core.types import SimulationSnapshot


def test_procedural_corridor_initialization():
    sim = KinematicSimulator(map_name="procedural")
    topology = sim.get_network_topology()

    assert topology["map_name"] == "procedural"
    assert "intersections" in topology
    assert "roads" in topology
    assert len(topology["intersections"]) == 1
    assert len(topology["roads"]) == 8


def test_procedural_corridor_stepping_and_spawning():
    sim = KinematicSimulator(map_name="procedural")
    sim.reset()

    # Step forward 20 ticks (1.0 second)
    for _ in range(20):
        snapshot = sim.step(dt=0.05)

    assert isinstance(snapshot, SimulationSnapshot)
    assert snapshot.step == 20
    assert snapshot.timestamp > 0.0
    assert len(snapshot.signals) == 1
    assert snapshot.signals[0].intersection_id == "INT_CENTER"


def test_procedural_incident_and_surge():
    sim = KinematicSimulator(map_name="procedural")
    sim.reset()

    # Surge injection
    initial_count = len(sim.vehicles)
    sim.trigger_congestion_surge("west_in", vehicle_count=10)
    assert len(sim.vehicles) == initial_count + 10

    # Incident injection
    res = sim.inject_incident("west_in", severity=0.8, duration_sec=30.0)
    assert res is True


def test_bronx_geographic_initialization():
    sim = KinematicSimulator(map_name="bronx_fordham")
    topology = sim.get_network_topology()

    assert topology["map_name"] == "bronx_fordham"
    assert len(topology["intersections"]) >= 100
    assert len(topology["roads"]) >= 150
    assert len(topology["buildings"]) >= 200
    assert len(sim.route_pool) > 20


def test_bronx_stepping_and_multi_hop_movement():
    sim = KinematicSimulator(map_name="bronx_fordham")
    sim.reset()

    # Step forward 40 ticks
    for _ in range(40):
        snapshot = sim.step(dt=0.05)

    assert isinstance(snapshot, SimulationSnapshot)
    assert snapshot.step == 40
    assert len(snapshot.signals) == 27
    assert len(sim.vehicles) > 0

    # Verify vehicle coordinates stay within Bronx bounds
    for v in sim.vehicles.values():
        assert -600.0 <= v.x <= 800.0
        assert -1100.0 <= v.y <= 1000.0
        assert len(v.route) >= 2


def test_map_switching():
    sim = KinematicSimulator(map_name="procedural")
    assert sim.map_name == "procedural"
    assert len(sim.intersections) == 1

    sim.load_map("bronx_fordham")
    assert sim.map_name == "bronx_fordham"
    assert len(sim.intersections) > 100
    assert len(sim.buildings) > 200

    sim.load_map("procedural")
    assert sim.map_name == "procedural"
    assert len(sim.intersections) == 1


def test_bronx_reserves_managed_junctions_and_separates_duplicate_directions():
    sim = KinematicSimulator(map_name="bronx_fordham")
    outbound = sim.roads["road_5697114_1_fwd"]
    inbound = sim.roads["road_5697114_2_rev"]
    assert outbound.from_node == inbound.to_node
    direct, _ = sim._get_lane_point_and_tangent(outbound, 0, at_end=False)
    opposite, _ = sim._get_lane_point_and_tangent(inbound, 0, at_end=True)
    assert direct != opposite
    assert set(sim.signal_controllers) <= sim.managed_junctions
    assert len(sim.managed_junctions) > len(sim.signal_controllers)
    assert sim.get_network_topology()["roads"][0]["left_edge"]


def test_bronx_normal_demand_flows_without_vehicle_body_overlaps():
    sim = KinematicSimulator(map_name="bronx_fordham")
    sim.set_demand(classroom_demand(sim, "normal"), "normal")

    # Sample oriented car bodies once per simulated second while the full demand
    # clears, so this catches transient intersection conflicts as well as stalls.
    for tick in range(18_000):
        sim.step(dt=0.05)
        if tick % 20 == 0:
            vehicles = list(sim.vehicles.values())
            occupancy = VehicleOccupancy(vehicles)
            overlaps = [v for v in vehicles if occupancy.blocked(v)]
            assert not overlaps, f"overlap at {sim.sim_time:.1f}s: {[(v.id, v.current_road_id, round(v.x, 1), round(v.y, 1), v.route, v.route_index, round(v.distance_on_road, 1)) for v in overlaps[:3]]}; reservations={sim.junction_reservations}"

    counts = sim.demand_counts()
    assert counts["completed"] == counts["scheduled"] == 75
    assert counts["active"] == counts["pending"] == 0
