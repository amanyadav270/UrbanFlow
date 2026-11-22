"""UrbanFlow AI — Phase 4 Infrastructure Tests.

Validates:
1. Routing KPI calculations using hand-computed toy examples.
2. Segregation of completed vs. incomplete trips.
3. Physical spillback event counting, clustering, and duration.
4. Route changes and diversion rate calculations.
5. Counterfactual seed reproducibility across Philox PRNG streams.
6. Bit-for-bit exogenous input equality checks.
7. Resumability and incomplete run detection.
8. Cryptographic preservation of existing models and campaign datasets.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, List

import pytest
import numpy as np
from numpy.random import Generator, Philox

from urbanflow.core.kinematic_sim import KinematicSimulator, RoadSegment, SimulatedVehicle
from urbanflow.intelligence.benchmark_kpis import RoutingKPIEvaluator, RoutingKPISummary
from scripts.run_dynamic_routing_benchmark import (
    derive_stream_seed,
    verify_exogenous_matching,
    ExogenousMismatchError,
    init_benchmark_metadata_db,
    is_benchmark_run_complete,
    BenchmarkScenarioConfig,
    CLASSIFIER_10MIN_PATH,
    EXPECTED_CLASSIFIER_SHA256,
)


@pytest.fixture
def mock_roads() -> Dict[str, Dict[str, Any]]:
    """Simple 3-road network with known lengths and speed limits."""
    return {
        "road_A": {"length": 500.0, "speed_limit": 10.0, "lanes": 2},  # free-flow time = 50s
        "road_B": {"length": 400.0, "speed_limit": 10.0, "lanes": 2},  # free-flow time = 40s
        "road_alt": {"length": 600.0, "speed_limit": 10.0, "lanes": 1},  # free-flow time = 60s
    }


def test_kpi_calculations_hand_computed_toy_example(mock_roads):
    """Verifies that all KPI mathematical formulas match hand-calculated values exactly."""
    evaluator = RoutingKPIEvaluator(
        roads=mock_roads,
        alternative_roads={"road_alt"},
        warmup_sec=600.0,
    )

    # 4 Completed trips:
    # Trip 1: arrival=700s (post-warmup), travel_time=100s, wait_time=20s, route=[road_A, road_B] (t_free=90s) -> delay=10s
    # Trip 2: arrival=800s (post-warmup), travel_time=150s, wait_time=50s, route=[road_A, road_B] (t_free=90s) -> delay=60s
    # Trip 3: arrival=900s (post-warmup), travel_time=80s, wait_time=0s, route=[road_A] (t_free=50s) -> delay=30s
    # Trip 4: arrival=300s (warmup -> EXCLUDED from completed statistics), travel_time=60s, wait=5s
    completed_trips = [
        {"vehicle_id": "v1", "arrival_time": 700.0, "travel_time_sec": 100.0, "wait_time_sec": 20.0, "final_route": ["road_A", "road_B"], "reroute_count": 0, "distance_traveled_m": 900.0},
        {"vehicle_id": "v2", "arrival_time": 800.0, "travel_time_sec": 150.0, "wait_time_sec": 50.0, "final_route": ["road_A", "road_B"], "reroute_count": 1, "distance_traveled_m": 900.0},
        {"vehicle_id": "v3", "arrival_time": 900.0, "travel_time_sec": 80.0, "wait_time_sec": 0.0, "final_route": ["road_A"], "reroute_count": 0, "distance_traveled_m": 500.0},
        {"vehicle_id": "v4", "arrival_time": 300.0, "travel_time_sec": 60.0, "wait_time_sec": 5.0, "final_route": ["road_A"], "reroute_count": 0, "distance_traveled_m": 500.0},
    ]

    # 2 Incomplete active trips at conclusion:
    active_vehicles = [
        {"vehicle_id": "v_inc1", "wait_time": 40.0, "distance_on_road": 50.0, "reroute_count": 1},
        {"vehicle_id": "v_inc2", "wait_time": 10.0, "distance_on_road": 150.0, "reroute_count": 0},
    ]

    # Post-warmup telemetry
    telemetry = [
        {"road_id": "road_A", "timestamp": 650.0, "queue_length_m": 20.0, "vc_ratio": 0.40},
        {"road_id": "road_B", "timestamp": 650.0, "queue_length_m": 40.0, "vc_ratio": 0.60},
    ]

    kpi = evaluator.calculate_kpis(
        completed_trips=completed_trips,
        active_vehicles=active_vehicles,
        telemetry=telemetry,
        duration_sec=3600.0,
    )

    # 1. Trip Volumes
    assert kpi.total_spawned_trips == 6
    assert kpi.completed_trips_count == 3
    assert kpi.warmup_completed_trips_count == 1
    assert kpi.incomplete_trips_count == 2
    # Post-warmup demand denominator = 3 completed + 2 incomplete = 5
    assert kpi.completion_rate == pytest.approx(3 / 5, rel=1e-3)
    # Throughput = (3 / (3600 - 600)) * 3600 = (3 / 3000) * 3600 = 3.6 veh/hr
    assert kpi.network_throughput_veh_hr == pytest.approx(3.60, rel=1e-2)

    # 2. Travel Time & Delay (Trips 1, 2, 3 only)
    # Mean travel time = (100 + 150 + 80) / 3 = 110.0s
    assert kpi.mean_travel_time_sec == pytest.approx(110.0, rel=1e-2)
    # Median travel time = 100.0s
    assert kpi.median_travel_time_sec == pytest.approx(100.0, rel=1e-2)
    # Mean delay = (10 + 60 + 30) / 3 = 33.33s
    assert kpi.mean_delay_sec == pytest.approx(33.33, rel=1e-2)
    # Mean wait = (20 + 50 + 0) / 3 = 23.33s
    assert kpi.mean_waiting_time_sec == pytest.approx(23.33, rel=1e-2)
    # Total completed travel time = 330.0s
    assert kpi.total_completed_travel_time_sec == pytest.approx(330.0, rel=1e-2)

    # 3. Incomplete Trips Segregation
    assert kpi.incomplete_active_vehicles == 2
    assert kpi.incomplete_accumulated_wait_time_sec == pytest.approx(50.0, rel=1e-2)
    assert kpi.incomplete_mean_distance_progress_m == pytest.approx(100.0, rel=1e-2)
    # Total network wait = 70.0 (completed) + 50.0 (incomplete) = 120.0s
    assert kpi.total_network_wait_time_sec == pytest.approx(120.0, rel=1e-2)


def test_completed_vs_incomplete_trip_segregation(mock_roads):
    """Verifies that incomplete trips are strictly segregated and never count as completed trips."""
    evaluator = RoutingKPIEvaluator(roads=mock_roads, warmup_sec=600.0)

    # Only 1 vehicle completed post-warmup
    completed = [
        {"vehicle_id": "v_done", "arrival_time": 750.0, "travel_time_sec": 120.0, "wait_time_sec": 10.0, "final_route": ["road_A"], "reroute_count": 0, "distance_traveled_m": 500.0}
    ]

    # 5 vehicles stuck in gridlock (incomplete)
    active = [
        {"vehicle_id": f"v_stuck_{i}", "wait_time": 500.0, "distance_on_road": 10.0, "reroute_count": 0}
        for i in range(5)
    ]

    kpi = evaluator.calculate_kpis(
        completed_trips=completed,
        active_vehicles=active,
        telemetry=[],
        duration_sec=3600.0,
    )

    # Completed count must strictly be 1
    assert kpi.completed_trips_count == 1
    # Mean travel time reflects ONLY the completed trip
    assert kpi.mean_travel_time_sec == pytest.approx(120.0, rel=1e-2)
    # Incomplete count must strictly be 5
    assert kpi.incomplete_trips_count == 5
    assert kpi.incomplete_active_vehicles == 5
    # Incomplete wait time tracks the 5 x 500 = 2500s delay
    assert kpi.incomplete_accumulated_wait_time_sec == pytest.approx(2500.0, rel=1e-2)
    # Completion rate is 1 / 6
    assert kpi.completion_rate == pytest.approx(1 / 6, rel=1e-2)


def test_spillback_event_counting_and_duration():
    """Verifies that physical spillback events are clustered into contiguous durations."""
    # Road length = 100m. Threshold = 90m (or min(90, 85) = 85m).
    roads = {"road_1": {"length": 100.0, "speed_limit": 10.0}}
    evaluator = RoutingKPIEvaluator(roads=roads, warmup_sec=600.0, spillback_occupancy_threshold=0.90)

    # Sample interval = 5.0s
    telemetry = [
        # Warmup (ignored)
        {"road_id": "road_1", "timestamp": 580.0, "queue_length_m": 95.0},
        {"road_id": "road_1", "timestamp": 595.0, "queue_length_m": 95.0},
        # Event 1: t=605, 610, 615 (3 samples -> 15s duration)
        {"road_id": "road_1", "timestamp": 600.0, "queue_length_m": 20.0},
        {"road_id": "road_1", "timestamp": 605.0, "queue_length_m": 92.0},
        {"road_id": "road_1", "timestamp": 610.0, "queue_length_m": 95.0},
        {"road_id": "road_1", "timestamp": 615.0, "queue_length_m": 90.0},
        {"road_id": "road_1", "timestamp": 620.0, "queue_length_m": 15.0},
        # Event 2: t=640, 645 (2 samples -> 10s duration)
        {"road_id": "road_1", "timestamp": 640.0, "queue_length_m": 96.0},
        {"road_id": "road_1", "timestamp": 645.0, "queue_length_m": 91.0},
        {"road_id": "road_1", "timestamp": 650.0, "queue_length_m": 30.0},
    ]

    count, total_dur, mean_dur, max_dur = evaluator.detect_spillback_events(telemetry, sample_interval_sec=5.0)

    assert count == 2
    assert total_dur == pytest.approx(25.0, rel=1e-2)
    assert mean_dur == pytest.approx(12.5, rel=1e-2)
    assert max_dur == pytest.approx(15.0, rel=1e-2)


def test_route_change_and_diversion_rate_calculations(mock_roads):
    """Verifies total reroutes and diversion rates across completed and active vehicles."""
    evaluator = RoutingKPIEvaluator(roads=mock_roads, warmup_sec=600.0)

    completed = [
        {"vehicle_id": "v1", "arrival_time": 700.0, "reroute_count": 0, "final_route": ["road_A"]},
        {"vehicle_id": "v2", "arrival_time": 800.0, "reroute_count": 1, "final_route": ["road_A", "road_alt"]},
        {"vehicle_id": "v3", "arrival_time": 900.0, "reroute_count": 2, "final_route": ["road_A", "road_B"]},
    ]
    active = [
        {"vehicle_id": "v4", "reroute_count": 1, "wait_time": 10.0, "distance_on_road": 20.0},
        {"vehicle_id": "v5", "reroute_count": 0, "wait_time": 5.0, "distance_on_road": 50.0},
    ]

    kpi = evaluator.calculate_kpis(
        completed_trips=completed,
        active_vehicles=active,
        telemetry=[],
        duration_sec=3600.0,
    )

    # Total reroute commands = 0 + 1 + 2 + 1 + 0 = 4
    assert kpi.total_route_changes == 4
    # Unique diverted vehicles = v2, v3, v4 = 3
    assert kpi.unique_diverted_vehicles_count == 3
    # Overall diversion rate = 3 / 5 = 0.60
    assert kpi.overall_diversion_rate == pytest.approx(0.60, rel=1e-2)
    # Completed diversion rate = 2 / 3 = 0.667
    assert kpi.completed_diversion_rate == pytest.approx(2 / 3, rel=1e-2)


def test_seed_and_scenario_reproducibility_across_regimes():
    """Verifies that identical root seeds generate bit-for-bit identical vehicle sequences."""
    root_seed = 5001

    stream_d1 = derive_stream_seed(root_seed, "driver_attributes")
    stream_s1 = derive_stream_seed(root_seed, "spawn_arrivals")

    stream_d2 = derive_stream_seed(root_seed, "driver_attributes")
    stream_s2 = derive_stream_seed(root_seed, "spawn_arrivals")

    # Derivations must be deterministic
    assert stream_d1 == stream_d2
    assert stream_s1 == stream_s2

    # Instantiate two independent simulators with identical seeds
    sim1 = KinematicSimulator(
        map_name="bronx_fordham",
        rng_spawn=Generator(Philox(stream_s1)),
        rng_driver=Generator(Philox(stream_d1)),
    )
    sim2 = KinematicSimulator(
        map_name="bronx_fordham",
        rng_spawn=Generator(Philox(stream_s2)),
        rng_driver=Generator(Philox(stream_d2)),
    )

    # Initial pre-seeded fleet must match exactly
    assert len(sim1.vehicles) == len(sim2.vehicles)
    for v_id in sim1.vehicles:
        assert v_id in sim2.vehicles
        v1 = sim1.vehicles[v_id]
        v2 = sim2.vehicles[v_id]
        assert v1.desired_speed == v2.desired_speed
        assert v1.route == v2.route
        assert v1.distance_on_road == v2.distance_on_road

    # Advance both by 15 physics steps
    for _ in range(15):
        sim1.step(0.25)
        sim2.step(0.25)

    assert sim1.sim_time == sim2.sim_time
    assert len(sim1.vehicles) == len(sim2.vehicles)
    for v_id in sim1.vehicles:
        v1 = sim1.vehicles[v_id]
        v2 = sim2.vehicles[v_id]
        assert v1.speed == pytest.approx(v2.speed, abs=1e-6)
        assert v1.distance_on_road == pytest.approx(v2.distance_on_road, abs=1e-6)


def test_exogenous_input_equality_checks():
    """Verifies that counterfactual exogenous audits catch any discrepancy between regimes."""
    trace_a = {
        "stream_seeds": {"driver": 111, "spawn": 222, "incident": 333},
        "initial_fleet": [{"id": "v0", "type": "sedan", "desired_speed": 11.2, "initial_route": ["r1"]}],
        "spawns": [{"id": "v1", "time": 5.0, "type": "sedan", "desired_speed": 10.5, "initial_route": ["r1", "r2"]}],
        "incident": {"road": "r1", "severity": 0.8, "start_sec": 1200.0, "duration_sec": 600.0},
    }

    # Identical trace B
    trace_b = json.loads(json.dumps(trace_a))
    assert verify_exogenous_matching(trace_a, trace_b) is True

    # Modified driver speed in trace B
    trace_b_bad = json.loads(json.dumps(trace_a))
    trace_b_bad["spawns"][0]["desired_speed"] = 14.0
    with pytest.raises(ExogenousMismatchError, match="Exogenous vehicle attribute mismatch"):
        verify_exogenous_matching(trace_a, trace_b_bad)

    # Modified incident schedule in trace B
    trace_b_bad_inc = json.loads(json.dumps(trace_a))
    trace_b_bad_inc["incident"]["start_sec"] = 1250.0
    with pytest.raises(ExogenousMismatchError, match="Incident onset/severity schedule mismatch"):
        verify_exogenous_matching(trace_a, trace_b_bad_inc)


def test_resume_behavior_and_incomplete_run_detection(tmp_path):
    """Verifies that existing complete runs are skipped while incomplete or corrupted runs are flagged."""
    db_path = tmp_path / "test_benchmark.db"
    init_benchmark_metadata_db(db_path)

    run_id_success = "RUN_FAM05_S5001_REACTIVE"
    summary_file = tmp_path / f"{run_id_success}_summary.json"
    summary_file.write_text(json.dumps({
        "run_id": run_id_success,
        "mean_travel_time_sec": 123.45,
    }), encoding="utf-8")

    # Record SUCCESS run in DB
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    c.execute("""
        INSERT INTO benchmark_runs (run_id, status, summary_json_path)
        VALUES (?, ?, ?)
    """, (run_id_success, "SUCCESS", str(summary_file)))
    conn.commit()
    conn.close()

    assert is_benchmark_run_complete(run_id_success, db_path, tmp_path) is True

    # Incomplete run (status = FAILED)
    run_id_failed = "RUN_FAM05_S5002_REACTIVE"
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    c.execute("""
        INSERT INTO benchmark_runs (run_id, status, summary_json_path)
        VALUES (?, ?, ?)
    """, (run_id_failed, "FAILED", ""))
    conn.commit()
    conn.close()

    assert is_benchmark_run_complete(run_id_failed, db_path, tmp_path) is False

    # Run with deleted summary JSON
    run_id_missing_json = "RUN_FAM05_S5003_REACTIVE"
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    c.execute("""
        INSERT INTO benchmark_runs (run_id, status, summary_json_path)
        VALUES (?, ?, ?)
    """, (run_id_missing_json, "SUCCESS", str(tmp_path / "nonexistent.json")))
    conn.commit()
    conn.close()

    assert is_benchmark_run_complete(run_id_missing_json, db_path, tmp_path) is False


def test_preservation_of_existing_datasets_and_models():
    """Verifies that baseline models and campaign datasets are cryptographically intact."""
    assert CLASSIFIER_10MIN_PATH.exists(), f"Classifier missing: {CLASSIFIER_10MIN_PATH}"
    actual_hash = hashlib.sha256(CLASSIFIER_10MIN_PATH.read_bytes()).hexdigest().upper()
    assert actual_hash == EXPECTED_CLASSIFIER_SHA256, (
        f"Classifier hash mismatch: expected {EXPECTED_CLASSIFIER_SHA256}, got {actual_hash}"
    )

    # Check 15-min and 5-min network classifiers
    c_15 = CLASSIFIER_10MIN_PATH.parent / "lgb_network_spillback_classifier_15min.joblib"
    c_5 = CLASSIFIER_10MIN_PATH.parent / "lgb_network_spillback_classifier_5min.joblib"
    assert c_15.exists()
    assert c_5.exists()
    assert hashlib.sha256(c_15.read_bytes()).hexdigest().upper() == "429E172E160CDC486A4667B2ED6513CCA7E017C9F1BB4B403204DCEF18B5B198"
    assert hashlib.sha256(c_5.read_bytes()).hexdigest().upper() == "5B301A531D1D8A9223CBEE3C076ED1F56DBF0D181580B10E3D5EB25E732028E9"


def test_online_observation_buffer_density_aliases():
    """Verifies that OnlineObservationBuffer accepts both density naming conventions."""
    from urbanflow.intelligence.routing import OnlineObservationBuffer
    from urbanflow.core.network_topology import NetworkTopology

    topo = NetworkTopology.get_instance()
    buf = OnlineObservationBuffer(topo)
    test_road = next(iter(topo.roads.keys()))

    # Case 1: density_veh_km_lane alone (campaign dataset convention)
    for i, val in enumerate([10.0, 20.0, 30.0, 40.0]):
        buf.update({test_road: {"density_veh_km_lane": val, "avg_speed_kmh": 30.0, "queue_length_m": 5.0}}, timestamp=float(i * 5))
    feats1 = buf.build_feature_dict(test_road)
    assert feats1 is not None
    assert feats1["density_lag_1"] == 30.0
    assert feats1["density_lag_2"] == 20.0
    assert feats1["density_lag_3"] == 10.0
    assert feats1["density_roll_mean"] == pytest.approx(25.0, rel=1e-3)

    # Case 2: density_per_km_lane alone (state estimator convention)
    buf = OnlineObservationBuffer(topo)
    for i, val in enumerate([15.0, 25.0, 35.0, 45.0]):
        buf.update({test_road: {"density_per_km_lane": val, "avg_speed_kmh": 30.0, "queue_length_m": 5.0}}, timestamp=float(i * 5))
    feats2 = buf.build_feature_dict(test_road)
    assert feats2 is not None
    assert feats2["density_lag_1"] == 35.0
    assert feats2["density_lag_2"] == 25.0
    assert feats2["density_lag_3"] == 15.0
    assert feats2["density_roll_mean"] == pytest.approx(30.0, rel=1e-3)

    # Case 3: Both present -> prefers density_per_km_lane
    buf = OnlineObservationBuffer(topo)
    for i in range(4):
        buf.update({test_road: {
            "density_per_km_lane": 50.0,
            "density_veh_km_lane": 10.0,
            "avg_speed_kmh": 30.0,
            "queue_length_m": 5.0,
        }}, timestamp=float(i * 5))
    feats3 = buf.build_feature_dict(test_road)
    assert feats3 is not None
    assert feats3["density_lag_1"] == 50.0
    assert feats3["density_roll_mean"] == 50.0


def test_online_observation_buffer_perimeter_exit_scaling():
    """Verifies that perimeter exit roads output unscaled speed limits matching training features."""
    from urbanflow.intelligence.routing import OnlineObservationBuffer
    from urbanflow.core.network_topology import NetworkTopology

    topo = NetworkTopology.get_instance()
    buf = OnlineObservationBuffer(topo)

    # Identify a perimeter exit road (empty downstream set)
    perim_road = None
    for r_id in topo.roads:
        if not topo.get_downstream_roads(r_id, 1):
            perim_road = r_id
            break

    assert perim_road is not None, "No perimeter exit road found in topology"
    raw_limit = topo.get_road_speed_limit(perim_road)

    for i in range(4):
        buf.update({perim_road: {"avg_speed_kmh": 30.0, "queue_length_m": 5.0}}, timestamp=float(i * 5))

    feats = buf.build_feature_dict(perim_road)
    assert feats is not None
    assert feats["is_perimeter_exit"] == 1.0
    # Must match raw unscaled speed limit (8.9 or 11.2 m/s), NOT multiplied by 3.6
    assert feats["down_1hop_min_speed_kmh"] == raw_limit
    assert feats["down_2hop_min_speed_kmh"] == raw_limit
    assert feats["down_3hop_min_speed_kmh"] == raw_limit

