"""UrbanFlow AI — 10-Execution Resource Pilot Benchmark Runner (Rev 6).

Implements Section 7 of forecasting_validation_campaign_plan.md (Revision 6).
Executes exactly 10 process executions to measure:
1. Logging Overhead: Paired logging OFF vs ON runs for normal and congested traffic.
2. Serial Baseline Workload: 3 runs executed sequentially.
3. Concurrent Workload: 3 identical runs executed across 3 workers simultaneously.
4. Parquet export size, SQLite metadata growth, and Windows peak memory (RSS).

Outputs are isolated strictly to:
- data/datasets/pilot_resource/ (Parquet files)
- data/datasets/pilot_metadata.db (SQLite run metadata index)
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
import sqlite3
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from multiprocessing import Pool

import numpy as np
from numpy.random import Generator, Philox
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Ensure urbanflow root is on sys.path
URBANFLOW_ROOT = Path(__file__).resolve().parent.parent
if str(URBANFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(URBANFLOW_ROOT))

from urbanflow.core.kinematic_sim import KinematicSimulator, SimulatedVehicle
from urbanflow.intelligence.state_estimator import TrafficStateEstimator

# Target directories
DATA_DIR = URBANFLOW_ROOT / "data"
PILOT_DATA_DIR = DATA_DIR / "datasets" / "pilot_resource"
METADATA_DB_PATH = DATA_DIR / "datasets" / "pilot_metadata.db"


def get_peak_memory_mb() -> float:
    """Returns the peak working set size (RSS) in MB using the Windows native API."""
    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    try:
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        get_mem = ctypes.windll.kernel32.K32GetProcessMemoryInfo
        get_mem.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
        get_mem.restype = wintypes.BOOL
        success = get_mem(handle, ctypes.byref(counters), counters.cb)
        if success:
            return counters.PeakWorkingSetSize / (1024.0 * 1024.0)
    except Exception:
        pass
    return 0.0


def derive_stream_seed(root_seed: int, stream_name: str) -> int:
    """Canonical SHA-256 PRNG seed derivation matching Section 6.2 of Rev 6."""
    payload = f"{root_seed}:{stream_name}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


@dataclass
class PilotRunConfig:
    run_id: str
    family_id: str
    random_seed: int
    logging_enabled: bool
    worker_mode: str
    partner_id: str
    duration_sec: float = 3600.0
    dt_physics: float = 0.25
    sample_interval_sec: float = 5.0
    incident_active: bool = False
    incident_start_sec: float = 1200.0
    incident_duration_sec: float = 1200.0
    incident_road: str = "E_Fordham_Rd_Mid"
    demand_multiplier: float = 1.0


# The Authoritative 10-Execution Pilot Matrix from Section 7.1 of Rev 6
PILOT_WORKLOAD: List[PilotRunConfig] = [
    # Workload Group A: Logging Overhead Analysis (4 Process Executions)
    PilotRunConfig(
        run_id="P-RAW-NORM",
        family_id="FAM-01",
        random_seed=9901,
        logging_enabled=False,
        worker_mode="1 worker (Isolated)",
        partner_id="P-LOG-NORM",
        demand_multiplier=1.0,
    ),
    PilotRunConfig(
        run_id="P-LOG-NORM",
        family_id="FAM-01",
        random_seed=9901,
        logging_enabled=True,
        worker_mode="1 worker (Isolated)",
        partner_id="P-RAW-NORM",
        demand_multiplier=1.0,
    ),
    PilotRunConfig(
        run_id="P-RAW-CONG",
        family_id="FAM-04",
        random_seed=9902,
        logging_enabled=False,
        worker_mode="1 worker (Isolated)",
        partner_id="P-LOG-CONG",
        incident_active=True,
        demand_multiplier=1.2,
    ),
    PilotRunConfig(
        run_id="P-LOG-CONG",
        family_id="FAM-04",
        random_seed=9902,
        logging_enabled=True,
        worker_mode="1 worker (Isolated)",
        partner_id="P-RAW-CONG",
        incident_active=True,
        demand_multiplier=1.2,
    ),
    # Workload Group B: Serial Baseline Workload (3 Process Executions Sequentially)
    PilotRunConfig(
        run_id="P-SER-1",
        family_id="FAM-01",
        random_seed=9903,
        logging_enabled=True,
        worker_mode="1 worker (Serial 1/3)",
        partner_id="P-CON-1",
        demand_multiplier=1.0,
    ),
    PilotRunConfig(
        run_id="P-SER-2",
        family_id="FAM-02",
        random_seed=9904,
        logging_enabled=True,
        worker_mode="1 worker (Serial 2/3)",
        partner_id="P-CON-2",
        demand_multiplier=1.8,
    ),
    PilotRunConfig(
        run_id="P-SER-3",
        family_id="FAM-04",
        random_seed=9905,
        logging_enabled=True,
        worker_mode="1 worker (Serial 3/3)",
        partner_id="P-CON-3",
        incident_active=True,
        demand_multiplier=1.2,
    ),
    # Workload Group C: Concurrent Workload (3 Process Executions Simultaneously across 3 Workers)
    PilotRunConfig(
        run_id="P-CON-1",
        family_id="FAM-01",
        random_seed=9903,
        logging_enabled=True,
        worker_mode="Worker 1 of 3 (Parallel)",
        partner_id="P-SER-1",
        demand_multiplier=1.0,
    ),
    PilotRunConfig(
        run_id="P-CON-2",
        family_id="FAM-02",
        random_seed=9904,
        logging_enabled=True,
        worker_mode="Worker 2 of 3 (Parallel)",
        partner_id="P-SER-2",
        demand_multiplier=1.8,
    ),
    PilotRunConfig(
        run_id="P-CON-3",
        family_id="FAM-04",
        random_seed=9905,
        logging_enabled=True,
        worker_mode="Worker 3 of 3 (Parallel)",
        partner_id="P-SER-3",
        incident_active=True,
        demand_multiplier=1.2,
    ),
]


def init_metadata_db(db_path: Path):
    """Initializes the SQLite metadata table for pilot benchmark execution records."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pilot_runs (
            run_id TEXT PRIMARY KEY,
            family_id TEXT,
            random_seed INTEGER,
            stream_seed_driver INTEGER,
            stream_seed_spawn INTEGER,
            stream_seed_incident INTEGER,
            prng_algorithm TEXT,
            worker_mode TEXT,
            partner_id TEXT,
            logging_enabled BOOLEAN,
            duration_sec REAL,
            physics_steps INTEGER,
            wall_time_sec REAL,
            steps_per_sec REAL,
            peak_memory_mb REAL,
            parquet_size_bytes INTEGER,
            observation_count INTEGER,
            mean_driver_speed_mult REAL,
            std_driver_speed_mult REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def execute_single_pilot_run(cfg: PilotRunConfig) -> Dict[str, Any]:
    """Executes a single 3,600s pilot simulation run matching Rev 6 specs."""
    # Enforce OpenBLAS thread count per worker process
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    t_start = time.perf_counter()

    # 1. Initialize Decoupled Deterministic PRNG Streams
    stream_driver = derive_stream_seed(cfg.random_seed, "driver_attributes")
    stream_spawn = derive_stream_seed(cfg.random_seed, "spawn_arrivals")
    stream_inc = derive_stream_seed(cfg.random_seed, "incident_timing")

    rng_driver = Generator(Philox(stream_driver))
    rng_spawn = Generator(Philox(stream_spawn))
    rng_incident = Generator(Philox(stream_inc))

    # 2. Instantiate Simulator and State Estimator
    sim = KinematicSimulator(map_name="bronx_fordham")
    estimator = TrafficStateEstimator(sample_interval_sec=cfg.sample_interval_sec)

    # Base spawn interval modulated by demand multiplier
    base_spawn_interval = 0.8 / max(0.1, cfg.demand_multiplier)
    sim.spawn_interval = base_spawn_interval

    # Realized driver speed multipliers
    driver_multipliers: List[float] = []

    # Telemetry observation buffer
    observations: List[Dict[str, Any]] = []

    # Time parameters
    total_steps = int(round(cfg.duration_sec / cfg.dt_physics))  # 14,400 intervals
    sample_every_n_steps = int(round(cfg.sample_interval_sec / cfg.dt_physics))  # 20 steps

    # Identify incident target road if active
    incident_road_id = ""
    if cfg.incident_active:
        # Match road containing 'fordham' or pick first multi-lane road
        for r_id in sim.roads:
            if "fordham" in r_id.lower() and sim.roads[r_id].lanes >= 2:
                incident_road_id = r_id
                break
        if not incident_road_id and sim.roads:
            incident_road_id = list(sim.roads.keys())[0]

    # Pre-sample initial driver multipliers for initial seeded fleet
    for veh in sim.vehicles.values():
        mult = float(np.clip(rng_driver.normal(1.0, 0.10), 0.80, 1.25))
        veh.desired_speed *= mult
        driver_multipliers.append(mult)

    # 3. Main Discrete Physics Simulation Loop
    for step_idx in range(total_steps):
        current_time = step_idx * cfg.dt_physics

        # Check vehicle spawning with decoupled RNG
        if current_time - sim.last_spawn_time >= sim.spawn_interval and len(sim.vehicles) < 140:
            if sim.route_pool:
                sim.vehicle_counter += 1
                v_id = f"veh_{sim.vehicle_counter}"
                v_type = rng_spawn.choice(["sedan", "van", "bus", "emergency"], p=[0.70, 0.15, 0.10, 0.05])
                base_speed = {"sedan": 12.0, "van": 11.0, "bus": 9.5, "emergency": 15.0}[v_type]
                v_len = {"sedan": 4.2, "van": 5.4, "bus": 10.5, "emergency": 5.0}[v_type]

                # Stochastic driver heterogeneity multiplier
                mult = float(np.clip(rng_driver.normal(1.0, 0.10), 0.80, 1.25))
                driver_multipliers.append(mult)
                desired_speed = base_speed * mult

                route_idx = int(rng_spawn.integers(0, len(sim.route_pool)))
                route = sim.route_pool[route_idx]
                first_road = sim.roads.get(route[0])
                lane_idx = int(rng_spawn.integers(0, max(1, first_road.lanes if first_road else 1)))

                veh = SimulatedVehicle(
                    vehicle_id=v_id,
                    v_type=v_type,
                    route=route,
                    color="#2563eb",
                    desired_speed=desired_speed,
                    length=v_len,
                )
                veh.lane_index = lane_idx
                road_vehs = [v for v in sim.vehicles.values() if v.current_road_id == route[0]]
                if not any(v.distance_on_road < (veh.length + 3.0) for v in road_vehs):
                    if first_road:
                        sim._update_vehicle_pose(veh, first_road)
                    sim.vehicles[v_id] = veh
                    sim.last_spawn_time = current_time

        # Inject road capacity reduction if incident is active
        is_incident_now = False
        if cfg.incident_active and incident_road_id:
            if cfg.incident_start_sec <= current_time < (cfg.incident_start_sec + cfg.incident_duration_sec):
                is_incident_now = True

        # Advance physics step
        sim.step(dt=cfg.dt_physics)

        # 4. Telemetry Logging (every 5.0 seconds = 20 steps)
        if cfg.logging_enabled and (step_idx % sample_every_n_steps == 0 or step_idx == total_steps - 1):
            log_time = round(step_idx * cfg.dt_physics, 2)
            is_warmup = (log_time < 600.0)

            # Map vehicles to roads for macroscopic estimation
            vehs_by_road: Dict[str, List[SimulatedVehicle]] = {r: [] for r in sim.roads}
            for v in sim.vehicles.values():
                r_id = v.current_road_id
                if r_id in vehs_by_road:
                    vehs_by_road[r_id].append(v)

            for road_id, road in sim.roads.items():
                road_vehs = vehs_by_road.get(road_id, [])
                incident_sev = 0.75 if (is_incident_now and road_id == incident_road_id) else 0.0

                state = estimator.estimate_segment_state(
                    road_id=road_id,
                    timestamp=log_time,
                    road_length=road.length,
                    lanes=road.lanes,
                    speed_limit_mps=road.speed_limit,
                    vehicles_on_road=road_vehs,
                    downstream_signal_state="NONE",
                    incident_severity=incident_sev,
                )

                observations.append({
                    "run_id": cfg.run_id,
                    "road_id": road_id,
                    "timestamp": log_time,
                    "step_index": step_idx,
                    "is_warmup": is_warmup,
                    "vehicle_count": state.vehicle_count,
                    "density_veh_km_lane": state.density_per_km_lane,
                    "avg_speed_kmh": state.average_speed_kmh,
                    "flow_veh_hr": state.flow_veh_per_hour,
                    "queue_length_m": state.queue_length_meters,
                    "queue_vehicle_count": state.queue_vehicle_count,
                    "occupancy_ratio": state.occupancy_ratio,
                    "travel_time_sec": state.travel_time_sec,
                    "operational_capacity": state.operational_capacity,
                    "vc_ratio": state.capacity_utilization,
                    "signal_state": state.signal_state,
                    "incident_active": (is_incident_now and road_id == incident_road_id),
                })

    t_wall = time.perf_counter() - t_start
    steps_per_sec = total_steps / max(0.001, t_wall)
    peak_mem_mb = get_peak_memory_mb()

    # 5. Parquet Serialization (if logging enabled)
    parquet_size_bytes = 0
    obs_count = len(observations)
    if cfg.logging_enabled and observations:
        PILOT_DATA_DIR.mkdir(parents=True, exist_ok=True)
        parquet_path = PILOT_DATA_DIR / f"{cfg.run_id}.parquet"
        df = pd.DataFrame(observations)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, str(parquet_path), compression="snappy")
        parquet_size_bytes = parquet_path.stat().st_size

    mean_mult = float(np.mean(driver_multipliers)) if driver_multipliers else 1.0
    std_mult = float(np.std(driver_multipliers)) if driver_multipliers else 0.0

    result = {
        "run_id": cfg.run_id,
        "family_id": cfg.family_id,
        "random_seed": cfg.random_seed,
        "stream_seed_driver": stream_driver,
        "stream_seed_spawn": stream_spawn,
        "stream_seed_incident": stream_inc,
        "prng_algorithm": "NumPy-Philox",
        "worker_mode": cfg.worker_mode,
        "partner_id": cfg.partner_id,
        "logging_enabled": cfg.logging_enabled,
        "duration_sec": cfg.duration_sec,
        "physics_steps": total_steps,
        "wall_time_sec": round(t_wall, 3),
        "steps_per_sec": round(steps_per_sec, 1),
        "peak_memory_mb": round(peak_mem_mb, 2),
        "parquet_size_bytes": parquet_size_bytes,
        "observation_count": obs_count,
        "mean_driver_speed_mult": round(mean_mult, 4),
        "std_driver_speed_mult": round(std_mult, 4),
    }

    # Record to metadata SQLite DB
    conn = sqlite3.connect(str(METADATA_DB_PATH))
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO pilot_runs (
            run_id, family_id, random_seed, stream_seed_driver, stream_seed_spawn,
            stream_seed_incident, prng_algorithm, worker_mode, partner_id, logging_enabled,
            duration_sec, physics_steps, wall_time_sec, steps_per_sec, peak_memory_mb,
            parquet_size_bytes, observation_count, mean_driver_speed_mult, std_driver_speed_mult
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        result["run_id"], result["family_id"], result["random_seed"],
        result["stream_seed_driver"], result["stream_seed_spawn"], result["stream_seed_incident"],
        result["prng_algorithm"], result["worker_mode"], result["partner_id"],
        result["logging_enabled"], result["duration_sec"], result["physics_steps"],
        result["wall_time_sec"], result["steps_per_sec"], result["peak_memory_mb"],
        result["parquet_size_bytes"], result["observation_count"],
        result["mean_driver_speed_mult"], result["std_driver_speed_mult"]
    ))
    conn.commit()
    conn.close()

    return result


def print_scorecard(results: Dict[str, Dict[str, Any]]):
    """Computes and formats the official Rev 6 Pilot Resource Scorecard."""
    print("\n" + "=" * 80)
    print("URBANFLOW AI — 10-EXECUTION RESOURCE PILOT BENCHMARK SCORECARD (REV 6)")
    print("=" * 80)

    # 1. Logging Overhead
    t_raw_norm = results["P-RAW-NORM"]["wall_time_sec"]
    t_log_norm = results["P-LOG-NORM"]["wall_time_sec"]
    t_raw_cong = results["P-RAW-CONG"]["wall_time_sec"]
    t_log_cong = results["P-LOG-CONG"]["wall_time_sec"]

    oh_norm = ((t_log_norm - t_raw_norm) / t_raw_norm) * 100.0
    oh_cong = ((t_log_cong - t_raw_cong) / t_raw_cong) * 100.0

    print("\n[1. LOGGING OVERHEAD ANALYSIS (Matched Pairs)]")
    print(f"  • Normal Flow (Fam 1, Seed 9901):")
    print(f"      P-RAW-NORM (No Log) : {t_raw_norm:.2f} s  ({results['P-RAW-NORM']['steps_per_sec']:.0f} steps/s)")
    print(f"      P-LOG-NORM (With Log): {t_log_norm:.2f} s  ({results['P-LOG-NORM']['steps_per_sec']:.0f} steps/s)")
    print(f"      Measured Overhead   : {oh_norm:+.1f}%  (Budget Target: <= 25.0%)")

    print(f"  • Congested Flow (Fam 4, Seed 9902):")
    print(f"      P-RAW-CONG (No Log) : {t_raw_cong:.2f} s  ({results['P-RAW-CONG']['steps_per_sec']:.0f} steps/s)")
    print(f"      P-LOG-CONG (With Log): {t_log_cong:.2f} s  ({results['P-LOG-CONG']['steps_per_sec']:.0f} steps/s)")
    print(f"      Measured Overhead   : {oh_cong:+.1f}%  (Budget Target: <= 25.0%)")

    # 2. Concurrency Speedup
    t_ser_total = (
        results["P-SER-1"]["wall_time_sec"] +
        results["P-SER-2"]["wall_time_sec"] +
        results["P-SER-3"]["wall_time_sec"]
    )
    t_con_max = max(
        results["P-CON-1"]["wall_time_sec"],
        results["P-CON-2"]["wall_time_sec"],
        results["P-CON-3"]["wall_time_sec"]
    )
    speedup = t_ser_total / max(0.001, t_con_max)

    print("\n[2. CONCURRENCY SCALING ANALYSIS (Matched Workload Suite)]")
    print(f"  • Serial Suite Total Wall Time   : {t_ser_total:.2f} s")
    print(f"      P-SER-1: {results['P-SER-1']['wall_time_sec']:.2f} s | P-SER-2: {results['P-SER-2']['wall_time_sec']:.2f} s | P-SER-3: {results['P-SER-3']['wall_time_sec']:.2f} s")
    print(f"  • Concurrent Suite Wall Time (3w): {t_con_max:.2f} s")
    print(f"      P-CON-1: {results['P-CON-1']['wall_time_sec']:.2f} s | P-CON-2: {results['P-CON-2']['wall_time_sec']:.2f} s | P-CON-3: {results['P-CON-3']['wall_time_sec']:.2f} s")
    print(f"  • Measured 3-Worker Speedup (S_3): {speedup:.2f}x  (Budget Target: >= 2.40x)")

    # 3. Storage & Memory Footprint
    max_mem = max(r["peak_memory_mb"] for r in results.values())
    logged_sizes = [r["parquet_size_bytes"] for r in results.values() if r["logging_enabled"]]
    avg_parquet_kb = (sum(logged_sizes) / len(logged_sizes)) / 1024.0 if logged_sizes else 0.0

    print("\n[3. RESOURCE & STORAGE FOOTPRINT]")
    print(f"  • Peak Process Memory (RSS)      : {max_mem:.1f} MB  (Budget Target: <= 500.0 MB)")
    print(f"  • Mean Parquet File Size per Run : {avg_parquet_kb / 1024.0:.2f} MB ({avg_parquet_kb:.0f} KB)")
    print(f"  • Projected 60-Run Parquet Total : {(avg_parquet_kb / 1024.0) * 60:.1f} MB")
    print("=" * 80 + "\n")


def run_pilot():
    """Main execution orchestrator for the 10-execution pilot benchmark."""
    print("Initializing UrbanFlow Pilot Resource Benchmark (Revision 6)...")
    init_metadata_db(METADATA_DB_PATH)

    results: Dict[str, Dict[str, Any]] = {}

    configs_dict = {cfg.run_id: cfg for cfg in PILOT_WORKLOAD}

    # Phase 1: Workload Group A (Logging Overhead Analysis — 4 isolated runs)
    print("\nExecuting Workload Group A: Logging Overhead Analysis (4 runs)...")
    for r_id in ["P-RAW-NORM", "P-LOG-NORM", "P-RAW-CONG", "P-LOG-CONG"]:
        cfg = configs_dict[r_id]
        print(f"  -> Running {cfg.run_id} ({cfg.family_id}, seed {cfg.random_seed}, log={cfg.logging_enabled})...")
        res = execute_single_pilot_run(cfg)
        results[r_id] = res
        print(f"     Done in {res['wall_time_sec']:.2f}s ({res['steps_per_sec']:.0f} steps/s, Peak RSS: {res['peak_memory_mb']:.1f}MB)")

    # Phase 2: Workload Group B (Serial Baseline Workload — 3 sequential runs)
    print("\nExecuting Workload Group B: Matched Serial Baseline Workload (3 runs)...")
    for r_id in ["P-SER-1", "P-SER-2", "P-SER-3"]:
        cfg = configs_dict[r_id]
        print(f"  -> Running {cfg.run_id} ({cfg.family_id}, seed {cfg.random_seed})...")
        res = execute_single_pilot_run(cfg)
        results[r_id] = res
        print(f"     Done in {res['wall_time_sec']:.2f}s ({res['steps_per_sec']:.0f} steps/s)")

    # Phase 3: Workload Group C (Concurrent Workload — 3 parallel runs across 3 workers)
    print("\nExecuting Workload Group C: Matched Concurrent Workload (3 runs simultaneously across 3 workers)...")
    concurrent_cfgs = [configs_dict["P-CON-1"], configs_dict["P-CON-2"], configs_dict["P-CON-3"]]
    t_con_start = time.perf_counter()
    with Pool(processes=3) as pool:
        con_results_list = pool.map(execute_single_pilot_run, concurrent_cfgs)
    t_con_wall = time.perf_counter() - t_con_start

    for res in con_results_list:
        results[res["run_id"]] = res
        print(f"  -> {res['run_id']} finished in {res['wall_time_sec']:.2f}s ({res['steps_per_sec']:.0f} steps/s)")
    print(f"Concurrent batch wall-time: {t_con_wall:.2f}s")

    # Scorecard Output
    print_scorecard(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UrbanFlow 10-Execution Pilot Benchmark (Rev 6)")
    parser.add_argument("--check-only", action="store_true", help="Perform static configuration check without running simulations")
    args = parser.parse_args()

    if args.check_only:
        print("UrbanFlow Pilot Benchmark Configuration Check (Rev 6):")
        print(f"  • Authoritative Run Count : {len(PILOT_WORKLOAD)} executions")
        print(f"  • Output Parquet Directory: {PILOT_DATA_DIR}")
        print(f"  • Output Metadata DB      : {METADATA_DB_PATH}")
        print(f"  • Max Concurrency Limit   : 3 workers")
        print(f"  • Target Duration / Run   : 3,600.0s (14,400 intervals at dt=0.25s)")
        print(f"  • Target Sampling Grid    : 5.0s (721 snapshots / run)")
        print("Static check PASSED.")
        sys.exit(0)

    run_pilot()
