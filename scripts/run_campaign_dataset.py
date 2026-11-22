"""UrbanFlow AI — 60-Run Validation Campaign Dataset Orchestrator (Rev 6).

Implements Section 2, Section 5, and Section 6 of forecasting_validation_campaign_plan.md (Rev 6).

Executes the exact 60-run matrix:
- Families 1-4: 6 seeds each for Train (24 runs)
- Families 1-4: 2 seeds each for Val (8 runs)
- Families 1-4: 2 seeds each for In-Dist Test (8 runs)
- Families 5-6: 10 seeds each for Held-Out Test (20 runs)
Total = 60 simulation runs.

Specifications:
- Duration: 3,600s (14,400 intervals at dt=0.25s)
- Telemetry: 5.0s sampling (721 snapshots / run) across 210 roads = 151,410 rows / run
- Decoupled SHA-256 PRNG streams (Philox) for driver attributes, spawn arrivals, and incident timing
- Stochastic driver heterogeneity: TruncNorm(1.0, 0.10, [0.80, 1.25])
- Resumable: Checks existing Parquet and metadata; skips already completed runs
- Storage: Parquet files to data/datasets/campaign/{family_id}/{run_id}.parquet
           Metadata index to data/datasets/campaign_metadata.db
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

URBANFLOW_ROOT = Path(__file__).resolve().parent.parent
if str(URBANFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(URBANFLOW_ROOT))

from urbanflow.core.kinematic_sim import KinematicSimulator, SimulatedVehicle
from urbanflow.intelligence.state_estimator import TrafficStateEstimator

DATA_DIR = URBANFLOW_ROOT / "data"
CAMPAIGN_DATA_DIR = DATA_DIR / "datasets" / "campaign"
CAMPAIGN_METADATA_DB = DATA_DIR / "datasets" / "campaign_metadata.db"


def derive_stream_seed(root_seed: int, stream_name: str) -> int:
    """Canonical SHA-256 PRNG seed derivation matching Section 6.2 of Rev 6."""
    payload = f"{root_seed}:{stream_name}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


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


@dataclass
class CampaignRunConfig:
    run_id: str
    family_id: str
    family_name: str
    random_seed: int
    partition: str  # 'TRAIN', 'VAL', 'IN_DIST_TEST', 'HELD_OUT_TEST'
    duration_sec: float = 3600.0
    dt_physics: float = 0.25
    sample_interval_sec: float = 5.0
    demand_multiplier: float = 1.0
    surge_start_sec: float = -1.0
    surge_duration_sec: float = 0.0
    surge_multiplier: float = 1.0
    incident_active: bool = False
    incident_start_sec: float = 1200.0
    incident_duration_sec: float = 1200.0
    incident_severity: float = 0.75
    incident_road: str = "E_Fordham_Rd_Mid"
    description: str = ""


def build_canonical_60_run_matrix() -> List[CampaignRunConfig]:
    """
    Constructs the canonical 60-run matrix locked in Section 2.1 of Rev 6:
    - FAM-01: Normal Baseline Equilibrium (Seeds 1001-1010)
    - FAM-02: Peak Commute Surge Wave (Seeds 2001-2010)
    - FAM-03: Bi-Directional Corridor Pulse (Seeds 3001-3010)
    - FAM-04: Mid-Block Lane Closure (Seeds 4001-4010)
    - FAM-05: Downstream Intersection Gridlock (Seeds 5001-5010) [Held-Out]
    - FAM-06: Cascading Multi-Incident (Seeds 6001-6010) [Held-Out]
    """
    runs: List[CampaignRunConfig] = []

    # Helper to assign partition for standard families (1-4)
    def get_standard_partition(seed_offset: int) -> str:
        # seed_offset: 1..10
        if 1 <= seed_offset <= 6:
            return "TRAIN"
        elif 7 <= seed_offset <= 8:
            return "VAL"
        else:
            return "IN_DIST_TEST"

    # Family 1: Normal Baseline Equilibrium
    for idx in range(1, 11):
        seed = 1000 + idx
        runs.append(CampaignRunConfig(
            run_id=f"RUN_FAM01_S{seed}",
            family_id="FAM-01",
            family_name="Normal Baseline Equilibrium",
            random_seed=seed,
            partition=get_standard_partition(idx),
            demand_multiplier=1.0,
            description="Free-flow and moderate demand baseline without incidents.",
        ))

    # Family 2: Peak Commute Surge Wave
    for idx in range(1, 11):
        seed = 2000 + idx
        runs.append(CampaignRunConfig(
            run_id=f"RUN_FAM02_S{seed}",
            family_id="FAM-02",
            family_name="Peak Commute Surge Wave",
            random_seed=seed,
            partition=get_standard_partition(idx),
            demand_multiplier=1.2,
            surge_start_sec=1200.0,
            surge_duration_sec=900.0,
            surge_multiplier=1.8,
            description="Commute demand surge wave between t=1200s and 2100s.",
        ))

    # Family 3: Bi-Directional Corridor Pulse
    for idx in range(1, 11):
        seed = 3000 + idx
        runs.append(CampaignRunConfig(
            run_id=f"RUN_FAM03_S{seed}",
            family_id="FAM-03",
            family_name="Bi-Directional Corridor Pulse",
            random_seed=seed,
            partition=get_standard_partition(idx),
            demand_multiplier=1.3,
            surge_start_sec=1000.0,
            surge_duration_sec=1200.0,
            surge_multiplier=1.5,
            description="Recurring arterial traffic pulses.",
        ))

    # Family 4: Mid-Block Lane Closure
    for idx in range(1, 11):
        seed = 4000 + idx
        runs.append(CampaignRunConfig(
            run_id=f"RUN_FAM04_S{seed}",
            family_id="FAM-04",
            family_name="Mid-Block Lane Closure",
            random_seed=seed,
            partition=get_standard_partition(idx),
            demand_multiplier=1.1,
            incident_active=True,
            incident_start_sec=1200.0,
            incident_duration_sec=1200.0,
            incident_severity=0.75,
            incident_road="E_Fordham_Rd_Mid",
            description="Mid-block arterial lane drop on E Fordham Rd from t=1200s to 2400s.",
        ))

    # Family 5: Downstream Intersection Gridlock (Held-Out Test 100%)
    for idx in range(1, 11):
        seed = 5000 + idx
        runs.append(CampaignRunConfig(
            run_id=f"RUN_FAM05_S{seed}",
            family_id="FAM-05",
            family_name="Downstream Intersection Gridlock",
            random_seed=seed,
            partition="HELD_OUT_TEST",
            demand_multiplier=1.25,
            incident_active=True,
            incident_start_sec=1200.0,
            incident_duration_sec=1000.0,
            incident_severity=0.95,
            incident_road="E_Fordham_Rd_Mid",
            description="Catastrophic downstream intersection blockage creating spillback.",
        ))

    # Family 6: Cascading Multi-Incident (Held-Out Test 100%)
    for idx in range(1, 11):
        seed = 6000 + idx
        runs.append(CampaignRunConfig(
            run_id=f"RUN_FAM06_S{seed}",
            family_id="FAM-06",
            family_name="Cascading Multi-Incident",
            random_seed=seed,
            partition="HELD_OUT_TEST",
            demand_multiplier=1.3,
            incident_active=True,
            incident_start_sec=1000.0,
            incident_duration_sec=1500.0,
            incident_severity=0.80,
            incident_road="E_Fordham_Rd_Mid",
            description="Multi-corridor cascading bottleneck shockwaves.",
        ))

    return runs


def init_campaign_metadata_db(db_path: Path):
    """Initializes the SQLite metadata schema for the 60-run campaign."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS campaign_runs (
            run_id TEXT PRIMARY KEY,
            family_id TEXT,
            family_name TEXT,
            random_seed INTEGER,
            stream_seed_driver INTEGER,
            stream_seed_spawn INTEGER,
            stream_seed_incident INTEGER,
            partition TEXT,
            prng_algorithm TEXT,
            duration_sec REAL,
            physics_steps INTEGER,
            wall_time_sec REAL,
            steps_per_sec REAL,
            peak_memory_mb REAL,
            parquet_path TEXT,
            parquet_size_bytes INTEGER,
            observation_count INTEGER,
            status TEXT,
            mean_driver_speed_mult REAL,
            std_driver_speed_mult REAL,
            config_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def is_run_completed(run_id: str, db_path: Path, campaign_dir: Path) -> bool:
    """
    Verifies whether a run has already successfully landed on disk and in metadata.
    Reads Parquet metadata/table to ensure the file is complete, valid, and not corrupted.
    """
    if not db_path.exists():
        return False
    try:
        conn = sqlite3.connect(str(db_path))
        c = conn.cursor()
        c.execute("SELECT parquet_path, observation_count, status FROM campaign_runs WHERE run_id = ?", (run_id,))
        row = c.fetchone()
        conn.close()
        if row and row[2] == "SUCCESS" and row[1] == 151410:
            p_file = Path(row[0])
            if p_file.exists() and p_file.stat().st_size > 500000:
                # Rigorous check: verify Parquet schema and exact row count to prevent accepting corrupted files
                meta = pq.read_metadata(str(p_file))
                if meta.num_rows == 151410 and meta.num_columns >= 17:
                    return True
    except Exception:
        pass
    return False


def execute_campaign_run(cfg: CampaignRunConfig) -> Dict[str, Any]:
    """Executes a single 3,600s campaign simulation run and saves Parquet observations."""
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    # Destination paths
    family_dir = CAMPAIGN_DATA_DIR / cfg.family_id
    family_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = family_dir / f"{cfg.run_id}.parquet"

    t_start = time.perf_counter()

    # Decoupled SHA-256 PRNG streams
    stream_driver = derive_stream_seed(cfg.random_seed, "driver_attributes")
    stream_spawn = derive_stream_seed(cfg.random_seed, "spawn_arrivals")
    stream_inc = derive_stream_seed(cfg.random_seed, "incident_timing")

    rng_driver = Generator(Philox(stream_driver))
    rng_spawn = Generator(Philox(stream_spawn))
    rng_incident = Generator(Philox(stream_inc))

    # Instantiate Simulator with decoupled RNGs
    sim = KinematicSimulator(map_name="bronx_fordham", rng_spawn=rng_spawn, rng_driver=rng_driver)
    estimator = TrafficStateEstimator(sample_interval_sec=cfg.sample_interval_sec)

    base_spawn_interval = 0.8 / max(0.1, cfg.demand_multiplier)
    sim.spawn_interval = base_spawn_interval

    driver_multipliers: List[float] = []
    observations: List[Dict[str, Any]] = []

    total_steps = int(round(cfg.duration_sec / cfg.dt_physics))  # 14,400 intervals
    sample_every_n = int(round(cfg.sample_interval_sec / cfg.dt_physics))  # 20 steps

    incident_road_id = cfg.incident_road
    # Jitter incident onset slightly using rng_incident stream if incident is active
    incident_start = cfg.incident_start_sec
    if cfg.incident_active and cfg.incident_start_sec > 0:
        jitter = float(rng_incident.uniform(-30.0, 30.0))
        incident_start = max(650.0, cfg.incident_start_sec + jitter)

    # Pre-sample multipliers for initial fleet
    for veh in sim.vehicles.values():
        mult = float(np.clip(rng_driver.normal(1.0, 0.10), 0.80, 1.25))
        veh.desired_speed *= mult
        driver_multipliers.append(mult)

    # Main stepping loop
    for step_idx in range(total_steps):
        current_time = step_idx * cfg.dt_physics

        # Demand surge handling
        if cfg.surge_start_sec > 0:
            if cfg.surge_start_sec <= current_time < (cfg.surge_start_sec + cfg.surge_duration_sec):
                sim.spawn_interval = max(0.25, base_spawn_interval / cfg.surge_multiplier)
            else:
                sim.spawn_interval = base_spawn_interval

        # Incident status
        is_incident_now = False
        if cfg.incident_active and incident_road_id:
            if incident_start <= current_time < (incident_start + cfg.incident_duration_sec):
                is_incident_now = True

        # Advance physics step
        sim.step(dt=cfg.dt_physics)

        # Telemetry logging (every 5.0 seconds = 20 steps, or final step)
        if step_idx % sample_every_n == 0 or step_idx == total_steps - 1:
            log_time = round(step_idx * cfg.dt_physics, 2)
            is_warmup = (log_time < 600.0)

            vehs_by_road: Dict[str, List[SimulatedVehicle]] = {r: [] for r in sim.roads}
            for v in sim.vehicles.values():
                r_id = v.current_road_id
                if r_id in vehs_by_road:
                    vehs_by_road[r_id].append(v)

            for road_id, road in sim.roads.items():
                road_vehs = vehs_by_road.get(road_id, [])
                incident_sev = cfg.incident_severity if (is_incident_now and road_id == incident_road_id) else 0.0

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
                    "family_id": cfg.family_id,
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

    # Write Parquet with snappy compression
    df = pd.DataFrame(observations)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, str(parquet_path), compression="snappy")
    parquet_size_bytes = parquet_path.stat().st_size

    mean_mult = float(np.mean(driver_multipliers)) if driver_multipliers else 1.0
    std_mult = float(np.std(driver_multipliers)) if driver_multipliers else 0.0

    result = {
        "run_id": cfg.run_id,
        "family_id": cfg.family_id,
        "family_name": cfg.family_name,
        "random_seed": cfg.random_seed,
        "stream_seed_driver": stream_driver,
        "stream_seed_spawn": stream_spawn,
        "stream_seed_incident": stream_inc,
        "partition": cfg.partition,
        "prng_algorithm": "NumPy-Philox",
        "duration_sec": cfg.duration_sec,
        "physics_steps": total_steps,
        "wall_time_sec": round(t_wall, 3),
        "steps_per_sec": round(steps_per_sec, 1),
        "peak_memory_mb": round(peak_mem_mb, 2),
        "parquet_path": str(parquet_path),
        "parquet_size_bytes": parquet_size_bytes,
        "observation_count": len(observations),
        "status": "SUCCESS",
        "mean_driver_speed_mult": round(mean_mult, 4),
        "std_driver_speed_mult": round(std_mult, 4),
        "config_json": json.dumps(asdict(cfg)),
    }

    # Record in campaign metadata DB
    conn = sqlite3.connect(str(CAMPAIGN_METADATA_DB))
    c = conn.cursor()
    c.execute("""
        INSERT OR REPLACE INTO campaign_runs (
            run_id, family_id, family_name, random_seed, stream_seed_driver, stream_seed_spawn,
            stream_seed_incident, partition, prng_algorithm, duration_sec, physics_steps,
            wall_time_sec, steps_per_sec, peak_memory_mb, parquet_path, parquet_size_bytes,
            observation_count, status, mean_driver_speed_mult, std_driver_speed_mult, config_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        result["run_id"], result["family_id"], result["family_name"], result["random_seed"],
        result["stream_seed_driver"], result["stream_seed_spawn"], result["stream_seed_incident"],
        result["partition"], result["prng_algorithm"], result["duration_sec"], result["physics_steps"],
        result["wall_time_sec"], result["steps_per_sec"], result["peak_memory_mb"],
        result["parquet_path"], result["parquet_size_bytes"], result["observation_count"],
        result["status"], result["mean_driver_speed_mult"], result["std_driver_speed_mult"],
        result["config_json"]
    ))
    conn.commit()
    conn.close()

    return result


def main():
    parser = argparse.ArgumentParser(description="UrbanFlow 60-Run Campaign Dataset Generator (Rev 6)")
    parser.add_argument("--dry-run", action="store_true", help="Print plan and verify matrix without executing runs")
    parser.add_argument("--workers", type=int, default=3, choices=[1, 2, 3], help="Parallel worker processes (max 3)")
    parser.add_argument("--run-id", type=str, default="", help="Execute a specific run ID only")
    args = parser.parse_args()

    matrix = build_canonical_60_run_matrix()
    init_campaign_metadata_db(CAMPAIGN_METADATA_DB)

    print("=" * 80)
    print("URBANFLOW AI — 60-RUN FORECASTING VALIDATION CAMPAIGN GENERATOR (REV 6)")
    print("=" * 80)
    print(f"Total Canonical Matrix : {len(matrix)} runs")
    print(f"Storage Directory      : {CAMPAIGN_DATA_DIR}")
    print(f"Metadata Database      : {CAMPAIGN_METADATA_DB}")
    print(f"Concurrency Bound      : {args.workers} workers (Max 3)")

    # Partition Summary
    part_counts = {}
    for r in matrix:
        part_counts[r.partition] = part_counts.get(r.partition, 0) + 1
    print("\nPartition Allocation:")
    for p, count in sorted(part_counts.items()):
        print(f"  • {p:<15}: {count} runs")

    if args.dry_run:
        print("\n[DRY RUN MODE]: Matrix verified. Zero simulations executed.")
        return

    # Filter completed runs (Resumability)
    pending_runs = [r for r in matrix if not is_run_completed(r.run_id, CAMPAIGN_METADATA_DB, CAMPAIGN_DATA_DIR)]
    if args.run_id:
        pending_runs = [r for r in pending_runs if r.run_id == args.run_id]

    print(f"\nExecution Status:")
    print(f"  • Already Completed : {len(matrix) - len(pending_runs)} runs")
    print(f"  • Pending Execution : {len(pending_runs)} runs")

    if not pending_runs:
        print("All campaign runs are already successfully generated and verified.")
        return

    # Execute pending runs
    print(f"\nLaunching {len(pending_runs)} runs using {args.workers} worker processes...")
    t_start = time.perf_counter()
    if args.workers == 1:
        for r in pending_runs:
            print(f"--> Starting {r.run_id} ({r.family_id}, seed {r.random_seed}, {r.partition})...")
            res = execute_campaign_run(r)
            print(f"    Finished {r.run_id} in {res['wall_time_sec']}s ({res['steps_per_sec']} steps/s)")
    else:
        with Pool(processes=args.workers) as pool:
            for res in pool.imap_unordered(execute_campaign_run, pending_runs):
                print(f"--> Completed {res['run_id']} ({res['family_id']}, seed {res['random_seed']}) in {res['wall_time_sec']}s")

    t_total = time.perf_counter() - t_start
    print(f"\nFinished batch in {t_total:.2f}s wall-time.")


if __name__ == "__main__":
    main()
