"""UrbanFlow AI — Counterfactual Dynamic Routing Benchmark Orchestrator (Phase 4).

Executes paired counterfactual benchmarks across three experimental regimes:
  A. Baseline: No dynamic rerouting (static distance-based routes).
  B. Reactive: Dynamic routing using observed V/C and queue telemetry (risk weight = 0).
  C. Predictive Capacity-Aware: Dynamic routing using observed V/C, queue telemetry,
     and advisory spillback risk from the frozen 10-minute network classifier.

Key Principles:
  1. Paired counterfactual reproducibility:
     Decoupled Philox PRNG streams ensure identical vehicle arrivals, driver heterogeneity,
     origin-destination demands, and incident schedules across all three regimes.
  2. Exogenous input verification:
     Verifies bit-for-bit equality of inputs before comparing performance outcomes.
  3. Strict accounting:
     All outputs written to data/datasets/dynamic_routing_benchmark/.
     No modification of existing 60-run campaign files or baseline models.
  4. Explicit failure handling:
     Corrupt or partial runs are never silently ignored; status is explicitly tracked in SQLite.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import sqlite3
import sys
import time
from typing import Dict, Any, List, Optional, Set, Tuple

import numpy as np
from numpy.random import Generator, Philox
import pandas as pd

URBANFLOW_ROOT = Path(__file__).resolve().parent.parent
if str(URBANFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(URBANFLOW_ROOT))

# Safe Windows imports
import lightgbm as lgb  # noqa: F401

from urbanflow.core.kinematic_sim import KinematicSimulator, SimulatedVehicle
from urbanflow.intelligence.routing import RouteCostEvaluator, PredictiveRoutingController
from urbanflow.intelligence.state_estimator import TrafficStateEstimator
from urbanflow.intelligence.benchmark_kpis import RoutingKPIEvaluator, RoutingKPISummary

logger = logging.getLogger("dynamic_routing_benchmark")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DATA_DIR = URBANFLOW_ROOT / "data"
BENCHMARK_OUTPUT_DIR = DATA_DIR / "datasets" / "dynamic_routing_benchmark"
BENCHMARK_METADATA_DB = BENCHMARK_OUTPUT_DIR / "benchmark_metadata.db"
REPORTS_DIR = URBANFLOW_ROOT / "docs" / "reports"

CLASSIFIER_10MIN_PATH = (
    DATA_DIR / "models" / "campaign_network_classifiers" / "lgb_network_spillback_classifier_10min.joblib"
)
EXPECTED_CLASSIFIER_SHA256 = (
    "9B93E5DB672F692E509DC3012658413BC713FC6348CCD5D044B33470FF568C54"
)


class ExogenousMismatchError(RuntimeError):
    """Raised when counterfactual exogenous inputs do not match across regimes."""
    pass


@dataclass
class BenchmarkScenarioConfig:
    """Configuration for a paired counterfactual scenario evaluation."""
    scenario_id: str
    family_id: str
    family_name: str
    seed: int
    duration_sec: float = 3600.0
    dt_physics: float = 0.25
    sample_interval_sec: float = 5.0
    eval_interval_sec: float = 30.0
    demand_multiplier: float = 1.0
    surge_start_sec: float = -1.0
    surge_duration_sec: float = 0.0
    surge_multiplier: float = 1.0
    incident_active: bool = False
    incident_start_sec: float = 1200.0
    incident_duration_sec: float = 1200.0
    incident_severity: float = 0.75
    incident_road: str = "road_586133572_120_fwd"
    description: str = ""

    def get_config_hash(self) -> str:
        d = asdict(self)
        d_str = json.dumps(d, sort_keys=True)
        return hashlib.sha256(d_str.encode("utf-8")).hexdigest()[:16]


def derive_stream_seed(root_seed: int, stream_name: str) -> int:
    """Canonical SHA-256 PRNG seed derivation matching Rev 6."""
    payload = f"{root_seed}:{stream_name}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def get_held_out_scenarios() -> List[BenchmarkScenarioConfig]:
    """Constructs the canonical 20 held-out evaluation scenarios (Families 5 and 6)."""
    scenarios: List[BenchmarkScenarioConfig] = []

    # Family 5: Downstream Intersection Gridlock (Seeds 5001-5010)
    for idx in range(1, 11):
        seed = 5000 + idx
        scenarios.append(BenchmarkScenarioConfig(
            scenario_id=f"SCENARIO_FAM05_S{seed}",
            family_id="FAM-05",
            family_name="Downstream Intersection Gridlock",
            seed=seed,
            demand_multiplier=1.25,
            incident_active=True,
            incident_start_sec=1200.0,
            incident_duration_sec=1000.0,
            incident_severity=0.95,
            incident_road="road_1082220764_167_fwd",
            description="Catastrophic downstream arterial blockage creating spillback.",
        ))

    # Family 6: Cascading Multi-Incident (Seeds 6001-6010)
    for idx in range(1, 11):
        seed = 6000 + idx
        scenarios.append(BenchmarkScenarioConfig(
            scenario_id=f"SCENARIO_FAM06_S{seed}",
            family_id="FAM-06",
            family_name="Cascading Multi-Incident",
            seed=seed,
            demand_multiplier=1.30,
            incident_active=True,
            incident_start_sec=1000.0,
            incident_duration_sec=1500.0,
            incident_severity=0.80,
            incident_road="road_1082220763_166_fwd",
            description="Multi-corridor cascading bottleneck shockwaves.",
        ))

    return scenarios


def init_benchmark_metadata_db(db_path: Path):
    """Initializes SQLite schema for tracking dynamic routing benchmark runs."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS benchmark_runs (
            run_id TEXT PRIMARY KEY,
            scenario_id TEXT,
            family_id TEXT,
            seed INTEGER,
            regime TEXT,
            status TEXT,  -- 'SUCCESS', 'FAILED', 'INCOMPLETE'
            duration_sec REAL,
            eval_duration_sec REAL,
            completed_trips INTEGER,
            incomplete_trips INTEGER,
            network_throughput_veh_hr REAL,
            mean_travel_time_sec REAL,
            mean_delay_sec REAL,
            mean_waiting_time_sec REAL,
            mean_queue_length_m REAL,
            max_queue_length_m REAL,
            spillback_events INTEGER,
            spillback_duration_sec REAL,
            total_route_changes INTEGER,
            diversion_rate REAL,
            mean_operational_vc REAL,
            peak_operational_vc REAL,
            alternative_vkt_km REAL,
            config_hash TEXT,
            model_hash TEXT,
            summary_json_path TEXT,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def is_benchmark_run_complete(run_id: str, db_path: Path, output_dir: Path) -> bool:
    """Verifies whether a benchmark run has already successfully landed on disk."""
    if not db_path.exists():
        return False
    try:
        conn = sqlite3.connect(str(db_path))
        c = conn.cursor()
        c.execute("SELECT status, summary_json_path FROM benchmark_runs WHERE run_id = ?", (run_id,))
        row = c.fetchone()
        conn.close()
        if row and row[0] == "SUCCESS":
            json_file = Path(row[1])
            if json_file.exists() and json_file.stat().st_size > 50:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("run_id") == run_id and "mean_travel_time_sec" in data:
                    return True
    except Exception:
        pass
    return False


def verify_exogenous_matching(
    trace_a: Dict[str, Any],
    trace_b: Dict[str, Any],
    trace_c: Optional[Dict[str, Any]] = None,
) -> bool:
    """Verifies bit-for-bit equality of exogenous vehicle generation and incidents across regimes."""
    # 1. Compare Seeds
    if trace_a["stream_seeds"] != trace_b["stream_seeds"]:
        raise ExogenousMismatchError(
            f"PRNG Stream seeds mismatch between Baseline and Reactive: {trace_a['stream_seeds']} vs {trace_b['stream_seeds']}"
        )
    if trace_c and trace_a["stream_seeds"] != trace_c["stream_seeds"]:
        raise ExogenousMismatchError(
            f"PRNG Stream seeds mismatch between Baseline and Predictive: {trace_a['stream_seeds']} vs {trace_c['stream_seeds']}"
        )

    # 2. Compare Initial Fleet at t=0
    if trace_a["initial_fleet"] != trace_b["initial_fleet"]:
        raise ExogenousMismatchError("Initial vehicle fleet at t=0 mismatch between Baseline and Reactive.")
    if trace_c and trace_a["initial_fleet"] != trace_c["initial_fleet"]:
        raise ExogenousMismatchError("Initial vehicle fleet at t=0 mismatch between Baseline and Predictive.")

    # 3. Compare Incident Schedule
    if trace_a["incident"] != trace_b["incident"]:
        raise ExogenousMismatchError("Incident onset/severity schedule mismatch between Baseline and Reactive.")
    if trace_c and trace_a["incident"] != trace_c["incident"]:
        raise ExogenousMismatchError("Incident onset/severity schedule mismatch between Baseline and Predictive.")

    # 4. Compare Exogenous Spawn Attributes for common arrivals matched by Vehicle ID
    spawns_a = {v["id"]: v for v in trace_a["spawns"]}
    spawns_b = {v["id"]: v for v in trace_b["spawns"]}
    spawns_c = {v["id"]: v for v in trace_c["spawns"]} if trace_c else None

    def _sort_key(k: str):
        match = re.search(r"\d+", k)
        return int(match.group()) if match else k

    common_ids_ab = sorted(set(spawns_a.keys()) & set(spawns_b.keys()), key=_sort_key)
    if not common_ids_ab:
        raise ExogenousMismatchError("No common spawned vehicles found between Baseline and Reactive.")
    for v_id in common_ids_ab:
        sa, sb = spawns_a[v_id], spawns_b[v_id]
        if sa["type"] != sb["type"] or sa["desired_speed"] != sb["desired_speed"] or sa["initial_route"] != sb["initial_route"]:
            raise ExogenousMismatchError(
                f"Exogenous vehicle attribute mismatch for {v_id} between Baseline and Reactive: {sa} vs {sb}"
            )

    if spawns_c:
        common_ids_ac = sorted(set(spawns_a.keys()) & set(spawns_c.keys()), key=_sort_key)
        if not common_ids_ac:
            raise ExogenousMismatchError("No common spawned vehicles found between Baseline and Predictive.")
        for v_id in common_ids_ac:
            sa, sc = spawns_a[v_id], spawns_c[v_id]
            if sa["type"] != sc["type"] or sa["desired_speed"] != sc["desired_speed"] or sa["initial_route"] != sc["initial_route"]:
                raise ExogenousMismatchError(
                    f"Exogenous vehicle attribute mismatch for {v_id} with Predictive: {sa} vs {sc}"
                )

    # 5. Compare Scheduled Spawns (100% deterministic arrival stream)
    sched_a = trace_a.get("scheduled_spawns")
    sched_b = trace_b.get("scheduled_spawns")
    sched_c = trace_c.get("scheduled_spawns") if trace_c else None
    if sched_a and sched_b:
        if len(sched_a) != len(sched_b):
            raise ExogenousMismatchError(
                f"Scheduled spawns count mismatch between Baseline ({len(sched_a)}) and Reactive ({len(sched_b)})."
            )
        for sa, sb in zip(sched_a, sched_b):
            if sa != sb:
                raise ExogenousMismatchError(
                    f"Scheduled spawn mismatch between Baseline and Reactive: {sa} vs {sb}"
                )
    if sched_a and sched_c:
        if len(sched_a) != len(sched_c):
            raise ExogenousMismatchError(
                f"Scheduled spawns count mismatch between Baseline ({len(sched_a)}) and Predictive ({len(sched_c)})."
            )
        for sa, sc in zip(sched_a, sched_c):
            if sa != sc:
                raise ExogenousMismatchError(
                    f"Scheduled spawn mismatch between Baseline and Predictive: {sa} vs {sc}"
                )

    return True


def execute_single_regime_run(
    scenario: BenchmarkScenarioConfig,
    regime: str,
    output_dir: Path,
    db_path: Path,
) -> Tuple[RoutingKPISummary, Dict[str, Any]]:
    """Executes a single simulation run under the specified routing regime.

    Args:
        scenario: Scenario parameters and demand configuration.
        regime: 'baseline', 'reactive', or 'predictive'.
        output_dir: Destination path for results.
        db_path: SQLite metadata database.

    Returns:
        Tuple of (RoutingKPISummary, exogenous_trace).
    """
    run_id = f"RUN_{scenario.family_id}_S{scenario.seed}_{regime.upper()}"
    family_dir = output_dir / scenario.family_id
    family_dir.mkdir(parents=True, exist_ok=True)
    summary_json_path = family_dir / f"{run_id}_summary.json"

    # Cryptographic model hash check
    model_sha256 = "N/A"
    if regime == "predictive":
        if not CLASSIFIER_10MIN_PATH.exists():
            raise FileNotFoundError(f"10-minute network classifier missing at {CLASSIFIER_10MIN_PATH}")
        model_sha256 = hashlib.sha256(CLASSIFIER_10MIN_PATH.read_bytes()).hexdigest().upper()
        if model_sha256 != EXPECTED_CLASSIFIER_SHA256:
            raise ValueError(f"Classifier SHA-256 tampered: expected {EXPECTED_CLASSIFIER_SHA256}, got {model_sha256}")

    # Decoupled PRNG streams
    stream_driver = derive_stream_seed(scenario.seed, "driver_attributes")
    stream_spawn = derive_stream_seed(scenario.seed, "spawn_arrivals")
    stream_incident = derive_stream_seed(scenario.seed, "incident_timing")

    rng_driver = Generator(Philox(stream_driver))
    rng_spawn = Generator(Philox(stream_spawn))
    rng_incident = Generator(Philox(stream_incident))

    # Instantiate Simulator
    sim = KinematicSimulator(
        map_name="bronx_fordham",
        rng_spawn=rng_spawn,
        rng_driver=rng_driver,
    )
    base_spawn_interval = 0.8 / max(0.1, scenario.demand_multiplier)
    sim.spawn_interval = base_spawn_interval

    estimator = TrafficStateEstimator(sample_interval_sec=scenario.sample_interval_sec)

    # Instantiate Routing Controller (if not baseline)
    controller: Optional[PredictiveRoutingController] = None
    if regime != "baseline":
        cost_evaluator = RouteCostEvaluator(simulator=sim, mode=regime)
        controller = PredictiveRoutingController(
            route_evaluator=cost_evaluator,
            mode=regime,
            classifier_path=CLASSIFIER_10MIN_PATH if regime == "predictive" else None,
            max_diversion_quota=0.25,
            vc_admission_cap=0.80,
            cooldown_sec=180.0,
            min_improvement_sec=45.0,
            evaluation_interval_sec=scenario.eval_interval_sec,
            strict_quota_floor=True,
            allow_node_destination=True,
        )

    # Track exogenous inputs for reproducibility assertion
    exogenous_trace: Dict[str, Any] = {
        "stream_seeds": {
            "driver": stream_driver,
            "spawn": stream_spawn,
            "incident": stream_incident,
        },
        "initial_fleet": [
            {
                "id": v.id,
                "type": v.type,
                "initial_route": list(v.route),
                "desired_speed": round(v.desired_speed, 4),
            }
            for v in sim.vehicles.values()
        ],
        "spawns": [],
        "incident": {},
    }

    # Incident setup
    incident_start = scenario.incident_start_sec
    if scenario.incident_active and scenario.incident_start_sec > 0:
        jitter = float(rng_incident.uniform(-30.0, 30.0))
        incident_start = max(650.0, scenario.incident_start_sec + jitter)
        exogenous_trace["incident"] = {
            "road": scenario.incident_road,
            "severity": scenario.incident_severity,
            "start_sec": round(incident_start, 2),
            "duration_sec": scenario.incident_duration_sec,
        }

    total_steps = int(round(scenario.duration_sec / scenario.dt_physics))
    sample_every_n = int(round(scenario.sample_interval_sec / scenario.dt_physics))
    eval_every_n = int(round(scenario.eval_interval_sec / scenario.dt_physics))

    telemetry: List[Dict[str, Any]] = []
    latest_telemetry_map: Dict[str, Any] = {}
    routing_decisions_log: List[Dict[str, Any]] = []

    known_veh_ids: Set[str] = {v.id for v in sim.vehicles.values()}

    try:
        for step_idx in range(total_steps):
            current_time = round(step_idx * scenario.dt_physics, 2)

            # Demand surge modulation
            if scenario.surge_start_sec > 0:
                if scenario.surge_start_sec <= current_time < (scenario.surge_start_sec + scenario.surge_duration_sec):
                    sim.spawn_interval = max(0.25, base_spawn_interval / scenario.surge_multiplier)
                else:
                    sim.spawn_interval = base_spawn_interval

            # Incident status
            is_incident_now = False
            if scenario.incident_active and scenario.incident_road:
                if incident_start <= current_time < (incident_start + scenario.incident_duration_sec):
                    is_incident_now = True

            # Physics step
            sim.step(dt=scenario.dt_physics)

            # Record newly spawned vehicles for exogenous audit
            for v_id, v in sim.vehicles.items():
                if v_id not in known_veh_ids:
                    known_veh_ids.add(v_id)
                    exogenous_trace["spawns"].append({
                        "id": v.id,
                        "time": current_time,
                        "type": v.type,
                        "initial_route": list(v.route),
                        "desired_speed": round(v.desired_speed, 4),
                    })

            # Telemetry Sampling (every 5.0s)
            if step_idx % sample_every_n == 0 or step_idx == total_steps - 1:
                vehs_by_road: Dict[str, List[SimulatedVehicle]] = {r: [] for r in sim.roads}
                for v in sim.vehicles.values():
                    if v.current_road_id in vehs_by_road:
                        vehs_by_road[v.current_road_id].append(v)

                latest_telemetry_map.clear()
                for road_id, road in sim.roads.items():
                    road_vehs = vehs_by_road.get(road_id, [])
                    sev = scenario.incident_severity if (is_incident_now and road_id == scenario.incident_road) else 0.0

                    state = estimator.estimate_segment_state(
                        road_id=road_id,
                        timestamp=current_time,
                        road_length=road.length,
                        lanes=road.lanes,
                        speed_limit_mps=road.speed_limit,
                        vehicles_on_road=road_vehs,
                        downstream_signal_state="NONE",
                        incident_severity=sev,
                    )
                    obs = {
                        "road_id": road_id,
                        "timestamp": current_time,
                        "vehicle_count": state.vehicle_count,
                        "density_per_km_lane": state.density_per_km_lane,
                        "occupancy_ratio": state.occupancy_ratio,
                        "avg_speed_kmh": state.average_speed_kmh,
                        "flow_veh_hr": state.flow_veh_per_hour,
                        "queue_length_m": state.queue_length_meters,
                        "operational_capacity": state.operational_capacity,
                        "vc_ratio": state.capacity_utilization,
                        "incident_active": (is_incident_now and road_id == scenario.incident_road),
                    }
                    telemetry.append(obs)
                    latest_telemetry_map[road_id] = obs

                if controller:
                    controller.update_telemetry(latest_telemetry_map, timestamp=current_time)

            # Routing Controller Evaluation Window (every 30.0s)
            if controller and (step_idx % eval_every_n == 0) and step_idx > 0:
                eval_res = controller.evaluate_and_reroute(
                    sim,
                    current_time=current_time,
                    telemetry_map=latest_telemetry_map,
                )
                if eval_res.get("eligible_candidates_count", 0) > 0:
                    routing_decisions_log.append(eval_res)

        # Calculate KPIs using RoutingKPIEvaluator
        evaluator = RoutingKPIEvaluator(roads=sim.roads)
        active_veh_dicts = [
            {
                "vehicle_id": v.id,
                "wait_time": v.wait_time,
                "distance_on_road": v.distance_on_road,
                "reroute_count": v.reroute_count,
            }
            for v in sim.vehicles.values()
        ]
        never_entered = getattr(sim, "get_never_entered_vehicles", lambda: [])()
        kpi_summary = evaluator.calculate_kpis(
            completed_trips=sim.completed_trips,
            active_vehicles=active_veh_dicts,
            telemetry=telemetry,
            run_id=run_id,
            scenario_family=scenario.family_id,
            regime=regime,
            seed=scenario.seed,
            duration_sec=scenario.duration_sec,
            sample_interval_sec=scenario.sample_interval_sec,
            never_entered_vehicles=never_entered,
        )

        exogenous_trace["scheduled_spawns"] = list(getattr(sim, "all_scheduled_spawns", []))

        # Write summary JSON
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(kpi_summary.to_dict(), f, indent=2)

        # Write routing decisions diagnostic log if controller was active
        if routing_decisions_log:
            decisions_json_path = family_dir / f"{run_id}_decisions.json"
            with open(decisions_json_path, "w", encoding="utf-8") as f:
                json.dump(routing_decisions_log, f, indent=2)

        # Record in SQLite metadata DB
        conn = sqlite3.connect(str(db_path))
        c = conn.cursor()
        c.execute("""
            INSERT OR REPLACE INTO benchmark_runs (
                run_id, scenario_id, family_id, seed, regime, status,
                duration_sec, eval_duration_sec, completed_trips, incomplete_trips,
                network_throughput_veh_hr, mean_travel_time_sec, mean_delay_sec,
                mean_waiting_time_sec, mean_queue_length_m, max_queue_length_m,
                spillback_events, spillback_duration_sec, total_route_changes,
                diversion_rate, mean_operational_vc, peak_operational_vc,
                alternative_vkt_km, config_hash, model_hash, summary_json_path,
                error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            scenario.scenario_id,
            scenario.family_id,
            scenario.seed,
            regime,
            "SUCCESS",
            kpi_summary.duration_sec,
            kpi_summary.eval_duration_sec,
            kpi_summary.completed_trips_count,
            kpi_summary.incomplete_trips_count,
            kpi_summary.network_throughput_veh_hr,
            kpi_summary.mean_travel_time_sec,
            kpi_summary.mean_delay_sec,
            kpi_summary.mean_waiting_time_sec,
            kpi_summary.mean_queue_length_m,
            kpi_summary.max_queue_length_m,
            kpi_summary.spillback_event_count,
            kpi_summary.total_spillback_duration_sec,
            kpi_summary.total_route_changes,
            kpi_summary.overall_diversion_rate,
            kpi_summary.mean_operational_vc,
            kpi_summary.peak_operational_vc,
            kpi_summary.alternative_vkt_km,
            scenario.get_config_hash(),
            model_sha256,
            str(summary_json_path),
            None,
        ))
        conn.commit()
        conn.close()

        return kpi_summary, exogenous_trace

    except Exception as exc:
        err_msg = str(exc)
        logger.error(f"Benchmark run {run_id} failed: {err_msg}")
        conn = sqlite3.connect(str(db_path))
        c = conn.cursor()
        c.execute("""
            INSERT OR REPLACE INTO benchmark_runs (
                run_id, scenario_id, family_id, seed, regime, status, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (run_id, scenario.scenario_id, scenario.family_id, scenario.seed, regime, "FAILED", err_msg))
        conn.commit()
        conn.close()
        raise


def run_counterfactual_triad(
    scenario: BenchmarkScenarioConfig,
    output_dir: Path,
    db_path: Path,
    regimes: Optional[List[str]] = None,
    force: bool = False,
) -> Dict[str, RoutingKPISummary]:
    """Runs a paired triad (Baseline, Reactive, Predictive) for a single scenario seed."""
    regimes = regimes or ["baseline", "reactive", "predictive"]
    results: Dict[str, RoutingKPISummary] = {}
    traces: Dict[str, Dict[str, Any]] = {}

    for regime in regimes:
        run_id = f"RUN_{scenario.family_id}_S{scenario.seed}_{regime.upper()}"
        if not force and is_benchmark_run_complete(run_id, db_path, output_dir):
            logger.info(f"Skipping already completed run: {run_id}")
            summary_path = output_dir / scenario.family_id / f"{run_id}_summary.json"
            with open(summary_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            results[regime] = RoutingKPISummary(**data)
            continue

        logger.info(f"Executing counterfactual run: {run_id} ({scenario.family_name})...")
        kpi, trace = execute_single_regime_run(scenario, regime, output_dir, db_path)
        results[regime] = kpi
        traces[regime] = trace

    # Pre-comparison exogenous matching verification
    if len(traces) >= 2:
        trace_base = traces.get("baseline")
        trace_react = traces.get("reactive")
        trace_pred = traces.get("predictive")
        if trace_base and trace_react:
            verify_exogenous_matching(trace_base, trace_react, trace_pred)
            logger.info(f"Verified 100% exogenous input matching for {scenario.scenario_id}.")

    return results


def generate_benchmark_scorecard(
    summaries: List[Dict[str, Any]],
    output_json_path: Path,
    output_md_path: Path,
):
    """Compiles machine-readable results JSON and human-readable markdown scorecard."""
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_md_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    # Markdown scorecard generation
    lines = [
        "# UrbanFlow AI — Dynamic Routing Counterfactual Benchmark Scorecard",
        "",
        f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        "",
        "## 1. Experimental Regimes Evaluated",
        "- **Regime A (Baseline):** No dynamic rerouting (static shortest path).",
        "- **Regime B (Reactive):** Dynamic routing using observed V/C and queue telemetry (risk weight $\\lambda_{\\text{pred}} = 0$).",
        "- **Regime C (Predictive):** Predictive capacity-aware routing with advisory spillback risk from frozen 10-min classifier.",
        "",
        "## 2. Benchmark Summary Table",
        "",
        "| Scenario Family | Seed | Regime | Completed Trips | Mean Travel Time (s) | Mean Delay (s) | Throughput (veh/hr) | Spillback Events | Spillback Dur (s) | Total Reroutes | Diversion Rate | Peak V/C | Alt VKT (km) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for s in summaries:
        lines.append(
            f"| {s.get('scenario_family')} | {s.get('seed')} | {s.get('regime')} | "
            f"{s.get('completed_trips_count')} | {s.get('mean_travel_time_sec'):.1f} | "
            f"{s.get('mean_delay_sec'):.1f} | {s.get('network_throughput_veh_hr'):.1f} | "
            f"{s.get('spillback_event_count')} | {s.get('total_spillback_duration_sec'):.1f} | "
            f"{s.get('total_route_changes')} | {s.get('overall_diversion_rate'):.3f} | "
            f"{s.get('peak_operational_vc'):.2f} | {s.get('alternative_vkt_km'):.2f} |"
        )

    lines.extend([
        "",
        "## 3. Incomplete Trips & Network Integrity Accounting",
        "Trips unfinished at simulation conclusion are strictly segregated from completed trip statistics.",
        "",
    ])

    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    logger.info(f"Landed scorecard to {output_md_path} and JSON to {output_json_path}.")


def main():
    parser = argparse.ArgumentParser(description="UrbanFlow AI — Counterfactual Dynamic Routing Benchmark Orchestrator")
    parser.add_argument("--pilot", action="store_true", help="Run a single-seed pilot (Seed 5001 across all 3 regimes).")
    parser.add_argument("--family", choices=["FAM-05", "FAM-06", "all"], default="all", help="Target scenario family.")
    parser.add_argument("--seed", type=int, default=None, help="Execute a specific seed.")
    parser.add_argument("--regimes", nargs="+", choices=["baseline", "reactive", "predictive"], default=["baseline", "reactive", "predictive"])
    parser.add_argument("--dry-run", action="store_true", help="Validate configs, PRNG streams, and model hashes without running simulations.")
    parser.add_argument("--force", action="store_true", help="Rerun scenarios even if marked completed in database.")
    parser.add_argument("--output-dir", type=str, default=str(BENCHMARK_OUTPUT_DIR), help="Output directory for benchmark files.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    db_path = out_dir / "benchmark_metadata.db"
    init_benchmark_metadata_db(db_path)

    scenarios = get_held_out_scenarios()

    if args.pilot:
        scenarios = [s for s in scenarios if s.seed == 5001]
    elif args.family != "all":
        scenarios = [s for s in scenarios if s.family_id == args.family]

    if args.seed is not None:
        scenarios = [s for s in scenarios if s.seed == args.seed]

    logger.info(f"Loaded {len(scenarios)} benchmark scenario targets across regimes {args.regimes}.")

    if args.dry_run:
        logger.info("DRY-RUN: Validating model pre-load hash and PRNG seed derivation...")
        if not CLASSIFIER_10MIN_PATH.exists():
            raise FileNotFoundError(f"Missing classifier at {CLASSIFIER_10MIN_PATH}")
        h = hashlib.sha256(CLASSIFIER_10MIN_PATH.read_bytes()).hexdigest().upper()
        assert h == EXPECTED_CLASSIFIER_SHA256, f"Hash mismatch: {h} != {EXPECTED_CLASSIFIER_SHA256}"
        for sc in scenarios:
            s_d = derive_stream_seed(sc.seed, "driver_attributes")
            s_s = derive_stream_seed(sc.seed, "spawn_arrivals")
            s_i = derive_stream_seed(sc.seed, "incident_timing")
            assert len({s_d, s_s, s_i}) == 3, "PRNG stream collision detected"
        logger.info("DRY-RUN PASSED: All configs, model hashes, and PRNG streams validated successfully.")
        return

    all_summaries: List[Dict[str, Any]] = []
    for sc in scenarios:
        logger.info(f"Starting counterfactual triad execution for {sc.scenario_id} (Seed {sc.seed})...")
        triad_results = run_counterfactual_triad(
            scenario=sc,
            output_dir=out_dir,
            db_path=db_path,
            regimes=args.regimes,
            force=args.force,
        )
        for regime in args.regimes:
            if regime in triad_results:
                all_summaries.append(triad_results[regime].to_dict())

    if all_summaries:
        json_out = out_dir / "dynamic_routing_benchmark_results.json"
        md_out = out_dir / "dynamic_routing_benchmark_scorecard.md"
        generate_benchmark_scorecard(all_summaries, json_out, md_out)
        logger.info(f"Benchmark execution completed successfully for {len(scenarios)} scenario(s).")


if __name__ == "__main__":
    main()
