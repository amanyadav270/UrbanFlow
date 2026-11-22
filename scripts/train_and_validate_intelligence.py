"""Phase 6: Multi-Scenario Dataset Generation, Training, and Benchmark Suite.

Executes 5 reproducible simulation scenarios:
1. Normal Traffic
2. Rush-Hour Demand
3. Sudden Traffic Surge
4. Road Incident (Lane Closure / Breakdown)
5. Downstream Intersection Blockage

Records all observations into SQLite via DatasetRecorder, builds chronological lag datasets,
trains LightGBM multi-horizon models, and evaluates against Persistence, Historical Average,
and Moving Average baselines using MAE, RMSE, MAPE, Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
"""
from __future__ import annotations

import sys
import time
import random
import sqlite3
from pathlib import Path
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from urbanflow.core.kinematic_sim import KinematicSimulator
from urbanflow.intelligence.state_estimator import TrafficStateEstimator
from urbanflow.intelligence.capacity_estimator import RoadCapacityEstimator
from urbanflow.intelligence.dataset_recorder import DatasetRecorder, ScenarioConfig
from urbanflow.intelligence.congestion_predictor import ModularCongestionPredictor, CongestionEvaluationSuite


def run_scenario(
    config: ScenarioConfig,
    recorder: DatasetRecorder,
    estimator: TrafficStateEstimator,
    capacity_est: RoadCapacityEstimator,
) -> int:
    """Executes one deterministic simulation scenario and logs observations."""
    print(f"--> Running Scenario: {config.scenario_name} (ID: {config.scenario_id}, Seed: {config.random_seed})")
    recorder.record_scenario(config)

    random.seed(config.random_seed)
    sim = KinematicSimulator(map_name="bronx_fordham")
    sim.reset()

    # Pre-warm simulation
    sim.spawn_interval = config.base_spawn_interval
    dt = 0.25  # Highly performant physics stepping
    total_steps = int(config.duration_seconds / dt)
    log_interval_steps = int(5.0 / dt)  # Record states every 5.0 simulation seconds

    recorded_observations = 0

    # Pick arterial roads for incident injection if configured
    arterial_roads = [r.id for r in sim.roads.values() if r.lanes >= 2]
    target_incident_road = config.incident_road_id or (arterial_roads[0] if arterial_roads else "road_1")

    surge_triggered = False
    incident_applied = False

    t_start = time.perf_counter()

    for step_idx in range(total_steps):
        current_time = sim.sim_time

        # Surge handling
        if config.surge_start_sec > 0 and current_time >= config.surge_start_sec and not surge_triggered:
            print(f"    [t={current_time:.1f}s] Triggering Demand Surge (+25 vehicles)", flush=True)
            sim.trigger_congestion_surge(target_incident_road, vehicle_count=25)
            sim.spawn_interval = max(0.2, config.base_spawn_interval / config.surge_multiplier)
            surge_triggered = True

        if config.surge_start_sec > 0 and current_time >= (config.surge_start_sec + config.surge_duration_sec):
            sim.spawn_interval = config.base_spawn_interval

        # Incident handling
        if config.incident_start_sec > 0 and current_time >= config.incident_start_sec and not incident_applied:
            print(f"    [t={current_time:.1f}s] Injecting Incident on {target_incident_road} (Severity: {config.incident_severity:.1%})", flush=True)
            sim.inject_incident(target_incident_road, severity=config.incident_severity, duration_sec=config.incident_duration_sec)
            incident_applied = True

        snapshot = sim.step(dt=dt)

        # Periodic observation logging (O(V) single-pass vehicle grouping)
        if step_idx % log_interval_steps == 0:
            vehs_by_road: dict = {}
            for v in sim.vehicles.values():
                vehs_by_road.setdefault(v.current_road_id, []).append(v)

            states = []
            for r_id, road in sim.roads.items():
                vehs = vehs_by_road.get(r_id, [])
                target_node = sim.intersections.get(road.to_node)
                downstream_sig = "NONE"
                if target_node and target_node.has_signals:
                    is_ns = abs(road.points[-1].x - road.points[0].x) < abs(road.points[-1].y - road.points[0].y) if len(road.points) >= 2 else True
                    phase_info = sim.phases[sim.signal_phase]
                    downstream_sig = phase_info["ns"] if is_ns else phase_info["ew"]

                sev = config.incident_severity if (incident_applied and r_id == target_incident_road and current_time < (config.incident_start_sec + config.incident_duration_sec)) else 0.0

                state = estimator.estimate_segment_state(
                    road_id=r_id,
                    timestamp=current_time,
                    road_length=road.length,
                    lanes=road.lanes,
                    speed_limit_mps=road.speed_limit,
                    vehicles_on_road=vehs,
                    downstream_signal_state=downstream_sig,
                    incident_severity=sev,
                )
                states.append(state)

            recorder.record_observations_batch(config.scenario_id, states)
            recorded_observations += len(states)

    t_elapsed = time.perf_counter() - t_start
    print(f"    Completed {config.duration_seconds}s ({total_steps} steps) in {t_elapsed:.2f}s wall-time. Logged {recorded_observations} observations.\n", flush=True)
    return recorded_observations


def main():
    print("=" * 75)
    print("  URBANFLOW AI — PREDICTIVE INTELLIGENCE BENCHMARK & VALIDATION SUITE")
    print("=" * 75)

    scenario_ids = [
        "scen_01_normal",
        "scen_02_rush_hour",
        "scen_03_traffic_surge",
        "scen_04_road_incident",
        "scen_05_bottleneck_blockage",
    ]

    db_path = PROJECT_ROOT / "data" / "datasets" / "traffic_history.db"
    existing_records = 0
    if db_path.exists():
        try:
            with sqlite3.connect(db_path) as conn:
                c = conn.cursor()
                c.execute("SELECT COUNT(*) FROM observations")
                row = c.fetchone()
                existing_records = row[0] if row else 0
        except Exception:
            existing_records = 0

    if existing_records < 200000:
        if db_path.exists():
            try:
                db_path.unlink()
            except Exception:
                pass
        recorder = DatasetRecorder(db_path)
        estimator = TrafficStateEstimator()
        capacity_est = RoadCapacityEstimator()
        total_records = 0
        for scen in scenarios:
            total_records += run_scenario(scen, recorder, estimator, capacity_est)
    else:
        print(f"--> Found verified dataset with {existing_records:,} observations in {db_path.name}. Proceeding to training.", flush=True)
        recorder = DatasetRecorder(db_path)
        total_records = existing_records

    print("=" * 75)
    print(f"Total Observations Collected Across 5 Scenarios: {total_records:,}")
    print("=" * 75)

    # Load all scenarios into consolidated chronological training dataset
    dfs = []
    for s_id in scenario_ids:
        df_s = recorder.load_scenario_dataframe(s_id)
        dfs.append(df_s)

    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Combined Training DataFrame: {combined_df.shape[0]} rows, {combined_df.shape[1]} columns")

    # Train and evaluate modular predictor
    print("\n--- Training LightGBM Models & Evaluating Baselines (5m, 10m, 15m) ---")
    t_train_start = time.perf_counter()
    predictor = ModularCongestionPredictor()
    evaluation_report = predictor.train_and_evaluate(combined_df, train_ratio=0.60, val_ratio=0.20)
    t_train_end = time.perf_counter()

    # Persist fitted models for live simulation broadcaster
    models_dir = PROJECT_ROOT / "data" / "models"
    saved = predictor.save_models(models_dir)
    print(f"Persisted {len(saved)} trained model artifacts to {models_dir}")
    print(f"Training and evaluation completed in {t_train_end - t_train_start:.2f} seconds.\n")

    # Print comprehensive performance comparison table
    for horizon in ["5min", "10min", "15min"]:
        print("=" * 75)
        print(f"  FORECAST HORIZON: {horizon.upper()} ({predictor.HORIZONS[horizon]} SECONDS AHEAD)")
        print("=" * 75)

        for target in ["avg_speed", "density", "queue_length"]:
            print(f"\nTarget Variable: [{target}]")
            print(f"{'Method / Model':<22} | {'MAE':<10} | {'RMSE':<10} | {'MAPE (%)':<10}")
            print("-" * 60)

            target_eval = evaluation_report.get(horizon, {}).get(target, {})
            for model_name in ["Persistence", "HistoricalAverage", "MovingAverage", "LightGBM"]:
                metrics = target_eval.get(model_name, {})
                mae_str = f"{metrics.get('MAE', 0.0):.3f}"
                rmse_str = f"{metrics.get('RMSE', 0.0):.3f}"
                mape_str = f"{metrics.get('MAPE_percent', 0.0):.2f}%"
                print(f"{model_name:<22} | {mae_str:<10} | {rmse_str:<10} | {mape_str:<10}")

        # Classification metrics for congestion event detection
        speed_cls = evaluation_report.get(horizon, {}).get("avg_speed", {}).get("Classification_Metrics", {})
        if speed_cls:
            print(f"\nCongestion Event Detection (Speed < 50% Free-Flow):")
            print(f"{'Method / Model':<20} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'ROC-AUC':<10}")
            print("-" * 75)
            for m_name in ["Persistence", "MovingAverage", "LightGBM"]:
                c_met = speed_cls.get(m_name, {})
                acc = f"{c_met.get('Accuracy', 0.0):.4f}"
                prec = f"{c_met.get('Precision', 0.0):.4f}"
                rec = f"{c_met.get('Recall', 0.0):.4f}"
                f1 = f"{c_met.get('F1_Score', 0.0):.4f}"
                auc = f"{c_met.get('ROC_AUC', 0.0):.4f}"
                print(f"{m_name:<20} | {acc:<10} | {prec:<10} | {rec:<10} | {f1:<10} | {auc:<10}")

    print("\nBenchmark and evaluation successfully completed.")


if __name__ == "__main__":
    main()
