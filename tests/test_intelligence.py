"""Unit test suite for Phase 1 - 4 Predictive Traffic Intelligence Foundation."""
import pytest
from pathlib import Path
import numpy as np
import pandas as pd

from urbanflow.intelligence.state_estimator import TrafficStateEstimator, SegmentTrafficState
from urbanflow.intelligence.capacity_estimator import RoadCapacityEstimator, CapacityReport
from urbanflow.intelligence.dataset_recorder import DatasetRecorder, ScenarioConfig
from urbanflow.intelligence.congestion_predictor import (
    PersistenceBaseline,
    HistoricalAverageBaseline,
    MovingAverageBaseline,
    LightGBMPredictor,
    CongestionEvaluationSuite,
    ModularCongestionPredictor,
)
from urbanflow.server.websocket_feed import SimulationBroadcaster


class DummyVehicle:
    def __init__(self, speed: float = 10.0, length: float = 4.5):
        self.speed = speed
        self.length = length


def test_traffic_state_estimator_calculation():
    estimator = TrafficStateEstimator()
    vehs = [DummyVehicle(speed=0.0), DummyVehicle(speed=0.5), DummyVehicle(speed=12.0)]

    state = estimator.estimate_segment_state(
        road_id="road_test_1",
        timestamp=100.0,
        road_length=100.0,
        lanes=2,
        speed_limit_mps=11.2,
        vehicles_on_road=vehs,
        downstream_signal_state="r",
        incident_severity=0.0,
    )

    assert isinstance(state, SegmentTrafficState)
    assert state.road_id == "road_test_1"
    assert state.vehicle_count == 3
    assert state.queue_vehicle_count == 2
    assert state.queue_length_meters > 0.0
    assert state.density_per_km_lane > 0.0
    assert state.data_source == "simulation_synthetic"
    assert state.signal_state == "r"


def test_capacity_estimator_theoretical_vs_operational():
    cap_est = RoadCapacityEstimator()

    # Normal free-flow condition
    normal = cap_est.estimate_capacity(
        road_id="road_norm",
        road_length_meters=150.0,
        lanes=2,
        speed_limit_kmh=50.0,
        observed_flow_veh_hr=800.0,
        density_veh_km_lane=15.0,
        queue_length_meters=0.0,
        downstream_signal_state="G",
        incident_severity=0.0,
    )

    # Degraded incident condition (same road geometry and signal presence)
    incident = cap_est.estimate_capacity(
        road_id="road_inc",
        road_length_meters=150.0,
        lanes=2,
        speed_limit_kmh=50.0,
        observed_flow_veh_hr=800.0,
        density_veh_km_lane=45.0,
        queue_length_meters=80.0,
        downstream_signal_state="r",
        incident_severity=0.75,
    )

    assert normal.theoretical_capacity_veh_hr > 1500.0
    assert normal.operational_capacity_veh_hr > 1000.0
    assert normal.level_of_service in ("A", "B", "C")
    assert normal.data_source == "simulation_synthetic"

    # Incident should severely degrade operational capacity while preserving theoretical baseline
    assert incident.theoretical_capacity_veh_hr == normal.theoretical_capacity_veh_hr
    assert incident.operational_capacity_veh_hr < normal.operational_capacity_veh_hr * 0.4
    assert incident.vc_ratio > normal.vc_ratio
    assert incident.limiting_factor == "INCIDENT"


def test_los_classification_all_bands_and_boundaries():
    """Verifies that all continuous V/C bands A through F are classified correctly without gaps."""
    cap_est = RoadCapacityEstimator()

    # Boundary test vector: (vc_value, expected_los, expected_risk)
    test_cases = [
        (0.00, "A", "LOW"),
        (0.20, "A", "LOW"),
        (0.349, "A", "LOW"),
        (0.35, "B", "LOW"),        # Restored LOS B category
        (0.45, "B", "LOW"),        # Restored LOS B category
        (0.549, "B", "LOW"),       # Restored LOS B category
        (0.55, "C", "MODERATE"),
        (0.65, "C", "MODERATE"),
        (0.749, "C", "MODERATE"),
        (0.75, "D", "HIGH"),
        (0.85, "D", "HIGH"),
        (0.899, "D", "HIGH"),
        (0.90, "E", "CRITICAL"),
        (0.999, "E", "CRITICAL"),
        (1.00, "E", "CRITICAL"),
        (1.001, "F", "CRITICAL"),
        (1.50, "F", "CRITICAL"),
        (3.00, "F", "CRITICAL"),
    ]

    for vc, exp_los, exp_risk in test_cases:
        los, risk, color = cap_est.classify_los(vc)
        assert los == exp_los, f"V/C {vc} expected LOS {exp_los}, got {los}"
        assert risk == exp_risk, f"V/C {vc} expected Risk {exp_risk}, got {risk}"
        assert color.startswith("#"), f"Color for {los} should be hex code"


def test_capacity_signal_timing_and_degradation():
    """Verifies green ratio handling, transient phase throttling, and lack of double-counting."""
    cap_est = RoadCapacityEstimator()

    # Unsignalized vs default signalized baseline
    unsig = cap_est.estimate_capacity("r1", 100.0, 1, 50.0, 500.0, 10.0, 0.0, "NONE")
    sig_g = cap_est.estimate_capacity("r2", 100.0, 1, 50.0, 500.0, 10.0, 0.0, "G")
    sig_y = cap_est.estimate_capacity("r3", 100.0, 1, 50.0, 500.0, 10.0, 0.0, "y")
    sig_r = cap_est.estimate_capacity("r4", 100.0, 1, 50.0, 500.0, 10.0, 0.0, "r")

    # Unsignalized green ratio = 1.0; Signalized default = 0.48
    assert unsig.green_split == 1.0
    assert unsig.green_split_source == "unsignalized"
    assert sig_g.green_split == 0.48
    assert sig_g.green_split_source == "assumed_default_0.48"
    assert unsig.theoretical_capacity_veh_hr > sig_g.theoretical_capacity_veh_hr

    # Actual timing override
    custom_timing = cap_est.estimate_capacity("r5", 100.0, 1, 50.0, 500.0, 10.0, 0.0, "G", actual_green_ratio=0.65)
    assert custom_timing.green_split == 0.65
    assert custom_timing.green_split_source == "actual_timing"
    assert custom_timing.theoretical_capacity_veh_hr > sig_g.theoretical_capacity_veh_hr

    # Transient phase behavior: G has 0 penalty, y has 0.10, r has 0.25 (no double-counting compound collapse)
    assert sig_g.operational_capacity_veh_hr == sig_g.theoretical_capacity_veh_hr
    assert sig_y.operational_capacity_veh_hr == round(sig_g.theoretical_capacity_veh_hr * 0.90, 1)
    assert sig_r.operational_capacity_veh_hr == round(sig_g.theoretical_capacity_veh_hr * 0.75, 1)

    # Incident and queue spillback degradation
    spill = cap_est.estimate_capacity("r6", 100.0, 1, 50.0, 500.0, 10.0, 100.0, "G")
    assert spill.operational_capacity_veh_hr < sig_g.operational_capacity_veh_hr
    assert spill.limiting_factor == "QUEUE_SPILLBACK"

    # Combined degradation capping at 90% loss
    severe = cap_est.estimate_capacity("r7", 100.0, 1, 50.0, 500.0, 10.0, 100.0, "r", incident_severity=0.90)
    assert severe.operational_capacity_veh_hr >= 150.0  # Physical discharge floor


def test_dataset_recorder_and_chronological_feature_leakage_prevention(tmp_path):
    db_file = tmp_path / "test_traffic.db"
    recorder = DatasetRecorder(db_file)

    config = ScenarioConfig(
        scenario_id="test_scen_1",
        scenario_name="Test Scenario",
        random_seed=42,
        duration_seconds=300.0,
        base_spawn_interval=1.0,
    )
    recorder.record_scenario(config)

    # Generate dummy sequential observations
    states = []
    for t in range(0, 300, 5):
        s = SegmentTrafficState(
            road_id="road_A",
            timestamp=float(t),
            vehicle_count=int(t % 15),
            density_per_km_lane=float(t % 25),
            average_speed_kmh=max(5.0, 45.0 - (t % 20)),
            flow_veh_per_hour=float(500 + t * 2),
            queue_length_meters=float((t % 10) * 2),
            queue_vehicle_count=int(t % 5),
            occupancy_ratio=0.25,
            travel_time_sec=12.0,
            theoretical_capacity=3600.0,
            operational_capacity=2800.0,
            capacity_utilization=0.35,
            signal_state="G",
            incident_active=False,
            data_source="simulation_synthetic",
        )
        states.append(s)

    recorder.record_observations_batch("test_scen_1", states)
    df = recorder.load_scenario_dataframe("test_scen_1")
    assert len(df) == len(states)

    # Build supervised features for 30s horizon
    X, y = DatasetRecorder.build_supervised_features(df, horizon_sec=30, target_col="avg_speed")
    assert not X.empty
    assert len(X) == len(y)
    assert "target_future" not in X.columns
    assert "current_val" in X.columns
    assert "avg_speed_lag_1" in X.columns
    assert "avg_speed_roll_mean" in X.columns


def test_modular_prediction_baselines_and_lightgbm():
    # Synthetic dataset
    np.random.seed(42)
    N = 120
    t_vals = np.linspace(20.0, 45.0, N) + np.random.normal(0, 2.0, N)
    df = pd.DataFrame({
        "current_val": t_vals,
        "density": np.random.uniform(5, 30, N),
        "flow": np.random.uniform(200, 1200, N),
        "queue_length": np.random.uniform(0, 50, N),
        "occupancy": np.random.uniform(0.05, 0.40, N),
        "capacity_utilization": np.random.uniform(0.1, 0.8, N),
        "signal_is_red": np.random.choice([0, 1], N),
        "incident_active": np.zeros(N),
        "avg_speed_roll_mean": t_vals,
        "avg_speed_roll_std": np.ones(N) * 1.5,
        "avg_speed_lag_1": np.roll(t_vals, 1),
        "avg_speed_lag_2": np.roll(t_vals, 2),
        "avg_speed_lag_3": np.roll(t_vals, 3),
        "density_lag_1": np.random.uniform(5, 30, N),
        "density_lag_2": np.random.uniform(5, 30, N),
        "density_lag_3": np.random.uniform(5, 30, N),
        "queue_lag_1": np.random.uniform(0, 50, N),
        "queue_lag_2": np.random.uniform(0, 50, N),
        "queue_lag_3": np.random.uniform(0, 50, N),
    })
    y = pd.Series(t_vals * 0.9 + np.random.normal(0, 1.0, N))

    # Evaluate Persistence
    persist = PersistenceBaseline()
    preds_p = persist.predict(df)
    eval_p = CongestionEvaluationSuite.evaluate_regression(y.values, preds_p)
    assert "MAE" in eval_p
    assert "RMSE" in eval_p
    assert eval_p["MAE"] > 0.0

    # Evaluate Historical Average
    hist_avg = HistoricalAverageBaseline()
    hist_avg.fit(df.iloc[:80], y.iloc[:80])
    preds_h = hist_avg.predict(df.iloc[80:])
    eval_h = CongestionEvaluationSuite.evaluate_regression(y.iloc[80:].values, preds_h)
    assert "MAE" in eval_h

    # Evaluate LightGBM
    lgb_model = LightGBMPredictor(n_estimators=30)
    lgb_model.fit(df.iloc[:80], y.iloc[:80])
    preds_lgb = lgb_model.predict(df.iloc[80:])
    eval_lgb = CongestionEvaluationSuite.evaluate_regression(y.iloc[80:].values, preds_lgb)
    assert eval_lgb["MAE"] >= 0.0

    # Evaluate Classification Suite
    cls_eval = CongestionEvaluationSuite.evaluate_congestion_classification(y.iloc[80:].values, preds_lgb)
    assert "Accuracy" in cls_eval
    assert "F1_Score" in cls_eval
    assert "ROC_AUC" in cls_eval


def test_modular_predictor_model_routing_and_online_inference():
    """Verifies that predictor routes each horizon and target to its empirically best model."""
    predictor = ModularCongestionPredictor()

    # Check routing matrix
    assert predictor.BEST_MODELS["5min"]["avg_speed"] == "LightGBM"
    assert predictor.BEST_MODELS["5min"]["density"] == "Persistence"
    assert predictor.BEST_MODELS["5min"]["queue_length"] == "Persistence"

    assert predictor.BEST_MODELS["10min"]["avg_speed"] == "LightGBM"
    assert predictor.BEST_MODELS["10min"]["density"] == "Persistence"
    assert predictor.BEST_MODELS["10min"]["queue_length"] == "LightGBM"

    assert predictor.BEST_MODELS["15min"]["avg_speed"] == "MovingAverage"
    assert predictor.BEST_MODELS["15min"]["density"] == "MovingAverage"
    assert predictor.BEST_MODELS["15min"]["queue_length"] == "MovingAverage"

    # Online feature test payload
    features = {
        "current_val": 35.0,
        "avg_speed": 35.0,
        "density": 18.0,
        "flow": 650.0,
        "queue_length": 5.0,
        "occupancy": 0.12,
        "capacity_utilization": 0.45,
        "signal_is_red": 0,
        "incident_active": 0,
        "avg_speed_roll_mean": 34.0,
        "avg_speed_roll_std": 1.2,
        "avg_speed_lag_1": 34.5,
        "avg_speed_lag_2": 33.8,
        "avg_speed_lag_3": 35.2,
        "density_roll_mean": 17.5,
        "density_lag_1": 17.0,
        "density_lag_2": 16.5,
        "density_lag_3": 18.0,
        "queue_roll_mean": 4.5,
        "queue_lag_1": 4.0,
        "queue_lag_2": 3.0,
        "queue_lag_3": 5.0,
    }

    # Test 15-minute horizon (All targets use MovingAverage - no ML weights required)
    pred_15 = predictor.predict_online(features, horizon="15min", timestamp=500.0)
    assert pred_15["horizon"] == "15min"
    assert pred_15["horizon_seconds"] == 900
    assert pred_15["prediction_timestamp"] == 500.0
    assert pred_15["models_used"]["avg_speed"] == "MovingAverage"
    assert pred_15["models_used"]["density"] == "MovingAverage"
    assert pred_15["models_used"]["queue_length"] == "MovingAverage"
    assert pred_15["predicted_avg_speed"] == 34.0
    assert pred_15["predicted_density"] == 17.5
    assert pred_15["predicted_queue_length"] == 4.5
    assert pred_15["inference_status"] == "complete"
    assert pred_15["congestion_probability"] is not None


def test_modular_predictor_missing_model_safety():
    """Verifies that un-fitted ML models report 'unavailable' rather than fabricating data."""
    fresh_predictor = ModularCongestionPredictor()

    features = {
        "current_val": 30.0,
        "avg_speed": 30.0,
        "density": 15.0,
        "flow": 500.0,
        "queue_length": 0.0,
        "occupancy": 0.10,
        "capacity_utilization": 0.40,
        "signal_is_red": 0,
        "incident_active": 0,
    }

    # 5-minute horizon specifies LightGBM for speed, which is un-fitted in fresh_predictor
    pred_5 = fresh_predictor.predict_online(features, horizon="5min", timestamp=100.0)
    assert pred_5["models_used"]["avg_speed"] == "LightGBM"
    assert pred_5["predicted_avg_speed"] is None  # Never fabricated!
    assert pred_5["statuses"]["avg_speed"] == "unavailable"
    assert pred_5["congestion_probability"] is None
    assert pred_5["risk_level"] == "UNKNOWN"

    # Density uses Persistence, so it IS available
    assert pred_5["models_used"]["density"] == "Persistence"
    assert pred_5["predicted_density"] == 15.0
    assert pred_5["statuses"]["density"] == "available"
    assert pred_5["inference_status"] == "partial"


def test_modular_predictor_save_and_load(tmp_path):
    """Verifies that model artifacts can be serialized and reloaded cleanly."""
    predictor = ModularCongestionPredictor()

    # Fit small model for 5min avg_speed
    df = pd.DataFrame({
        "current_val": [20.0, 25.0, 30.0, 35.0] * 10,
        "density": [5.0, 10.0, 15.0, 20.0] * 10,
        "flow": [200.0, 400.0, 600.0, 800.0] * 10,
        "queue_length": [0.0, 2.0, 5.0, 10.0] * 10,
        "occupancy": [0.1, 0.2, 0.3, 0.4] * 10,
        "capacity_utilization": [0.2, 0.4, 0.6, 0.8] * 10,
        "signal_is_red": [0, 0, 1, 1] * 10,
        "incident_active": [0, 0, 0, 0] * 10,
        "avg_speed_roll_mean": [20.0, 25.0, 30.0, 35.0] * 10,
        "avg_speed_roll_std": [1.0, 1.0, 1.0, 1.0] * 10,
        "avg_speed_lag_1": [20.0, 20.0, 25.0, 30.0] * 10,
        "avg_speed_lag_2": [20.0, 20.0, 20.0, 25.0] * 10,
        "avg_speed_lag_3": [20.0, 20.0, 20.0, 20.0] * 10,
        "density_lag_1": [5.0, 5.0, 10.0, 15.0] * 10,
        "density_lag_2": [5.0, 5.0, 5.0, 10.0] * 10,
        "density_lag_3": [5.0, 5.0, 5.0, 5.0] * 10,
        "queue_lag_1": [0.0, 0.0, 2.0, 5.0] * 10,
        "queue_lag_2": [0.0, 0.0, 0.0, 2.0] * 10,
        "queue_lag_3": [0.0, 0.0, 0.0, 0.0] * 10,
    })
    y = pd.Series([22.0, 27.0, 31.0, 34.0] * 10)

    predictor.models["5min"]["avg_speed"].fit(df, y)
    assert predictor.models["5min"]["avg_speed"].is_fitted

    # Save to tmp directory
    save_dir = tmp_path / "models"
    saved = predictor.save_models(save_dir)
    assert "5min_avg_speed" in saved

    # Reload into new predictor
    reloaded_predictor = ModularCongestionPredictor()
    assert not reloaded_predictor.models["5min"]["avg_speed"].is_fitted
    success = reloaded_predictor.load_models(save_dir)
    assert success
    assert reloaded_predictor.models["5min"]["avg_speed"].is_fitted


def test_broadcaster_intelligence_payload_integration():
    """Verifies that SimulationBroadcaster generates telemetry with real predictions and handles horizon changes."""
    from urbanflow.core.kinematic_sim import KinematicSimulator

    sim = KinematicSimulator()
    broadcaster = SimulationBroadcaster(sim)

    # Step simulation once to generate vehicle states
    snapshot = sim.step(dt=0.05)
    intel = broadcaster._compute_intelligence(snapshot)

    assert "horizon" in intel
    assert intel["horizon"] == "10min"
    assert "bottlenecks" in intel
    assert "road_heatmap" in intel
    assert "capacity_summary" in intel
    assert intel["capacity_summary"]["methodology"] == "HCM-inspired operational V/C heuristic"

    # Check bottleneck telemetry fields
    for b in intel["bottlenecks"]:
        assert "road_id" in b
        assert "current_speed" in b
        assert "predicted_speed" in b
        assert "model_used" in b
        assert "inference_status" in b
        assert "los" in b
        assert b["los"] in ("A", "B", "C", "D", "E", "F")
        assert "vc_ratio" in b

    # Test horizon switching via handle_client_message
    import asyncio
    asyncio.run(broadcaster.handle_client_message({"action": "set_horizon", "horizon": "15min"}))
    assert broadcaster.active_horizon == "15min"

    intel_15 = broadcaster._compute_intelligence(snapshot)
    assert intel_15["horizon"] == "15min"
    assert intel_15["horizon_seconds"] == 900
    for b in intel_15["bottlenecks"]:
        assert b["horizon"] == "15min"
        # 15min routes to MovingAverage
        assert b["model_used"] == "MovingAverage"
        assert b["predicted_speed"] is not None
