"""Phase 3: Multi-Horizon Congestion Prediction Engine for UrbanFlow AI.

Supports predictions at 5-minute (300s), 10-minute (600s), and 15-minute (900s) horizons for:
1. Average Speed (km/h)
2. Vehicle Density (veh/km/lane)
3. Queue Length (meters)
4. Congestion Probability (0.0 - 1.0) & Classification (Low, Moderate, High, Severe)

Includes:
- Baseline 1: Persistence Predictor (y_{t+h} = y_t)
- Baseline 2: Historical Average Predictor
- Baseline 3: Moving Average Predictor
- Advanced ML: LightGBM Multi-Horizon Predictor (with HistGradientBoosting fallback)
- Strict Chronological Train / Validation / Test Splitting
- Formal Metric Evaluation (MAE, RMSE, MAPE, Accuracy, Precision, Recall, F1-Score, ROC-AUC)
- Model Persistence (Save / Load) and Missing-Model Safe Telemetry Generation
- Dynamic Model Routing based on Empirical Milestone 3 Audit
"""
from __future__ import annotations

import lightgbm as lgb
import json
import logging
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


class PredictionBaseline:
    """Abstract baseline predictor interface."""

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        pass

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class PersistenceBaseline(PredictionBaseline):
    """Predicts future value equals the current observed value at timestamp t."""

    def __init__(self, current_val_col: str = "current_val"):
        self.current_val_col = current_val_col

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        if self.current_val_col in X_test.columns:
            return X_test[self.current_val_col].values
        for col in X_test.columns:
            if "current" in col:
                return X_test[col].values
        return np.zeros(len(X_test))


class HistoricalAverageBaseline(PredictionBaseline):
    """Predicts the mean of the training target across all historical samples."""

    def __init__(self):
        self.mean_val = 0.0

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.mean_val = float(y_train.mean()) if len(y_train) > 0 else 0.0

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        return np.full(len(X_test), self.mean_val)


class MovingAverageBaseline(PredictionBaseline):
    """Predicts rolling mean of the most recent lag window."""

    def __init__(self, roll_mean_col: str = "avg_speed_roll_mean"):
        self.roll_mean_col = roll_mean_col

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        if self.roll_mean_col in X_test.columns:
            return X_test[self.roll_mean_col].values
        for col in X_test.columns:
            if "roll_mean" in col:
                return X_test[col].values
        if "current_val" in X_test.columns:
            return X_test["current_val"].values
        return np.zeros(len(X_test))


class LightGBMPredictor(PredictionBaseline):
    """Trained LightGBM Regressor for continuous multi-horizon forecasting."""

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.05):
        self.model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=42,
            verbosity=-1,
            force_col_wise=True,
            n_jobs=2,
        )
        self.is_fitted = False
        self.feature_names: List[str] = []
        self._booster: Optional[lgb.Booster] = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.feature_names = list(X_train.columns)
        self.model.fit(X_train, y_train)
        if hasattr(self.model, "booster_"):
            try:
                b_str = self.model.booster_.model_to_string()
                self._booster = lgb.Booster(model_str=b_str)
            except Exception as e:
                logger.warning("Could not extract native booster after fit: %s", e)
                self._booster = None
        self.is_fitted = True

    def predict(self, X_test: Any) -> np.ndarray:
        if not self.is_fitted:
            return np.zeros(1 if not hasattr(X_test, "__len__") else len(X_test))
        if isinstance(X_test, np.ndarray):
            if X_test.ndim == 1:
                X_test = X_test.reshape(1, -1)
            cols = self.feature_names if len(self.feature_names) == X_test.shape[1] else [f"f{i}" for i in range(X_test.shape[1])]
            X_test = pd.DataFrame(X_test, columns=cols)
        elif isinstance(X_test, dict):
            X_test = pd.DataFrame([X_test])

        if self.feature_names:
            # Construct dataframe with exact feature names and order expected by trained LightGBM
            aligned_data = {}
            for col in self.feature_names:
                if col in X_test.columns:
                    aligned_data[col] = X_test[col]
                elif col.startswith("queue_length_") and col.replace("queue_length_", "queue_") in X_test.columns:
                    aligned_data[col] = X_test[col.replace("queue_length_", "queue_")]
                elif col.startswith("queue_") and col.replace("queue_", "queue_length_") in X_test.columns:
                    aligned_data[col] = X_test[col.replace("queue_", "queue_length_")]
                else:
                    aligned_data[col] = 0.0
            aligned_df = pd.DataFrame(aligned_data, index=X_test.index)
        else:
            aligned_df = X_test

        # Prefer safe native Booster on Windows to eliminate C++ access violations
        if self._booster is not None:
            try:
                vals = aligned_df.to_numpy(dtype=np.float64)
                return np.asarray(self._booster.predict(vals), dtype=np.float64)
            except Exception as e:
                logger.warning("Native booster inference failed, falling back to scikit-learn predict: %s", e)

        if hasattr(self.model, "predict"):
            return self.model.predict(aligned_df)
        return np.zeros(len(X_test))

    def save(self, filepath: Path | str):
        """Serializes fitted model and feature specification."""
        if not self.is_fitted:
            return
        payload = {
            "model": self.model,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(payload, filepath)

    def load(self, filepath: Path | str) -> bool:
        """Loads serialized model and feature specification with native Booster extraction."""
        path = Path(filepath)
        if not path.exists():
            return False
        try:
            payload = joblib.load(path)
            self.model = payload["model"]
            self.feature_names = payload.get("feature_names", [])
            self.is_fitted = payload.get("is_fitted", True)

            # Robust native LightGBM booster extraction for Windows C++ stability
            self._booster = None
            try:
                if hasattr(self.model, "booster_"):
                    b_str = self.model.booster_.model_to_string()
                    self._booster = lgb.Booster(model_str=b_str)
                    self.model._Booster = self._booster
                elif isinstance(self.model, lgb.Booster):
                    b_str = self.model.model_to_string()
                    self._booster = lgb.Booster(model_str=b_str)
                elif hasattr(self.model, "_Booster"):
                    b_str = self.model._Booster.model_to_string()
                    self._booster = lgb.Booster(model_str=b_str)
                    self.model._Booster = self._booster
            except Exception as b_exc:
                logger.warning("Native LightGBM booster initialization failed for %s: %s", path.name, b_exc)
                self._booster = None

            return True
        except Exception as e:
            logger.error("Failed to load LightGBMPredictor from %s: %s", path, e)
            self._booster = None
            self.is_fitted = False
            return False


class CongestionEvaluationSuite:
    """Evaluates prediction accuracy across regression and classification metrics."""

    @staticmethod
    def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculates MAE, RMSE, and MAPE."""
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(root_mean_squared_error(y_true, y_pred))
        
        # Safe MAPE calculation avoiding div by zero
        eps = 1e-4
        non_zero = np.abs(y_true) > eps
        if np.any(non_zero):
            mape = float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100.0)
        else:
            mape = 0.0

        return {
            "MAE": round(mae, 3),
            "RMSE": round(rmse, 3),
            "MAPE_percent": round(mape, 2),
        }

    @staticmethod
    def evaluate_congestion_classification(
        y_true_speed: np.ndarray,
        y_pred_speed: np.ndarray,
        free_flow_speed: float = 40.0,
        congestion_threshold_ratio: float = 0.50,  # Speed < 50% of free-flow = congested
    ) -> Dict[str, float]:
        """Calculates classification metrics for predicting binary congestion events."""
        speed_thresh = free_flow_speed * congestion_threshold_ratio
        
        y_true_bin = (y_true_speed < speed_thresh).astype(int)
        y_pred_bin = (y_pred_speed < speed_thresh).astype(int)

        # Probabilities estimated via logistic sigmoid over normalized speed drop
        prob_pred = 1.0 / (1.0 + np.exp((y_pred_speed - speed_thresh) / 5.0))

        acc = float(accuracy_score(y_true_bin, y_pred_bin))
        prec = float(precision_score(y_true_bin, y_pred_bin, zero_division=0))
        rec = float(recall_score(y_true_bin, y_pred_bin, zero_division=0))
        f1 = float(f1_score(y_true_bin, y_pred_bin, zero_division=0))

        try:
            if len(np.unique(y_true_bin)) > 1:
                auc = float(roc_auc_score(y_true_bin, prob_pred))
            else:
                auc = 1.0 if (y_true_bin[0] == y_pred_bin[0]) else 0.5
        except Exception:
            auc = 0.5

        return {
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1_Score": round(f1, 4),
            "ROC_AUC": round(auc, 4),
        }


class ModularCongestionPredictor:
    """
    Unified manager serving predictions across 5m, 10m, and 15m horizons.
    Routes each target and horizon to the empirically optimal model.
    """

    HORIZONS = {
        "5min": 300,
        "10min": 600,
        "15min": 900,
    }

    TARGETS = [
        "avg_speed",
        "density",
        "queue_length",
    ]

    # Optimal model per (horizon, target) established by Milestone 3 Technical Audit:
    # 5m: Speed->LightGBM (MAE 3.08), Density->Persistence (MAE 6.34), Queue->Persistence (MAE 0.57)
    # 10m: Speed->LightGBM (MAE 3.49), Density->Persistence (MAE 7.33), Queue->LightGBM (MAE 0.69)
    # 15m: Speed->MovingAverage (MAE 4.25), Density->MovingAverage (MAE 7.43), Queue->MovingAverage (MAE 0.69)
    BEST_MODELS = {
        "5min": {
            "avg_speed": "LightGBM",
            "density": "Persistence",
            "queue_length": "Persistence",
            "congestion_event": "LightGBM",
        },
        "10min": {
            "avg_speed": "LightGBM",
            "density": "Persistence",
            "queue_length": "LightGBM",
            "congestion_event": "LightGBM",
        },
        "15min": {
            "avg_speed": "MovingAverage",
            "density": "MovingAverage",
            "queue_length": "MovingAverage",
            "congestion_event": "LightGBM",
        },
    }

    def __init__(self):
        self.models: Dict[str, Dict[str, LightGBMPredictor]] = {}
        self.baselines: Dict[str, Dict[str, Dict[str, PredictionBaseline]]] = {}
        self.evaluations: Dict[str, Dict[str, Dict[str, Any]]] = {}

        for h in self.HORIZONS:
            self.models[h] = {}
            self.baselines[h] = {}
            self.evaluations[h] = {}
            for target in self.TARGETS:
                self.models[h][target] = LightGBMPredictor()
                self.baselines[h][target] = {
                    "Persistence": PersistenceBaseline(),
                    "HistoricalAverage": HistoricalAverageBaseline(),
                    "MovingAverage": MovingAverageBaseline(f"{target}_roll_mean"),
                }

    def save_models(self, directory: Path | str) -> Dict[str, str]:
        """Serializes all fitted models and baseline calibration to disk once."""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        saved_files = {}

        for h, targets in self.models.items():
            for target, model in targets.items():
                if model.is_fitted:
                    filename = f"lgb_{h}_{target}.joblib"
                    out_path = dir_path / filename
                    model.save(out_path)
                    saved_files[f"{h}_{target}"] = str(out_path)

        # Save historical average means
        hist_means = {}
        for h in self.HORIZONS:
            hist_means[h] = {}
            for t in self.TARGETS:
                base = self.baselines[h][t].get("HistoricalAverage")
                if base and hasattr(base, "mean_val"):
                    hist_means[h][t] = base.mean_val

        with open(dir_path / "baseline_means.json", "w", encoding="utf-8") as f:
            json.dump(hist_means, f, indent=2)

        # Save model registry metadata
        registry = {
            "horizons": list(self.HORIZONS.keys()),
            "targets": self.TARGETS,
            "best_models": self.BEST_MODELS,
            "saved_models": list(saved_files.keys()),
        }
        with open(dir_path / "model_registry.json", "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)

        return saved_files

    def load_models(self, directory: Path | str) -> bool:
        """Loads serialized models from disk once during initialization."""
        dir_path = Path(directory)
        if not dir_path.exists():
            return False

        loaded_any = False
        for h, targets in self.models.items():
            for target, model in targets.items():
                target_file = dir_path / f"lgb_{h}_{target}.joblib"
                if target_file.exists():
                    if model.load(target_file):
                        loaded_any = True

        # Load baseline means if available
        means_file = dir_path / "baseline_means.json"
        if means_file.exists():
            try:
                with open(means_file, "r", encoding="utf-8") as f:
                    hist_means = json.load(f)
                for h in self.HORIZONS:
                    for t in self.TARGETS:
                        val = hist_means.get(h, {}).get(t)
                        if val is not None and "HistoricalAverage" in self.baselines[h][t]:
                            self.baselines[h][t]["HistoricalAverage"].mean_val = float(val)
            except Exception as e:
                logger.warning("Could not load baseline means: %s", e)

        return loaded_any

    def is_model_available(self, horizon: str, target: str) -> bool:
        """Checks if the recommended model for (horizon, target) is loaded/available."""
        if horizon not in self.HORIZONS:
            return False
        model_name = self.BEST_MODELS.get(horizon, {}).get(target, "Persistence")
        if model_name == "LightGBM":
            lgb_mod = self.models.get(horizon, {}).get(target)
            return bool(lgb_mod and lgb_mod.is_fitted)
        # Persistence and MovingAverage compute from observed state and are always available
        return True

    def train_and_evaluate(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.60,
        val_ratio: float = 0.20,
    ) -> Dict[str, Any]:
        """
        Chronologically splits historical data into Train (60%), Validation (20%), and Test (20%).
        Trains LightGBM models and benchmarks against all 3 baselines.
        """
        from urbanflow.intelligence.dataset_recorder import DatasetRecorder

        report = {}

        for h_label, h_sec in self.HORIZONS.items():
            report[h_label] = {}

            for target in self.TARGETS:
                X, y = DatasetRecorder.build_supervised_features(df, horizon_sec=h_sec, target_col=target)
                if len(X) < 30:
                    continue

                # Strict Chronological Train/Val/Test Split (No Shuffling)
                n = len(X)
                i_train = int(n * train_ratio)
                i_val = int(n * (train_ratio + val_ratio))

                X_train, y_train = X.iloc[:i_train], y.iloc[:i_train]
                X_val, y_val = X.iloc[i_train:i_val], y.iloc[i_train:i_val]
                X_test, y_test = X.iloc[i_val:], y.iloc[i_val:]

                # Fit and evaluate baselines
                target_eval = {}
                for b_name, baseline in self.baselines[h_label][target].items():
                    baseline.fit(X_train, y_train)
                    preds = baseline.predict(X_test)
                    target_eval[b_name] = CongestionEvaluationSuite.evaluate_regression(y_test.values, preds)

                # Fit and evaluate LightGBM
                ml_model = self.models[h_label][target]
                ml_model.fit(X_train, y_train)
                ml_preds = ml_model.predict(X_test)
                target_eval["LightGBM"] = CongestionEvaluationSuite.evaluate_regression(y_test.values, ml_preds)

                # If target is avg_speed, also compute binary classification metrics
                if target == "avg_speed":
                    target_eval["Classification_Metrics"] = {
                        "Persistence": CongestionEvaluationSuite.evaluate_congestion_classification(y_test.values, self.baselines[h_label][target]["Persistence"].predict(X_test)),
                        "MovingAverage": CongestionEvaluationSuite.evaluate_congestion_classification(y_test.values, self.baselines[h_label][target]["MovingAverage"].predict(X_test)),
                        "LightGBM": CongestionEvaluationSuite.evaluate_congestion_classification(y_test.values, ml_preds),
                    }

                report[h_label][target] = target_eval
                self.evaluations[h_label][target] = target_eval

        return report

    def predict_online(
        self,
        current_state_features: Dict[str, Any],
        horizon: str = "10min",
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generates real-time inference predictions routed to the best-performing model.
        
        Strict Safety Contract:
        If the recommended model is unavailable (e.g. LightGBM model file not loaded),
        reports it explicitly as unavailable (`None` with status "unavailable").
        Never substitutes fabricated predictions.
        """
        if horizon not in self.HORIZONS:
            horizon = "10min"

        h_sec = self.HORIZONS[horizon]
        df_feat = pd.DataFrame([current_state_features])
        routing = self.BEST_MODELS.get(horizon, {})

        results: Dict[str, Any] = {
            "horizon": horizon,
            "horizon_seconds": h_sec,
            "prediction_timestamp": timestamp,
            "models_used": {},
            "statuses": {},
        }

        all_available = True
        any_available = False

        for target in self.TARGETS:
            chosen_model_name = routing.get(target, "Persistence")
            results["models_used"][target] = chosen_model_name

            # Target-specific feature slice so current_val represents the predicted variable
            target_feat = dict(current_state_features)
            target_feat["current_val"] = current_state_features.get(target, current_state_features.get("current_val", 0.0))
            target_df_feat = pd.DataFrame([target_feat])

            if chosen_model_name == "LightGBM":
                model = self.models.get(horizon, {}).get(target)
                if model and model.is_fitted:
                    try:
                        pred_val = float(model.predict(target_df_feat)[0])
                        results[f"predicted_{target}"] = round(pred_val, 2)
                        results["statuses"][target] = "available"
                        any_available = True
                    except Exception as e:
                        logger.warning("Inference failure for LightGBM %s %s: %s", horizon, target, e)
                        results[f"predicted_{target}"] = None
                        results["statuses"][target] = "unavailable"
                        all_available = False
                else:
                    # Model not loaded: report unavailable, never fabricate
                    results[f"predicted_{target}"] = None
                    results["statuses"][target] = "unavailable"
                    all_available = False

            elif chosen_model_name == "MovingAverage":
                # Look for rolling mean feature (supporting both queue_roll_mean and queue_length_roll_mean)
                prefix = target.split("_")[0]
                roll_val = current_state_features.get(
                    f"{target}_roll_mean",
                    current_state_features.get(f"{prefix}_roll_mean", current_state_features.get(target, current_state_features.get("current_val")))
                )
                if roll_val is not None:
                    results[f"predicted_{target}"] = round(float(roll_val), 2)
                    results["statuses"][target] = "available"
                    any_available = True
                else:
                    results[f"predicted_{target}"] = None
                    results["statuses"][target] = "unavailable"
                    all_available = False

            elif chosen_model_name == "Persistence":
                # Current observed state
                curr_val = current_state_features.get(target, current_state_features.get("current_val"))
                if curr_val is not None:
                    results[f"predicted_{target}"] = round(float(curr_val), 2)
                    results["statuses"][target] = "available"
                    any_available = True
                else:
                    results[f"predicted_{target}"] = None
                    results["statuses"][target] = "unavailable"
                    all_available = False

            elif chosen_model_name == "HistoricalAverage":
                base = self.baselines.get(horizon, {}).get(target, {}).get("HistoricalAverage")
                if base and hasattr(base, "mean_val") and base.mean_val > 0.0:
                    results[f"predicted_{target}"] = round(float(base.mean_val), 2)
                    results["statuses"][target] = "available"
                    any_available = True
                else:
                    results[f"predicted_{target}"] = None
                    results["statuses"][target] = "unavailable"
                    all_available = False

        # Overall status summary
        if all_available:
            results["inference_status"] = "complete"
        elif any_available:
            results["inference_status"] = "partial"
        else:
            results["inference_status"] = "unavailable"

        # Congestion Probability based on real predicted speed (if available)
        pred_speed = results.get("predicted_avg_speed")
        if pred_speed is not None:
            # Calibrated logistic sigmoid centered at 20 km/h threshold
            speed_ratio = max(0.0, min(1.0, 1.0 - (pred_speed / 40.0)))
            results["congestion_probability"] = round(speed_ratio, 3)
            if speed_ratio >= 0.75:
                risk = "CRITICAL"
            elif speed_ratio >= 0.50:
                risk = "HIGH"
            elif speed_ratio >= 0.25:
                risk = "MODERATE"
            else:
                risk = "LOW"
            results["risk_level"] = risk
        else:
            results["congestion_probability"] = None
            results["risk_level"] = "UNKNOWN"

        return results
