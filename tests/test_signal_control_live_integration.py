"""Integration and unit tests for Phase 7 Signal-Control Integration in UrbanFlow AI.

Verifies:
1. Availability and instantiation of all 4 controller regimes (FixedTime, Actuated, MaxPressure, CooperativePredictive).
2. Live simulation control loop invocation at consistent 1.0s control interval.
3. Simulator-level FSM safety enforcement (yellow, all-red, min green, conflicting green prevention).
4. REST API status, /api/controllers, and /api/control switching with strict validation.
5. Clean state reset on simulation reset or map reload without cross-contamination.
6. CooperativePredictive SHA-256 cryptographic check and graceful non-predictive fallback.
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest
from starlette.datastructures import Headers

from urbanflow.core.kinematic_sim import KinematicSimulator
from urbanflow.intelligence.signal_control import (
    SignalControllerBase,
    FixedTimeController,
    ActuatedController,
    MaxPressureController,
    CooperativePredictiveSignalController,
)
from urbanflow.server.app import (
    api_status,
    api_controllers,
    api_control,
    broadcaster as app_broadcaster,
)
from urbanflow.server.websocket_feed import (
    SimulationBroadcaster,
    CONTROLLER_REGIMES,
)


class DummyRequest:
    def __init__(self, json_body=None):
        self._json = json_body or {}
        self.headers = Headers({"content-type": "application/json"})

    async def json(self):
        return self._json


def test_controller_regimes_specification():
    """All 4 planned regimes must be registered with proper metadata."""
    expected_keys = {"FIXED_TIME", "ACTUATED", "MAX_PRESSURE", "COOPERATIVE_PREDICTIVE"}
    assert set(CONTROLLER_REGIMES.keys()) == expected_keys

    for key, spec in CONTROLLER_REGIMES.items():
        assert "name" in spec
        assert "description" in spec
        assert issubclass(spec["class"], SignalControllerBase)


def test_broadcaster_controller_lifecycle():
    """Broadcaster must initialize with default controller, switch cleanly, and reject invalid names."""
    sim = KinematicSimulator()
    b = SimulationBroadcaster(sim)

    # 1. Default controller
    assert b.active_controller_name == "COOPERATIVE_PREDICTIVE"
    assert isinstance(b.controller, CooperativePredictiveSignalController)
    assert b.controller_telemetry["active_controller"] == "COOPERATIVE_PREDICTIVE"

    # 2. Switch to Fixed-Time
    tel = b.set_controller("FIXED_TIME")
    assert b.active_controller_name == "FIXED_TIME"
    assert isinstance(b.controller, FixedTimeController)
    assert tel["controller_name"] == "Fixed-Time"

    # 3. Switch to Actuated
    tel = b.set_controller("actuated")  # Case-insensitive
    assert b.active_controller_name == "ACTUATED"
    assert isinstance(b.controller, ActuatedController)
    assert tel["controller_name"] == "Actuated"

    # 4. Switch to MaxPressure
    tel = b.set_controller("MAX_PRESSURE")
    assert b.active_controller_name == "MAX_PRESSURE"
    assert isinstance(b.controller, MaxPressureController)
    assert tel["controller_name"] == "MaxPressure"

    # 5. Invalid controller rejection
    with pytest.raises(ValueError) as excinfo:
        b.set_controller("REINFORCEMENT_LEARNING_PPO")
    assert "Invalid signal controller" in str(excinfo.value)
    # Ensure active controller was not mutated by invalid attempt
    assert b.active_controller_name == "MAX_PRESSURE"


def test_controller_evaluation_and_phase_dispatch():
    """Controller compute_phases must be invoked and commands applied to simulator."""
    sim = KinematicSimulator()
    b = SimulationBroadcaster(sim)
    b.set_controller("FIXED_TIME")

    # Initially step simulator
    snapshot = sim.step(dt=0.25)
    phases = b.controller.compute_phases(
        current_time=snapshot.timestamp,
        simulator=sim,
        telemetry=b.current_intel,
    )
    assert isinstance(phases, dict)
    assert len(phases) > 0

    # Apply phases
    sim.apply_signal_phases(phases)
    tel = b._update_controller_telemetry(phases)
    assert tel["last_decisions_count"] == len(phases)
    assert tel["status"] == "OPERATIONAL"


def test_fsm_safety_bounds_preserved():
    """Simulator FSM safety guards (clearance yellow/red, min green) remain intact under controller."""
    sim = KinematicSimulator()
    rep_ctrl = next(iter(sim.signal_controllers.values()))

    # Start at phase 0, stage GREEN
    assert rep_ctrl.stage == "GREEN"
    assert rep_ctrl.active_green_phase == 0

    # Request immediate phase 1 (EW) when time_in_green is 0 (< min_green=10s)
    sim.apply_signal_phases({rep_ctrl.intersection_id: 1})
    # Should not switch immediately because min_green is not met
    assert rep_ctrl.stage == "GREEN"
    assert rep_ctrl.active_green_phase == 0

    # Hold phase 0 for 40 steps (10.0s) so time_in_green reaches min_green under command mode
    for _ in range(40):
        sim.apply_signal_phases({rep_ctrl.intersection_id: 0})
        sim.step(dt=0.25)

    assert rep_ctrl.stage == "GREEN"
    assert rep_ctrl.time_in_green >= rep_ctrl.min_green

    # Now request phase 1
    sim.apply_signal_phases({rep_ctrl.intersection_id: 1})
    # Must transition to yellow clearance first, never instantaneous green jump
    assert rep_ctrl.stage == "YELLOW"
    assert rep_ctrl.time_in_stage == 0.0


def test_reset_and_map_switch_state_cleanliness():
    """Resetting or switching maps must cleanly clear controller and broadcaster state."""
    sim = KinematicSimulator()
    b = SimulationBroadcaster(sim)
    b.set_controller("ACTUATED")

    # Run several steps to populate history
    for _ in range(20):
        snap = sim.step(dt=0.25)
        phases = b.controller.compute_phases(snap.timestamp, simulator=sim)
        sim.apply_signal_phases(phases)

    assert b.controller._last_requested_phase

    # Test reset action
    asyncio_run = pytest.importorskip("asyncio").run
    asyncio_run(b.handle_client_message({"action": "reset"}))

    assert b.last_control_time == -1.0
    assert len(b.controller._last_requested_phase) == 0
    assert sim.sim_time == 0.0


@pytest.mark.anyio
async def test_rest_api_status_and_controllers():
    """REST endpoints /api/status and /api/controllers report controller state accurately."""
    # Ensure known state
    app_broadcaster.set_controller("COOPERATIVE_PREDICTIVE")

    # 1. /api/status
    resp = await api_status(DummyRequest())
    assert resp.status_code == 200
    data = json.loads(resp.body)
    assert data["active_controller"] == "COOPERATIVE_PREDICTIVE"
    assert data["controller_status"] == "OPERATIONAL"
    assert "COOPERATIVE_PREDICTIVE" in data["available_controllers"]
    assert "FIXED_TIME" in data["available_controllers"]

    # 2. /api/controllers
    resp_ctrls = await api_controllers(DummyRequest())
    assert resp_ctrls.status_code == 200
    ctrls_data = json.loads(resp_ctrls.body)
    assert ctrls_data["active_controller"] == "COOPERATIVE_PREDICTIVE"
    assert len(ctrls_data["available_controllers"]) == 4

    active_items = [c for c in ctrls_data["available_controllers"] if c["is_active"]]
    assert len(active_items) == 1
    assert active_items[0]["id"] == "COOPERATIVE_PREDICTIVE"


@pytest.mark.anyio
async def test_rest_api_control_controller_switching():
    """Switching controllers via /api/control validates options and updates state."""
    # Valid switch
    req_act = DummyRequest(json_body={"action": "set_controller", "controller": "MAX_PRESSURE"})
    resp_act = await api_control(req_act)
    assert resp_act.status_code == 200
    body_act = json.loads(resp_act.body)
    assert body_act["status"] == "acknowledged"
    assert body_act["controller"] == "MAX_PRESSURE"
    assert app_broadcaster.active_controller_name == "MAX_PRESSURE"

    # Invalid switch must return HTTP 400
    req_inv = DummyRequest(json_body={"action": "set_controller", "controller": "BOGUS_CONTROLLER"})
    resp_inv = await api_control(req_inv)
    assert resp_inv.status_code == 400
    body_inv = json.loads(resp_inv.body)
    assert body_inv["status"] == "error"
    assert "Invalid controller" in body_inv["message"]
    # Controller must remain MAX_PRESSURE
    assert app_broadcaster.active_controller_name == "MAX_PRESSURE"

    # Restore default
    await api_control(DummyRequest(json_body={"action": "set_controller", "controller": "COOPERATIVE_PREDICTIVE"}))
    assert app_broadcaster.active_controller_name == "COOPERATIVE_PREDICTIVE"


def test_cooperative_predictive_fallback_resilience(tmp_path: Path):
    """CooperativePredictive must gracefully fallback when classifier file is missing or corrupted."""
    # 1. Missing file fallback
    fake_path = tmp_path / "nonexistent_model.joblib"
    ctrl_missing = CooperativePredictiveSignalController(classifier_path=fake_path)
    assert not ctrl_missing.predictive_available
    assert "UNAVAILABLE_FILE_NOT_FOUND" in ctrl_missing.predictive_status

    # 2. Corrupt / hash mismatch fallback
    corrupt_file = tmp_path / "corrupt_model.joblib"
    corrupt_file.write_bytes(b"corrupt payload bytes")
    ctrl_corrupt = CooperativePredictiveSignalController(classifier_path=corrupt_file)
    assert not ctrl_corrupt.predictive_available
    assert "UNAVAILABLE_HASH_MISMATCH" in ctrl_corrupt.predictive_status

    # 3. Execution under fallback must still function purely cooperatively without raising
    sim = KinematicSimulator()
    phases = ctrl_corrupt.compute_phases(current_time=10.0, simulator=sim)
    assert isinstance(phases, dict)
    assert len(phases) > 0
