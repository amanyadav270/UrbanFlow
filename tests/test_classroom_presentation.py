import asyncio
import math

import pytest

from urbanflow.core.demand import classroom_demand
from urbanflow.core.kinematic_sim import KinematicSimulator
from urbanflow.rl.live import PPOController
from urbanflow.server.websocket_feed import SimulationBroadcaster


def test_trained_classroom_model_decides_for_four_agents_and_fails_closed(monkeypatch):
    pytest.importorskip("torch")

    async def run():
        sim = KinematicSimulator("classroom_grid")
        demand = classroom_demand(sim, "rush_hour")
        sim.set_demand(demand, "rush_hour")
        driver = SimulationBroadcaster(sim)
        driver.prepare_classroom_model()
        assert driver.is_paused and sim.departures == tuple(demand)
        assert driver.active_controller_name == "PPO_NEIGHBOR"
        assert isinstance(driver.controller, PPOController)
        assert driver.controller_telemetry["controller_name"] == "Multi-Agent RL Model"
        assert driver.active_routing_mode == "OFF"
        monkeypatch.setattr(driver, "_compute_intelligence", lambda *a, **kw: {})
        await driver.handle_client_message({"action": "set_speed", "multiplier": 20})
        task = asyncio.create_task(driver.run_loop())
        try:
            for cycle in range(3):
                await driver.handle_client_message({"action": "step_decision"})
                while driver.step_pending:
                    await asyncio.sleep(.01)
                assert driver.is_paused
                rows = list(driver.signal_decisions)[-4:]
                assert {row["junction"] for row in rows} == set(sim.signal_controllers)
                for row in rows:
                    assert row["time"] == pytest.approx(.05 + cycle * 5)
                    assert row["requested"] in (0, 1)
                    assert row["observation"]["action"] in (0, 1)
                    assert all(math.isfinite(p) for p in row["observation"]["probabilities"])
                    assert sum(row["observation"]["probabilities"]) == pytest.approx(1)
            assert driver.controller_telemetry["policy_cycles"] == 3
            await driver.handle_client_message({"action": "reset"})
            assert not driver.signal_decisions and not driver.controller.observations
            assert driver.controller_telemetry["policy_cycles"] == 0

            def fail(**_):
                raise ValueError("inference test error")

            monkeypatch.setattr(driver.controller, "compute_phases", fail)
            await driver.handle_client_message({"action": "step_decision"})
            while driver.step_pending:
                await asyncio.sleep(.01)
            assert driver.is_paused and not driver.signal_decisions
            assert driver.controller_telemetry["status"] == "ERROR"
            assert driver.active_controller_name == "PPO_NEIGHBOR"
            with pytest.raises(ValueError, match="inference test error"):
                await driver.handle_client_message({"action": "resume"})
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(asyncio.wait_for(run(), timeout=12))
