import pytest

from urbanflow.core.kinematic_sim import KinematicSimulator
from urbanflow.core.route_experiment import run_route_experiment
from urbanflow.core.demand import classroom_demand


def test_route_report_records_active_checkpoint_and_probe_waits():
    pytest.importorskip("torch")
    sim = KinematicSimulator("classroom_grid")
    demand = classroom_demand(sim, "rush_hour", seed=6101)
    roads = ["road_w_in_nw", "road_nw_ne", "road_ne_out_e"]
    report = run_route_experiment(
        sim.map_name, "rush_hour", demand, roads, "PPO_NEIGHBOR", {},
        controllers=("PPO_NEIGHBOR",), detailed=True,
    )
    result = report["results"][0]
    assert result["checkpoint"] == "data/classroom_ppo_directional_v2/neighbor/weights.pt"
    assert result["controller"] == "Multi-Agent RL Model · Directional v2"
    assert result["action_selection"] == "argmax"
    assert result["actual_entry"] - result["departure"] == result["entry_wait"]
    assert result["arrival"] - result["departure"] == result["seconds"]
    assert [row["intersection"] for row in result["intersections"]] == ["INT_NW", "INT_NE"]
    assert 0 < sum(row["stopped"] for row in result["intersections"]) <= result["stopped"] + .01
    assert any(row["action"] == 1 and row["outcome"] == "clearance_started" for row in result["decisions"])
    assert report["trace"] and result["ticks"]
