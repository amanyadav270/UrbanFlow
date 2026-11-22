"""Compare one preselected PPO checkpoint with three SUMO baselines on DOT count replays."""
import argparse
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from urbanflow.core.observed_demand import OBSERVED_WINDOWS, count_events, measured_slots


SEEDS = range(4101, 4106)
CHECKPOINT = "models/ppo_real_mix_s0/best.zip"
PRESETS = ("normal", "rush_hour", "surge", "city_shift", "event_release")


def write_departures(path, events, seed, demand_module):
    approach = {"NB": "bottom", "SB": "top", "EB": "left", "WB": "right"}
    rng = random.Random(seed + 1_000_000)
    routes = ET.Element("routes")
    routes.append(ET.fromstring(demand_module.VTYPE))
    for index, (time, direction) in enumerate(events):
        source = approach[direction]
        exits = [a for a in demand_module.APPROACHES if a != source]
        weights = [60 if a == demand_module.OPPOSITE[source] else 20 for a in exits]
        target = rng.choices(exits, weights)[0]
        vehicle = ET.SubElement(routes, "vehicle", id=f"atr-{index}", type="car",
                                depart=f"{time:.3f}", departSpeed="max")
        ET.SubElement(vehicle, "route", edges=f"{demand_module.IN_EDGE[source]} {demand_module.OUT_EDGE[target]}")
    ET.ElementTree(routes).write(path, encoding="utf-8", xml_declaration=True)


def preset_events(sim, preset, seed):
    from urbanflow.core.demand import classroom_demand

    for record in classroom_demand(sim, preset, seed):
        a, b = sim.roads[record.route[0]].points
        dx, dy = b.x - a.x, b.y - a.y
        direction = ("EB" if dx > 0 else "WB") if abs(dx) > abs(dy) else ("NB" if dy > 0 else "SB")
        yield record.time, direction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("research", type=Path, help="Location of the original marl-traffic project")
    parser.add_argument("output", type=Path, help="Directory for new comparison CSVs")
    parser.add_argument("--presets", action="store_true", help="Benchmark UrbanFlow's five synthetic presets")
    args = parser.parse_args()
    sys.path.insert(1, str(args.research.resolve()))
    from stable_baselines3 import PPO
    from marl_traffic import demand
    from marl_traffic.baselines.sumo_programs import make_fixed_net, make_fixed_net_split, run_program, webster_split_plan
    from marl_traffic.environment.single_env import SingleIntersectionEnv
    from marl_traffic.evaluation.evaluate import run_env_policy, save_csv

    network = args.research.resolve() / "networks" / "single_intersection"
    cfg, net = network / "scenario.sumocfg", network / "network.net.xml"
    model = PPO.load(str(args.research.resolve() / CHECKPOINT), device="cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.presets:
        from urbanflow.core.kinematic_sim import KinematicSimulator
        sim = KinematicSimulator("classroom_grid")

    def policy(observation, _rng):
        return int(model.predict(observation, deterministic=True)[0])

    with TemporaryDirectory(prefix="urbanflow-dot-") as directory:
        workspace = Path(directory)
        equal = make_fixed_net(net, workspace / "equal.net.xml", 10)
        for window in PRESETS if args.presets else OBSERVED_WINDOWS:
            events_by_seed = {seed: list(preset_events(sim, window, seed) if args.presets else count_events(window, seed))
                              for seed in SEEDS}
            duration = 600 if args.presets else 3600
            if args.presets:
                nominal = list(preset_events(sim, window, 42))
                hours = max(time for time, _ in nominal) / 3600
                ns = sum(direction in ("NB", "SB") for _, direction in nominal) / (2 * hours)
                ew = len(nominal) / (2 * hours) - ns
                # The Webster formula needs unsaturated inputs. Use the same bounded total for all presets.
                factor = min(1, 1500 / (ns + ew))
                ns, ew = ns * factor, ew * factor
            else:
                slots = measured_slots(window)
                ns = sum(n for (_, direction), n in slots.items() if direction in ("NB", "SB")) / 2
                ew = sum(slots.values()) / 2 - ns
            plan = webster_split_plan(ns, ew)
            greens = plan["green_ns"], plan["green_ew"]
            baselines = {
                "fixed-equal-10": equal,
                f"fixed-webster({greens[0]}/{greens[1]})": make_fixed_net_split(net, workspace / "webster.net.xml", *greens),
                "sumo-actuated": network / "network_actuated.net.xml",
            }
            results = {name: [] for name in (*baselines, "PPO(real_mix_s0)")}
            for seed in SEEDS:
                route = workspace / "departures.rou.xml"
                write_departures(route, events_by_seed[seed], seed, demand)
                for i, (name, baseline) in enumerate(baselines.items()):
                    results[name].append(run_program(cfg, baseline, seed, workspace / f"baseline-{i}.xml",
                                                     route_file=route, episode_seconds=duration))
                env = SingleIntersectionEnv(cfg, route_file=route, reward_mode="delta_n", reward_scale=5.0,
                                            episode_seconds=duration)
                results["PPO(real_mix_s0)"].append(run_env_policy(env, policy, seed, workspace / "ppo.xml"))
            target = args.output / (f"preset_{window}.csv" if args.presets else f"{window}.csv")
            save_csv(results, target)
            print(f"Saved {target}", flush=True)


if __name__ == "__main__":
    main()
