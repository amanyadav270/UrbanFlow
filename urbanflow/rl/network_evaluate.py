"""Compare isolated network candidates on identical held-out schedules."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path
from statistics import mean

from urbanflow.core.classroom_evaluation import compare_classroom
from urbanflow.core.demand import classroom_demand
from urbanflow.core.kinematic_sim import KinematicSimulator
from .live import CANDIDATES
from .network_campaign import NETWORKS


def replay(task):
    network, scenario, seed, directory = task
    sim = KinematicSimulator(network)
    departures = classroom_demand(sim, scenario, seed)
    report = compare_classroom(departures, scenario, map_name=network, candidate_directory=directory)
    print(network, scenario, seed, flush=True)
    return [
        {"network": network, "scenario": scenario, "seed": seed,
         "controller": name, "scheduled": result["scheduled"], "completed": result["completed"],
         "delay_sec": result["completed_delay_mean_sec"], "mean_queue": result["mean_queue"],
         "demand_sha256": report["demand_sha256"], "duration_seconds": report["duration_seconds"]}
        for name, result in report["results"].items()
    ]


def summarize(rows):
    """Compare aligned seed pairs, requiring comparable completion."""
    groups = {}
    for row in rows:
        key = (row["network"], row["scenario"])
        groups.setdefault(key, {}).setdefault(row["controller"], {})[row["seed"]] = row
    summary = []
    for (network, scenario), controllers in groups.items():
        candidate = controllers["PPO_NEIGHBOR"]
        comparisons = {}
        for name, baseline in controllers.items():
            if name == "PPO_NEIGHBOR":
                continue
            pairs = [(candidate[seed], baseline[seed]) for seed in sorted(candidate)]
            delays = [rl["delay_sec"] - rule["delay_sec"] for rl, rule in pairs
                      if rl["delay_sec"] is not None and rule["delay_sec"] is not None]
            completion = [rl["completed"] - rule["completed"] for rl, rule in pairs]
            comparisons[name] = {
                "mean_delay_difference_sec": mean(delays) if delays else None,
                "mean_completed_difference": mean(completion),
                "joint_wins": sum(rl["delay_sec"] is not None and rule["delay_sec"] is not None
                                  and rl["delay_sec"] < rule["delay_sec"]
                                  and rl["completed"] >= rule["completed"] for rl, rule in pairs),
                "seeds": len(pairs),
            }
        summary.append({"network": network, "scenario": scenario, "against": comparisons})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=CANDIDATES)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--first-seed", type=int, default=5101)
    args = parser.parse_args()
    output = args.root / "held_out.csv"
    if output.exists():
        parser.error("Output already exists.")
    tasks = [(network, scenario, seed, args.root / network / "neighbor")
             for network in NETWORKS for scenario in ("normal", "city_shift")
             for seed in range(args.first_seed, args.first_seed + args.seeds)]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool, output.open("x", newline="", encoding="utf-8") as file:
        writer = None
        rows = []
        for result in pool.map(replay, tasks):
            if writer is None:
                writer = csv.DictWriter(file, fieldnames=list(result[0]))
                writer.writeheader()
            writer.writerows(result)
            rows.extend(result)
            file.flush()
    (args.root / "held_out_summary.json").write_text(
        json.dumps(summarize(rows), indent=2) + "\n", encoding="utf-8"
    )
