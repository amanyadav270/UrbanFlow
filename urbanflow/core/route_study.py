"""Repeat the isolated Classroom Grid route experiment and retain paired runs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import gzip
import json
from pathlib import Path
from statistics import mean, stdev

from urbanflow.core.demand import classroom_demand
from urbanflow.core.kinematic_sim import KinematicSimulator
from urbanflow.core.route_experiment import run_route_experiment


ROUTES = {
    "west_east": ["road_w_in_nw", "road_nw_ne", "road_ne_out_e"],
    "north_south": ["road_n_in_nw", "road_nw_sw", "road_sw_out_s"],
}
DEPARTURES = (0, 35)
SEEDS = range(6101, 6106)


def run_case(output, label, departure, seed):
    path = output / f"{label}-{departure}-{seed}.json.gz"
    if path.is_file():
        try:
            with gzip.open(path, "rt", encoding="utf-8") as log:
                json.load(log)
            return
        except (EOFError, OSError, ValueError):
            pass
    sim = KinematicSimulator("classroom_grid")
    demand = classroom_demand(sim, "rush_hour", seed)
    report = run_route_experiment(
        sim.map_name, "rush_hour", demand, ROUTES[label], "PPO_NEIGHBOR", {},
        departure_seconds=departure, network_horizon=180, detailed=True,
    )
    report.update(seed=seed, demand=[asdict(row) for row in demand])
    temporary = path.with_suffix(".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as log:
        json.dump(report, log, separators=(",", ":"))
    temporary.replace(path)


def study(output, workers=1):
    output.mkdir(parents=True, exist_ok=True)
    cases = [(output, label, departure, seed) for label in ROUTES for departure in DEPARTURES for seed in SEEDS]
    if workers == 1:
        for case in cases:
            run_case(*case)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            jobs = [pool.submit(run_case, *case) for case in cases]
            for job in jobs:
                job.result()
    summaries = {}
    for label in ROUTES:
        summaries[label] = {}
        for departure in DEPARTURES:
            samples = {}
            for seed in SEEDS:
                with gzip.open(output / f"{label}-{departure}-{seed}.json.gz", "rt", encoding="utf-8") as log:
                    report = json.load(log)
                for result in report["results"]:
                    samples.setdefault(result["controller"], []).append(result)
            batch = {}
            for controller, results in samples.items():
                completed = [row["seconds"] for row in results if row["seconds"] is not None]
                batch[controller] = {
                    "completed": len(completed), "runs": len(results),
                    "completion_percent": 100 * len(completed) / len(results),
                    "mean_trip": round(mean(completed), 2) if completed else None,
                    "sample_sd": round(stdev(completed), 2) if len(completed) > 1 else None,
                    "mean_intersection_wait": round(mean(sum(v["stopped"] for v in row["intersections"])
                                                         for row in results), 2),
                    "mean_wait_by_junction": {
                        node: round(mean(sum(v["stopped"] for v in row["intersections"] if v["intersection"] == node)
                                         for row in results), 2)
                        for node in sorted({v["intersection"] for row in results for v in row["intersections"]})
                    },
                    "mean_network_completions_180s": mean(row["network"]["completed"] for row in results),
                    "mean_network_stopped_vehicle_seconds_180s": round(mean(row["network"]["stopped_vehicle_seconds"]
                                                                             for row in results), 2),
                }
            summaries[label][departure] = batch
            print(label, departure, json.dumps(batch), flush=True)
    (output / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument('--workers', type=int, choices=range(1, 9), default=1)
    args = parser.parse_args()
    study(args.output, args.workers)
