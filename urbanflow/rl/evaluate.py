"""Run six controllers on matched held-out Classroom schedules."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev

from urbanflow.core.classroom_evaluation import compare_classroom
from urbanflow.core.demand import classroom_demand
from urbanflow.core.kinematic_sim import KinematicSimulator
from .live import POLICIES, ROOT, load

METRICS = ("completed", "scheduled", "pending", "active", "future_departures",
           "completed_delay_mean_sec", "completed_delay_p95_sec", "mean_queue",
           "peak_queue", "spillback_seconds")


def run(task):
    scenario, seed = task
    sim = KinematicSimulator("classroom_grid")
    for key in POLICIES:
        load(sim, key)
    report = compare_classroom(classroom_demand(sim, scenario, seed), scenario, include_ppo=True)
    return {**report, "seed": seed}


def evaluate(out, jobs):
    tasks = [(scenario, seed) for scenario in ("normal", "city_shift", "incident")
             for seed in range(5001, 5006)]
    out.mkdir(parents=True, exist_ok=True)
    hashes = {name: hashlib.sha256((ROOT / variant / "weights.pt").read_bytes()).hexdigest()
              for name, variant in POLICIES.items()}
    records, summaries = [], {}
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for report in pool.map(run, tasks):
            records.append(report)
            print(report["scenario"], report["seed"], flush=True)
    with (out / "per-seed.csv").open("w", encoding="utf-8", newline="") as file:
        fields = ("scenario", "seed", "controller", "checkpoint_sha256", "demand_sha256", "duration_seconds",
                  "routing", "restriction", "completion_rate", "unfinished") + METRICS
        writer = csv.DictWriter(file, fields)
        writer.writeheader()
        for report in records:
            for controller, result in report["results"].items():
                writer.writerow({**{field: report[field] for field in ("scenario", "seed", "demand_sha256",
                                  "duration_seconds", "routing", "restriction")},
                                 **{field: result[field] for field in METRICS},
                                 "controller": controller, "checkpoint_sha256": hashes.get(controller),
                                 "completion_rate": result["completed"] / result["scheduled"],
                                 "unfinished": result["scheduled"] - result["completed"]})
    for scenario in {scenario for scenario, _ in tasks}:
        matches = [record for record in records if record["scenario"] == scenario]
        summaries[scenario] = {}
        for key in matches[0]["results"]:
            rows = [record["results"][key] for record in matches]
            delays = [r["completed_delay_mean_sec"] for r in rows if r["completed_delay_mean_sec"] is not None]
            summaries[scenario][key] = {
                "mean_delay": mean(delays) if delays else None,
                "delay_sd": stdev(delays) if len(delays) > 1 else None,
                "completion_rate": mean(r["completed"] / r["scheduled"] for r in rows),
                "mean_queue": mean(r["mean_queue"] for r in rows),
            }
    (out / "results.json").write_text(json.dumps({"summary": summaries, "checkpoint_sha256": hashes,
                                    "records": records}, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps(summaries), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/classroom_ppo/evaluation"))
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    evaluate(args.out, args.jobs)
