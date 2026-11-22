"""Record matched Classroom Grid PPO behavior under directional demand."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
from statistics import mean

import numpy as np
import torch

from .env import DECISION_SECONDS, DT, SCHEMA, observe, requests
from .live import ROOT, load_directory
from urbanflow.core.demand import project_events
from urbanflow.core.kinematic_sim import KinematicSimulator

SEEDS = (2501, 2502, 2503)
DURATION = 180
PATTERNS = ("balanced", "east_west", "north_south", "reversal")


def schedule(sim, pattern, seed):
    rng = random.Random(seed)
    events = []
    for i in range(230):
        t = i * .65
        favored = "ew" if pattern == "east_west" else "ns"
        if pattern == "reversal":
            favored = "ew" if t < 75 else "ns"
        east_west = rng.random() < (.5 if pattern == "balanced" else .85 if favored == "ew" else .15)
        side = rng.choice(("west", "east") if east_west else ("north", "south"))
        events.append((t, side))
    return project_events(sim, events, pattern, rng)


def trial(policy, pattern, seed, sample, details):
    sim = KinematicSimulator("classroom_grid")
    sim.set_demand(schedule(sim, pattern, seed), pattern)
    generator = torch.Generator().manual_seed(seed + 9000)
    counts, aligned, queues, switch_probs, transitions = Counter(), Counter(), [], [], []
    stages = {}
    last = {}
    for tick in range(round(DURATION / DT)):
        sim.step(DT, snapshot=False)
        for node, signal in sim.signal_controllers.items():
            phase = (signal.stage, signal.active_green_phase)
            if phase != stages.get(node):
                if stages and node in stages:
                    transitions.append([node, round(sim.sim_time, 2), *phase,
                                        last.get(node)])
                stages[node] = phase
        if tick % round(DECISION_SECONDS / DT):
            continue
        agents, obs = observe(sim, True)
        with torch.inference_mode():
            logits, _ = policy(torch.from_numpy(obs))
            p = logits.softmax(-1)
            actions = torch.multinomial(p, 1, generator=generator).flatten() if sample else p.argmax(-1)
        phases = requests(sim, agents, actions.tolist())
        for index, node in enumerate(agents):
            signal = sim.signal_controllers[node]
            direction = int(signal.active_green_phase)
            gap = float(obs[index, 1 - direction] - obs[index, direction])
            if signal.stage == "GREEN" and signal.time_in_green + 1e-8 >= signal.min_green and abs(gap) >= .04:
                key = "opposing" if gap > 0 else "current"
                aligned[key] += 1
                aligned[key + "_switches"] += int(actions[index])
        sim.apply_signal_phases(phases)
        for index, node in enumerate(agents):
            signal = sim.signal_controllers[node]
            action = int(actions[index])
            counts[action] += 1
            ns, ew = map(float, obs[index, :2])
            queues.append(ns + ew)
            switch_probs.append(float(p[index, 1]))
            last[node] = [round(sim.sim_time, 2), action, phases[node]]
            if details is not None:
                details.write(json.dumps({"pattern": pattern, "seed": seed, "node": node, "time": sim.sim_time,
                    "features": dict(zip(SCHEMA, obs[index].tolist())), "dtype": str(obs.dtype),
                    "logits": logits[index].tolist(), "probabilities": p[index].tolist(),
                    "action": action, "requested": phases[node], "stage_after_request": signal.stage}) + "\n")
    if details is not None:
        details.write(json.dumps({"pattern": pattern, "seed": seed, "transitions": transitions}) + "\n")
    return {"seed": seed, "hold": counts[0], "switch": counts[1], "completed": sim.total_arrived,
            "entered": sim.total_arrived + len(sim.vehicles), "mean_queue": round(mean(queues), 4),
            "switch_probability_range": [round(min(switch_probs), 4), round(max(switch_probs), 4)],
            "directional_decisions": dict(aligned),
            "yellow_transitions": sum(row[2] == "YELLOW" for row in transitions),
            "green_transitions": sum(row[2] == "GREEN" for row in transitions)}


def probe(policy):
    _, start = observe(KinematicSimulator("classroom_grid"), True)
    cases = np.repeat(start[:1], 4, axis=0)
    cases[:, :2] = ((.75, .02), (.02, .75), (.75, .75), (.02, .02))
    cases[:, SCHEMA.index("elapsed_green")] = .5
    with torch.inference_mode():
        logits, _ = policy(torch.from_numpy(cases))
        return logits.tolist(), logits.softmax(-1).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "neighbor")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sim = KinematicSimulator("classroom_grid")
    policy = load_directory(sim, "PPO_NEIGHBOR", args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".jsonl").open("w", encoding="utf-8") as details:
        report = {"selection": "categorical" if args.sample else "argmax",
                  "checkpoint": str(args.checkpoint), "probe": probe(policy),
                  "scenarios": {name: [trial(policy, name, seed, args.sample, details) for seed in SEEDS]
                                for name in PATTERNS}}
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
