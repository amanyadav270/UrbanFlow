"""Train one Classroom PPO variant, using validation return to select weights."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.distributions import Categorical
from torch.nn.functional import cross_entropy

from .env import ClassroomEnv, DT, DECISION_SECONDS, SCHEMA, fingerprint
from .policy import Policy
from urbanflow.core.classroom_evaluation import classroom_controller


def warm_start(policy, optimizer, scenarios, duration, map_name, neighbor, demos):
    """Fit Actuated requests on training-only schedules before PPO updates."""
    observations, actions = [], []
    for index in range(demos):
        env = ClassroomEnv()
        obs = env.reset(1400 + index, scenarios[index % len(scenarios)], neighbor, duration, map_name)
        teacher = classroom_controller("ACTUATED")
        while True:
            sim = env.sim
            phases = teacher.compute_phases(sim.sim_time, simulator=sim, telemetry=None)
            labels = []
            for node in env.agents:
                current = sim.signal_controllers[node].active_green_phase
                requested = phases.get(node, current)
                labels.append(int(requested in (0, 1) and requested != current))
            observations.append(torch.from_numpy(obs))
            actions.append(torch.tensor(labels))
            obs, _, done = env.step(labels)
            if done:
                break
    batch, labels = torch.cat(observations), torch.cat(actions)
    # Switching is rare. Weight the two classes equally so the actor learns
    # observable differences between a hold and a switch.
    counts = torch.bincount(labels, minlength=2).clamp_min(1)
    weights = (labels.numel() / (2 * counts)).float()
    for _ in range(8):
        for indices in torch.randperm(len(labels)).split(128):
            loss = cross_entropy(policy(batch[indices])[0], labels[indices], weight=weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def rollout(policy, seed, scenario, neighbor, duration, greedy=False, map_name="classroom_grid"):
    env = ClassroomEnv()
    obs = env.reset(seed, scenario, neighbor, duration, map_name)
    records = []
    while True:
        row = torch.from_numpy(obs)
        with torch.no_grad():
            logits, value = policy(row)
            distribution = Categorical(logits=logits)
            action = logits.argmax(-1) if greedy else distribution.sample()
        obs, reward, done = env.step(action.tolist())
        records.append((row, action, distribution.log_prob(action), value,
                        torch.as_tensor(reward, dtype=torch.float32)))
        if done:
            return records, env


def train(args):
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    policy = Policy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=.0003)
    neighbor = args.variant == "neighbor"
    duration = 20 if args.smoke else args.duration
    directory = args.out / args.variant
    scenarios = ["normal", "city_shift", "incident" if args.map_name == "classroom_grid" else "rush_hour"]
    directory.mkdir(parents=True, exist_ok=True)
    start, best = time.monotonic(), float("-inf")
    if args.teacher_demos and not args.smoke:
        warm_start(policy, optimizer, scenarios, duration, args.map_name, neighbor, args.teacher_demos)
    with (directory / "training.jsonl").open("w", encoding="utf-8") as log:
        for episode in range(1, (1 if args.smoke else args.episodes) + 1):
            seed = int(rng.integers(100, 1000))
            scenario = str(rng.choice(scenarios))
            records, env = rollout(policy, seed, scenario, neighbor, duration, map_name=args.map_name)
            observations, actions, old, values, rewards = zip(*records)
            remaining, discounted = torch.zeros_like(rewards[0]), []
            for reward in reversed(rewards):
                remaining = reward + .99 * remaining
                discounted.append(remaining)
            targets = torch.cat(discounted[::-1])
            advantages = targets - torch.cat(values)
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
            batch, selected, old_probability = map(torch.cat, (observations, actions, old))
            for _ in range(4):
                logits, prediction = policy(batch)
                distribution = Categorical(logits=logits)
                ratio = (distribution.log_prob(selected) - old_probability).exp()
                objective = torch.minimum(ratio * advantages, ratio.clamp(.8, 1.2) * advantages).mean()
                loss = -objective + .5 * (targets - prediction).square().mean() - .01 * distribution.entropy().mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), .5)
                optimizer.step()
            validation, _ = rollout(policy, 1000, "city_shift", neighbor, duration, greedy=True, map_name=args.map_name)
            score = float(torch.stack([row[-1] for row in validation]).mean())
            summary = {"episode": episode, "seed": seed, "scenario": scenario,
                       "validation_reward": score, "wall_seconds": round(time.monotonic() - start, 3)}
            log.write(json.dumps(summary) + "\n")
            log.flush()
            print(json.dumps(summary), flush=True)
            if score > best:
                best = score
                torch.save(policy.state_dict(), directory / "weights.pt")
                metadata = {"variant": args.variant, "map_name": args.map_name, "topology": fingerprint(env.sim), "features": SCHEMA,
                            "dt": DT, "decision_seconds": DECISION_SECONDS, "smoke": args.smoke,
                            "training_seed": args.seed, "validation_seed": 1000, "torch": str(torch.__version__),
                            "duration": duration, "selected": summary, "episode_budget": args.episodes,
                            "teacher_demos": 0 if args.smoke else args.teacher_demos}
                (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=("local", "neighbor"))
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--duration", type=int, default=180)
    parser.add_argument("--map-name", default="classroom_grid")
    parser.add_argument("--teacher-demos", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("data/classroom_ppo"))
    train(parser.parse_args())
