"""Train a versioned directional classroom policy with imitation and PPO."""
import argparse
import json
from pathlib import Path

import torch
from torch.distributions import Categorical

from .diagnose import PATTERNS, schedule
from .env import ClassroomEnv, DECISION_SECONDS, DT, SCHEMA, fingerprint, observe
from .policy import Policy
from urbanflow.core.kinematic_sim import KinematicSimulator


def rollout(policy, pattern, seed, mode):
    env = ClassroomEnv()
    sim = env.sim = KinematicSimulator("classroom_grid")
    sim.set_demand(schedule(sim, pattern, seed), pattern)
    env.duration, env.neighbor_aware = 180, True
    sim.step(DT, snapshot=False)
    env.agents, obs = observe(sim, True)
    rows = []
    while True:
        x = torch.from_numpy(obs)
        with torch.no_grad():
            logits, value = policy(x)
            distribution = Categorical(logits=logits)
            if mode == "teacher":
                actions = []
                for i, node in enumerate(env.agents):
                    signal = sim.signal_controllers[node]
                    current = signal.active_green_phase
                    pressure = obs[i, :2] - .35 * obs[i, 2:4]
                    actions.append(int(signal.stage == "GREEN" and signal.time_in_green >= signal.min_green
                                       and pressure[1 - current] > pressure[current] + .035))
                action = torch.tensor(actions)
            elif mode == "ppo":
                action = distribution.sample()
            else:
                action = logits.argmax(-1)
        obs, reward, done = env.step(action.tolist())
        # Keep the storage penalties from the classroom objective. Remove the
        # blanket switch cost for this candidate because the FSM limits switching.
        reward = torch.as_tensor(reward, dtype=torch.float32) + .02 * action
        rows.append((x, action, distribution.log_prob(action), value, reward))
        if done:
            return rows, sim


def main(directory):
    torch.set_num_threads(1)
    torch.manual_seed(682)
    directory.mkdir(parents=True, exist_ok=False)
    policy = Policy()
    examples = [record for i in range(12)
                for record in rollout(policy, PATTERNS[i % 4], 1700 + i, "teacher")[0]]
    obs, target = (torch.cat(tuple(row[col] for row in examples)) for col in (0, 1))
    optimizer = torch.optim.Adam(policy.parameters(), lr=.003)
    counts = torch.bincount(target, minlength=2)
    weights = (target.numel() / (2 * counts)).float()
    for _ in range(45):
        for indices in torch.randperm(len(target)).split(128):
            loss = torch.nn.functional.cross_entropy(policy(obs[indices])[0], target[indices], weight=weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    optimizer = torch.optim.Adam(policy.parameters(), lr=.0001)
    best = -float("inf")
    with (directory / "training.jsonl").open("w", encoding="utf-8") as log:
        for episode in range(1, 5):
            rows, _ = rollout(policy, PATTERNS[episode - 1], 1800 + episode, "ppo")
            observations, actions, old, values, rewards = zip(*rows)
            remaining, discounted = torch.zeros_like(rewards[0]), []
            for reward in reversed(rewards):
                remaining = reward + .99 * remaining
                discounted.append(remaining)
            targets = torch.cat(discounted[::-1])
            advantages = targets - torch.cat(values)
            advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
            batch, selected, old_probability = map(torch.cat, (observations, actions, old))
            for _ in range(4):
                logits, scores = policy(batch)
                distribution = Categorical(logits=logits)
                ratio = (distribution.log_prob(selected) - old_probability).exp()
                objective = torch.minimum(ratio * advantages, ratio.clamp(.8, 1.2) * advantages).mean()
                loss = -objective + .5 * (targets - scores).square().mean() - .01 * distribution.entropy().mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), .5)
                optimizer.step()
            validation = [rollout(policy, pattern, 1900 + i, "greedy") for i, pattern in enumerate(PATTERNS)]
            score = sum(sim.total_arrived for _, sim in validation)
            summary = {"episode": episode, "completed": [sim.total_arrived for _, sim in validation],
                       "training_switches": int(selected.sum()), "validation_total": score}
            print(json.dumps(summary), flush=True)
            log.write(json.dumps(summary) + "\n")
            log.flush()
            if score > best:
                best = score
                torch.save(policy.state_dict(), directory / "weights.pt")
                metadata = {"variant": "neighbor", "map_name": "classroom_grid",
                            "topology": fingerprint(validation[0][1]), "features": SCHEMA, "dt": DT,
                            "decision_seconds": DECISION_SECONDS, "smoke": False, "training_seed": 682,
                            "duration": 180, "torch": str(torch.__version__), "episode_budget": 4,
                            "teacher_examples": counts.tolist(), "teacher_rollouts": 12,
                            "teacher_seed_range": [1700, 1711], "ppo_seed_range": [1801, 1804],
                            "validation_seed_range": [1900, 1903], "selected": summary}
                (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    main(parser.parse_args().directory)
