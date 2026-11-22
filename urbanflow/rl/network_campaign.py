"""Train isolated candidates on each selectable UrbanFlow map."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from urbanflow.core.districts import DISTRICTS
from .train import train

NETWORKS = ("classroom_grid", *DISTRICTS, "bronx_fordham", "procedural")


def train_network(task):
    map_name, episodes, duration, root, teacher_demos = task
    print(f"Training {map_name}", flush=True)
    train(SimpleNamespace(variant="neighbor", seed=117, episodes=episodes,
                          map_name=map_name, duration=duration, smoke=False,
                          out=root / map_name, teacher_demos=teacher_demos))
    print(f"Finished {map_name}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--duration", type=int, default=180)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--teacher-demos", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("data/network_ppo_candidates"))
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Choose a fresh output directory to preserve previous candidate runs.")
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(train_network, [(name, args.episodes, args.duration, args.out, args.teacher_demos)
                                     for name in NETWORKS]))
