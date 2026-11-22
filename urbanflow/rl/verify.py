"""Verify the saved Classroom Grid policies using the launcher's Python."""
import sys

from urbanflow.core.kinematic_sim import KinematicSimulator
from urbanflow.rl.env import observe
from urbanflow.rl.live import CLASSROOM_NEIGHBOR, POLICIES, ROOT, infer, load, load_directory


def main():
    import torch

    print(f"Python: {sys.executable}")
    print(f"PyTorch: {torch.__version__} (CPU inference)")
    simulator = KinematicSimulator("classroom_grid")
    for key, variant in POLICIES.items():
        load(simulator, key)
        print(f"{variant}: checkpoint compatibility and inference passed for "
              f"{len(simulator.signal_controllers)} agents.")
    if CLASSROOM_NEIGHBOR != ROOT / "neighbor":
        original = load_directory(simulator, "PPO_NEIGHBOR", ROOT / "neighbor")
        agents, observations = observe(simulator, True)
        infer(original, observations, len(agents))
        print("Original neighbor checkpoint preserved and verified.")


if __name__ == "__main__":
    main()
