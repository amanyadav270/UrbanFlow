"""Optional checkpoint loading and live policy decisions."""
from functools import lru_cache
import importlib.util
import json
from pathlib import Path

from urbanflow.intelligence.signal_control import SignalControllerBase
from .env import DECISION_SECONDS, DT, SCHEMA, fingerprint, observe, requests

POLICIES = {"PPO_LOCAL": "local", "PPO_NEIGHBOR": "neighbor"}
ROOT = Path(__file__).resolve().parents[2] / "data" / "classroom_ppo"
CANDIDATES = ROOT.parent / "network_ppo_candidates"
CLASSROOM_NEIGHBOR = ROOT.parent / "classroom_ppo_directional_v2" / "neighbor"


@lru_cache(maxsize=8)
def _load(directory, topology, variant, metadata_stamp, weights_stamp):
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    expected = {"variant": variant, "topology": topology, "features": list(SCHEMA),
                "dt": DT, "decision_seconds": DECISION_SECONDS}
    if any(metadata.get(key) != value for key, value in expected.items()):
        fields = ", ".join(key for key, value in expected.items() if metadata.get(key) != value)
        raise ValueError(f"Checkpoint is incompatible. Mismatched fields: {fields}.")
    if metadata.get("smoke") is not False:
        raise ValueError("Smoke checkpoint is excluded from live comparisons.")
    import torch
    from .policy import Policy

    torch.set_num_threads(1)
    policy = Policy()
    policy.load_state_dict(torch.load(directory / "weights.pt", map_location="cpu", weights_only=True))
    return policy.eval()


def checkpoint_directory(sim, key):
    variant = POLICIES[key]
    if sim.map_name != "classroom_grid":
        return CANDIDATES / sim.map_name / variant
    return CLASSROOM_NEIGHBOR if variant == "neighbor" else ROOT / variant


def load(sim, key):
    if importlib.util.find_spec("torch") is None:
        raise ValueError("PyTorch is missing. Run INSTALL_RL_DEPENDENCIES.bat with the same Python used to launch UrbanFlow.")
    variant = POLICIES[key]
    directory = checkpoint_directory(sim, key)
    if not (directory / "metadata.json").is_file():
        raise ValueError(f"There is no PPO ({variant}) checkpoint for this network.")
    policy = load_directory(sim, key, directory)
    try:
        agents, observations = observe(sim, variant == "neighbor")
        if sim.map_name == "classroom_grid" and len(agents) != 4:
            raise ValueError(f"Expected four intersection agents, found {len(agents)}.")
        infer(policy, observations, len(agents))
    except Exception as error:
        raise ValueError(f"Checkpoint inference validation failed: {error}") from error
    return policy


def load_directory(sim, key, directory):
    variant = POLICIES[key]
    try:
        return _load(directory, fingerprint(sim), variant,
                     (directory / "metadata.json").stat().st_mtime_ns,
                     (directory / "weights.pt").stat().st_mtime_ns)
    except Exception as error:
        raise ValueError(f"Cannot load {variant} checkpoint: {error}") from error


def infer(policy, observations, count):
    import torch

    obs = torch.from_numpy(observations)
    if obs.shape != (count, len(SCHEMA)) or not torch.isfinite(obs).all():
        raise ValueError("Invalid observation shape or values.")
    with torch.inference_mode():
        logits, values = policy(obs)
        if logits.shape != (count, 2) or values.shape != (count,):
            raise ValueError("Unexpected actor or critic output shape.")
        if not (torch.isfinite(logits).all() and torch.isfinite(values).all()):
            raise ValueError("Non-finite actor or critic output.")
        probabilities = logits.softmax(-1)
        if not torch.isfinite(probabilities).all():
            raise ValueError("Non-finite action probabilities.")
        return probabilities, probabilities.argmax(-1).tolist()


class PPOController(SignalControllerBase):
    decision_interval = DECISION_SECONDS

    def __init__(self, sim, key, *, directory=None):
        self.policy = load(sim, key) if directory is None else load_directory(sim, key, directory)
        path = (checkpoint_directory(sim, key) if directory is None else Path(directory)).resolve()
        self.checkpoint = ((path / "weights.pt").relative_to(ROOT.parent.parent).as_posix()
                           if path.is_relative_to(ROOT.parent.parent) else path.name + "/weights.pt")
        self.neighbor = POLICIES[key] == "neighbor"
        self.observations = {}

    def reset(self):
        self.observations = {}

    def compute_phases(self, current_time, simulator=None, telemetry=None):
        counts = {}
        agents, obs = observe(simulator, self.neighbor, queue_counts=counts)
        probabilities, actions = infer(self.policy, obs, len(agents))
        self.observations = {node: {"features": dict(zip(SCHEMA, obs[i].tolist())),
                                    "probabilities": probabilities[i].tolist(),
                                    "stopped": counts[node],
                                    "action": actions[i]} for i, node in enumerate(agents)}
        return requests(simulator, agents, actions)
