# Classroom experiment branch

The dashboard starts paused with Classroom grid, fixed-time signals and the
seed-42 synthetic Rush hour schedule. Select Start to run, then focus a junction
from Traffic signals to see the latest request and the current simulator stage.

Experiments runs the active schedule through the four baseline controllers.
Each replay starts with an empty network and uses the same departure records,
with routing off. Download the CSV from the result table. The API at
`/api/experiments/matched` also returns the demand hash and evaluation horizon.
Delay is measured for completed trips as system travel time minus free-flow
travel time on the original route, floored at zero. Check completion and the
cutoff counts alongside it. Archived results have their own SUMO study heading.

## Measured smoke experiment

Ran `compare_classroom(scenario="normal")` with its default seed 42 and
75 scheduled departures. All four controllers completed all 75 trips.

| Controller | Mean completed-trip delay (s) |
|---|---:|
| Fixed-Time | 47.98 |
| Actuated | 32.27 |
| MaxPressure | 39.61 |
| Cooperative Predictive | 42.50 |

This is one matched schedule. The held-out evaluation below uses separate seeds.

## Shared PPO policies

Install the base requirements and optional CPU PyTorch requirements.

```powershell
python -m pip install -r requirements.txt -r requirements-rl.txt
$env:URBANFLOW_PORT = '8002'
python run_demo.py --no-browser
python -m urbanflow.rl.train local --episodes 12 --seed 100 --duration 180
python -m urbanflow.rl.train neighbor --episodes 12 --seed 100 --duration 180
python -m urbanflow.rl.evaluate --jobs 4
```

For a wiring check use `python -m urbanflow.rl.train local --smoke --out data/smoke`.
Smoke checkpoints are ineligible for the live demo. The bundled weights were
trained with CPU PyTorch 2.14.0. Each run used 12 episodes drawn from Normal,
Commuter reversal and Downstream restriction using schedule seeds sampled from
100–999. Validation used seed 1000; the test seeds are 5001–5005.
Local selected episode 3 (run 117.641 s); neighbor selected episode 1
(run 117.922 s). The metadata retains the original topology digest and the
verified digest after excluding the transient current phase field.

The simulator advances at 50 ms. The first decision is at 0.05 s, then every
5 s. Each of the four junctions uses the same actor and critic. The actor
samples during training and chooses the larger probability during replay.
Action 0 requests the current green; action 1 requests the opposite green.
All requests go through the existing simulator FSM. Rollouts end at the first
decision boundary at or after the episode horizon. PPO clips the policy ratio
to 0.8–1.2 and uses discounted returns (0.99), four optimization passes per
episode, Adam at 0.0003, and entropy weight 0.01.

The exact feature names and order are stored in checkpoint metadata. Local
features group queues and outgoing storage use by NS/EW, followed by green
bits, elapsed green divided by maximum green, and approach-presence bits.
Queue ratios use a 7.5 m storage cell and the most occupied link in the
group. Neighbor slots follow junction ID order and contain presence,
both queue ratios and both green bits. Unused slots are zero, including all
neighbor slots for the local policy. The per-junction reward penalizes mean
queue ratio, outgoing occupancy at or above 75% (0.2), and switch requests
(0.02). Both variants use the same reward.

## Demo sequence

1. Start Fixed-Time on Classroom with Downstream restriction. The scenario
   slows the eastbound internal link to 3 m/s from 20 to 110 simulated seconds.
2. Select PPO · Neighbor under Simulation settings. It resets to a paused run
   on the same demand. Click **Step decision** and open **Traffic signals**.
   Select two junctions to inspect their actual feature values, requests and
   simulator stages. Start playback to observe clearance and green changes.
3. Open **Experiments**, run a matched comparison, view the upstream
   queue traces and export the comparison CSV. The held-out CSV includes
   seed, departure hash, policy hash and fixed evaluation window.

## Measured held-out results

Each cell is mean completed-trip delay in seconds (sample SD across five
seeds), followed by completion percentage. All replays used the same departures
within each scenario and seed, routing off, and the same fixed cutoff per
schedule. Raw metrics, hashes and one-second queue samples are in
`data/classroom_ppo/evaluation/`.

| Controller | Normal | Commuter reversal | Downstream restriction |
|---|---:|---:|---:|
| Fixed-Time | 51.88 (2.36), 93.1% | 144.76 (2.39), 30.2% | 99.56 (5.16), 35.4% |
| Actuated | 41.89 (2.26), 100.0% | 127.33 (9.12), 50.1% | 89.63 (6.68), 50.9% |
| MaxPressure | 46.03 (3.37), 97.1% | 131.83 (6.77), 45.5% | 88.17 (5.89), 46.2% |
| Cooperative Predictive | 47.86 (2.95), 97.6% | 134.33 (8.72), 45.1% | 89.35 (8.01), 44.9% |
| PPO Local | 65.19 (4.49), 87.2% | 172.17 (6.09), 34.8% | 114.15 (6.04), 38.3% |
| PPO Neighbor | 65.19 (4.49), 87.2% | 172.61 (5.26), 34.6% | 114.15 (6.04), 38.3% |

Both policies had higher completed-trip delay than all four baselines on
all three scenario averages. Neighbor features tied Local on Normal and
Downstream restriction, and added 0.44 s of delay on Commuter reversal.
Actuated had the highest completion rate in each scenario. The restriction
trace shows why the preset works for an inspector demo: on seed 5001,
Fixed-Time's northwest queue was 2 before the restriction (19 s), 19 during
it (80 s) and 7 at 160 s. PPO's queue was 2, 12 and 18 at those same times.

The two training commands used `D:\TRAFFIC\marl-traffic\.venv\Scripts\python.exe`.
Evaluation used that interpreter for CPU Torch, with the Delivery
virtual environment site-packages first on `PYTHONPATH` to preserve the pinned
simulator dependencies. Both checkpoints reloaded in the evaluator workers
and the preview server. The Windows full suite ran with the Delivery
virtual environment using `python -m pytest tests -q`: 233 passed, with one
Starlette test-client deprecation warning.
