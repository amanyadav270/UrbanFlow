# Classroom Grid policy diagnostics

## Root cause

Live and training both construct observations with `urbanflow.rl.env.observe` and map HOLD (0) or SWITCH (1) with `requests`. The original checkpoint produced changing but near-even probabilities, and SWITCH never won argmax in the 1,728 logged decisions. Its greedy validation score stayed at −0.1752409 through all twelve 180-second training episodes, and selection retained episode one. The logs do not contain original training action counts or support a convergence claim. Training sampled categorical actions and vintage validation used argmax. The reward included a flat 0.02 switch cost, a plausible contributor to HOLD bias. Its independent effect was not isolated.

The original queue signals represent the maximum stopped-vehicle share of any controlled inbound link on the N/S or E/W axis. Storage is estimated from link length and lane count at 7.5 m per vehicle, and the input clips to [0,1]. Stopped means below 0.5 m/s. These are two axis features within the 31-feature float32 schema. Separate cardinal-direction inputs do not exist in the checkpoint. Regression tests rebuilt the two features from simulator vehicles by road, checked nonidentical agent observations, and verified opposite-direction phase requests, clearance and a vehicle stopping at red and proceeding on green.

The old model's SWITCH probabilities were 48.54% for high N/S, 47.43% for high E/W, 47.55% for both high and 48.42% for both low in the isolated observation probe. Directional v2's corresponding values were 99.87%, 99.56%, 2.93%, and 24.73%. Varying only the queue columns creates combinations outside typical trajectories, so the matched simulator results below are the behavioral check.

## Retraining and held-out results

Directional v2 used 12 simulator-generated imitation rollouts with an explicit directional pressure teacher during training, followed by four PPO episodes on distinct seeds. This candidate retained the same network architecture, observation schema, 0.05 s physics and five-second decisions. The training reward removed the flat switching penalty while retaining the queue and downstream storage terms. Episode two was selected on validation seeds 1900–1903 by total completed trips. The teacher is absent from live inference. Live uses greedy neural decisions and the existing minimum-green, yellow, all-red, reservation and maximum-green rules. The original checkpoint is preserved.

Each row uses three 180-second runs with 230 scheduled departures per run and seeds 2501–2503. The directional schedule selects the favored entry axis with 85% probability. Reversal changes the favored axis at 75 seconds. Hold and switch totals cover all four agents, including decisions during clearance. Trips are the mean across seeds.

| Demand | Original greedy HOLD / SWITCH | Original sampled HOLD / SWITCH | Directional v2 HOLD / SWITCH | Trips original greedy / sampled / v2 |
|---|---:|---:|---:|---:|
| Balanced | 432 / 0 | 249 / 183 | 320 / 112 | 64.33 / 60.00 / 59.33 |
| Heavy E/W | 432 / 0 | 248 / 184 | 346 / 86 | 44.00 / 51.67 / 58.67 |
| Heavy N/S | 432 / 0 | 249 / 183 | 357 / 75 | 59.33 / 48.00 / 63.33 |
| Reversal | 432 / 0 | 248 / 184 | 340 / 92 | 44.33 / 51.67 / 58.67 |

Directional v2 completed 720 trips across all seeds and scenarios, compared with 636 for original greedy and 634 for original categorical sampling. At minimum-green eligibility, with a queue-input gap of at least four percentage points, Directional v2 switched in 319 of 344 decisions when the opposing queue was larger and 3 of 335 when the currently served queue was larger. Directional v2 recorded 375 transitions to green, including automatic safety-cycle transitions. The balanced scenario regressed versus the original, and this study only compares policies on the tested synthetic runs.

The first retrained and archived teacher candidates were also evaluated. Their results are retained in this folder. Neither was selected because their heavy N/S results regressed substantially.

## Reproduce

```powershell
python -m urbanflow.rl.diagnose --output docs/diagnostics/original-greedy.json
python -m urbanflow.rl.diagnose --sample --output docs/diagnostics/original-sampled.json
python -m urbanflow.rl.diagnose --checkpoint data/classroom_ppo_directional_v2/neighbor --output docs/diagnostics/directional-v2-greedy.json
```

The JSONL logs include real per-agent input vectors, data type, raw logits, probabilities, requests and timed signal transitions. The decision history in the browser distinguishes requests from applied greens and shows stopped-vehicle counts separately from the normalized queue feature. Clearance estimates are earliest stage-end times and may extend while a junction remains occupied.
