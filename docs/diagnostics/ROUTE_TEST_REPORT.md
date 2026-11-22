# Route Test Drive investigation

## Checkpoint and historical result

The current Classroom Grid route test and the main simulation both instantiate the live controller factory. Each route result records the checkpoint on the actual controller instance. The current PPO run used `data/classroom_ppo_directional_v2/neighbor/weights.pt` (SHA-256 `6c0816d6124188e16cfa38d67753aec06a5742236e9572655d1474af40687a5f`), the same observation builder and normalization as live, and deterministic argmax at five simulated seconds. Neither checkpoint was changed.

The reported historical result was 98.75 s for PPO and 71.00 s for Fixed-Time, with 112 background departures and a 350 s cutoff. The historical report supplied in the brief lacks the exact map, road IDs, demand schedule, and checkpoint path needed for an exact replay. In the source inspected before this change, the exact experimental suffix was added for non-Classroom maps, where the loader resolves a network-specific checkpoint. That makes the old label insufficient evidence of Directional v2. The new browser run was a separate Classroom Grid case with 246 scheduled background departures and a 300 s cutoff.

## What slowed a current Directional v2 run

For West → East, departure 0, seed 6101, Directional v2 took **62.25 s** and Fixed-Time took **57.30 s**. Both entered at 0.05 s. Directional v2 stopped **7.40 s at North-West**, versus **17.35 s** for Fixed-Time, then lost **12.45 s at North-East**, where Fixed-Time did not stop.

At North-East the Directional v2 car entered the approach on east–west green at 24.15 s. The agent requested N/S at 30.05 s (75.46% SWITCH), before the test car reached the stop line. The test car stopped at 37.35 s. The agent still chose HOLD at 40.05 s (99.50% HOLD). The agent requested E/W at 45.05 s (95.40% SWITCH), and the simulator gave it green permission at 49.55 s. It crossed at 51.65 s. Of the stopped time there, 7.75 s occurred on the opposing green, 4.45 s in clearance, and 0.25 s in other motion constraints. No nearby leader was recorded during that stop. This is evidence of an arrival missing a downstream green in this run. The recorded decisions were valid and fresh at their five-second cadence.

Directional v2 trains on traffic queue and downstream storage terms. It has no objective for escorting the test sedan or coordinating a green wave for its particular arrival time. At West → East departure zero, it averaged 55.6 background trips completed versus Fixed-Time's 51.0 at the same 180 s horizon, and 5,942.71 versus 7,001.02 stopped vehicle-seconds. Its designated sedan was still 2.37 s slower on average. Actuated completed more background trips and recorded fewer stopped vehicle-seconds than Directional v2 in this group. These are separate measures of performance.

Additional training would be justified if corridor travel time becomes a deployment objective. A useful comparison would train a separate candidate on multi-origin demand with route travel time or delay by direction in the objective, then validate on held-out routes and departure times as well as network queues, throughput, and safety. This change preserves the current policy.

## Repeated experiment

Each group uses five independently seeded Rush Hour schedules (6101–6105) and an identical sedan, departure time, fixed route, background schedule, cutoff and empty starting state across controllers within a run. Routing is off. Both routes use the Classroom Grid. Network statistics exclude the test car and use a common 180 s horizon. Trip means and sample standard deviations use completed tests. Intersection stopping includes observed waiting for unfinished tests up to the cutoff. Wait buckets use observed signal or traffic context and can overlap in reality. A leader ID is recorded when the simulator's gap identifies a nearby leader. Green permission excludes a known conflicting junction reservation, and the actual crossing is recorded separately.

See [the summary](route-study/summary.json) and the compressed per-seed logs in `route-study/`. The logs include the departure schedule, checkpoint path, each test vehicle's trajectory samples or approach ticks, times and observed vehicles, and signal decisions at the test intersections.

```powershell
python -m urbanflow.core.route_study docs/diagnostics/route-study --workers 4
```

## Results by route and departure

### West → East at 0 s

| Controller | Mean trip (s) | Sample SD (s) | Completed | Mean stopped at intersections (s) |
|---|---:|---:|---:|---:|
| Multi-Agent RL Model · Directional v2 | 57.01 | 6.85 | 5/5 (100%) | 14.79 |
| Fixed-Time | 54.64 | 2.79 | 5/5 (100%) | 18.16 |
| Actuated | 41.83 | 4.04 | 5/5 (100%) | 4.72 |
| MaxPressure | 37.59 | 0.44 | 5/5 (100%) | 2.30 |
| Cooperative Predictive | 37.59 | 0.44 | 5/5 (100%) | 2.30 |

### West → East at 35 s

| Controller | Mean trip (s) | Sample SD (s) | Completed | Mean stopped at intersections (s) |
|---|---:|---:|---:|---:|
| Multi-Agent RL Model · Directional v2 | 78.20 | 8.31 | 5/5 (100%) | 30.73 |
| Fixed-Time | 120.86 | 6.46 | 5/5 (100%) | 72.39 |
| Actuated | 78.67 | 37.67 | 5/5 (100%) | 33.55 |
| MaxPressure | 78.98 | 11.09 | 5/5 (100%) | 33.79 |
| Cooperative Predictive | 79.65 | 11.57 | 5/5 (100%) | 34.21 |

### North → South at 0 s

| Controller | Mean trip (s) | Sample SD (s) | Completed | Mean stopped at intersections (s) |
|---|---:|---:|---:|---:|
| Multi-Agent RL Model · Directional v2 | 35.25 | 10.95 | 5/5 (100%) | 5.17 |
| Fixed-Time | 30.01 | 3.99 | 5/5 (100%) | 0.67 |
| Actuated | 54.77 | 33.90 | 5/5 (100%) | 23.39 |
| MaxPressure | 55.44 | 29.07 | 5/5 (100%) | 23.60 |
| Cooperative Predictive | 55.52 | 29.20 | 5/5 (100%) | 23.75 |

### North → South at 35 s

| Controller | Mean trip (s) | Sample SD (s) | Completed | Mean stopped at intersections (s) |
|---|---:|---:|---:|---:|
| Multi-Agent RL Model · Directional v2 | 111.56 | 45.16 | 5/5 (100%) | 65.00 |
| Fixed-Time | 56.50 | 17.46 | 5/5 (100%) | 16.05 |
| Actuated | 108.51 | 13.80 | 5/5 (100%) | 52.90 |
| MaxPressure | 216.61 | 64.04 | 4/5 (80%) | 179.08 |
| Cooperative Predictive | 223.76 | 64.39 | 4/5 (80%) | 181.07 |

North → South at 35 s was Directional v2's largest deficit versus Fixed-Time. The v2 car averaged 46.40 s stopped at South-West, versus 7.18 s under Fixed-Time. The v2 South-West wait averaged 32.43 s in opposing green, 9.24 s in clearance, 3.35 s in junction conflict, 1.22 s in nearby leader, 0.16 s in other.
