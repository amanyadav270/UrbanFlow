# Delivery verification

## Bronx and Single Corridor update

- All 211 Python tests passed. The counterfactual-demand test checks identical departure schedules and greater entry blocking under an explicit cap. The uncapped run can still queue at a physically occupied entrance.
- Both comparison endpoints ran through HTTP on Bronx and Single Corridor, using the active network's repeatable Normal demand. Each returned all four signal controllers or all three routing modes as appropriate.
- In the Normal motion checks, each of these maps cleared all 75 trips. The Bronx regression sampled oriented vehicle bodies once per simulated second and found no overlaps. Single Corridor also had zero overlaps at the same sampling cadence.
- Bronx opposing traffic now follows separated carriageways. The simulator uses one active driving lane per direction, protects irregular junctions with exclusive reservations, and checks vehicle footprints near other vehicles. Fast-forward shows sampled poses directly so it does not draw straight-line motion across turns.

## Clearance and routing update

- Inspection of the reported live run found 80 stopped vehicles and two junctions holding all-red for thousands of simulated seconds. The downstream storage check now measures available space behind the lead vehicle's rear bumper.
- The new regression demonstrates that a long bus can leave too little room to admit a following car even when its center looks far enough away. The car enters once the full clearance fits.
- Live routing comparison held Fixed-Time constant over the same 246 Rush Hour departures. Off, Reactive and Predictive each completed 88 trips and accepted zero reroutes, yielding a tie. Live progress and rejection of a stale result were verified.
- A full 457-trip Peak Demand replay using CooperativePredictive signals and Predictive routing completed in 1842.9 simulated seconds. The longest network-wide standstill in one-second samples was 13 seconds. All 209 Python tests passed. JavaScript syntax and routing results rendering, tie messaging, and apply payload checks passed.

## Map dashboard and route planner

- Chromium with WebGL rendered the desktop Metro dashboard and Harbor at a 412 px mobile viewport. Browser interactions covered route estimation and highlighting, playback and the live vehicle-count chart, road search, incident injection, the simulation settings dialog, saved comparisons, map switching and the traffic overlay toggle.
- Python covered route detours around a sufficiently restricted segment and restoration of normal speeds at incident expiry. All 226 tests passed. Route travel time is an estimate from current link speeds, independent of the saved SUMO benchmark results.

## Saved comparisons, CSV import and city playback

- The live API returned stored four-controller SUMO results for all eight built-in presets without starting a simulation. Five new synthetic reference cases were evaluated on seeds 4101 through 4105 with the existing PPO checkpoint and three SUMO baselines. PPO beat SUMO Actuated on four of the five synthetic cases. A fixed-time baseline had the best mean delay in each of those cases. The measured evening profile still leads with PPO.
- Over live HTTP, the bundled raw count CSV loaded 1,388 trips on Harbor City. Changing volume to 4× produced 5,552 trips. Downloading and re-uploading that expanded schedule preserved its count. The regression test checks exact route, timing and lane round-trip. An uploaded 75-trip classroom CSV ran all four UrbanFlow controllers live, each completing all 75 trips.
- A WebSocket client timed playback over a four-second wall-clock interval with at least 100 active vehicles. At the 20× setting Metro improved from 6.48× to 9.71× and Harbor from 5.59× to 7.11×. The app keeps 50 ms physics steps and reports the achieved playback rate. Performance varies with the fleet, selected controller and machine.
- Both city scene meshes were built in a headless Three.js check with finite geometry and the expected signal counts. Metro built 146 buildings and 144 trees. Harbor built 186 buildings and 190 trees. The new appearance and dashboard interactions lacked an interactive browser walkthrough.

## City networks and measured traffic counts

- Matched all 48 bundled count records against the supplied NYC DOT source export. There are 16 directional quarter-hour count records in each window. The morning, midday and evening windows contain 573, 347 and 468 vehicles respectively. Tests confirm that UrbanFlow's Harbor city schedule preserves those counts by interval and direction. The individual trips in both simulators are generated from the measured counts.
- Verified over HTTP and the WebSocket that both city networks load and stream moving vehicles and live signal states while replaying the 468-vehicle evening dataset. Verified both dataset download endpoints and all 12 benchmark scenarios. The new city views lacked an interactive browser walkthrough.
- Under Actuated, Metro cleared all 360 commuter reversal trips and Harbor cleared all 425 event release trips. Both runs used a 50 ms physics step. Sampling oriented bodies once per simulated second found zero overlaps on both runs after setting city stop bars back to clear long turning vehicles. The Python suite passed 222 tests, and the targeted bus-tail clearance regression passed separately.
- A full Harbor city replay of the measured evening profile cleared all 468 scheduled vehicles by 3,778.2 simulated seconds using Actuated control.
- Ran the existing `ppo_real_mix_s0` checkpoint and three SUMO baselines with identical generated departures on seeds 4101 through 4105. Results are mean time loss plus mean departure delay, measured in seconds. The existing checkpoints and archive were retained.

| Count window (Bronx, Jan 2013) | Saved PPO | SUMO Actuated | Lower delay |
|---|---:|---:|---|
| Jan 8 at 07:00 | 9.09 | 8.58 | Actuated |
| Jan 9 at 12:00 | 7.43 | 7.34 | Actuated |
| Jan 10 at 17:00 | 7.84 | 8.18 | PPO |

## City maps and RL benchmark dashboard

- All 216 Python tests passed on Windows. The five application JavaScript modules passed syntax checks.
- The report predates the current upload cleanup; Riverside boulevard, Freight district and University quarter are archived and are not active selectable networks in this copy.
- The retained live networks are Classroom Grid, Midtown Avenues, Metro Center, Harbor City, Bronx, NY and Single Corridor.
- The archived benchmark endpoint returned nine SUMO studies with five shared seeds per study, and the CSV endpoint returned the source file. Rankings use mean time loss plus mean departure delay. Six scenarios were led by an RL entry, using scenario-specific checkpoints in the archive.
- Reran the four single-intersection scenarios with the original PPO checkpoints and SUMO baselines on seeds 3001 through 3005 in an isolated copy. All four fresh CSVs matched the archived CSVs byte for byte. PPO ranked first in all four single-intersection runs. The five profile and grid studies were checked from the archive.
- Browser rendering and interaction on the new district maps and benchmark dialog lacked an interactive walkthrough. The UrbanFlow simulator does not run these SUMO checkpoints as live traffic controllers.

## Controller comparison update

- Matched comparison exercised through the live API on Normal, Rush Hour, Peak Demand and the included CSV. All four actual controllers ran each schedule. Normal completed 75 in each regime, and Adaptive ranked higher on queue and waiting measures. Rush Hour completed 122 with Adaptive versus 88 with Fixed-Time. Peak Demand completed 125 with Adaptive versus 88 with Fixed-Time. The uploaded sample tied.
- Observed progress for each of the four replays in the live Rush Hour comparison. All 208 Python tests passed.
- Applying a stale result after switching demand was rejected. The live uploaded run completed, auto-paused, replayed and reset. Routing published a real evaluation timestamp.
- Frontend component checks covered recommendation rendering, the guarded Apply action, ties and completed playback formatting. The saved browser permission still prevents an interactive visual walkthrough.

## Map workspace update

- The revised 600-second Peak Demand replay completed 240 trips, with 31 cars moving in the final sample. Every sampled second with an active vehicle recorded movement.
- Live Fixed-Time classroom fast-forward measured 16.3 simulated seconds per wall-clock second at the 20x setting. Pause and reset were verified over the same connection.
- Regression tests check simultaneous release on separated right turns and continue to check parallel straight traffic, occupancy, and clearance.
- A camera math check at 1191 by 614 confirmed that an off-center road location keeps its screen position while zooming. The saved browser permission prevented an interactive layout pass.

## Green-light and gridlock update

- All 205 Python tests passed. New regressions cover simultaneous green release from parallel lanes, turning vehicles ignoring a stopped car in the adjacent exit lane, and phase changes completing their clearance.
- The old live Peak Demand run with CooperativePredictive had 80 active cars and zero cars moving in six consecutive one-second samples. In the updated 600-second replay of that scenario, 222 trips completed, 28 cars were moving at the last sample, and every one-second sample with active vehicles recorded movement.
- The updated live server accepted map, demand, controller, routing, speed, pause and reset commands over its WebSocket and reported the corresponding state. The JavaScript modules passed syntax checks. The browser permission gate denied a repeat browser walkthrough of the revised dashboard.

## Earlier dashboard checks

The classroom simulation and browser were tested on Windows with Python 3.11.

| Check | Observed result |
|---|---|
| Python regression suite | 202 passed, including the added classroom safety, motion and replay tests |
| Frontend JavaScript | All five application modules passed Node syntax checks |
| Classroom physics | 4 junctions and 24 directed roads. Both matched controller modes reached at least 30 moving vehicles during the regression runs. Assertions covered turn continuity, same-lane spacing, red-light stopping, connector reservations and trip accounting |
| Windows launcher | Started the packaged application using the batch launcher on a test loopback port |
| Live HTTP | Switched all three maps, all four controller selections and all three routing modes. Accepted the corrected example CSV and a JSON schedule. Rejected a duplicate ID |
| Live WebSocket | Received topology and moving vehicle snapshots, verified pause, reset to an empty frame and replayed the same uploaded departure twice |
| Browser | Observed a live WebGL map and telemetry, Start, Pause, a speed change, top-down and angled overview, the guided Adaptive switch and the measured comparison table. Captured both views below |
| Model and dataset files | Verified against the source archive when packaging |

- [Earlier top-down dashboard with live vehicles](classroom-dashboard.jpg)
- [Earlier 3D dashboard and matched comparison](classroom-comparison.jpg)
- [Complete measured comparison JSON](classroom-comparison.json)

## Limits of verification

The browser file picker reached the upload endpoint and displayed its validation error for an invalid road in the original example. The example was corrected and the successful CSV and JSON upload paths were verified over live HTTP. The browser permission gate declined the final camera and inspector pass after the screenshots. Drag-to-orbit, pan, zoom and the final follow-vehicle walkthrough therefore lack end-to-end browser sign-off.

The comparison covers one synthetic schedule with routing off. MaxPressure, CooperativePredictive and the routing modes have existing automated coverage and accepted live mode switches. Their operational gains, the transfer of trained forecasts to the classroom grid, real-world signal safety, and the optional SUMO setup were outside the verified results. Classroom signal heads are grouped by approach axis, with individual turning movements governed by the conflict-zone reservation.
