# Presenting a route test drive

The route planner answers two questions. **Find route** estimates a path
from the live simulated conditions. **Drive and compare** holds that path
fixed and asks what changes when the signals use a different controller.
The gold marker replays a sedan's recorded simulator positions. The result
includes entry waiting and reports unfinished trips separately.

## Three ways to use it

1. **Corridor retiming.** A traffic engineer can compare a fixed plan and
   an adaptive controller on the same commuter route before a field trial.
   FHWA describes evaluating retiming with probe cars driven over the
   same routes before and after a change.
   [FHWA Traffic Signal Timing Manual](https://ops.fhwa.dot.gov/publications/fhwahop08024/chapter7.htm)
2. **Work-zone planning.** Use Classroom's Downstream restriction scenario
   and choose a route through the affected corridor. Compare timing plans
   before deciding whether to collect field data or try a retiming plan.
   [FHWA work-zone management guidance](https://ops.fhwa.dot.gov/wz/resources/publications/trans_mgmt_plans/sec4.htm)
3. **Shift-change or commuter planning.** Choose Commuter reversal and
   compare the same trip in the two directions. Check the Experiments
   completion and queue metrics before recommending a plan. The route test
   measures one traveler, while a reliability study requires repeated runs.
   [FHWA adaptive signal validation guidance](https://ops.fhwa.dot.gov/publications/fhwahop13031/chap4.htm)

## Two-minute demo

1. Choose Classroom grid, Rush hour demand, and Fixed-Time. Find the route
   from INT_NW to INT_SE while the simulation is paused at its start.
2. Click **Drive and compare**. Point to the gold sedan and note where it
   stops. Each comparison replay starts from an empty network with the
   same background departures, places the sedan on the same route due
   at simulation time 45 s, and keeps rerouting off.
3. At arrival, explain the time difference. In the verified seed-42 Rush
   hour run, Fixed-Time took 95.00 s, Actuated took 85.70 s, and both
   PPO variants took 50.50 s. PPO's test car arrived 44.50 s sooner than
   under Fixed-Time. Then open Experiments to show completion and queueing
   across the network. One favorable trip does not establish overall
   controller performance.

This is a simulated comparison of signal strategies. Field claims would
require measured demand and independent probe runs on the real corridor.
