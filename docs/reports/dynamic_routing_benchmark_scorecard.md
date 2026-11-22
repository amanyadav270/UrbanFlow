# UrbanFlow AI — Dynamic Routing Counterfactual Benchmark Scorecard

Generated at: 2026-11-14 03:34:10 UTC

## 1. Experimental Regimes Evaluated
- **Regime A (Baseline):** No dynamic rerouting (static shortest path).
- **Regime B (Reactive):** Dynamic routing using observed V/C and queue telemetry (risk weight $\lambda_{\text{pred}} = 0$).
- **Regime C (Predictive):** Predictive capacity-aware routing with advisory spillback risk from frozen 10-min classifier.

## 2. Benchmark Summary Table

| Scenario Family | Seed | Regime | Completed Trips | Mean Travel Time (s) | Mean Delay (s) | Throughput (veh/hr) | Spillback Events | Spillback Dur (s) | Total Reroutes | Diversion Rate | Peak V/C | Alt VKT (km) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FAM-05 | 5001 | baseline | 3474 | 97.5 | 40.4 | 4168.8 | 167 | 1750.0 | 0 | 0.000 | 12.49 | 63.18 |
| FAM-05 | 5001 | reactive | 3482 | 97.7 | 40.5 | 4178.4 | 168 | 1879.8 | 13 | 0.003 | 12.52 | 63.12 |
| FAM-05 | 5001 | predictive | 3469 | 98.3 | 41.1 | 4162.8 | 170 | 1875.0 | 18 | 0.004 | 12.52 | 63.36 |

## 3. Incomplete Trips & Network Integrity Accounting
Trips unfinished at simulation conclusion are strictly segregated from completed trip statistics.

