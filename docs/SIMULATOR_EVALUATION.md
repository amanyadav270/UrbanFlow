# UrbanFlow AI — Traffic Simulator Evaluation & Strategy

## 1. Context & Motivation

To avoid architectural vendor lock-in, UrbanFlow AI evaluates contemporary traffic simulation engines against our requirements:
1. Compatibility with existing SUMO models from the `marl-traffic` research.
2. Fast simulation rollout for real-time reinforcement learning and scenario forecasting.
3. Lightweight execution on integrated graphics and standard CPUs.
4. Clean, modular Python API integration.

---

## 2. Simulator Comparison Matrix

| Evaluation Criteria | SUMO (TraCI / libsumo) | CityFlow | Aimsun / Vissim | UrbanFlow Kinematic Micro-Engine |
|---|---|---|---|---|
| **License & Access** | Open source (EPL 2.0) | Open source (Apache 2.0) | Commercial / Proprietary | Built-in (Native Python/NumPy) |
| **Execution Speed** | Moderate (~1x–5x real-time via TraCI, faster via libsumo) | Ultra-fast (~100x–500x real-time) | Fast (optimized C++) | Ultra-fast (~100x+ real-time) |
| **External Dependencies** | SUMO binaries + XML network toolchain | C++ compilation toolchain (often Linux-centric) | Proprietary Windows installation + dongle/license | **Zero external dependencies** (pure Python/NumPy) |
| **Existing Research Compatibility** | **100% native** (`.net.xml`, `.rou.xml`) | Requires format conversion | Requires proprietary network conversion | Compatible via standard graph format |
| **Multi-Agent RL Suitability** | Good, but TraCI IPC can bottleneck large multi-intersection sync | Excellent, designed specifically for RL | Fair (scripting API exists, heavy) | Excellent (direct vectorized state access) |
| **Traffic Signal Flexibility** | Complete (actuated, NEMA, coordinated, multi-stage) | Simplified (fixed phase switches) | Complete | Complete (NEMA 8-phase and custom stages) |
| **Hardware Overhead** | Moderate RAM / CPU | Low RAM, multi-threaded CPU | High | **Extremely lightweight** (<150 MB RAM) |

---

## 3. Simulator Evaluation Findings

1. **SUMO (Retained as Primary High-Fidelity Simulator):**
   - *Strengths:* Standard in transportation research; matches existing `marl-traffic` models and datasets; full microscopic lane dynamics and detailed car-following behavior.
   - *Weaknesses:* TraCI IPC latency can limit scale; binary dependency on Windows.
   - *Verdict:* Retain full support via `SumoAdapter`.

2. **CityFlow (Future High-Scale Exploration):**
   - *Strengths:* Extremely fast multi-agent simulation for reinforcement learning on large grids.
   - *Weaknesses:* Windows compilation and wheels are historically fragile; lacks European/US complex actuated signal phasing.
   - *Verdict:* Support as an optional accelerated backend adapter once CityFlow Windows builds stabilize.

3. **UrbanFlow Native Kinematic Engine (Adopted as Built-in Core Engine):**
   - *Strengths:* Zero external install required; starts immediately; deterministic; runs anywhere; built-in Intelligent Driver Model (IDM) and MOBIL lane changing; ideal for instant browser demos and rapid digital twin forecasting.
   - *Verdict:* Implemented as default backend engine for zero-friction startup, with seamless hot-swapping to SUMO via `SimulatorAdapter`.

---

## 4. Dual-Engine Architecture Recommendation

UrbanFlow AI adopts a **hybrid dual-engine architecture**:
- **Default Engine:** UrbanFlow Native Kinematic Simulator for instant zero-dependency execution, live browser digital twin demonstration, and rapid look-ahead congestion forecasting.
- **High-Fidelity Engine:** SUMO Adapter connecting to TraCI / libsumo when executing high-fidelity runs or benchmarking against the `marl-traffic` research baseline.
