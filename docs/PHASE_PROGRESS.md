# UrbanFlow MARL phase progress

## Phase 0

- Baseline commit: `7eb9a7d`. Windows Python 3.11 full suite: 226 passed in 41.53 s. Campaign contracts separately: 8 passed in 2.46 s.
- Updated dependency files, launchers, architecture and port references. Removed the obsolete handoff packager and its two handoff documents. Added Git rules for generated caches and virtual environments.
- Fixed the successful-reroute timestamp and passed named features to the fitted spillback calibrators. Regression tests check prediction equivalence and the first successful routing log.
- Fresh Windows virtual environment: 228 passed in 35.59 s with scikit-learn version warnings treated as errors. Starlette reported one test-client deprecation warning. The launcher started and the status endpoint returned online on an isolated port. POSIX restart remains untested.

## Phase order conflict

The profiling instructions require a before-any-changes matrix for networks and an RL controller introduced in Phases 2 and 3. Keep the v11 baseline for configurations it supports and capture the full three-network matrix and golden trajectories immediately before Phase 6 optimizations. Record unsupported baseline cells as unavailable.
