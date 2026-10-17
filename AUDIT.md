# Source and evidence audit

## Coverage

The submitted ZIP includes benchmark code, train / simulation / serial / hardware scripts, four model weights, three flight videos and PDF reports. Python files parse successfully. Cache-only entries were excluded from the clean project because they are not reproducible source.

- 24 map definitions; 23 represented in the result JSON.
- Missing evaluated map: `5x5_snake_corridor`.
- 15 registered algorithms with actual source and archived result rows.
- 3,450 unique experiment records = 15 × 23 × 2 × 5.
- Seeds: 42, 123, 456, 789, 101112.
- Prioritized Sweeping has a compiled cache entry in the submitted ZIP, but no `.py` implementation. It is not counted as a 16th runnable algorithm or reconstructed from bytecode.

## Archived success rate is a selected maximum

`original/benchmark/experiments/experiment_runner.py::_find_best_performance` selects the highest evaluation success rate in training history and compares it with final evaluation for learning algorithms. The summary takes that selected result. Planning algorithms instead use final evaluation. Checkpoint evaluation counts differ (including small deep-learning evaluations), and final evaluation uses a different count. Taking a maximum across repeated, unequally sized tests can inflate apparent performance and prevents a fair final-test ranking.

The web table reports the mean and **sample standard deviation across the five archived records** for each algorithm/map/mode. This is not a confidence interval, a variance estimate across all episodes, or proof of statistical superiority. Mean steps and time also come from historical records. No 3,450-run experiment was rerun for this showcase.

## Planning transition model

The original Value Iteration implementation builds deterministic transitions even when the environment is slippery, and it does not properly remove continuation value at terminal holes / goals. Its archived behavior is therefore not a valid general comparison of correctly implemented stochastic planning against model-free methods. The browser solver independently implements equal-probability perpendicular slips and terminal-aware Bellman updates; its results must not be substituted into the original archive.

## Seed handling

The original runner sets NumPy's seed. It does not fully seed the environment action-space RNG or all deep-learning RNG / backend sources. Thus a numeric seed in the archived table is not a guarantee of identical reruns. The small browser tabular engine uses its own seeded PRNG for exploration and environment slips; its evaluation uses seed + 1000 and no learning. It has a 100-step cap per episode and γ = 0.99. Because of discounting, Value Iteration maximizes expected discounted reward, not necessarily the undiscounted 100-step success rate.

## Checkpoint identity and environment

The supplied UAV `train.py` uses target-network maximum Q for its TD target, rather than online-network selection followed by target-network evaluation. Consequently the `ddqn_s.pth` filename alone cannot establish Double DQN training. It is presented as a saved UAV checkpoint.

Its network is 36 → 512 → 256 → 128 → 4 (ReLU hidden layers). The small deployment environment has a 6×6 full map with boundary walls and a 4×4 inner navigation region. Obstacles block movement; they do not terminate the episode like FrozenLake holes. The finite-state export uses the original full-map encoding with agent=2 and goal=3. In browser slipped mode, the same fixed policy is stress-tested under new stochastic transitions; no claim is made that this is an exact replay of the original flight controller or stochastic deployment script.

## What was checked

- Parsed original Python source without invoking hardware scripts.
- Validated record uniqueness, cardinalities and defined/evaluated map coverage.
- Ran the supplied Q-learning code on deterministic `4x4_default`, up to 1,000 training episodes. The original early-stop logic stopped at 200 in this one run; final evaluation reported 100% goal success and 6 mean steps. PyTorch and hardware were not involved.
- Exported the small checkpoint with a restricted tensor-only decoder and NumPy inference; checked action conversion and deterministic goal-reaching replay.
- Ran independent Node engine tests against known analytic transition and value cases, as well as seeded learning and artifact invariants.

No full deep-learning retraining, serial integration test, new flight, or automated visual browser QA is claimed. The recordings are the team's submitted evidence.
