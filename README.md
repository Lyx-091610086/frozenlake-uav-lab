# FrozenLake Lab — From Policy to Flight

Interactive project showcase for a Sun Yat-sen University robotics course team: **朱嘉语、林妍汐、毛星懿、刘仲晗、刘礼豪**. Portfolio presentation by **Yanxi Lin / 林妍汐**.

**Demo:** https://frozenlake-robotics-lab.free-harp-2721.chatgpt.site

**Repository:** https://github.com/Lyx-091610086/frozenlake-uav-lab (private at creation).

## What is here

- A bilingual English / Chinese browser playground: Value Iteration, seeded Q-Learning and SARSA, 24 editable map definitions, deterministic / slippery dynamics, manual controls, policy / value inspection, and 100-episode independent evaluation.
- A finite-state inference export from the supplied small UAV checkpoint (`ddqn_s.pth`), on its original fixed **blocking-wall** map.
- An interactive explorer for **3,450 archived records**: **15 algorithms × 23 evaluated maps × 2 modes × 5 seeds**. There are 24 map definitions; `5x5_snake_corridor` has no archived records.
- Three original UAV flight recordings, the submitted report, original Python source, saved model weights, and experiment summaries.

This is a **team project**, not a claim of sole authorship. The interactive browser implementation was added for presentation. Original research artifacts are kept separately and have not been silently repaired or relabeled as new experiments.

## Quick start

```sh
python -m http.server 8000 --directory docs
# Open http://localhost:8000
node --test tests/engine.test.cjs
```

No Python ML dependencies or build step are needed for the website. The browser performs actual local tabular learning and simulation; no external API, pretrained service, or hardware is required.

See [GETTING_STARTED.md](GETTING_STARTED.md) for Python reproduction, [AUDIT.md](AUDIT.md) for important scientific limitations, and [PROVENANCE.md](PROVENANCE.md) for the supplied materials and checksums.

## Layout

| Path | Contents |
|---|---|
| `docs/` | Buildless static website, browser engine, UI, exported data, video and report assets |
| `original/benchmark/` | Original multi-algorithm framework and archived experiment summary |
| `original/uav/` | Original training, simulation, serial interface and hardware scripts; four `.pth` weights |
| `scripts/` | Explicit reproduction entry point and data / checkpoint export utilities |
| `tests/` | Meaningful checks for terminal states, slip dynamics, planning, training and supplied artifacts |

## Reading the evidence

Archived learning-algorithm success rates are **the best of checkpoint evaluations and final evaluation**, rather than a uniform final test. Evaluation sample counts differ; original planning models also contain transition issues. Use the archive as historical exploratory evidence, not a controlled leaderboard. Browser evaluation is a separate new calculation and never modifies that archive.

The checkpoint filename contains `ddqn`, but the supplied training script uses a **vanilla DQN target**. We therefore label this export “Saved UAV checkpoint”, rather than assert it proves Double DQN. The fixed UAV map uses blocking obstacles, whereas benchmark FrozenLake holes terminate an episode.

## Distribution and hardware

The archive supplied for this task contains no explicit open-source license. No new license is asserted over teammates' work or third-party dependencies. Attribution is retained. The hardware scripts are provided for inspection and are **not started by the website or reproduction wrapper**. Recorded flight videos are demonstrations; no new hardware experiment or flight-success estimate is claimed.
