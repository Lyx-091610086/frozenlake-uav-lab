# Run the project

## 1. Browser demo (recommended first)

Use Python 3 to serve the static `docs` directory:

```sh
python -m http.server 8000 --directory docs
```

Open http://localhost:8000. Choose a map and a policy method; run the policy, move manually, toggle slip, inspect Q-values, edit holes, or evaluate 100 new episodes. Saved UAV checkpoint mode fixes the map and uses blocking walls. The result explorer is historical data, independent from the live simulation.

Node.js 18+ can run the engine checks without installing any package:

```sh
node --test tests/engine.test.cjs
```

## 2. Original Python benchmark

Use an isolated Python environment. Python 3.10 or 3.11 is recommended for compatibility with the original pinned dependencies. The original framework declares its full dependencies in `original/benchmark/requirements.txt`. For the small tabular reproduction below, the following subset suffices:

```sh
python -m venv .venv
# macOS / Linux
. .venv/bin/activate
# Windows PowerShell alternative: .venv\Scripts\Activate.ps1
python -m pip install gymnasium==0.29.1 numpy==1.24.4 matplotlib==3.7.5 pillow==10.4.0 PyYAML==6.0.2 pandas seaborn
python scripts/run_benchmark.py --algorithm q_learning --map 4x4_default --episodes 1000 --deterministic --seed 42
```

New outputs go under `runs/`, leaving historical results untouched. The wrapper prints `FINAL evaluation` explicitly. A local smoke run reached the goal in 6 mean steps with 100% success in that run's final evaluation; this is one small check, not a rerun of the 3,450-record benchmark or a guarantee of seed reproducibility.

Deep methods require a compatible PyTorch installation in addition to the original requirements. Choose the official PyTorch build appropriate for your own CPU / GPU; the browser demo does not require it. Hardware code additionally depends on the actual serial device, SDKs and equipment used by the team. No hardware commands are issued by these instructions.

## 3. Regenerate the presentation data

```sh
python scripts/export_site_data.py
# Needs NumPy; restricted tensor-only reader for these supplied weights only
python scripts/export_saved_policy.py
```

The checkpoint export evaluates the network at all 16 positions on the supplied small UAV map, then converts action order from up/down/left/right to the browser's left/down/right/up. It does not train a new network. The restricted decoder rejects unrecognized pickle globals and unsupported tensor layouts; it is intentionally not a general PyTorch loader.

## 4. GitHub hosting

The repository is ready to upload as-is. Keep the root README, original source, scripts, tests and `docs` folder. Do not commit `.venv`, `runs`, caches, `.openai`, or credentials. GitHub Pages can serve the static website from **main / docs** if your repository visibility and account plan allow it. A private repository does not automatically make a Pages site private; choose the publication audience explicitly before changing hosting settings.

No npm build, database, backend secrets or API keys are needed.

For the managed host used by this demo, run `python scripts/build_site.py` to copy `docs/` to `dist/`. The generated `dist/` directory is excluded from source control.
