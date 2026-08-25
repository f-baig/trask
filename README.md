# Trask/RaceLab

For a more in depth write up of this experiment please visit: https://f-baig.github.io/trask-writeup/.

Trask/RaceLab is a research harness for building playable racing circuits and evaluating one camera-grounded racing player. It provides a browser UI for creating 2D or 3D environments, describing controlled comparisons in an experiment chat, inspecting replay trajectories, and branching a replay from a chosen point.

The normal player is always the predictive-skills agent:

- **2D:** forward-cone camera image + scalar speed
- **3D:** first-person camera image + scalar speed

It does not receive a map, centerline, checkpoint state, route progress, or simulator telemetry. The harness supplies reusable controller skills and runs their model calls alongside tick-rate control.

This project was coauthored with Codex, even though it isn't in the contributor list.

## Setup

### Prerequisites

Install these system packages once: **Python 3.12+**, **Node.js 20+** (which
includes `npm`), and `curl`. No Python or JavaScript package needs to be
installed globally: the launcher creates `.venv` and installs the harness's
FastAPI/Pydantic/Uvicorn backend, native renderer (`pygame-ce`, NumPy), test and
plotting tools, plus the React/Vite frontend dependencies locally.

From the repository root, use at most these two commands:

```bash
./racelab configure  # optional: choose OpenAI or Anthropic and enter an API key
./racelab            # installs dependencies if needed, starts API + UI, and opens the browser
```

The launcher opens [http://127.0.0.1:5173](http://127.0.0.1:5173). Press `Ctrl+C` once to stop both services.

An API key is required for model-backed environment creation and player runs. The offline compiler, manual game, deterministic circuit verification, and reference tooling work without one. `configure` writes the selected key to a local `.env` file with restricted permissions; it is not committed.

Useful maintenance commands:

```bash
./racelab test       # backend tests plus frontend type/build checks
./racelab doctor     # check prerequisites and service health
./racelab clean      # reset local runs/environments/experiments, installs, and caches; preserves .env
```

For a first-time setup without launching the browser, run `./racelab setup`.

## UI quick guide

For a video guide on how to use the platform, please click this [link](https://drive.google.com/file/d/1O0Z3zvPD0jQaz7snuznjkROrxUF8uqCV/view?usp=sharing). Otherwise the below description should be fairly sufficient.

| Tab | Use it for |
| --- | --- |
| **Coordinator** | Describe and create a 2D or 3D circuit. It compiles and certifies the environment, but never launches a player run. You can place the start/finish line and player grid slot in the brief (for example: “start/finish in the top right; player starts P3”). |
| **Draw** | Sketch a closed centerline, save it, then send `use /drawing-id` to Coordinator. |
| **Environments** | Check the compiled layout: top-down in 2D, orbitable preview in 3D. |
| **Experiments** | Describe a comparison in natural language. The fixed predictive-skills player runs the requested conditions, seeds, and pace settings. |
| **Runs** | Review outcome, ticks, calls, tokens, controllers, and trajectory. Click a path point to fork a continuation or correction. |

### Experiment variables

Each experiment keeps the predictive-skills player fixed and can vary these rollout inputs:

| Variable | Available values |
| --- | --- |
| **Track condition** | Normal conditions, control delay, shifted obstacles, low grip, worn tires, heavy car, rear weight bias, high drag, or high downforce. |
| **Seed** | One or more deterministic circuit variants compiled from the same brief. |
| **Player aggression** | A value from 0 (cautious) to 1 (maximum attack), controlling how assertively the player carries pace. |
| **Tick budget** | The maximum simulation ticks allowed before a rollout times out. |

For example: “Compare seeds 11, 29, and 47 under low grip and high drag at 0.6 and 0.9 aggression, with a 1,200-tick budget.”

## Repository map

| Location | Purpose |
| --- | --- |
| [`frontend/`](frontend/) | React/Vite research UI. |
| [`backend/harness/`](backend/harness/) | Compiler, engine, player policies, experiment service, API, and persistence. |
| [`backend/harness/context/`](backend/harness/context/README.md) | Selectively retrieved structured context for the environment and player harnesses. |
| [`docs/generation-harness.md`](docs/generation-harness.md) | Environment-generation harness and evaluation notes. |
| [`docs/reflex-harness.md`](docs/reflex-harness.md) | Player/controller harness design. |
| [`docs/racing-3d.md`](docs/racing-3d.md) | 3D renderer and elevation model. |
| [`scripts/`](scripts/) | Reproducible evaluation and diagnostic scripts. |
| `.harness-data/` | Local generated environments, runs, experiments, chats, and artifacts (created at runtime; not committed). |

## Development and diagnostics

The UI is the intended researcher interface. These commands are useful when working on the engine itself:

```bash
./racelab play2d
./racelab play3d
./racelab engine-check
```

For separate API/UI processes, use `make api` and `make ui` after `./racelab setup`. The API is served on port 8000 and Vite proxies it for the browser UI.
