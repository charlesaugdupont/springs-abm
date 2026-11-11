# SPRINGS-ABM: A Spatial Agent-Based Model of Diarrheal Disease Transmission

*SPRINGS-ABM was developed as part of the SPRINGS project (Horizon Europe, grant No. 101057764). It models diarrheal disease transmission in Akuse, Ghana, to evaluate the health and economic impact of WASH, vaccination, and infrastructure interventions.*

## 1. Overview

SPRINGS-ABM is a high-performance, agent-based model designed to simulate the spread of diarrheal pathogens within a realistic, spatially explicit community. The model captures the dynamics of household structures, agent movement, environmental factors, and behavioral economics to provide a detailed view of disease transmission.

The simulation is set in a geographical context inspired by Akuse, Ghana, using real-world data from OpenStreetMap to generate the environment. The model is built with a modular architecture, making it extensible and maintainable for future research.


## 2. Core Features

*   **Agent Demographics:** Agents are situated within households, with distinct roles as adults or children, influencing their behaviors and susceptibility.
*   **Spatial Environment:** The simulation unfolds on a grid generated from OpenStreetMap (OSM) data, featuring key points of interest such as residences, schools, places of worship, and water sources.
*   **Three-Pathogen Dynamics:** The model simulates three distinct pathogens with different transmission pathways:
    *   **Rotavirus:** A viral pathogen spreading through human-to-human contact and contaminated water sources. The model includes vaccination dynamics.
    *   **Campylobacter:** A bacterial pathogen with a zoonotic transmission pathway, where infection risk is linked to an environmental animal density layer.
    *   **Cryptosporidium:** A protozoan parasite with four routes (household person-to-person, community co-location, waterborne via a persistent oocyst reservoir, and a small ruminant-only zoonotic route). Two features set it apart mechanically: most childhood infections are **subclinical** yet still shed and transmit, and its environmental stage **accumulates and decays over weeks** rather than flipping between clean and contaminated.
*   **Behavioral Economics:** Parent agents make care-seeking decisions for their sick children based on principles of Cumulative Prospect Theory (CPT). Each agent is assigned a behavioral "persona" with unique parameters for risk and loss aversion.
*   **Modular & High-Performance:**
    *   Written in Python, using **PyTorch** for efficient tensor computations, enabling potential GPU acceleration.
    *   A clean, object-oriented design separates concerns into distinct `systems` (e.g., movement, illness), `pathogens`, and `factories`.
    *   Configuration is managed via **Pydantic**, ensuring type safety and clear parameter definition.
    *   Simulation output is handled by **Zarr** and **XArray**, allowing for efficient storage and analysis of large-scale, multi-dimensional data.

## 3. Project Structure
The project is organized into several key directories. Note there are two parallel workflows: the original single-run CLI (`main.py` + `abm/simulation_analysis/`) described in §5 below, and a newer, actively-developed multi-replicate sweep/calibration framework (`experiments/`) described in §6 — most of the project's calibration and published results are produced via the latter. A browser-based web app for running single scenarios (`webapp/` + `frontend/`) is described in §7.

```
├── README.md
├── requirements.txt        # Full dependencies: model, grid generation, experiments, plotting
├── requirements-web.txt    # Narrower dependency set for the deployed web app (see §7)
├── pytest.ini              # Test configuration (see §8)
├── Dockerfile              # Web app image: builds frontend/ and serves it from webapp/
├── cloudbuild.yaml         # Cloud Build pipeline: deploys the web app to Cloud Run on push to main
├── config.py               # Pydantic-based configuration for all model parameters
├── main.py                 # Entry point for the original single-run simulation workflow (see §5)
├── run_viz.py              # Quick single-run visualization (spatial + epidemic curves)
├── sensitivity.py          # One-at-a-time parameter sensitivity sweeps (bridges main.py and experiments/)
├── inspect_grid.py         # Debug/visualization tool for inspecting spatial grid layers
├── model-reference.html    # Standalone model reference (open in a browser)
├── abm/                    # Core source code for the agent-based model
│ ├── agent/                # Agent-level logic: CPT/utility math, illness severity & duration formulas
│ ├── data/                 # Real-world input data (e.g. Akuse water-sampling points)
│ ├── economy/              # Household economy in Ghana cedis (used when EconomyConfig.model == "cedi")
│ ├── environment/          # Spatial grid creation, generation, and OpenStreetMap integration
│ ├── evidence/             # Transcribed published estimates (MAL-ED, rotavirus cohorts) used for parameters and targets
│ ├── factories/            # Builders that populate agents and the environment at model init
│ ├── model/                # The main model class, day/night step loop, and data collection
│ ├── pathogens/            # Pathogen-specific transmission and progression (Rotavirus, Campylobacter, Cryptosporidium) + the name->class registry
│ ├── simulation_analysis/  # Plotting/output helpers for the legacy main.py workflow (see §5)
│ ├── systems/              # Core per-day processes: movement, illness, care-seeking, demography, household, environment, economics
│ └── utils/                # Utilities (RNG seeding)
├── experiments/            # Actively-developed sweep/calibration framework (see §6)
│ ├── calibration/          # Calibration against MAL-ED and DHS targets: LHS search, joint history matching, out-of-sample validation
│ ├── care_seeking/         # Care-seeking/economics OAT, interaction, and DHS-calibration sweeps
│ ├── gsa/                  # Global sensitivity analysis (Morris screening, then Sobol indices)
│ ├── perf/                 # Timing benchmarks and the regression_check.py fingerprint harness (see §8)
│ ├── shocks/               # Ecological water-contamination disturbance sweeps
│ ├── vaccination/          # Vaccination rate/efficacy sweep
│ ├── metrics.py            # Reusable per-run and post-hoc metrics shared across all sweeps
│ ├── observation.py        # MAL-ED observation layer: what surveillance would have detected in a run
│ ├── presets.py            # Named override sets for recommended non-default settings
│ ├── legacy.py             # Override sets that reproduce earlier model generations
│ └── orchestrator.py       # Shared parallel sweep-execution engine
├── webapp/                 # FastAPI backend for the web app: JSON API under /api, serves the built frontend (see §7)
├── frontend/               # React + TypeScript + Vite single-page app for the web app (see §7)
├── scripts/                # Offline tooling: settlement grid builder, GHS-POP population layer, basemap baker, DHS tabulations
├── model_tests/            # Pytest suite for the model core (pathogens, illness mechanics, demography, economy, vaccination, water network)
├── notebooks/              # Exploratory data analysis (e.g. Ghana animal-ownership estimation)
├── grids/                  # Generated spatial grids, one directory per grid_id (the web app uses 7d9ce7c720a6)
├── cache/                  # OpenStreetMap query cache (generated, not committed)
└── outputs/                # Simulation outputs (generated, not committed)
```

## 4. Setup and Installation

The model requires Python >= 3.13.

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/charlesaugdupont/springs-abm.git
    cd springs-abm
    ```

2.  **Create and Activate a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    This covers the model, grid generation, and the experiments. The web app needs `requirements-web.txt` on top (see §7).

### External data

Running the model and the experiments doesn't need any of the raw data below: what the model takes from it is already in the repository, as parameters in `config.py` and as the grids in `grids/`. The raw files are needed only to rebuild those inputs, and none of them is committed:

| Data | Source | Used by |
|---|---|---|
| Ghana DHS 2014 and 2022 recode files: household (HR), person (PR), children (KR), women (IR) | [The DHS Program](https://dhsprogram.com); registration required, and the files may not be redistributed | `scripts/dhs_tables.py` |
| `ghana_data.csv`: the SPRINGS compiled child-level survey file (cluster coordinates, household WASH, livestock ownership) | SPRINGS project | `scripts/dhs_tables.py`, `notebooks/ghana_data.ipynb` (reads `abm/data/ghana_data.csv`) |
| GHS-POP R2023A population tiles (100 m; epochs 2020, 2025, 2030) | JRC Global Human Settlement Layer (openly licensed) | `scripts/add_population_layer.py` |

Keep these outside the repository, or in a `data/` folder at its root (ignored by both git and Docker), and pass their location on the command line. With the files laid out as the scripts expect:

```bash
python -m scripts.dhs_tables --data-dir "path/to/SPRINGS DATA"          # writes outputs/dhs/dhs_tables.txt
python -m scripts.add_population_layer --grid-id <GRID_ID> \
    --tiles "path/to/SPRINGS DATA/GHS POP/2025/*.zip"
```

`scripts/dhs_tables.py` expects the unzipped recode folders under `GHS 2014/` and `GHS 2022/` (e.g. `GHS 2022/GHKR8CDT/GHKR8CFL.DTA`; see `FILES` in the script), with `ghana_data.csv` beside them. OpenStreetMap data for grid creation is downloaded automatically and cached in `cache/`.

## 5. Simulation Workflow (single-run, via `main.py`)

This is the original workflow for running and inspecting one simulation at a time. For multi-replicate parameter sweeps and calibration (the primary way this project now generates results), see §6 instead.

Running an experiment is a three-step process managed via `main.py`.

### Step 1: Create the Spatial Grid

First, generate the spatial environment for the simulation. This command fetches data from OpenStreetMap and creates the multi-layered grid file. **This step only needs to be run once.**

```bash
python main.py create-grid
```

After running, a unique Grid ID will be printed to the console. You will need this ID for the next step. The grid files are saved in the grids/ directory.


### Step 2: Run the Simulation
Next, run the agent-based simulation on the generated grid.

```bash
python main.py simulate --grid-id <GRID_ID_FROM_STEP_1>
```

- Replace <GRID_ID_FROM_STEP_1> with the actual ID from the previous step.

This will run a single, complete simulation. A unique Experiment Name (e.g., run_20260121_170000_grid_...) will be printed to the console. You will need this name to plot the results.

You can override default simulation parameters using command-line arguments:

- `--agents <N>`: Set the number of agents (e.g., `--agents 5000`).
- `--steps <N>`: Set the number of simulation steps (e.g., `--steps 200`).

### Step 3: Plot the Results

Finally, use the experiment name from Step 2 to generate relevant figures.


- **Plot Epidemic Curves (Prevalence vs. Time)**
    ```bash
    python main.py plot-curves --experiment-name <EXPERIMENT_NAME_FROM_STEP_2>
    ```
- **Plot Care-Seeking Behavior**
    ```bash
    python main.py plot-care --experiment-name <EXPERIMENT_NAME_FROM_STEP_2>
    ```
- **Plot Child Illness Duration and Severity**
    ```bash
    python main.py plot-child-illness --experiment-name <EXPERIMENT_NAME_FROM_STEP_2>
    ```

Plots are displayed on-screen and saved to the `outputs/<experiment_name>/simulation_results/` directory.

## 6. Sweep & Calibration Experiments (`experiments/`)

For parameter sweeps, calibration searches, and multi-replicate scenario comparisons, use the `experiments/` package rather than `main.py`. It's built around a shared engine (`experiments/orchestrator.py`) that handles parallel replicate execution, seeding, and tidy Parquet output, so each individual experiment script only needs to declare *what* to sweep and *what to measure*.

Each experiment lives in its own subpackage (`experiments/calibration/`, `experiments/care_seeking/`, `experiments/shocks/`, `experiments/vaccination/`) and is run as a module, e.g.:

```bash
python -m experiments.vaccination.run_vaccination_sweep --grid-id <GRID_ID> --pilot     # fast smoke test first
python -m experiments.vaccination.run_vaccination_sweep --grid-id <GRID_ID>             # full sweep
python -m experiments.vaccination.run_vaccination_sweep --plot-only                     # replot without rerunning
```

Common flags across these scripts: `--pilot` (small grid/reps/steps for a quick sanity check before committing to a full run), `--workers N` (parallel processes), `--plot-only`. Results and figures are written to `experiments/outputs/<experiment_name>/`. See each script's module docstring for its specific design (parameters swept, metrics recorded), and `experiments/orchestrator.py`'s docstring for the shared engine's design intent.

`experiments/shocks/` has two scripts: `run_shock_sweep.py` (a single rectangular duration x magnitude shock window; produces the `shocks_day200` result) and `run_shock_scenarios.py` (GoodBYE-paper-inspired cyclical background stress + punctuated/persistent shock recovery shapes, layered on the same underlying mechanism; produces `shocks_scenarios_named`).

## 7. Web App (`webapp/` + `frontend/`)

A browser interface for configuring and running single simulations: a FastAPI backend (`webapp/`) exposing a JSON API under `/api`, and a React + TypeScript + Vite frontend (`frontend/`) whose charts render client-side. It runs on the `7d9ce7c720a6` grid, executes one simulation at a time, and sits behind a single shared password.

**Run locally** (two terminals, from the repository root):

```bash
pip install -r requirements.txt -r requirements-web.txt
WEBAPP_SHARED_PASSWORD=<choose-a-password> uvicorn webapp.app:app --port 8000
```

```bash
cd frontend
npm ci
npm run dev     # http://localhost:5173 - proxies /api to the backend on :8000
```

Alternatively, `npm run build` writes the frontend into `webapp/frontend_dist/`, after which the backend alone serves the whole app at http://localhost:8000.

**Deployment:** the `Dockerfile` builds the frontend and a slim Python image (CPU-only PyTorch, no geo or plotting stack), and `cloudbuild.yaml` deploys it to Cloud Run on every push to `main`. The service must keep `--max-instances=1` (the SQLite job store and single-worker executor assume one process) and `--no-cpu-throttling` (simulations run in a background thread after the submitting request returns). Outside local dev (`WEBAPP_ENV` other than `dev`), the app refuses to start unless `WEBAPP_SECRET_KEY` and `WEBAPP_SHARED_PASSWORD` are set; on Cloud Run they come from Secret Manager.

## 8. Testing

```bash
pytest          # both suites; see pytest.ini for why --pyargs is required
```

Two packages: `webapp/tests/` (FastAPI layer + the parameter-registry completeness check) and `model_tests/` (the model core). In addition, `experiments/perf/regression_check.py` is a byte-for-byte fingerprint harness used to prove that a refactor leaves behaviour untouched:

```bash
python -m experiments.perf.regression_check                      # vs the saved golden
python -m experiments.perf.regression_check --determinism        # bit-reproducibility
python -m experiments.perf.regression_check --pathogens rota campy --allow-new-keys
```

The last form is how the Cryptosporidium work was validated: it reproduces a golden captured *before* the third pathogen existed, proving the framework refactors left rotavirus and campylobacter bit-identical.

The web app's end-to-end tests (Playwright) live in `frontend/e2e/`. `npm run test:e2e` (from `frontend/`) starts the Vite dev server and the backend itself; the backend is launched from `venv/bin/uvicorn`, so it expects the virtual environment at `venv/`. The `live-*` specs run only against a deployed URL and are skipped unless `SPRINGS_LIVE_URL` and `SPRINGS_LIVE_PASSWORD` are set.

## 9. Configuration

All model parameters are defined in `config.py` using Pydantic models. To alter the model's behavior (e.g., pathogen infectiousness, number of agents, agent persona ranges), you can modify the values in this file before running a simulation.

## Contact
Charles Dupont - c.a.dupont@uva.nl

Computational Science Lab, University of Amsterdam
