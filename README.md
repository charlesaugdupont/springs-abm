# SPRINGS-ABM: A Spatial Agent-Based Model of Diarrheal Disease Transmission

*SPRINGS-ABM was developed as part of the SPRINGS project (Horizon Europe, grant No. 101057764). It models diarrheal disease transmission in Akuse, Ghana, to evaluate the health and economic impact of WASH, vaccination, and infrastructure interventions.*

## 1. Overview

SPRINGS-ABM is a high-performance, agent-based model designed to simulate the spread of diarrheal pathogens within a realistic, spatially explicit community. The model captures the dynamics of household structures, agent movement, environmental factors, and behavioral economics to provide a detailed view of disease transmission.

The simulation is set in a geographical context inspired by Akuse, Ghana, using real-world data from OpenStreetMap to generate the environment. The model is built with a modular architecture, making it extensible and maintainable for future research.


## 2. Core Features

*   **Agent Demographics:** Agents are situated within households, with distinct roles as adults or children, influencing their behaviors and susceptibility.
*   **Spatial Environment:** The simulation unfolds on a grid generated from OpenStreetMap (OSM) data, featuring key points of interest such as residences, schools, places of worship, and water sources.
*   **Dual Pathogen Dynamics:** The model simulates two distinct pathogens with different transmission pathways:
    *   **Rotavirus:** A viral pathogen spreading through human-to-human contact and contaminated water sources. The model includes vaccination dynamics.
    *   **Campylobacter:** A bacterial pathogen with a zoonotic transmission pathway, where infection risk is linked to an environmental animal density layer.
*   **Behavioral Economics:** Parent agents make care-seeking decisions for their sick children based on principles of Cumulative Prospect Theory (CPT). Each agent is assigned a behavioral "persona" with unique parameters for risk and loss aversion.
*   **Modular & High-Performance:**
    *   Written in Python, using **PyTorch** for efficient tensor computations, enabling potential GPU acceleration.
    *   A clean, object-oriented design separates concerns into distinct `systems` (e.g., movement, illness), `pathogens`, and `factories`.
    *   Configuration is managed via **Pydantic**, ensuring type safety and clear parameter definition.
    *   Simulation output is handled by **Zarr** and **XArray**, allowing for efficient storage and analysis of large-scale, multi-dimensional data.

## 3. Project Structure
The project is organized into several key directories. Note there are two parallel workflows: the original single-run CLI (`main.py` + `abm/simulation_analysis/`) described in §5 below, and a newer, actively-developed multi-replicate sweep/calibration framework (`experiments/`) described in §6 — most of the project's calibration and published results are produced via the latter.

```
├── README.md
├── requirements.txt
├── inspect_grid.py         # Debug/visualization tool for inspecting spatial grid layers
├── run_viz.py              # Quick single-run visualization (spatial + epidemic curves)
├── sensitivity.py          # One-at-a-time parameter sensitivity sweeps (bridges main.py and experiments/)
├── main.py                 # Entry point for the original single-run simulation workflow (see §5)
├── config.py               # Pydantic-based configuration for all model parameters
├── pyproject.toml          # Project dependencies and metadata
├── abm/                    # Core source code for the agent-based model
│ ├── agent/                # Agent-level logic: CPT/utility math, illness severity & duration formulas
│ ├── data/                 # Real-world input data (e.g. Akuse water-sampling points)
│ ├── environment/          # Spatial grid creation, generation, and OpenStreetMap integration
│ ├── factories/            # Builders that populate agents and the environment at model init
│ ├── model/                # The main model class, day/night step loop, and data collection
│ ├── pathogens/            # Pathogen-specific transmission and progression (Rotavirus, Campylobacter)
│ ├── simulation_analysis/  # Plotting/output helpers for the legacy main.py workflow (see §5)
│ ├── systems/              # Core per-day processes: movement, illness, care-seeking, environment, economics
│ └── utils/                # Utilities (RNG seeding)
├── experiments/            # Actively-developed sweep/calibration framework (see §6)
│ ├── calibration/          # Epidemic transmission-parameter calibration (LHS search)
│ ├── care_seeking/         # Care-seeking/economics OAT, interaction, and DHS-calibration sweeps
│ ├── shocks/               # Ecological water-contamination disturbance sweeps
│ ├── vaccination/          # Vaccination rate/efficacy sweep
│ ├── metrics.py            # Reusable per-run and post-hoc metrics shared across all sweeps
│ └── orchestrator.py       # Shared parallel sweep-execution engine
├── notebooks/              # Exploratory data analysis (e.g. Ghana animal-ownership estimation)
├── grids/                  # Cached generated spatial grids (one per unique grid_id)
└── cache/                  # OpenStreetMap query cache (raw API responses)
```

## 4. Setup and Installation

The model requires Python >= 3.13.

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/charlesaugdupont/springs-abm.git
    cd charlesaugdupont-springs-abm
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

## 7. Configuration

All model parameters are defined in `config.py` using Pydantic models. To alter the model's behavior (e.g., pathogen infectiousness, number of agents, agent persona ranges), you can modify the values in this file before running a simulation.

## Contact
Charles Dupont - c.a.dupont@uva.nl

Computational Science Lab, University of Amsterdam