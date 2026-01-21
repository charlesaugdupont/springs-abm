# SPRINGS-ABM: A Spatial Agent-Based Model of Diarrheal Disease Transmission

## 1. Overview

SPRINGS-ABM is a high-performance, agent-based model designed to simulate the spread of diarrheal pathogens within a realistic, spatially explicit community. The model captures the dynamics of household structures, agent movement, environmental factors, and behavioral economics to provide a detailed view of disease transmission.

The simulation is set in a geographical context inspired by Akuse, Ghana, using real-world data from OpenStreetMap to generate the environment. The model is built with a modular architecture, making it extensible and maintainable for future research.

---

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

---

## 3. Project Stucture
The project is organized into several key directories:

```
└── charlesaugdupont-springs-abm/
├── README.md
├── main.py                 # Main entry point for running the simulation workflow
├── config.py               # Pydantic-based configuration for all model parameters
├── pyproject.toml          # Project dependencies and metadata
├── abm/                    # Core source code for the agent-based model
│ ├── agent/                # Agent-specific logic (illness, CPT)
│ ├── environment/          # Spatial grid creation and generation
│ ├── factories/            # Factories for building agents and the environment
│ ├── model/                # The main model class, stepping logic, and data collection
│ ├── network/              # Social network creation
│ ├── pathogens/            # Logic for pathogen-specific transmission and progression
│ ├── systems/              # Core behavioral and environmental processes (movement, etc.)
│ └── simulation_analysis/  # Plotting and experiment management scripts
├── grids/                  # Output directory for generated spatial grids
└── outputs/                # Output directory for simulation results and plots
```
---

## 4. Setup and Installation

The model requires Python >= 3.13.

1.  **Clone the Repository:**
    ```bash
    git clone <repository_url>
    cd charlesaugdupont-springs-abm
    ```

2.  **Create and Activate a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    The project uses `pyproject.toml` to manage dependencies. Install the project and all required libraries with pip:
    ```bash
    pip install .
    ```

---

## 5. Simulation Workflow

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

- `--agents <N>`: Set the number of agents (e.g., `--agents 500`).
- `--steps <N>`: Set the number of simulation steps (e.g., `--steps 200`).

### Step 3: Plot the Results

Finally, use the experiment name from Step 2 to generate plots from the simulation output.


- **Plot Epidemic Curves (Prevalence vs. Time):**
    ```bash
    python main.py plot-curves --experiment-name <EXPERIMENT_NAME_FROM_STEP_2>
    ```
- **Plot Final Agent State Distributions (Violin Plots):**
    ```bash
    python main.py plot-violins --experiment-name <EXPERIMENT_NAME_FROM_STEP_2>
    ```
- **Plot Final Health vs. Wealth (2D Density Scatter Plot):**
    ```bash
    python main.py plot-scatter --experiment-name <EXPERIMENT_NAME_FROM_STEP_2>
    ```

Plots are displayed on-screen and saved to the `outputs/<experiment_name>/simulation_results/` directory.

---

## 6. Configuration

All model parameters are defined in `config.py` using Pydantic models. To alter the model's behavior (e.g., pathogen infectiousness, number of agents, agent persona ranges), you can modify the values in this file before running a simulation.