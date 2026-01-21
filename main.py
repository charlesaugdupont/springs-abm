# main.py

import argparse
import os
import time

from abm.simulation_analysis.intervention_sweep import run_simulation_sweep
from abm.simulation_analysis.plots import (
    plot_epidemic_curves,
    plot_final_state_violins,
    plot_final_state_scatter
)
from abm.environment.grid_generator import create_and_save_realistic_grid
from config import SVEIRCONFIG

def main():
    """Provides a command-line interface for the SVEIR model experimental workflow."""
    parser = argparse.ArgumentParser(
        description="Run stages of the SVEIR model experiment.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'stage',
        choices=['create-grid', 'simulate', 'plot-curves', 'plot-violins', 'plot-scatter'],
        help=(
            "The stage of the experiment to run:\n"
            "  'create-grid'  - Generate the realistic base grid (run once).\n"
            "  'simulate'     - Run simulations and save results.\n"
            "  'plot-curves'  - Compare incidence curves.\n"
            "  'plot-violins' - Compare final agent health and wealth.\n"
            "  'plot-scatter' - Generate density scatter plots of final agent states.\n"
        )
    )
    parser.add_argument('-n', '--agents', type=int, help="Number of agents in the simulation.")
    parser.add_argument('-r', '--repetitions', type=int, help="Repetitions for each scenario.")
    parser.add_argument('-c', '--cores', type=int, help="Number of CPU cores for parallel processing.")
    parser.add_argument('-s', '--steps', type=int, help="Number of simulation steps.")
    parser.add_argument('-g', '--grid-id', type=str, help="REQUIRED for 'simulate': The unique ID of the grid to use.")
    parser.add_argument('-e', '--experiment-name', type=str, help="REQUIRED for plotting stages: The name of the experiment run to plot.")
    args = parser.parse_args()

    # --- Load base config and override with CLI args where provided ---
    config = SVEIRCONFIG.model_copy(deep=True)
    if args.agents: config.number_agents = args.agents
    if args.steps: config.step_target = args.steps
    if args.repetitions: config.experiment_params.repetitions = args.repetitions
    if args.cores: config.experiment_params.num_cores = args.cores

    # --- STAGE 1: CREATE GRID ---
    if args.stage == 'create-grid':
        create_and_save_realistic_grid()

    # --- STAGE 3: RUN SIMULATION ---
    elif args.stage == 'simulate':
        if not args.grid_id:
            parser.error("'simulate' stage requires --grid-id.")
        grid_path = os.path.join("grids", args.grid_id)
        if not os.path.exists(grid_path):
            parser.error(f"Grid '{args.grid_id}' not found. Please run 'create-grid' first.")

        timestr = time.strftime("%Y%m%d_%H%M%S")
        experiment_name = f"run_{timestr}_grid_{args.grid_id}"
        print(f"--- Stage: Running Simulation Sweep ---")
        print(f" Experiment Name:  {experiment_name}")
        print(f" Using Grid:       {args.grid_id}")
        print(f" Parameters:       {config.number_agents} agents, {config.step_target} steps, {config.experiment_params.repetitions} reps, {config.experiment_params.num_cores} cores.\n")

        config.spatial_creation_args.grid_id = args.grid_id
        run_simulation_sweep(config, experiment_name)

    # --- PLOTTING STAGES ---
    else:
        if not args.experiment_name:
            parser.error(f"The '{args.stage}' stage requires --experiment-name.")
        
        # For plotting, we need to know the parameters the experiment was run with
        # A more robust solution would be to save the config with the results
        print(f"Plotting results for experiment: {args.experiment_name}")
        if args.stage == "plot-curves":
            plot_epidemic_curves(args.experiment_name, config)
        elif args.stage == "plot-violins":
            plot_final_state_violins(args.experiment_name, config)
        elif args.stage == 'plot-scatter':
            plot_final_state_scatter(args.experiment_name, config)

if __name__ == "__main__":
    main()