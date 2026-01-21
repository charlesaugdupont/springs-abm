# main.py
import argparse
import os
import time
import pickle
import traceback

from abm.simulation_analysis.plots import (
    plot_epidemic_curves,
    plot_final_state_violins,
    plot_final_state_scatter
)
from abm.simulation_analysis.experiment_config import get_full_results_path, get_sim_runs_path
from abm.model.initialize_model import SVEIRModel
from abm.environment.grid_generator import create_and_save_realistic_grid
from config import SVEIRCONFIG

def run_single_simulation(config, experiment_name: str) -> dict:
    """Runs one simulation instance and returns key results."""
    run_name = f"sim_run_{config.seed}"
    sim_runs_path = get_sim_runs_path(experiment_name)

    try:
        model = SVEIRModel(model_identifier=run_name, root_path=sim_runs_path)
        model.set_model_parameters(**config.model_dump())
        model.initialize_model(verbose=False)
        model.run()

        time_series = model.get_time_series_data()
        final_states = model.get_final_agent_states()
        return {
            'run_name': run_name,
            'proportion_infected': model.get_proportion_infected_at_least_once(),
            'prevalence_curve': time_series['prevalence'],
            'final_health': final_states['health'],
            'final_wealth': final_states['wealth'],
        }
    except Exception:
        print(f"\n--- ERROR IN SIMULATION: {run_name} ---")
        traceback.print_exc()
        return {'run_name': run_name, 'proportion_infected': -1.0}

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
            "  'simulate'     - Run a single simulation and save results.\n"
            "  'plot-curves'  - Compare incidence curves.\n"
            "  'plot-violins' - Compare final agent health and wealth.\n"
            "  'plot-scatter' - Generate density scatter plots of final agent states.\n"
        )
    )
    parser.add_argument('-n', '--agents', type=int, help="Number of agents in the simulation.")
    parser.add_argument('-s', '--steps', type=int, help="Number of simulation steps.")
    parser.add_argument('-g', '--grid-id', type=str, help="REQUIRED for 'simulate': The unique ID of the grid to use.")
    parser.add_argument('-e', '--experiment-name', type=str, help="REQUIRED for plotting stages: The name of the experiment run to plot.")
    args = parser.parse_args()

    # --- Load base config and override with CLI args where provided ---
    config = SVEIRCONFIG.model_copy(deep=True)
    if args.agents: config.number_agents = args.agents
    if args.steps: config.step_target = args.steps

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
        print(f"--- Stage: Running Single Simulation ---")
        print(f" Experiment Name:  {experiment_name}")
        print(f" Using Grid:       {args.grid_id}")
        print(f" Parameters:       {config.number_agents} agents, {config.step_target} steps.\n")

        config.spatial_creation_args.grid_id = args.grid_id
        
        # Run the simulation
        result = run_single_simulation(config, experiment_name)
        
        # Save the result in a list to maintain compatibility with plotting functions
        results_list = [result]
        full_results_path = get_full_results_path(experiment_name, config)
        with open(full_results_path, 'wb') as f:
            pickle.dump(results_list, f)
        print(f"\nFull results saved to: {full_results_path}")

    # --- PLOTTING STAGES ---
    else:
        if not args.experiment_name:
            parser.error(f"The '{args.stage}' stage requires --experiment-name.")
        
        print(f"Plotting results for experiment: {args.experiment_name}")
        if args.stage == "plot-curves":
            plot_epidemic_curves(args.experiment_name, config)
        elif args.stage == "plot-violins":
            plot_final_state_violins(args.experiment_name, config)
        elif args.stage == 'plot-scatter':
            plot_final_state_scatter(args.experiment_name, config)

if __name__ == "__main__":
    main()