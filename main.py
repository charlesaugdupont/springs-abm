# main.py

import argparse
import os
import time

from abm.policy_computation.sweep_runner import compute_all_policies_for_sweep
from abm.simulation_analysis.intervention_sweep import run_simulation_sweep
from abm.simulation_analysis.plots import (
    plot_heatmap,
    plot_epidemic_curves,
    plot_final_state_violins,
    plot_final_state_scatter
)
from abm.simulation_analysis.experiment_config import (
    get_results_path,
    get_policy_set_id,
    get_policy_set_path,
    COST_SUBSIDY_FACTORS,
    EFFICACY_MULTIPLIERS,
    INFECTION_RISK_LEVELS
)
from abm.environment.grid_generator import create_and_save_realistic_grid
from config import SVEIRCONFIG

def main():
    """
    Provides a command-line interface to run the different stages of the
    SVEIR model experimental workflow.
    """
    parser = argparse.ArgumentParser(
        description="Run stages of the SVEIR model experiment.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'stage',
        choices=[
            'create-grid',
            'precompute',
            'simulate',
            'plot-heatmap',
            'plot-curves',
            'plot-violins',
            'plot-scatter'
        ],
        help=(
            "The stage of the experiment to run:\n"
            "  'create-grid'  - Generate the realistic base grid from real-world data (run once).\n"
            "  'precompute'   - Generate all policy files.\n"
            "  'simulate'     - Run simulations and save the results.\n"
            "  'plot-heatmap' - Generate the summary heatmap from saved results.\n"
            "  'plot-curves'  - Compare incidence curves.\n"
            "  'plot-violins' - Compare final agent health and wealth levels.\n"
            "  'plot-scatter' - Generate density scatter plots of final agent states.\n"
        )
    )

    parser.add_argument('-n', '--agents', type=int, default=250, help="Number of agents in the simulation.")
    parser.add_argument('-r', '--repetitions', type=int, default=5, help="Repetitions for each scenario.")
    parser.add_argument('-c', '--cores', type=int, default=6, help="Number of CPU cores for parallel processing.")
    parser.add_argument('-s', '--steps', type=int, default=SVEIRCONFIG.step_target, help=f"Simulation steps (default: {SVEIRCONFIG.step_target}).")
    parser.add_argument('-g', '--grid-id', type=str, help="REQUIRED for 'simulate': The unique ID of the grid to use.")
    parser.add_argument('-p', '--policy-set-id', type=str, help="REQUIRED for 'simulate': The unique ID of the policy set to use for the simulation.")
    parser.add_argument('-e', '--experiment-name', type=str, help="REQUIRED for 'plot-heatmap': The unique name of the experiment run to plot.")

    args = parser.parse_args()

    # --- STAGE 1: CREATE GRID ---
    if args.stage == 'create-grid':
        create_and_save_realistic_grid()

    # --- STAGE 2: PRECOMPUTE POLICIES ---
    elif args.stage == 'precompute':
        print("--- Stage: Pre-computing Policies ---")
        # Calculate the ID that WILL be generated, so we can print it for the user.
        base_config = SVEIRCONFIG
        policy_set_id = get_policy_set_id(
            cost_factors=COST_SUBSIDY_FACTORS,
            efficacy_multipliers=EFFICACY_MULTIPLIERS,
            risk_levels=INFECTION_RISK_LEVELS,
            base_config=base_config
        )
        print(f"Based on current configs, the Policy Set ID will be: {policy_set_id}")
        print("Starting computation...")
        
        compute_all_policies_for_sweep()
        
        print("\n--- Pre-computation Complete ---")
        print(f"Policy Set ID: {policy_set_id}")
        print(f"Saved in directory: {get_policy_set_path(policy_set_id)}")
        print("\nUse this ID for the 'simulate' stage:")
        print(f"  uv run main.py simulate --policy-set-id {policy_set_id}")


    # --- STAGE 3: RUN SIMULATION ---
    elif args.stage == 'simulate':
        # Check for the required argument
        if not args.policy_set_id or not args.grid_id:
            parser.error("The 'simulate' stage requires BOTH --grid-id AND --policy-set-id arguments.")

        # Verify that the chosen policy set actually exists
        policy_path = get_policy_set_path(args.policy_set_id)
        if not os.path.exists(policy_path):
            parser.error(f"The specified policy set '{args.policy_set_id}' was not found in '{policy_path}'. Please run 'precompute' first.")

        # Verify that the chosen grid actually exists
        grid_path = os.path.join("grids", args.grid_id)
        if not os.path.exists(grid_path):
            parser.error(f"The specified grid '{args.grid_id}' was not found in '{grid_path}'. Please run 'create-grid' first.")
                    
        # Make the experiment name even more descriptive
        timestr = time.strftime("%Y%m%d_%H%M%S")
        experiment_name = f"run_{timestr}_grid_{args.grid_id}_policies_{args.policy_set_id}"

        print(f"--- Stage: Running Simulation Sweep ---")
        print(f" Experiment Name:  {experiment_name}")
        print(f" Using Grid:       {args.grid_id}")
        print(f" Using Policy Set: {args.policy_set_id}")
        print(f" Parameters:       {args.agents} agents, {args.steps} steps, {args.repetitions} reps, {args.cores} cores.\n")

        run_simulation_sweep(
            number_agents=args.agents,
            repetitions=args.repetitions,
            num_cores=args.cores,
            steps=args.steps,
            experiment_name=experiment_name,
            grid_id=args.grid_id,
            policy_set_id=args.policy_set_id
        )
        print("\nTo generate the heatmap for this run, use the following command:")
        print(f"  uv run main.py plot-heatmap --experiment-name {experiment_name} --agents {args.agents} --repetitions {args.repetitions}")

    # --- (Optional) PlOTTING STAGES ---
    elif args.stage == 'plot-heatmap':
        if not args.experiment_name:
            parser.error("The 'plot-heatmap' stage requires the --experiment-name argument.")    
        results_grid_path = get_results_path(args.experiment_name, args.agents, args.repetitions)
        if not os.path.exists(results_grid_path):
            parser.error(f"Results grid file not found at '{results_grid_path}'. Make sure the experiment name and parameters are correct.")
        plot_heatmap(results_grid_path)

    elif args.stage == "plot-curves":
        if not args.experiment_name:
            parser.error("The 'plot-curves' stage requires the --experiment-name argument.")
        plot_epidemic_curves(args.experiment_name, args.agents, args.repetitions)

    elif args.stage == "plot-violins":
        if not args.experiment_name:
            parser.error("The 'plot-violins' stage requires the --experiment-name argument.")
        plot_final_state_violins(args.experiment_name, args.agents, args.repetitions)

    elif args.stage == 'plot-scatter':
        if not args.experiment_name:
            print("The 'plot-scatter' stage requires the --experiment-name argument.")    
        plot_final_state_scatter(args.experiment_name, args.agents, args.repetitions)

if __name__ == "__main__":
    main()