# abm/simulation_analysis/plots.py
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
from typing import Dict

from config import SVEIRConfig
from .experiment_config import get_full_results_path

def plot_heatmap(results_grid_file: str, config: SVEIRConfig):
    """Loads a saved summary grid and generates the analysis heatmap."""
    print(f"Generating heatmap from {results_grid_file}...")
    results_grid = np.load(results_grid_file)
    exp_params = config.experiment_params

    plt.figure(figsize=(12, 10))
    # Flip grid for intuitive visualization (high efficacy at top)
    results_flipped = np.flipud(results_grid)
    annot_data = np.char.mod('%.3f', results_flipped)
    annot_data[results_flipped < 0] = 'FAIL'

    ax = sns.heatmap(
        results_flipped, annot=annot_data, fmt="s", cmap="viridis_r",
        xticklabels=[f"{x:.2f}" for x in exp_params.cost_subsidy_factors],
        yticklabels=[f"{y:.2f}" for y in reversed(exp_params.efficacy_multipliers)],
        linewidths=.5, annot_kws={"size": 12}
    )
    ax.set_title("Impact of Interventions on Population Attack Rate", fontsize=16, pad=20)
    ax.set_xlabel("Cost Subsidy Factor (Lower is Cheaper Healthcare)", fontsize=12)
    ax.set_ylabel("Health Efficacy Multiplier (Higher is Better Healthcare)", fontsize=12)

    output_dir = os.path.dirname(results_grid_file)
    base_name = os.path.splitext(os.path.basename(results_grid_file))[0]
    output_filename = os.path.join(output_dir, f"heatmap_{base_name}.png")
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    print(f"Heatmap saved to {output_filename}")
    plt.show()

def _load_and_aggregate_results(experiment_name: str, config: SVEIRConfig) -> Dict | None:
    """Helper to load full results and aggregate them by scenario."""
    full_results_file = get_full_results_path(experiment_name, config)
    if not os.path.exists(full_results_file):
        print(f"Error: Full results file not found at '{full_results_file}'.")
        return None
    with open(full_results_file, 'rb') as f:
        raw_results = pickle.load(f)

    aggregated = {}
    for res in raw_results:
        key = (res['efficacy'], res['subsidy'])
        if key not in aggregated:
            aggregated[key] = {'prevalence': [], 'health': [], 'wealth': []}
        if 'prevalence_curve' in res:
            aggregated[key]['prevalence'].append(res['prevalence_curve'])
        if 'final_health' in res:
            aggregated[key]['health'].extend(res['final_health'])
            aggregated[key]['wealth'].extend(res['final_wealth'])
    return aggregated

def plot_epidemic_curves(experiment_name: str, config: SVEIRConfig):
    """Plots and compares epidemic prevalence curves for key scenarios."""
    print(f"Generating epidemic curves for {experiment_name}...")
    aggregated_results = _load_and_aggregate_results(experiment_name, config)
    if not aggregated_results: return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    exp_params = config.experiment_params

    scenarios = {
        "Baseline (No Intervention)": (1.0, 1.0),
        "Best Intervention": (max(exp_params.efficacy_multipliers), min(exp_params.cost_subsidy_factors)),
    }

    for label, (eff, sub) in scenarios.items():
        closest_key = min(aggregated_results.keys(), key=lambda k: abs(k[0]-eff) + abs(k[1]-sub))
        curves = aggregated_results[closest_key]['prevalence']
        if not curves: continue

        max_len = max(len(c) for c in curves) if curves else 0
        padded = [np.pad(c, (0, max_len - len(c)), 'constant') for c in curves]
        mean_curve, std_curve = np.mean(padded, axis=0), np.std(padded, axis=0)
        timesteps = np.arange(len(mean_curve))

        line, = ax.plot(timesteps, mean_curve, label=f"{label}\n(E:{closest_key[0]:.2f}, S:{closest_key[1]:.2f})", lw=2)
        ax.fill_between(timesteps, mean_curve - std_curve, mean_curve + std_curve, alpha=0.15, color=line.get_color())

    ax.set_title("Impact of Interventions on Epidemic Progression (Prevalence)", fontsize=16, pad=15)
    ax.set_xlabel("Time (Days)", fontsize=12)
    ax.set_ylabel("Number of Currently Infected Agents", fontsize=12)
    ax.legend(title="Scenario", fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_ylim(bottom=0)

    output_dir = os.path.dirname(get_full_results_path(experiment_name, config))
    output_filename = os.path.join(output_dir, f"prevalence_curves_{experiment_name}.png")
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    print(f"Prevalence curves saved to {output_filename}")
    plt.show()

def plot_final_state_violins(experiment_name: str, config: SVEIRConfig):
    """Creates violin plots comparing final agent health and wealth."""
    print(f"Generating violin plots for {experiment_name}...")
    aggregated_results = _load_and_aggregate_results(experiment_name, config)
    if not aggregated_results: return

    exp_params = config.experiment_params
    baseline_key = min(aggregated_results.keys(), key=lambda k: abs(k[0]-1.0) + abs(k[1]-1.0))
    best_key = min(aggregated_results.keys(), key=lambda k: abs(k[0]-max(exp_params.efficacy_multipliers)) + abs(k[1]-min(exp_params.cost_subsidy_factors)))

    plot_data = []
    for health_val in aggregated_results[baseline_key]['health']:
        plot_data.append({'Value': health_val, 'Metric': 'Health', 'Scenario': 'Baseline'})
    for wealth_val in aggregated_results[baseline_key]['wealth']:
        plot_data.append({'Value': wealth_val, 'Metric': 'Wealth', 'Scenario': 'Baseline'})
    for health_val in aggregated_results[best_key]['health']:
        plot_data.append({'Value': health_val, 'Metric': 'Health', 'Scenario': 'Best Intervention'})
    for wealth_val in aggregated_results[best_key]['wealth']:
        plot_data.append({'Value': wealth_val, 'Metric': 'Wealth', 'Scenario': 'Best Intervention'})
    df = pd.DataFrame(plot_data)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    fig.suptitle('Distribution of Final Agent States by Scenario', fontsize=18)

    sns.violinplot(ax=axes[0], data=df[df['Metric'] == 'Health'], x='Scenario', y='Value', hue='Scenario', inner='quartile', legend=False)
    axes[0].set_title('Final Health', fontsize=14)
    axes[0].set_ylabel('State Value (1-100)', fontsize=12)
    axes[0].set_xlabel(None)

    sns.violinplot(ax=axes[1], data=df[df['Metric'] == 'Wealth'], x='Scenario', y='Value', hue='Scenario', inner='quartile', legend=False)
    axes[1].set_title('Final Wealth', fontsize=14)
    axes[1].set_ylabel(None)
    axes[1].set_xlabel(None)

    max_val = config.steering_parameters.max_state_value
    axes[0].set_ylim(0, max_val + 5)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_dir = os.path.dirname(get_full_results_path(experiment_name, config))
    output_filename = os.path.join(output_dir, f"violin_plots_{experiment_name}.png")
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    print(f"Violin plots saved to {output_filename}")
    plt.show()

def plot_final_state_scatter(experiment_name: str, config: SVEIRConfig):
    """Creates 2D density scatter plots of final agent states."""
    print(f"Generating scatter plots for {experiment_name}...")
    aggregated_results = _load_and_aggregate_results(experiment_name, config)
    if not aggregated_results: return

    exp_params = config.experiment_params
    baseline_key = min(aggregated_results.keys(), key=lambda k: abs(k[0]-1.0) + abs(k[1]-1.0))
    best_key = min(aggregated_results.keys(), key=lambda k: abs(k[0]-max(exp_params.efficacy_multipliers)) + abs(k[1]-min(exp_params.cost_subsidy_factors)))

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
    fig.suptitle('Distribution of Final Agent States by Scenario', fontsize=18)
    max_val = config.steering_parameters.max_state_value

    # Plot Baseline
    im1 = axes[0].hist2d(x=aggregated_results[baseline_key]['wealth'], y=aggregated_results[baseline_key]['health'],
                        bins=50, range=[[0, max_val], [0, max_val]], cmap='plasma', norm=LogNorm())
    fig.colorbar(im1[3], ax=axes[0], label='Number of Agents (Log Scale)')
    axes[0].set_title('Baseline Scenario', fontsize=14)
    axes[0].set_xlabel('Final Wealth', fontsize=12)
    axes[0].set_ylabel('Final Health', fontsize=12)

    # Plot Best Intervention
    im2 = axes[1].hist2d(x=aggregated_results[best_key]['wealth'], y=aggregated_results[best_key]['health'],
                        bins=50, range=[[0, max_val], [0, max_val]], cmap='plasma', norm=LogNorm())
    fig.colorbar(im2[3], ax=axes[1], label='Number of Agents (Log Scale)')
    axes[1].set_title('Best Intervention Scenario', fontsize=14)
    axes[1].set_xlabel('Final Wealth', fontsize=12)

    axes[0].set_xlim(0, max_val + 5)
    axes[0].set_ylim(0, max_val + 5)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_dir = os.path.dirname(get_full_results_path(experiment_name, config))
    output_filename = os.path.join(output_dir, f"scatter_plots_{experiment_name}.png")
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    print(f"Scatter plots saved to {output_filename}")
    plt.show()