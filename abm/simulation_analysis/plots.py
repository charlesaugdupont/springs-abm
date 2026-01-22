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

def _load_and_aggregate_results(experiment_name: str, config: SVEIRConfig) -> Dict | None:
    """
    Helper to load full results and aggregate them.
    """
    full_results_file = get_full_results_path(experiment_name, config)
    if not os.path.exists(full_results_file):
        print(f"Error: Full results file not found at '{full_results_file}'.")
        return None
    with open(full_results_file, 'rb') as f:
        raw_results = pickle.load(f)

    # All results are now part of a single "scenario"
    aggregated = {'prevalence': [], 'health': [], 'wealth': []}
    for res in raw_results:
        if 'prevalence_curve' in res and res.get('proportion_infected', -1) >= 0:
            aggregated['prevalence'].append(res['prevalence_curve'])
        if 'final_health' in res and res.get('proportion_infected', -1) >= 0:
            aggregated['health'].extend(res['final_health'])
            aggregated['wealth'].extend(res['final_wealth'])
    return aggregated

def plot_epidemic_curves(experiment_name: str, config: SVEIRConfig):
    """Plots the average epidemic prevalence curve across all repetitions."""
    print(f"Generating epidemic curves for {experiment_name}...")
    aggregated_results = _load_and_aggregate_results(experiment_name, config)
    if not aggregated_results or not aggregated_results['prevalence']:
        print("No valid results found to plot.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    curves = aggregated_results['prevalence']
    max_len = max(len(c) for c in curves) if curves else 0
    padded = [np.pad(c, (0, max_len - len(c)), 'constant') for c in curves]
    mean_curve, std_curve = np.mean(padded, axis=0), np.std(padded, axis=0)
    timesteps = np.arange(len(mean_curve))

    line, = ax.plot(timesteps, mean_curve, label=f"Mean of {len(curves)} repetitions", lw=2)
    ax.fill_between(timesteps, mean_curve - std_curve, mean_curve + std_curve, alpha=0.15, color=line.get_color())

    ax.set_title("Epidemic Progression (Prevalence)", fontsize=16, pad=15)
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
    """Creates violin plots showing the distribution of final agent health and wealth."""
    print(f"Generating violin plots for {experiment_name}...")
    aggregated_results = _load_and_aggregate_results(experiment_name, config)
    if not aggregated_results or not aggregated_results['health']:
        print("No valid results found to plot.")
        return

    plot_data = []
    for health_val in aggregated_results['health']:
        plot_data.append({'Value': health_val, 'Metric': 'Health'})
    for wealth_val in aggregated_results['wealth']:
        plot_data.append({'Value': wealth_val, 'Metric': 'Wealth'})
    df = pd.DataFrame(plot_data)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharey=True)
    fig.suptitle('Distribution of Final Agent States', fontsize=18)

    sns.violinplot(ax=axes[0], data=df[df['Metric'] == 'Health'], y='Value', inner='quartile')
    axes[0].set_title('Final Health', fontsize=14)
    axes[0].set_ylabel('State Value (Normalized)', fontsize=12)
    axes[0].set_xlabel(None)

    sns.violinplot(ax=axes[1], data=df[df['Metric'] == 'Wealth'], y='Value', inner='quartile')
    axes[1].set_title('Final Wealth', fontsize=14)
    axes[1].set_ylabel(None)
    axes[1].set_xlabel(None)

    axes[0].set_ylim(-0.05, 1.05)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_dir = os.path.dirname(get_full_results_path(experiment_name, config))
    output_filename = os.path.join(output_dir, f"violin_plots_{experiment_name}.png")
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    print(f"Violin plots saved to {output_filename}")
    plt.show()

def plot_final_state_scatter(experiment_name: str, config: SVEIRConfig):
    """Creates a 2D density scatter plot of final agent states."""
    print(f"Generating scatter plot for {experiment_name}...")
    aggregated_results = _load_and_aggregate_results(experiment_name, config)
    if not aggregated_results or not aggregated_results['health']:
        print("No valid results found to plot.")
        return

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.suptitle('Distribution of Final Agent States', fontsize=18)

    im = ax.hist2d(x=aggregated_results['wealth'], y=aggregated_results['health'],
                   bins=50, range=[[0, 1], [0, 1]], cmap='plasma', norm=LogNorm())
    fig.colorbar(im[3], ax=ax, label='Number of Agents (Log Scale)')
    ax.set_title(f'Final Agent States from a Single Simulation Run', fontsize=14)
    ax.set_xlabel('Final Wealth (Normalized)', fontsize=12)
    ax.set_ylabel('Final Health (Normalized)', fontsize=12)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_dir = os.path.dirname(get_full_results_path(experiment_name, config))
    output_filename = os.path.join(output_dir, f"scatter_plot_{experiment_name}.png")
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    print(f"Scatter plot saved to {output_filename}")
    plt.show()