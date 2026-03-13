# abm/simulation_analysis/plots.py
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict

from .experiment_config import get_existing_results_path

def _load_and_aggregate_results(experiment_name: str) -> Dict | None:
    """
    Helper to load full results and aggregate them.
    """
    # Use the new config-agnostic path finder
    try:
        full_results_file = get_existing_results_path(experiment_name)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None

    with open(full_results_file, 'rb') as f:
        raw_results = pickle.load(f)

    # All results are now part of a single "scenario"
    aggregated = {
        'prevalence': [], 'health': [], 'wealth': [], 'initial_health': [], 
        'initial_wealth': [], 'care_seeking_count': [], 'is_parent': [],
        'alpha':[], 'gamma':[], 'lambda':[],
    }
    for res in raw_results:
        if 'prevalence_curve' in res and res.get('proportion_infected', -1) >= 0:
            aggregated['prevalence'].append(res['prevalence_curve'])
        if 'final_health' in res and res.get('proportion_infected', -1) >= 0:
            aggregated['health'].extend(res['final_health'])
            aggregated['wealth'].extend(res['final_wealth'])
            if 'initial_health' in res:
                aggregated['initial_health'].extend(res['initial_health'])
                aggregated['initial_wealth'].extend(res['initial_wealth'])
                aggregated['care_seeking_count'].extend(res['care_seeking_count'])
                aggregated['is_parent'].extend(res['is_parent'])
                if 'alpha' in res:
                    aggregated['alpha'].extend(res['alpha'])
                    aggregated['gamma'].extend(res['gamma'])
                    aggregated['lambda'].extend(res['lambda'])
    return aggregated

def plot_epidemic_curves(experiment_name: str):
    """Plots the average epidemic prevalence curve across all repetitions."""
    print(f"Generating epidemic curves for {experiment_name}...")
    aggregated_results = _load_and_aggregate_results(experiment_name)
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

    # Save to the same directory we found the results in
    output_dir = os.path.dirname(get_existing_results_path(experiment_name))
    output_filename = os.path.join(output_dir, f"prevalence_curves_{experiment_name}.png")
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    print(f"Prevalence curves saved to {output_filename}")
    plt.show()

def plot_care_seeking_analysis(experiment_name: str):
    """Analyzes and plots care-seeking behavior against agent endowments and personas."""
    print(f"Generating care-seeking analysis plots for {experiment_name}...")
    aggregated_results = _load_and_aggregate_results(experiment_name)
    
    if not aggregated_results or not aggregated_results.get('care_seeking_count'):
        print("No valid care-seeking results found to plot.")
        return

    # Create a DataFrame for easier analysis
    df = pd.DataFrame({
        'initial_wealth': aggregated_results['initial_wealth'],
        'final_wealth': aggregated_results['wealth'],
        'final_health': aggregated_results['health'],
        'care_seeking_count': aggregated_results['care_seeking_count'],
        'is_parent': aggregated_results['is_parent'],
        'alpha': aggregated_results.get('alpha', []),
        'gamma': aggregated_results.get('gamma', []),
        'lambda': aggregated_results.get('lambda', [])
    })

    # Filter for only parents, as they are the decision-makers
    parents_df = df[df['is_parent']].copy()
    if parents_df.empty:
        print("No parent agents found in the simulation results.")
        return
        
    # Add jitter for better visualization in scatter plots
    parents_df['care_seeking_jitter'] = parents_df['care_seeking_count'] + np.random.normal(0, 0.1, size=len(parents_df))

    # --- Plotting ---
    sns.set_theme(style="whitegrid")
    # MODIFIED: Changed to a 3x2 grid and increased figsize
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('Analysis of Parental Care-Seeking Decisions', fontsize=20)

    # --- ROW 1: CORE METRICS ---
    # 1. Distribution of Care-Seeking Choices
    ax1 = axes[0, 0]
    sns.histplot(data=parents_df, x='care_seeking_count', discrete=True, ax=ax1, stat="percent")
    ax1.set_title('Frequency of Care-Seeking Decisions', fontsize=14)
    ax1.set_xlabel('Number of Times Care Was Sought', fontsize=12)
    ax1.set_ylabel('Percentage of Parents (%)', fontsize=12)
    if not parents_df.empty:
        max_count = parents_df['care_seeking_count'].max()
        if not np.isnan(max_count) and max_count > 0:
            ax1.set_xticks(np.arange(max_count + 1))

    # 2. Initial Wealth vs. Care-Seeking
    ax2 = axes[0, 1]
    sns.regplot(data=parents_df, x='initial_wealth', y='care_seeking_jitter', ax=ax2,
                scatter_kws={'alpha': 0.4, 's': 15}, line_kws={'color': 'red'})
    ax2.set_title('Initial Wealth vs. Care-Seeking', fontsize=14)
    ax2.set_xlabel('Initial Wealth', fontsize=12)
    ax2.set_ylabel('Number of Times Care Was Sought', fontsize=12)
    ax2.set_ylim(bottom=-0.5)

    # --- ROW 2: BEHAVIORAL PERSONAS ---
    # 3. Alpha (Utility) vs. Care-Seeking
    ax3 = axes[1, 0]
    sns.regplot(data=parents_df, x='alpha', y='care_seeking_jitter', ax=ax3,
                scatter_kws={'alpha': 0.4, 's': 15}, line_kws={'color': 'green'})
    ax3.set_title('Alpha (Health/Wealth Utility) vs. Care-Seeking', fontsize=14)
    ax3.set_xlabel('Alpha Value (Higher = More Wealth-Focused)', fontsize=12)
    ax3.set_ylabel('Number of Times Care Was Sought', fontsize=12)
    ax3.set_ylim(bottom=-0.5)

    # 4. Lambda (Loss Aversion) vs. Care-Seeking
    ax4 = axes[1, 1]
    sns.regplot(data=parents_df, x='lambda', y='care_seeking_jitter', ax=ax4,
                scatter_kws={'alpha': 0.4, 's': 15}, line_kws={'color': 'green'})
    ax4.set_title('Lambda (Loss Aversion) vs. Care-Seeking', fontsize=14)
    ax4.set_xlabel('Lambda Value (Higher = More Loss Averse)', fontsize=12)
    ax4.set_ylabel('Number of Times Care Was Sought', fontsize=12)
    ax4.set_ylim(bottom=-0.5)

    # --- ROW 3: OUTCOMES ---
    # 5. Final Wealth vs. Care-Seeking
    ax5 = axes[2, 0]
    sns.regplot(data=parents_df, x='final_wealth', y='care_seeking_jitter', ax=ax5,
                scatter_kws={'alpha': 0.4, 's': 15}, line_kws={'color': 'purple'})
    ax5.set_title('Outcome: Final Wealth vs. Care-Seeking', fontsize=14)
    ax5.set_xlabel('Final Wealth', fontsize=12)
    ax5.set_ylabel('Number of Times Care Was Sought', fontsize=12)
    ax5.set_ylim(bottom=-0.5)

    # 6. Gamma (Probability Weighting) vs. Care-Seeking
    ax6 = axes[2, 1]
    sns.regplot(data=parents_df, x='gamma', y='care_seeking_jitter', ax=ax6,
                scatter_kws={'alpha': 0.4, 's': 15}, line_kws={'color': 'green'})
    ax6.set_title('Gamma (Risk Perception) vs. Care-Seeking', fontsize=14)
    ax6.set_xlabel('Gamma Value (Lower = More Pessimistic)', fontsize=12)
    ax6.set_ylabel('Number of Times Care Was Sought', fontsize=12)
    ax6.set_ylim(bottom=-0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the plot
    output_dir = os.path.dirname(get_existing_results_path(experiment_name))
    output_filename = os.path.join(output_dir, f"care_seeking_analysis_{experiment_name}.png")
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    print(f"Care-seeking analysis plot saved to {output_filename}")
    plt.show()