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
    """Loads full results and aggregates them across repetitions."""
    try:
        full_results_file = get_existing_results_path(experiment_name)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None

    with open(full_results_file, 'rb') as f:
        raw_results = pickle.load(f)

    aggregated = {
        'u5_prevalence': [], 'health': [], 'wealth': [], 'initial_health': [],
        'initial_wealth': [], 'care_seeking_count': [], 'is_parent': [],
        'is_child': [], 'age': [],
        'alpha': [], 'gamma': [], 'lambda': [],
        'child_episode_log': [],
    }
    for res in raw_results:
        if res.get('proportion_infected', -1) < 0:
            continue
        if 'prevalence_curve' in res:
            aggregated['u5_prevalence'].append(res['prevalence_curve'])
        if 'final_health' in res:
            aggregated['health'].extend(res['final_health'])
            aggregated['wealth'].extend(res['final_wealth'])
        if 'initial_health' in res:
            aggregated['initial_health'].extend(res['initial_health'])
            aggregated['initial_wealth'].extend(res['initial_wealth'])
            aggregated['care_seeking_count'].extend(res['care_seeking_count'])
            aggregated['is_parent'].extend(res['is_parent'])
        if 'is_child' in res:
            aggregated['is_child'].extend(res['is_child'])
        if 'age' in res:
            aggregated['age'].extend(res['age'])
        if 'alpha' in res:
            aggregated['alpha'].extend(res['alpha'])
            aggregated['gamma'].extend(res['gamma'])
            aggregated['lambda'].extend(res['lambda'])
        if 'child_episode_log' in res:
            aggregated['child_episode_log'].extend(res['child_episode_log'])

    return aggregated


# ---------------------------------------------------------------------------
# Plot 1: Epidemic curves
# ---------------------------------------------------------------------------

def plot_epidemic_curves(experiment_name: str):
    """Plots the average epidemic prevalence curve across all repetitions."""
    print(f"Generating epidemic curves for {experiment_name}...")
    aggregated_results = _load_and_aggregate_results(experiment_name)
    if not aggregated_results or not aggregated_results['u5_prevalence']:
        print("No valid results found to plot.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    rota = aggregated_results['u5_prevalence'][0]["rota"]
    campy = aggregated_results['u5_prevalence'][0]["campy"]

    line1, = ax.plot(np.arange(len(rota)), rota, label=f"Rotavirus", lw=2, color="dodgerblue")
    line2, = ax.plot(np.arange(len(campy)), campy, label=f"Campylobacter", lw=2, color="crimson")

    ax.set_title("Epidemic Progression (<5 Prevalence)", fontsize=16, pad=15)
    ax.set_xlabel("Time (Days)", fontsize=12)
    ax.set_ylabel("Proportion of <5 Infected", fontsize=12)
    ax.legend(title="Scenario", fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_ylim(bottom=0)

    output_dir = os.path.dirname(get_existing_results_path(experiment_name))
    output_filename = os.path.join(output_dir, f"prevalence_curves_{experiment_name}.png")
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    print(f"Prevalence curves saved to {output_filename}")
    plt.show()


# ---------------------------------------------------------------------------
# Plot 2: Care-seeking analysis
# ---------------------------------------------------------------------------

def plot_care_seeking_analysis(experiment_name: str):
    """Analyzes and plots care-seeking behavior against agent endowments and personas."""
    print(f"Generating care-seeking analysis plots for {experiment_name}...")
    aggregated_results = _load_and_aggregate_results(experiment_name)

    if not aggregated_results or not aggregated_results.get('care_seeking_count'):
        print("No valid care-seeking results found to plot.")
        return

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

    parents_df = df[df['is_parent']].copy()
    if parents_df.empty:
        print("No parent agents found in the simulation results.")
        return

    parents_df['care_seeking_jitter'] = parents_df['care_seeking_count'] + np.random.normal(0, 0.1, size=len(parents_df))

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('Analysis of Parental Care-Seeking Decisions', fontsize=20)

    ax1 = axes[0, 0]
    sns.histplot(data=parents_df, x='care_seeking_count', discrete=True, ax=ax1, stat="percent")
    ax1.set_title('Frequency of Care-Seeking Decisions', fontsize=14)
    ax1.set_xlabel('Number of Times Care Was Sought', fontsize=12)
    ax1.set_ylabel('Percentage of Parents (%)', fontsize=12)
    if not parents_df.empty:
        max_count = parents_df['care_seeking_count'].max()
        if not np.isnan(max_count) and max_count > 0:
            ax1.set_xticks(np.arange(max_count + 1))

    ax2 = axes[0, 1]
    sns.regplot(data=parents_df, x='initial_wealth', y='care_seeking_jitter', ax=ax2,
                scatter_kws={'alpha': 0.4, 's': 15}, line_kws={'color': 'red'})
    ax2.set_title('Initial Wealth vs. Care-Seeking', fontsize=14)
    ax2.set_xlabel('Initial Wealth', fontsize=12)
    ax2.set_ylabel('Number of Times Care Was Sought', fontsize=12)
    ax2.set_ylim(bottom=-0.5)

    ax3 = axes[1, 0]
    sns.regplot(data=parents_df, x='alpha', y='care_seeking_jitter', ax=ax3,
                scatter_kws={'alpha': 0.4, 's': 15}, line_kws={'color': 'green'})
    ax3.set_title('Alpha (Health/Wealth Utility) vs. Care-Seeking', fontsize=14)
    ax3.set_xlabel('Alpha Value (Higher = More Wealth-Focused)', fontsize=12)
    ax3.set_ylabel('Number of Times Care Was Sought', fontsize=12)
    ax3.set_ylim(bottom=-0.5)

    ax4 = axes[1, 1]
    sns.regplot(data=parents_df, x='lambda', y='care_seeking_jitter', ax=ax4,
                scatter_kws={'alpha': 0.4, 's': 15}, line_kws={'color': 'green'})
    ax4.set_title('Lambda (Loss Aversion) vs. Care-Seeking', fontsize=14)
    ax4.set_xlabel('Lambda Value (Higher = More Loss Averse)', fontsize=12)
    ax4.set_ylabel('Number of Times Care Was Sought', fontsize=12)
    ax4.set_ylim(bottom=-0.5)

    ax5 = axes[2, 0]
    sns.regplot(data=parents_df, x='final_wealth', y='care_seeking_jitter', ax=ax5,
                scatter_kws={'alpha': 0.4, 's': 15}, line_kws={'color': 'purple'})
    ax5.set_title('Outcome: Final Wealth vs. Care-Seeking', fontsize=14)
    ax5.set_xlabel('Final Wealth', fontsize=12)
    ax5.set_ylabel('Number of Times Care Was Sought', fontsize=12)
    ax5.set_ylim(bottom=-0.5)

    ax6 = axes[2, 1]
    sns.regplot(data=parents_df, x='gamma', y='care_seeking_jitter', ax=ax6,
                scatter_kws={'alpha': 0.4, 's': 15}, line_kws={'color': 'green'})
    ax6.set_title('Gamma (Risk Perception) vs. Care-Seeking', fontsize=14)
    ax6.set_xlabel('Gamma Value (Lower = More Pessimistic)', fontsize=12)
    ax6.set_ylabel('Number of Times Care Was Sought', fontsize=12)
    ax6.set_ylim(bottom=-0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_dir = os.path.dirname(get_existing_results_path(experiment_name))
    output_filename = os.path.join(output_dir, f"care_seeking_analysis_{experiment_name}.png")
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    print(f"Care-seeking analysis plot saved to {output_filename}")
    plt.show()


# ---------------------------------------------------------------------------
# Plot 3: Child illness episode analysis
# ---------------------------------------------------------------------------

def plot_child_illness_analysis(experiment_name: str):
    """
    Visualises per-episode illness data for child agents across six panels:

    Row 1 — Episode distributions
        (a) Distribution of initial illness severity, by pathogen
        (b) Distribution of illness duration (days), by pathogen

    Row 2 — Temporal dynamics
        (c) Daily incidence of new child episodes over the simulation
        (d) Rolling 7-day mean severity over time, by pathogen

    Row 3 — Within-child burden
        (e) Episodes per child (histogram) — how many children had 0, 1, 2 … episodes
        (f) Severity vs. duration scatter, coloured by pathogen, with regression lines
    """
    print(f"Generating child illness analysis for {experiment_name}...")
    aggregated = _load_and_aggregate_results(experiment_name)

    if not aggregated or not aggregated.get('child_episode_log'):
        print("No child episode data found. Re-run the simulation to generate it.")
        return

    episodes = aggregated['child_episode_log']
    df = pd.DataFrame(episodes)

    # Severity labels for readability
    def _severity_label(s: float) -> str:
        if s < 0.2:  return 'Mild'
        if s < 0.7:  return 'Moderate'
        return 'Severe'

    df['severity_category'] = df['initial_severity'].apply(_severity_label)
    df['severity_category'] = pd.Categorical(
        df['severity_category'], categories=['Mild', 'Moderate', 'Severe'], ordered=True
    )

    pathogen_palette = {'rota': '#2196F3', 'campy': '#FF5722'}
    pathogens_present = sorted(df['pathogen'].unique())

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('Child Illness Episode Analysis', fontsize=20, y=1.01)

    # ------------------------------------------------------------------
    # (a) Severity distribution by pathogen
    # ------------------------------------------------------------------
    ax = axes[0, 0]
    for p in pathogens_present:
        sub = df[df['pathogen'] == p]['initial_severity']
        ax.hist(sub, bins=30, alpha=0.65, label=p.capitalize(),
                color=pathogen_palette.get(p), edgecolor='white', linewidth=0.4)
    ax.set_title('Distribution of Initial Illness Severity', fontsize=14)
    ax.set_xlabel('Initial Severity (0–1)', fontsize=12)
    ax.set_ylabel('Number of Episodes', fontsize=12)
    ax.legend(title='Pathogen', fontsize=10)
    ax.set_xlim(0, 1)

    # ------------------------------------------------------------------
    # (b) Duration distribution by pathogen
    # ------------------------------------------------------------------
    ax = axes[0, 1]
    max_dur = int(df['initial_duration'].max()) if not df.empty else 14
    bins = np.arange(0, max_dur + 2) - 0.5
    for p in pathogens_present:
        sub = df[df['pathogen'] == p]['initial_duration']
        ax.hist(sub, bins=bins, alpha=0.65, label=p.capitalize(),
                color=pathogen_palette.get(p), edgecolor='white', linewidth=0.4)
    ax.set_title('Distribution of Illness Duration', fontsize=14)
    ax.set_xlabel('Initial Duration (Days)', fontsize=12)
    ax.set_ylabel('Number of Episodes', fontsize=12)
    ax.legend(title='Pathogen', fontsize=10)
    ax.set_xticks(range(0, max_dur + 2))

    # ------------------------------------------------------------------
    # (c) Daily incidence of new child episodes
    # ------------------------------------------------------------------
    ax = axes[1, 0]
    if not df.empty:
        max_ts = int(df['timestep'].max())
        all_days = np.arange(0, max_ts + 1)
        for p in pathogens_present:
            counts = df[df['pathogen'] == p].groupby('timestep').size().reindex(all_days, fill_value=0)
            ax.plot(all_days, counts.values, color=pathogen_palette.get(p),
                    alpha=0.4, linewidth=0.8)
            # 7-day rolling mean
            rolling = pd.Series(counts.values).rolling(7, min_periods=1).mean()
            ax.plot(all_days, rolling.values, color=pathogen_palette.get(p),
                    linewidth=2, label=f'{p.capitalize()} (7-day avg)')
    ax.set_title('Daily New Child Illness Episodes', fontsize=14)
    ax.set_xlabel('Simulation Day', fontsize=12)
    ax.set_ylabel('New Episodes', fontsize=12)
    ax.legend(title='Pathogen', fontsize=10)
    ax.set_ylim(bottom=0)

    # ------------------------------------------------------------------
    # (d) Rolling mean severity over time
    # ------------------------------------------------------------------
    ax = axes[1, 1]
    if not df.empty:
        max_ts = int(df['timestep'].max())
        all_days = np.arange(0, max_ts + 1)
        for p in pathogens_present:
            daily_mean = (
                df[df['pathogen'] == p]
                .groupby('timestep')['initial_severity']
                .mean()
                .reindex(all_days)
            )
            # Raw (faint) + 7-day rolling mean (bold)
            ax.plot(all_days, daily_mean.values, color=pathogen_palette.get(p),
                    alpha=0.25, linewidth=0.8)
            rolling_sev = daily_mean.rolling(7, min_periods=1).mean()
            ax.plot(all_days, rolling_sev.values, color=pathogen_palette.get(p),
                    linewidth=2, label=f'{p.capitalize()} (7-day avg)')
    ax.set_title('Mean Episode Severity Over Time', fontsize=14)
    ax.set_xlabel('Simulation Day', fontsize=12)
    ax.set_ylabel('Mean Initial Severity', fontsize=12)
    ax.legend(title='Pathogen', fontsize=10)
    ax.set_ylim(0, 1)

    # ------------------------------------------------------------------
    # (e) Episodes per child
    # ------------------------------------------------------------------
    ax = axes[2, 0]
    if not df.empty:
        episodes_per_child = df.groupby('agent_idx').size()
        # Include children who had zero episodes
        is_child_arr = np.array(aggregated.get('is_child', []), dtype=bool)
        if is_child_arr.any():
            all_child_indices = np.where(is_child_arr)[0]
            full_counts = pd.Series(0, index=all_child_indices)
            full_counts.update(episodes_per_child)
        else:
            full_counts = episodes_per_child

        max_ep = int(full_counts.max())
        bins_ep = np.arange(0, max_ep + 2) - 0.5
        ax.hist(full_counts.values, bins=bins_ep, color='steelblue',
                edgecolor='white', linewidth=0.5)
        ax.set_xticks(range(0, max_ep + 1))
        mean_ep = full_counts.mean()
        ax.axvline(mean_ep, color='crimson', linestyle='--', linewidth=1.5,
                   label=f'Mean = {mean_ep:.2f}')
        ax.legend(fontsize=10)
    ax.set_title('Episodes per Child Agent', fontsize=14)
    ax.set_xlabel('Number of Illness Episodes', fontsize=12)
    ax.set_ylabel('Number of Children', fontsize=12)

    # ------------------------------------------------------------------
    # (f) Severity vs. duration scatter with regression
    # ------------------------------------------------------------------
    ax = axes[2, 1]
    if not df.empty:
        for p in pathogens_present:
            sub = df[df['pathogen'] == p]
            # Jitter duration slightly for readability
            jittered_dur = sub['initial_duration'] + np.random.uniform(-0.25, 0.25, size=len(sub))
            ax.scatter(sub['initial_severity'], jittered_dur,
                       color=pathogen_palette.get(p), alpha=0.35, s=12, label=p.capitalize())
            # Regression line
            if len(sub) > 2:
                z = np.polyfit(sub['initial_severity'], sub['initial_duration'], 1)
                x_line = np.linspace(sub['initial_severity'].min(), sub['initial_severity'].max(), 100)
                ax.plot(x_line, np.polyval(z, x_line),
                        color=pathogen_palette.get(p), linewidth=2)
    ax.set_title('Severity vs. Duration per Episode', fontsize=14)
    ax.set_xlabel('Initial Severity (0-1)', fontsize=12)
    ax.set_ylabel('Initial Duration (Days)', fontsize=12)
    ax.legend(title='Pathogen', fontsize=10)
    ax.set_xlim(0, 1)

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    output_dir = os.path.dirname(get_existing_results_path(experiment_name))
    output_filename = os.path.join(output_dir, f"child_illness_analysis_{experiment_name}.png")
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    print(f"Child illness analysis saved to {output_filename}")
    plt.show()
