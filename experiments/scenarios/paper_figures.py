# experiments/scenarios/paper_figures.py
"""
The manuscript's Results figures, one per experiment, drawn from the reruns of
the revision (one measured year after
the two-year warm-up, the adopted vaccine, 40 seeds for the key contrasts) and
their analysis outputs, laid out for print in Nature Communications: 180 mm
wide (two columns), Arial-metric sans-serif (Liberation Sans) at 5-7 pt, bold
lowercase panel letters at 8 pt, editable text in the PDF (TrueType), white
background.

    python -m experiments.scenarios.paper_figures [1a 1b 2 3a 3b] --results <dir> --out <dir>

`--results` holds one folder per part, named as the reruns write them
(PART_DIRS: rev_1a_care_levers, rev_1b_learning, rev_2_shocks,
rev_3a_vaccine_limits, rev_3b_vaccine_interventions), each with its
results.parquet and the `analysis` folder that experiments.scenarios.analysis
writes (`python -m experiments.scenarios.analysis 1a_rev`, ...). The figures of
the first round (runs of four to six years, 8 seeds) were drawn by this
module's earlier version (experiments_v5_code.patch).

Colours (dataviz validator, white surface): categorical blue #2a78d6 and
orange #eb6834 (all-pairs CVD-safe); a sequential blue ramp for ordered levels
(#86b6ef to #0d366b); diverging blue-red with a neutral midpoint for signed
changes; ink #0b0b0b, secondary #52514e.
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

MM = 1 / 25.4
WIDTH = 180 * MM
BLUE, ORANGE = "#2a78d6", "#eb6834"
RAMP = {250: "#86b6ef", 300: "#6da7ec", 400: "#3987e5", 450: "#2a78d6", 550: "#1c5cab", 650: "#104281",
        700: "#0d366b", 200: "#9ec5f4", 150: "#b7d3f6", 100: "#cde2fb"}
INK, INK2, MUTED, HAIR = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
NEUTRAL = "#f0efec"
LIGHT_BLUE = "#b7d3f6"
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_red", ["#0d366b", "#1c5cab", "#3987e5", "#86b6ef", "#cde2fb", NEUTRAL,
                 "#fbd3cf", "#f4a49c", "#e8706a", "#c93a3a", "#8f1f24"])
LABEL = "gsa.design_id"
FONT = "Liberation Sans"


def style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [FONT, "Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "legend.title_fontsize": 6.5,
        "axes.linewidth": 0.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "xtick.major.pad": 2, "ytick.major.pad": 2, "axes.labelpad": 3,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.1, "lines.markersize": 3.5,
        "patch.linewidth": 0, "legend.frameon": False, "legend.handlelength": 1.6, "legend.borderaxespad": 0.2,
        "legend.labelspacing": 0.3, "legend.handletextpad": 0.5,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "figure.constrained_layout.h_pad": 2 * MM, "figure.constrained_layout.w_pad": 2 * MM,
        "figure.constrained_layout.hspace": 0.04, "figure.constrained_layout.wspace": 0.04,
    })


def panel(ax, letter: str, title: str = "") -> None:
    """A bold lowercase letter (8 pt) at the panel's top left, on the left edge of the axes' labels, followed by a
    short title (7 pt). An empty title is set now so that the layout keeps room above the axes; the letter and
    the title are placed when the figure is saved (see `save`)."""
    ax.set_title(" ", loc="left", fontsize=7, pad=5)
    ax._panel = (letter, title)


def _place_panels(fig) -> None:
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for ax in fig.axes:
        if not hasattr(ax, "_panel"):
            continue
        letter, title = ax._panel
        x0 = inv.transform((ax.get_tightbbox(rend, for_layout_only=True).x0, 0))[0]
        y0 = inv.transform((0, ax._left_title.get_window_extent(rend).y0))[1]
        t = fig.text(x0, y0, letter, fontsize=8, fontweight="bold", va="bottom", ha="left")
        if title:
            x1 = inv.transform((t.get_window_extent(rend).x1, 0))[0]
            fig.text(x1 + 1.6 * MM / fig.get_figwidth(), y0, title, fontsize=7, va="bottom", ha="left")


def save(fig, out_dir: str, name: str) -> dict:
    _place_panels(fig)
    os.makedirs(out_dir, exist_ok=True)
    paths = {"pdf": os.path.join(out_dir, f"{name}.pdf"), "png": os.path.join(out_dir, f"{name}.png")}
    fig.savefig(paths["pdf"])
    fig.savefig(paths["png"], dpi=300)
    plt.close(fig)
    return paths


#: each part's folder under --results (the reruns' output folders, experiments.scenarios.designs)
PART_DIRS = {"1a": "rev_1a_care_levers", "1b": "rev_1b_learning", "2": "rev_2_shocks",
             "3a": "rev_3a_vaccine_limits", "3b": "rev_3b_vaccine_interventions"}


def _results(results_dir: str, part: str, *files: str) -> pd.DataFrame:
    files = files or ("results.parquet",)
    return pd.concat([pd.read_parquet(os.path.join(results_dir, PART_DIRS[part], f)) for f in files],
                     ignore_index=True)


def _analysis_dir(results_dir: str, part: str) -> str:
    """A part's analysis outputs: in `<part>/analysis/` as experiments.scenarios.analysis writes them, or
    directly in `<part>/`."""
    d = os.path.join(results_dir, PART_DIRS[part], "analysis")
    return d if os.path.isdir(d) else os.path.join(results_dir, PART_DIRS[part])


def _level(df, label, metric, seeds=None):
    """Mean and SE over seeds of `metric` in scenario `label` (only `seeds`, if given: a sweep whose levels
    ran with fewer seeds than its reference is drawn on the seeds they share)."""
    g = df[df[LABEL] == label]
    if seeds is not None:
        g = g[g["seed"].isin(seeds)]
    v = g[metric]
    return float(v.mean()), float(v.sem())


def _seeds(df, label):
    return set(df.loc[df[LABEL] == label, "seed"])


def _hgrid(ax, axis="y"):
    ax.grid(axis=axis, color=HAIR, lw=0.4)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Experiment 1A: what holds care-seeking at 0.6
# ---------------------------------------------------------------------------

#: the policies panel: (group, [(scenario, label)]); credit for a visit is a sensitivity (decision R2: borrowing
#: stays off, as calibrated)
POLICY_GROUPS = [
    ("Free care", [("price_free", "All care free"),
                   ("free_ors_zinc", "Free ORS and zinc at drug sellers"),
                   ("free_primary_insured", "Free primary care for NHIS members")]),
    ("Insurance", [("nhis_universal", "NHIS for everyone"),
                   ("nhis_universal_free_primary", "NHIS for everyone, primary care free")]),
    ("Cash transfers", [("leap_q1_x1", "LEAP grants, lowest quintile"),
                        ("leap_q1_x2", "Twice the grant, lowest quintile"),
                        ("leap_q12_x1", "LEAP grants, two lowest quintiles"),
                        ("leap_q12_x2", "Twice the grant, two lowest quintiles")]),
    ("Lost earnings", [("compensate_visit_time", "Earnings lost to a visit repaid"),
                       ("compensate_caring_time", "Earnings lost to caring repaid")]),
    ("Credit for a visit", [("credit_1d", "Up to 1 day of consumption"),
                            ("credit_3d", "Up to 3 days of consumption"),
                            ("credit_10d", "Up to 10 days of consumption")]),
]
PERCEIVED = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.6, 0.8, 1.0)       # 0.8: today (the baseline)


def fig_1a(results_dir: str, out_dir: str) -> dict:
    from experiments.scenarios.analysis import paired
    style()
    df = _results(results_dir, "1a")
    labels = set(df[LABEL])
    care, care_q1, treated = "econ_episode_care_seeking_rate", "econ_care_seeking_rate_q1", "exp_treated_share"
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(WIDTH, 96 * MM), layout="constrained",
                                     gridspec_kw={"width_ratios": [0.78, 1.5]})

    # a: perceived efficacy of care (care works with probability 0.8 throughout): care sought, and episodes
    # effectively treated; the sweep on the seeds its levels share (today's scenario has more)
    ax = ax_a
    name = lambda p: "baseline" if p == 0.8 else f"perceived_{p:g}"
    seeds = set.intersection(*[_seeds(df, name(p)) for p in PERCEIVED if name(p) in labels])
    ax.axvline(0.8, color=HAIR, lw=0.8, zorder=0)
    ax.text(0.8, 1.035, "today", ha="center", va="bottom", fontsize=6, color=INK2)
    for metric, ls, mfc, lab in ((care, "-", BLUE, "care sought"),
                                 (treated, (0, (3, 1.5)), "white", "effectively treated")):
        pts = [(p, *_level(df, name(p), metric, seeds)) for p in PERCEIVED if name(p) in labels]
        xs, ys, es = map(np.array, zip(*pts))
        ax.errorbar(xs, ys, yerr=es, color=BLUE, ls=ls, marker="o", mfc=mfc, mec=BLUE, mew=0.7, ms=3.2,
                    elinewidth=0.6, capsize=0, lw=1.1, label=lab)
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1.03)
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("Perceived probability that care works")
    ax.set_ylabel("Share of episodes")
    panel(ax, "a", "Perceived efficacy of care")

    # b: policies against today, all households and the lowest quintile (paired by seed, +- SE)
    ax = ax_b
    rows, ylabels, group_rows = [], [], []
    y = 0.0
    for g, items in POLICY_GROUPS:
        items = [(lab, text) for lab, text in items if lab in labels]
        if not items:
            continue
        group_rows.append((g, y))
        y -= 0.85
        for lab, text in items:
            rows.append((lab, y))
            ylabels.append((y, text))
            y -= 1.0
        y -= 0.45
    lo, hi = 0.0, 0.0
    right = {}                                   # per row, the rightmost end of its error bars
    for metric, color, off, lab in ((care, BLUE, 0.16, "all households"), (care_q1, ORANGE, -0.16, "lowest quintile")):
        d = [paired(df, metric, l, "baseline") for l, _ in rows]
        xs = np.array([100 * r["diff"] for r in d])
        es = np.array([100 * r["se"] for r in d])
        lo, hi = min(lo, float(np.min(xs - es))), max(hi, float(np.max(xs + es)))
        for (l, _), x1 in zip(rows, xs + es):
            right[l] = max(right.get(l, -np.inf), float(x1))
        ax.errorbar(xs, [yy + off for _, yy in rows], xerr=es, fmt="o", color=color, ms=3.2, mec="white", mew=0.5,
                    elinewidth=0.7, capsize=0, label=lab)
    ax.axvline(0, color=INK2, lw=0.5)
    ax.set_yticks([yy for yy, _ in ylabels], [t for _, t in ylabels])
    ax.tick_params(axis="y", length=0)
    for g, yy in group_rows:
        ax.text(-0.005, yy, g, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=6.5,
                fontweight="bold")
    ax.set_ylim(y + 0.75, 0.45)
    step = 10
    ax.set_xlim(step * np.floor(min(lo, -2) / step) if lo < -2 else -5, step * np.ceil(hi / step) + 2)
    ax.set_xlabel("Change in care sought per episode against today (percentage points)")
    _hgrid(ax, "x")
    ax.spines["left"].set_visible(False)
    base, base_q1 = _level(df, "baseline", care)[0], _level(df, "baseline", care_q1)[0]
    # the legend at the right of the group with the most room there, its top level with the group's header
    spans = []
    for k, (g, yy) in enumerate(group_rows):
        nxt = group_rows[k + 1][1] if k + 1 < len(group_rows) else -np.inf
        members = [l for l, ry in rows if nxt < ry < yy]
        spans.append((max(right[l] for l in members), len(members), yy))
    _, _, y_top = min(spans, key=lambda t: (t[0] if t[1] >= 2 else np.inf, -t[1]))
    y0, y1 = ax.get_ylim()
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, (y_top + 0.5 - y0) / (y1 - y0)),
              title=f"Today: {base:.2f} of episodes (lowest quintile {base_q1:.2f})", alignment="left")
    panel(ax, "b", "Policies")
    return save(fig, out_dir, "fig_1a_care_seeking")


# ---------------------------------------------------------------------------
# Experiment 1B: learning whether care works
# ---------------------------------------------------------------------------

DURATION_SHADES = {2: RAMP[250], 4: RAMP[300], 8: RAMP[400], 16: RAMP[550], 26: RAMP[700]}
WEIGHT_SHADES = {0.3: RAMP[300], 0.6: RAMP[650]}


def _mean_weekly(df, label, key, k=1):
    from experiments.scenarios.analysis import rolling, weekly
    xs = [rolling(weekly(r, key), k) if k > 1 else weekly(r, key) for _, r in df[df[LABEL] == label].iterrows()]
    return np.nanmean(np.vstack(xs), axis=0)


def fig_1b(results_dir: str, out_dir: str) -> dict:
    """The rerun of 1B in one year: stock-outs from the end of the warm-up (B1) and what restores trust after a
    26-week one (B4), followed to the end of the measured year."""
    from experiments.scenarios.analysis import _stockout_start
    style()
    df = _results(results_dir, "1b")
    labels = set(df[LABEL])
    an = _analysis_dir(results_dir, "1b")
    summ = pd.read_csv(os.path.join(an, "recovery_1b1_summary.csv"))
    b4 = pd.read_csv(os.path.join(an, "recovery_1b4.csv"))
    care_key = "wk_episodes_ended_with_care/wk_episodes_ended"
    start = _stockout_start(df)
    w_start = start // 7
    w_back = (start + 182) // 7                    # supplies return after the 26-week stock-out
    weeks = lambda w, w0: np.asarray(w) - w0

    fig = plt.figure(figsize=(WIDTH, 118 * MM), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, 1])
    ax_a, ax_b = top.subplots(1, 2, gridspec_kw={"width_ratios": [1.55, 1]})
    ax_c, ax_d = bottom.subplots(1, 2, gridspec_kw={"width_ratios": [1.55, 1]})

    # a: trust through stock-outs of 2-26 weeks (neighbours' weight 0.6), and care sought for the longest
    ax = ax_a
    ref = _mean_weekly(df, "B1_s0.6_none", "wk_belief_success")
    w = np.arange(len(ref))
    keep = weeks(w, w_start) >= -8
    x = weeks(w, w_start)[keep]
    ax.axvspan(0, 26, color=NEUTRAL, lw=0, zorder=0)
    handles = []
    for n_weeks, color in DURATION_SHADES.items():
        lab = f"B1_s0.6_stockout{n_weeks}w"
        if lab in labels:
            ax.plot(x, _mean_weekly(df, lab, "wk_belief_success")[keep], color=color, lw=1.1)
            handles.append(Line2D([], [], color=color, lw=1.1, label=f"{n_weeks} weeks"))
    ax.plot(x, ref[keep], color=INK2, lw=0.8)
    handles.append(Line2D([], [], color=INK2, lw=0.8, label="no stock-out"))
    for lab, ls in (("B1_s0.6_none", (0, (1, 1.2))), ("B1_s0.6_stockout26w", "-")):
        c = _mean_weekly(df, lab, care_key, k=8)
        ax.plot(x, c[keep], color=ORANGE, lw=0.8, ls=ls)
    leg1 = ax.legend(handles=handles, title="Belief that care works, after a stock-out of", loc="lower left",
                     bbox_to_anchor=(0.43, 0.0), alignment="left", ncols=2)
    ax.add_artist(leg1)
    ax.legend(handles=[Line2D([], [], color=ORANGE, lw=0.8, label="26-week stock-out"),
                       Line2D([], [], color=ORANGE, lw=0.8, ls=(0, (1, 1.2)), label="no stock-out")],
              title="Care sought per episode", loc="lower left", bbox_to_anchor=(0.0, 0.0), alignment="left")
    ax.set_xlim(-8, x[-1])
    ax.set_xticks(np.arange(0, 53, 13))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Weeks since the stock-out began (shaded: the longest, 26 weeks)")
    ax.set_ylabel("Mean belief, or share of episodes")
    panel(ax, "a", "Trust through stock-outs (neighbours' weight 0.6)")

    # b: trust still missing at the end of the year (last 4 weeks), against the same seed without a stock-out
    ax = ax_b
    ax.axhline(0, color=INK2, lw=0.5)
    for wgt, dx in ((0.3, -0.5), (0.6, 0.5)):
        sm = summ[summ["social_weight"] == wgt].sort_values("weeks")
        if sm.empty:
            continue
        ax.errorbar(sm["weeks"] + dx, sm["belief_gap_end"], yerr=sm["belief_gap_end_se"], color=WEIGHT_SHADES[wgt],
                    marker="o", ms=3.2, mec="white", mew=0.5, elinewidth=0.6, capsize=0, lw=1.1,
                    label=f"{wgt:g}")
    sm0 = summ[summ["social_weight"] == 0.0]
    if len(sm0):
        ax.errorbar(sm0["weeks"], sm0["belief_gap_end"], yerr=sm0["belief_gap_end_se"], fmt="s", color=MUTED,
                    ms=3.0, mec="white", mew=0.5, elinewidth=0.6, capsize=0, label="0 (own visits only)")
    ax.set_xticks([2, 4, 8, 16, 26])
    ax.set_xlim(0, 28.5)
    ax.set_xlabel("Stock-out (weeks)")
    ax.set_ylabel("Mean belief at the year's end,\nagainst no stock-out")
    ax.legend(title="Neighbours' weight", loc="lower left", alignment="left")
    panel(ax, "b", "Trust still missing at the year's end")

    # c: what restores trust after a 26-week stock-out: the belief against the same seed without a stock-out
    ax = ax_c
    x = weeks(w, w_back)
    keep = x >= -26
    ax.axvspan(0, 13, color=NEUTRAL, lw=0, zorder=0)
    ax.text(13.5, 0.02, "free care", ha="left", va="bottom", fontsize=6, color=INK2, transform=ax.get_xaxis_transform())
    for lab, color, ls, text in (("B4_collapse", INK2, "-", "none"),
                                 ("B4_campaign", BLUE, "-", "campaign"),
                                 ("B4_free_care", ORANGE, "-", "free care"),
                                 ("B4_both", BLUE, (0, (3, 1.2)), "both")):
        if lab in labels:
            y = _mean_weekly(df, lab, "wk_belief_success") - ref
            ax.plot(x[keep], y[keep], color=color, ls=ls, lw=1.1 if lab != "B4_collapse" else 0.9, label=text)
    ax.axhline(0, color=INK2, lw=0.5)
    ax.set_xlim(-26, x[-1])
    ax.set_xticks(np.arange(-26, x[-1] + 1, 13))
    ax.set_xlabel("Weeks since supplies returned (from -26: the stock-out)")
    ax.set_ylabel("Mean belief, against no stock-out")
    lo = ax.get_ylim()[0]
    ax.set_ylim(lo, 0.25 * abs(lo))                   # room above zero (no series goes there) for the legend
    ax.legend(loc="upper right", ncols=4, borderaxespad=0.2)
    panel(ax, "c", "Restoring trust (weight 0.6)")

    # d: care sought per episode after supplies return, against no intervention (paired by seed, +- SE), while
    # free care lasts (weeks 0-13) and after it (13 to the year's end)
    ax = ax_d
    base = b4[b4["scenario"] == "B4_collapse"].set_index("seed")
    rows = [("B4_campaign", "Information campaign"), ("B4_free_care", "Three months of free care"), ("B4_both", "Both")]
    rows = [(sc, t) for sc, t in rows if sc in set(b4["scenario"])]
    y = np.arange(len(rows))[::-1].astype(float)
    for col, color, off, text in (("care_0_13w", ORANGE, 0.15, "weeks 0-13 (free care lasts)"),
                                  ("care_13w_end", BLUE, -0.15, "week 13 to the year's end")):
        d = [(b4[b4["scenario"] == sc].set_index("seed")[col] - base[col]).dropna() for sc, _ in rows]
        ax.errorbar([100 * v.mean() for v in d], y + off, xerr=[100 * v.sem() for v in d], fmt="o", color=color,
                    ms=3.2, mec="white", mew=0.5, elinewidth=0.7, capsize=0, label=text)
    ax.axvline(0, color=INK2, lw=0.5)
    ax.set_yticks(y, [t for _, t in rows])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.6, len(rows) - 0.1 + 0.6)
    _hgrid(ax, "x")
    ax.set_xlabel("Care sought per episode against\nno intervention (percentage points)")
    ax.legend(loc="upper right", borderaxespad=0.0)
    panel(ax, "d", "Care-seeking after supplies return")
    return save(fig, out_dir, "fig_1b_learning")


# ---------------------------------------------------------------------------
# Experiment 2: weather shocks and risk-management strategies
# ---------------------------------------------------------------------------

#: "economic adjustment" is Wren et al.'s (2025) name for a coping strategy (here: consumption cut by 20% while
#: earnings are low or prices high); the model leaves out its cost to nutrition, so the label only describes it
STRATEGY_NAMES = [("storage", "Storage (savings groups)"), ("exchange", "Exchange (LEAP, two lowest quintiles)"),
                  ("infrastructure", "Infrastructure (WASH, flood zone)"),
                  ("adjustment", "Economic adjustment (consumption cut)"), ("mobility", "Mobility (packaged water)")]
REGIME_COLS = [("C-hi", "\nStrong\ncycle"), ("F-slow", "Flood"), ("D-slow", "Drought"), ("C+D", "Cycle +\ndrought")]
BURDEN_COLS = [("D-slow", "episodes", 1, "Drought:\nepisodes"),
               ("D-slow", "care_per_episode", -1, "Drought:\ncare lost"),
               ("D-slow", "zero_cash", 1, "Drought:\nno cash"),
               ("D-slow", "poverty", 1, "Drought:\npoverty"),
               ("C+D", "care_per_episode", -1, "Cycle +\ndrought:\ncare lost"),
               ("F-slow", "fz_zero_cash", 1, "Flood\nzone:\nno cash")]
#: children's diarrhoea episodes: the three pathogens' onsets (child episode log)
EPISODE_KEYS = ("wk_episodes_rota", "wk_episodes_campy", "wk_episodes_crypto")
MONTH_DAYS = (0, 59, 120, 181, 243, 304)          # 1 Jan, Mar, May, Jul, Sep, Nov of the measured year
MONTH_NAMES = ("Jan", "Mar", "May", "Jul", "Sep", "Nov")


def _heatmap(ax, z, bold, xlabels, ylabels, cmap, lim, fmt, cbar_label):
    """Signed changes as coloured cells. Cells that meet the rule (`bold`) are drawn in full colour with bold
    numbers; the others faint (their colour 65% of the way to white) with grey numbers, so that the eye goes
    to the effects that are there."""
    norm = Normalize(vmin=-lim, vmax=lim)
    rgba = cmap(norm(np.nan_to_num(z)))
    faint = ~bold & np.isfinite(z)
    rgba[faint, :3] = 1.0 - 0.35 * (1.0 - rgba[faint, :3])
    rgba[~np.isfinite(z)] = (1.0, 1.0, 1.0, 0.0)
    ax.imshow(rgba, aspect="auto")
    im = ScalarMappable(norm=norm, cmap=cmap)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            if np.isfinite(z[i, j]):
                v = z[i, j]
                strong = bool(bold[i, j])
                ax.text(j, i, fmt.format(0.0 if abs(v) < 0.5 else v), ha="center", va="center", fontsize=6.5,
                        fontweight="bold" if strong else "normal",
                        color=("white" if abs(v) > 0.55 * lim else INK) if strong else INK2)
    ax.set_xticks(range(len(xlabels)), xlabels)
    ax.set_yticks(range(len(ylabels)), ylabels)
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-0.5, z.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, z.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.2)
    ax.tick_params(which="minor", length=0)
    cb = ax.figure.colorbar(im, ax=ax, shrink=0.9, aspect=18, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(width=0.4, length=2)
    cb.set_label(cbar_label, fontsize=6.5)


def _mean_weekly_centred(df, label, keys, k=4, seeds=None):
    """The mean over seeds of a weekly series (a key "a/b" is a ratio; several keys are summed), as a centred
    k-week rolling mean (for drawing; the analysis uses trailing means)."""
    from experiments.scenarios.analysis import weekly
    keys = (keys,) if isinstance(keys, str) else tuple(keys)
    g = df[df[LABEL] == label]
    if seeds is not None:
        g = g[g["seed"].isin(seeds)]
    xs = [sum(weekly(r, key) for key in keys) for _, r in g.iterrows()]
    m = np.nanmean(np.vstack(xs), axis=0)
    return pd.Series(m).rolling(k, center=True, min_periods=k).mean().to_numpy()


def fig_2(results_dir: str, out_dir: str) -> dict:
    from experiments.scenarios import hooks as H
    style()
    df = _results(results_dir, "2")
    labels = set(df[LABEL])
    an = _analysis_dir(results_dir, "2")
    lev = pd.read_csv(os.path.join(an, "strategies_level_2.csv"))
    rule = pd.read_csv(os.path.join(an, "strategies_2.csv"))
    runs = pd.read_csv(os.path.join(an, "resilience_runs_2.csv"))
    runs = runs[runs["variant"] == "main"]
    fig = plt.figure(figsize=(WIDTH, 136 * MM), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[0.82, 1.12])
    ax_a, ax_b, ax_c = top.subplots(1, 3)
    ax_d, ax_e = bottom.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.3]})
    t0 = H.SHOCK_START_DAY                                   # 1 January of the measured year
    w0 = int(t0 // 7)
    day = lambda w: np.asarray(w) * 7 + 3.5 - t0             # mid-week, days since 1 January
    rains = (H.DROUGHT_START_DAY + H.DROUGHT_DAYS, H.DROUGHT_START_DAY + H.DROUGHT_DAYS + H.FIRST_RAINS_DAYS)
    bands = {"drought": (H.DROUGHT_START_DAY, H.DROUGHT_START_DAY + H.DROUGHT_DAYS), "first rains": rains,
             "flood": (H.FLOOD_START_DAY, H.FLOOD_START_DAY + H.FLOOD_DAYS)}

    def frame(ax, which, ylabel, names=True):
        """The shaded periods (named inside the axes, at the bottom, where the series leave room), month ticks."""
        for name in which:
            a, b = bands[name]
            ax.axvspan(a - t0, b - t0, color=NEUTRAL if name != "first rains" else "#e7f1fb", lw=0, zorder=0)
            if names:
                ax.text((a + b) / 2 - t0, 0.02, name if name != "first rains" else "rains", fontsize=6,
                        color=INK2, transform=ax.get_xaxis_transform(), ha="center", va="bottom")
        ax.set_xticks(MONTH_DAYS, MONTH_NAMES)
        ax.set_xlim(0, 365)
        ax.set_xlabel("Month of the measured year")
        ax.set_ylabel(ylabel)

    def line(ax, scenario, keys, k=4, **kw):
        if scenario not in labels:
            return
        y = _mean_weekly_centred(df, scenario, keys, k)
        w = np.arange(len(y))
        keep = w >= w0
        ax.plot(day(w[keep]), y[keep], **kw)

    # a: children's diarrhoea episodes through the drought, with and without the environmental pathways
    ax = ax_a
    frame(ax, ("drought", "first rains"), "Children's episodes per week")
    line(ax, "R0_none", EPISODE_KEYS, color=INK2, lw=0.8, label="no shock", zorder=3)
    line(ax, "D-slow_none_noenv", EPISODE_KEYS, color=ORANGE, lw=1.0, ls=(0, (1.2, 1.2)),
         label="drought, no pathways", zorder=2)
    line(ax, "D-slow_none", EPISODE_KEYS, color=ORANGE, lw=1.1, label="drought, with pathways", zorder=4)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.07))
    panel(ax, "a", "Diarrhoea in children")

    # b: care sought per episode through a drought and a flood (and the drought with one day of credit)
    care_key = "wk_episodes_ended_with_care/wk_episodes_ended"
    ax = ax_b
    frame(ax, ("drought", "flood"), "Care sought per episode", names=False)
    line(ax, "R0_none", care_key, color=INK2, lw=0.8, label="no shock", zorder=4)
    line(ax, "F-slow_none", care_key, color=BLUE, lw=1.1, label="flood", zorder=3)
    line(ax, "D-slow_none", care_key, color=ORANGE, lw=1.1, label="drought", zorder=2)
    line(ax, "D-slow_none_credit1d", care_key, color=ORANGE, lw=1.0, ls=(0, (3, 1.2)),
         label="drought, one day of credit", zorder=1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower left")
    panel(ax, "b", "Care-seeking")

    # c: households without cash, town-wide and in the flood zone
    ax = ax_c
    frame(ax, ("drought", "flood"), "Households without cash")
    line(ax, "R0_none", "wk_zero_cash_share", color=INK2, lw=0.8, label="no shock", zorder=4)
    line(ax, "F-slow_none", "wk_zero_cash_share", color=BLUE, lw=1.1, label="flood: town", zorder=3)
    line(ax, "F-slow_none", "wk_fz_zero_cash_share", color=BLUE, lw=1.1, ls=(0, (3, 1.2)),
         label="flood: flood zone", zorder=2)
    line(ax, "D-slow_none", "wk_zero_cash_share", color=ORANGE, lw=1.1, label="drought: town", zorder=1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    panel(ax, "c", "Households without cash")

    # d: care sought from the regime's start, strategy minus no strategy under the same regime (points)
    names = [n for k, n in STRATEGY_NAMES]
    z = np.full((len(STRATEGY_NAMES), len(REGIME_COLS)), np.nan)
    bold = np.zeros_like(z, dtype=bool)
    for i, (st, _) in enumerate(STRATEGY_NAMES):
        for j, (rg, _) in enumerate(REGIME_COLS):
            r = lev[(lev["regime"] == rg) & (lev["strategy"] == st) & (lev["outcome"] == "care_per_episode")]
            if len(r):
                r = r.iloc[0]
                z[i, j] = 100 * r["diff"]
                need = int(np.ceil(7 * r["n"] / 8.0 - 1e-9))
                same = max(r["n_lower"], r["n"] - r["n_lower"]) >= need
                bold[i, j] = bool(abs(r["diff"]) > 2 * r["se"] and same)
    lim = max(16.0, float(np.ceil(np.nanmax(np.abs(z)) / 4) * 4)) if np.isfinite(z).any() else 16.0
    _heatmap(ax_d, z, bold, [n for _, n in REGIME_COLS], names, DIVERGING.reversed(), lim, "{:+.0f}",
             "Change in care sought per episode (points)")
    panel(ax_d, "d", "Care-seeking with each strategy")

    # e: how much of the burden a shock adds each strategy absorbs (% of that burden without a strategy)
    cols = [c for c in BURDEN_COLS if len(runs[(runs["outcome"] == c[1]) & (runs["regime"] == c[0])])]
    z = np.full((len(STRATEGY_NAMES), len(cols)), np.nan)
    bold = np.zeros_like(z, dtype=bool)
    for j, (rg, oc, sign, _) in enumerate(cols):
        sub = runs[(runs["outcome"] == oc) & (runs["regime"] == rg)]
        none = sign * sub[sub["strategy"] == "none"].set_index("seed")["excess"]
        for i, (st, _) in enumerate(STRATEGY_NAMES):
            x = sign * sub[sub["strategy"] == st].set_index("seed")["excess"]
            seeds = x.index.intersection(none.index)
            if len(seeds):
                z[i, j] = 100 * (x.loc[seeds].mean() - none.loc[seeds].mean()) / none.loc[seeds].mean()
            r = rule[(rule["regime"] == rg) & (rule["outcome"] == oc) & (rule["strategy"] == st)]
            bold[i, j] = bool(len(r) and (r["better"].iloc[0] or r["worse"].iloc[0]))
    _heatmap(ax_e, z, bold, [n for *_, n in cols], names, DIVERGING, 100, "{:+.0f}",
             "Change in the shock's burden (%)")
    ax_e.tick_params(axis="y", labelleft=False)
    panel(ax_e, "e", "Burden of the shock with each strategy")
    return save(fig, out_dir, "fig_2_shocks")


# ---------------------------------------------------------------------------
# Experiment 3A: what limits the rotavirus vaccine
# ---------------------------------------------------------------------------

#: the vaccine's four components, each made ideal in turn (coverage: the doses given; take: a dose given
#: protects; depth: protection against infection while it lasts; duration: how long it lasts)
COMPONENT_NAMES = [("vacc_coverage_ideal", "Coverage: every child given 3 doses"),
                   ("vacc_take_ideal", "Take: every dose takes"),
                   ("grid_v0.95_d616", "Depth: protection against infection 95%"),
                   ("grid_v0.55_dlife", "Duration: lifelong protection against infection"),
                   ("vacc_all_ideal", "All four together")]
SEQUENTIAL = LinearSegmentedColormap.from_list("blues", ["#f5f9fe", RAMP[100], RAMP[250], RAMP[450], RAMP[700]])


def fig_3a(results_dir: str, out_dir: str) -> dict:
    style()
    an = _analysis_dir(results_dir, "3a")
    grid = pd.read_csv(os.path.join(an, "grid_3a.csv"))
    comps = pd.read_csv(os.path.join(an, "components_3a.csv"))
    table = pd.read_csv(os.path.join(an, "table_3a.csv"), index_col=0)
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(WIDTH, 70 * MM), layout="constrained",
                                           gridspec_kw={"width_ratios": [1.0, 1.15, 1.05]})

    # a: fall in episodes at 0-24 months over the grid of protection against infection and its duration
    ax = ax_a
    durs = ["475", "616", "861", "life"]
    v_effs = sorted(grid["v_eff"].unique())
    z = np.full((len(v_effs), len(durs)), np.nan)
    flag = np.zeros_like(z, dtype=bool)
    for _, r in grid.iterrows():
        i, j = v_effs.index(r["v_eff"]), durs.index(str(r["duration"]))
        z[i, j] = 100 * r["fall_episodes_0_24m"]
        flag[i, j] = not bool(r["consistent_with_ghana"])
    im = ax.imshow(z, cmap=SEQUENTIAL, vmin=0, vmax=100, origin="lower", aspect="auto")
    for i in range(len(v_effs)):
        for j in range(len(durs)):
            ax.text(j, i, f"{z[i, j]:.0f}{'*' if flag[i, j] else ''}", ha="center", va="center", fontsize=6.5,
                    color="white" if z[i, j] > 58 else INK)
    n_plaus = sum(v <= 0.85 for v in v_effs)
    ax.add_patch(Rectangle((-0.5, -0.5), 3, n_plaus, fill=False, ec=INK, lw=0.7, ls=(0, (2.5, 1.5))))
    ax.add_patch(Rectangle((0.5, v_effs.index(0.55) - 0.5), 1, 1, fill=False, ec=ORANGE, lw=1.5))
    ax.set_xticks(range(len(durs)), ["15.6", "20.2", "28.3", "for life"])
    ax.set_yticks(range(len(v_effs)), [f"{v:.2f}" for v in v_effs])
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel("Mean duration of protection (months)")
    ax.set_ylabel("Protection against infection")
    cb = fig.colorbar(im, ax=ax, shrink=0.92, aspect=16, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(width=0.4, length=2)
    cb.set_label("Fall in episodes, 0-24 months (%)", fontsize=6.5)
    panel(ax, "a", "Protection and its duration")

    # b: the fall against today's schedule when one component is made ideal (rule: > 2 SE and > 10%)
    ax = ax_b
    sub = comps[comps["reference"] == "vacc_default"].set_index("scenario")
    rows = [(sc, name) for sc, name in COMPONENT_NAMES if sc in sub.index]
    y = np.arange(len(rows))[::-1]
    share = np.array([100 * sub.loc[sc, "reduction_share"] for sc, _ in rows])
    se = np.array([100 * sub.loc[sc, "se"] * sub.loc[sc, "reduction_share"] / sub.loc[sc, "reduction"]
                   for sc, _ in rows])
    meets = np.array([bool(sub.loc[sc, "limits"]) for sc, _ in rows])
    ax.barh(y, share, color=[BLUE if m else LIGHT_BLUE for m in meets], height=0.6)
    ax.errorbar(share, y, xerr=se, fmt="none", ecolor=INK, elinewidth=0.6, capsize=0)
    for yy, v, e in zip(y, share, se):
        ax.text(v + e + 1.5, yy, f"{v:.0f}%", va="center", fontsize=6.5)
    ax.axvline(10, color=INK2, lw=0.6, ls=(0, (1, 1.2)))
    ax.set_yticks(y, [n for _, n in rows])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Fall in episodes at 0-24 months\nagainst today's schedule (%)")
    ax.spines["left"].set_visible(False)
    ax.text(10, 1.0, " 10%", transform=ax.get_xaxis_transform(), ha="left", va="top", fontsize=6, color=INK2)
    panel(ax, "b", "Making one component ideal")

    # c: where today's effect comes from: the protection against illness switched off part by part, and
    # (the sensitivity of the revision) made to wane over four years (mean) instead of lasting for life
    ax = ax_c
    variants = [("vacc_default", "Today's schedule"), ("vacc_no_prior_infection_equivalent", "No extra prior\ninfection"),
                ("vacc_no_severity_reduction", "No cut in severity"),
                ("vacc_illness_wanes_4y", "Protection against\nillness wanes (4 years)")]
    variants = [(sc, lab) for sc, lab in variants if sc in table.index]
    y = np.arange(len(variants))[::-1].astype(float)
    hgt = 0.36
    for k, (metric, color, lab) in enumerate((("episodes_0_24m", BLUE, "episodes, 0-24 months"),
                                              ("severe_0_59m", ORANGE, "severe episodes, 0-59 months"))):
        vals = np.array([-100 * table.loc[sc, f"{metric}_vs_none"] for sc, _ in variants])
        errs = np.array([100 * table.loc[sc, f"{metric}_vs_none_se"] for sc, _ in variants])
        ys = y + (0.5 - k) * hgt
        ax.barh(ys, vals, height=hgt * 0.88, color=color, label=lab)
        ax.errorbar(vals, ys, xerr=errs, fmt="none", ecolor=INK, elinewidth=0.6)
        for yy, v, e in zip(ys, vals, errs):
            ax.text(v + e + 1.5, yy, f"{v:.0f}", va="center", fontsize=6.5)
    ax.set_yticks(y, [lab for _, lab in variants])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, len(variants) + 0.35)          # room for the legend above the first group
    ax.set_xlabel("Fall against\nno vaccination (%)")
    ax.legend(loc="upper right", borderaxespad=0.0)
    panel(ax, "c", "Where today's effect comes from")
    return save(fig, out_dir, "fig_3a_vaccine_limits")


# ---------------------------------------------------------------------------
# Experiment 3B: interventions on the vaccine
# ---------------------------------------------------------------------------

INTERVENTION_NAMES = [("neonatal", "First dose at birth"), ("neonatal_take_ideal", "First dose at birth, every dose takes"),
                      ("timely", "Every dose on time"), ("booster_9m", "Booster at 9 months (upper bound)"),
                      ("booster_9m_wanes_4y", "Booster, if protection against illness wanes (4 y)"),
                      ("opv_separate", "OPV at a separate visit"),
                      ("wash_q12", "WASH for the two lowest quintiles"),
                      ("takewash_wash_q12", "WASH, with its effect on take")]
QUINTILE_SERIES = [("neonatal_take_ideal", "First dose at birth, every dose takes", BLUE),
                   ("booster_9m", "Booster at 9 months", ORANGE),
                   ("wash_q12", "WASH for the two lowest quintiles", "#1baf7a")]


def fig_3b(results_dir: str, out_dir: str) -> dict:
    from experiments.scenarios.analysis import paired
    style()
    df = _results(results_dir, "3b")
    t = pd.read_csv(os.path.join(_analysis_dir(results_dir, "3b"), "table_3b.csv")).set_index("scenario")
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(WIDTH, 66 * MM), layout="constrained",
                                     gridspec_kw={"width_ratios": [1.35, 1]})

    # a: further fall in episodes (0-24 months) and severe episodes (0-59 months) against today's schedule
    ax = ax_a
    rows = [(sc, n) for sc, n in INTERVENTION_NAMES if sc in t.index]
    y = np.arange(len(rows))[::-1].astype(float)
    v = np.array([-100 * t.loc[sc, "rel"] for sc, _ in rows])
    e = np.array([100 * t.loc[sc, "se"] / (t.loc[sc, "episodes_0_24m"] - t.loc[sc, "diff"]) for sc, _ in rows])
    helps = np.array([bool(t.loc[sc, "helps"]) for sc, _ in rows])
    sv = np.array([-100 * t.loc[sc, "severe_0_59m_rel"] for sc, _ in rows])
    se_ = np.array([100 * t.loc[sc, "severe_0_59m_rel_se"] for sc, _ in rows])
    ax.axvline(0, color=INK2, lw=0.5)
    ax.errorbar(sv, y - 0.17, xerr=se_, fmt="o", color=ORANGE, ms=3.4, mec="white", mew=0.5, elinewidth=0.7, capsize=0)
    ax.errorbar(v, y + 0.17, xerr=e, fmt="none", ecolor=BLUE, elinewidth=0.7, capsize=0)
    ax.plot(v[helps], y[helps] + 0.17, "o", color=BLUE, ms=3.6, mec="white", mew=0.5)
    ax.plot(v[~helps], y[~helps] + 0.17, "o", color=BLUE, ms=3.4, mfc="white", mec=BLUE, mew=0.8)
    ax.set_yticks(y, [n for _, n in rows])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-10, 100)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    _hgrid(ax, "x")
    ax.set_xlabel("Fall against today's schedule (%)")
    handles = [Line2D([], [], color=BLUE, marker="o", ls="", ms=3.6, mec="white", label="episodes, 0-24 months: helps"),
               Line2D([], [], color=BLUE, marker="o", ls="", ms=3.4, mfc="white", mec=BLUE,
                      label="episodes, 0-24 months: does not"),
               Line2D([], [], color=ORANGE, marker="o", ls="", ms=3.4, mec="white", label="severe episodes, 0-59 months")]
    ax.legend(handles=handles, loc="lower right")
    panel(ax, "a", "Interventions")

    # b: fall in rotavirus episodes under five, by wealth quintile, for the three that help (+- SE)
    ax = ax_b
    ax.axhline(0, color=INK2, lw=0.5)
    for k, (sc, name, color) in enumerate(QUINTILE_SERIES):
        vals, errs = [], []
        for q in range(1, 6):
            r = paired(df, f"exp_rota_episodes_per_child_year_q{q}", sc, "vacc_default")
            vals.append(-100 * r["diff"] / r["reference"])
            errs.append(100 * r["se"] / r["reference"])
        xs = np.arange(1, 6) + (k - 1) * 0.2
        ax.errorbar(xs, vals, yerr=errs, fmt="o", color=color, ms=3.4, mec="white", mew=0.5, elinewidth=0.7,
                    capsize=0, label=name)
    ax.set_xticks(range(1, 6), ["1\n(lowest)", "2", "3", "4", "5\n(highest)"])
    ax.set_xlabel("Wealth quintile")
    ax.set_ylabel("Fall in rotavirus episodes under five (%)")
    ax.set_ylim(-25, 52)
    ax.set_yticks(np.arange(-20, 41, 10))
    _hgrid(ax, "y")
    ax.legend(loc="upper left")
    panel(ax, "b", "By wealth quintile")
    return save(fig, out_dir, "fig_3b_vaccine_interventions")


# ---------------------------------------------------------------------------
# Step 3 of the revision: the headline effects over calibrated parameter sets
# ---------------------------------------------------------------------------

#: step 3's contrasts (experiments.scenarios.param_sets.CONTRASTS), short labels for the figure, grouped by
#: experiment in the order of the paper
PS_GROUPS = [
    ("1A Care-seeking", [("free_care", "All care free"), ("free_ors", "Free ORS and zinc"),
                         ("leap", "LEAP grants (lowest quintile)"), ("credit", "One day of credit"),
                         ("perceived", "Care perceived to work 0.2")]),
    ("1B Trust", [("stockout_belief", "Stock-out: trust at its end"), ("stockout_care", "Stock-out: care sought"),
                  ("campaign", "Campaign: trust 13-16 weeks on")]),
    ("2 Shocks", [("drought_cash", "Drought: no cash"), ("drought_care", "Drought: care sought"),
                  ("drought_poverty", "Drought: poverty at year end"), ("flood_cash", "Flood: no cash, flood zone"),
                  ("adjustment_cash", "Adjustment: drought's no cash"),
                  ("adjustment_care", "Adjustment: drought's care"),
                  ("pathways", "Pathways: children's episodes")]),
    ("3A Vaccine", [("vacc_episodes", "Vaccine: episodes"), ("vacc_severe", "Vaccine: severe episodes"),
                    ("depth", "Protection against infection 0.95"),
                    ("illness_protection", "No extra prior infection")]),
    ("3B Vaccine interventions", [("neonatal", "Dose at birth: severe episodes"), ("booster", "Booster at 9 months"),
                                  ("wash", "WASH, two lowest quintiles")]),
]


def plot_param_sets(by_set: pd.DataFrame, verdict: pd.DataFrame, out_dir: str, adopted: str = "e7") -> dict:
    """Each headline effect that is beyond 2 SE at the adopted set (step 2, all its seeds), in every parameter
    set, as a multiple of the adopted set's effect: a dot per set (8 paired seeds, +-2 SE), filled where the
    effect keeps its sign beyond 2 SE; the adopted set's own first 8 seeds in orange; on the right, the share of
    sets doing so, in bold where the rule calls it robust (at least 80%). Effects not beyond 2 SE at the adopted
    set are not tested and not drawn."""
    style()
    v_by = verdict.set_index("contrast")
    rows, ylabels, group_rows = [], [], []
    y = 0.0
    for g, items in PS_GROUPS:
        items = [(k, t) for k, t in items if k in v_by.index and bool(v_by.loc[k, "tested"])]
        if not items:
            continue
        group_rows.append((g, y))
        y -= 0.85
        for k, t in items:
            rows.append((k, y))
            ylabels.append((y, t))
            y -= 1.0
        y -= 0.4
    fig, ax = plt.subplots(figsize=(WIDTH, (14 + 4.2 * (-y)) * MM), layout="constrained")
    lo, hi = 0.0, 2.0
    for k, yy in rows:
        v = v_by.loc[k]
        g = by_set[by_set["contrast"] == k]
        scale = float(v["adopted_diff"])
        x = g["diff"].to_numpy() / scale
        e = 2 * g["se"].to_numpy() / abs(scale)
        own = (g["set"] == adopted).to_numpy()
        meets = g["meets"].to_numpy(dtype=bool)
        n_other = int((~own).sum())
        jit = np.zeros(len(g))
        jit[~own] = np.linspace(-0.3, 0.3, n_other) if n_other > 1 else 0.0
        ys = yy + jit
        lo, hi = min(lo, float(np.nanmin(x - e))), max(hi, float(np.nanmax(x + e)))
        for m, mfc in ((meets & ~own, BLUE), (~meets & ~own, "white")):
            ax.errorbar(x[m], ys[m], xerr=e[m], fmt="o", ms=2.4, color=BLUE, mfc=mfc, mec=BLUE, mew=0.6,
                        elinewidth=0.45, capsize=0, zorder=3)
        ax.errorbar(x[own], ys[own], xerr=e[own], fmt="D", ms=2.8, color=ORANGE, mec="white", mew=0.4,
                    elinewidth=0.6, capsize=0, zorder=4)
        txt = f"{v['share_meeting']:.0%} of {int(v['sets'])}" + ("  robust" if v["robust"] else "")
        ax.text(1.01, yy, txt, transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=6.5,
                fontweight="bold" if v["robust"] else "normal", color=INK if v["robust"] else INK2)
    ax.axvline(0, color=INK2, lw=0.5)
    ax.axvline(1, color=HAIR, lw=0.8, zorder=0)
    ax.set_yticks([yy for yy, _ in ylabels], [t for _, t in ylabels])
    ax.tick_params(axis="y", length=0)
    for g, yy in group_rows:
        ax.text(-0.005, yy, g, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=6.5,
                fontweight="bold")
    ax.set_ylim(y + 0.7, 0.45)
    ax.set_xlim(np.floor(lo * 2) / 2 - 0.1, np.ceil(hi * 2) / 2 + 0.1)
    ax.set_xlabel("Effect in the parameter set, as a multiple of the adopted set's effect (±2 SE)")
    _hgrid(ax, "x")
    ax.spines["left"].set_visible(False)
    handles = [Line2D([], [], color=ORANGE, marker="D", ls="", ms=2.8, mec="white", label="adopted set, its first 8 seeds"),
               Line2D([], [], color=BLUE, marker="o", ls="", ms=2.4, label="same sign, beyond 2 SE"),
               Line2D([], [], color=BLUE, marker="o", ls="", ms=2.4, mfc="white", label="not")]
    ax.legend(handles=handles, loc="lower right", ncols=3, bbox_to_anchor=(1.0, 1.0), borderaxespad=0.1)
    return save(fig, out_dir, "fig_param_sets")


PARTS = {"1a": fig_1a, "1b": fig_1b, "2": fig_2, "3a": fig_3a, "3b": fig_3b}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("parts", nargs="*", default=None)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    for part in a.parts or sorted(PARTS):
        print(part, PARTS[part](a.results, a.out))


if __name__ == "__main__":
    main()
