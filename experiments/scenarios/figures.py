# experiments/scenarios/figures.py
"""
Figures for the v5 experiments, from the analysis outputs
(experiments.scenarios.analysis). One function per experiment part; each
writes a PNG (and a PDF for the manuscript) into the part's analysis folder.

    python -m experiments.scenarios.figures 3a [--analysis <dir>]

Colours: one validated categorical pair (blue #2a78d6, orange #eb6834; dataviz
validator, light surface), a sequential blue ramp for magnitudes, text in ink
tokens; thin marks, recessive grid, direct labels where few.
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e2dd", "#fcfcfb"
OUT_ROOT = os.path.join("experiments", "outputs", "experiments_v5")


def _style():
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})


def _save(fig, out_dir: str, name: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, f"{name}.png")
    fig.savefig(png, dpi=200)
    fig.savefig(os.path.join(out_dir, f"{name}.pdf"))
    plt.close(fig)
    return png


# ---------------------------------------------------------------------------
# 3A: what limits the vaccine
# ---------------------------------------------------------------------------

COMPONENT_LABELS = {
    "vacc_coverage_ideal": "every child given 3 doses",
    "vacc_take_ideal": "every dose takes",
    "grid_v0.95_d616": "95% protection against infection",
    "grid_v0.55_dlife": "lifelong protection against infection",
    "vacc_all_ideal": "all four together",
}


def fig_3a(analysis_dir: str) -> str:
    _style()
    grid = pd.read_csv(os.path.join(analysis_dir, "grid_3a.csv"))
    comps = pd.read_csv(os.path.join(analysis_dir, "components_3a.csv"))
    table = pd.read_csv(os.path.join(analysis_dir, "table_3a.csv"), index_col=0)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3), layout="constrained",
                             gridspec_kw={"width_ratios": [1.05, 1.15, 0.95]})

    # A: the fall in episodes at 0-24 months over the protection grid
    ax = axes[0]
    durs = ["475", "616", "861", "life"]
    v_effs = sorted(grid["v_eff"].unique())
    z = np.full((len(v_effs), len(durs)), np.nan)
    flag = np.zeros_like(z, dtype=bool)
    for _, r in grid.iterrows():
        i, j = v_effs.index(r["v_eff"]), durs.index(str(r["duration"]))
        z[i, j] = 100 * r["fall_episodes_0_24m"]
        flag[i, j] = not bool(r["consistent_with_ghana"])
    im = ax.imshow(z, cmap="Blues", vmin=0, vmax=100, origin="lower", aspect="auto")
    for i in range(len(v_effs)):
        for j in range(len(durs)):
            if np.isfinite(z[i, j]):
                ax.text(j, i, f"{z[i, j]:.0f}{'*' if flag[i, j] else ''}", ha="center", va="center", fontsize=8,
                        color="white" if z[i, j] > 55 else INK)
    n_plaus = sum(v <= 0.85 for v in v_effs)
    ax.add_patch(plt.Rectangle((-0.5, -0.5), 3, n_plaus, fill=False, ec=INK2, lw=1, ls="--"))
    ax.add_patch(plt.Rectangle((0.5, v_effs.index(0.55) - 0.5), 1, 1, fill=False, ec=ORANGE, lw=2))
    ax.set_xticks(range(len(durs)), ["15.6", "20.2", "28.3", "lifelong"])
    ax.set_yticks(range(len(v_effs)), [f"{v:.2f}" for v in v_effs])
    ax.set_xlabel("Mean duration of protection (months)")
    ax.set_ylabel("Protection against infection while protected")
    ax.set_title("A  Fall in rotavirus episodes, 0-24 months (%)", loc="left", fontsize=10, color=INK)
    ax.text(0.0, -0.2, "orange: today's values; dashed: plausible range;\n* outside Ghana's observed impact",
            transform=ax.transAxes, fontsize=7.5, color=INK2, va="top")
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK2)

    # B: what making each component ideal adds at the default
    ax = axes[1]
    sub = comps[comps["reference"] == "vacc_default"].copy()
    sub["label"] = sub["scenario"].map(COMPONENT_LABELS)
    sub = sub.dropna(subset=["label"])
    y = np.arange(len(sub))[::-1]
    share = 100 * sub["reduction_share"].to_numpy()
    se = 100 * (sub["se"] / (sub["reduction"] / sub["reduction_share"])).to_numpy()
    meets = sub["limits"].to_numpy(dtype=bool)
    light = "#a9c8ee"
    ax.barh(y, share, color=[BLUE if m else light for m in meets], height=0.55)
    ax.errorbar(share, y, xerr=se, fmt="none", ecolor=INK2, elinewidth=1, capsize=0)
    for yy, v, e in zip(y, share, se):
        ax.text(v + e + 1.5, yy, f"{v:.0f}%", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, sub["label"])
    ax.axvline(10, color=INK2, lw=0.8, ls=":")
    ax.set_xlim(0, max(share + se) * 1.15)
    ax.set_xlabel("Further fall in episodes, 0-24 months (% of today's)")
    ax.set_title("B  Making one part of the vaccine ideal", loc="left", fontsize=10, color=INK)
    ax.grid(axis="x", color=GRID, lw=0.6)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="meets the rule (> 2 SE and > 10%)"),
                       Patch(color=light, label="does not")], loc="upper right", frameon=False, fontsize=7.5)

    # C: where today's effect comes from (structural parts of the protection against illness)
    ax = axes[2]
    variants = [("vacc_default", "today's\nschedule"),
                ("vacc_no_prior_infection_equivalent", "without the\nextra prior\ninfection"),
                ("vacc_no_severity_reduction", "without the\n40% cut in\nseverity")]
    x = np.arange(len(variants))
    wdt = 0.36
    for k, (metric, label, color) in enumerate((("episodes_0_24m", "all episodes, 0-24 months", BLUE),
                                                 ("severe_0_59m", "severe episodes, 0-59 months", ORANGE))):
        vals = np.array([-100 * table.loc[sc, f"{metric}_vs_none"] for sc, _ in variants])
        errs = np.array([100 * table.loc[sc, f"{metric}_vs_none_se"] for sc, _ in variants])
        xs = x + (k - 0.5) * wdt
        ax.bar(xs, vals, width=wdt * 0.92, color=color, label=label)
        ax.errorbar(xs, vals, yerr=errs, fmt="none", ecolor=INK2, elinewidth=1)
        for xx, v, e in zip(xs, vals, errs):
            ax.text(xx, v + e + 1.5, f"{v:.0f}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x, [lab for _, lab in variants])
    ax.set_ylim(0, 85)
    ax.set_ylabel("Fall against no vaccination (%)")
    ax.set_title("C  Where today's effect comes from", loc="left", fontsize=10, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(loc="upper right", frameon=False, fontsize=7.5)
    return _save(fig, analysis_dir, "fig_3a_vaccine_limits")


# ---------------------------------------------------------------------------
# 1A: the edges of care-seeking and the policies that cross them
# ---------------------------------------------------------------------------

POLICY_LABELS = [
    ("price_free", "all care free"),
    ("free_ors_zinc", "free ORS and zinc (drug sellers)"),
    ("free_primary_insured", "free primary care with NHIS"),
    ("nhis_universal", "NHIS for everyone"),
    ("nhis_universal_free_primary", "NHIS for everyone, primary care free"),
    ("leap_q1_x1", "LEAP grants to the lowest quintile"),
    ("leap_q1_x2", "  ... twice the grant"),
    ("leap_q12_x1", "LEAP grants to the two lowest quintiles"),
    ("leap_q12_x2", "  ... twice the grant"),
    ("compensate_visit_time", "earnings lost to a visit compensated"),
    ("compensate_caring_time", "earnings lost to caring compensated"),
]
CREDIT_SHADES = {0.0: "#9ec1ec", 1.0: "#5f9be2", 3.0: "#2a78d6", 10.0: "#174d91"}


def _level(df, label, metric):
    v = df.loc[df["gsa.design_id"] == label, metric]
    return float(v.mean()), float(v.sem())


def fig_1a(analysis_dir: str, results: list[str]) -> str:
    from experiments.scenarios.analysis import paired
    _style()
    df = pd.concat([pd.read_parquet(r) for r in results], ignore_index=True)
    m, q1 = "econ_episode_care_seeking_rate", "econ_care_seeking_rate_q1"
    labels = set(df["gsa.design_id"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), layout="constrained", gridspec_kw={"width_ratios": [1.1, 0.9, 1.25]})

    # A: price of care, with emergency credit
    ax = axes[0]
    mults = [0.0, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.1, 0.5, 1.0, 2.0]
    name = lambda x: "baseline" if x == 1.0 else ("price_free" if x == 0 else f"price_x{x:g}")
    pos = {x: i for i, x in enumerate(mults)}
    for days, shade in CREDIT_SHADES.items():
        pts = []
        for x in mults:
            lab = name(x) if days == 0 else (f"credit_{days:g}d_price_x{x:g}" if x >= 0.1 else None)
            if lab in labels:
                pts.append((pos[x], *_level(df, lab, m)))
        if pts:
            xs, ys, es = zip(*pts)
            ax.errorbar(xs, ys, yerr=es, color=shade, lw=2, marker="o", ms=5, mec=SURF, capsize=0,
                        label="no credit" if days == 0 else
                        f"credit up to {days:g} day{'' if days == 1 else 's'} of consumption")
    ax.set_xticks(range(len(mults)), ["free", ".001", ".0025", ".005", ".0075", ".01", ".1", ".5", "1\n(today)", "2"])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Out-of-pocket price, multiple of today's")
    ax.set_ylabel("Care sought, share of episodes")
    ax.set_title("A  Price of care, with and without credit", loc="left", fontsize=10, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(loc="lower left", frameon=False, fontsize=7.5)

    # B: efficacy of care, as perceived (actual 0.8) and actual (perceived 0.8)
    ax = axes[1]
    for prefix, levels, color, lab in (("perceived_", (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.6, 0.8, 1.0), BLUE,
                                        "perceived (care works with 0.8)"),
                                       ("actual_", (0.4, 0.6, 0.8, 0.95), ORANGE, "actual (perceived as 0.8)")):
        pts = []
        for p in levels:
            l = "baseline" if p == 0.8 else f"{prefix}{p:g}"
            if l in labels:
                pts.append((p, *_level(df, l, m)))
        if pts:
            xs, ys, es = zip(*pts)
            ax.errorbar(xs, ys, yerr=es, color=color, lw=2, marker="o", ms=5, mec=SURF, capsize=0, label=lab)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Probability that care works")
    ax.set_title("B  Efficacy of care", loc="left", fontsize=10, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(loc="lower right", frameon=False, fontsize=7.5)

    # C: policies, change in care-seeking against today (all, and the lowest quintile)
    ax = axes[2]
    pol = [(l, t) for l, t in POLICY_LABELS if l in labels]
    y = np.arange(len(pol))[::-1]
    for k, (metric, color, lab) in enumerate(((m, BLUE, "all households"), (q1, ORANGE, "lowest quintile"))):
        d = [paired(df, metric, l, "baseline") for l, _ in pol]
        xs = np.array([100 * r["diff"] for r in d])
        es = np.array([100 * r["se"] for r in d])
        ax.errorbar(xs, y + (0.17 if k == 0 else -0.17), xerr=es, fmt="o", color=color, ms=5, mec=SURF,
                    elinewidth=1, capsize=0, label=lab)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y, [t for _, t in pol])
    ax.set_xlabel("Change in care sought per episode (percentage points)")
    ax.set_title("C  Policies, against today", loc="left", fontsize=10, color=INK)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.legend(loc="lower right", frameon=False, fontsize=7.5)
    base, base_q1 = _level(df, "baseline", m)[0], _level(df, "baseline", q1)[0]
    ax.text(0.99, 1.01, f"today: {base:.2f} (lowest quintile {base_q1:.2f})", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7.5, color=INK2)
    return _save(fig, analysis_dir, "fig_1a_care_levers")


# ---------------------------------------------------------------------------
# 2: shocks and strategies
# ---------------------------------------------------------------------------

REGIME_LABELS = {"R0": "no shock", "C-lo": "seasonal cycle, mild", "C-hi": "seasonal cycle, strong",
                 "C-rare": "strong cycle, one year in three", "F-fast": "flood, fast recovery",
                 "F-slow": "flood, slow recovery", "D-fast": "drought, fast recovery",
                 "D-slow": "drought, slow recovery", "C+F": "strong cycle + flood", "C+D": "strong cycle + drought"}
STRATEGY_LABELS = {"storage": "storage (savings groups)", "exchange": "exchange (LEAP, two lowest quintiles)",
                   "infrastructure": "infrastructure (WASH in the flood zone)",
                   "adjustment": "economic adjustment (eat less)", "mobility": "mobility (switch water source)"}


def _rolling_mean_series(df, label, key, k=4):
    from experiments.scenarios.analysis import rolling, weekly
    xs = [rolling(weekly(r, key), k) for _, r in df[df["gsa.design_id"] == label].iterrows()]
    return np.nanmean(np.vstack(xs), axis=0) if xs else None


SHORT_REGIME = {"C-hi": "strong\ncycle", "F-slow": "flood\n(slow)", "D-slow": "drought\n(slow)",
                "C+D": "cycle +\ndrought", "C+F": "cycle +\nflood",
                "D-slow|care": "drought:\ncare\nlost", "D-slow|cash": "drought:\nno\ncash",
                "D-slow|poverty": "drought:\npoverty", "C+D|care": "cycle +\ndrought:\ncare lost",
                "F-slow|fzcash": "flood:\nno cash,\nflood zone"}


def _strategy_heatmap(ax, fig, z, regimes, strategies, title, fmt="{:+.0f}", cmap="RdBu", lim=None):
    lim = lim or (np.nanmax(np.abs(z)) if np.isfinite(z).any() else 1)
    im = ax.imshow(z, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            if np.isfinite(z[i, j]):
                ax.text(j, i, fmt.format(z[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if abs(z[i, j]) > 0.6 * lim else INK)
    ax.set_xticks(range(len(regimes)), [SHORT_REGIME.get(r, r) for r in regimes])
    ax.set_yticks(range(len(strategies)), [STRATEGY_LABELS[s] for s in strategies])
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    cb = ax.figure.colorbar(im, ax=ax, shrink=0.85)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK2)


def fig_2(analysis_dir: str, results: list[str]) -> str:
    from experiments.scenarios import hooks as H
    _style()
    df = pd.concat([pd.read_parquet(r) for r in results], ignore_index=True)
    lev = pd.read_csv(os.path.join(analysis_dir, "strategies_level_2.csv"))
    runs = pd.read_csv(os.path.join(analysis_dir, "resilience_runs_2.csv"))
    fig = plt.figure(figsize=(13.5, 9.4), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, 1.05])
    axes = np.array([top.subplots(1, 2), bottom.subplots(1, 2)])
    years = lambda w: w * 7 / 365.0
    wk0 = 104
    shade = lambda ax: [ax.axvspan(years(a / 7), years(b / 7), color=GRID, lw=0) for a, b in
                        ((H.DROUGHT_START_DAY, H.DROUGHT_START_DAY + H.DROUGHT_DAYS),
                         (H.FLOOD_START_DAY, H.FLOOD_START_DAY + H.FLOOD_DAYS))]
    xlabel = "Years since the start of the run (shaded: the drought, then the flood)"

    # A: care-seeking through the shocks (no strategy)
    ax = axes[0, 0]
    shade(ax)
    for label, color, lw in (("R0_none", INK2, 1.2), ("D-slow_none", ORANGE, 2), ("F-slow_none", BLUE, 2)):
        y = _rolling_mean_series(df, label, "wk_episodes_ended_with_care/wk_episodes_ended")
        if y is not None:
            t = years(np.arange(len(y)))
            ax.plot(t[wk0:], y[wk0:], color=color, lw=lw, label=REGIME_LABELS[label.split("_")[0]])
    ax.set_ylim(0, 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Care sought per episode (4-week mean of 8 seeds)")
    ax.set_title("A  Care-seeking through a drought and a flood", loc="left", fontsize=10, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(loc="lower left", frameon=False, fontsize=7.5)

    # B: households without cash, town and flood zone
    ax = axes[0, 1]
    shade(ax)
    for label, key, color, ls, lab in (("D-slow_none", "wk_zero_cash_share", ORANGE, "-", "drought, slow recovery: town"),
                                       ("F-slow_none", "wk_zero_cash_share", BLUE, "-", "flood, slow recovery: town"),
                                       ("F-slow_none", "wk_fz_zero_cash_share", BLUE, "--",
                                        "flood, slow recovery: flood zone"),
                                       ("R0_none", "wk_zero_cash_share", INK2, "-", "no shock: town")):
        y = _rolling_mean_series(df, label, key)
        if y is not None:
            t = years(np.arange(len(y)))
            ax.plot(t[wk0:], y[wk0:], color=color, ls=ls, lw=1.2 if label == "R0_none" else 2, label=lab)
    ax.set_ylim(0, 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Households without cash (4-week mean of 8 seeds)")
    ax.set_title("B  Households without cash", loc="left", fontsize=10, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(loc="upper left", frameon=False, fontsize=7.5)

    # C: strategies against no strategy under the same regime: care-seeking from the shocks' start
    regimes = [r for r in ("C-hi", "F-slow", "D-slow", "C+D") if r in set(lev["regime"])]
    strategies = [s for s in STRATEGY_LABELS if s in set(lev["strategy"])]
    z = np.full((len(strategies), len(regimes)), np.nan)
    for i, st in enumerate(strategies):
        for j, rg in enumerate(regimes):
            row = lev[(lev["regime"] == rg) & (lev["strategy"] == st) & (lev["outcome"] == "care_per_episode")]
            if len(row):
                z[i, j] = 100 * row["diff"].iloc[0]
    _strategy_heatmap(axes[1, 0], fig, z, regimes, strategies,
                      "C  Care sought from year 3 on: strategy minus none (points)")

    # D: resilience: how much a strategy changes the burden the shock adds (against the same strategy without
    # the shock), in % of that burden without a strategy; negative = the strategy absorbs part of the shock.
    # Only burdens the shock adds by more than 2 SE without a strategy are shown.
    cols = [("D-slow", "care_per_episode", -1, "D-slow|care"), ("D-slow", "zero_cash", 1, "D-slow|cash"),
            ("D-slow", "poverty", 1, "D-slow|poverty"), ("C+D", "care_per_episode", -1, "C+D|care"),
            ("F-slow", "fz_zero_cash", 1, "F-slow|fzcash")]
    shown, zd = [], []
    for rg, oc, sign, key in cols:
        sub = runs[(runs["outcome"] == oc) & (runs["regime"] == rg)]
        none = sign * sub[sub["strategy"] == "none"].set_index("seed")["excess"]
        if len(none) < 2 or not (none.mean() > 2 * none.std(ddof=1) / np.sqrt(len(none))):
            continue
        colv = []
        for st in strategies:
            x = sign * sub[sub["strategy"] == st].set_index("seed")["excess"]
            seeds = x.index.intersection(none.index)
            colv.append(100 * (x.loc[seeds].mean() - none.loc[seeds].mean()) / none.loc[seeds].mean()
                        if len(seeds) else np.nan)
        shown.append(key)
        zd.append(colv)
    zd = np.array(zd, dtype=float).T if zd else np.full((len(strategies), 0), np.nan)
    _strategy_heatmap(axes[1, 1], fig, zd, shown, strategies,
                      "D  Burden the shock adds: change with the strategy (%)", cmap="RdBu_r", lim=100)
    axes[1, 1].text(0.0, -0.17, "negative: the strategy absorbs part of the shock's burden", transform=axes[1, 1].transAxes,
                    fontsize=7.5, color=INK2, va="top")
    return _save(fig, analysis_dir, "fig_2_shocks")


# ---------------------------------------------------------------------------
# 3B: vaccine interventions
# ---------------------------------------------------------------------------

INTERVENTION_LABELS = [
    ("neonatal", "first dose at birth (0-5 days)"),
    ("neonatal_take_ideal", "  ... and every dose takes"),
    ("timely", "every dose on time (6, 10, 14 weeks)"),
    ("booster_9m", "booster at 9 months (upper bound)"),
    ("opv_separate", "OPV at a separate visit (take x1.3)"),
    ("wash_q12", "water and sanitation, two lowest quintiles"),
    ("takewash_wash_q12", "  ... with SHINE's effect on take (D4)"),
]


def fig_3b(analysis_dir: str) -> str:
    _style()
    t = pd.read_csv(os.path.join(analysis_dir, "table_3b.csv"))
    rows = [(sc, lab) for sc, lab in INTERVENTION_LABELS if sc in set(t["scenario"])]
    t = t.set_index("scenario").loc[[sc for sc, _ in rows]]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.4), layout="constrained", gridspec_kw={"width_ratios": [1.25, 1]})

    # A: change in rotavirus illness against today's schedule (D4 against its own reference)
    ax = axes[0]
    y = np.arange(len(rows))[::-1]
    v = -100 * t["rel"].to_numpy()
    e = 100 * (t["se"] / (t["episodes_0_24m"] - t["diff"])).to_numpy()
    helps = t["helps"].to_numpy(dtype=bool)
    light = "#a9c8ee"
    ax.barh(y + 0.18, v, height=0.34, color=[BLUE if h else light for h in helps], label="all episodes, 0-24 months")
    ax.errorbar(v, y + 0.18, xerr=e, fmt="none", ecolor=INK2, elinewidth=1)
    sv = -100 * t["severe_0_59m_rel"].to_numpy()
    se_ = 100 * t["severe_0_59m_rel_se"].to_numpy()
    ax.barh(y - 0.18, sv, height=0.34, color=ORANGE, label="severe episodes, 0-59 months")
    ax.errorbar(sv, y - 0.18, xerr=se_, fmt="none", ecolor=INK2, elinewidth=1)
    for yy, a_, ea, b_, eb in zip(y, v, e, sv, se_):
        ax.text(max(a_, 0) + ea + 1.5, yy + 0.18, f"{a_:.0f}%", va="center", fontsize=7.5, color=INK2)
        ax.text(max(b_, 0) + eb + 1.5, yy - 0.18, f"{b_:.0f}%", va="center", fontsize=7.5, color=INK2)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y, [lab for _, lab in rows])
    ax.set_xlabel("Further fall in rotavirus episodes against today's schedule (%)")
    ax.set_title("A  Interventions on the vaccine", loc="left", fontsize=10, color=INK)
    ax.grid(axis="x", color=GRID, lw=0.6)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="all episodes, 0-24 months (helps, by the rule)"),
                       Patch(color=light, label="all episodes, 0-24 months (does not)"),
                       Patch(color=ORANGE, label="severe episodes, 0-59 months")],
              loc="center right", frameon=False, fontsize=7.5)
    ax.set_xlim(min(0, np.nanmin(np.r_[v - e, sv - se_])) - 2, 100)

    # B: by wealth quintile, rotavirus episodes under five
    ax = axes[1]
    z = np.array([[-100 * t.loc[sc, f"q{k}_rel"] for k in range(1, 6)] for sc, _ in rows])
    lim = max(10.0, np.nanmax(np.abs(z)))
    im = ax.imshow(z, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            ax.text(j, i, f"{z[i, j]:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(z[i, j]) > 0.6 * lim else INK)
    ax.set_xticks(range(5), ["lowest", "2", "3", "4", "highest"])
    ax.set_yticks(range(len(rows)), [lab for _, lab in rows])
    ax.set_xlabel("Wealth quintile")
    ax.set_title("B  Fall in rotavirus episodes under five, by quintile (%)", loc="left", fontsize=10, color=INK)
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.outline.set_visible(False)
    return _save(fig, analysis_dir, "fig_3b_vaccine_interventions")


# ---------------------------------------------------------------------------
# 1B: learning how well care works
# ---------------------------------------------------------------------------

STOCKOUT_SHADES = {2: "#c5daf3", 4: "#9ec1ec", 8: "#5f9be2", 16: "#2a78d6", 26: "#174d91"}


def _mean_series(df, label, key, k=1):
    from experiments.scenarios.analysis import rolling, weekly
    xs = [rolling(weekly(r, key), k) if k > 1 else weekly(r, key) for _, r in df[df["gsa.design_id"] == label].iterrows()]
    return np.nanmean(np.vstack(xs), axis=0) if xs else None


def fig_1b(analysis_dir: str, b1: str, b4: str | None = None, b2: str | None = None) -> str:
    from experiments.scenarios.designs import STOCKOUT_START
    _style()
    d1 = pd.read_parquet(b1)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), layout="constrained")
    yrs = lambda w: np.asarray(w) * 7 / 365.0
    wk0 = 130
    so = STOCKOUT_START / 365.0

    # A: trust through stock-outs (social weight 0.6), with care-seeking for the longest
    ax = axes[0, 0]
    for weeks, color in STOCKOUT_SHADES.items():
        y = _mean_series(d1, f"B1_s0.6_stockout{weeks}w", "wk_belief_success")
        if y is not None:
            ax.plot(yrs(np.arange(len(y)))[wk0:], y[wk0:], color=color, lw=1.8, label=f"{weeks}-week stock-out")
    y = _mean_series(d1, "B1_s0.6_none", "wk_belief_success")
    if y is not None:
        ax.plot(yrs(np.arange(len(y)))[wk0:], y[wk0:], color=INK2, lw=1.2, label="no stock-out")
    c = _mean_series(d1, "B1_s0.6_stockout26w", "wk_episodes_ended_with_care/wk_episodes_ended", k=8)
    if c is not None:
        ax.plot(yrs(np.arange(len(c)))[wk0:], c[wk0:], color=ORANGE, lw=1.5, ls="--",
                label="care sought per episode, 26-week stock-out (8-week mean)")
    ax.axvline(so, color=INK2, lw=0.8, ls=":")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Years since the start of the run (stock-outs start at the dotted line)")
    ax.set_ylabel("Mean belief that care works (actual 0.8; during a stock-out 0.2)")
    ax.set_title("A  Trust after stock-outs (neighbours' weight 0.6)", loc="left", fontsize=10, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(loc="lower right", frameon=False, fontsize=7)

    # B: recovery time of trust against the stock-out's duration; a seed that has not recovered by the end of
    # the run counts at the weeks it had left (a lower bound); open markers: more than half did not recover
    ax = axes[0, 1]
    summ = pd.read_csv(os.path.join(analysis_dir, "recovery_1b1_summary.csv"))
    for w, color in ((0.3, BLUE), (0.6, ORANGE)):
        g = summ[summ["social_weight"] == w].sort_values("weeks")
        if g.empty:
            continue
        pts = [(int(r.weeks), float(r.belief_recovery_weeks_median), r.belief_not_recovered > r.n / 2)
               for r in g.itertuples()]
        xs, ys, cs = zip(*pts)
        ax.plot(xs, ys, "-", color=color, lw=2, label=f"neighbours' weight {w:g}")
        for x_, y_, c_ in pts:
            ax.plot(x_, y_, "o", ms=7, color=color, mfc=SURF if c_ else color, mec=color, mew=1.8)
    xs = np.array([2, 26])
    ax.plot(xs, xs, color=INK2, lw=0.8, ls=":", label="recovery as long as the stock-out")
    ax.set_xlabel("Stock-out duration (weeks)")
    ax.set_ylabel("Weeks until trust is back (median of 8 seeds)")
    ax.set_title("B  Trust takes far longer to recover than the stock-out lasted", loc="left", fontsize=10, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(loc="upper left", frameon=False, fontsize=7.5)
    ax.text(0.99, 0.02, "open: more than half the seeds not recovered by the end of the run (a lower bound)",
            transform=ax.transAxes, ha="right", fontsize=7, color=INK2)

    # C: what restores trust (B4)
    ax = axes[1, 0]
    if b4 is not None and os.path.exists(b4):
        d4 = pd.read_parquet(b4)
        ref = _mean_series(d1, "B1_s0.6_none", "wk_belief_success")
        for label, color, ls, lab in (("B4_collapse", INK2, "-", "no intervention"),
                                      ("B4_campaign", BLUE, "-", "information campaign (half the households)"),
                                      ("B4_free_care", ORANGE, "-", "three months of free care"),
                                      ("B4_both", BLUE, "--", "both")):
            y = _mean_series(d4, label, "wk_belief_success")
            if y is not None and ref is not None:
                t = yrs(np.arange(len(y)))
                ax.plot(t[wk0:], (y - ref)[wk0:], color=color, ls=ls, lw=1.8, label=lab)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.axvline((STOCKOUT_START + 182) / 365.0, color=INK2, lw=0.8, ls=":")
        ax.set_xlabel("Years since the start of the run (supplies return at the dotted line)")
        ax.set_ylabel("Mean belief, against no stock-out")
        ax.legend(loc="lower right", frameon=False, fontsize=7.5)
    ax.set_title("C  What restores trust after a 26-week stock-out", loc="left", fontsize=10, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6)

    # D: down and up the efficacy ramp (B2)
    ax = axes[1, 1]
    paths_f = os.path.join(analysis_dir, "paths_1b2.csv")
    if os.path.exists(paths_f):
        from matplotlib.lines import Line2D
        pth = pd.read_csv(paths_f)
        handles = []
        for sc, color, name in (("B2_s0_ramp365", "#9ec1ec", "own outcomes only"),
                                ("B2_s0.3_ramp365", BLUE, "neighbours' weight 0.3"),
                                ("B2_s0.6_ramp365", "#174d91", "neighbours' weight 0.6"),
                                ("B2_s0.6_ramp730", ORANGE, "weight 0.6, two years each way")):
            g = pth[pth["scenario"] == sc].groupby(["leg", "efficacy"])["belief"].mean()
            if g.empty:
                continue
            ax.plot(g["down"].index, g["down"].values, "-o", color=color, ms=4, lw=1.8)
            ax.plot(g["up"].index, g["up"].values, "--o", color=color, ms=4, lw=1.8, mfc=SURF)
            handles.append(Line2D([], [], color=color, lw=2.5, label=name))
        ax.plot([0.1, 0.8], [0.1, 0.8], color=INK2, lw=0.8, ls=":")
        handles += [Line2D([], [], color=INK2, lw=1.8, marker="o", ms=4, label="on the way down"),
                    Line2D([], [], color=INK2, lw=1.8, ls="--", marker="o", ms=4, mfc=SURF, label="on the way back up"),
                    Line2D([], [], color=INK2, lw=0.8, ls=":", label="belief = actual probability")]
        ax.set_xlabel("Actual probability that care works (lowered to 0.1 and raised back, a year each way)")
        ax.set_ylabel("Mean belief that care works")
        ax.set_ylim(0.05, 0.85)
        ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=7.5)
    ax.set_title("D  Trust lags the quality of care, down and back up", loc="left", fontsize=10, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6)
    return _save(fig, analysis_dir, "fig_1b_learning")


FIGURES = {"3a": ("exp3a_vaccine_limits", fig_3a), "1a": ("exp1a_care_levers", fig_1a),
           "2": ("exp2_shocks", fig_2), "3b": ("exp3b_vaccine_interventions", fig_3b)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=sorted(FIGURES))
    ap.add_argument("--analysis", default=None)
    ap.add_argument("--results", nargs="*", default=None, help="results.parquet file(s), for figures drawn from runs")
    a = ap.parse_args(argv)
    name, fn = FIGURES[a.part]
    out = a.analysis or os.path.join(OUT_ROOT, name, "analysis")
    if a.part in ("1a", "2"):
        print(fn(out, a.results or [os.path.join(OUT_ROOT, name, "results.parquet")]))
    else:
        print(fn(out))


if __name__ == "__main__":
    main()
