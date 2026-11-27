# experiments/scenarios/paper1_figures.py
"""
Paper 1's figures (PLOS Computational Biology), drawn from the tables that
experiments.calibration.diagnostics_v5 and experiments.scenarios.paper1_analysis
write:

    python -m experiments.scenarios.paper1_figures            # every figure
    python -m experiments.scenarios.paper1_figures calibration

Each figure is written as PDF (vector, editable text, for the LaTeX draft), as
TIFF (600 dpi, RGB, LZW, for submission) and as PNG (150 dpi, to look at) to
experiments/outputs/experiments_v5/paper1/figures/ (--out to change it).

PLOS's rules for figures: 6.68-19.05 cm wide, at most 22.23 cm tall, Arial at
8-12 pt (Liberation Sans, with Arial's metrics, where Arial is missing). Colours
as in experiments/scenarios/paper_figures.py (dataviz validator, white surface):
categorical blue #2a78d6 and orange #eb6834 (CVD-safe as a pair), a sequential
blue ramp for ordered levels, ink #0b0b0b and secondary #52514e.

Figures:
  calibration  Fig 2, calibration by history matching. A: the share of the search
               box not ruled out before each wave. B: each calibrated parameter's
               position in its search range, for the design points of waves 2 and 4
               and for the 11 calibrated sets. C: the 11 sets' fit to the 41 targets.
  observation  Fig 3, what a birth cohort detects (experiment 8; no vaccine, 16 seeds).
               A: infections by age, all and as MAL-ED's design detects them, against
               MAL-ED. B: the share of infections each design detects. C: the share of
               detected infections each design attributes to diarrhoea, against the
               share of all infections that cause an episode.
  validation   Fig 4, data not used in calibration. A: the vaccine's effect (the
               calibration ran without it) against Ghana's fall in rotavirus
               hospitalisations. B: diarrhoea by wealth against the DHS.
  routes       Fig 5, transmission routes (experiment 7; no vaccine). A: each route's
               share of infections in the 11 calibrated sets. B: the adopted set's
               shares against the fall in infections when the route is switched off.
  decisions    Fig 6, care-seeking decisions (experiment 9; today's vaccine, 40 seeds).
               A: care-seeking by wealth quintile under CPT and its variants, against
               the DHS. B: each variant's change in care-seeking.
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

CM = 1 / 2.54
MAX_WIDTH = 19.05 * CM
MAX_HEIGHT = 22.23 * CM
BLUE, ORANGE = "#2a78d6", "#eb6834"
RAMP = {100: "#cde2fb", 150: "#b7d3f6", 200: "#9ec5f4", 250: "#86b6ef", 400: "#3987e5", 550: "#1c5cab",
        650: "#104281", 700: "#0d366b"}
INK, INK2, MUTED, HAIR = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
FONT_PT, LETTER_PT = 8, 10

DIAG = os.path.join("experiments", "outputs", "history_matching_v5", "diagnostics")
PAPER1 = os.path.join("experiments", "outputs", "experiments_v5", "paper1")
OUT = os.path.join(PAPER1, "figures")

ASYMP = ("pathogens[rota].asymptomatic_relative_infectiousness|pathogens[campy].asymptomatic_relative_infectiousness"
         "|pathogens[crypto].asymptomatic_relative_infectiousness")


def _pathogen_params(p: str, own: list[tuple[str, str]]) -> list[tuple[str, str]]:
    common = [("symptomatic_fraction", "Symptomatic fraction"),
              ("infant_susceptibility_at_birth", "Infant susceptibility at birth"),
              ("infant_susceptibility_ramp_months", "Infant susceptibility ramp"),
              ("detectable_shedding_days_symptomatic", "Shedding, symptomatic"),
              ("detectable_shedding_days_asymptomatic", "Shedding, asymptomatic")]
    return [(f"pathogens[{p}].{k}", v) for k, v in own + common]


#: the 37 calibrated parameters by group, with short labels (S1 Appendix has their definitions)
PARAM_GROUPS = (
    ("Rotavirus", _pathogen_params("rota", [("infection_prob_mean", "Transmission probability"),
                                            ("adult_relative_susceptibility", "Adults' relative susceptibility"),
                                            ("water_dose_scale", "Water dose scale"),
                                            ("importation_rate", "Importation rate")])
     + [("pathogens[rota].post_infection_refractory_days", "Refractory period")]),
    ("Campylobacter", _pathogen_params("campy", [("human_animal_interaction_rate", "Contact with animals"),
                                                 ("fecal_oral_prob", "Faecal–oral transmission"),
                                                 ("food_borne_prob", "Food-borne infection")])),
    ("Cryptosporidium", _pathogen_params("crypto", [("household_transmission_prob", "Household transmission"),
                                                    ("community_infection_prob", "Community infection"),
                                                    ("adult_relative_susceptibility", "Adults' relative susceptibility"),
                                                    ("water_dose_scale", "Water dose scale"),
                                                    ("zoonotic_interaction_rate", "Contact with ruminants")])),
    ("All three pathogens", [(ASYMP, "Asymptomatic infectiousness"),
                             ("steering_parameters.household_exposure_sdlog", "Spread of household exposure")]),
    ("Household economy", [("economy.income_shock_sd", "Income shock SD"),
                           ("economy.income_shock_halflife_days", "Income shock half-life"),
                           ("economy.buffer_days", "Cash buffer"),
                           ("economy.savings_rate", "Savings rate"),
                           ("economy.earning_spread_sdlog", "Spread of earnings"),
                           ("economy.cost_primary_insured", "Primary care cost, insured"),
                           ("economy.buffer_wealth_elasticity", "Buffer's wealth elasticity")]),
)


def _detection_targets(p: str) -> list[tuple[str, str]]:
    return [(f"{p}_obs_{k}", v) for k, v in (
        ("infections_per_child_year_0_24m", "Infections, 0–24 mo"),
        ("attributable_episodes_per_child_year_0_24m", "Attributable episodes, 0–24 mo"),
        ("infant_ratio_0_6_vs_6_12m", "Infections, 0–6 vs 6–12 mo"),
        ("second_to_first_year_ratio", "Infections, 2nd vs 1st year"),
        ("naive_decline_15_24_vs_6_12m", "Naive rate, 15–24 vs 6–12 mo"),
        ("prior_infection_rate_ratio_12_24m", "Prior vs naive rate, 12–24 mo"),
        ("crude_to_consecutive_ratio", "Positive stools vs runs"),
        ("median_age_first_detected_infection_months", "Median age at first detection"))]


#: the 41 targets by group, with short labels
TARGET_GROUPS = (
    ("Rotavirus, detected (MAL-ED)", _detection_targets("rota")),
    ("Rotavirus, true (Vellore)", [("rota_true_frac_infected_by_6m", "Infected by 6 mo"),
                                   ("rota_true_primary_share_0_36m", "First infections' share, 0–36 mo")]),
    ("Campylobacter, detected (MAL-ED)", _detection_targets("campy")),
    ("Cryptosporidium, detected (MAL-ED)", _detection_targets("crypto")),
    ("Routes and carriage", [("campy_zoonotic_fraction_post_burnin", "Campylobacter: share from animals"),
                             ("crypto_household_fraction_post_burnin", "Cryptosporidium: share in households"),
                             ("crypto_zoonotic_fraction_post_burnin", "Cryptosporidium: share from animals"),
                             ("rota_adult_to_child_carriage_ratio_post_burnin", "Rotavirus: adult vs child carriage"),
                             ("crypto_adult_to_child_carriage_ratio_post_burnin",
                              "Cryptosporidium: adult vs child carriage")]),
    ("Care-seeking by wealth quintile (DHS)", [(f"econ_care_seeking_rate_q{q}", f"Care sought, quintile {q}")
                                              for q in range(1, 6)]
     + [(f"econ_cannot_pay_facility_share_q{q}", f"Cannot pay, quintile {q}") for q in range(1, 6)]),
)


def style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
        "font.size": FONT_PT, "axes.titlesize": FONT_PT, "axes.labelsize": FONT_PT, "xtick.labelsize": FONT_PT,
        "ytick.labelsize": FONT_PT, "legend.fontsize": FONT_PT, "axes.edgecolor": INK2, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def letter(ax, s: str, x: float = 0.0, y: float = 1.0, fig=None) -> None:
    """A panel letter at the axes' top left (figure coordinates if fig is given)."""
    if fig is not None:
        fig.text(x, y, s, fontsize=LETTER_PT, fontweight="bold", va="top", ha="left")
    else:
        ax.text(x, y, s, transform=ax.transAxes, fontsize=LETTER_PT, fontweight="bold", va="bottom", ha="right")


def save(fig, name: str, out: str) -> list[str]:
    """PDF, TIFF (600 dpi, RGB, LZW) and PNG (150 dpi); refuses a figure larger than PLOS allows."""
    w, h = fig.get_size_inches()
    if w > MAX_WIDTH + 1e-6 or h > MAX_HEIGHT + 1e-6:
        raise ValueError(f"{name}: {w / CM:.2f} x {h / CM:.2f} cm is larger than PLOS allows")
    os.makedirs(out, exist_ok=True)
    base = os.path.join(out, name)
    fig.savefig(base + ".pdf")
    fig.savefig(base + ".png", dpi=150)
    from PIL import Image
    tmp = base + "_600.png"
    fig.savefig(tmp, dpi=600)
    with Image.open(tmp) as im:
        im.convert("RGB").save(base + ".tif", compression="tiff_lzw", dpi=(600, 600))
    os.remove(tmp)
    plt.close(fig)
    return [base + ext for ext in (".pdf", ".tif", ".png")]


def _rows(groups) -> list[tuple[str, str | None]]:
    """(label, key) rows with a header row (key None) before each group."""
    rows = []
    for g, items in groups:
        rows.append((g, None))
        rows += [(lab, key) for key, lab in items]
    return rows


def _label_rows(ax, rows) -> None:
    y = np.arange(len(rows))
    ax.set_yticks(y)
    ax.set_yticklabels([lab for lab, _ in rows])
    for t, (_, key) in zip(ax.get_yticklabels(), rows):
        if key is None:
            t.set_fontweight("bold")
            t.set_color(INK)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(rows) - 0.5, -0.7)
    for i, (_, key) in enumerate(rows):
        if key is None and i > 0:
            ax.axhline(i - 0.5, color=HAIR, lw=0.6, zorder=0)


# ---------------------------------------------------------------------------
# Fig 2: calibration by history matching
# ---------------------------------------------------------------------------

def fig_calibration(diag: str = DIAG, out: str = OUT) -> list[str]:
    nroy = pd.read_csv(os.path.join(diag, "nroy.csv")).sort_values("wave")
    ranges = pd.read_csv(os.path.join(diag, "design_ranges.csv"))
    sets = pd.read_csv(os.path.join(diag, "sets_box.csv"))
    fit = pd.read_csv(os.path.join(diag, "fit_sets.csv"))
    new_seeds = os.path.join(diag, "fit.csv")                   # the adopted defaults on 8 new seeds
    fit_new = pd.read_csv(new_seeds).set_index("target") if os.path.exists(new_seeds) else None
    p_rows, t_rows = _rows(PARAM_GROUPS), _rows(TARGET_GROUPS)
    known_p = {k for _, k in p_rows if k}
    known_t = {k for _, k in t_rows if k}
    if known_p != set(ranges.parameter) or known_t != set(fit.target):
        raise ValueError("the figure's parameter or target list differs from the tables'")

    row_mm = 3.25
    fig_h = MAX_HEIGHT - 0.6 * CM
    fig = plt.figure(figsize=(MAX_WIDTH, fig_h))
    W, H = MAX_WIDTH / CM * 10, fig_h / CM * 10                        # figure size in mm

    def axes(left, bottom, width, height):                             # mm from the bottom left
        return fig.add_axes([left / W, bottom / H, width / W, height / H])

    # panel C (targets) sets the height of the tall panels; B is aligned at the top with the same rows
    c_h, b_h = row_mm * (len(t_rows) + 0.7), row_mm * (len(p_rows) + 0.7)
    bottom_c = 10.0
    top_tall = bottom_c + c_h
    axB = axes(47.0, top_tall - b_h, 37.0, b_h)
    axC = axes(146.0, bottom_c, 41.0, c_h)
    axA = axes(18.0, top_tall + 17.0, 52.0, H - top_tall - 23.0)

    # A: the share of the box not ruled out before each wave
    axA.plot(nroy.wave, nroy.nroy_share_before, color=BLUE, lw=1.5, marker="o", ms=4, zorder=3)
    axA.set_yscale("log")
    axA.set_xticks(nroy.wave)
    axA.set_xlim(0.7, nroy.wave.max() + 0.5)
    axA.set_ylim(1e-6, 4.0)
    axA.set_xlabel("Wave")
    axA.set_ylabel("Share of search box\nnot ruled out")
    last = int(nroy.wave.max())
    for w, v in zip(nroy.wave, nroy.nroy_share_before):
        txt = "whole box" if v >= 1 else f"{100 * v:.2g}%" if v >= 1e-4 else f"{100 * v:.1g}%"
        if w == last:
            txt += ", also within\nthe design rules"
        axA.annotate(txt, (w, v), xytext=(5, 2 if w < last else -1), textcoords="offset points",
                     fontsize=FONT_PT, color=INK, va="bottom" if w < last else "center")
    letter(axA, "A", fig=fig, x=2 / W, y=(H - 3.0) / H)

    # B: each parameter's position in its search range
    _label_rows(axB, p_rows)
    y_of = {k: i for i, (_, k) in enumerate(p_rows) if k}
    for i, (_, k) in enumerate(p_rows):
        if k:
            axB.plot([0, 1], [i, i], color=HAIR, lw=0.6, zorder=1)
    for w, dy, colour, lw in ((2, -0.17, RAMP[150], 3.2), (4, 0.17, RAMP[250], 3.2)):
        r = ranges[ranges.wave == w]
        for _, row in r.iterrows():
            i = y_of[row.parameter]
            axB.plot([row.q025, row.q975], [i + dy, i + dy], color=colour, lw=lw, solid_capstyle="butt", zorder=2)
    others = sets[~sets.adopted]
    axB.scatter(others.unit, others.parameter.map(y_of), s=5, color=INK2, lw=0, zorder=4)
    ad = sets[sets.adopted]
    axB.scatter(ad.unit, ad.parameter.map(y_of), s=16, marker="D", color=ORANGE, edgecolor="white", lw=0.5, zorder=5)
    axB.set_xlim(-0.03, 1.03)
    axB.set_xticks([0, 0.5, 1])
    axB.set_xticklabels(["lower\nbound", "middle", "upper\nbound"])
    axB.set_xlabel("Position in search range")
    axB.spines["left"].set_visible(False)
    letter(axB, "B", fig=fig, x=2 / W, y=(top_tall + 4.5) / H)

    # C: the sets' fit to the targets
    _label_rows(axC, t_rows)
    yt = {k: i for i, (_, k) in enumerate(t_rows) if k}
    axC.axvspan(-3, 3, color="#f6f5f2", zorder=0)
    for x in (-3, 3):
        axC.axvline(x, color=MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
    axC.axvline(0, color=INK2, lw=0.6, zorder=1)
    o = fit[~fit.adopted]
    axC.scatter(o.signed.clip(-4.4, 4.4), o.target.map(yt), s=5, color=INK2, lw=0, zorder=4)
    a = fit[fit.adopted]
    axC.scatter(a.signed.clip(-4.4, 4.4), a.target.map(yt), s=16, marker="D", color=ORANGE, edgecolor="white",
                lw=0.5, zorder=6)
    if fit_new is not None:
        f = fit_new.reindex(a.target)
        axC.scatter(f.signed.clip(-4.4, 4.4), a.target.map(yt), s=16, marker="D", facecolor="none",
                    edgecolor=ORANGE, lw=0.8, zorder=5)
    axC.set_xlim(-4.6, 4.6)
    axC.set_xticks([-3, 0, 3])
    axC.set_xlabel("Standardised error, model − target")
    axC.spines["left"].set_visible(False)
    letter(axC, "C", fig=fig, x=96 / W, y=(top_tall + 4.5) / H)

    # legends, beside panel A
    lx = 100 / W
    handles = [Patch(facecolor=RAMP[150], label="B: after wave 1 (95% of wave 2's points)"),
               Patch(facecolor=RAMP[250], label="B: after wave 3 (95% of wave 4's points)"),
               Line2D([], [], ls="", marker="o", ms=2.6, color=INK2, label="The other 10 calibrated sets"),
               Line2D([], [], ls="", marker="D", ms=4, color=ORANGE, mec="white", mew=0.5,
                      label="Adopted set, verification seeds")]
    if fit_new is not None:
        handles.append(Line2D([], [], ls="", marker="D", ms=4, mfc="none", mec=ORANGE, mew=0.8,
                              label="Adopted set, 8 new seeds (C)"))
    handles.append(Patch(facecolor="#f6f5f2", edgecolor=MUTED, lw=0.6, ls=(0, (3, 2)),
                         label="C: within 3 standardised errors"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(lx, (H - 6.0) / H), handlelength=1.6,
               borderaxespad=0, labelspacing=0.35)
    return save(fig, "fig2_calibration", out)


# ---------------------------------------------------------------------------
# Shared pieces of Figs 3-6
# ---------------------------------------------------------------------------

PATHOGEN_NAMES = {"rota": "Rotavirus", "campy": "Campylobacter", "crypto": "Cryptosporidium"}
SEEDS_NOTE = "mean ± 2 SE over seeds"


def _xgrid(ax, ticks) -> None:
    ax.set_xticks(ticks)
    for x in ticks:
        ax.axvline(x, color=HAIR, lw=0.5, zorder=0)


def _fig(height_cm: float, width_cm: float = MAX_WIDTH / CM):
    fig = plt.figure(figsize=(width_cm * CM, height_cm * CM))
    W, H = width_cm * 10, height_cm * 10

    def axes(left, bottom, width, height):                             # mm from the bottom left
        return fig.add_axes([left / W, bottom / H, width / W, height / H])
    return fig, axes, W, H


# ---------------------------------------------------------------------------
# Fig 3: what a birth cohort detects (experiment 8)
# ---------------------------------------------------------------------------

OBS_LABELS = {"maled": "MAL-ED: a stool a month,\nand diarrhoeal stools",
              "fortnightly": "A stool a fortnight", "weekly": "A stool a week", "daily": "A stool a day",
              "no_diarrhoeal_stools": "No diarrhoeal stools", "all_diarrhoeal_stools": "Every diarrhoeal stool",
              "windows_half": "Detection windows halved", "windows_double": "Detection windows doubled"}


def fig_observation(paper1: str = PAPER1, out: str = OUT) -> list[str]:
    designs = pd.read_csv(os.path.join(paper1, "p1_obs_designs.csv"))
    by_age = pd.read_csv(os.path.join(paper1, "p1_obs_by_age.csv"))
    if set(designs.design) != set(OBS_LABELS):
        raise ValueError("the figure's design labels differ from the table's designs")
    order = list(OBS_LABELS)
    fig, axes, W, H = _fig(19.4)
    left, gap, pw = 44.0, 9.0, 42.0                                    # mm
    xs = [left + i * (pw + gap) for i in range(3)]
    rowA, rowB, rowC = (130.0, 46.0), (72.0, 40.0), (16.0, 40.0)        # (bottom, height) in mm
    mid = (by_age.age_lo_months + by_age.age_hi_months) / 2
    by_age = by_age.assign(mid=mid)
    for j, p in enumerate(("rota", "campy", "crypto")):
        g = by_age[by_age.pathogen == p]
        ax = axes(xs[j], rowA[0], pw, rowA[1])
        ax.plot(g.mid, g.true, color=INK2, lw=1.2, ls=(0, (4, 2)), zorder=2)
        ax.fill_between(g.mid, g.detected - 2 * g.detected_se, g.detected + 2 * g.detected_se, color=RAMP[150],
                        lw=0, zorder=1)
        ax.plot(g.mid, g.detected, color=BLUE, lw=1.5, marker="o", ms=3, zorder=3)
        ax.plot(g.mid, g.maled, color=ORANGE, lw=1.2, marker="s", ms=3, zorder=4)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.set_xlim(0, 24)
        ax.set_ylim(0, None)
        ax.set_title(PATHOGEN_NAMES[p], fontsize=FONT_PT, fontweight="bold", fontstyle="italic", loc="left")
        r = float(np.corrcoef(g.detected, g.maled)[0, 1])
        ax.text(0.98, 0.04, f"r = {r:.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=FONT_PT,
                color=INK2)
        ax.set_xlabel("Age (months)")
        if j == 0:
            ax.set_ylabel("Infections per child-year")
            letter(ax, "A", fig=fig, x=2 / W, y=(rowA[0] + rowA[1] + 8) / H)
    handles = [Line2D([], [], color=INK2, lw=1.2, ls=(0, (4, 2)), label="Model, all infections"),
               Line2D([], [], color=BLUE, lw=1.5, marker="o", ms=3, label="Model, detected by MAL-ED's design"),
               Patch(facecolor=RAMP[150], label="± 2 SE (16 seeds)"),
               Line2D([], [], color=ORANGE, lw=1.2, marker="s", ms=3, label="MAL-ED")]
    fig.legend(handles=handles, loc="upper left", ncol=4, bbox_to_anchor=(12 / W, (H - 1.5) / H),
               borderaxespad=0, handlelength=1.8, columnspacing=1.2)

    def design_rows(ax, first: bool):
        y = np.arange(len(order))
        ax.set_yticks(y)
        ax.set_yticklabels([OBS_LABELS[d] for d in order] if first else [])
        ax.set_ylim(len(order) - 0.5, -0.6)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        if first:
            ax.get_yticklabels()[0].set_fontweight("bold")
        ax.axhspan(-0.5, 0.5, color="#f6f5f2", zorder=0)
        return {d: i for i, d in enumerate(order)}

    for j, p in enumerate(("rota", "campy", "crypto")):
        d = designs[designs.pathogen == p].set_index("design")
        ax = axes(xs[j], rowB[0], pw, rowB[1])
        yy = design_rows(ax, j == 0)
        _xgrid(ax, [0, 50, 100])
        ax.scatter(100 * d.detected_over_true, d.index.map(yy), s=18, color=BLUE, zorder=3)
        ax.set_xlim(-4, 108)
        ax.set_xlabel("Infections detected (%)")
        if j == 0:
            letter(ax, "B", fig=fig, x=2 / W, y=(rowB[0] + rowB[1] + 7) / H)
        ax = axes(xs[j], rowC[0], pw, rowC[1])
        yy = design_rows(ax, j == 0)
        true_share = 100 * d.loc["maled", "true_episodes"] / d.loc["maled", "true_infections"]
        top = 100 * max(d.attributable_fraction_0_24m.max(), true_share / 100)
        ticks = [0, 20, 40, 60, 80] if top > 45 else [0, 10, 20, 30, 40] if top > 22 else [0, 5, 10, 15, 20, 25]
        _xgrid(ax, ticks)
        ax.axvline(true_share, color=ORANGE, lw=1.0, ls=(0, (3, 2)), zorder=2)
        ax.text(true_share, -0.75, f" all infections: {true_share:.0f}%", color=INK, fontsize=FONT_PT,
                va="bottom", ha="left")
        ax.scatter(100 * d.attributable_fraction_0_24m.drop("no_diarrhoeal_stools"),
                   d.drop("no_diarrhoeal_stools").index.map(yy), s=18, color=BLUE, zorder=3)
        ax.text(0.5, yy["no_diarrhoeal_stools"], "  (none tested)", color=MUTED, fontsize=FONT_PT, va="center")
        ax.set_xlim(0, ticks[-1] * 1.08)
        ax.set_xlabel("Detected infections\nattributed to diarrhoea (%)")
        if j == 0:
            letter(ax, "C", fig=fig, x=2 / W, y=(rowC[0] + rowC[1] + 9) / H)
    return save(fig, "fig3_observation", out)


# ---------------------------------------------------------------------------
# Fig 4: validation against data not used in calibration
# ---------------------------------------------------------------------------

def fig_validation(paper1: str = PAPER1, out: str = OUT) -> list[str]:
    from experiments.scenarios.analysis import GHANA_FALL
    eff = pd.read_csv(os.path.join(paper1, "p1_3a_effects.csv"))
    sets = pd.read_csv(os.path.join(paper1, "p1_ps_sets.csv"))
    wealth = pd.read_csv(os.path.join(paper1, "p1_wealth_gradient.csv"))
    e = eff[eff.scenario == "vacc_default"].set_index("outcome")
    fig, axes, W, H = _fig(8.6)
    # A: the vaccine's effect
    ax = axes(52.0, 12.0, 52.0, 50.0)
    rows = [("Model: all episodes,\nunder 2 years", -e.loc["episodes 0-24 m", "change_pct"],
             e.loc["episodes 0-24 m", "se_pct"], sets.vacc_fall_episodes_0_24m),
            ("Model: severe episodes,\nunder 5 years", -e.loc["severe episodes 0-59 m", "change_pct"],
             e.loc["severe episodes 0-59 m", "se_pct"], sets.vacc_fall_severe_0_59m)]
    _xgrid(ax, [0, 20, 40, 60, 80])
    for i, (_, mean, se, per_set) in enumerate(rows):
        ax.scatter(per_set, np.full(len(per_set), i + 0.22), s=6, color=INK2, lw=0, zorder=3)
        ax.plot([mean - 2 * se, mean + 2 * se], [i, i], color=BLUE, lw=1.6, zorder=3)
        ax.scatter([mean], [i], s=26, color=BLUE, zorder=4)
    g, lo, hi = (100 * v for v in GHANA_FALL)
    ax.plot([lo, hi], [2, 2], color=ORANGE, lw=1.6, zorder=3)
    ax.scatter([g], [2], s=26, marker="s", color=ORANGE, zorder=4)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels([r[0] for r in rows] + ["Ghana: rotavirus\nhospitalisations, under 5"])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(2.6, -0.6)
    ax.set_xlim(0, 85)
    ax.set_xlabel("Fall with today's vaccine (%)")
    handles = [Line2D([], [], color=BLUE, lw=1.6, marker="o", ms=4, label="Adopted set, 40 seeds (± 2 SE)"),
               Line2D([], [], ls="", marker="o", ms=2.4, color=INK2, label="Each calibrated set, 8 seeds"),
               Line2D([], [], color=ORANGE, lw=1.6, marker="s", ms=4, label="Armah et al. 2016 (95% CI)")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(12 / W, (H - 1.5) / H), ncol=1,
               borderaxespad=0, handlelength=1.6)
    letter(ax, "A", fig=fig, x=2 / W, y=(H - 2) / H)
    # B: diarrhoea by wealth
    ax = axes(128.0, 12.0, 56.0, 50.0)
    q = wealth.quintile
    ax.axhline(1, color=HAIR, lw=0.6, zorder=0)
    ax.errorbar(q, wealth.model_rel_q5, yerr=2 * wealth.model_rel_q5_se, color=BLUE, lw=1.5, marker="o", ms=4,
                capsize=0, elinewidth=1.0, zorder=3, label="Model, 40 seeds (± 2 SE)")
    ax.plot(q, wealth.dhs2022_rel_q5, color=ORANGE, lw=1.5, marker="s", ms=4, zorder=3, label="DHS 2022")
    ax.plot(q, wealth.dhs2014_rel_q5, color=MUTED, lw=1.0, marker="s", ms=3, ls=(0, (3, 2)), zorder=2,
            label="DHS 2014")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(["1\npoorest", "2", "3", "4", "5\nrichest"])
    ax.set_xlabel("Wealth quintile")
    ax.set_ylabel("Two-week diarrhoea prevalence,\nrelative to the richest quintile")
    ax.set_ylim(0.8, 2.2)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=1, borderaxespad=0, handlelength=1.8)
    letter(ax, "B", fig=fig, x=108 / W, y=(H - 2) / H)
    return save(fig, "fig4_validation", out)


# ---------------------------------------------------------------------------
# Fig 5: transmission routes (experiment 7 and the calibrated sets)
# ---------------------------------------------------------------------------

ROUTE_LABELS = {"h2h": "Person to person", "water": "Drinking water", "imported": "Imported",
                "zoonotic": "Animals", "fecal_oral": "Faecal–oral, in households", "food_borne": "Food",
                "household": "Within households", "community": "In the community"}


def fig_routes(paper1: str = PAPER1, out: str = OUT) -> list[str]:
    from experiments.scenarios.paper1 import ROUTES
    knock = pd.read_csv(os.path.join(paper1, "p1_routes.csv")).set_index(["pathogen", "route"])
    sets = pd.read_csv(os.path.join(paper1, "p1_ps_sets.csv"))
    groups = []
    for p in ("rota", "campy", "crypto"):
        groups.append((PATHOGEN_NAMES[p], [((p, r), ROUTE_LABELS[r]) for q, r in ROUTES if q == p]))
    rows = _rows(groups)
    row_mm = 4.2
    fig, axes, W, H = _fig(row_mm * (len(rows) + 0.7) / 10 + 3.6)
    h = row_mm * (len(rows) + 0.7)
    axA, axB = axes(44.0, 14.0, 52.0, h), axes(128.0, 14.0, 56.0, h)
    for ax in (axA, axB):
        _label_rows(ax, rows)
        ax.spines["left"].set_visible(False)
        _xgrid(ax, [0, 25, 50, 75, 100])
        ax.set_xlim(-2, 102)
    axB.set_yticklabels([])
    yi = {k: i for i, (_, k) in enumerate(rows) if k}
    for (p, r), i in yi.items():
        col = f"{p}_{r}_fraction_post_burnin"
        o = sets[~sets.adopted]
        axA.scatter(100 * o[col], np.full(len(o), i), s=7, color=INK2, lw=0, zorder=3)
        a = sets[sets.adopted]
        axA.scatter(100 * a[col], [i], s=18, marker="D", color=ORANGE, edgecolor="white", lw=0.5, zorder=4)
        k = knock.loc[(p, r)]
        share, fall, se = k.share_pct, k.fall_all_infections_pct, k.fall_all_infections_se
        axB.plot([share, fall], [i, i], color=RAMP[150], lw=2.4, solid_capstyle="butt", zorder=1)
        axB.plot([fall - 2 * se, fall + 2 * se], [i, i], color=BLUE, lw=1.0, zorder=3)
        axB.scatter([fall], [i], s=20, color=BLUE, zorder=4)
        axB.scatter([share], [i], s=20, facecolor="white", edgecolor=INK2, lw=0.9, zorder=5)
    axA.set_xlabel("Share of infections (%)")
    axB.set_xlabel("Share of infections, and fall in\ninfections with the route switched off (%)")
    hA = [Line2D([], [], ls="", marker="o", ms=2.8, color=INK2, label="The other 10 calibrated sets"),
          Line2D([], [], ls="", marker="D", ms=4, color=ORANGE, mec="white", mew=0.5, label="Adopted set")]
    hB = [Line2D([], [], ls="", marker="o", ms=4, mfc="white", mec=INK2, mew=0.9,
                 label="Share of infections, adopted set"),
          Line2D([], [], color=BLUE, lw=1.0, marker="o", ms=4, label="Fall when switched off, 16 seeds (± 2 SE)")]
    axA.legend(handles=hA, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=1, borderaxespad=0.3,
               handlelength=1.2)
    axB.legend(handles=hB, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=1, borderaxespad=0.3,
               handlelength=1.6)
    letter(axA, "A", fig=fig, x=2 / W, y=(H - 2) / H)
    letter(axB, "B", fig=fig, x=100 / W, y=(H - 2) / H)
    return save(fig, "fig5_routes", out)


# ---------------------------------------------------------------------------
# Fig 6: care-seeking decisions (experiment 9)
# ---------------------------------------------------------------------------

DECISION_LABELS = {"no_probability_weighting": "No probability weighting", "no_loss_aversion": "No loss aversion",
                   "linear_value": "Linear value function", "expected_utility": "Expected utility\n(all three)",
                   "no_cash_constraint": "No cash constraint"}


def fig_decisions(paper1: str = PAPER1, out: str = OUT) -> list[str]:
    from experiments.calibration.history_matching import ECON_CARE_TARGETS
    dec = pd.read_csv(os.path.join(paper1, "p1_decision.csv")).set_index(["variant", "output"])
    fig, axes, W, H = _fig(7.6)
    # A: care-seeking by quintile
    ax = axes(20.0, 12.0, 80.0, 56.0)
    q = np.arange(1, 6)
    dhs = np.array([ECON_CARE_TARGETS[k][0] for k in q])               # (value, 95% CI low, high)
    dhs_err = np.array([[ECON_CARE_TARGETS[k][0] - ECON_CARE_TARGETS[k][1] for k in q],
                        [ECON_CARE_TARGETS[k][2] - ECON_CARE_TARGETS[k][0] for k in q]])

    def series(v):
        return np.array([dec.loc[(v, f"econ_care_seeking_rate_q{k}"), "value"] for k in q])
    for v in ("no_probability_weighting", "no_loss_aversion", "linear_value"):
        ax.plot(q, 100 * series(v), color=RAMP[150], lw=1.0, zorder=2)
    ax.plot(q, 100 * series("expected_utility"), color=BLUE, lw=1.2, ls=(0, (4, 2)), zorder=3)
    ax.plot(q, 100 * series("cpt"), color=BLUE, lw=1.6, marker="o", ms=4, zorder=4)
    ax.plot(q, 100 * series("no_cash_constraint"), color=INK2, lw=1.6, marker="^", ms=4, zorder=4)
    ax.errorbar(q, 100 * dhs, yerr=100 * dhs_err, color=ORANGE, lw=0, elinewidth=1.2, marker="s", ms=4, zorder=5)
    ax.set_xticks(q)
    ax.set_xticklabels(["1\npoorest", "2", "3", "4", "5\nrichest"])
    ax.set_xlabel("Wealth quintile")
    ax.set_ylabel("Episodes taken for care (%)")
    ax.set_ylim(40, 100)
    handles = [Line2D([], [], color=BLUE, lw=1.6, marker="o", ms=4, label="CPT (the model)"),
               Line2D([], [], color=RAMP[150], lw=1.0, label="One CPT feature removed"),
               Line2D([], [], color=BLUE, lw=1.2, ls=(0, (4, 2)), label="Expected utility"),
               Line2D([], [], color=INK2, lw=1.6, marker="^", ms=4, label="No cash constraint"),
               Line2D([], [], color=ORANGE, lw=0, marker="s", ms=4, label="DHS 2022 (95% CI), the target")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.02, 0.86), borderaxespad=0, handlelength=1.8,
              labelspacing=0.3)
    letter(ax, "A", fig=fig, x=2 / W, y=(H - 2) / H)
    # B: change in care-seeking, each variant against CPT
    ax = axes(148.0, 12.0, 38.0, 56.0)
    vs = list(DECISION_LABELS)
    _xgrid(ax, [0, 10, 20, 30])
    ax.axvline(0, color=INK2, lw=0.6, zorder=1)
    for i, v in enumerate(vs):
        d, se = (100 * dec.loc[(v, "episode_care_seeking_rate"), c] for c in ("diff", "se"))
        ax.plot([d - 2 * se, d + 2 * se], [i, i], color=BLUE, lw=1.2, zorder=3)
        ax.scatter([d], [i], s=20, color=BLUE if v != "no_cash_constraint" else INK2, zorder=4)
    ax.set_yticks(range(len(vs)))
    ax.set_yticklabels([DECISION_LABELS[v] for v in vs])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(len(vs) - 0.5, -0.5)
    ax.set_xlim(-3, 36)
    ax.set_xlabel("Change in episodes\ntaken for care (points)")
    letter(ax, "B", fig=fig, x=112 / W, y=(H - 2) / H)
    return save(fig, "fig6_decisions", out)


# ---------------------------------------------------------------------------
# Supporting figures
# ---------------------------------------------------------------------------

OUTPUT_LABELS = {
    **{f"{p}_infections_per_child_year_0_24m": f"{PATHOGEN_NAMES[p]}: infections, 0–24 mo" for p in PATHOGEN_NAMES},
    **{f"{p}_symptomatic_episodes_per_child_year_0_24m": f"{PATHOGEN_NAMES[p]}: episodes, 0–24 mo"
       for p in PATHOGEN_NAMES},
    **{f"{p}_obs_infections_per_child_year_0_24m": f"{PATHOGEN_NAMES[p]}: detected, 0–24 mo" for p in PATHOGEN_NAMES},
    "episode_care_seeking_rate": "Episodes taken for care", "sev59": "Rotavirus: severe episodes, 0–59 mo"}


def s1_seeds(paper1: str = PAPER1, out: str = OUT) -> list[str]:
    """S1 Fig: how many seeds, and the warm-up."""
    seeds = pd.read_csv(os.path.join(paper1, "p1_seeds.csv"))
    yearly = pd.read_csv(os.path.join(paper1, "p1_warmup_yearly.csv"))
    fig, axes, W, H = _fig(15.0)
    # A: the relative SE of the mean at 16 and 40 seeds, no vaccine
    s = seeds[seeds.scenario == "vacc_none"].reset_index(drop=True)
    ax = axes(56.0, 82.0, 52.0, 58.0)
    _xgrid(ax, [0, 1, 2, 3, 4, 5])
    for i, r in s.iterrows():
        ax.plot([100 * r.se_rel_40, 100 * r.se_rel_16], [i, i], color=RAMP[150], lw=2, zorder=1)
        ax.scatter([100 * r.se_rel_16], [i], s=16, facecolor="white", edgecolor=BLUE, lw=1, zorder=3)
        ax.scatter([100 * r.se_rel_40], [i], s=16, color=BLUE, zorder=3)
        ax.text(5.6, i, f"{int(r.n_stable)}", va="center", ha="right", fontsize=FONT_PT, color=INK)
    ax.text(5.6, -1.0, "n*", va="bottom", ha="right", fontsize=FONT_PT, fontstyle="italic")
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels([OUTPUT_LABELS.get(o, o) for o in s.output])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(len(s) - 0.5, -1.2)
    ax.set_xlim(0, 5.8)
    ax.set_xlabel("Standard error of the mean (% of the mean)")
    ax.legend(handles=[Line2D([], [], ls="", marker="o", ms=4, mfc="white", mec=BLUE, label="16 seeds"),
                       Line2D([], [], ls="", marker="o", ms=4, color=BLUE, label="40 seeds")],
              loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, borderaxespad=0.2)
    letter(ax, "A", fig=fig, x=2 / W, y=(H - 2) / H)
    # B: the vaccine effect's SE against the number of seeds
    e = seeds[seeds.scenario == "vaccine effect"]
    ax = axes(136.0, 90.0, 48.0, 48.0)
    for out_, lab, mk in (("rota_symptomatic_episodes_per_child_year_0_24m", "Episodes, 0–24 mo", "o"),
                          ("sev59", "Severe episodes, 0–59 mo", "s")):
        g = e[e.output == out_].sort_values("n")
        ax.plot(g.n, g.effect_se_pct, color=BLUE, lw=1.4, marker=mk, ms=4, label=lab)
    nn = np.linspace(8, 40, 50)
    ref = e[(e.output == "sev59") & (e.n == 8)].effect_se_pct.iloc[0]
    ax.plot(nn, ref * np.sqrt(8 / nn), color=MUTED, lw=0.8, ls=(0, (3, 2)), label="Falling as 1/√n")
    ax.set_xticks([8, 16, 24, 32, 40])
    ax.set_ylim(0, None)
    ax.set_xlabel("Seeds")
    ax.set_ylabel("SE of the vaccine's effect (points)")
    ax.legend(loc="upper right", borderaxespad=0.2, handlelength=1.6)
    letter(ax, "B", fig=fig, x=118 / W, y=(H - 2) / H)
    # C: the warm-up, infections 0-24 months by year relative to years 4-10
    names = {"rota_infections_0_24m": "rota", "campy_infections_0_24m": "campy", "crypto_infections_0_24m": "crypto"}
    for j, (o, p) in enumerate(names.items()):
        ax = axes(20.0 + j * 58.0, 14.0, 46.0, 44.0)
        ax.axvspan(2.5, 3.5, color="#f6f5f2", zorder=0)
        ax.axhline(100, color=HAIR, lw=0.6, zorder=0)
        for sc, ls, lab in (("long_vacc_none", "-", "No vaccine"), ("long_vacc_default", (0, (4, 2)), "Today's vaccine")):
            g = yearly[(yearly.output == o) & (yearly.scenario == sc)]
            later = g[g.year >= 4].groupby("seed").value.mean()
            rel = 100 * g.pivot(index="seed", columns="year", values="value").div(later, axis=0)
            m, se = rel.mean(), rel.std(ddof=1) / np.sqrt(len(rel))
            ax.fill_between(m.index, m - 2 * se, m + 2 * se, color=RAMP[100], lw=0, zorder=1)
            ax.plot(m.index, m, color=BLUE, lw=1.4, ls=ls, marker="o", ms=2.5, zorder=3, label=lab)
        ax.set_xticks([1, 2, 3, 4, 6, 8, 10])
        ax.set_xlim(0.6, 10.4)
        ax.set_title(PATHOGEN_NAMES[p], fontsize=FONT_PT, fontstyle="italic", fontweight="bold", loc="left")
        ax.set_xlabel("Year of the run")
        if j == 0:
            ax.set_ylabel("Infections, 0–24 mo\n(% of years 4–10)")
            ax.legend(loc="upper right", borderaxespad=0.2, handlelength=2.0)
            letter(ax, "C", fig=fig, x=2 / W, y=(14.0 + 44.0 + 10.0) / H)
        ax.text(3, ax.get_ylim()[1], "measured", ha="center", va="top", fontsize=FONT_PT, color=INK2)
    return save(fig, "S1_Fig_seeds_warmup", out)


MORRIS_GROUPS = (
    ("Decisions", [("cpt_alpha_centre", "Weight on wealth vs health (α)"),
                   ("cpt_gamma_centre", "Probability weighting (γ)"),
                   ("cpt_lambda_upper", "Loss aversion (λ)"),
                   ("cpt_theta_eta", "Value curvature (θ, η)"),
                   ("steering_parameters.child_health_weight", "Weight on the child's health"),
                   ("steering_parameters.natural_worsening_prob", "Untreated episode worsens"),
                   ("steering_parameters.parent_stress_health_impact", "Parent's stress from illness"),
                   ("steering_parameters.untreated_severity_penalty", "Severity penalty, untreated"),
                   ("economy.wealth_scale_days", "Wealth scale")]),
    ("Treatment", [("steering_parameters.treatment_success_prob", "Treatment succeeds"),
                   ("steering_parameters.treatment_harm_reduction", "Treatment's harm reduction"),
                   ("pathogens[campy].treatment_clears_infection_prob", "Campylobacter: treatment clears")]),
    ("Natural history", [("pathogens[rota].exposure_period", "Rotavirus: latent period"),
                         ("pathogens[campy].exposure_period", "Campylobacter: latent period"),
                         ("pathogens[crypto].exposure_period", "Cryptosporidium: latent period"),
                         ("pathogens[rota].recovery_rate", "Rotavirus: recovery rate"),
                         ("pathogens[campy].recovery_rate", "Campylobacter: recovery rate"),
                         ("pathogens[crypto].recovery_rate", "Cryptosporidium: recovery rate")]),
    ("Immunity", [("pathogens[rota].immunity_decay_per_infection", "Rotavirus: susceptibility after infection"),
                  ("steering_parameters.prior_infection_immunity_factor",
                   "Campylobacter: susceptibility after infection"),
                  ("pathogens[crypto].immunity_decay_per_infection",
                   "Cryptosporidium: susceptibility after infection"),
                  ("pathogens[rota].symptomatic_reduction_per_prior_infection", "Rotavirus: fewer symptoms after"),
                  ("pathogens[crypto].symptomatic_reduction_per_prior_infection",
                   "Cryptosporidium: fewer symptoms after"),
                  ("illness_mechanics.severity_reduction_per_infection", "Lower severity after infection")]),
    ("Environment and contacts", [("pathogens[rota].water_decay_rate", "Rotavirus: die-off in water"),
                                  ("pathogens[crypto].oocyst_decay_rate", "Cryptosporidium: die-off in water"),
                                  ("campy_beta_poisson_alpha", "Campylobacter: dose-response shape"),
                                  ("steering_parameters.max_contacts_per_visit", "Contacts per visit")]),
)


def s2_morris(paper1: str = PAPER1, out: str = OUT) -> list[str]:
    """S2 Fig: Morris screening of the 28 fixed parameters against the 41 targets."""
    tab = pd.read_csv(os.path.join(paper1, "p1_morris.csv"))
    tab = tab[tab.scale == "target tolerance"]
    rows = _rows(MORRIS_GROUPS)
    if {k for _, k in rows if k} != set(tab.factor):
        raise ValueError("the figure's factor list differs from the table's")
    row_mm = 4.2
    fig, axes, W, H = _fig(row_mm * (len(rows) + 0.7) / 10 + 2.6)
    ax = axes(66.0, 14.0, 110.0, row_mm * (len(rows) + 0.7))
    _label_rows(ax, rows)
    ax.spines["left"].set_visible(False)
    ax.set_xscale("log")
    zero_x = 0.0012                                       # effects of exactly 0 (the output never moves)
    ax.set_xlim(0.0009, 6)
    ax.axvspan(0.0009, 0.0017, color="#f6f5f2", zorder=0)
    ax.axvline(1, color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=1)
    yi = {k: i for i, (_, k) in enumerate(rows) if k}
    noise = tab[~tab.above_noise]
    real = tab[tab.above_noise]
    ax.scatter(noise.mu_star.where(noise.mu_star > 0, zero_x).clip(lower=zero_x), noise.factor.map(yi), s=6,
               facecolor="none", edgecolor=MUTED, lw=0.5, zorder=2)
    ax.set_xticks([zero_x, 0.01, 0.1, 1])
    ax.set_xticklabels(["0", "0.01", "0.1", "1"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.LogLocator(subs=np.arange(2, 10) * 0.1, numticks=20))
    ax.set_xticks([x for x in ax.get_xticks(minor=True) if x > 0.002], minor=True)
    ax.scatter(real.mu_star, real.factor.map(yi), s=16, color=BLUE, edgecolor="white", lw=0.4, zorder=3)
    ax.set_xlabel("Effect across the parameter's range, μ*Δ (in units of the target's tolerance)")
    ax.legend(handles=[Line2D([], [], ls="", marker="o", ms=4, color=BLUE, label="Above the seeds' noise (μ* > 2 floors)"),
                       Line2D([], [], ls="", marker="o", ms=3, mfc="none", mec=MUTED, label="Within the noise")],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.3)
    ax.text(1.06, -0.55, "one tolerance", fontsize=FONT_PT, color=INK2, va="center")
    return save(fig, "S2_Fig_morris", out)


def s3_docking(docking: str | None = None, out: str = OUT) -> list[str]:
    """S3 Fig: the model's rotavirus code against its SEIR equations."""
    d = docking or os.path.join("experiments", "outputs", "verification", "seir_docking")
    curves = pd.read_csv(os.path.join(d, "curves.csv"))
    summ = pd.read_csv(os.path.join(d, "summary.csv"))
    fig, axes, W, H = _fig(7.6)
    ax = axes(20.0, 12.0, 92.0, 56.0)
    shades = [RAMP[k] for k in (150, 250, 400, 550, 700)]
    for colour, (r0, g) in zip(shades, curves.groupby("R0")):
        g = g[g.day <= 120]
        ax.fill_between(g.day, g.new_model_mean - g.new_model_sd, g.new_model_mean + g.new_model_sd, color=colour,
                        alpha=0.35, lw=0)
        ax.plot(g.day, g.new_model_mean, color=colour, lw=1.4, label=f"R0 = {r0:g}")
        ax.plot(g.day, g.new_mean_field, color=INK, lw=0.7, ls=(0, (2, 1.5)))
    ax.plot([], [], color=INK, lw=0.7, ls=(0, (2, 1.5)), label="SEIR equations")
    ax.set_xlim(0, 120)
    ax.set_ylim(0, None)
    ax.set_xlabel("Day")
    ax.set_ylabel("New infections a day")
    ax.legend(loc="upper right", borderaxespad=0.2, handlelength=1.8)
    letter(ax, "A", fig=fig, x=2 / W, y=(H - 2) / H)
    ax = axes(136.0, 12.0, 48.0, 56.0)
    r = np.linspace(1.05, 5.2, 200)
    from scipy.optimize import brentq
    fs = [brentq(lambda z: z - (1 - np.exp(-x * z)), 1e-9, 1.0) for x in r]
    ax.plot(r, fs, color=INK, lw=0.8, label="Final-size relation")
    se = summ.attack_sd / np.sqrt(summ.reps)
    ax.errorbar(summ.R0, summ.attack_mean, yerr=2 * se, color=BLUE, lw=0, elinewidth=1.0, marker="o", ms=4,
                label=f"Model, {int(summ.reps.iloc[0])} runs each (± 2 SE)")
    ax.set_xlabel("R0")
    ax.set_ylabel("Final attack rate")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", borderaxespad=0.2)
    letter(ax, "B", fig=fig, x=118 / W, y=(H - 2) / H)
    return save(fig, "S3_Fig_docking", out)


def s4_emulators(diag: str = DIAG, out: str = OUT) -> list[str]:
    """S4 Fig: the emulators of waves 1-3 predicting wave 4's runs, per target."""
    v = pd.read_csv(os.path.join(diag, "validation.csv")).set_index("target")
    rows = _rows(TARGET_GROUPS)
    row_mm = 3.3
    fig, axes, W, H = _fig(row_mm * (len(rows) + 0.7) / 10 + 2.2)
    h = row_mm * (len(rows) + 0.7)
    axA, axB = axes(62.0, 14.0, 54.0, h), axes(130.0, 14.0, 54.0, h)
    for ax in (axA, axB):
        _label_rows(ax, rows)
        ax.spines["left"].set_visible(False)
    axB.set_yticklabels([])
    yi = {k: i for i, (_, k) in enumerate(rows) if k}
    _xgrid(axA, [0, 0.5, 1, 1.5])
    axA.axvline(1, color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=1)
    axA.scatter(v.rms_standardised_error, v.index.map(yi), s=14, color=BLUE, zorder=3)
    axA.set_xlim(0, 1.75)
    axA.set_xlabel("Root mean square standardised\nerror (1: uncertainty right)")
    _xgrid(axB, [0, 0.5, 1])
    axB.scatter(v.r2.clip(lower=-0.45), v.index.map(yi), s=14, color=BLUE, zorder=3)
    for t, r in v.r2.items():
        if r < -0.45:
            axB.annotate(f"{r:.1f}", (-0.45, yi[t]), xytext=(4, 0), textcoords="offset points", va="center",
                         fontsize=FONT_PT, color=INK2)
    axB.set_xlim(-0.5, 1.05)
    axB.set_xlabel("Share of wave 4's variance\nexplained (R²)")
    letter(axA, "A", fig=fig, x=2 / W, y=(H - 2) / H)
    letter(axB, "B", fig=fig, x=120 / W, y=(H - 2) / H)
    return save(fig, "S4_Fig_emulators", out)


def s5_sobol(diag: str = DIAG, out: str = OUT) -> list[str]:
    """S5 Fig: the emulators' total Sobol indices, targets by calibrated parameters."""
    s = pd.read_csv(os.path.join(diag, "sobol.csv"))
    t_rows, p_rows = _rows(TARGET_GROUPS), _rows(PARAM_GROUPS)
    p_keys = [k for _, k in p_rows if k]
    p_labels = {k: lab for lab, k in p_rows if k}
    p_group = {}
    for g, items in PARAM_GROUPS:
        for k, _ in items:
            p_group[k] = g
    M = np.full((len(t_rows), len(p_keys)), np.nan)
    ti = {k: i for i, (_, k) in enumerate(t_rows) if k}
    for _, r in s.iterrows():
        M[ti[r.target], p_keys.index(r.parameter)] = min(max(r.ST, 0.0), 1.0)
    cell = 3.15
    fig, axes, W, H = _fig(MAX_HEIGHT / CM - 0.3)
    w, h = cell * len(p_keys), cell * len(t_rows)
    ax = axes(190.5 - 4.0 - w, 6.0, w, h)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("st", ["#ffffff", RAMP[100], RAMP[250], RAMP[400], RAMP[550], RAMP[700]])
    cmap.set_bad("#f2f1ee")
    ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(t_rows)))
    ax.set_yticklabels([lab for lab, _ in t_rows])
    for tl, (_, k) in zip(ax.get_yticklabels(), t_rows):
        if k is None:
            tl.set_fontweight("bold")
            tl.set_color(INK)
    for i, (_, k) in enumerate(t_rows):
        if k is None:
            ax.axhspan(i - 0.5, i + 0.5, color="white", zorder=2)
    ax.set_xticks(range(len(p_keys)))
    ax.set_xticklabels([p_labels[k] for k in p_keys], rotation=90)
    ax.xaxis.tick_top()
    prev = None
    for j, k in enumerate(p_keys):
        if prev is not None and p_group[k] != p_group[prev]:
            ax.axvline(j - 0.5, color=INK2, lw=0.6)
        prev = k
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cax = axes(4.0, 8.0, 3.0, 40.0)
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=cmap), cax=cax)
    cb.set_label("Total Sobol index")
    cb.outline.set_visible(False)
    return save(fig, "S5_Fig_sobol", out)


FIGURES = {"calibration": fig_calibration, "observation": fig_observation, "validation": fig_validation,
           "routes": fig_routes, "decisions": fig_decisions, "s1": s1_seeds, "s2": s2_morris, "s3": s3_docking,
           "s4": s4_emulators, "s5": s5_sobol}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="paper 1's figures")
    ap.add_argument("figures", nargs="*", help=f"some of {', '.join(FIGURES)} (default: all)")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)
    bad = sorted(set(args.figures) - set(FIGURES))
    if bad:
        ap.error(f"unknown figures {bad}; choose from {list(FIGURES)}")
    style()
    for name in args.figures or FIGURES:
        print(name, FIGURES[name](out=args.out))


if __name__ == "__main__":
    main()
