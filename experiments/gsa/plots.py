# experiments/gsa/plots.py
"""Diagnostic plots for the GSA stages (matplotlib, saved to PNG)."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _short(path: str) -> str:
    """Compact label from a config dot-path, e.g. pathogens[rota].x -> rota.x."""
    return path.replace("pathogens[", "").replace("]", "").replace("steering_parameters.", "") \
               .replace("illness_mechanics.", "im.").replace("_", " ")


def plot_morris(morris_df, metric: str, out_path: str, top: int = 20):
    """mu* (importance) vs sigma (interactions/nonlinearity); label the top params."""
    d = morris_df.head(top)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(d["mu_star"], d["sigma"], s=30, c="#2196F3")
    for _, r in d.iterrows():
        ax.annotate(_short(r["path"]), (r["mu_star"], r["sigma"]),
                    fontsize=7, xytext=(4, 2), textcoords="offset points")
    ax.set_xlabel("mu*  (mean |elementary effect| - importance)")
    ax.set_ylabel("sigma  (spread of effects - interactions / nonlinearity)")
    ax.set_title(f"Morris screening: {metric}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_sobol(sobol_df, metric: str, out_path: str, top: int = 15):
    """Horizontal bars of ST and S1 (with conf whiskers) for the top params."""
    d = sobol_df.head(top).iloc[::-1]  # largest at top
    y = range(len(d))
    fig, ax = plt.subplots(figsize=(8, max(3, 0.42 * len(d))))
    ax.barh([i + 0.2 for i in y], d["ST"], height=0.4, xerr=d["ST_conf"],
            color="#1565C0", label="ST (total)", error_kw={"elinewidth": 0.8})
    ax.barh([i - 0.2 for i in y], d["S1"], height=0.4, xerr=d["S1_conf"],
            color="#90CAF9", label="S1 (first-order)", error_kw={"elinewidth": 0.8})
    ax.set_yticks(list(y))
    ax.set_yticklabels([_short(p) for p in d["path"]], fontsize=8)
    ax.set_xlabel("Sobol index")
    ax.set_title(f"Sobol indices: {metric}")
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
