"""drift_figures.py: the figures of plan.md section 8, drawn from results/drift_*.csv.

Usage: uv run python -m scripts.drift_figures   (writes results/figures/*.png)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.drift_windows import PUBLIC_A, STUDY_B

OUT = Path(__file__).resolve().parent.parent / "results"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
TFM, BOOSTER = "#2a78d6", "#eb6834"   # categorical slots 1 and 2, checked with the dataviz validator
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
                     "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlelocation": "left"})
SIGN = {"H1": "positive: the TFM lead grows", "H2": "positive: a refit helps TFMs more",
        "H3": "positive: the refit beyond an intercept\nhelps TFMs less (log loss)",
        "H4": "positive: TFMs lose ground (log loss)", "H5": "positive: TFMs recover a larger share"}
TITLES = {"H1": "H1: lead in ROC AUC\n(covariate minus control)", "H2": "H2: lead in refit gain, AUC\n(covariate minus control)",
          "H3": "H3: lead in S1 - S0*, log loss\n(prior minus control)", "H4": "H4: lead in log loss\n(covariate minus control)",
          "H5": "H5: share difference, region\n(TFM minus booster)"}


def save(fig, name: str) -> None:
    (OUT / "figures").mkdir(exist_ok=True)
    fig.savefig(OUT / "figures" / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def primary() -> None:
    """Per-dataset value of each primary test, with the Hodges-Lehmann estimate and its 95 % interval."""
    v, t = pd.read_csv(OUT / "drift_values.csv").set_index("dataset"), pd.read_csv(OUT / "drift_primary.csv").set_index("test")
    fig, axes = plt.subplots(1, 5, figsize=(13, 3.2))
    for ax, k in zip(axes, TITLES):
        d = v[k].dropna().sort_values()
        x = d.to_numpy()
        ax.scatter(x, np.arange(len(x)) + 1, s=22, color=MUTED, zorder=3)
        for i, name in enumerate(d.index):
            if name == "hill-valley-with-noise":
                ax.annotate("hill-valley", (x[i], i + 1), xytext=(5, 0), textcoords="offset points", fontsize=7, color=MUTED, va="center")
        r = t.loc[k]
        ax.axvline(0, color=INK, lw=0.8)
        ax.plot([r.lo, r.hi], [0, 0], color=TFM, lw=3, solid_capstyle="round", zorder=4)
        ax.scatter([r.est], [0], s=40, marker="D", color=TFM, edgecolor=SURFACE, linewidth=1.5, zorder=5)
        ax.set_yticks([])
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.set_title(TITLES[k])
        ax.set_xlabel(f"n = {int(r.n_nonzero)}, p = {r.p:.2f}, Holm {r.holm_p:.2f}\n{SIGN[k]}")
    fig.tight_layout()
    fig.text(0.01, -0.1, "grey dots: one dataset each, ordered by value (each panel has its own x scale); "
             "blue diamond and bar: Hodges-Lehmann estimate and 95 % interval",
             fontsize=7.5, color=MUTED)
    save(fig, "primary_tests")


def lead_by_block() -> None:
    """The change of the S0 lead in ROC AUC since T1 under every arm, one line per Study B dataset.
    hill-valley-with-noise is left out: its lead (0.4, boosters at chance) falls by 0.24 to 0.45 in the noise,
    swap and region arms and would shrink the other lines to a band. thyroid-sick-euthyroid has no concept arms."""
    d = pd.read_csv(OUT / "drift_leads.csv")
    d = d[(d.strategy == "S0") & (d.metric == "auc") & d.dataset.isin(STUDY_B) & (d.dataset != "hill-valley-with-noise")]
    arms = {"none": "control", "cov": "covariate", "prior0.5": "prior, r = 0.5", "prior2": "prior, r = 2",
            "noise": "concept: noise", "swap": "concept: swap", "region": "concept: region"}
    fig, axes = plt.subplots(1, 7, figsize=(14, 2.8), sharey=True)
    for ax, (arm, title) in zip(axes, arms.items()):
        a = d[d.scenario == arm].pivot(index="block", columns="dataset", values="lead")
        a = a - a.loc[1]
        ax.plot(a.index, a.to_numpy(), color=MUTED, lw=0.8, alpha=0.45)
        ax.plot(a.index, a.median(axis=1), color=TFM, lw=2.2)
        ax.axhline(0, color=INK, lw=0.6)
        ax.set_xticks([1, 2, 3, 4], ["T1", "T2", "T3", "T4"])
        ax.set_title(title)
    axes[0].set_ylabel("change of TFM minus booster\nROC AUC (S0), since T1")
    fig.text(0.01, -0.1, "grey: one dataset; blue: median over nine datasets (eight in the concept arms), hill-valley-with-noise left out.\n"
             "The x axis is the block, not the dose; the dose grows with the block (T4 = full dose). In the concept arms a stale model "
             "loses ROC AUC on the relabeled rows, and a sharper model loses more.", fontsize=7.5, color=MUTED)
    fig.tight_layout()
    save(fig, "lead_by_block")


def h5() -> None:
    """Recovered share of each family in the region arm; datasets failing the guard are left out."""
    c = pd.read_csv(OUT / "drift_concept.csv")
    c = c[(c.scenario == "region") & c.guard].sort_values("diff")
    fig, ax = plt.subplots(figsize=(6, 4))
    y = np.arange(len(c))
    ax.hlines(y, c.booster_share, c.tfm_share, color=GRID, lw=3, zorder=1)
    ax.scatter(c.tfm_share, y, s=36, color=TFM, zorder=3, label="TFMs")
    ax.scatter(c.booster_share, y, s=36, color=BOOSTER, zorder=3, label="boosters")
    ax.set_yticks(y, c.dataset)
    ax.set_xlabel("share of own oracle gain recovered by S1 at T4 (ROC AUC)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    ax.set_title("Region arm: recovered share, ordered by difference")
    fig.text(0.01, -0.12, f"{len(c)} datasets past the oracle-gain guard; hill-valley-with-noise fails it (boosters at chance) and is left out.\n"
             "Hodges-Lehmann estimate of the TFM minus booster difference: see drift_primary.csv (H5).", fontsize=7.5, color=MUTED)
    save(fig, "h5_shares")


def study_a() -> None:
    """Study A, ROC AUC over T1 to T4: family means under S0 (with its bootstrap band), S1 and S2."""
    c = pd.read_csv(OUT / "drift_study_a_curves.csv")
    c = c[(c.metric == "auc") & c.unit.isin(["tfm", "booster", "tfm_no_tabfm"]) & c.strategy.isin(["S0", "S1", "S2"])]
    fig, axes = plt.subplots(3, 3, figsize=(11, 8), sharex=True)
    for ax, ds in zip(axes.flat, PUBLIC_A):
        for unit, color in (("tfm", TFM), ("booster", BOOSTER)):
            d = c[(c.dataset == ds) & (c.unit == unit)]
            s0 = d[d.strategy == "S0"].set_index("block")
            ax.fill_between(s0.index, s0.lo, s0.hi, color=color, alpha=0.18, lw=0)
            ax.plot(s0.index, s0.est, color=color, lw=2)
            for st, ls in (("S1", "--"), ("S2", ":")):
                x = d[d.strategy == st].set_index("block").est
                ax.plot([1, *x.index], [s0.est[1], *x], color=color, lw=1.4, ls=ls)
        x = c[(c.dataset == ds) & (c.unit == "tfm_no_tabfm") & (c.strategy == "S2")].set_index("block").est
        ax.plot([1, *x.index], [c[(c.dataset == ds) & (c.unit == "tfm") & (c.strategy == "S0") & (c.block == 1)].est.iloc[0], *x],
                color=TFM, lw=1.4, ls="-.")
        ax.set_xticks([1, 2, 3, 4], ["T1", "T2", "T3", "T4"])
        ax.set_title(ds)
    for ax in axes.flat[len(PUBLIC_A):]:
        ax.set_axis_off()
    handles = [plt.Line2D([], [], color=TFM, lw=2), plt.Line2D([], [], color=BOOSTER, lw=2),
               plt.Line2D([], [], color=MUTED, lw=2), plt.Line2D([], [], color=MUTED, lw=1.4, ls="--"),
               plt.Line2D([], [], color=MUTED, lw=1.4, ls=":"), plt.Line2D([], [], color=TFM, lw=1.4, ls="-.")]
    fig.legend(handles, ["TFMs", "boosters", "S0 (band: 95 % bootstrap of rows)", "S1", "S2",
                         "S2, TFMs without TabFM (its context is capped at 5,000 rows)"], ncol=3, frameon=False,
               loc="lower center", bbox_to_anchor=(0.5, 0.025))
    fig.text(0.01, 0.005, "Each panel has its own y scale. The bands resample rows independently and ignore serial dependence "
             "(elec in particular); at T1 S1 and S2 are S0.", fontsize=7.5, color=MUTED)
    fig.supylabel("ROC AUC", color=MUTED)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, "study_a_curves")


def study_a_lead() -> None:
    """Study A, the lead (TFMs minus boosters) in ROC AUC with its own paired bootstrap band, by strategy."""
    c = pd.read_csv(OUT / "drift_study_a_curves.csv")
    c = c[(c.metric == "auc") & (c.unit == "lead") & c.strategy.isin(["S0", "S1", "S2"])]
    fig, axes = plt.subplots(3, 3, figsize=(11, 8), sharex=True)
    for ax, ds in zip(axes.flat, PUBLIC_A):
        for st, color in (("S0", MUTED), ("S1", "#1baf7a"), ("S2", "#6250d6")):  # not the family colors
            d = c[(c.dataset == ds) & (c.strategy == st)].set_index("block")
            ax.fill_between(d.index, d.lo, d.hi, color=color, alpha=0.18, lw=0)
            ax.plot(d.index, d.est, color=color, lw=1.8)
        ax.axhline(0, color=INK, lw=0.6)
        ax.set_xticks([1, 2, 3, 4], ["T1", "T2", "T3", "T4"])
        ax.set_title(ds)
    for ax in axes.flat[len(PUBLIC_A):]:
        ax.set_axis_off()
    handles = [plt.Line2D([], [], color=col, lw=2) for col in (MUTED, "#1baf7a", "#6250d6")]
    fig.legend(handles, ["S0", "S1", "S2"], ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.supylabel("TFMs minus boosters, ROC AUC", color=MUTED)
    fig.text(0.01, -0.04, "Band: 95 % percentile interval of a paired bootstrap of rows (the same resample for every model); "
             "T1 has S0 only. Each panel has its own y scale.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "study_a_lead")


if __name__ == "__main__":
    primary()
    lead_by_block()
    h5()
    study_a()
    study_a_lead()
