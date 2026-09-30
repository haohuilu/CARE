"""Figures for the paper, drawn from the analysis CSVs in ``RESULTS_DIR``.

Run ``python -m care.figures`` after ``care.analysis`` has produced
``sar_main.csv``.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from . import config as cfg

DATASET_LABEL = {"pairs_medqa_race": "MedQA race", "pairs_medqa_gender": "MedQA sex",
                 "pairs_medmcqa_race": "MedMCQA race", "pairs_medmcqa_gender": "MedMCQA sex"}
MARKER = {"pairs_medqa_race": "o", "pairs_medqa_gender": "s",
          "pairs_medmcqa_race": "^", "pairs_medmcqa_gender": "D"}
MODELS = ["MedGemma-4B", "Med42-8B", "Gemma-4-E4B", "Llama-3.1-8B"]
COLOR = {"MedGemma-4B": "#8c6d31", "Med42-8B": "#1b3a6b",
         "Gemma-4-E4B": "#6a9fb5", "Llama-3.1-8B": "#c1121f"}
GROUP_COLOR = {"white": "#1b3a6b", "male": "#1b3a6b", "black": "#c1121f", "female": "#c1121f"}
PROMPT_LS = {"P1": "-", "P2": ":", "P3": "--"}


def apply_style() -> None:
    """Serif, 9 pt, light grid: the style used for every figure."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Liberation Serif"],
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "axes.axisbelow": True, "lines.linewidth": 1.2, "lines.markersize": 4,
    })


def _models(df: pd.DataFrame) -> list:
    """Models present in the data, in the paper's order first."""
    present = list(dict.fromkeys(df.model))
    return [m for m in MODELS if m in present] + [m for m in present if m not in MODELS]


def _save(fig, name: str) -> str:
    os.makedirs(cfg.FIGS_DIR, exist_ok=True)
    path = os.path.join(cfg.FIGS_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    return path


def gap_forest(sar: pd.DataFrame, alpha: float = 0.10) -> str:
    """Difference in SAR between groups with 95% intervals, one panel per prompt."""
    apply_style()
    t = sar[(sar.method == "tlp") & (sar.alpha == alpha)]
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.4), sharey=True)
    for ax, prompt in zip(axes, ("P1", "P2", "P3")):
        sub, y, labels = t[t.prompt == prompt], 0, []
        for d in list(DATASET_LABEL)[::-1]:
            for m in _models(t)[::-1]:
                row = sub[(sub.dataset == d) & (sub.model == m)]
                if row.empty:
                    continue
                r = row.iloc[0]
                col = "#c1121f" if (r.sar_gap_lo > 0 or r.sar_gap_hi < 0) else "0.45"
                ax.plot([r.sar_gap_lo, r.sar_gap_hi], [y, y], color=col, lw=1.2)
                ax.plot([r.sar_gap], [y], "o", ms=3.4, color=col)
                labels.append(f"{DATASET_LABEL[d]} · {m}")
                y += 1
        ax.axvline(0, color="0.2", lw=0.8)
        ax.set_title(prompt)
        ax.set_xlabel("SAR gap (reference $-$ demographic)")
        ax.set_xlim(-0.12, 0.16)
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(labels, fontsize=6.5)
    axes[0].set_ylim(-0.8, len(labels) - 0.2)
    fig.tight_layout()
    return _save(fig, "fig_gap_forest.pdf")


def tlp_vs_ce(sar: pd.DataFrame, alpha: float = 0.10) -> str:
    """Token-probability vs elicited confidence: SAR scatter and where a threshold exists."""
    apply_style()
    a = sar[sar.alpha == alpha]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.2), gridspec_kw={"width_ratios": [1, 1.05]})
    p = a[a.prompt == "P1"].pivot_table(index=["dataset", "model"], columns="method",
                                        values="sar_ref").reset_index()
    models = _models(a)
    for col in ("tlp", "ce"):
        if col not in p:
            p[col] = np.nan
    for _, r in p.iterrows():
        ax1.scatter(r.tlp, r.ce, s=26, color=COLOR.get(r.model, "0.5"), marker=MARKER[r.dataset],
                    edgecolor="white", linewidth=0.5, zorder=3)
    ax1.plot([0, 1], [0, 1], color="0.65", lw=0.8, ls=(0, (3, 3)), zorder=1)
    ax1.set_xlim(-0.03, 1.03)
    ax1.set_ylim(-0.03, 1.03)
    ax1.set_xlabel("SAR, token probability")
    ax1.set_ylabel("SAR, elicited confidence")
    ax1.set_title("(a) Same answers, two confidence sources", fontsize=8.5)
    handles = [Line2D([], [], marker="o", ls="", color=COLOR.get(m, "0.5"), label=m, markersize=5)
               for m in models]
    handles += [Line2D([], [], marker=MARKER[d], ls="", color="0.4", label=l, markersize=4.5)
                for d, l in DATASET_LABEL.items()]
    ax1.legend(handles=handles, fontsize=5.8, loc="upper left", frameon=False,
               handletextpad=0.3, borderaxespad=0.2)
    feas = (a.assign(ok=(a.sar_ref > 0) & (a.sar_dem > 0)).groupby(["model", "method"]).ok.sum()
            .unstack().reindex(index=models, columns=["tlp", "ce"]).fillna(0))
    x, w = np.arange(len(models)), 0.36
    for off, meth, col, lab in ((-w / 2, "tlp", "#1b3a6b", "token probability"),
                                (w / 2, "ce", "#bdbdbd", "elicited confidence")):
        bars = ax2.bar(x + off, [feas.loc[m, meth] for m in models], w, color=col, label=lab)
        for b in bars:
            ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.2, f"{int(b.get_height())}",
                     ha="center", va="bottom", fontsize=6.5)
    ax2.set_xticks(x)
    n_combos = int(a.groupby("model").apply(lambda g: g[g.method == "tlp"][["dataset", "prompt"]]
                                            .drop_duplicates().shape[0], include_groups=False).max())
    ax2.set_xticklabels(models, fontsize=6.8, rotation=12)
    ax2.set_ylim(0, n_combos + 1)
    ax2.set_ylabel(f"combinations with safe automation\nfor both groups (of {n_combos})")
    ax2.set_title("(b) Where a safe threshold exists", fontsize=8.5, pad=18)
    ax2.legend(fontsize=6.5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout()
    return _save(fig, "fig_tlp_vs_ce.pdf")


def sar_curves(curves: pd.DataFrame, model: str, method: str = "tlp") -> str:
    """SAR against the error budget, per group and prompt, one panel per dataset."""
    apply_style()
    d = curves[(curves.model == model) & (curves.method == method)]
    datasets = [k for k in DATASET_LABEL if k in set(d.dataset)]
    fig, axes = plt.subplots(1, len(datasets), figsize=(1.9 * len(datasets) + 0.6, 2.5),
                             sharey=True, squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        s = d[d.dataset == ds]
        for prompt in ("P1", "P3"):
            for g in cfg.DATASETS[ds]["groups"]:
                t = s[(s.prompt == prompt) & (s.group == g)].sort_values("alpha")
                ax.plot(t.alpha, t.sar, color=GROUP_COLOR[g], ls=PROMPT_LS[prompt],
                        label=f"{g}, {prompt}")
        ax.axvline(0.10, color="0.6", lw=0.7, ls=(0, (2, 2)))
        ax.set_title(DATASET_LABEL[ds])
        ax.set_xlabel(r"error budget $\alpha$")
        ax.set_ylim(-0.02, 1.02)
    axes[0][0].set_ylabel("safe automation rate")
    axes[0][-1].legend(loc="lower right", fontsize=6.5)
    fig.suptitle(model, y=1.02, fontsize=9)
    fig.tight_layout()
    return _save(fig, f"fig_sar_curves_{model}_{method}.pdf")


def main() -> int:
    sar = pd.read_csv(os.path.join(cfg.RESULTS_DIR, "sar_main.csv"))
    curves = pd.read_csv(os.path.join(cfg.RESULTS_DIR, "curves_main.csv"))
    print(gap_forest(sar))
    print(tlp_vs_ce(sar))
    for m in _models(curves):
        print(sar_curves(curves, m))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
