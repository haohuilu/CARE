"""LaTeX tables for the paper, written from the analysis CSVs in ``RESULTS_DIR``.

Run ``python -m care.tables`` after ``care.analysis``; every number in the
tables comes from ``sar_main.csv``, ``contrast_main.csv`` and ``sar_appendix.csv``.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg

RES = Path(cfg.RESULTS_DIR)
OUT = Path(cfg.TABLES_DIR)

DS = {"pairs_medqa_race": "MedQA, race (White vs Black)",
      "pairs_medqa_gender": "MedQA, sex (male vs female)",
      "pairs_medmcqa_race": "MedMCQA, race (White vs Black)",
      "pairs_medmcqa_gender": "MedMCQA, sex (male vs female)"}
MODELS = ["MedGemma-4B", "Med42-8B", "Gemma-4-E4B", "Llama-3.1-8B"]
API_ORDER = [("GPT-6-Sol", "ce"), ("GPT-5.6-Terra", "ce"), ("Claude-Sonnet-5", "ce"),
             ("GPT-4.1-mini", "tlp"), ("GPT-4.1-mini", "ce")]


def present(df, order):
    """Models in ``df``, the paper's order first, then any others."""
    have = list(dict.fromkeys(df.model))
    return [m for m in order if m in have] + [m for m in have if m not in order]


def f3(x):
    return "--" if pd.isna(x) else f"{x:.3f}"


def pair(a, b):
    return f"{f3(a)} / {f3(b)}"


def signed(x):
    if pd.isna(x):
        return "--"
    if round(float(x), 3) == 0:
        return "0.000"
    return f"{x:+.3f}".replace("-", "$-$")


def ci(lo, hi):
    return f"[{signed(lo)}, {signed(hi)}]"


def write(name, lines):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text("\n".join(lines) + "\n")


def main() -> int:
    sar = pd.read_csv(RES / "sar_main.csv")
    con = pd.read_csv(RES / "contrast_main.csv")
    api = pd.read_csv(RES / "sar_appendix.csv")

    # --------------------------------------------------------------------------- #
    # Table 3: safe automation by group, P1, TLP, alpha = 0.10
    # --------------------------------------------------------------------------- #
    t = sar[(sar.method == "tlp") & (sar.alpha == 0.10) & (sar.prompt == "P1")]
    L = [r"\begin{table*}[!htbp]", r"\centering", r"\small",
         r"\caption{Safe automation by demographic group under the bare prompt (P1), "
         r"with token-probability confidence and an error budget of $\alpha=0.10$. "
         r"Each cell pair gives the reference group (White or male) followed by the "
         r"demographic group (Black or female); each group comprises 1{,}000 cases. "
         r"$\Delta$ is the reference minus the demographic safe automation rate, with a "
         r"95\% paired bootstrap interval; values in bold have intervals that exclude "
         r"zero. Held-out error is the error rate among cases accepted by thresholds "
         r"fitted on the remaining pairs, and should not exceed 0.10.}",
         r"\label{tbl:sar}", r"\vspace{4pt}",
         r"\setlength{\tabcolsep}{5pt}",
         r"\begin{tabular}{lccccc}", r"\toprule",
         r"\textbf{Model} & \textbf{Accuracy} & \textbf{ECE} & \textbf{SAR} & "
         r"\textbf{$\Delta$ [95\% CI]} & \textbf{Held-out error} \\",
         r" & ref / dem & ref / dem & ref / dem & & ref / dem \\", r"\midrule"]
    for i, d in enumerate(DS):
        if i:
            L.append(r"\addlinespace")
        L.append(r"\multicolumn{6}{l}{\textit{%s}} \\" % DS[d])
        for m in present(t, MODELS):
            rows = t[(t.dataset == d) & (t.model == m)]
            if rows.empty:
                continue
            r = rows.iloc[0]
            gap = f"{signed(r.sar_gap)} {ci(r.sar_gap_lo, r.sar_gap_hi)}"
            if r.sar_gap_lo > 0 or r.sar_gap_hi < 0:
                gap = r"\textbf{" + gap + "}"
            L.append(f"\\quad {m} & {pair(r.acc_ref, r.acc_dem)} & {pair(r.ece_ref, r.ece_dem)} & "
                     f"{pair(r.sar_ref, r.sar_dem)} & {gap} & {pair(r.cf_err_ref, r.cf_err_dem)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write("tbl_sar.tex", L)

    # --------------------------------------------------------------------------- #
    # Table 4: prompt effects relative to P1, alpha = 0.10
    # --------------------------------------------------------------------------- #
    c = con[con.alpha == 0.10].copy()
    c["axis"] = np.where(c.dataset.str.contains("race"), "race", "sex")
    L = [r"\begin{table*}[!htbp]", r"\centering", r"\small",
         r"\caption{Effect of the bias-aware (P2) and equity-aware (P3) prompts relative to "
         r"the bare prompt (P1) at $\alpha=0.10$. Each cell summarises eight combinations "
         r"of model and dataset. \emph{Narrowed} counts contrasts in which the difference "
         r"between groups decreased with a 95\% interval excluding zero; \emph{Reduced} "
         r"counts contrasts in which average safe automation across both groups decreased "
         r"with an interval excluding zero. $\Delta$Gap, $\Delta$SAR and $\Delta$Accuracy "
         r"are medians of the change relative to P1. No contrast showed a significant "
         r"widening of the difference or an increase in automation. With elicited "
         r"confidence, safe automation was zero for most combinations under every "
         r"prompt, so these contrasts are uninformative rather than evidence of no effect.}",
         r"\label{tbl:prompt}", r"\vspace{4pt}",
         r"\setlength{\tabcolsep}{4.5pt}",
         r"\begin{tabular}{llcccccccc}", r"\toprule",
         r" & & \multicolumn{4}{c}{\textbf{Race}} & \multicolumn{4}{c}{\textbf{Sex}} \\",
         r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
         r"\textbf{Confidence} & \textbf{Prompt} & Narrowed & $\Delta$Gap & Reduced & $\Delta$SAR "
         r"& Narrowed & $\Delta$Gap & Reduced & $\Delta$SAR \\", r"\midrule"]
    for meth, label in (("tlp", "Token probability"), ("ce", "Elicited")):
        for k, cond in enumerate(("P2", "P3")):
            cells = []
            for ax in ("race", "sex"):
                x = c[(c.method == meth) & (c.condition == cond) & (c.axis == ax)]
                cells += [f"{int(x.improves.sum())}/{len(x)}", signed(x.d_gap.median()),
                          f"{int(x.costs_automation.sum())}/{len(x)}", signed(x.d_sar_mean.median())]
            L.append(f"{label if k == 0 else ''} & {cond} & " + " & ".join(cells) + r" \\")
        if meth == "tlp":
            L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write("tbl_prompt.tex", L)

    # --------------------------------------------------------------------------- #
    # Table 5: deployment checks (cross-fitting and group-blind threshold), TLP
    # --------------------------------------------------------------------------- #
    s10 = sar[(sar.method == "tlp") & (sar.alpha == 0.10)].groupby("model").mean(numeric_only=True)
    s05 = sar[(sar.method == "tlp") & (sar.alpha == 0.05)].groupby("model").mean(numeric_only=True)
    L = [r"\begin{table*}[!htbp]", r"\centering", r"\small",
         r"\caption{Deployment checks with token-probability confidence, averaged over the "
         r"four datasets and three prompts (12 combinations per model). Held-out values "
         r"use thresholds fitted on four fifths of the pairs and applied to the remaining "
         r"fifth; a held-out error close to the budget indicates that the threshold "
         r"transfers to new cases. Group-blind cost is the safe automation each group "
         r"forgoes when a single threshold, safe for both groups, replaces group-specific "
         r"thresholds.}",
         r"\label{tbl:audit}", r"\vspace{4pt}",
         r"\setlength{\tabcolsep}{5pt}",
         r"\begin{tabular}{lccccc}", r"\toprule",
         r" & \multicolumn{4}{c}{\textbf{Error budget $\alpha=0.10$}} & "
         r"\textbf{$\alpha=0.05$} \\", r"\cmidrule(lr){2-5}\cmidrule(lr){6-6}",
         r"\textbf{Model} & SAR & Held-out coverage & Held-out error & Group-blind cost & "
         r"Held-out error \\",
         r" & ref / dem & ref / dem & ref / dem & ref / dem & ref / dem \\", r"\midrule"]
    for m in [m for m in present(sar, MODELS) if m in s10.index and m in s05.index]:
        a, b = s10.loc[m], s05.loc[m]
        L.append(f"{m} & {pair(a.sar_ref, a.sar_dem)} & {pair(a.cf_cov_ref, a.cf_cov_dem)} & "
                 f"{pair(a.cf_err_ref, a.cf_err_dem)} & {pair(a.cons_cost_ref, a.cons_cost_dem)} & "
                 f"{pair(b.cf_err_ref, b.cf_err_dem)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write("tbl_audit.tex", L)

    # --------------------------------------------------------------------------- #
    # Table 6: commercial models, MedQA race
    # --------------------------------------------------------------------------- #
    have = set(zip(api.model, api.method))
    ORDER = [k for k in API_ORDER if k in have] + sorted(have - set(API_ORDER))
    L = [r"\begin{table*}[!htbp]", r"\centering", r"\small",
         r"\caption{Commercial models on the MedQA race set (1{,}000 pairs). Only "
         r"GPT-4.1-mini returns token probabilities (TLP); the other models were evaluated "
         r"with elicited confidence (CE). Cell pairs give the White group followed by the "
         r"Black group. $^{\dagger}$Both groups reach a safe automation rate of one because "
         r"the error rate is already within the budget; the difference is then zero by "
         r"construction and carries no information about disparity.}",
         r"\label{tbl:api}", r"\vspace{4pt}",
         r"\setlength{\tabcolsep}{4pt}",
         r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{lllccccc}", r"\toprule",
         r" & & & & & \multicolumn{2}{c}{\textbf{$\alpha=0.10$}} & \textbf{$\alpha=0.05$} \\",
         r"\cmidrule(lr){6-7}\cmidrule(lr){8-8}",
         r"\textbf{Model} & \textbf{Conf.} & \textbf{Prompt} & \textbf{Accuracy} & \textbf{ECE} & "
         r"SAR & $\Delta$ [95\% CI] & SAR \\",
         r" & & & ref / dem & ref / dem & ref / dem & & ref / dem \\", r"\midrule"]
    for j, (m, meth) in enumerate(ORDER):
        if j:
            L.append(r"\addlinespace")
        for k, cond in enumerate(("P1", "P2", "P3")):
            q10 = api[(api.model == m) & (api.method == meth) & (api.prompt == cond) & (api.alpha == 0.10)]
            q05 = api[(api.model == m) & (api.method == meth) & (api.prompt == cond) & (api.alpha == 0.05)]
            if q10.empty or q05.empty:
                continue
            r10, r05 = q10.iloc[0], q05.iloc[0]
            dag = r"$^{\dagger}$" if r10.ceiling else ""
            dag5 = r"$^{\dagger}$" if r05.ceiling else ""
            L.append(f"{m if k == 0 else ''} & {meth.upper() if k == 0 else ''} & {cond} & "
                     f"{pair(r10.acc_ref, r10.acc_dem)} & {pair(r10.ece_ref, r10.ece_dem)} & "
                     f"{pair(r10.sar_ref, r10.sar_dem)}{dag} & {signed(r10.sar_gap)} {ci(r10.sar_gap_lo, r10.sar_gap_hi)} & "
                     f"{pair(r05.sar_ref, r05.sar_dem)}{dag5} \\\\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    write("tbl_api.tex", L)
    print("tables written to", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
