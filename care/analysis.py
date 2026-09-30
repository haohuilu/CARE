"""Safe automation analysis.

A referral system accepts a model's answer when its confidence clears a
threshold and refers the case to a clinician otherwise. For an error budget
``alpha``, the safe automation rate (SAR) of a group is the largest share of its
cases that some threshold can accept while the error rate among accepted cases
stays at or below ``alpha``; it is zero when no threshold meets the budget.

For each dataset, model and confidence source, :func:`run` reports

* SAR per group, the difference between groups and a paired bootstrap interval,
  with accuracy and expected calibration error alongside;
* the effect of each prompt relative to the bare prompt on the difference and on
  average SAR, with the same paired bootstrap;
* cross-fitted (out-of-sample) coverage and error of the thresholds, and the
  automation each group gives up under a single group-blind threshold.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from . import config as cfg
from .data import load_records, restrict_to_groups, validate

# A group whose cases are all referred has an empty accepted set, so its error
# rate is legitimately undefined; those NaNs are meaningful, not faults.
warnings.filterwarnings("ignore", message="Mean of empty slice")
np.seterr(invalid="ignore", divide="ignore")

ALPHAS = (0.05, 0.10, 0.15)
ALPHA_GRID = np.round(np.arange(0.02, 0.301, 0.01), 3)
SESOI = 0.02          # smallest gap reduction worth calling an improvement


# --------------------------------------------------------------------------- #
# Loading and calibration
# --------------------------------------------------------------------------- #

def load_slice(dataset: str, model: str, conf_method: str = "tlp",
               strict: bool = False, quiet: bool = True):
    """Load and validate one dataset x model x confidence-method slice.

    Parameters
    ----------
    dataset, model, conf_method : str
        Slice identifiers.
    strict : bool
        Raise on validation errors rather than returning them.
    quiet : bool
        Suppress the printed validation summary.

    Returns
    -------
    tuple of (pandas.DataFrame or None, SanityReport)
        Group-restricted records and the validation report. The frame is None
        when validation found a hard error.
    """
    df = load_records(dataset, model, conf_method)
    rep = validate(df, strict=strict)
    if not quiet:
        for e in rep.errors:
            print(f"    ERROR   {e}")
        for w in rep.warnings:
            print(f"    warning {w}")
    if not rep.ok:
        return None, rep
    df = restrict_to_groups(df)
    if cfg.DATASETS[dataset]["paired"] and len(df):
        # keep only complete pairs, so the design stays matched
        spec = cfg.DATASETS[dataset]
        comp = df.groupby(["prompt", "pair_id"])["group"].transform(
            lambda s: tuple(sorted(s)) == tuple(sorted(spec["groups"])))
        df = df[comp.astype(bool)].reset_index(drop=True)
    return df, rep


def save(df: pd.DataFrame, name: str) -> str:
    """Write a results table to ``RESULTS_DIR`` and return its path."""
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    path = os.path.join(cfg.RESULTS_DIR, name)
    df.to_csv(path, index=False)
    return path


def _n_bins_default(n: int) -> int:
    """Return the default bin count for a sample of size ``n``.

    Parameters
    ----------
    n : int
        Number of records.

    Returns
    -------
    int
        ``ceil(n ** (1/3))``, floored at 1.
    """
    return max(1, int(np.ceil(n ** (1.0 / 3.0))))


def ece(conf, correct, scheme: str = "equal_mass", n_bins: int | None = None) -> float:
    """Expected calibration error.

    Parameters
    ----------
    conf : array_like
        Predicted confidence in [0, 1].
    correct : array_like
        Binary correctness indicators.
    scheme : {"equal_mass", "equal_width"}
        Binning scheme. ``equal_mass`` places roughly equal counts per bin and
        is the default; ``equal_width`` reproduces the conventional fixed-grid
        estimator for appendix comparison.
    n_bins : int, optional
        Number of bins; defaults to ``ceil(n ** (1/3))`` for ``equal_mass`` and
        10 for ``equal_width``.

    Returns
    -------
    float
        Sample-weighted mean absolute gap between confidence and accuracy
        across bins; ``numpy.nan`` for an empty input.
    """
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    m = np.isfinite(conf) & np.isfinite(correct)
    conf, correct = conf[m], correct[m]
    n = conf.size
    if n == 0:
        return np.nan

    if scheme == "equal_width":
        nb = n_bins or 10
        edges = np.linspace(0.0, 1.0, nb + 1)
        idx = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, nb - 1)
    elif scheme == "equal_mass":
        nb = n_bins or _n_bins_default(n)
        qs = np.quantile(conf, np.linspace(0, 1, nb + 1)[1:-1]) if nb > 1 else np.array([])
        idx = np.clip(np.digitize(conf, qs, right=False), 0, nb - 1)
    else:
        raise ValueError(f"unknown scheme {scheme!r}")

    total = 0.0
    for b in range(nb):
        s = idx == b
        cnt = int(s.sum())
        if cnt:
            total += cnt * abs(conf[s].mean() - correct[s].mean())
    return float(total / n)


# --------------------------------------------------------------------------- #
# Safe automation rate
# --------------------------------------------------------------------------- #

def safe_coverage(conf, correct, alpha: float, min_accepted: int = cfg.MIN_ACCEPTED) -> float:
    """Largest coverage a threshold can reach with error among accepted <= alpha.

    A threshold can only cut between distinct confidence values, so tied cases
    are accepted or referred together. This matters for coarse confidence
    sources, where ignoring ties would overstate what is achievable.

    Values are compared exactly, never rounded: token probabilities of strong
    models often lie within 1e-7 of 1, and rounding would merge most cases into
    one artificial tie block (a knife-edge where SAR jumps between 0 and 0.6 on
    a single case). Elicited confidences are exact multiples of 0.01, so their
    genuine ties are preserved.

    Parameters
    ----------
    conf, correct : array_like
        Confidence and correctness for one group.
    alpha : float
        Error budget.
    min_accepted : int
        Smallest accepted set counted, so a budget is not "met" on a handful of
        cases.

    Returns
    -------
    float
        Safe automation rate in [0, 1] over all cases, counting cases without a
        confidence as referred; 0 when no threshold meets the budget.
    """
    return _safe_cut(conf, correct, alpha, min_accepted)[1]


def _safe_cut(conf, correct, alpha: float, min_accepted: int = cfg.MIN_ACCEPTED):
    """Threshold and coverage behind :func:`safe_coverage`.

    Returns
    -------
    tuple of (float, float)
        ``(tau, coverage)``: cases with confidence >= tau are accepted.
        ``tau`` is ``inf`` (accept nothing) when no threshold meets the budget.
    """
    c = np.asarray(conf, dtype=float)
    y = np.asarray(correct, dtype=float)
    n = c.size                       # every case, including ones with no confidence
    if n == 0:
        return np.inf, np.nan
    # A case without a usable confidence (an unparsable elicited reply) cannot
    # be auto-accepted: it is referred, so it stays in the denominator.
    m = np.isfinite(c)
    c, y = c[m], y[m]
    if c.size == 0:
        return np.inf, 0.0
    order = np.argsort(-c, kind="stable")
    c, y = c[order], y[order]
    ends = np.flatnonzero(np.r_[c[1:] < c[:-1], True]) + 1      # tie-block ends
    err = np.cumsum(1.0 - y)[ends - 1] / ends
    ok = (err <= alpha) & (ends >= min_accepted)
    if not ok.any():
        return np.inf, 0.0
    k = ends[ok].max()
    return float(c[k - 1]), float(k / n)


def crossfit(conf, correct, folds, alpha: float) -> dict:
    """Out-of-sample check of a group's safe threshold.

    SAR picks the threshold on the same cases it is scored on, which flatters
    it. Here the threshold is chosen on all folds but one and applied to the
    held-out fold, as a hospital would set it on audit data and then deploy
    it. The realised error among accepted cases shows whether the budget still
    holds on new cases.

    Parameters
    ----------
    conf, correct : array_like
        One group's confidence and correctness.
    folds : array_like of int
        Fold label per case (shared across groups and conditions, so pairs stay
        together).
    alpha : float
        Error budget.

    Returns
    -------
    dict
        ``cov`` (share of all cases accepted out of sample) and ``err`` (error
        rate among them; NaN if none accepted).
    """
    c = np.asarray(conf, dtype=float)
    y = np.asarray(correct, dtype=float)
    folds = np.asarray(folds)
    acc_n = err_n = 0
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        tau, _ = _safe_cut(c[tr], y[tr], alpha)
        take = te & np.isfinite(c) & (c >= tau)
        acc_n += int(take.sum())
        err_n += int((1.0 - y[take]).sum())
    return {"cov": acc_n / c.size if c.size else np.nan,
            "err": err_n / acc_n if acc_n else np.nan}


def conservative_global(groups_data: dict, alpha: float) -> dict:
    """Coverage per group under one shared threshold that is safe for every group.

    This is the fallback when group labels cannot be used at inference: a single
    threshold, set so that in each group either nothing is accepted or at least
    ``MIN_ACCEPTED`` cases are accepted with an error rate at most ``alpha``.
    The most permissive such threshold is chosen. Comparing each group's
    coverage here with its own SAR gives the automation given up by refusing to
    use group-specific thresholds.

    Parameters
    ----------
    groups_data : dict
        Group label to ``(conf, correct)`` arrays.
    alpha : float
        Error budget.

    Returns
    -------
    dict
        Group label to coverage under the conservative threshold, plus ``tau``
        (``inf`` if only referring everyone satisfies the budget).
    """
    pooled = np.concatenate([np.asarray(c, float) for c, _ in groups_data.values()])
    pooled = pooled[np.isfinite(pooled)]
    best_tau, best_cov = np.inf, -1.0
    for t in np.unique(pooled)[::-1]:
        ok, n_acc = True, 0
        for c, y in groups_data.values():
            acc = np.asarray(c, float) >= t
            k = int(acc.sum())
            n_acc += k
            # A group either has nothing accepted, or enough accepted cases for
            # the budget to be shown, as in safe_coverage; a few lucky correct
            # answers do not demonstrate safety.
            if k and (k < cfg.MIN_ACCEPTED or (1.0 - np.asarray(y, float)[acc].mean()) > alpha):
                ok = False
                break
        if ok and n_acc >= cfg.MIN_ACCEPTED and n_acc > best_cov:
            best_tau, best_cov = float(t), n_acc
    out = {"tau": best_tau}
    for g, (c, _) in groups_data.items():
        c = np.asarray(c, float)
        out[g] = float((c >= best_tau).mean()) if np.isfinite(best_tau) else 0.0
    return out


def _aligned(df: pd.DataFrame, groups, conds):
    """Per condition and group, arrays aligned on units common to all conditions.

    Parameters
    ----------
    df : pandas.DataFrame
        Records for one slice, all conditions.
    groups : tuple of str
        Reference and demographic group.
    conds : list of str
        Conditions to align.

    Returns
    -------
    tuple of (dict, dict)
        ``units[g]`` -- the common unit ids for group g -- and
        ``arr[cond][g]`` -- ``(conf, correct)`` aligned to ``units[g]``.
    """
    units, arr = {}, {c: {} for c in conds}
    for g in groups:
        sets = [set(df[(df["prompt"] == c) & (df["group"] == g)]["question_id"]) for c in conds]
        units[g] = np.array(sorted(set.intersection(*sets))) if sets else np.array([])
        for c in conds:
            s = (df[(df["prompt"] == c) & (df["group"] == g)]
                 .drop_duplicates("question_id").set_index("question_id").loc[units[g]])
            arr[c][g] = (s["conf_raw"].to_numpy(float), s["correct"].to_numpy(float))
    return units, arr


def _ci(draws, level=0.95):
    """Percentile interval of finite bootstrap draws.

    Parameters
    ----------
    draws : array_like
        Bootstrap replicates.
    level : float
        Interval coverage.

    Returns
    -------
    tuple of (float, float)
        Lower and upper bound, NaN if too few finite draws.
    """
    d = np.asarray(draws, dtype=float)
    d = d[np.isfinite(d)]
    if d.size < 50:
        return np.nan, np.nan
    q = (1 - level) / 2
    return float(np.quantile(d, q)), float(np.quantile(d, 1 - q))


def analyse_slice(df: pd.DataFrame, dataset: str, alphas=ALPHAS, n_boot: int = cfg.N_BOOT,
                  seed: int = cfg.SEED, baseline: str | None = None):
    """SAR, gap, and paired mitigation effects for one dataset x model x method.

    Parameters
    ----------
    df : pandas.DataFrame
        Group-restricted records, all conditions.
    dataset : str
        Dataset key, which fixes the groups and whether the design is paired.
    alphas : sequence of float
        Error budgets.
    n_boot : int
        Bootstrap replicates.
    seed : int
        Random seed.
    baseline : str, optional
        Reference condition; ``P1`` if present, else the first condition.

    Returns
    -------
    tuple of (list of dict, list of dict)
        Per-condition rows and per-contrast (condition vs baseline) rows.
    """
    spec = cfg.DATASETS[dataset]
    ref, dem = spec["groups"]
    conds = sorted(df["prompt"].unique())
    baseline = baseline or ("P1" if "P1" in conds else conds[0])
    if spec["paired"]:
        both = set(df[df["group"] == ref]["question_id"]) & set(df[df["group"] == dem]["question_id"])
        df = df[df["question_id"].isin(both)]
    units, arr = _aligned(df, (ref, dem), conds)
    n_ref, n_dem = len(units[ref]), len(units[dem])
    if min(n_ref, n_dem) < 2 * cfg.MIN_ACCEPTED:
        return [], []

    # A paired design resamples one set of pairs for both groups; otherwise
    # each group is resampled on its own. Either way the same draw is applied
    # to every condition, which is what makes the mitigation contrast paired.
    paired = spec["paired"] and np.array_equal(units[ref], units[dem])
    rng = np.random.default_rng(seed)
    if paired:
        idx_r = idx_d = rng.integers(0, n_ref, size=(n_boot, n_ref))
    else:
        idx_r = rng.integers(0, n_ref, size=(n_boot, n_ref))
        idx_d = rng.integers(0, n_dem, size=(n_boot, n_dem))

    # Cross-fitting folds over units; in a paired design both groups share them.
    n_folds = 5
    fold_r = np.random.default_rng(seed + 1).permutation(np.arange(n_ref) % n_folds)
    fold_d = fold_r if paired else np.random.default_rng(seed + 2).permutation(np.arange(n_dem) % n_folds)

    cond_rows, gap_draws, mean_draws = [], {}, {}
    for c in conds:
        (cr, yr), (cd, yd) = arr[c][ref], arr[c][dem]
        base = {"dataset": dataset, "prompt": c, "paired": paired,
                "n_ref": n_ref, "n_dem": n_dem,
                "acc_ref": float(yr.mean()), "acc_dem": float(yd.mean()),
                # share of cases with a usable confidence (below 1 only for CE)
                "parse_rate_ref": float(np.isfinite(cr).mean()),
                "parse_rate_dem": float(np.isfinite(cd).mean()),
                "acc_gap": float(yr.mean() - yd.mean()),
                "ece_ref": ece(cr, yr), "ece_dem": ece(cd, yd),
                "mean_conf_ref": float(np.nanmean(cr)), "mean_conf_dem": float(np.nanmean(cd))}
        for a in alphas:
            s_r, s_d = safe_coverage(cr, yr, a), safe_coverage(cd, yd, a)
            dr = np.array([safe_coverage(cr[i], yr[i], a) for i in idx_r])
            dd = np.array([safe_coverage(cd[j], yd[j], a) for j in idx_d])
            gap_draws[(c, a)] = dr - dd
            mean_draws[(c, a)] = (dr + dd) / 2        # automation for the average case
            lo, hi = _ci(dr - dd)
            cons = conservative_global({ref: (cr, yr), dem: (cd, yd)}, a)
            cf_r, cf_d = crossfit(cr, yr, fold_r, a), crossfit(cd, yd, fold_d, a)
            cond_rows.append({**base, "alpha": a, "sar_ref": s_r, "sar_dem": s_d,
                              "sar_gap": s_r - s_d, "sar_gap_lo": lo, "sar_gap_hi": hi,
                              "dem_infeasible": s_d == 0.0, "ref_infeasible": s_r == 0.0,
                              # SAR = 1: every case can be accepted within the budget
                              # (error rate <= alpha), so no threshold is needed and
                              # a zero gap is a ceiling, not evidence of equal SAR.
                              "ceiling_ref": s_r >= 1.0, "ceiling_dem": s_d >= 1.0,
                              "ceiling": bool(s_r >= 1.0 and s_d >= 1.0),
                              "sar_mean": (s_r + s_d) / 2,
                              # one threshold safe for both groups, and what it costs
                              "cons_tau": cons["tau"], "cons_cov_ref": cons[ref],
                              "cons_cov_dem": cons[dem],
                              "cons_cost_ref": s_r - cons[ref], "cons_cost_dem": s_d - cons[dem],
                              # thresholds chosen on 4/5 of pairs, applied to the rest
                              "cf_cov_ref": cf_r["cov"], "cf_cov_dem": cf_d["cov"],
                              "cf_err_ref": cf_r["err"], "cf_err_dem": cf_d["err"]})

    contrast_rows = []
    for c in conds:
        if c == baseline:
            continue
        for a in alphas:
            pt = next(r["sar_gap"] for r in cond_rows if r["prompt"] == c and r["alpha"] == a) - \
                next(r["sar_gap"] for r in cond_rows if r["prompt"] == baseline and r["alpha"] == a)
            d = gap_draws[(c, a)] - gap_draws[(baseline, a)]
            lo, hi = _ci(d)
            row_c = next(r for r in cond_rows if r["prompt"] == c and r["alpha"] == a)
            row_b = next(r for r in cond_rows if r["prompt"] == baseline and r["alpha"] == a)
            # both conditions at SAR = 1 for both groups: the gap cannot move, so
            # the contrast says nothing about the prompt
            ceil = row_c["ceiling"] and row_b["ceiling"]
            dm = mean_draws[(c, a)] - mean_draws[(baseline, a)]
            mlo, mhi = _ci(dm)
            contrast_rows.append({
                "dataset": dataset, "condition": c, "baseline": baseline, "alpha": a,
                "d_gap": pt, "d_gap_lo": lo, "d_gap_hi": hi,
                # negative d_gap = the condition narrowed the gap
                "improves": bool(np.isfinite(hi) and hi < 0),
                "worsens": bool(np.isfinite(lo) and lo > 0),
                "rules_out_meaningful_improvement": bool(np.isfinite(lo) and lo > -SESOI and not ceil),
                "ceiling": bool(ceil),
                # what the prompt does to automation overall (average of the groups);
                # negative = fewer cases can be auto-accepted safely
                "d_sar_mean": row_c["sar_mean"] - row_b["sar_mean"],
                "d_sar_mean_lo": mlo, "d_sar_mean_hi": mhi,
                "d_acc_mean": (row_c["acc_ref"] + row_c["acc_dem"] - row_b["acc_ref"] - row_b["acc_dem"]) / 2,
                "costs_automation": bool(np.isfinite(mhi) and mhi < 0),
                "adds_automation": bool(np.isfinite(mlo) and mlo > 0),
            })
    return cond_rows, contrast_rows


def sar_curves(df: pd.DataFrame, dataset: str, grid=ALPHA_GRID) -> pd.DataFrame:
    """SAR per group over a grid of error budgets, for plotting.

    Parameters
    ----------
    df : pandas.DataFrame
        Group-restricted records for one slice.
    dataset : str
        Dataset key.
    grid : array_like
        Error budgets.

    Returns
    -------
    pandas.DataFrame
        One row per condition, group, and budget.
    """
    ref, dem = cfg.DATASETS[dataset]["groups"]
    rows = []
    for c, sub in df.groupby("prompt"):
        for g in (ref, dem):
            s = sub[sub["group"] == g]
            for a in grid:
                rows.append({"dataset": dataset, "prompt": c, "group": g, "alpha": float(a),
                             "sar": safe_coverage(s["conf_raw"], s["correct"], a)})
    return pd.DataFrame(rows)


def ece_by_subject(df: pd.DataFrame, min_n: int = 30) -> pd.DataFrame:
    """ECE per specialty, overall and per group, where a specialty label exists.

    Parameters
    ----------
    df : pandas.DataFrame
        Group-restricted records carrying ``subject``.
    min_n : int
        Smallest subject-by-group cell reported.

    Returns
    -------
    pandas.DataFrame
        One row per condition and subject.
    """
    if "subject" not in df.columns or df["subject"].isna().all():
        return pd.DataFrame()
    rows = []
    for (c, subj), s in df.dropna(subset=["subject"]).groupby(["prompt", "subject"]):
        if len(s) < min_n:
            continue
        rec = {"prompt": c, "subject": subj, "n": len(s),
               "accuracy": float(s["correct"].mean()), "ece": ece(s["conf_raw"], s["correct"])}
        for g, sg in s.groupby("group"):
            if len(sg) >= min_n:
                rec[f"ece_{g}"] = ece(sg["conf_raw"], sg["correct"])
        rows.append(rec)
    return pd.DataFrame(rows)


def run(datasets=None, models=None, methods=("tlp",), alphas=ALPHAS,
        n_boot: int = cfg.N_BOOT, verbose: bool = True, tag: str = ""):
    """Run the safe-automation analysis over every available slice.

    Parameters
    ----------
    datasets, models : sequence of str, optional
        Slices to analyse; default all registered datasets and non-excluded models.
    methods : sequence of str
        Confidence sources.
    alphas : sequence of float
        Error budgets.
    n_boot : int
        Bootstrap replicates.
    verbose : bool
        Print progress.
    tag : str
        Suffix for the output files, so that separate runs (main text,
        appendix, all-models comparison) do not overwrite one another.

    Returns
    -------
    dict of pandas.DataFrame
        ``sar``, ``contrast``, ``curves``, ``letter_mass``, ``ece_subject``.
    """
    datasets = list(datasets or cfg.DATASETS)
    models = list(models or cfg.MODELS)
    sar, con, curves, lmass, subj = [], [], [], [], []

    for ds in datasets:
        for mdl in models:
            for meth in methods:
                try:
                    df, rep = load_slice(ds, mdl, meth)
                except (FileNotFoundError, KeyError):
                    continue
                if df is None or len(df) == 0 or df["conf_raw"].isna().all():
                    why = "; ".join(rep.errors) if df is None else "no usable confidence values"
                    print(f"  skipped {ds}/{cfg.short(mdl)}/{meth}: {why}")
                    continue
                ident = {"model": cfg.short(mdl), "model_key": mdl, "method": meth,
                         "near_chance": bool(rep.degenerate)}
                cr, ct = analyse_slice(df, ds, alphas, n_boot)
                sar += [{**r, **ident} for r in cr]
                con += [{**r, **ident} for r in ct]
                cv = sar_curves(df, ds)
                if len(cv):
                    curves.append(cv.assign(**ident))
                if "letter_mass" in df.columns and df["letter_mass"].notna().any():
                    lm = df["letter_mass"].dropna()
                    lmass.append({"dataset": ds, **ident, "n": int(lm.size),
                                  "median": float(lm.median()),
                                  "p10": float(lm.quantile(.10)),
                                  "share_below_0.5": float((lm < 0.5).mean())})
                es = ece_by_subject(df)
                if len(es):
                    subj.append(es.assign(dataset=ds, **ident))
                if verbose:
                    print(f"  analysed {ds}/{cfg.short(mdl)}/{meth}")

    out = {"sar": pd.DataFrame(sar), "contrast": pd.DataFrame(con),
           "curves": pd.concat(curves, ignore_index=True) if curves else pd.DataFrame(),
           "letter_mass": pd.DataFrame(lmass),
           "ece_subject": pd.concat(subj, ignore_index=True) if subj else pd.DataFrame()}
    for k, v in out.items():
        if len(v):
            save(v, f"{k}{'_' + tag if tag else ''}.csv")
    return out


def main() -> int:
    """Reproduce the paper's analysis from the cached responses in ``DATA_ROOT``."""
    run(datasets=list(cfg.DATASETS), models=cfg.LOCAL_MODELS, methods=("tlp", "ce"), tag="main")
    run(datasets=["pairs_medqa_race"], models=cfg.API_MODELS, methods=("tlp", "ce"),
        verbose=False, tag="appendix")
    print("CSV files written to", cfg.RESULTS_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
