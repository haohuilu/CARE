"""Analyses added in response to peer review (revision R1).

Everything here is computed from cached model responses; no model is called.
``python -m care.revision`` writes one CSV per analysis to ``RESULTS_DIR`` with
the prefix ``r1_`` and a short ``r1_summary.md``.

Analyses that need only the responses already cached for the submitted paper:

=================  ==========================================================
``pair_level``     within-pair difference in confidence and in the probability
                   of the keyed answer; how often the selected answer changes;
                   discordant correctness
``ranking``        AUROC of confidence for correctness and area under the
                   risk-coverage curve, per group, with paired differences
``common``         one cross-fitted threshold applied to both groups: held-out
                   coverage and accepted-case error per group and their
                   difference; also group-specific and group-safe thresholds,
                   and thresholds with a safety margin
``directional``    counts of comparisons favouring each variant at each
                   tolerance, with non-estimable comparisons set aside
``permutation``    within-pair label-swap test of the directional pattern
``pooled``         cluster bootstrap of the pooled mean difference
``mixed``          mixed model of the pair-level confidence difference
``bca``            percentile and BCa intervals for the SAR difference
``by_split``       results by source split (benchmark-exposure check)
``sensitivity``    results after removing pairs caught by the screens in
                   ``pair_flags.csv``
``tlp_vs_ce``      answer agreement between the two confidence procedures and
                   a comparison of confidence sources on a fixed set of answers
``power``          sample size needed for a stated difference in coverage
=================  ==========================================================

Analyses that need the new inference runs (control pairs, control prompts):

=================  ==========================================================
``controls``       placebo, both-edited and reverse-direction sets beside the
                   demographic sets they control for
``edit_effect``    verbatim benchmark text against the same text with one
                   neutral word inserted
``prompt_controls``contrasts that separate persona, length and fairness wording
=================  ==========================================================
"""

from __future__ import annotations

import functools
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

from . import config as cfg
from .analysis import _ci, _safe_cut, analyse_slice, load_slice, safe_coverage
from .prompts import CONDITIONS, CONTROL_CONDITIONS, CONTROL_CONTRASTS

ALPHAS = (0.05, 0.10, 0.15)
PRIMARY_ALPHA = 0.10
SESOI = 0.02                   # smallest difference of interest (two points)
NON_ESTIMABLE = 0.01           # SAR at or below this means "no usable threshold"
N_FOLDS = 5


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

@functools.lru_cache(maxsize=None)
def _slice(dataset: str, model: str, method: str, conditions: tuple):
    try:
        df, _ = load_slice(dataset, model, method, conditions=list(conditions))
    except (FileNotFoundError, KeyError):
        return None
    if df is None or not len(df) or df["conf_raw"].isna().all():
        return None
    return df


@functools.lru_cache(maxsize=None)
def _raw(dataset: str, model: str, cond: str) -> pd.DataFrame:
    """Raw cache lines for one condition (needed for per-option probabilities)."""
    path = os.path.join(cfg.DATA_ROOT, dataset, model, "calibration_cache", f"{cond}.jsonl")
    rows = []
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)


def arrays(dataset: str, model: str, method: str = "tlp", conditions=CONDITIONS):
    """Pair-aligned arrays for one dataset and model.

    Parameters
    ----------
    dataset, model, method : str
        Slice identifiers.
    conditions : sequence of str
        Prompting conditions; pairs are kept only if complete under all.

    Returns
    -------
    dict or None
        ``ids`` (pair ids, sorted) and, per condition, ``c``, ``y`` and
        ``pred``: arrays of shape (2, n) with the reference variant in row 0
        and the demographic variant in row 1. None if the slice is not on disk.
    """
    df = _slice(dataset, model, method, tuple(conditions))
    if df is None:
        return None
    ref, dem = cfg.DATASETS[dataset]["groups"]
    ids = None
    for c in conditions:
        for g in (ref, dem):
            s = set(df[(df["prompt"] == c) & (df["group"] == g)]["question_id"])
            ids = s if ids is None else ids & s
    ids = np.array(sorted(ids or []))
    if ids.size < 2 * cfg.MIN_ACCEPTED:
        return None
    out = {"ids": ids, "dataset": dataset, "model": model, "method": method}
    for c in conditions:
        blk = {"c": [], "y": [], "pred": []}
        for g in (ref, dem):
            s = (df[(df["prompt"] == c) & (df["group"] == g)]
                 .drop_duplicates("question_id").set_index("question_id").loc[ids])
            blk["c"].append(s["conf_raw"].to_numpy(float))
            blk["y"].append(s["correct"].to_numpy(float))
            blk["pred"].append(s["pred_idx"].to_numpy(float))
        out[c] = {k: np.vstack(v) for k, v in blk.items()}
    return out


def gold_prob(dataset: str, model: str, cond: str, ids) -> np.ndarray:
    """Token probability of the keyed answer, shape (2, n); NaN where absent."""
    raw = _raw(dataset, model, cond)
    out = np.full((2, len(ids)), np.nan)
    if raw.empty or "tlp_option_probs" not in raw.columns:
        return out
    raw = raw[raw["tlp_option_probs"].notna()].drop_duplicates(
        ["question_id", "variant"], keep="last")
    key = {(str(q), str(v).lower()): json.loads(p).get(a)
           for q, v, p, a in zip(raw["question_id"], raw["demographic"],
                                 raw["tlp_option_probs"], raw["answer_idx"])}
    for g, name in enumerate(cfg.DATASETS[dataset]["groups"]):
        out[g] = [key.get((str(i), name), np.nan) for i in ids]
    return out


def _slices(datasets, models, method="tlp", conditions=CONDITIONS):
    for ds in datasets:
        for mdl in models:
            a = arrays(ds, mdl, method, conditions)
            if a is not None:
                yield ds, mdl, a


def _ident(ds, mdl, prompt=None, **kw):
    d = {"dataset": ds, "source": cfg.source_of(ds), "axis": cfg.axis_of(ds),
         "kind": cfg.DATASETS[ds].get("kind", "main"), "model": cfg.short(mdl)}
    if prompt is not None:
        d["prompt"] = prompt
    d.update(kw)
    return d


def _boot_idx(n: int, n_boot: int, seed: int = cfg.SEED):
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def save(df: pd.DataFrame, name: str) -> str:
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    path = os.path.join(cfg.RESULTS_DIR, name)
    df.to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------- #
# Pair-level analyses
# --------------------------------------------------------------------------- #

def _mean_ci(d, idx):
    """Mean of paired differences with a percentile bootstrap interval."""
    d = np.asarray(d, float)
    ok = np.isfinite(d)
    if ok.sum() < 20:
        return np.nan, np.nan, np.nan
    draws = np.nanmean(np.where(ok, d, np.nan)[idx], axis=1)
    lo, hi = _ci(draws)
    return float(d[ok].mean()), lo, hi


def pair_level(datasets, models, method="tlp", conditions=CONDITIONS,
               n_boot: int = cfg.N_BOOT) -> pd.DataFrame:
    """Within-pair differences that do not depend on any threshold.

    Returns
    -------
    pandas.DataFrame
        Per dataset, model and prompt: mean reference-minus-demographic
        difference in confidence (``d_conf``) and in the probability of the
        keyed answer (``d_gold``), each with a paired bootstrap interval and a
        Wilcoxon signed-rank p-value; the share of pairs whose selected answer
        differs between versions; and the discordant correctness counts with an
        exact McNemar p-value.
    """
    rows = []
    for ds, mdl, a in _slices(datasets, models, method, conditions):
        n = a["ids"].size
        idx = _boot_idx(n, n_boot)
        for c in conditions:
            cc, y, pred = a[c]["c"], a[c]["y"], a[c]["pred"]
            d = cc[0] - cc[1]
            m, lo, hi = _mean_ci(d, idx)
            row = _ident(ds, mdl, c, method=method, n_pairs=n, d_conf=m, d_conf_lo=lo,
                         d_conf_hi=hi, d_conf_median=float(np.nanmedian(d)),
                         share_conf_ref_higher=float(np.nanmean(d > 0)),
                         share_conf_dem_higher=float(np.nanmean(d < 0)))
            nz = d[np.isfinite(d) & (d != 0)]
            row["d_conf_wilcoxon_p"] = float(stats.wilcoxon(nz).pvalue) if nz.size >= 20 else np.nan
            if method == "tlp":
                g = gold_prob(ds, mdl, c, a["ids"])
                gm, glo, ghi = _mean_ci(g[0] - g[1], idx)
                row.update(d_gold=gm, d_gold_lo=glo, d_gold_hi=ghi)
            both = np.isfinite(pred[0]) & np.isfinite(pred[1])
            changed = (pred[0] != pred[1]) & both
            cm, clo, chi = _mean_ci(np.where(both, changed.astype(float), np.nan), idx)
            b = int(((y[0] == 1) & (y[1] == 0)).sum())      # only the reference version correct
            k = int(((y[0] == 0) & (y[1] == 1)).sum())      # only the demographic version correct
            row.update(answer_changed=cm, answer_changed_lo=clo, answer_changed_hi=chi,
                       acc_ref=float(y[0].mean()), acc_dem=float(y[1].mean()),
                       only_ref_correct=b, only_dem_correct=k,
                       mcnemar_p=float(stats.binomtest(b, b + k, 0.5).pvalue) if b + k else np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Threshold-free ranking quality
# --------------------------------------------------------------------------- #

def auroc(conf, correct) -> float:
    """Probability that a correct answer has higher confidence than an
    incorrect one (ties count one half). NaN if either class is empty."""
    c, y = np.asarray(conf, float), np.asarray(correct, float)
    ok = np.isfinite(c)
    c, y = c[ok], y[ok]
    n1 = int(y.sum())
    n0 = y.size - n1
    if not n1 or not n0:
        return np.nan
    r = stats.rankdata(c)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def aurc(conf, correct) -> float:
    """Area under the risk-coverage curve (lower is better).

    The mean, over coverage levels, of the error rate among the cases accepted
    at that level. Tied cases are accepted together; cases without a confidence
    are accepted last.
    """
    c, y = np.asarray(conf, float), np.asarray(correct, float)
    n = c.size
    if not n:
        return np.nan
    c = np.where(np.isfinite(c), c, -np.inf)
    order = np.argsort(-c, kind="stable")
    c, y = c[order], y[order]
    ends = np.flatnonzero(np.r_[c[1:] < c[:-1], True]) + 1
    risk = np.cumsum(1.0 - y)[ends - 1] / ends
    size = np.diff(np.r_[0, ends])
    return float((risk * size).sum() / n)


def ranking(datasets, models, method="tlp", conditions=CONDITIONS,
            n_boot: int = cfg.N_BOOT) -> pd.DataFrame:
    """AUROC and AURC per group with paired bootstrap differences.

    Separates how well confidence ranks answers from how many answers are
    correct: a group can have lower SAR because it is less accurate, because
    its confidence ranks its answers less well, or both.
    """
    rows = []
    for ds, mdl, a in _slices(datasets, models, method, conditions):
        n = a["ids"].size
        idx = _boot_idx(n, n_boot)
        for c in conditions:
            cc, y = a[c]["c"], a[c]["y"]
            row = _ident(ds, mdl, c, method=method, n_pairs=n,
                         acc_ref=float(y[0].mean()), acc_dem=float(y[1].mean()))
            for name, fn in (("auroc", auroc), ("aurc", aurc)):
                r, d = fn(cc[0], y[0]), fn(cc[1], y[1])
                draws = np.array([fn(cc[0][i], y[0][i]) - fn(cc[1][i], y[1][i]) for i in idx])
                lo, hi = _ci(draws)
                row.update({f"{name}_ref": r, f"{name}_dem": d, f"{name}_diff": r - d,
                            f"{name}_diff_lo": lo, f"{name}_diff_hi": hi})
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Thresholds: fitting rules and cross-fitted evaluation
# --------------------------------------------------------------------------- #

def cp_upper(errors, n, level: float):
    """One-sided Clopper-Pearson upper bound on a binomial proportion."""
    errors, n = np.asarray(errors, float), np.asarray(n, float)
    ub = stats.beta.ppf(level, errors + 1, np.maximum(n - errors, 1e-12))
    return np.where(errors >= n, 1.0, ub)


def fit_tau(conf, correct, alpha: float, rule: str = "point", delta: float = 0.05,
            min_accepted: int = cfg.MIN_ACCEPTED) -> float:
    """Most permissive threshold a fitting rule allows.

    Parameters
    ----------
    conf, correct : array_like
        Fitting sample.
    alpha : float
        Error tolerance among accepted cases.
    rule : {"point", "ucb", "ucb_bonf"}
        ``point``: the accepted-case error in the fitting sample is at most
        alpha (the rule behind SAR). ``ucb``: the one-sided ``1 - delta``
        Clopper-Pearson upper bound on that error is at most alpha, a safety
        margin that shrinks as the accepted set grows. ``ucb_bonf``: the same
        bound made simultaneous over 20 candidate thresholds at evenly spaced
        coverage levels by a Bonferroni correction, which gives a guarantee of
        the kind in Geifman and El-Yaniv (2017): with probability at least
        ``1 - delta`` the selected threshold has true accepted-case error at
        most alpha.
    delta : float
        Allowed failure probability for the two ``ucb`` rules.

    Returns
    -------
    float
        Threshold; cases with confidence at or above it are accepted. ``inf``
        (accept nothing) when no threshold qualifies.
    """
    if rule == "point":
        return _safe_cut(conf, correct, alpha, min_accepted)[0]
    c, y = np.asarray(conf, float), np.asarray(correct, float)
    ok = np.isfinite(c)
    c, y = c[ok], y[ok]
    if not c.size:
        return np.inf
    order = np.argsort(-c, kind="stable")
    c, y = c[order], y[order]
    ends = np.flatnonzero(np.r_[c[1:] < c[:-1], True]) + 1
    errs = np.cumsum(1.0 - y)[ends - 1]
    if rule == "ucb":
        level = 1.0 - delta
    elif rule == "ucb_bonf":
        grid = np.unique(ends[np.searchsorted(ends, np.linspace(0.05, 1.0, 20) * c.size)
                              .clip(max=ends.size - 1)])
        keep = np.isin(ends, grid)
        ends, errs = ends[keep], errs[keep]
        level = 1.0 - delta / max(1, ends.size)
    else:
        raise ValueError(f"unknown rule {rule!r}")
    good = (cp_upper(errs, ends, level) <= alpha) & (ends >= min_accepted)
    if not good.any():
        return np.inf
    return float(c[ends[good].max() - 1])


def both_tau(c0, y0, c1, y1, alpha: float, min_accepted: int = cfg.MIN_ACCEPTED) -> float:
    """Most permissive single threshold that meets the tolerance in each group.

    Same definition as :func:`care.analysis.conservative_global` for two groups
    (in each group either nothing is accepted, or at least ``min_accepted``
    cases are accepted with error at most alpha), computed without a Python
    loop over candidate thresholds so that it can be refitted in a bootstrap.
    """
    cand = np.unique(np.concatenate([c0[np.isfinite(c0)], c1[np.isfinite(c1)]]))[::-1]
    if not cand.size:
        return np.inf
    ok = np.ones(cand.size, bool)
    total = np.zeros(cand.size)
    for c, y in ((c0, y0), (c1, y1)):
        m = np.isfinite(c)
        order = np.argsort(-c[m], kind="stable")
        cs, errs = c[m][order], np.r_[0.0, np.cumsum(1.0 - y[m][order])]
        k = np.searchsorted(-cs, -cand, side="right")          # cases with confidence >= t
        e = errs[k]
        ok &= (k == 0) | ((k >= min_accepted) & (e / np.maximum(k, 1) <= alpha))
        total += k
    ok &= total >= min_accepted
    return float(cand[ok][-1]) if ok.any() else np.inf


def crossfit_accept(c, y, folds, alpha: float, scope: str = "pooled", rule: str = "point"):
    """Held-out acceptance decisions under cross-fitted thresholds.

    Parameters
    ----------
    c, y : numpy.ndarray
        Confidence and correctness, shape (2, n), reference variant first.
    folds : numpy.ndarray
        Fold label per pair. Both versions of a pair share a fold.
    alpha : float
        Error tolerance.
    scope : {"pooled", "both", "group"}
        ``pooled``: one threshold fitted on the two groups' training cases
        taken together, the policy of an institution that applies a single
        threshold to everyone. ``both``: one threshold that meets the tolerance
        in each group separately in the training folds. ``group``: a separate
        threshold per group.
    rule : str
        Fitting rule, see :func:`fit_tau`; ``both`` supports ``point`` only.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        Boolean acceptance of shape (2, n), and per group the number of folds
        in which nothing was accepted.
    """
    acc = np.zeros(c.shape, dtype=bool)
    empty = np.zeros(2, dtype=int)
    fin = np.isfinite(c)
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        if scope == "pooled":
            t = fit_tau(c[:, tr].ravel(), y[:, tr].ravel(), alpha, rule)
            taus = (t, t)
        elif scope == "both":
            t = both_tau(c[0, tr], y[0, tr], c[1, tr], y[1, tr], alpha)
            taus = (t, t)
        elif scope == "group":
            taus = tuple(fit_tau(c[g, tr], y[g, tr], alpha, rule) for g in (0, 1))
        else:
            raise ValueError(f"unknown scope {scope!r}")
        for g in (0, 1):
            take = te & fin[g] & (c[g] >= taus[g])
            acc[g] |= take
            empty[g] += int(not take.any())
    return acc, empty


def _held_out_stats(acc, y):
    n = acc.shape[1]
    k = acc.sum(axis=1)
    e = ((1.0 - y) * acc).sum(axis=1)
    cov = k / n
    err = np.where(k > 0, e / np.maximum(k, 1), np.nan)
    return cov, err, k, e


def common_threshold(datasets, models, method="tlp", conditions=CONDITIONS, alphas=ALPHAS,
                     n_boot: int = cfg.N_BOOT, margin_alpha: float = PRIMARY_ALPHA,
                     seed: int = cfg.SEED) -> pd.DataFrame:
    """Held-out coverage and accepted-case error by group under cross-fitted thresholds.

    For every dataset, model and prompt, thresholds are fitted on four fifths
    of the pairs and applied to the remaining fifth. The bootstrap resamples
    pairs and refits the thresholds in each replicate, so the intervals include
    the uncertainty of the threshold itself; a resampled pair keeps its fold, so
    no pair is in both the fitting and the held-out part.

    Returns
    -------
    pandas.DataFrame
        One row per dataset, model, prompt, tolerance, scope and rule, with
        held-out coverage, accepted counts and error per group, the reference
        minus demographic differences with intervals, the number of pairs
        accepted for one version only, and the number of folds with no
        accepted case.
    """
    plan = [(a, "pooled", "point") for a in alphas] + \
           [(a, "both", "point") for a in alphas] + \
           [(a, "group", "point") for a in alphas] + \
           [(margin_alpha, sc, ru) for sc in ("pooled", "group") for ru in ("ucb", "ucb_bonf")]
    rows = []
    for ds, mdl, a in _slices(datasets, models, method, conditions):
        n = a["ids"].size
        folds = np.random.default_rng(seed + 1).permutation(np.arange(n) % N_FOLDS)
        idx = _boot_idx(n, n_boot, seed)
        for c in conditions:
            cc, y = a[c]["c"], a[c]["y"]
            for alpha, scope, rule in plan:
                acc, empty = crossfit_accept(cc, y, folds, alpha, scope, rule)
                cov, err, k, e = _held_out_stats(acc, y)
                draws = np.empty((len(idx), 4))
                for b, i in enumerate(idx):
                    ab, _ = crossfit_accept(cc[:, i], y[:, i], folds[i], alpha, scope, rule)
                    cb, eb, _, _ = _held_out_stats(ab, y[:, i])
                    draws[b] = (cb[0] - cb[1], eb[0] - eb[1], eb[0], eb[1])
                ci = [_ci(draws[:, j]) for j in range(4)]
                d = acc[0].astype(float) - acc[1].astype(float)
                rows.append(_ident(
                    ds, mdl, c, method=method, alpha=alpha, scope=scope, rule=rule, n_pairs=n,
                    cov_ref=cov[0], cov_dem=cov[1], d_cov=cov[0] - cov[1],
                    d_cov_lo=ci[0][0], d_cov_hi=ci[0][1],
                    n_acc_ref=int(k[0]), n_acc_dem=int(k[1]),
                    n_err_ref=int(e[0]), n_err_dem=int(e[1]),
                    err_ref=err[0], err_ref_lo=ci[2][0], err_ref_hi=ci[2][1],
                    err_dem=err[1], err_dem_lo=ci[3][0], err_dem_hi=ci[3][1],
                    d_err=err[0] - err[1], d_err_lo=ci[1][0], d_err_hi=ci[1][1],
                    only_ref_accepted=int((d > 0).sum()), only_dem_accepted=int((d < 0).sum()),
                    sd_pair_diff=float(d.std(ddof=1)),
                    empty_folds_ref=int(empty[0]), empty_folds_dem=int(empty[1]),
                    in_sample_sar_ref=safe_coverage(cc[0], y[0], alpha),
                    in_sample_sar_dem=safe_coverage(cc[1], y[1], alpha)))
    return pd.DataFrame(rows)


def power(common: pd.DataFrame, effect: float = SESOI, available: dict | None = None) -> pd.DataFrame:
    """Pairs needed to detect a difference in held-out coverage of ``effect``.

    Uses the standard deviation of the within-pair difference in acceptance
    under the pooled cross-fitted threshold (two-sided 5% test).

    Parameters
    ----------
    common : pandas.DataFrame
        Output of :func:`common_threshold`.
    effect : float
        Difference in coverage to detect.
    available : dict, optional
        Dataset name to the number of pairs the rules can build.
    """
    c = common[(common["scope"] == "pooled") & (common["rule"] == "point")].copy()
    z = stats.norm.ppf(0.975)
    out = c[["dataset", "model", "prompt", "alpha", "n_pairs", "sd_pair_diff"]].copy()
    out["se_at_n"] = out["sd_pair_diff"] / np.sqrt(out["n_pairs"])
    out["ci_half_width_at_n"] = z * out["se_at_n"]
    for pw in (0.80, 0.90):
        out[f"n_for_{int(pw * 100)}pct_power"] = np.ceil(
            ((z + stats.norm.ppf(pw)) * out["sd_pair_diff"] / effect) ** 2)
    if available:
        out["n_available"] = out["dataset"].map(available)
        out["ci_half_width_at_available"] = z * out["sd_pair_diff"] / np.sqrt(out["n_available"])
    return out


# --------------------------------------------------------------------------- #
# The directional pattern: counts and dependence-aware tests
# --------------------------------------------------------------------------- #

def _gap(c, y, alpha):
    """SAR difference and whether it is estimable.

    A SAR at or below ``NON_ESTIMABLE`` means no usable threshold. When that is
    so for both groups the comparison is non-estimable and takes no part in
    directional counts.
    """
    r, d = safe_coverage(c[0], y[0], alpha), safe_coverage(c[1], y[1], alpha)
    return r - d, max(r, d) > NON_ESTIMABLE


def _cube(datasets, models, method="tlp", conditions=CONDITIONS):
    """Per dataset: pair ids common to all models, and (model, prompt) arrays."""
    out = {}
    for ds in datasets:
        got = {m: arrays(ds, m, method, conditions) for m in models}
        got = {m: a for m, a in got.items() if a is not None}
        if not got:
            continue
        ids = functools.reduce(np.intersect1d, [a["ids"] for a in got.values()])
        blk = {}
        for m, a in got.items():
            sel = np.searchsorted(a["ids"], ids)
            for c in conditions:
                blk[(m, c)] = (a[c]["c"][:, sel], a[c]["y"][:, sel])
        out[ds] = {"ids": ids, "blk": blk}
    return out


def _pattern(cube, alpha, pick=None, flip=None):
    """Summary statistics of the directional pattern.

    ``pick[ds]`` selects pairs (a bootstrap resample); ``flip[ds]`` marks pairs
    whose two versions are swapped (a permutation).
    """
    gaps, est, dconf, key = [], [], [], []
    for ds, d in cube.items():
        for (m, c), (cc, y) in d["blk"].items():
            if pick is not None:
                cc, y = cc[:, pick[ds]], y[:, pick[ds]]
            if flip is not None:
                f = flip[ds]
                cc = np.where(f, cc[::-1], cc)
                y = np.where(f, y[::-1], y)
            g, e = _gap(cc, y, alpha)
            gaps.append(g)
            est.append(e)
            dconf.append(np.nanmean(cc[0] - cc[1]))
            key.append((ds, m))
    gaps, est, dconf = np.array(gaps), np.array(est), np.array(dconf)
    pos, neg = int(((gaps > 0) & est).sum()), int(((gaps < 0) & est).sum())
    avg = pd.Series(np.where(est, gaps, np.nan)).groupby([f"{a}|{b}" for a, b in key]).mean()
    return {"n_pos": pos, "n_neg": neg, "n_zero": int(((gaps == 0) & est).sum()),
            "n_non_estimable": int((~est).sum()), "net": pos - neg,
            "mean_gap": float(gaps[est].mean()) if est.any() else np.nan,
            "median_gap": float(np.median(gaps[est])) if est.any() else np.nan,
            "avg_pos": int((avg > 0).sum()), "avg_neg": int((avg < 0).sum()),
            "avg_net": int((avg > 0).sum() - (avg < 0).sum()),
            "mean_d_conf": float(dconf.mean())}


def directional(datasets, models, method="tlp", conditions=CONDITIONS, alphas=ALPHAS) -> pd.DataFrame:
    """Counts of comparisons favouring each variant, overall and by attribute."""
    cube = _cube(datasets, models, method, conditions)
    rows = []
    for alpha in alphas:
        for label, keep in [("all", lambda d: True)] + \
                [(ax, lambda d, ax=ax: cfg.axis_of(d) == ax)
                 for ax in sorted({cfg.axis_of(d) for d in cube})]:
            sub = {d: v for d, v in cube.items() if keep(d)}
            if sub:
                rows.append({"alpha": alpha, "subset": label,
                             "n_comparisons": sum(len(v["blk"]) for v in sub.values()),
                             **_pattern(sub, alpha)})
    return pd.DataFrame(rows)


def _vignette_keys(cube, tie_axes: bool):
    """Randomisation unit of every pair: the pair itself, or (with
    ``tie_axes``) the source vignette shared by a race and a sex pair."""
    keys = {}
    for ds, d in cube.items():
        pre = cfg.source_of(ds) if tie_axes else ds
        keys[ds] = np.array([f"{pre}:{i}" for i in d["ids"]])
    return keys


def permutation(datasets, models, method="tlp", conditions=CONDITIONS, alphas=ALPHAS,
                n_perm: int = 5000, tie_axes: bool = False, seed: int = cfg.SEED) -> pd.DataFrame:
    """Within-pair label-swap test of the directional pattern.

    Under the hypothesis that the descriptor has no effect, the two versions of
    a pair are exchangeable. Each permutation swaps the reference and
    demographic labels of a random subset of pairs, and applies the same swap
    to every model and every prompt, since all of them answered the same two
    texts. The 48 comparisons therefore keep their dependence in every
    permutation, which is what a sign test over 48 comparisons ignores.

    Parameters
    ----------
    n_perm : int
        Number of random label swaps.
    tie_axes : bool
        Swap the race pair and the sex pair built from one vignette together.
        The default swaps them independently, which the hypothesis permits.

    Returns
    -------
    pandas.DataFrame
        Per tolerance and statistic: the observed value, the mean and 2.5-97.5
        percentiles of its permutation distribution, and a two-sided p-value.
    """
    cube = _cube(datasets, models, method, conditions)
    keys = _vignette_keys(cube, tie_axes)
    units = np.unique(np.concatenate(list(keys.values())))
    where = {ds: np.searchsorted(units, k) for ds, k in keys.items()}
    rng = np.random.default_rng(seed)
    names = ("net", "avg_net", "mean_gap", "mean_d_conf")
    rows = []
    for alpha in alphas:
        obs = _pattern(cube, alpha)
        null = np.empty((n_perm, len(names)))
        for b in range(n_perm):
            f = rng.random(units.size) < 0.5
            st = _pattern(cube, alpha, flip={ds: f[w] for ds, w in where.items()})
            null[b] = [st[k] for k in names]
        for j, k in enumerate(names):
            x = null[:, j][np.isfinite(null[:, j])]
            p = (1 + int((np.abs(x) >= abs(obs[k]) - 1e-12).sum())) / (x.size + 1)
            rows.append({"alpha": alpha, "statistic": k, "observed": obs[k],
                         "n_pos": obs["n_pos"], "n_neg": obs["n_neg"],
                         "n_non_estimable": obs["n_non_estimable"],
                         "null_mean": float(x.mean()), "null_lo": float(np.quantile(x, .025)),
                         "null_hi": float(np.quantile(x, .975)), "p_two_sided": p,
                         "n_perm": n_perm, "tie_axes": tie_axes,
                         "n_units": int(units.size)})
    return pd.DataFrame(rows)


def pooled_bootstrap(datasets, models, method="tlp", conditions=CONDITIONS, alphas=ALPHAS,
                     n_boot: int = cfg.N_BOOT, seed: int = cfg.SEED) -> pd.DataFrame:
    """Cluster bootstrap of the pooled directional summaries.

    Pairs are resampled within each pair set and the same resample is applied
    to every model and prompt, so the interval for the pooled mean difference
    reflects the dependence between comparisons that share pairs.
    """
    cube = _cube(datasets, models, method, conditions)
    rng = np.random.default_rng(seed)
    names = ("mean_gap", "median_gap", "n_pos", "net", "avg_pos", "mean_d_conf")
    rows = []
    for alpha in alphas:
        obs = _pattern(cube, alpha)
        draws = np.empty((n_boot, len(names)))
        for b in range(n_boot):
            pick = {ds: rng.integers(0, d["ids"].size, d["ids"].size) for ds, d in cube.items()}
            st = _pattern(cube, alpha, pick=pick)
            draws[b] = [st[k] for k in names]
        for j, k in enumerate(names):
            lo, hi = _ci(draws[:, j])
            rows.append({"alpha": alpha, "statistic": k, "observed": obs[k], "lo": lo, "hi": hi,
                         "share_draws_above_zero": float((draws[:, j] > 0).mean())})
    return pd.DataFrame(rows)


def mixed_model(datasets, models, conditions=CONDITIONS) -> pd.DataFrame:
    """Mixed model of the pair-level difference in confidence.

    One observation per pair, model and prompt: the reference-minus-demographic
    difference in confidence (and, separately, in the probability of the keyed
    answer). A random intercept for the source vignette accounts for the same
    vignette being answered by every model under every prompt and, where a
    vignette is in both a race and a sex set, in both. Fixed effects are the
    attribute means.
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("  statsmodels is not installed: mixed model skipped (pip install statsmodels)")
        return pd.DataFrame()
    recs = []
    for ds, mdl, a in _slices(datasets, models, "tlp", conditions):
        for c in conditions:
            g = gold_prob(ds, mdl, c, a["ids"])
            recs.append(pd.DataFrame({
                "vignette": [f"{cfg.source_of(ds)}:{i}" for i in a["ids"]],
                "axis": cfg.axis_of(ds), "source": cfg.source_of(ds),
                "model": cfg.short(mdl), "prompt": c,
                "d_conf": a[c]["c"][0] - a[c]["c"][1], "d_gold": g[0] - g[1]}))
    if not recs:
        return pd.DataFrame()
    long = pd.concat(recs, ignore_index=True)
    axes = sorted(long["axis"].unique())
    for ax in axes:                                  # numeric indicators, one per attribute
        long[f"is_{ax}"] = (long["axis"] == ax).astype(float)
    groups = pd.factorize(long["vignette"])[0]
    rows = []
    for outcome in ("d_conf", "d_gold"):
        keep = long[outcome].notna().to_numpy()
        d, g = long[keep], groups[keep]
        specs = [("overall", f"{outcome} ~ 1", {"Intercept": float(d[outcome].mean())})]
        if len(axes) > 1:
            specs.append(("by attribute", f"{outcome} ~ 0 + " + " + ".join(f"is_{a}" for a in axes),
                          {f"is_{a}": float(d.loc[d["axis"] == a, outcome].mean()) for a in axes}))
        for label, formula, plain in specs:
            # Powell first: L-BFGS can stop on the zero-variance boundary with
            # fixed effects far from the sample mean.
            fit = None
            for how in ("powell", "lbfgs"):
                try:
                    fit = smf.mixedlm(formula, d, groups=g).fit(reml=True, method=how)
                    break
                except Exception as e:
                    print(f"  mixed model {outcome}/{label}/{how} failed: {type(e).__name__}: {e}")
            if fit is None:
                continue
            ci = fit.conf_int()
            for term in fit.fe_params.index:
                rows.append({"outcome": outcome, "model": label, "term": term,
                             "estimate": float(fit.fe_params[term]), "se": float(fit.bse[term]),
                             "lo": float(ci.loc[term, 0]), "hi": float(ci.loc[term, 1]),
                             "p": float(fit.pvalues[term]), "plain_mean": plain[term],
                             "n_obs": int(fit.nobs),
                             "n_vignettes": int(d["vignette"].nunique()),
                             "var_vignette": float(fit.cov_re.iloc[0, 0]),
                             "var_residual": float(fit.scale)})
    return pd.DataFrame(rows)


def overlap(datasets) -> pd.DataFrame:
    """Number of source vignettes shared between pair sets of one corpus."""
    ids = {}
    for ds in datasets:
        path = os.path.join(cfg.PAIRS_DIR, f"{ds}.jsonl")
        if os.path.exists(path):
            ids[ds] = {json.loads(l)["pair_id"] for l in open(path, encoding="utf-8") if l.strip()}
    rows = []
    names = sorted(ids)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if cfg.source_of(a) == cfg.source_of(b):
                rows.append({"set_a": a, "set_b": b, "n_a": len(ids[a]), "n_b": len(ids[b]),
                             "shared_vignettes": len(ids[a] & ids[b])})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Interval methods for the SAR difference
# --------------------------------------------------------------------------- #

def bca(datasets, models, method="tlp", conditions=CONDITIONS, alpha: float = PRIMARY_ALPHA,
        n_boot: int = cfg.N_BOOT) -> pd.DataFrame:
    """Percentile and bias-corrected accelerated intervals for the SAR difference.

    SAR is a maximum over thresholds, so its bootstrap distribution is lumpy
    and can be far from centred on the estimate; ``share_draws_below`` shows
    how far (0.5 is centred). BCa corrects for that median bias and for
    skewness, using a leave-one-pair-out jackknife for the acceleration.
    """
    rows = []
    for ds, mdl, a in _slices(datasets, models, method, conditions):
        n = a["ids"].size
        idx = _boot_idx(n, n_boot)
        for c in conditions:
            cc, y = a[c]["c"], a[c]["y"]
            est, ok = _gap(cc, y, alpha)
            draws = np.array([_gap(cc[:, i], y[:, i], alpha)[0] for i in idx])
            lo, hi = _ci(draws)
            below = float(((draws < est) + 0.5 * (draws == est)).mean())
            keep = np.ones(n, bool)
            jack = np.empty(n)
            for i in range(n):
                keep[i] = False
                jack[i] = _gap(cc[:, keep], y[:, keep], alpha)[0]
                keep[i] = True
            dj = jack.mean() - jack
            denom = 6.0 * (dj ** 2).sum() ** 1.5
            acc_ = float((dj ** 3).sum() / denom) if denom > 0 else 0.0
            blo = bhi = np.nan
            if 0 < below < 1:
                z0 = stats.norm.ppf(below)
                q = [stats.norm.cdf(z0 + (z0 + z) / (1 - acc_ * (z0 + z)))
                     for z in stats.norm.ppf([0.025, 0.975])]
                blo, bhi = (float(v) for v in np.quantile(draws, q))
            rows.append(_ident(ds, mdl, c, method=method, alpha=alpha, sar_gap=est, estimable=ok,
                               pct_lo=lo, pct_hi=hi, bca_lo=blo, bca_hi=bhi,
                               share_draws_below=below, acceleration=acc_,
                               distinct_draw_values=int(np.unique(draws).size)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Subsets: source split and sensitivity screens
# --------------------------------------------------------------------------- #

def load_flags() -> pd.DataFrame:
    """The per-pair table written by ``python -m care.controls``."""
    path = os.path.join(cfg.PAIRS_DIR, "pair_flags.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    f = pd.read_csv(path, dtype={"pair_id": str}, keep_default_na=False)
    for col in ("same_sex_kin", "spouse_noun"):
        f[col] = f[col].astype(str).str.lower().eq("true")
    return f


def _subset_rows(ds, mdl, a, conditions, mask, label, alpha, n_boot, min_sar_n=200):
    rows = []
    n = int(mask.sum())
    if n < 20:
        return rows
    idx = _boot_idx(n, n_boot)
    for c in conditions:
        cc, y = a[c]["c"][:, mask], a[c]["y"][:, mask]
        m, lo, hi = _mean_ci(cc[0] - cc[1], idx)
        row = _ident(ds, mdl, c, subset=label, n_pairs=n, acc_ref=float(y[0].mean()),
                     acc_dem=float(y[1].mean()), d_conf=m, d_conf_lo=lo, d_conf_hi=hi,
                     auroc_ref=auroc(cc[0], y[0]), auroc_dem=auroc(cc[1], y[1]))
        if n >= min_sar_n:                      # SAR on a small subset is noise
            g, ok = _gap(cc, y, alpha)
            draws = np.array([_gap(cc[:, i], y[:, i], alpha)[0] for i in idx])
            glo, ghi = _ci(draws)
            row.update(sar_ref=safe_coverage(cc[0], y[0], alpha),
                       sar_dem=safe_coverage(cc[1], y[1], alpha),
                       sar_gap=g, sar_gap_lo=glo, sar_gap_hi=ghi, estimable=ok)
        rows.append(row)
    return rows


def by_split(datasets, models, conditions=CONDITIONS, alpha: float = PRIMARY_ALPHA,
             n_boot: int = cfg.N_BOOT) -> pd.DataFrame:
    """Results by source split.

    A model that has seen a benchmark split in training may be more confident
    on its verbatim wording. Test items are the least likely to have been used
    for instruction tuning, so a difference confined to training-split items
    would point to exposure rather than to the descriptor.
    """
    flags = load_flags()
    if flags.empty:
        print("  pair_flags.csv not found: run `python -m care.controls` first")
        return pd.DataFrame()
    rows = []
    for ds, mdl, a in _slices(datasets, models, "tlp", conditions):
        sp = flags[flags["dataset"] == ds].set_index("pair_id")["split"]
        split = pd.Series(a["ids"]).map(sp).to_numpy()
        for label in pd.unique(split[pd.notna(split)]):
            rows += _subset_rows(ds, mdl, a, conditions, split == label, f"split={label}",
                                 alpha, n_boot)
    return pd.DataFrame(rows)


SCREENS = {
    # label: (which attribute it applies to, function of the flags frame -> pairs REMOVED)
    "no other person of the patient's sex, no spouse noun":
        ("gender", lambda f: f["same_sex_kin"] | f["spouse_noun"] | (f["n_patient_np"].astype(int) > 1)),
    "no spouse noun": ("gender", lambda f: f["spouse_noun"]),
    "no race tier A condition": ("race", lambda f: f["race_tier_a"].ne("")),
    "no race tier A or B condition":
        ("race", lambda f: f["race_tier_a"].ne("") | f["race_tier_b"].ne("")),
}


def sensitivity(datasets, models, conditions=CONDITIONS, alpha: float = PRIMARY_ALPHA,
                n_boot: int = cfg.N_BOOT) -> tuple:
    """Results after removing pairs caught by each screen.

    Returns
    -------
    tuple of (pandas.DataFrame, pandas.DataFrame)
        Per-slice results on each screened subset (and on all pairs, for
        reference), and the directional counts on each subset.
    """
    flags = load_flags()
    if flags.empty:
        print("  pair_flags.csv not found: run `python -m care.controls` first")
        return pd.DataFrame(), pd.DataFrame()
    rows, counts = [], []
    for label, (axis, removed) in SCREENS.items():
        sub = [d for d in datasets if cfg.axis_of(d) == axis]
        keep_ids = {}
        for ds in sub:
            f = flags[flags["dataset"] == ds]
            keep_ids[ds] = set(f.loc[~removed(f), "pair_id"])
        for ds, mdl, a in _slices(sub, models, "tlp", conditions):
            mask = np.isin(a["ids"], list(keep_ids[ds]))
            rows += _subset_rows(ds, mdl, a, conditions, np.ones(mask.size, bool), "all pairs",
                                 alpha, n_boot)
            rows += _subset_rows(ds, mdl, a, conditions, mask, label, alpha, n_boot)
        cube = _cube(sub, models, "tlp", conditions)
        for name, restrict in (("all pairs", False), (label, True)):
            c2 = cube
            if restrict:
                c2 = {}
                for ds, d in cube.items():
                    m = np.isin(d["ids"], list(keep_ids[ds]))
                    c2[ds] = {"ids": d["ids"][m],
                              "blk": {k: (c[:, m], y[:, m]) for k, (c, y) in d["blk"].items()}}
            if c2:
                counts.append({"screen": label, "axis": axis, "subset": name,
                               "pairs": int(sum(d["ids"].size for d in c2.values())),
                               **_pattern(c2, alpha)})
    res = pd.DataFrame(rows).drop_duplicates(["dataset", "model", "prompt", "subset"]) \
        if rows else pd.DataFrame()
    return res, pd.DataFrame(counts)


# --------------------------------------------------------------------------- #
# Confidence source on a fixed set of answers
# --------------------------------------------------------------------------- #

def tlp_vs_ce(datasets, models, conditions=CONDITIONS, alpha: float = PRIMARY_ALPHA) -> pd.DataFrame:
    """Token-probability and elicited confidence compared on the same answers.

    The two procedures use different answer instructions, so the model may
    select different answers under each. This reports how often they agree and
    then removes the difference: for the answer given under the elicited
    procedure, confidence is taken either from the stated number or from the
    token probability the model assigned to that same option. Both are scored
    against the same answers, so any difference is due to the confidence source.
    """
    from .infer import parse_ce, LETTERS
    rows = []
    for ds in datasets:
        K = cfg.DATASETS[ds]["n_options"]
        groups = cfg.DATASETS[ds]["groups"]
        for mdl in models:
            for c in conditions:
                raw = _raw(ds, mdl, c)
                if raw.empty or "tlp_option_probs" not in raw or "ce_raw_text" not in raw:
                    continue
                key = ["question_id", "demographic"]
                t = raw[raw["tlp_option_probs"].notna()].drop_duplicates(key, keep="last")
                e = raw[raw["ce_raw_text"].notna()].drop_duplicates(key, keep="last")
                j = t[key + ["answer_idx", "tlp_answer", "tlp_conf", "tlp_correct",
                             "tlp_option_probs"]].merge(e[key + ["ce_raw_text"]], on=key)
                if j.empty:
                    continue
                parsed = [parse_ce(x, LETTERS[:K]) for x in j["ce_raw_text"]]
                j["ce_answer"] = [a for a, _ in parsed]
                j["ce_conf"] = [np.nan if v is None else v for _, v in parsed]
                j["ce_correct"] = (j["ce_answer"] == j["answer_idx"]).astype(float)
                # token probability of the option chosen under the elicited procedure
                j["tlp_on_ce"] = [json.loads(p).get(a, np.nan) if a else np.nan
                                  for p, a in zip(j["tlp_option_probs"], j["ce_answer"])]
                for g in groups:
                    s = j[j["demographic"].astype(str).str.lower() == g]
                    if len(s) < 2 * cfg.MIN_ACCEPTED:
                        continue
                    ok = s["ce_answer"].notna()
                    same = ok & (s["ce_answer"] == s["tlp_answer"])
                    rows.append(_ident(
                        ds, mdl, c, group=g, n=len(s), ce_parsed=float(ok.mean()),
                        answers_agree=float(same.sum() / max(1, ok.sum())),
                        acc_tlp=float(s["tlp_correct"].mean()),
                        acc_ce=float(s["ce_correct"].mean()),
                        acc_where_agree=float(s.loc[same, "ce_correct"].mean()) if same.any() else np.nan,
                        # same answers (those given under the elicited procedure), two sources
                        auroc_ce_conf=auroc(s["ce_conf"], s["ce_correct"]),
                        auroc_tlp_on_ce_answers=auroc(s["tlp_on_ce"], s["ce_correct"]),
                        sar_ce_conf=safe_coverage(s["ce_conf"], s["ce_correct"], alpha),
                        sar_tlp_on_ce_answers=safe_coverage(s["tlp_on_ce"], s["ce_correct"], alpha),
                        distinct_ce_values=int(s["ce_conf"].nunique()),
                        # as submitted: each source with its own answers
                        sar_tlp_own_answers=safe_coverage(s["tlp_conf"], s["tlp_correct"], alpha),
                        auroc_tlp_own_answers=auroc(s["tlp_conf"], s["tlp_correct"])))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Analyses that need the new inference runs
# --------------------------------------------------------------------------- #

def controls(models, conditions=CONDITIONS, alpha: float = PRIMARY_ALPHA,
             n_boot: int = cfg.N_BOOT) -> pd.DataFrame:
    """Each control set beside the demographic set it controls for.

    For every model, source and prompt: the difference between the two versions
    in the control set and in its parent set, on the same statistics. The
    placebo row is the size of difference that changing one clinically
    irrelevant word produces.
    """
    names = [d for d in cfg.CONTROL_DATASETS if cfg.DATASETS[d]["kind"] != "full"]
    parents = sorted({cfg.DATASETS[d]["parent"] for d in names})
    pl = pair_level(names + parents, models, "tlp", conditions, n_boot)
    if pl.empty or not pl["dataset"].isin(names).any():
        return pd.DataFrame()
    rk = ranking(names + parents, models, "tlp", conditions, n_boot)
    gaps = []
    for ds, mdl, a in _slices(names + parents, models, "tlp", conditions):
        idx = _boot_idx(a["ids"].size, n_boot)
        for c in conditions:
            cc, y = a[c]["c"], a[c]["y"]
            g, ok = _gap(cc, y, alpha)
            lo, hi = _ci(np.array([_gap(cc[:, i], y[:, i], alpha)[0] for i in idx]))
            gaps.append({"dataset": ds, "model": cfg.short(mdl), "prompt": c, "sar_gap": g,
                         "sar_gap_lo": lo, "sar_gap_hi": hi, "sar_estimable": ok})
    out = pl.merge(rk[["dataset", "model", "prompt", "auroc_diff", "auroc_diff_lo",
                       "auroc_diff_hi", "aurc_diff", "aurc_diff_lo", "aurc_diff_hi"]],
                   on=["dataset", "model", "prompt"]).merge(
        pd.DataFrame(gaps), on=["dataset", "model", "prompt"])
    out["parent"] = out["dataset"].map(lambda d: cfg.DATASETS[d].get("parent", d))
    return out.sort_values(["parent", "model", "prompt", "kind"]).reset_index(drop=True)


def edit_effect(models, conditions=CONDITIONS, n_boot: int = cfg.N_BOOT) -> pd.DataFrame:
    """Verbatim benchmark text against the same text with one neutral word added.

    The male version of the submitted sex pairs is the verbatim benchmark text;
    the male version of the both-edited pairs is that text with one neutral
    word inserted. Their difference, on the same vignettes, is the effect of
    an edit that carries no demographic information. The female versions give
    the effect of adding a second edit to an already edited text.
    """
    rows = []
    for src in ("medqa", "medmcqa"):
        base, ed = f"pairs_{src}_gender", f"pairs_{src}_gender_edited"
        for mdl in models:
            a, b = arrays(base, mdl, "tlp", conditions), arrays(ed, mdl, "tlp", conditions)
            if a is None or b is None:
                continue
            ids = np.intersect1d(a["ids"], b["ids"])
            ia, ib = np.searchsorted(a["ids"], ids), np.searchsorted(b["ids"], ids)
            idx = _boot_idx(ids.size, n_boot)
            for c in conditions:
                for g, name in enumerate(cfg.DATASETS[base]["groups"]):
                    d = a[c]["c"][g, ia] - b[c]["c"][g, ib]
                    m, lo, hi = _mean_ci(d, idx)
                    ya, yb = a[c]["y"][g, ia], b[c]["y"][g, ib]
                    rows.append({"source": src, "model": cfg.short(mdl), "prompt": c,
                                 "version": name,
                                 "comparison": ("verbatim vs one neutral word" if g == 0 else
                                                "sex-substituted vs substituted plus neutral word"),
                                 "n_pairs": int(ids.size), "d_conf": m, "d_conf_lo": lo,
                                 "d_conf_hi": hi, "acc_without_word": float(ya.mean()),
                                 "acc_with_word": float(yb.mean()),
                                 "sar_without_word": safe_coverage(a[c]["c"][g, ia], ya, PRIMARY_ALPHA),
                                 "sar_with_word": safe_coverage(b[c]["c"][g, ib], yb, PRIMARY_ALPHA)})
    return pd.DataFrame(rows)


def prompt_controls(datasets, models, alphas=(PRIMARY_ALPHA,), n_boot: int = cfg.N_BOOT) -> pd.DataFrame:
    """Prompt contrasts that separate persona, length and fairness wording.

    Each contrast in :data:`care.prompts.CONTROL_CONTRASTS` is evaluated with
    the paper's paired bootstrap on the pairs complete under both prompts.
    """
    rows = []
    for ds in datasets:
        for mdl in models:
            for cond, base, what in CONTROL_CONTRASTS:
                df = _slice(ds, mdl, "tlp", (base, cond))
                if df is None:
                    continue
                _, con = analyse_slice(df, ds, alphas, n_boot, baseline=base)
                for r in con:
                    rows.append({**_ident(ds, mdl), "contrast": f"{cond} - {base}",
                                 "isolates": what, **{k: v for k, v in r.items() if k != "dataset"}})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def _fmt(x, pct=True, signed=True):
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{100 * x:{'+' if signed else ''}.1f}" if pct else f"{x:.3f}"


def summarise(res: dict) -> str:
    """Short markdown summary of the main numbers, for the response letter."""
    L = ["# Revision analyses: summary", "",
         "Differences are reference (White or male) minus demographic (Black or female), "
         "in percentage points unless stated.", ""]
    d = res.get("directional")
    if d is not None and len(d):
        L += ["## Directional counts (estimable comparisons only)", "",
              "| alpha | subset | favour reference | favour demographic | zero | non-estimable |",
              "|---|---|---|---|---|---|"]
        for _, r in d.iterrows():
            L.append(f"| {r.alpha:.2f} | {r.subset} | {r.n_pos} | {r.n_neg} | {r.n_zero} | "
                     f"{r.n_non_estimable} |")
        L.append("")
    p = res.get("permutation")
    if p is not None and len(p):
        L += ["## Label-swap permutation test", "",
              "| alpha | statistic | observed | null 2.5-97.5% | p |", "|---|---|---|---|---|"]
        for _, r in p.iterrows():
            L.append(f"| {r.alpha:.2f} | {r.statistic} | {r.observed:.4g} | "
                     f"{r.null_lo:.4g} to {r.null_hi:.4g} | {r.p_two_sided:.4f} |")
        L.append("")
    b = res.get("pooled")
    if b is not None and len(b):
        L += ["## Pooled summaries, cluster bootstrap", "",
              "| alpha | statistic | observed | 95% interval |", "|---|---|---|---|"]
        for _, r in b.iterrows():
            L.append(f"| {r.alpha:.2f} | {r.statistic} | {r.observed:.4g} | "
                     f"{r.lo:.4g} to {r.hi:.4g} |")
        L.append("")
    m = res.get("mixed")
    if m is not None and len(m):
        L += ["## Mixed model of the pair-level difference (probability scale)", "",
              "| outcome | term | estimate | 95% interval | p |", "|---|---|---|---|---|"]
        for _, r in m.iterrows():
            L.append(f"| {r.outcome} | {r.term} | {r.estimate:+.5f} | {r.lo:+.5f} to {r.hi:+.5f} | "
                     f"{r.p:.4g} |")
        L.append("")
    c = res.get("common")
    if c is not None and len(c):
        s = c[(c.scope == "pooled") & (c.rule == "point") & (c.alpha == PRIMARY_ALPHA)]
        L += [f"## One cross-fitted threshold for both groups (alpha = {PRIMARY_ALPHA})", "",
              f"- comparisons: {len(s)}",
              f"- median difference in held-out coverage: {_fmt(s.d_cov.median())}",
              f"- coverage higher for the reference variant in {(s.d_cov > 0).sum()}, "
              f"for the demographic variant in {(s.d_cov < 0).sum()}",
              f"- intervals excluding zero: {((s.d_cov_lo > 0) | (s.d_cov_hi < 0)).sum()}",
              f"- median interval width: {_fmt((s.d_cov_hi - s.d_cov_lo).median(), signed=False)}",
              f"- median difference in held-out accepted-case error: {_fmt(s.d_err.median())}",
              f"- held-out error above alpha: reference {(s.err_ref > PRIMARY_ALPHA).sum()}, "
              f"demographic {(s.err_dem > PRIMARY_ALPHA).sum()} of {len(s)}", ""]
        t = c[(c.alpha == PRIMARY_ALPHA) & (c.scope.isin(["pooled", "group"]))]
        L += ["### Held-out error by fitting rule (mean over comparisons)", "",
              "| scope | rule | coverage ref | coverage dem | error ref | error dem | "
              "share above alpha |", "|---|---|---|---|---|---|---|"]
        for (sc, ru), g in t.groupby(["scope", "rule"]):
            over = float(((g.err_ref > PRIMARY_ALPHA).mean() + (g.err_dem > PRIMARY_ALPHA).mean()) / 2)
            L.append(f"| {sc} | {ru} | {g.cov_ref.mean():.3f} | {g.cov_dem.mean():.3f} | "
                     f"{g.err_ref.mean():.3f} | {g.err_dem.mean():.3f} | {over:.2f} |")
        L.append("")
    pl = res.get("pair_level")
    if pl is not None and len(pl):
        L += ["## Pair-level differences", "",
              f"- comparisons: {len(pl)}",
              f"- mean confidence higher for the reference variant in {(pl.d_conf > 0).sum()}; "
              f"interval excludes zero in {((pl.d_conf_lo > 0) | (pl.d_conf_hi < 0)).sum()}",
              f"- median difference in confidence: {_fmt(pl.d_conf.median())}",
              f"- selected answer changes between versions: median "
              f"{_fmt(pl.answer_changed.median(), signed=False)}% of pairs "
              f"(range {_fmt(pl.answer_changed.min(), signed=False)} to "
              f"{_fmt(pl.answer_changed.max(), signed=False)})", ""]
    r = res.get("ranking")
    if r is not None and len(r):
        L += ["## Ranking quality", "",
              f"- AUROC higher for the reference variant in {(r.auroc_diff > 0).sum()} of {len(r)}; "
              f"interval excludes zero in {((r.auroc_diff_lo > 0) | (r.auroc_diff_hi < 0)).sum()}",
              f"- median AUROC difference: {r.auroc_diff.median():+.4f}",
              f"- AURC lower (better) for the reference variant in {(r.aurc_diff < 0).sum()} of {len(r)}", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    """Run the revision analyses on whatever responses are cached."""
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", choices=["cache", "controls", "all"], default="all",
                    help="cache: analyses of the submitted responses; controls: new runs")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="pair sets for the cache stage (default: the paper's four; "
                         "pass the *_full sets once they have been run)")
    ap.add_argument("--n-boot", type=int, default=cfg.N_BOOT)
    ap.add_argument("--n-perm", type=int, default=5000)
    ap.add_argument("--tag", default="", help="suffix for output files")
    ap.add_argument("--only", nargs="*", default=None, help="run only these analyses")
    args = ap.parse_args(argv)

    datasets = list(args.datasets or cfg.MAIN_DATASETS)
    models, nb = cfg.LOCAL_MODELS, args.n_boot
    tag = f"_{args.tag}" if args.tag else ""
    res = {}

    def step(name, fn):
        if args.only and name not in args.only:
            return
        print(f"[{name}]", flush=True)
        out = fn()
        frames = out if isinstance(out, tuple) else (out,)
        for suffix, f in zip(("", "_counts"), frames):
            if f is not None and len(f):
                save(f, f"r1_{name}{suffix}{tag}.csv")
        res[name] = frames[0]

    if args.stage in ("cache", "all"):
        step("pair_level", lambda: pair_level(datasets, models, n_boot=nb))
        step("ranking", lambda: ranking(datasets, models, n_boot=nb))
        step("directional", lambda: directional(datasets, models))
        step("permutation", lambda: permutation(datasets, models, n_perm=args.n_perm))
        step("pooled", lambda: pooled_bootstrap(datasets, models, n_boot=nb))
        step("mixed", lambda: mixed_model(datasets, models))
        step("overlap", lambda: overlap(datasets))
        step("common", lambda: common_threshold(datasets, models, n_boot=nb))
        if "common" in res and len(res["common"]):
            avail = {}
            for d in datasets:
                p = os.path.join(cfg.PAIRS_DIR, f"{cfg.DATASETS[d].get('parent', d)}_full.jsonl")
                if os.path.exists(p):
                    avail[d] = sum(1 for _ in open(p, encoding="utf-8")) // 2
            step("power", lambda: power(res["common"], available=avail or None))
        step("bca", lambda: bca(datasets, models, n_boot=nb))
        step("by_split", lambda: by_split(datasets, models, n_boot=nb))
        step("sensitivity", lambda: sensitivity(datasets, models, n_boot=nb))
        step("tlp_vs_ce", lambda: tlp_vs_ce(datasets, models))
    if args.stage in ("controls", "all"):
        step("controls", lambda: controls(models, n_boot=nb))
        step("edit_effect", lambda: edit_effect(models, n_boot=nb))
        step("prompt_controls", lambda: prompt_controls(list(cfg.MAIN_DATASETS), models, n_boot=nb))
        for name in ("controls", "edit_effect", "prompt_controls"):
            if name in res and not len(res[name]):
                print(f"  {name}: no responses on disk yet (run the revision notebook first)")

    # The summary is rebuilt from the files on disk, so a partial re-run
    # (--only) refreshes it without recomputing everything.
    for name in ("directional", "permutation", "pooled", "mixed", "common", "pair_level", "ranking"):
        path = os.path.join(cfg.RESULTS_DIR, f"r1_{name}{tag}.csv")
        if name not in res and os.path.exists(path):
            res[name] = pd.read_csv(path)
    path = os.path.join(cfg.RESULTS_DIR, f"r1_summary{tag}.md")
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    open(path, "w", encoding="utf-8").write(summarise(res))
    print("summary ->", path)
    print("CSV files written to", cfg.RESULTS_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
