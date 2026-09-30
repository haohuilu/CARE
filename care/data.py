"""Loading, schema validation, and sanity checks for stored inference outputs.

The validation in this module is deliberately loud. Several checks raise rather
than warn, because a downstream subgroup comparison built on an invalid split is
worse than no comparison at all.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from . import config as cfg

# --------------------------------------------------------------------------- #
# Record schema
# --------------------------------------------------------------------------- #


def _letter_to_idx(x) -> float:
    """Convert an option letter to a zero-based index.

    Parameters
    ----------
    x : object
        Option label such as ``"A"``, ``"c"``, or an integer-like value.

    Returns
    -------
    float
        Zero-based index, or ``numpy.nan`` when unparsable.
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip().upper()
    if len(s) == 1 and "A" <= s <= "Z":
        return float(ord(s) - ord("A"))
    try:
        return float(int(s))
    except (TypeError, ValueError):
        return np.nan


def discover_conditions(dataset: str, model: str, data_root: str | None = None) -> list:
    """List the prompting conditions present for a slice, in sorted order.

    Condition labels are read from the cache filenames rather than assumed, so a
    run using a different naming scheme (P1, P2, ...) needs no code change and
    old and new runs can coexist.

    Parameters
    ----------
    dataset, model : str
        Slice identifiers.
    data_root : str, optional
        Override for :data:`care.config.DATA_ROOT`.

    Returns
    -------
    list of str
        Condition labels found on disk; empty if the directory is absent.
    """
    d = os.path.join(data_root or cfg.DATA_ROOT, dataset, model, "calibration_cache")
    if not os.path.isdir(d):
        return []
    return sorted(f[:-6] for f in os.listdir(d) if f.endswith(".jsonl"))


def load_records(dataset: str, model: str, conf_method: str = "tlp",
                 conditions=None, data_root: str | None = None) -> pd.DataFrame:
    """Load stored inference records into the canonical CARE schema.

    Parameters
    ----------
    dataset : str
        Key in :data:`care.config.DATASETS`.
    model : str
        Model directory name.
    conf_method : {"tlp", "sc", "ce"}
        Confidence source. ``tlp`` uses token log-probabilities.
    conditions : sequence of str, optional
        Prompting conditions to load; discovered from the cache directory when
        omitted.
    data_root : str, optional
        Override for :data:`care.config.DATA_ROOT`.

    Returns
    -------
    pandas.DataFrame
        One row per model response.

    Raises
    ------
    FileNotFoundError
        If a condition's cache file is missing.
    KeyError
        If ``dataset`` is not registered.
    """
    if dataset not in cfg.DATASETS:
        raise KeyError(f"unknown dataset {dataset!r}; known: {sorted(cfg.DATASETS)}")
    spec = cfg.DATASETS[dataset]
    root = data_root or cfg.DATA_ROOT
    K = spec["n_options"]
    if conditions is None:
        conditions = discover_conditions(dataset, model, root)
        if not conditions:
            raise FileNotFoundError(
                os.path.join(root, dataset, model, "calibration_cache"))

    rows = []
    for cond in conditions:
        path = os.path.join(root, dataset, model, "calibration_cache", f"{cond}.jsonl")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        bad = 0
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    bad += 1           # partial line left by an interrupted write
        if bad:
            print(f"  note: skipped {bad} malformed line(s) in {path}")

    raw = pd.DataFrame(rows)
    m = conf_method
    # One file can hold several confidence methods, either on one line per item
    # (earlier runs) or on separate lines per method (current runs). Keep only
    # the lines that carry this method -- a result, or a recorded parse
    # failure -- and the latest line per item, so re-runs do not double count.
    has = pd.Series(False, index=raw.index)
    if f"{m}_conf" in raw.columns:
        has |= raw[f"{m}_conf"].notna()
    if f"{m}_parse_ok" in raw.columns:
        has |= raw[f"{m}_parse_ok"].eq(False)
    if m == "tlp" and "tlp_option_probs" in raw.columns:
        has |= raw["tlp_option_probs"].notna()
    # An empty reply is an API failure, not a model answer: treat as not run.
    rt = f"{m}_raw_text"
    if rt in raw.columns and f"{m}_parse_ok" in raw.columns and "api_params" in raw.columns:
        empty = (raw[f"{m}_parse_ok"].eq(False) & raw["api_params"].notna()
                 & raw[rt].fillna("").astype(str).str.strip().eq(""))
        if empty.any():
            print(f"  note: {int((empty & has).sum())} empty {m} replies in {dataset}/{model} "
                  "treated as not run (re-run inference to fill them)")
        has &= ~empty
    if m == "ce" and "ce_version" in raw.columns:
        from .infer import CE_VERSION
        old = has & raw["ce_version"].ne(CE_VERSION)
        if old.any():
            print(f"  note: {int(old.sum())} CE records in {dataset}/{model} use an older CE "
                  "instruction and are ignored (re-run inference to replace them)")
        has &= ~old
    elif m == "ce" and has.any():
        print(f"  note: CE records in {dataset}/{model} carry no instruction version "
              "and are ignored (re-run inference to replace them)")
        has &= False
    # Local-model records must say how they were produced; those made by older
    # code (no reasoning pre-fill, single precision) are superseded.
    if m in ("tlp", "ce"):
        from .infer import NO_THINK_PREFIX
        local = raw["api_params"].isna() if "api_params" in raw.columns else pd.Series(True, index=raw.index)
        want = next((p for k, p in NO_THINK_PREFIX.items() if k in model.lower()), "")
        col = f"{m}_think_prefix"
        ok = raw[col].eq(want) if col in raw.columns else pd.Series(False, index=raw.index)
        if m == "tlp":
            ok &= raw["tlp_precision"].eq("fp64") if "tlp_precision" in raw.columns else False
        stale = has & local & ~ok
        if stale.any():
            print(f"  note: {int(stale.sum())} {m} records in {dataset}/{model} were made by older "
                  "code and are ignored (re-run inference to replace them)")
        has &= ~(local & ~ok)
    raw = raw[has]
    if m == "ce" and "ce_raw_text" in raw.columns and not raw.empty:
        # Re-parse from the stored reply, so parser improvements reach records
        # already on disk. Empty API replies were removed above.
        from .infer import parse_ce, LETTERS
        letters = LETTERS[:K]
        parsed = [parse_ce(t, letters) if isinstance(t, str) else (None, None)
                  for t in raw["ce_raw_text"]]
        raw = raw.copy()
        raw["ce_answer"] = [a for a, _ in parsed]
        raw["ce_conf"] = [c for _, c in parsed]
        raw["ce_parse_ok"] = [a is not None for a, _ in parsed]
        raw["ce_correct"] = [int(a == g) if a else 0 for (a, _), g in zip(parsed, raw["answer_idx"])]
    if raw.empty:
        raise FileNotFoundError(f"no {m} records for {dataset}/{model}")
    raw = raw.drop_duplicates(["question_id", "variant", "condition"], keep="last")
    raw = raw.reset_index(drop=True)
    out = pd.DataFrame({
        "dataset": dataset,
        "model": model,
        "prompt": raw["condition"].astype(str),
        "conf_method": m,
        "question_id": raw["question_id"].astype(str),
        "group": raw["demographic"].astype(str).str.lower(),
        "conf_raw": pd.to_numeric(raw.get(f"{m}_conf"), errors="coerce"),
        "pred_idx": raw.get(f"{m}_answer").map(_letter_to_idx),
        "gold_idx": raw["answer_idx"].map(_letter_to_idx),
        "correct": pd.to_numeric(raw.get(f"{m}_correct"), errors="coerce"),
    })

    variant = raw["variant"].astype(str).str.lower().str.replace("adv_", "", regex=False)
    if spec["paired"]:
        out["variant"] = variant
        out["pair_id"] = out["question_id"]
    else:
        out["variant"] = None
        out["pair_id"] = None

    # parse_ok: tlp is always parsed from probabilities; sc/ce carry a flag.
    if m == "tlp":
        out["parse_ok"] = raw.get("tlp_option_probs").notna() & out["conf_raw"].notna()
    else:
        col = raw[f"{m}_parse_ok"] if f"{m}_parse_ok" in raw.columns else pd.Series(True, index=raw.index)
        out["parse_ok"] = col.eq(True).to_numpy()


    out["correct"] = out["correct"].fillna(0).astype(int)
    out["n_options"] = K
    # Optional fields: specialty label, and the TLP validity diagnostic (the
    # unnormalised probability the model put on any option letter).
    out["subject"] = raw["subject"] if "subject" in raw.columns else None
    lm = f"{m}_letter_mass"
    out["letter_mass"] = pd.to_numeric(raw[lm], errors="coerce") if lm in raw.columns else np.nan
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Sanity checks
# --------------------------------------------------------------------------- #

@dataclass
class SanityReport:
    """Result of validating one dataset x model x confidence-method slice.

    Attributes
    ----------
    dataset, model, conf_method : str
        What was validated.
    per_group : pandas.DataFrame
        Per condition and group: n, accuracy, mean confidence, parse rate.
    acc_gap : pandas.Series
        Reference-minus-demographic accuracy difference per condition.
    bad_pairs, gold_mismatches : pandas.DataFrame
        Pair-integrity and gold-agreement violations (empty for unpaired data).
    errors : list of str
        Conditions that invalidate the slice. Non-empty means do not analyse.
    warnings : list of str
        Findings that change interpretation but do not invalidate the data.
    degenerate : bool
        True if accuracy is not meaningfully above the chance rate ``1/K``.
    chance_p : float
        One-sided binomial p-value for accuracy exceeding chance.
    """

    dataset: str
    model: str
    conf_method: str
    per_group: pd.DataFrame
    acc_gap: pd.Series
    bad_pairs: pd.DataFrame
    gold_mismatches: pd.DataFrame
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    degenerate: bool = False
    chance_p: float = float("nan")

    @property
    def ok(self) -> bool:
        """Whether the slice is fit for subgroup analysis.

        Returns
        -------
        bool
            True when no hard errors were recorded.
        """
        return not self.errors


def validate(df: pd.DataFrame, parse_imbalance_tol: float = 0.05,
             strict: bool = True) -> SanityReport:
    """Validate a loaded slice and summarise its subgroup structure.

    Checks, in order: per-group parse-failure rate (group-imbalanced failure is
    an error, because it silently confounds every subgroup comparison); pair
    integrity; within-pair gold agreement; per-group accuracy, confidence and
    size; and whether accuracy sits at or below the chance rate.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of :func:`load_records`.
    parse_imbalance_tol : float
        Maximum tolerated difference in parse-failure rate between groups.
    strict : bool
        If True, raise on any hard error instead of only recording it.

    Returns
    -------
    SanityReport
        Structured findings.

    Raises
    ------
    ValueError
        If ``strict`` and a hard error was found.
    """
    dataset = df["dataset"].iloc[0]
    model = df["model"].iloc[0]
    conf_method = df["conf_method"].iloc[0]
    spec = cfg.DATASETS[dataset]
    ref, dem = spec["ref"], spec["dem"]
    K = int(df["n_options"].iloc[0])

    errors: list[str] = []
    warnings: list[str] = []

    per_group = (
        df.groupby(["prompt", "group"])
        .agg(n=("correct", "size"), accuracy=("correct", "mean"),
             mean_conf=("conf_raw", "mean"), parse_rate=("parse_ok", "mean"))
        .reset_index()
    )

    # -- 1. parse failures ------------------------------------------------- #
    pr = df.groupby("group")["parse_ok"].mean()
    overall_parse = float(df["parse_ok"].mean())
    if overall_parse < 0.5:
        errors.append(
            f"parse rate is {overall_parse:.2%} overall for conf_method={conf_method!r}: "
            "the confidence column is unusable for this model."
        )
    in_scope = [g for g in (ref, dem) if g in pr.index]
    if len(in_scope) == 2:
        imb = abs(pr[ref] - pr[dem])
        if imb > parse_imbalance_tol:
            # Not an error: an unparsed reply is referred, so the SAR already
            # accounts for it. It is reported so the gap can be read correctly.
            warnings.append(
                f"parse-failure rate differs by {imb:.2%} between groups "
                f"({ref}={1-pr[ref]:.2%}, {dem}={1-pr[dem]:.2%}); subgroup "
                "unparsed cases are referred, so part of any SAR gap is parse failure."
            )

    # -- 2/3. pair integrity and gold agreement ---------------------------- #
    if spec["paired"]:
        comp = (df.groupby(["prompt", "pair_id"])["variant"]
                .apply(lambda s: tuple(sorted(s))).rename("composition").reset_index())
        expected = tuple(sorted((ref, dem)))
        bad_pairs = comp[comp["composition"] != expected]
        if len(bad_pairs):
            warnings.append(f"{len(bad_pairs)} pair(s) lack exactly one {ref} and one {dem} "
                            "record (incomplete run?); they are left out of the analysis.")
        nun = df.groupby(["prompt", "pair_id"])["gold_idx"].nunique()
        gold_mismatches = nun[nun > 1].rename("n_distinct_gold").reset_index()
        if len(gold_mismatches):
            errors.append(f"{len(gold_mismatches)} pair(s) disagree on the gold answer.")
    else:
        bad_pairs = pd.DataFrame()
        gold_mismatches = pd.DataFrame()
        warnings.append(
            f"{dataset} is not a matched-pair design as run here: {spec['note']} "
            "Group contrasts are observational; counterfactual claims are not supported."
        )

    # -- 4/5. subgroup accuracy and degeneracy ----------------------------- #
    acc = df.groupby(["prompt", "group"])["correct"].mean().unstack()
    if ref in acc.columns and dem in acc.columns:
        acc_gap = (acc[ref] - acc[dem]).rename(f"acc_gap_{ref}_minus_{dem}")
    else:
        acc_gap = pd.Series(dtype=float)
        errors.append(f"groups {ref}/{dem} not both present; found {list(acc.columns)}.")

    chance = 1.0 / K
    n_tot = int(len(df))
    n_hit = int(df["correct"].sum())
    overall_acc = float(df["correct"].mean())
    # Degenerate means "not usefully above chance", not merely "below chance":
    # a model a hair above the chance rate on a few hundred items is
    # indistinguishable from guessing, and its calibration statistics describe
    # a confidently uninformative model rather than anything about fairness.
    p_above = float(stats.binomtest(n_hit, n_tot, chance, alternative="greater").pvalue)
    degenerate = (overall_acc <= chance) or (p_above > 0.05)
    if degenerate:
        warnings.append(
            f"accuracy {overall_acc:.3f} is not meaningfully above the {K}-option "
            f"chance rate {chance:.3f} (one-sided binomial p={p_above:.3f}): this "
            "model is confidently uninformative here and is excluded from the "
            "main analysis."
        )

    extra = set(df["group"].unique()) - {ref, dem}
    if extra:
        n_extra = int(df["group"].isin(extra).sum())
        warnings.append(
            f"{n_extra} record(s) in group(s) {sorted(extra)} are outside the "
            f"{ref}/{dem} contrast and are excluded from group metrics."
        )

    rep = SanityReport(dataset, model, conf_method, per_group, acc_gap,
                       bad_pairs, gold_mismatches, errors, warnings, degenerate,
                       p_above)
    if strict and errors:
        raise ValueError(f"[{dataset}/{model}/{conf_method}] " + " | ".join(errors))
    return rep


def restrict_to_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Drop records outside the dataset's two-group contrast.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of :func:`load_records`.

    Returns
    -------
    pandas.DataFrame
        Records whose ``group`` is the dataset's reference or demographic group.
    """
    spec = cfg.DATASETS[df["dataset"].iloc[0]]
    return df[df["group"].isin(spec["groups"])].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Construction audit
# --------------------------------------------------------------------------- #

