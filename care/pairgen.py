"""Rule-based construction of demographic minimal pairs.

The counterfactual claim this study makes is that changing a demographic
descriptor, and nothing else, can change the reliability of a model's
confidence. That claim is only as good as the pairs it is measured on. Pairs
produced by a generative rewriter are not minimal: the rewriter inserts
narrative context, and if it is given an attack objective it will differ
systematically between the two arms. This module therefore builds pairs by
deterministic string substitution only. No model is called, the rule table is
small enough to print in an appendix, and any reviewer can reproduce the output
from the source corpus.

Two pair types are produced.

Gender pairs swap the patient's gender markers. The source is restricted to
vignettes carrying male markers only, because the male-to-female direction is
unambiguous word for word (``his`` is always possessive, ``him`` always
objective) whereas the reverse is not (``her`` is both). Grammatical agreement
is preserved by the swap, so both arms read as naturally authored text.

Race pairs insert a race descriptor into the patient noun phrase in both arms,
so that the two variants are symmetric: each carries exactly one inserted word
and neither is the unmarked original.

Vignettes whose clinical content depends on sex are excluded from gender pairs,
and anything the rules cannot handle confidently is routed to manual review
rather than silently included.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass

import pandas as pd

from . import config as cfg

# --------------------------------------------------------------------------- #
# Rule tables (reported in the paper's appendix)
# --------------------------------------------------------------------------- #

# Patient gender markers, male -> female. Order matters: longer forms first so
# that "himself" is not consumed by the "him" rule.
GENDER_RULES: list[tuple[str, str]] = [
    ("himself", "herself"),
    ("gentleman", "lady"),
    ("gentlemen", "ladies"),
    ("man", "woman"),
    ("men", "women"),
    ("male", "female"),
    ("males", "females"),
    ("boy", "girl"),
    ("boys", "girls"),
    ("he", "she"),
    ("him", "her"),
    ("his", "her"),
    ("mr", "ms"),
]

# Kinship and third-party nouns are deliberately NOT swapped: they refer to
# people other than the patient, and swapping them would change the vignette's
# family history rather than the patient's demographic descriptor.
PROTECTED_NOUNS = {
    "mother", "father", "sister", "brother", "son", "daughter", "wife",
    "husband", "grandmother", "grandfather", "aunt", "uncle", "niece",
    "nephew", "widow", "widower", "boyfriend", "girlfriend", "partner",
}

# Clinical content that is sex-specific. A gender swap over such a vignette
# produces a clinically incoherent item, so these are excluded outright.
SEX_SPECIFIC_EXCLUDE = [
    r"pregnan", r"gravida", r"\bpara\s*\d", r"gestation", r"obstetric",
    r"menstrua", r"menopaus", r"amenorrh", r"dysmenorrh", r"menarche",
    r"uterus", r"uterine", r"endometri", r"ovary", r"ovarian", r"fallopian",
    r"vagina", r"vulva", r"labia", r"eclampsia", r"preeclampsia",
    r"mammogra", r"\blactating\b", r"\blactation\b", r"breastfeed",
    r"postpartum", r"intrauterine",
    r"hysterectom", r"oophorectom", r"salping", r"\bpap smear", r"\bIUD\b",
    r"prostat", r"testic", r"testes", r"scrotum", r"scrotal", r"penis",
    r"penile", r"erectile", r"\bPSA\b", r"vasectom", r"circumcis",
    r"epididym", r"varicocele", r"cryptorchid", r"spermato", r"\bsemen\b",
    r"gynecomastia", r"\borchitis\b", r"\borchiectom",
    r"\bmale\s+urethra\b", r"\bmale\s+genital", r"\bmale\s+breast\b",
]

# Ambiguous terms that may or may not be sex-specific in context. These are not
# excluded automatically; they are flagged so a human decides.
SEX_AMBIGUOUS_REVIEW = [
    r"cervic",     # cervix vs cervical spine
    r"breast",     # breast cancer occurs in men, but rarely in these vignettes
    r"pelvic",
    r"\bOB\b", r"\bGYN\b",
]

# Spans that contain a gender word but do not describe the patient's gender.
# These are masked before substitution and restored afterwards, because
# swapping inside them would change clinical content rather than a descriptor.
# MEN is the clearest case: in medical text it is Multiple Endocrine Neoplasia,
# and a naive swap turns "MEN-2A" into "WOMEN-2A".
PROTECTED_PATTERNS = [
    r"\bMEN[\s-]?(?:1|2|3|I{1,3}|2A|2B)\b",   # multiple endocrine neoplasia
    r"\bmale[-\s]pattern\b",                   # male-pattern baldness
    r"\bmale[-\s]specific\b",
    r"\bman[-\s]made\b",
    r"\bmale\s+partners?\b",                   # a third party, not the patient
    r"\bmale\s+relatives?\b",
    r"\bmale\s+siblings?\b",
    r"\bmale\s+carriers?\b",
    r"\bmale\s+sex\s+hormones?\b",
]

RACE_TERMS = ("White", "Black")

# Conditions with a strong ancestry association. In such a vignette a race
# descriptor can be clinically informative, so a difference between the two
# arms would not be evidence of bias. They are excluded from race pairs, the
# counterpart of excluding sex-specific content from gender pairs.
RACE_RELEVANT_EXCLUDE = [
    r"sickle", r"thalass", r"\bg6pd\b", r"glucose-6-phosphate", r"cystic fibrosis",
    r"tay-?sachs", r"gaucher", r"hemochromatosis", r"haemochromatosis",
    r"sarcoid", r"keloid", r"familial mediterranean", r"\bhbs\b", r"hemoglobin s\b",
]

# Patient noun phrase, used as the insertion point for a race descriptor.
AGE_NP = re.compile(
    r"\b(?P<age>\d+[-\s]?(?:year|month|week|day)[-\s]?old)\s+"
    r"(?P<noun>man|woman|male|female|boy|girl|gentleman|lady|patient|child|infant|newborn)\b",
    re.I,
)

MALE_MARK = re.compile(r"\b(man|men|male|males|boy|boys|gentleman|gentlemen|he|him|his|himself)\b", re.I)
FEMALE_MARK = re.compile(r"\b(woman|women|female|females|girl|girls|lady|ladies|she|her|hers|herself)\b", re.I)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _match_case(src: str, tgt: str) -> str:
    """Apply the capitalisation pattern of one token to another.

    Parameters
    ----------
    src : str
        Token whose capitalisation should be copied.
    tgt : str
        Replacement token in lower case.

    Returns
    -------
    str
        ``tgt`` capitalised like ``src``.
    """
    if src.isupper():
        return tgt.upper()
    if src[:1].isupper():
        return tgt.capitalize()
    return tgt


def swap_gender(text: str) -> tuple[str, list]:
    """Rewrite male patient markers as female ones.

    Substitution is word-for-word on token boundaries, so grammatical agreement
    is preserved and sentence length is essentially unchanged. Kinship nouns in
    :data:`PROTECTED_NOUNS` are left alone, because they denote people other
    than the patient; a possessive preceding one is still swapped, so ``his
    mother`` becomes ``her mother`` and the family history is untouched.

    Parameters
    ----------
    text : str
        Vignette text carrying male markers.

    Returns
    -------
    tuple of (str, list)
        The rewritten text, and the list of ``(source, replacement)`` token
        substitutions actually applied, for audit.
    """
    applied: list[tuple[str, str]] = []
    mapping = {a: b for a, b in GENDER_RULES}

    # Mask protected spans so the substitution cannot reach inside them.
    masked: list[str] = []

    def _mask(m):
        masked.append(m.group(0))
        return f"\x00{len(masked) - 1}\x00"

    for pat in PROTECTED_PATTERNS:
        flags = 0 if pat.startswith(r"\bMEN") else re.I
        text = re.sub(pat, _mask, text, flags=flags)

    pattern = re.compile(
        r"\b(" + "|".join(sorted(mapping, key=len, reverse=True)) + r")\b(\.?)",
        re.I,
    )

    def sub(m):
        tok, dot = m.group(1), m.group(2)
        rep = _match_case(tok, mapping[tok.lower()])
        applied.append((tok, rep))
        return rep + dot

    out = pattern.sub(sub, text)
    out = re.sub(r"\x00(\d+)\x00", lambda m: masked[int(m.group(1))], out)
    return out, applied


def insert_race(text: str, race: str) -> tuple[str, bool]:
    """Insert a race descriptor into the patient noun phrase.

    The descriptor is placed immediately before the head noun of the first
    age-qualified patient noun phrase, which is the position it occupies in
    naturally written vignettes: ``A 67-year-old man`` becomes ``A 67-year-old
    Black man``. Both arms of a race pair receive an inserted word, so neither
    is the unmarked original.

    Parameters
    ----------
    text : str
        Vignette text.
    race : str
        Descriptor to insert, e.g. ``"Black"`` or ``"White"``.

    Returns
    -------
    tuple of (str, bool)
        The rewritten text and whether an insertion point was found. When no
        patient noun phrase is found the text is returned unchanged.
    """
    m = AGE_NP.search(text)
    if not m:
        return text, False
    i = m.start("noun")
    return text[:i] + race + " " + text[i:], True


def sex_specific_reason(text: str) -> str | None:
    """Return the reason a vignette is unsuitable for a gender swap, if any.

    Parameters
    ----------
    text : str
        Vignette text.

    Returns
    -------
    str or None
        The matched sex-specific term, or None if none was found.
    """
    low = text.lower()
    for pat in SEX_SPECIFIC_EXCLUDE:
        m = re.search(pat, low)
        if m:
            return m.group(0)
    return None


def needs_review_reason(text: str) -> str | None:
    """Return the reason a vignette should be reviewed by a human, if any.

    Parameters
    ----------
    text : str
        Vignette text.

    Returns
    -------
    str or None
        The matched ambiguous term, or None.
    """
    low = text.lower()
    for pat in SEX_AMBIGUOUS_REVIEW:
        m = re.search(pat, low)
        if m:
            return m.group(0)
    return None


def protected_noun_touched(applied: list, text_after: str) -> bool:
    """Check whether a kinship noun was altered by the swap.

    Parameters
    ----------
    applied : list
        Substitutions returned by :func:`swap_gender`.
    text_after : str
        The rewritten text.

    Returns
    -------
    bool
        True if any protected kinship noun appears to have been changed.
    """
    return any(a.lower() in PROTECTED_NOUNS for a, _ in applied)


# --------------------------------------------------------------------------- #
# Pair construction
# --------------------------------------------------------------------------- #

@dataclass
class PairSet:
    """A built set of minimal pairs plus the audit trail of its construction.

    Attributes
    ----------
    pairs : pandas.DataFrame
        One row per variant, two rows per pair.
    excluded : pandas.DataFrame
        Source items dropped, with the reason.
    review : pandas.DataFrame
        Items included but flagged for manual inspection.
    """

    pairs: pd.DataFrame
    excluded: pd.DataFrame
    review: pd.DataFrame


MEDQA_OPT = re.compile(r"\s([A-E]):")


def _norm(qid, stem, options, answer_idx, dataset, extra=None) -> dict:
    """Normalise one source item into the shape the pair builders expect.

    Parameters
    ----------
    qid : str
        Item identifier.
    stem : str
        Vignette text with the answer options removed.
    options : dict
        Mapping of option letter to option text.
    answer_idx : str
        Reference option letter.
    dataset : str
        Source dataset label.
    extra : dict, optional
        Additional fields to carry through (e.g. MedMCQA subject).

    Returns
    -------
    dict
        Normalised record.
    """
    r = {"question_id": str(qid), "stem": stem.strip(), "options": options,
         "answer_idx": answer_idx, "dataset": dataset}
    if extra:
        r.update(extra)
    return r


def load_medqa(splits=("test", "dev", "train"), root=None) -> list:
    """Load MedQA, splitting the answer options out of the question text.

    MedQA as distributed here stores the options inline in the question string
    as ``... A:opt1, B:opt2, ...``. The stem is everything before the first
    option marker, which is what a demographic substitution must act on.

    Parameters
    ----------
    splits : sequence of str
        Which splits to load.
    root : str, optional
        Directory containing ``test.json`` etc.

    Returns
    -------
    list of dict
        Normalised records with 5 options each.
    """
    root = root or os.path.join(cfg.SOURCE_DIR, "MedQA")
    out = []
    for sp in splits:
        path = os.path.join(root, f"{sp}.json")
        if not os.path.exists(path):
            continue
        d = json.load(open(path, encoding="utf-8"))
        rows = d["data"] if isinstance(d, dict) and "data" in d else d
        for r in rows:
            parts = MEDQA_OPT.split(r["question"])
            # parts = [stem, 'A', textA, 'B', textB, ...]
            if len(parts) < 3:
                continue
            stem = parts[0]
            opts = {}
            for k in range(1, len(parts) - 1, 2):
                opts[parts[k]] = parts[k + 1].rstrip(", ").strip()
            if len(opts) < 4 or r["answer"] not in opts:
                continue
            out.append(_norm(r["id"], stem, opts, r["answer"], "medqa",
                             {"split": sp}))
    return out


EXP_ANS = re.compile(r"^\W*(?:ans(?:wer)?|correct\s+answer)\s*(?:is|:|-|\.)?\s*\(?\s*([a-d])\b",
                     re.I)


def _medmcqa_base(rows) -> int:
    """Index base of MedMCQA's ``cop`` field in these files (0 or 1).

    Releases differ: the Hugging Face version numbers options 0-3, the original
    JSON 1-4. Reading one as the other shifts every gold label by one and drops
    every item answered D, so the base is inferred from the data rather than
    assumed.

    Parameters
    ----------
    rows : list of dict
        Raw MedMCQA records.

    Returns
    -------
    int
        0 or 1.

    Raises
    ------
    ValueError
        If the observed values fit neither convention.
    """
    vals = {int(r["cop"]) for r in rows if r.get("cop") is not None}
    if vals <= {0, 1, 2, 3} and 0 in vals:
        return 0
    if vals <= {1, 2, 3, 4} and 4 in vals:
        return 1
    raise ValueError(f"cannot infer MedMCQA option-index base from values {sorted(vals)}")


def check_medmcqa_labels(records, min_agreement: float = 0.9) -> dict:
    """Cross-check derived gold letters against letters stated in the explanation.

    Many MedMCQA explanations begin "Ans. a" or "Answer- A". Where they do, the
    stated letter must agree with the derived gold label; systematic
    disagreement means the labels are misaligned.

    Parameters
    ----------
    records : list of dict
        Normalised records carrying ``answer_idx`` and ``exp``.
    min_agreement : float
        Smallest acceptable agreement rate.

    Returns
    -------
    dict
        ``n_checked`` and ``agreement``.

    Raises
    ------
    ValueError
        If agreement falls below ``min_agreement``.
    """
    hits = tot = 0
    for r in records:
        m = EXP_ANS.search(r.get("exp") or "")
        if m:
            tot += 1
            hits += m.group(1).upper() == r["answer_idx"]
    agree = hits / tot if tot else float("nan")
    if tot >= 50 and agree < min_agreement:
        raise ValueError(f"MedMCQA gold labels disagree with explanations: "
                         f"{agree:.1%} agreement on {tot} items -- index misaligned?")
    return {"n_checked": tot, "agreement": agree}


def load_medmcqa(splits=("train", "dev"), root=None) -> list:
    """Load MedMCQA, keeping only labelled items, with a label-alignment check.

    MedMCQA is predominantly short factual questions rather than clinical
    vignettes, so only a small fraction carries a patient description; the pair
    builders filter for that automatically. The official test split is
    unlabelled and is skipped by default.

    Parameters
    ----------
    splits : sequence of str
        Which splits to load.
    root : str, optional
        Directory containing the split files.

    Returns
    -------
    list of dict
        Normalised records with 4 options each, carrying ``subject``.
    """
    root = root or os.path.join(cfg.SOURCE_DIR, "medmcqa")
    letters = ["A", "B", "C", "D"]
    out = []
    for sp in splits:
        path = os.path.join(root, f"{sp}.json")
        if not os.path.exists(path):
            continue
        raw = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        base = _medmcqa_base(raw)
        for r in raw:
            cop = r.get("cop")
            if cop is None:
                continue
            k = int(cop) - base
            if not 0 <= k < 4:
                continue
            opts = {L: str(r[f]) for L, f in zip(letters, ("opa", "opb", "opc", "opd"))}
            stem = re.sub(r"^[\W_]+", "", str(r["question"]))   # stray leading ". "
            rec = _norm(r["id"], stem, opts, letters[k], "medmcqa",
                        {"split": sp, "subject": r.get("subject_name")})
            rec["exp"] = r.get("exp")
            out.append(rec)
    check_medmcqa_labels(out)
    return out


LOADERS = {"medqa": load_medqa, "medmcqa": load_medmcqa}


def load_source(dataset: str = "medqa", **kw) -> list:
    """Load a source corpus by name.

    Parameters
    ----------
    dataset : {"medqa", "medmcqa"}
        Which corpus to load.
    **kw
        Passed to the loader.

    Returns
    -------
    list of dict
        Normalised records.
    """
    if dataset not in LOADERS:
        raise KeyError(f"unknown dataset {dataset!r}; known: {sorted(LOADERS)}")
    rows = LOADERS[dataset](**kw)
    kept = [r for r in rows if not_answerable_as_text(r) is None]
    why = Counter(not_answerable_as_text(r) for r in rows) - Counter({None: len(kept)})
    if len(kept) < len(rows):
        print(f"  {dataset}: dropped {len(rows) - len(kept)} of {len(rows)} items not "
              f"answerable from text ({dict(why)})")
    return kept


# Items whose answer depends on an image the model never sees, or whose options
# cannot be told apart, have no well-defined correct answer in a text-only
# setting; both arms of a pair would be guesses, adding noise to every metric.
IMAGE_REF = re.compile(
    r"<img|image_question|\bfigure\b"
    r"|\b(image|picture|photograph|photo|diagram|illustration|exhibit)s?\s+(below|above|shown|provided|attached)"
    r"|\b(shown|seen|depicted|displayed|illustrated|provided)\s+(in|on|by)\s+(the\s+)?"
    r"(accompanying\s+|following\s+|adjacent\s+)?(image|picture|photograph|photo|figure|diagram|exhibit|illustration)"
    r"|\bthe following\s+(image|picture|photograph|photo|figure|diagram|x-?ray|radiograph|film|ecg|ekg|tracing|slide)",
    re.I)


def not_answerable_as_text(rec: dict):
    """Reason an item cannot be answered from its text alone, or None.

    Parameters
    ----------
    rec : dict
        Normalised record with ``stem`` (or ``question``) and ``options``.

    Returns
    -------
    str or None
        ``"image"``, ``"duplicate_options"`` or ``"empty_option"``; None if usable.
    """
    opts = [str(v).strip() for v in rec["options"].values()]
    if IMAGE_REF.search(rec.get("stem", rec.get("question", "")) + " " + " ".join(opts)):
        return "image"
    if any(not o for o in opts):
        return "empty_option"
    if len({o.lower() for o in opts}) < len(opts):
        return "duplicate_options"
    return None


def build_gender_pairs(rows=None) -> PairSet:
    """Build male/female minimal pairs by deterministic gender swap.

    Parameters
    ----------
    rows : list of dict, optional
        Source records; loaded from the FairMedQA source if omitted.

    Returns
    -------
    PairSet
        Pairs, exclusions, and review flags.
    """
    rows = rows if rows is not None else load_source()
    out, exc, rev = [], [], []

    for r in rows:
        qid = str(r["question_id"])
        text = r["stem"]

        if FEMALE_MARK.search(text):
            exc.append({"question_id": qid, "reason": "not male-only (ambiguous swap direction)"})
            continue
        if not MALE_MARK.search(text):
            exc.append({"question_id": qid, "reason": "no gender marker"})
            continue
        if not AGE_NP.search(text):
            exc.append({"question_id": qid, "reason": "not a patient vignette"})
            continue
        why = sex_specific_reason(text)
        if why:
            exc.append({"question_id": qid, "reason": f"sex-specific content: {why}"})
            continue

        swapped, applied = swap_gender(text)
        if not applied:
            exc.append({"question_id": qid, "reason": "no substitution applied"})
            continue
        if protected_noun_touched(applied, swapped):
            exc.append({"question_id": qid, "reason": "kinship noun would be altered"})
            continue
        if FEMALE_MARK.search(text) or MALE_MARK.search(swapped):
            exc.append({"question_id": qid, "reason": "residual marker after swap"})
            continue

        flag = needs_review_reason(text)
        if flag:
            rev.append({"question_id": qid, "flag": flag, "text": text[:300]})

        for variant, t in (("male", text), ("female", swapped)):
            out.append({
                "pair_id": qid, "variant": variant, "group": variant,
                "axis": "gender", "question": t, "options": r["options"],
                "answer_idx": r["answer_idx"], "dataset": r["dataset"],
                "subject": r.get("subject"),
                "n_substitutions": len(applied),
                "flagged_for_review": bool(flag),
            })

    return PairSet(pd.DataFrame(out), pd.DataFrame(exc), pd.DataFrame(rev))


def build_race_pairs(rows=None, races=RACE_TERMS) -> PairSet:
    """Build race minimal pairs by symmetric descriptor insertion.

    Both arms receive one inserted descriptor at the same position, so neither
    arm is the unmarked original and the two differ by exactly one word.

    Parameters
    ----------
    rows : list of dict, optional
        Source records; loaded from the FairMedQA source if omitted.
    races : sequence of str
        The two descriptors to insert.

    Returns
    -------
    PairSet
        Pairs, exclusions, and review flags.
    """
    rows = rows if rows is not None else load_source()
    out, exc = [], []
    lower = {r.lower() for r in races}

    for r in rows:
        qid = str(r["question_id"])
        text = r["stem"]

        if any(w in text.lower() for w in lower | {"caucasian", "african american", "hispanic", "asian"}):
            exc.append({"question_id": qid, "reason": "race already stated in vignette"})
            continue
        hit = next((re.search(p, text.lower()).group(0) for p in RACE_RELEVANT_EXCLUDE
                    if re.search(p, text.lower())), None)
        if hit:
            exc.append({"question_id": qid, "reason": f"race-relevant condition: {hit}"})
            continue
        variants = {}
        ok = True
        for race in races:
            t, found = insert_race(text, race)
            if not found:
                ok = False
                break
            variants[race.lower()] = t
        if not ok:
            exc.append({"question_id": qid, "reason": "no patient noun phrase found"})
            continue

        for race in races:
            out.append({
                "pair_id": qid, "variant": race.lower(), "group": race.lower(),
                "axis": "race", "question": variants[race.lower()],
                "options": r["options"], "answer_idx": r["answer_idx"],
                "dataset": r["dataset"], "subject": r.get("subject"),
                "n_substitutions": 1, "flagged_for_review": False,
            })

    return PairSet(pd.DataFrame(out), pd.DataFrame(exc), pd.DataFrame())


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #

def write_inference_jsonl(ps: PairSet, path: str) -> str:
    """Write a pair set as JSONL ready for the inference runner.

    Parameters
    ----------
    ps : PairSet
        Built pairs.
    path : str
        Destination path.

    Returns
    -------
    str
        The path written.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for _, r in ps.pairs.iterrows():
            fh.write(json.dumps({
                "question_id": r["pair_id"], "pair_id": r["pair_id"],
                "variant": r["variant"], "demographic": r["group"],
                "axis": r["axis"], "question": r["question"],
                "options": r["options"], "answer_idx": r["answer_idx"],
                # specialty label (MedMCQA only), for comparison with studies
                # that stratify calibration by specialty
                "subject": (r.get("subject") if isinstance(r.get("subject"), str)
                            else None),
            }, ensure_ascii=False) + "\n")
    return path


# --------------------------------------------------------------------------- #
# Sampling and CLI
# --------------------------------------------------------------------------- #

def sample_pairs(ps: PairSet, n: int, seed: int = cfg.SEED,
                 stratify: str | None = None) -> PairSet:
    """Take a random subset of whole pairs, keeping both arms together.

    Parameters
    ----------
    ps : PairSet
        Built pairs.
    n : int
        Number of pairs to keep; all are kept if fewer exist.
    seed : int
        Sampling seed.
    stratify : str, optional
        Column to stratify on, e.g. ``"subject"`` for MedMCQA, so that the
        sample spans specialties in proportion to the source.

    Returns
    -------
    PairSet
        The sampled pairs, with the original exclusion and review tables.
    """
    import numpy as np
    df = ps.pairs
    ids = df.drop_duplicates("pair_id")[["pair_id"] + ([stratify] if stratify else [])]
    if len(ids) <= n:
        keep = ids["pair_id"]
    elif stratify and stratify in ids.columns and ids[stratify].notna().any():
        frac = n / len(ids)
        keep = (ids.groupby(stratify, group_keys=False)
                   .apply(lambda g: g.sample(max(1, round(len(g) * frac)),
                                             random_state=seed))["pair_id"])
        if len(keep) > n:
            keep = keep.sample(n, random_state=seed)
        elif len(keep) < n:                    # rounding left us short: top up
            rest = ids.loc[~ids["pair_id"].isin(keep), "pair_id"]
            keep = pd.concat([keep, rest.sample(n - len(keep), random_state=seed)])
    else:
        keep = ids["pair_id"].sample(n, random_state=seed)
    return PairSet(df[df["pair_id"].isin(set(keep))].copy(), ps.excluded, ps.review)


def main(argv=None) -> int:
    """Build the matched pairs used in the paper and write them as JSONL.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments.

    Returns
    -------
    int
        Process exit code.
    """
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=["medqa", "medmcqa"],
                    choices=["medqa", "medmcqa"])
    ap.add_argument("--axes", nargs="*", default=["race", "gender"],
                    choices=["race", "gender"])
    ap.add_argument("--n", type=int, default=1000, help="pairs per dataset and axis")
    ap.add_argument("--outdir", default=cfg.PAIRS_DIR)
    args = ap.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)
    print(f"{'dataset':10s} {'axis':7s} {'built':>7s} {'kept':>6s}")
    for ds in args.datasets:
        rows = load_source(ds)
        for axis in args.axes:
            ps = build_race_pairs(rows) if axis == "race" else build_gender_pairs(rows)
            built = len(ps.pairs) // 2
            ps = sample_pairs(ps, args.n, stratify="subject" if ds == "medmcqa" else None)
            write_inference_jsonl(ps, os.path.join(args.outdir, f"pairs_{ds}_{axis}.jsonl"))
            print(f"{ds:10s} {axis:7s} {built:7d} {len(ps.pairs) // 2:6d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
