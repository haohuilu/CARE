"""Control pair sets for the revision.

The submitted study compares White with Black and male with female versions of
one vignette. Three things other than the descriptor itself could produce a
difference between the two versions, and each control set below removes one.

``pairs_<src>_placebo``
    The race pairs with a clinically irrelevant descriptor in place of race:
    ``right-handed`` versus ``left-handed``, inserted at the same position in
    the same vignettes. Whatever difference these pairs show is the size of
    difference that one changed word produces on its own.

``pairs_<src>_gender_edited``
    The sex pairs with a neutral word inserted into BOTH versions. In the
    submitted sex pairs the male version is the verbatim benchmark text and
    only the female version is edited, so a model that has seen the benchmark
    could favour the male version for that reason alone. Here neither version
    is verbatim.

``pairs_<src>_gender_reverse``
    Sex pairs built in the opposite direction, from vignettes that are female
    in the benchmark. The female version is now the verbatim text and the male
    version is the edited one. If the male version is still favoured, exposure
    to the original wording cannot be the explanation.

``pairs_<src>_<axis>_full``
    Every pair the rules can build, rather than a sample of 1,000.

The module also writes ``pair_flags.csv``: per pair, the source split and a set
of screens used for sensitivity analyses (other people of the patient's sex in
the vignette, spouse nouns, and conditions for which race can bear on
management). The four pair files of the submitted paper are not touched; their
hashes are checked on every run.
"""

from __future__ import annotations

import hashlib
import json
import os
import re

import pandas as pd

from . import config as cfg
from .pairgen import (AGE_NP, FEMALE_MARK, MALE_MARK, PROTECTED_NOUNS, RACE_RELEVANT_EXCLUDE,
                      PairSet, _match_case, build_gender_pairs, build_race_pairs, insert_race,
                      load_source, sample_pairs, sex_specific_reason)

# First 16 hex characters of the SHA-256 of the pair files used in the paper.
PAPER_HASHES = {
    "pairs_medqa_race": "d32c84414db706f4",
    "pairs_medqa_gender": "084ad9446b00e11d",
    "pairs_medmcqa_race": "b2058037d8772b85",
    "pairs_medmcqa_gender": "e12d901caa9ff4a7",
}

# --------------------------------------------------------------------------- #
# Placebo descriptor
# --------------------------------------------------------------------------- #

# Vignettes in which handedness is stated or can matter clinically (language
# lateralisation, dominant-hand injury). The placebo must be irrelevant.
HANDEDNESS_RELEVANT = [
    r"hand(ed|edness)\b", r"aphasi", r"hemispher", r"\bdominant\s+(hand|arm|side|limb)",
    r"\bnon-?dominant\b", r"lateralis|lateraliz", r"\bwada\b", r"apraxi", r"neglect",
]


def handedness_reason(text: str):
    """The matched term if handedness is stated or clinically relevant, else None."""
    low = text.lower()
    for pat in HANDEDNESS_RELEVANT:
        m = re.search(pat, low)
        if m:
            return m.group(0)
    return None


def build_placebo_pairs(rows, terms=cfg.PLACEBO_TERMS) -> PairSet:
    """Right-/left-handed pairs at the race insertion point.

    The vignettes are those that qualify for race pairs (no race stated, no
    race-associated condition, a patient noun phrase to insert into), minus
    those in which handedness is relevant. Both versions receive one inserted
    word, exactly as in the race pairs.

    Parameters
    ----------
    rows : list of dict
        Normalised source records.
    terms : tuple of str
        Reference and comparison descriptor.

    Returns
    -------
    PairSet
    """
    base = build_race_pairs(rows)
    stems = {str(r["question_id"]): r for r in rows}
    out, exc = [], base.excluded.to_dict("records")
    for qid in dict.fromkeys(base.pairs["pair_id"]):
        r = stems[qid]
        why = handedness_reason(r["stem"])
        if why:
            exc.append({"question_id": qid, "reason": f"handedness relevant: {why}"})
            continue
        for t in terms:
            text, found = insert_race(r["stem"], t)
            assert found
            out.append({"pair_id": qid, "variant": t, "group": t, "axis": "placebo",
                        "question": text, "options": r["options"],
                        "answer_idx": r["answer_idx"], "dataset": r["dataset"],
                        "subject": r.get("subject"), "n_substitutions": 1,
                        "flagged_for_review": False})
    return PairSet(pd.DataFrame(out), pd.DataFrame(exc), pd.DataFrame())


def build_gender_edited_pairs(rows, term: str = cfg.PLACEBO_TERMS[0]) -> PairSet:
    """Sex pairs with one neutral word inserted into both versions.

    Parameters
    ----------
    rows : list of dict
        Normalised source records.
    term : str
        Neutral descriptor inserted before the patient noun in both versions.

    Returns
    -------
    PairSet
    """
    base = build_gender_pairs(rows)
    out, exc = [], base.excluded.to_dict("records")
    drop = set()
    for _, r in base.pairs.iterrows():
        why = handedness_reason(r["question"])
        text, found = insert_race(r["question"], term)
        if why or not found:
            drop.add(r["pair_id"])
            continue
        out.append({**r.to_dict(), "question": text, "axis": "gender_edited",
                    "n_substitutions": int(r["n_substitutions"]) + 1})
    for qid in sorted(drop):
        exc.append({"question_id": qid, "reason": "handedness relevant or no insertion point"})
    pairs = pd.DataFrame(out)
    pairs = pairs[~pairs["pair_id"].isin(drop)].reset_index(drop=True)
    return PairSet(pairs, pd.DataFrame(exc), base.review)


# --------------------------------------------------------------------------- #
# Reverse-direction sex pairs (female original -> male)
# --------------------------------------------------------------------------- #

REVERSE_RULES = {
    "herself": "himself", "lady": "gentleman", "ladies": "gentlemen",
    "woman": "man", "women": "men", "female": "male", "females": "males",
    "girl": "boy", "girls": "boys", "she": "he", "hers": "his",
    "ms": "mr", "mrs": "mr",
}
# "her" is possessive ("her mother" -> "his mother") or objective ("examined
# her" -> "examined him"). It is objective when followed by punctuation or one
# of these words, which cannot begin a noun phrase that "her" possesses.
HER_OBJECT_NEXT = {
    "to", "for", "with", "and", "or", "but", "in", "on", "at", "by", "from", "that", "when",
    "because", "if", "as", "was", "is", "has", "had", "about", "after", "before", "while",
    "until", "since", "into", "of", "during", "the", "a", "an", "this", "these", "those",
    "not", "which", "who", "would", "will", "can", "could", "should", "may", "might", "did",
    "does", "do", "were", "are", "be", "been", "than", "so", "there", "here", "again",
    "today", "yesterday", "now", "immediately", "up", "off", "through", "without", "against",
}
# After these verbs "her" + word can be either reading ("gave her medication",
# "saw her walking", "made her feel"); such vignettes are excluded, not guessed.
HER_AMBIGUOUS_PREV = {
    "give", "gives", "gave", "given", "giving", "tell", "tells", "told", "telling",
    "ask", "asks", "asked", "show", "shows", "showed", "shown", "send", "sends", "sent",
    "offer", "offers", "offered", "bring", "brings", "brought", "prescribe", "prescribes",
    "prescribed", "teach", "taught", "hand", "handed", "buy", "bought", "make", "makes",
    "made", "let", "lets", "help", "helps", "helped", "have", "see", "saw", "seen",
    "notice", "noticed", "find", "found", "hear", "heard", "watch", "watched", "observe",
    "observed", "keep", "kept", "leave", "left", "call", "called", "feed", "fed",
    "cause", "caused", "causes", "causing", "finding", "seeing", "noticing", "hearing",
    "watching", "observing", "making", "letting", "helping", "having", "keeping",
    "leaving", "calling", "showing", "telling", "asking", "sending", "bringing",
}
FEMALE_PROTECTED_PATTERNS = [
    r"\bfemale[-\s]pattern\b", r"\bfemale\s+partners?\b", r"\bfemale\s+relatives?\b",
    r"\bfemale\s+siblings?\b", r"\bfemale\s+carriers?\b", r"\bfemale\s+sex\s+hormones?\b",
]
# Female-coded content beyond the paper's exclusion list. The paper's list was
# written for male-coded vignettes, where these terms rarely occur.
REVERSE_EXTRA_EXCLUDE = [
    r"contracepti", r"\bocps?\b", r"hormone replacement", r"\blmp\b", r"last (menstrual )?period",
    r"gynecolog|gynaecolog", r"\bpcos\b", r"polycystic ov", r"nullipar", r"multipar", r"primipar",
    r"miscarriage", r"abortion", r"cesarean|caesarean", r"c-section", r"hirsut", r"\bturner",
    r"pessary", r"galactorrh", r"breast", r"cervix", r"cervical (cancer|smear|os|motion|dilat)",
    r"pelvic", r"bimanual", r"estrogen|oestrogen", r"progest", r"tamoxifen", r"\bhpv\b",
    r"\bg\d+\s*p\d+", r"deliver(y|ed)\b", r"\blabor\b|\blabour\b", r"trimester", r"\bfetus|\bfetal|foetal",
    r"\bob/?gyn\b", r"osteoporo", r"\bbrca",
]
SPOUSE_NOUNS = r"\b(husband|wife|boyfriend|girlfriend|fianc[eé]e?)s?\b"
FEMALE_KIN = r"\b(mother|sister|daughter|grandmother|aunt|niece|stepmother|mom)s?\b"
MALE_KIN = r"\b(father|brother|son|grandfather|uncle|nephew|stepfather|dad)s?\b"


def classify_her(text: str) -> list:
    """Reading of every ``her`` in a text.

    Parameters
    ----------
    text : str
        Vignette text.

    Returns
    -------
    list of tuple
        ``(start, end, reading)`` with reading ``"poss"``, ``"obj"`` or
        ``"ambiguous"``, in order of occurrence.
    """
    out = []
    for m in re.finditer(r"\bher\b", text, re.I):
        after = text[m.end():]
        nxt = re.match(r"\s+([A-Za-z][A-Za-z'-]*)", after)
        prev = re.search(r"([A-Za-z]+)\W*$", text[:m.start()])
        prev_w = prev.group(1).lower() if prev else ""
        if not nxt:                                   # punctuation, digit or end
            reading = "ambiguous" if re.match(r"\s+\d", after) else "obj"
        elif nxt.group(1).lower() in HER_OBJECT_NEXT:
            reading = "obj"
        elif prev_w in HER_AMBIGUOUS_PREV:
            reading = "ambiguous"
        else:
            reading = "poss"
        out.append((m.start(), m.end(), reading))
    return out


def swap_gender_reverse(text: str):
    """Rewrite female patient markers as male ones.

    Parameters
    ----------
    text : str
        Vignette text carrying female markers only.

    Returns
    -------
    tuple of (str or None, list)
        The rewritten text and the substitutions applied. The text is None
        when a ``her`` cannot be read confidently.
    """
    masked: list[str] = []

    def _mask(m):
        masked.append(m.group(0))
        return f"\x00{len(masked) - 1}\x00"

    for pat in FEMALE_PROTECTED_PATTERNS:
        text = re.sub(pat, _mask, text, flags=re.I)

    applied: list[tuple[str, str]] = []
    pieces, last = [], 0
    for s, e, reading in classify_her(text):
        if reading == "ambiguous":
            return None, [("her", "?")]
        rep = _match_case(text[s:e], "his" if reading == "poss" else "him")
        applied.append((text[s:e], rep))
        pieces += [text[last:s], rep]
        last = e
    text = "".join(pieces) + text[last:]

    pattern = re.compile(r"\b(" + "|".join(sorted(REVERSE_RULES, key=len, reverse=True))
                         + r")\b", re.I)

    def sub(m):
        rep = _match_case(m.group(1), REVERSE_RULES[m.group(1).lower()])
        applied.append((m.group(1), rep))
        return rep

    out = pattern.sub(sub, text)
    out = re.sub(r"\x00(\d+)\x00", lambda m: masked[int(m.group(1))], out)
    return out, applied


def build_gender_reverse_pairs(rows) -> PairSet:
    """Female-original sex pairs: the male version is the substituted one.

    Stricter than the submitted male-original pairs in two respects, because
    the purpose is a clean test of direction: vignettes that mention a spouse
    or partner, or a female relative, are left out. A spouse noun would turn
    ``her husband`` into ``his husband``; a female relative makes it impossible
    to tell by rule whether a later ``she`` is the patient.

    Parameters
    ----------
    rows : list of dict
        Normalised source records.

    Returns
    -------
    PairSet
    """
    out, exc = [], []
    for r in rows:
        qid, text = str(r["question_id"]), r["stem"]
        low = text.lower()
        if MALE_MARK.search(text):
            exc.append({"question_id": qid, "reason": "not female-only"})
            continue
        if not FEMALE_MARK.search(text):
            exc.append({"question_id": qid, "reason": "no gender marker"})
            continue
        if not AGE_NP.search(text):
            exc.append({"question_id": qid, "reason": "not a patient vignette"})
            continue
        why = sex_specific_reason(text) or next(
            (re.search(p, low).group(0) for p in REVERSE_EXTRA_EXCLUDE if re.search(p, low)), None)
        if why:
            exc.append({"question_id": qid, "reason": f"sex-specific content: {why}"})
            continue
        if re.search(SPOUSE_NOUNS, low):
            exc.append({"question_id": qid, "reason": "spouse or partner noun"})
            continue
        if re.search(FEMALE_KIN, low):
            exc.append({"question_id": qid, "reason": "female relative (third-party pronouns)"})
            continue
        swapped, applied = swap_gender_reverse(text)
        if swapped is None:
            exc.append({"question_id": qid, "reason": "ambiguous 'her'"})
            continue
        if not applied:
            exc.append({"question_id": qid, "reason": "no substitution applied"})
            continue
        if any(a.lower() in PROTECTED_NOUNS for a, _ in applied):
            exc.append({"question_id": qid, "reason": "kinship noun would be altered"})
            continue
        if FEMALE_MARK.search(swapped):
            exc.append({"question_id": qid, "reason": "residual marker after swap"})
            continue
        for variant, t in (("male", swapped), ("female", text)):
            out.append({"pair_id": qid, "variant": variant, "group": variant,
                        "axis": "gender_reverse", "question": t, "options": r["options"],
                        "answer_idx": r["answer_idx"], "dataset": r["dataset"],
                        "subject": r.get("subject"), "n_substitutions": len(applied),
                        "flagged_for_review": False})
    return PairSet(pd.DataFrame(out), pd.DataFrame(exc), pd.DataFrame())


# --------------------------------------------------------------------------- #
# Screens for sensitivity analyses
# --------------------------------------------------------------------------- #

# Conditions for which race can bear on the keyed answer, beyond the paper's
# exclusion list (which covers inherited conditions with strong ancestry
# association). Tier A: guidance or equations that have referred to race
# explicitly. Tier B: conditions with a well-known difference in incidence.
# These lists were not drawn up by a clinician; they are a screen for a
# sensitivity analysis, to be extended by clinical reviewers.
RACE_TIER_A = [
    r"hypertens", r"\bace inhibitor|\bacei\b|angiotensin", r"thiazide",
    r"calcium channel", r"amlodipine|lisinopril|enalapril|captopril|losartan|hydrochlorothiazide"
    r"|chlorthalidone", r"hydralazine", r"isosorbide dinitrate", r"heart failure",
    r"\begfr\b|glomerular filtration", r"creatinine clearance", r"spiromet|pulmonary function",
    r"angioedema", r"allopurinol", r"carbamazepine", r"\bhla-b",
]
RACE_TIER_B = [
    r"lupus|\bsle\b", r"glaucoma", r"myeloma", r"prostat", r"melanoma", r"basal cell",
    r"squamous cell carcinoma of the skin", r"vitamin d\b|rickets", r"lactose|lactase",
    r"fibroid|leiomyoma", r"\bfsgs\b|focal segmental", r"\bapol1\b", r"end-stage renal|esrd",
    r"chronic kidney disease", r"osteoporo", r"vitiligo", r"\bhiv\b", r"diabet", r"stroke",
    r"preterm|low birth weight", r"\bmgus\b", r"pseudofolliculitis", r"acral",
]


def _first(patterns, low):
    return next((re.search(p, low).group(0) for p in patterns if re.search(p, low)), None)


def pair_flags(dataset: str, pairs: pd.DataFrame, rows: list) -> pd.DataFrame:
    """Per-pair screens and source split for one pair set.

    Parameters
    ----------
    dataset : str
        Pair-set name.
    pairs : pandas.DataFrame
        One row per variant with ``pair_id``, ``variant`` and ``question``.
    rows : list of dict
        The source records the pairs were built from.

    Returns
    -------
    pandas.DataFrame
        One row per pair: ``split``; ``same_sex_kin`` (a relative of the
        original patient's sex is mentioned, so third-party pronouns may have
        been substituted); ``spouse_noun``; ``n_patient_np`` (age-qualified
        noun phrases; more than one means a second person is described);
        ``race_tier_a`` and ``race_tier_b`` (matched term or empty).
    """
    src = {str(r["question_id"]): r for r in rows}
    reverse = cfg.DATASETS.get(dataset, {}).get("kind") == "reverse"
    kin = FEMALE_KIN if reverse else MALE_KIN
    out = []
    for qid in dict.fromkeys(pairs["pair_id"]):
        r = src[qid]
        low = (r["stem"] + " " + " ".join(map(str, r["options"].values()))).lower()
        stem_low = r["stem"].lower()
        out.append({
            "dataset": dataset, "pair_id": qid, "split": r.get("split"),
            "subject": r.get("subject") if isinstance(r.get("subject"), str) else None,
            "same_sex_kin": bool(re.search(kin, stem_low)),
            "spouse_noun": bool(re.search(SPOUSE_NOUNS, stem_low)),
            "n_patient_np": len(AGE_NP.findall(r["stem"])),
            "race_tier_a": _first(RACE_TIER_A, low) or "",
            "race_tier_b": _first(RACE_TIER_B, low) or "",
        })
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #

def write_pairs(ps: PairSet, path: str, splits: dict) -> str:
    """Write a control pair set as JSONL for the inference runner.

    Same fields as the paper's pair files, plus ``split`` (the source split of
    the item), which the inference code ignores.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for _, r in ps.pairs.iterrows():
            fh.write(json.dumps({
                "question_id": r["pair_id"], "pair_id": r["pair_id"],
                "variant": r["variant"], "demographic": r["group"],
                "axis": r["axis"], "question": r["question"],
                "options": r["options"], "answer_idx": r["answer_idx"],
                "subject": r["subject"] if isinstance(r.get("subject"), str) else None,
                "split": splits.get(r["pair_id"]),
            }, ensure_ascii=False) + "\n")
    return path


def sha16(path: str) -> str:
    """First 16 hex characters of a file's SHA-256."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _restrict(ps: PairSet, ids) -> PairSet:
    ids = set(map(str, ids))
    return PairSet(ps.pairs[ps.pairs["pair_id"].isin(ids)].copy(), ps.excluded, ps.review)


def _paper_ids(outdir: str, name: str):
    """Pair ids of one of the paper's pair files, or None if it is not on disk."""
    path = os.path.join(outdir, f"{name}.jsonl")
    if not os.path.exists(path):
        return None
    return list(dict.fromkeys(json.loads(l)["pair_id"] for l in open(path, encoding="utf-8")
                              if l.strip()))


def review_sheet(ps: PairSet, n: int = 40, seed: int = cfg.SEED) -> str:
    """Markdown listing of sampled pairs with the changed words marked, for a
    human read-through before any compute is spent."""
    import difflib
    ids = pd.Series(list(dict.fromkeys(ps.pairs["pair_id"])))
    ids = ids.sample(min(n, len(ids)), random_state=seed)
    lines = []
    for pid in ids:
        a, b = ps.pairs[ps.pairs["pair_id"] == pid].to_dict("records")
        ta, tb = a["question"].split(), b["question"].split()
        ma, mb = [], []
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ta, tb).get_opcodes():
            ma += ta[i1:i2] if op == "equal" else [f"**{w}**" for w in ta[i1:i2]]
            mb += tb[j1:j2] if op == "equal" else [f"**{w}**" for w in tb[j1:j2]]
        lines += [f"### {pid}", f"- {a['variant']}: {' '.join(ma)}",
                  f"- {b['variant']}: {' '.join(mb)}", ""]
    return "\n".join(lines)


def build_all(outdir: str = cfg.PAIRS_DIR, n: int = 1000, full: bool = True,
              placebo_full: bool = False, sources=("medqa", "medmcqa")) -> pd.DataFrame:
    """Build every control pair set and the flags table.

    Parameters
    ----------
    outdir : str
        Where pair files are written; the paper's four files are expected here.
    n : int
        Pairs per sampled control set.
    full : bool
        Also write the full race and sex pools.
    placebo_full : bool
        Also write the placebo pairs for every race-eligible vignette.
    sources : sequence of str
        Source corpora.

    Returns
    -------
    pandas.DataFrame
        One row per pair file written: counts and hash.
    """
    os.makedirs(outdir, exist_ok=True)
    summary, flags = [], []

    def emit(name, ps, rows, splits):
        path = write_pairs(ps, os.path.join(outdir, f"{name}.jsonl"), splits)
        if len(ps.excluded):
            ps.excluded.to_csv(os.path.join(outdir, f"excluded_{name}.csv"), index=False)
        open(os.path.join(outdir, f"review_{name}.md"), "w", encoding="utf-8").write(
            f"# {name}: sampled pairs for reading\n\n" + review_sheet(ps))
        flags.append(pair_flags(name, ps.pairs, rows))
        summary.append({"dataset": name, "pairs": len(ps.pairs) // 2, "sha256_16": sha16(path),
                        "file": path})

    for src in sources:
        rows = load_source(src)
        splits = {str(r["question_id"]): r.get("split") for r in rows}
        strat = "subject" if src == "medmcqa" else None
        race_all, sex_all = build_race_pairs(rows), build_gender_pairs(rows)

        # The paper's files: verified, never rewritten; flags are computed for them.
        for axis, built in (("race", race_all), ("gender", sex_all)):
            name = f"pairs_{src}_{axis}"
            ids = _paper_ids(outdir, name)
            if ids is None:
                print(f"  note: {name}.jsonl not in {outdir}; run `python -m care.pairgen` "
                      "first so the controls can be matched to the paper's pairs")
                continue
            got = sha16(os.path.join(outdir, f"{name}.jsonl"))
            if got != PAPER_HASHES[name]:
                raise ValueError(f"{name}.jsonl has hash {got}, not the paper's "
                                 f"{PAPER_HASHES[name]}: the controls would not match")
            flags.append(pair_flags(name, _restrict(built, ids).pairs, rows))

        race_ids = _paper_ids(outdir, f"pairs_{src}_race")
        sex_ids = _paper_ids(outdir, f"pairs_{src}_gender")

        placebo = build_placebo_pairs(rows)
        emit(f"pairs_{src}_placebo",
             _restrict(placebo, race_ids) if race_ids else sample_pairs(placebo, n, stratify=strat),
             rows, splits)
        edited = build_gender_edited_pairs(rows)
        emit(f"pairs_{src}_gender_edited",
             _restrict(edited, sex_ids) if sex_ids else sample_pairs(edited, n, stratify=strat),
             rows, splits)
        emit(f"pairs_{src}_gender_reverse",
             sample_pairs(build_gender_reverse_pairs(rows), n, stratify=strat), rows, splits)
        if full:
            emit(f"pairs_{src}_race_full", race_all, rows, splits)
            emit(f"pairs_{src}_gender_full", sex_all, rows, splits)
        if placebo_full:
            emit(f"pairs_{src}_placebo_full", placebo, rows, splits)

    pd.concat(flags, ignore_index=True).to_csv(os.path.join(outdir, "pair_flags.csv"), index=False)
    out = pd.DataFrame(summary)
    out.drop(columns="file").to_csv(os.path.join(outdir, "control_pairs_manifest.csv"), index=False)
    return out


def seed_full_cache(results_root, model_dirs, conditions, sources=("medqa", "medmcqa"),
                    axes=("race", "gender", "placebo")) -> list:
    """Start each full-pool cache from the responses already on disk.

    The 1,000 pairs of a sampled set are part of the corresponding full pool
    and their text is identical, so their cached responses are valid for it.
    Copying them means the full run computes only the pairs that are new. A
    cache file that already exists is left alone.

    Parameters
    ----------
    results_root : str or Path
        Root of the inference cache.
    model_dirs : sequence of str
        Model directory names.
    conditions : sequence of str
        Prompting conditions.

    Returns
    -------
    list of str
        The cache files created.
    """
    import shutil
    made = []
    for src in sources:
        for axis in axes:
            for m in model_dirs:
                for c in conditions:
                    a = os.path.join(results_root, f"pairs_{src}_{axis}", m,
                                     "calibration_cache", f"{c}.jsonl")
                    b = os.path.join(results_root, f"pairs_{src}_{axis}_full", m,
                                     "calibration_cache", f"{c}.jsonl")
                    if os.path.exists(a) and not os.path.exists(b):
                        os.makedirs(os.path.dirname(b), exist_ok=True)
                        shutil.copyfile(a, b)
                        made.append(b)
    return made


def main(argv=None) -> int:
    """Build the control pair sets and print what was written."""
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--outdir", default=cfg.PAIRS_DIR)
    ap.add_argument("--n", type=int, default=1000, help="pairs per sampled control set")
    ap.add_argument("--no-full", action="store_true", help="skip the full race and sex pools")
    ap.add_argument("--placebo-full", action="store_true",
                    help="also write placebo pairs for every race-eligible vignette")
    args = ap.parse_args(argv)
    out = build_all(args.outdir, args.n, full=not args.no_full, placebo_full=args.placebo_full)
    print(out.drop(columns="file").to_string(index=False))
    print("flags ->", os.path.join(args.outdir, "pair_flags.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
