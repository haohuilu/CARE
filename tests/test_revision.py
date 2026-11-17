"""Tests for the revision code: control pairs, control prompts and the added analyses.

No network and no model calls. Tests that need the source corpora are skipped
when they are absent.
"""

from __future__ import annotations

import os
import re

import numpy as np
import pytest

from care import config as cfg
from care import revision as rv
from care.controls import (build_gender_edited_pairs, build_gender_reverse_pairs,
                           build_placebo_pairs, classify_her, handedness_reason,
                           swap_gender_reverse)
from care.pairgen import FEMALE_MARK, MALE_MARK


def _row(qid, stem, answer="A", dataset="medqa", **kw):
    return {"question_id": qid, "stem": stem, "answer_idx": answer, "dataset": dataset,
            "options": {"A": "one", "B": "two", "C": "three", "D": "four"}, **kw}


# --------------------------------------------------------------------------- #
# Control pairs
# --------------------------------------------------------------------------- #

def test_placebo_pairs_differ_by_one_word():
    rows = [_row("1", "A 45-year-old man presents with chest pain. He smokes."),
            _row("2", "A 30-year-old woman has aphasia after a stroke."),        # handedness matters
            _row("3", "A 60-year-old Black man presents with cough."),           # race stated
            _row("4", "A 9-year-old boy with sickle cell disease has pain.")]    # race-associated
    ps = build_placebo_pairs(rows)
    assert set(ps.pairs["pair_id"]) == {"1"}
    a, b = ps.pairs.sort_values("variant")["question"]
    assert a == "A 45-year-old left-handed man presents with chest pain. He smokes."
    assert b == "A 45-year-old right-handed man presents with chest pain. He smokes."
    assert set(ps.pairs["group"]) == set(cfg.PLACEBO_TERMS)
    assert handedness_reason("weakness of the dominant hand") is not None


def test_gender_edited_edits_both_versions():
    rows = [_row("1", "A 45-year-old man presents with chest pain. His pulse is 90/min.")]
    ps = build_gender_edited_pairs(rows)
    q = dict(zip(ps.pairs["variant"], ps.pairs["question"]))
    assert q["male"] == "A 45-year-old right-handed man presents with chest pain. His pulse is 90/min."
    assert q["female"] == "A 45-year-old right-handed woman presents with chest pain. Her pulse is 90/min."
    assert rows[0]["stem"] not in q.values(), "neither version may be the verbatim source text"


@pytest.mark.parametrize("text,want", [
    ("Her temperature is 37.0", ["poss"]),
    ("The physician examined her.", ["obj"]),
    ("She asked her to return.", ["obj"]),
    ("brought by her mother, who found her on the floor", ["poss", "obj"]),
    ("The nurse gave her medication at night.", ["ambiguous"]),
    ("after finding her seizing in her room", ["ambiguous", "poss"]),
    ("Her own account differs.", ["poss"]),
    ("They told her 3 times", ["ambiguous"]),
])
def test_classify_her(text, want):
    assert [r for _, _, r in classify_her(text)] == want


def test_reverse_swap():
    out, applied = swap_gender_reverse(
        "A 34-year-old woman presents with cough. She says her symptoms began when a "
        "colleague visited her. Her pulse is 80/min.")
    assert out == ("A 34-year-old man presents with cough. He says his symptoms began when a "
                   "colleague visited him. His pulse is 80/min.")
    assert len(applied) == 5
    assert swap_gender_reverse("The nurse gave her medication.")[0] is None


def test_reverse_pairs_original_is_female():
    rows = [
        _row("ok", "A 34-year-old woman presents with cough. Her pulse is 80/min."),
        _row("spouse", "A 34-year-old woman is brought by her husband with cough."),
        _row("kin", "A 9-year-old girl is brought by her mother. She has a fever."),
        _row("sexspec", "A 34-year-old woman presents with amenorrhea."),
        _row("mixed", "A 34-year-old woman says he hit her."),
        _row("male", "A 34-year-old man presents with cough."),
    ]
    ps = build_gender_reverse_pairs(rows)
    assert set(ps.pairs["pair_id"]) == {"ok"}
    q = dict(zip(ps.pairs["variant"], ps.pairs["question"]))
    assert q["female"] == rows[0]["stem"], "the female version must be the verbatim source"
    assert q["male"] == "A 34-year-old man presents with cough. His pulse is 80/min."
    assert not FEMALE_MARK.search(q["male"]) and not MALE_MARK.search(q["female"])
    why = dict(zip(ps.excluded["question_id"], ps.excluded["reason"]))
    assert "spouse" in why["spouse"] and "female relative" in why["kin"]
    assert "sex-specific" in why["sexspec"]


def test_control_datasets_registered():
    for src in ("medqa", "medmcqa"):
        for suffix in ("placebo", "gender_edited", "gender_reverse", "race_full", "gender_full"):
            spec = cfg.DATASETS[f"pairs_{src}_{suffix}"]
            assert spec["paired"] and spec["parent"] in cfg.DATASETS
    assert len(cfg.MAIN_DATASETS) == 4
    assert all(d not in cfg.CONTROL_DATASETS for d in cfg.MAIN_DATASETS)
    assert cfg.axis_of("pairs_medqa_gender_reverse") == "gender"
    assert cfg.axis_of("pairs_medmcqa_placebo") == "placebo"


def test_paper_pair_files_unchanged():
    """The control code must never alter the four pair files of the paper."""
    from care.controls import PAPER_HASHES, sha16
    found = 0
    for name, want in PAPER_HASHES.items():
        path = os.path.join(cfg.PAIRS_DIR, f"{name}.jsonl")
        if os.path.exists(path):
            found += 1
            assert sha16(path) == want, name
    if not found:
        pytest.skip(f"paper pair files not found under {cfg.PAIRS_DIR}")


# --------------------------------------------------------------------------- #
# Control prompts
# --------------------------------------------------------------------------- #

def test_control_prompts():
    from care.prompts import CONDITIONS, CONTROL_CONDITIONS, CONTROL_CONTRASTS, SYSTEM
    assert CONDITIONS == ("P1", "P2", "P3"), "the paper's conditions must not change"
    fair = re.compile(r"demograph|bias|equit|\brace\b|\bsex\b|gender|dispar|socioeconomic", re.I)
    for c in CONTROL_CONDITIONS:
        assert not fair.search(SYSTEM[c]), f"{c} must carry no fairness wording"
    assert fair.search(SYSTEM["P2"]) and fair.search(SYSTEM["P3"])
    persona = "board-certified physician"
    assert persona in SYSTEM["P4"] and persona in SYSTEM["P6"] and persona not in SYSTEM["P5"]
    n = {k: len(v.split()) for k, v in SYSTEM.items()}
    for c in ("P5", "P6"):                       # length-matched to the fairness prompts
        assert n["P2"] - 12 <= n[c] <= n["P3"] + 12, n
    assert n["P4"] < 35
    assert all(a in SYSTEM and b in SYSTEM for a, b, _ in CONTROL_CONTRASTS)


# --------------------------------------------------------------------------- #
# Analyses
# --------------------------------------------------------------------------- #

def test_auroc_and_aurc():
    c = np.array([0.9, 0.8, 0.7, 0.6])
    assert rv.auroc(c, [1, 1, 0, 0]) == 1.0
    assert rv.auroc(c, [0, 0, 1, 1]) == 0.0
    assert rv.auroc([0.5] * 4, [1, 0, 1, 0]) == 0.5            # all tied
    assert np.isnan(rv.auroc(c, [1, 1, 1, 1]))
    # risk at coverage 1/4, 2/4, 3/4, 4/4 = 0, 0, 1/3, 1/2
    assert rv.aurc(c, [1, 1, 0, 0]) == pytest.approx((0 + 0 + 1 / 3 + 1 / 2) / 4)
    # one tie block: every coverage level carries the overall error rate
    assert rv.aurc([0.5] * 4, [1, 0, 1, 0]) == pytest.approx(0.5)
    assert rv.aurc(c, [1, 1, 0, 0]) < rv.aurc(c, [0, 0, 1, 1])


def test_fit_tau_rules_are_nested():
    """A safety margin can only make the threshold stricter."""
    rng = np.random.default_rng(0)
    c = rng.uniform(0.3, 1, 2000)
    y = (rng.random(2000) < c).astype(float)
    t = {r: rv.fit_tau(c, y, 0.10, r) for r in ("point", "ucb", "ucb_bonf")}
    assert t["point"] <= t["ucb"] <= t["ucb_bonf"], t
    for r, tau in t.items():
        acc = c >= tau
        assert acc.sum() >= cfg.MIN_ACCEPTED and 1 - y[acc].mean() <= 0.10 + 1e-12, r
    # the bound itself: 0 errors in 30 gives 1 - 0.05 ** (1 / 30)
    assert float(rv.cp_upper(0, 30, 0.95)) == pytest.approx(1 - 0.05 ** (1 / 30))
    assert float(rv.cp_upper(30, 30, 0.95)) == 1.0
    assert rv.fit_tau(c, 1 - y, 0.01, "ucb") == np.inf        # nothing qualifies


def test_ucb_rule_holds_out_of_sample_more_often():
    """Held-out error exceeds the tolerance less often with a safety margin."""
    rng = np.random.default_rng(1)
    over = {"point": 0, "ucb": 0}
    for _ in range(60):
        c = rng.uniform(0.4, 1, 600)
        y = (rng.random(600) < c).astype(float)
        cn = rng.uniform(0.4, 1, 20000)
        yn = (rng.random(20000) < cn).astype(float)
        for r in over:
            acc = cn >= rv.fit_tau(c, y, 0.10, r)
            over[r] += bool(acc.any() and 1 - yn[acc].mean() > 0.10)
    assert over["ucb"] < over["point"], over
    assert over["ucb"] <= 0.15 * 60, over


def test_crossfit_accept_scopes():
    rng = np.random.default_rng(2)
    n = 1000
    c = rng.uniform(0.3, 1, (2, n))
    y = (rng.random((2, n)) < c).astype(float)
    folds = np.arange(n) % 5
    for scope in ("pooled", "both", "group"):
        acc, empty = rv.crossfit_accept(c, y, folds, 0.10, scope)
        assert acc.shape == (2, n) and acc.any() and empty.sum() == 0, scope
    # one threshold for both groups: acceptance depends on confidence alone
    acc, _ = rv.crossfit_accept(np.vstack([c[0], c[0]]), y, folds, 0.10, "pooled")
    assert (acc[0] == acc[1]).all()


def _fake_cube(effect: float, n: int = 400, seed: int = 0):
    """Four pair sets x two models x three prompts sharing pairs within a set."""
    rng = np.random.default_rng(seed)
    cube = {}
    for ds in cfg.MAIN_DATASETS:
        base = rng.uniform(0.3, 1, n)                    # item difficulty, shared by all
        blk = {}
        for m in ("m1", "m2"):
            for p in ("P1", "P2", "P3"):
                c_ref = np.clip(base + rng.normal(0, 0.03, n), 0.01, 1)
                c_dem = np.clip(base + rng.normal(0, 0.03, n), 0.01, 1)
                y_ref = (rng.random(n) < c_ref).astype(float)
                y_dem = (rng.random(n) < np.clip(c_dem - effect, 0, 1)).astype(float)
                blk[(m, p)] = (np.vstack([c_ref, c_dem]), np.vstack([y_ref, y_dem]))
        cube[ds] = {"ids": np.array([f"q{i:04d}" for i in range(n)]), "blk": blk}
    return cube


def test_permutation_test(monkeypatch):
    """No effect: the pattern is unremarkable. Planted effect: it is detected."""
    for effect, significant in ((0.0, False), (0.15, True)):
        monkeypatch.setattr(rv, "_cube", lambda *a, _e=effect, **k: _fake_cube(_e))
        out = rv.permutation(None, None, alphas=(0.10,), n_perm=300)
        p = out.set_index("statistic")["p_two_sided"]
        assert (p["mean_gap"] < 0.05) == significant, (effect, p.to_dict())
        assert (p["net"] < 0.05) == significant, (effect, p.to_dict())
        if significant:
            assert out["n_pos"].iloc[0] > out["n_neg"].iloc[0]


def test_non_estimable_comparisons_are_set_aside():
    """A SAR of 0.01 against 0 is the absence of a threshold, not a difference."""
    n = 1000
    c = np.linspace(1, 0.5, n)
    y = np.zeros(n)
    y[:10] = 1.0                                    # only the ten most confident are right
    ref = (c, y)
    dem = (c, np.zeros(n))
    # ten correct answers, then errors: at a 5% tolerance exactly ten can be accepted
    assert rv.safe_coverage(*ref, 0.05) == pytest.approx(0.01)
    gap, estimable = rv._gap(np.vstack([ref[0], dem[0]]), np.vstack([ref[1], dem[1]]), 0.05)
    assert gap == pytest.approx(0.01) and not estimable
    cube = {"pairs_medqa_race": {"ids": np.arange(n).astype(str),
                                 "blk": {("m", "P1"): (np.vstack([c, c]), np.vstack([y, np.zeros(n)]))}}}
    st = rv._pattern(cube, 0.05)
    assert st["n_pos"] == 0 and st["n_non_estimable"] == 1


def test_group_blind_threshold_on_the_boundary():
    """A group exactly on the tolerance is within it, as in safe_coverage."""
    from care.analysis import conservative_global, safe_coverage
    c = np.linspace(1, 0.5, 20)
    y = np.ones(20)
    y[-1] = 0.0                                     # 19 of 20 correct: error exactly 0.05
    assert safe_coverage(c, y, 0.05) == 1.0
    out = conservative_global({"a": (c, y), "b": (c, y)}, 0.05)
    assert out["a"] == 1.0 and out["b"] == 1.0, out


def test_power_table():
    import pandas as pd
    common = pd.DataFrame([{"dataset": "pairs_medqa_race", "model": "m", "prompt": "P1",
                            "alpha": 0.10, "scope": "pooled", "rule": "point",
                            "n_pairs": 1000, "sd_pair_diff": 0.2}])
    out = rv.power(common, effect=0.02, available={"pairs_medqa_race": 9699}).iloc[0]
    assert out["se_at_n"] == pytest.approx(0.2 / np.sqrt(1000))
    assert out["n_for_80pct_power"] == np.ceil(((1.959964 + 0.841621) * 0.2 / 0.02) ** 2)
    assert out["ci_half_width_at_available"] < out["ci_half_width_at_n"]


def test_both_tau_matches_reference_implementation():
    """The fast group-safe threshold equals the paper's implementation."""
    from care.analysis import conservative_global
    rng = np.random.default_rng(3)
    for trial in range(40):
        n = int(rng.integers(30, 400))
        c0, c1 = rng.uniform(0.3, 1, n), rng.uniform(0.3, 1, n)
        if trial % 3 == 0:                                # coarse values, many ties
            c0, c1 = np.round(c0, 1), np.round(c1, 1)
        y0 = (rng.random(n) < c0).astype(float)
        y1 = (rng.random(n) < c1 - rng.uniform(0, 0.3)).astype(float)
        for alpha in (0.05, 0.10, 0.15):
            want = conservative_global({"a": (c0, y0), "b": (c1, y1)}, alpha)["tau"]
            assert rv.both_tau(c0, y0, c1, y1, alpha) == want, (trial, alpha)
