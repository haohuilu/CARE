"""Tests for the core pipeline: pair construction, inference plumbing and analysis.

Run with ``pytest``. No network and no model calls. Tests that need the MedQA or
MedMCQA source files are skipped when those files are not present.
"""

from __future__ import annotations

import json
import os
import re

import numpy as np
import pandas as pd
import pytest

from care import config as cfg
from care.analysis import ece
from care.pairgen import (build_gender_pairs, build_race_pairs, load_source, swap_gender,
                          insert_race, sex_specific_reason, MALE_MARK, FEMALE_MARK)


def _need_source(name: str) -> None:
    sub = {"medqa": "MedQA", "medmcqa": "medmcqa"}[name]
    if not os.path.isdir(os.path.join(cfg.SOURCE_DIR, sub)):
        pytest.skip(f"{sub} source files not found under {cfg.SOURCE_DIR}")


def test_paired_contrast_null():
    """Identical prompts must give a prompt contrast of exactly zero."""
    from care.analysis import analyse_slice
    rng = np.random.default_rng(0)
    n, rows = 600, []
    for g, shift in (("white", 0.0), ("black", -0.12)):
        c = rng.uniform(0.3, 1.0, n)
        y = (rng.random(n) < np.clip(c + shift, 0, 1)).astype(int)
        rows.append(pd.DataFrame({"prompt": "P1", "group": g,
                                  "question_id": [f"q{i}" for i in range(n)],
                                  "conf_raw": c, "correct": y}))
    df = pd.concat(rows, ignore_index=True)
    df = pd.concat([df, df.assign(prompt="P2")], ignore_index=True)
    res, con = analyse_slice(df, "pairs_medqa_race", alphas=(0.10,), n_boot=200)
    c = con[0]
    assert c["d_gap"] == 0.0 and c["d_gap_lo"] == 0.0 and c["d_gap_hi"] == 0.0, c
    assert res[0]["sar_gap"] > 0, "planted disparity should give a positive SAR gap"


def test_ece_binning():
    """A perfectly calibrated sample should show a small ECE under both schemes."""
    rng = np.random.default_rng(cfg.SEED)
    p = rng.uniform(0.25, 1.0, 4000)
    y = (rng.random(4000) < p).astype(int)
    em = ece(p, y, scheme="equal_mass")
    ew = ece(p, y, scheme="equal_width", n_bins=10)
    assert em < 0.05 and ew < 0.05, f"ECE inflated on calibrated data: {em:.3f}/{ew:.3f}"

    # The equal-width estimator is the more biased one at small n.
    small = [ece(p[:130], y[:130], scheme=s, n_bins=n)
             for s, n in (("equal_mass", None), ("equal_width", 10))]
    assert small[1] >= small[0] - 0.02


def test_pairs_are_minimal():
    """Generated pairs must differ only in the demographic descriptor.

    This is the property the counterfactual claim depends on, so it is checked
    mechanically rather than asserted: race arms must differ by exactly one
    token, gender arms by a small fraction of tokens, every pair must share its
    reference answer, and no arm may retain a marker of the other group.
    """
    import difflib
    _need_source("medqa")
    rows = load_source("medqa")

    race = build_race_pairs(rows)
    assert len(race.pairs) > 0, "no race pairs built"
    for pid, sub in race.pairs.groupby("pair_id"):
        assert len(sub) == 2, f"pair {pid} has {len(sub)} arms"
        a, b = sub.iloc[0], sub.iloc[1]
        assert a["answer_idx"] == b["answer_idx"], f"pair {pid} gold differs"
        ta, tb = a["question"].split(), b["question"].split()
        ch = sum(max(i2 - i1, j2 - j1) for t, i1, i2, j1, j2
                 in difflib.SequenceMatcher(None, ta, tb).get_opcodes() if t != "equal")
        assert ch == 1, f"race pair {pid} differs by {ch} tokens, expected 1"

    gender = build_gender_pairs(rows)
    assert len(gender.pairs) > 0, "no gender pairs built"
    for pid, sub in gender.pairs.groupby("pair_id"):
        a, b = sub.iloc[0], sub.iloc[1]
        assert a["answer_idx"] == b["answer_idx"], f"pair {pid} gold differs"
        assert not FEMALE_MARK.search(a["question"]), f"pair {pid}: female marker in male arm"
        assert not MALE_MARK.search(b["question"]), f"pair {pid}: male marker in female arm"
        # Every changed token must be a gender marker from the rule table. This
        # is the property that matters and, unlike a fraction of the stem, it
        # does not weaken on short vignettes that are dense in pronouns.
        ta, tb = a["question"].split(), b["question"].split()
        for t, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ta, tb).get_opcodes():
            if t == "equal":
                continue
            left, right = ta[i1:i2], tb[j1:j2]
            assert len(left) == len(right), (
                f"gender pair {pid}: swap changed token count "
                f"({left!r} -> {right!r})")
            # The exact property: applying the substitution rules to the male
            # token must reproduce the female token, and nothing else may move.
            for x, y in zip(left, right):
                if x == y:
                    continue
                assert swap_gender(x)[0] == y, (
                    f"gender pair {pid}: {x!r} -> {y!r} is not a gender substitution")
        # Protected clinical compounds must survive the swap untouched.
        for pat in (r"MEN[\s-]?(?:1|2|3|I{1,3}|2A|2B)\b", r"male[-\s]pattern"):
            assert len(re.findall(pat, a["question"])) == len(re.findall(pat, b["question"])), (
                f"gender pair {pid} altered a protected clinical term")


def test_sex_specific_excluded():
    """Vignettes whose clinical content is sex-specific must not be swapped."""
    assert sex_specific_reason("A 30-year-old man with prostate cancer") is not None
    assert sex_specific_reason("She is gravida 2, para 1") is not None
    # Known false-positive traps that must NOT trigger exclusion.
    assert sex_specific_reason("Lactate dehydrogenase 1658 U/L") is None, "lactate wrongly excluded"
    assert sex_specific_reason("Ringer lactate solution was given") is None, "lactate wrongly excluded"
    assert sex_specific_reason("epidermal basement membrane with IgG") is None, "basement wrongly excluded"
    assert sex_specific_reason("cervical spine radiograph") is None, "cervical spine wrongly excluded"


def test_kinship_preserved():
    """A gender swap must not change people other than the patient."""
    txt = "A 40-year-old man reports that his mother had breast cancer and his brother has diabetes."
    out, _ = swap_gender(txt)
    assert "mother" in out and "brother" in out, "kinship noun was altered"
    assert "her mother" in out and "her brother" in out, "possessive not swapped"
    assert "woman" in out, "patient noun not swapped"


def test_race_insertion_symmetric():
    """Both race arms receive one inserted word at the same position."""
    txt = "A 67-year-old man with bladder cancer comes to the physician."
    w, okw = insert_race(txt, "White")
    b, okb = insert_race(txt, "Black")
    assert okw and okb
    assert w == "A 67-year-old White man with bladder cancer comes to the physician."
    assert b == "A 67-year-old Black man with bladder cancer comes to the physician."
    assert len(w.split()) == len(b.split()) == len(txt.split()) + 1


def test_safe_coverage():
    """SAR must match hand-computed values and respect confidence ties."""
    from care.analysis import safe_coverage
    # 20 cases, sorted confidence; first 12 correct, then wrong ones mixed in.
    conf = np.linspace(0.99, 0.30, 20)
    corr = np.array([1]*12 + [0, 1, 0, 1, 0, 0, 1, 0])
    # top-12 are all correct -> error 0; top-13 error 1/13=0.077; top-15 2/15=0.133
    assert safe_coverage(conf, corr, 0.10, min_accepted=10) == 14 / 20
    assert safe_coverage(conf, corr, 0.0, min_accepted=10) == 12 / 20
    # min_accepted guard: a perfect top-5 does not count when 10 are required
    assert safe_coverage(conf[:8], np.ones(8), 0.0, min_accepted=10) == 0.0
    # Ties: a threshold cannot split cases sharing one confidence value. Here
    # the 0.8 block is five correct then five wrong; splitting it after the
    # correct five would give 75% coverage at zero error, but no threshold can
    # do that, so the only valid cuts are 10 cases (error 0) and 20 (0.25).
    tied = np.array([0.9]*10 + [0.8]*10)
    y = np.array([1]*10 + [1]*5 + [0]*5)
    assert safe_coverage(tied, y, 0.10, min_accepted=5) == 0.5, "split a tie block"
    assert safe_coverage(tied, y, 0.25, min_accepted=5) == 1.0
    # infeasible budget -> 0, not missing
    assert safe_coverage(np.linspace(1, 0, 50), np.zeros(50), 0.10) == 0.0


def test_medmcqa_labels_aligned():
    """MedMCQA gold labels must match the letter stated in the explanations,
    and the check must fire if the option index is read with the wrong base."""
    from care.pairgen import load_medmcqa, check_medmcqa_labels
    _need_source("medmcqa")
    rows = load_medmcqa(splits=("dev",))
    rep = check_medmcqa_labels(rows)
    assert rep["agreement"] > 0.95, rep
    assert {r["answer_idx"] for r in rows} == {"A", "B", "C", "D"}, "an option letter is missing"
    shifted = [{**r, "answer_idx": "ABCD"[("ABCD".index(r["answer_idx"]) + 1) % 4]} for r in rows]
    try:
        check_medmcqa_labels(shifted)
    except ValueError:
        return
    raise AssertionError("an off-by-one label shift was not detected")


def test_race_relevant_excluded():
    """Vignettes with ancestry-linked conditions must not become race pairs."""
    from care.pairgen import build_race_pairs
    rows = [{"question_id": "1", "stem": "A 7-year-old boy with sickle cell disease has hip pain.",
             "options": {"A": "x", "B": "y"}, "answer_idx": "A", "dataset": "t"},
            {"question_id": "2", "stem": "A 7-year-old boy has hip pain after a fall.",
             "options": {"A": "x", "B": "y"}, "answer_idx": "A", "dataset": "t"}]
    ps = build_race_pairs(rows)
    assert set(ps.pairs["pair_id"]) == {"2"}, "race-relevant vignette was paired"


def test_conservative_global():
    """One shared threshold must keep every group within alpha."""
    from care.analysis import conservative_global, safe_coverage
    rng = np.random.default_rng(0)
    c1 = rng.uniform(0.5, 1, 200); y1 = (rng.random(200) < c1).astype(float)          # calibrated
    c2 = rng.uniform(0.5, 1, 200); y2 = (rng.random(200) < c2 - 0.08).astype(float)   # a bit worse
    out = conservative_global({"a": (c1, y1), "b": (c2, y2)}, 0.10)
    assert np.isfinite(out["tau"]) and out["a"] > 0 and out["b"] > 0, out
    for c, y, g in ((c1, y1, "a"), (c2, y2, "b")):
        acc = c >= out["tau"]
        assert 1 - y[acc].mean() <= 0.10 + 1e-9, f"group {g} breaches alpha"
        assert out[g] <= safe_coverage(c, y, 0.10, min_accepted=1) + 1e-9, "exceeds own SAR"
    # When one group can never meet the budget, the only safe shared policy is
    # to refer everyone: automation collapses for both groups.
    y3 = (rng.random(200) < c2 - 0.40).astype(float)
    worst = conservative_global({"a": (c1, y1), "b": (c2, y3)}, 0.10)
    assert worst["a"] < out["a"] and worst["b"] == 0.0, worst


def _fake_api(backend, model, chat_create=None, resp_create=None, msg_create=None):
    """APIModel wired to fake SDK endpoints, with backoff disabled."""
    from types import SimpleNamespace as NS
    from care import infer
    infer.time.sleep = lambda s: None
    api = infer.APIModel.__new__(infer.APIModel)
    api.backend, api.model = backend, model
    api.style = api.params = api.resolved_model = api.last_error = None
    api.client = NS(chat=NS(completions=NS(create=chat_create)),
                    responses=NS(create=resp_create), messages=NS(create=msg_create))
    return api


def test_api_values_not_cached():
    """A one-token TLP call must not truncate later CE calls or fix SC at T=0."""
    from types import SimpleNamespace as NS
    seen = []
    def create(*, model, messages, n=1, max_tokens=None, temperature=None,
               logprobs=None, top_logprobs=None):
        seen.append((max_tokens, temperature, n))
        ch = NS(message=NS(content="Answer=[B]\nConfidence=[80]"), logprobs=None)
        return NS(model=model, choices=[ch] * n)
    api = _fake_api("openai", "m", chat_create=create)
    api.generate("s", "u", max_tokens=1, logprobs=True)
    api.generate("s", "u", max_tokens=32)
    api.generate("s", "u", n=10, temperature=0.7, max_tokens=8)
    assert seen == [(1, 0.0, 1), (32, 0.0, 1), (8, 0.7, 10)], seen


def test_api_responses_fallback_persists():
    """After switching to the Responses API, later calls must keep working."""
    from types import SimpleNamespace as NS
    def chat(*, model, messages, n=1, max_tokens=None, max_completion_tokens=None,
             temperature=None, reasoning_effort=None, logprobs=None, top_logprobs=None):
        raise Exception("This model is only supported in v1/responses and not in v1/chat/completions.")
    calls = []
    def resp(*, model, instructions, input, max_output_tokens):
        calls.append(max_output_tokens)
        return NS(model=model, output_text="Answer=[D]\nConfidence=[70]")
    api = _fake_api("openai", "gpt-6-astra", chat_create=chat, resp_create=resp)
    outs = [api.generate("s", "u", max_tokens=32)[0] for _ in range(3)]
    assert all(o == ["Answer=[D]\nConfidence=[70]"] for o in outs), outs
    assert len(calls) == 3 and api.params["api"] == "responses"


def test_api_fail_fast_and_breaker():
    """Permanent errors must not be retried, and a failing run must stop early."""
    import json as _json, tempfile
    from care import infer
    tries = []
    class NotFound(Exception):
        status_code = 404
    def create(**kw):
        tries.append(1)
        raise NotFound("The model 'gpt-typo' does not exist")
    api = _fake_api("openai", "gpt-typo", chat_create=create)
    api.generate("s", "u", max_tokens=32)
    assert len(tries) == 1, f"permanent error retried {len(tries)} times"
    tries.clear()
    recs = [{"question_id": str(i), "pair_id": str(i), "variant": "white",
             "demographic": "white", "question": "q", "options": {"A": "a", "B": "b"},
             "answer_idx": "A"} for i in range(200)]
    root = tempfile.mkdtemp()
    try:
        infer.run_api(api, "gpt-typo", {"d": recs}, {"d": 2}, {"P1": "s"}, ["P1"], ["ce"],
                      root, max_workers=2, fail_fast_after=20)
    except RuntimeError as e:
        assert "does not exist" in str(e) and len(tries) < 60, (len(tries), e)
        return
    raise AssertionError("run did not stop on a permanent error")


def test_anthropic_temperature_negotiated():
    """A Claude model that rejects temperature must still be callable."""
    from types import SimpleNamespace as NS
    class BadRequest(Exception):
        status_code = 400
    def create(*, model, system, messages, max_tokens, temperature=None):
        if temperature is not None:
            raise BadRequest("temperature is not supported for this model")
        return NS(model=model, content=[NS(type="text", text="Answer=[A]\nConfidence=[90]")])
    api = _fake_api("anthropic", "claude-x", msg_create=create)
    outs, _ = api.generate("s", "u", max_tokens=32)
    assert outs == ["Answer=[A]\nConfidence=[90]"] and "temperature" not in api.params


def test_truncated_line_recovery():
    """A record cut off by a disconnect must not corrupt the next append."""
    import tempfile
    from pathlib import Path
    from care import infer
    p = Path(tempfile.mkdtemp()) / "C.jsonl"
    p.write_text('{"question_id": "1", "variant": "white", "tlp_conf": 0.9}\n{"question_id": "2", "vari')
    with infer.open_append(p) as fh:
        fh.write('{"question_id": "3", "variant": "white", "tlp_conf": 0.8}\n')
    good = []
    for line in p.read_text().splitlines():
        try:
            good.append(json.loads(line)["question_id"])
        except json.JSONDecodeError:
            pass
    assert good == ["1", "3"], good
    assert infer.done_keys(p, "tlp") == {("1", "white"), ("3", "white")}


def test_unparsed_counts_as_referred():
    """Cases without a confidence stay in the SAR denominator as referrals."""
    from care.analysis import safe_coverage
    conf = np.array([0.99] * 20 + [np.nan] * 20)
    corr = np.ones(40)
    assert safe_coverage(conf, corr, 0.10) == 0.5


def test_ce_parsing():
    """CE parsing accepts common formats and rejects an echoed template."""
    from care.infer import parse_ce
    L = list("ABCDE")
    assert parse_ce("Answer=[B]\nConfidence=[85]", L) == ("B", 0.85)
    assert parse_ce("Answer: C\nConfidence: 70%", L) == ("C", 0.70)
    assert parse_ce("answer = (D)\nconfidence = 95", L) == ("D", 0.95)
    assert parse_ce("Answer=[E]\nConfidence=[100]", list("ABCD")) == (None, None), "E invalid for 4 options"
    assert parse_ce("Answer=[A, B, C, D, E]\nConfidence=[an integer from 0 to 100]", L) == (None, None), \
        "echoed template scored as an answer"
    assert parse_ce("Answer=[B]", L) == (None, None), "missing confidence accepted"
    assert parse_ce("Answer=[B]\nConfidence=0.85", L) == ("B", 0.85), "probability read as 0%"
    assert parse_ce("Answer=[B]\nConfidence=[85.5]", L) == ("B", 0.855)
    assert parse_ce("Answer=[B]\nConfidence=[0]", L) == ("B", 0.0)
    assert parse_ce("**Answer:** B\n**Confidence:** 80%", L) == ("B", 0.8), "markdown reply"
    assert parse_ce("Answer=[A, B, C, D]\nConfidence=95", L) == (None, None), "echoed list"
    assert parse_ce("Answer=A: >60 breaths/min\nConfidence=100", L) == ("A", 1.0), "letter then text"
    assert parse_ce("A:\u202f>60/min\nConfidence:\u202f90", L) == ("A", 0.9), "no 'Answer' word"
    assert parse_ce("answer: a patient\nConfidence: 60", L) == (None, None), "lower-case word"
    assert parse_ce("Answer: Both\nConfidence: 50", L) == (None, None), "word starting with B"
    assert parse_ce("The answer is C.\nConfidence: 80", L) == ("C", 0.8)
    assert parse_ce("D\n100", L) == ("D", 1.0), "bare two-line reply"
    assert parse_ce("D\nConfidence: 95", L) == ("D", 0.95), "bare letter line, labelled confidence"
    assert parse_ce("A 45-year-old man\nConfidence: 95", L) == (None, None), "sentence starting with A"
    assert parse_ce("E\n90", list("ABCD")) == (None, None), "bare reply, invalid letter"


def test_api_quota_notag_and_streak():
    """Exhausted quota is not retried; letter-less TLP replies are recorded as
    referred; a run that starts failing midway stops instead of grinding on."""
    import json as _json, tempfile
    from types import SimpleNamespace as NS
    from care import infer
    tries = []
    class Quota(Exception):
        status_code = 429
    def create(**kw):
        tries.append(1)
        raise Quota("You exceeded your current quota (insufficient_quota)")
    api = _fake_api("openai", "m", chat_create=create)
    api.generate("s", "u", max_tokens=32)
    assert len(tries) == 1, f"quota error retried {len(tries)} times"

    # first-token distribution without any option letter
    tok = lambda s, lp: NS(token=s, logprob=lp)
    def no_letter(*, model, messages, n=1, max_tokens=None, temperature=None,
                  logprobs=None, top_logprobs=None):
        lp = NS(content=[NS(top_logprobs=[tok("The", -0.1), tok("Answer", -2.0)])])
        return NS(model=model, choices=[NS(message=NS(content="The"), logprobs=lp)])
    api = _fake_api("openai", "m", chat_create=no_letter)
    recs = [{"question_id": "q1", "pair_id": "q1", "variant": "white", "demographic": "white",
             "question": "x", "options": dict(zip("ABCDE", "abcde")), "answer_idx": "A"}]
    root = tempfile.mkdtemp()
    infer.run_api(api, "m", {"pairs_medqa_race": recs}, {"pairs_medqa_race": 5},
                  {"P1": "s"}, ["P1"], ["tlp"], root, max_workers=1)
    path = infer.out_path(root, "pairs_medqa_race", "m", "P1")
    rows = [_json.loads(l) for l in open(path)]
    assert len(rows) == 1 and rows[0]["tlp_parse_ok"] is False and rows[0]["tlp_conf"] is None
    assert infer.done_keys(path, "tlp") == {("q1", "white")}, "letter-less reply re-run forever"

    # 30 successes, then everything fails: must stop well before the end
    calls = {"n": 0}
    class Down(Exception):
        status_code = 400
    def flaky(*, model, messages, n=1, max_tokens=None, temperature=None,
              logprobs=None, top_logprobs=None):
        calls["n"] += 1
        if calls["n"] > 30:
            raise Down("credit balance is too low")
        return NS(model=model, choices=[NS(message=NS(content="Answer=[A]\nConfidence=[90]"),
                                           logprobs=None)])
    api = _fake_api("openai", "m", chat_create=flaky)
    many = [dict(recs[0], question_id=f"q{i}", pair_id=f"q{i}") for i in range(1000)]
    try:
        infer.run_api(api, "m2", {"pairs_medqa_race": many}, {"pairs_medqa_race": 5},
                      {"P1": "s"}, ["P1"], ["ce"], root, max_workers=1)
        stopped = False
    except RuntimeError:
        stopped = True
    assert stopped and calls["n"] < 200, f"kept calling after failures began ({calls['n']} calls)"


def test_incomplete_pair_keeps_slice():
    """One missing arm must drop that pair, not the whole slice."""
    import json as _json, tempfile, os
    from care import config as cfg
    from care.analysis import load_slice
    root = tempfile.mkdtemp()
    old = cfg.DATA_ROOT
    cfg.DATA_ROOT = root
    try:
        d = os.path.join(root, "pairs_medqa_race", "m", "calibration_cache")
        os.makedirs(d)
        rng = np.random.default_rng(0)
        with open(os.path.join(d, "P1.jsonl"), "w") as fh:
            for i in range(60):
                for g in ("white", "black"):
                    if i == 0 and g == "black":
                        continue                      # incomplete pair
                    c = float(rng.uniform(0.3, 1))
                    ok = int(rng.random() < c)
                    fh.write(_json.dumps({"question_id": f"q{i}", "variant": g, "demographic": g,
                             "condition": "P1", "model": "m", "answer_idx": "A",
                             "tlp_answer": "A" if ok else "B", "tlp_conf": c, "tlp_correct": ok,
                             "tlp_option_probs": _json.dumps({"A": c, "B": 1 - c}),
                             "tlp_think_prefix": "", "tlp_precision": "fp64"}) + "\n")
        df, rep = load_slice("pairs_medqa_race", "m", "tlp")
    finally:
        cfg.DATA_ROOT = old
    assert df is not None, f"slice dropped: {rep.errors}"
    assert len(df) == 118 and "q0" not in set(df.question_id), len(df)


def test_text_answerability_filter():
    """Image-dependent and indistinguishable-option items are excluded."""
    from care.pairgen import not_answerable_as_text as f
    o = dict(zip("ABCD", ["Aspirin", "Heparin", "Warfarin", "Alteplase"]))
    assert f({"stem": "A 60-year-old man has chest pain.", "options": o}) is None
    assert f({"stem": "His ECG is shown in the image below.", "options": o}) == "image"
    assert f({"stem": "The following X-ray was taken.", "options": o}) == "image"
    assert f({"stem": "Identify the lesion.", "options": dict(o, A='<img src="x"/>')}) == "image"
    assert f({"stem": "Q", "options": dict(o, D="aspirin")}) == "duplicate_options"
    assert f({"stem": "An x-ray shows a fracture of the radius.", "options": o}) is None, \
        "a described finding is answerable from text"


def test_crossfit_honest():
    """Cross-fitted thresholds keep the realised error near the budget on
    calibrated data, and expose in-sample optimism on pure noise."""
    from care.analysis import crossfit, safe_coverage
    rng = np.random.default_rng(1)
    n = 2000
    c = rng.uniform(0.3, 1.0, n)
    y = (rng.random(n) < c).astype(float)
    folds = rng.permutation(np.arange(n) % 5)
    cf = crossfit(c, y, folds, 0.10)
    sar = safe_coverage(c, y, 0.10)
    assert abs(cf["cov"] - sar) < 0.08, (cf, sar)
    assert cf["err"] < 0.14, cf
    # confidence unrelated to correctness (accuracy 0.9): in-sample SAR picks a
    # lucky cut, but out of sample the error is simply the base rate
    y2 = (rng.random(n) < 0.9).astype(float)
    cf2 = crossfit(c, y2, folds, 0.08)
    assert np.isnan(cf2["err"]) or cf2["err"] > 0.07, cf2


def test_thinking_model_empty_reply():
    """A thinking model that burns a small budget must not yield empty replies
    recorded as 'unparsed'; the budget is raised and old empty rows re-run."""
    import json as _json, tempfile
    from types import SimpleNamespace as NS
    from care import infer
    class BadReq(Exception):
        status_code = 400
    calls = []
    def create(*, model, system, messages, max_tokens, temperature=None, thinking=None):
        calls.append((max_tokens, temperature, thinking))
        if thinking is not None:
            raise BadReq("thinking cannot be disabled for this model")
        if temperature is not None:
            raise BadReq("temperature is not supported when thinking is enabled")
        if max_tokens < 500:     # budget used up by hidden thinking
            return NS(model=model, stop_reason="max_tokens",
                      content=[NS(type="thinking", thinking="...")])
        return NS(model=model, stop_reason="end_turn",
                  content=[NS(type="thinking", thinking="..."),
                           NS(type="text", text="Answer=[B]\nConfidence=[90]")])
    api = _fake_api("anthropic", "claude-x", msg_create=create)
    outs, _ = api.generate("s", "u", max_tokens=32)
    assert outs == ["Answer=[B]\nConfidence=[90]"], (outs, calls)
    assert api.params == {"max_tokens": 2048}, api.params

    # an old empty 'unparsed' row is not treated as done
    d = tempfile.mkdtemp()
    p = infer.out_path(d, "pairs_medqa_race", "m", "P1")
    with open(p, "w") as fh:
        fh.write(_json.dumps({"question_id": "q1", "variant": "white", "ce_answer": None,
                              "ce_conf": None, "ce_parse_ok": False, "ce_raw_text": "",
                              "api_params": "{}"}) + "\n")
        fh.write(_json.dumps({"question_id": "q3", "variant": "white", "ce_answer": None,
                              "ce_conf": None, "ce_parse_ok": False, "ce_raw_text": "",
                              "ce_version": infer.CE_VERSION}) + "\n")
        fh.write(_json.dumps({"question_id": "q2", "variant": "white", "ce_answer": None,
                              "ce_conf": None, "ce_parse_ok": False,
                              "ce_raw_text": "I cannot answer", "ce_version": infer.CE_VERSION}) + "\n")
    assert infer.done_keys(p, "ce") == {("q2", "white"), ("q3", "white")}, \
        "empty API reply must re-run; empty local greedy reply is a real (unparsed) answer"


def test_ce_version_rerun():
    """CE records made under an older instruction are re-run and ignored."""
    import json as _json, tempfile
    from care import infer
    d = tempfile.mkdtemp()
    p = infer.out_path(d, "pairs_medqa_race", "m", "P1")
    base = {"variant": "white", "ce_answer": "A", "ce_conf": 0.9, "ce_parse_ok": True}
    with open(p, "w") as fh:
        fh.write(_json.dumps({**base, "question_id": "old"}) + "\n")
        fh.write(_json.dumps({**base, "question_id": "old2", "ce_version": "deadbeef"}) + "\n")
        fh.write(_json.dumps({**base, "question_id": "new", "ce_version": infer.CE_VERSION}) + "\n")
        fh.write(_json.dumps({"question_id": "t", "variant": "white", "tlp_conf": 0.5}) + "\n")
    assert infer.done_keys(p, "ce") == {("new", "white")}
    assert infer.done_keys(p, "tlp") == {("t", "white")}, "TLP unaffected"


def test_ceiling_and_automation_cost():
    """SAR = 1 in both groups is flagged as a ceiling (the gap is uninformative),
    and a prompt that lowers accuracy shows up as a cost in average SAR."""
    from care.analysis import analyse_slice
    rng = np.random.default_rng(3)
    n = 600
    def arm(p_correct, cond):
        rows = []
        for g in ("white", "black"):
            c = rng.uniform(0.2, 1.0, n)
            y = (rng.random(n) < np.clip(c + p_correct, 0, 1)).astype(int)
            rows.append(pd.DataFrame({"prompt": cond, "group": g, "question_id": [f"q{i}" for i in range(n)],
                                      "conf_raw": c, "correct": y}))
        return pd.concat(rows)
    df = pd.concat([arm(0.0, "P1"), arm(-0.25, "P2")], ignore_index=True)
    rows, con = analyse_slice(df, "pairs_medqa_race", alphas=(0.10,), n_boot=200)
    c = con[0]
    assert c["d_sar_mean"] < 0 and c["costs_automation"], c
    assert not c["ceiling"]

    ok = pd.concat([arm(1.0, "P1"), arm(1.0, "P2")], ignore_index=True)   # always correct
    rows, con = analyse_slice(ok, "pairs_medqa_race", alphas=(0.10,), n_boot=200)
    assert all(r["ceiling"] for r in rows), rows
    assert con[0]["ceiling"] and not con[0]["rules_out_meaningful_improvement"], con[0]


def test_no_artificial_ties_near_one():
    """Confidences within 1e-7 of 1 must stay distinct: rounding them into one
    tie block turns SAR into a knife-edge between 0 and most of the cases."""
    from care.analysis import safe_coverage
    rng = np.random.default_rng(0)
    n = 1000
    gap = 10 ** rng.uniform(-14, -7, n)          # 1 - p, all within 1e-7 of 1
    rank = np.argsort(np.argsort(gap)) / n        # 0 = most confident
    y = (rng.random(n) > 0.2 * rank).astype(float)  # less sure -> more errors (~10% overall)
    sar = safe_coverage(1 - gap, y, 0.05)
    err_all = 1 - y.mean()
    assert err_all > 0.05 and 0.2 < sar < 1.0, (err_all, sar)


def test_medgemma_no_think_prefix():
    """MedGemma 1.5 opens with a reasoning trace; its turn must be pre-filled
    with an empty one, and no other model may be touched."""
    from types import SimpleNamespace as NS
    from care.infer import chat_text
    class Tok:
        chat_template = "x"
        def __init__(self, name): self.name_or_path = name
        def apply_chat_template(self, msgs, **kw): return "<start_of_turn>model\n"
    assert chat_text(Tok("google/medgemma-1.5-4b-it"), "s", "u").endswith("<unused94>thought\n<unused95>")
    for other in ("google/gemma-4-E4B-it", "m42-health/Llama3-Med42-8B", "google/medgemma-4b-it"):
        assert "<unused94>" not in chat_text(Tok(other), "s", "u"), other


def test_provenance_and_reparse():
    """Local records made by older code are re-run and ignored; CE replies are
    re-parsed on load, so parser fixes reach stored results."""
    import json as _json, tempfile, os
    from care import config as cfg, infer
    from care.data import load_records
    root = tempfile.mkdtemp(); old = cfg.DATA_ROOT; cfg.DATA_ROOT = root
    try:
        mdir = "google_medgemma-1.5-4b-it"
        p = infer.out_path(root, "pairs_medqa_race", mdir, "P1")
        pre = infer.NO_THINK_PREFIX["medgemma-1.5"]
        base = {"variant": "white", "demographic": "white", "condition": "P1", "model": mdir,
                "answer_idx": "D", "ce_version": infer.CE_VERSION}
        with open(p, "w") as fh:
            # made before the pre-fill: must be superseded
            fh.write(_json.dumps({**base, "question_id": "q1", "ce_answer": None, "ce_conf": None,
                                  "ce_parse_ok": False, "ce_raw_text": "<unused94>thought ..."}) + "\n")
            # current, stored as unparsed by an older parser: must be re-parsed
            fh.write(_json.dumps({**base, "question_id": "q2", "ce_answer": None, "ce_conf": None,
                                  "ce_parse_ok": False, "ce_raw_text": "D\n100",
                                  "ce_think_prefix": pre}) + "\n")
        class Tok:
            name_or_path = "google/medgemma-1.5-4b-it"
        assert infer.done_keys(p, "ce", require=infer.local_provenance(Tok(), "ce")) == {("q2", "white")}
        df = load_records("pairs_medqa_race", mdir, "ce")
    finally:
        cfg.DATA_ROOT = old
    assert list(df.question_id) == ["q2"], "record from older code was used"
    assert bool(df.parse_ok.iloc[0]) and df.conf_raw.iloc[0] == 1.0 and df.correct.iloc[0] == 1


def test_truncated_api_ce_rerun():
    """API CE replies left unparsed under a smaller output cap are re-run;
    parsed ones are kept."""
    import json as _json, tempfile
    from care import infer
    p = infer.out_path(tempfile.mkdtemp(), "pairs_medqa_race", "claude", "P3")
    base = {"variant": "white", "ce_version": infer.CE_VERSION, "api_params": "{}"}
    with open(p, "w") as fh:
        fh.write(_json.dumps({**base, "question_id": "cut", "ce_parse_ok": False, "ce_conf": None,
                              "ce_raw_text": "This presentation is consistent with"}) + "\n")
        fh.write(_json.dumps({**base, "question_id": "fine", "ce_parse_ok": True, "ce_conf": 0.9,
                              "ce_raw_text": "Answer: B\nConfidence: 90"}) + "\n")
        fh.write(_json.dumps({**base, "question_id": "refused", "ce_parse_ok": False, "ce_conf": None,
                              "ce_raw_text": "I can't pick one.", "ce_max_tokens": infer.CE_API_MAX_TOKENS}) + "\n")
    assert infer.done_keys(p, "ce") == {("fine", "white"), ("refused", "white")}
