"""Inference for the referral study: local token log-probabilities and API
elicited confidence, with resume, validity diagnostics and a run manifest.

Design points that matter for the analysis:

* TLP is computed from one forward pass. Before renormalising over the option
  letters we record ``letter_mass`` -- the probability the model put on *any*
  option letter as its next token. A low value means the model wanted to emit
  something else first ("The answer is..."), and TLP is then not a faithful
  confidence. It is reported per model rather than silently renormalised away.
* Local inference is batched with left padding and explicit position ids, so a
  batched result matches the unbatched one; the smoke test checks this per
  model rather than assuming it.
* API calls adapt to newer reasoning models, which reject ``temperature`` and
  ``max_tokens``; the parameters actually used are stored with every record.
* Every record is appended as soon as it exists, so a run can be interrupted
  and resumed without repeating paid calls. Failed calls are skipped, never
  written with a fabricated value.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

LETTERS = ["A", "B", "C", "D", "E"]
CODE_VERSION = "2026-09-23a"     # printed by the notebooks; bump on every change
# Output cap for API elicited confidence. Compliant replies need ~12 tokens and
# stop there; the cap only matters for models that explain first, which a small
# cap would cut off before the answer (and more often under longer prompts).
# Claude's explanations run 80-120 words, so 256 still cut ~0.2% of replies --
# some in the middle of "Answer: E / Confidence".
CE_API_MAX_TOKENS = 512

# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

TLP_INSTRUCTION = "Select the best option and output only the index letter ({opts})."
# The format line must not show the option list itself ("Answer=[A, B, C]"):
# models copy it verbatim instead of answering.
CE_INSTRUCTION = ("Select the best option ({opts}). Reply in exactly two lines, "
                  "with nothing before or after them:\n"
                  "Answer: <the letter of your chosen option>\n"
                  "Confidence: <an integer from 0 to 100: the probability that "
                  "your answer is correct>")
# Stored with every CE record; records made under another instruction are
# re-run, so every model is compared on the same prompt.
CE_VERSION = hashlib.sha256(CE_INSTRUCTION.encode()).hexdigest()[:8]


def user_prompt(rec: dict, method: str = "tlp") -> str:
    """Vignette, lettered options, and the method's output instruction.

    Parameters
    ----------
    rec : dict
        Pair record with ``question`` and ``options``.
    method : {"tlp", "ce"}
        Confidence method, which fixes the output instruction.

    Returns
    -------
    str
        The user turn.
    """
    ls = LETTERS[:len(rec["options"])]
    tail = (CE_INSTRUCTION if method == "ce" else TLP_INSTRUCTION).format(opts=", ".join(ls))
    return "\n".join([rec["question"].strip(), ""]
                     + [f"{L}: {rec['options'][L]}" for L in ls] + ["", tail])


# Accepted: "Answer: B", "Answer=[B]", "**Answer:** B", "Answer: <B>", "Answer=A: text".
# Rejected: an echoed list ("Answer=[A, B, C, D]") and words ("Answer: Both").
# The letter itself is case-sensitive, so "answer: a patient..." is not "A".
ANS_RE = re.compile(r"(?i:answer)\**\s*(?:[=:]|(?i:is))\s*\**\s*[\[(<]?\s*([A-E])"
                    r"(?![A-Za-z])(?!\s*[,/]\s*[A-E](?![A-Za-z]))")
# Fallback when the word "Answer" is missing: a first line such as "B: text",
# "(B)", or just "B" on its own line.
LEAD_RE = re.compile(r"^\W{0,3}([A-E])\s*(?:[:.)\]>]|$)")
# A bare two-line reply, "D" then "100", with the labels left out.
BARE_RE = re.compile(r"^\s*([A-E])\s*\n\s*(\d{1,3}(?:\.\d+)?)\s*(%?)\s*$")
CONF_RE = re.compile(r"(?i:confidence)\**\s*[=:]\s*\**\s*[\[(<]?\s*(\d{1,3}(?:\.\d+)?)\s*(%?)")


def parse_ce(text: str, letters) -> tuple:
    """(letter, confidence in [0, 1]) from an elicited-confidence reply.

    Parameters
    ----------
    text : str
        Model reply.
    letters : sequence of str
        Valid option letters.

    Returns
    -------
    tuple
        ``(letter, conf)``, or ``(None, None)`` if either part is missing.
    """
    text = text or ""
    bare = BARE_RE.match(text)
    if bare:
        letter, num, pct = bare.group(1), bare.group(2), bare.group(3)
    else:
        a, c = ANS_RE.search(text), CONF_RE.search(text)
        if not a:
            first = next((l for l in text.strip().splitlines() if l.strip()), "")
            a = LEAD_RE.match(first.strip())
        if not a or not c:
            return None, None
        letter, num, pct = a.group(1), c.group(1), c.group(2)
    letter = letter.upper()
    if letter not in letters:
        return None, None
    v = float(num)
    # "0.85" without a percent sign is a probability, not 0.85 percent
    if "." in num and v <= 1.0 and not pct:
        v *= 100.0
    return letter, min(max(v, 0.0), 100.0) / 100.0


# --------------------------------------------------------------------------- #
# Resume-safe I/O
# --------------------------------------------------------------------------- #

def out_path(results_root, dataset: str, model_dir: str, cond: str) -> Path:
    """Cache file for one dataset x model x condition, created if absent.

    Parameters
    ----------
    results_root : str or Path
        Root of the results tree.
    dataset, model_dir, cond : str
        Slice identifiers.

    Returns
    -------
    Path
        ``<root>/<dataset>/<model_dir>/calibration_cache/<cond>.jsonl``.
    """
    p = Path(results_root) / dataset / model_dir / "calibration_cache"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{cond}.jsonl"


def done_keys(path: Path, method: str, require: dict | None = None) -> set:
    """(question_id, variant) pairs already recorded for this method.

    A truncated final line from an interrupted run is ignored.

    Parameters
    ----------
    path : Path
        Cache file.
    method : str
        Confidence method.
    require : dict, optional
        Fields a record must carry with exactly these values to count as done
        -- how it was produced (prompt pre-fill, precision). Records made
        another way are run again.

    Returns
    -------
    set of tuple
        Completed keys.
    """
    if not path.exists():
        return set()
    seen = set()
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        # An empty API reply is a request problem: run it again. An empty local
        # greedy reply is the model's deterministic output and stays unparsed.
        empty = "api_params" in r and r.get(f"{method}_parse_ok") is False and \
            not str(r.get(f"{method}_raw_text", "x") or "").strip()
        if empty:
            continue
        if method == "ce" and r.get("ce_version") != CE_VERSION:
            continue            # made under an older CE instruction: run again
        if require and any(r.get(k) != v for k, v in require.items()):
            continue            # produced another way (older code): run again
        if method == "ce" and "api_params" in r and r.get("ce_parse_ok") is False \
                and r.get("ce_max_tokens") != CE_API_MAX_TOKENS:
            continue            # unparsed under a smaller output cap (likely cut off): run again
        if r.get(f"{method}_conf") is not None or r.get(f"{method}_parse_ok") is False:
            seen.add((str(r["question_id"]), r["variant"]))
    return seen


def open_append(path: Path):
    """Open a cache file for appending, repairing a truncated last line first.

    If a session died mid-write, the file ends without a newline; appending
    directly would glue the next record onto the partial one and corrupt both.
    The partial line is terminated so it is ignored on read, and the new
    records start on a clean line.

    Parameters
    ----------
    path : Path
        Cache file.

    Returns
    -------
    file object
        Opened in append mode.
    """
    if path.exists() and path.stat().st_size > 0:
        with open(path, "rb") as fh:
            fh.seek(-1, os.SEEK_END)
            last = fh.read(1)
        if last != b"\n":
            with open(path, "a", encoding="utf-8") as fh:     # builtin open, not open_append
                fh.write("\n")
    return open(path, "a", encoding="utf-8")


def base_record(rec: dict, cond: str, model: str) -> dict:
    """Fields shared by every result row.

    Parameters
    ----------
    rec : dict
        Pair record.
    cond : str
        Prompting condition.
    model : str
        Model identifier.

    Returns
    -------
    dict
        Identifying fields, including ``subject`` when the source carries it.
    """
    out = {"question_id": str(rec["question_id"]), "variant": rec["variant"],
           "demographic": rec["demographic"], "condition": cond, "model": model,
           "answer_idx": rec["answer_idx"]}
    if rec.get("subject") is not None:
        out["subject"] = rec["subject"]
    return out


# --------------------------------------------------------------------------- #
# Local models
# --------------------------------------------------------------------------- #

def pick_device():
    """Best available device and dtype.

    bfloat16 is preferred where supported: Gemma-family models are known to
    overflow in float16.

    Returns
    -------
    tuple of (str, torch.dtype)
        Device string and dtype.
    """
    import torch
    if torch.cuda.is_available():
        # Native bf16 needs compute capability >= 8 (A100, L4, H100). Older
        # cards such as the T4 may report support through slow emulation.
        native = torch.cuda.get_device_capability(0)[0] >= 8
        return "cuda", (torch.bfloat16 if native else torch.float16)
    if torch.backends.mps.is_available():
        try:
            (torch.ones(2, dtype=torch.bfloat16, device="mps") * 2).sum().item()
            return "mps", torch.bfloat16
        except Exception:
            return "mps", torch.float16
    return "cpu", torch.float32


def load_local(name: str, device: str, dtype):
    """Load a tokenizer and model, falling back to the image-text class.

    Some medical models (e.g. MedGemma-4B) are published as image-text models
    that the causal-LM auto class does not map; text-only prompting works
    through the image-text class.

    Parameters
    ----------
    name : str
        Hugging Face model id.
    device : str
        Target device.
    dtype : torch.dtype
        Weight dtype.

    Returns
    -------
    tuple of (tokenizer, model)
        Ready for inference, tokenizer set to left padding.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    kw = {_dtype_kwarg(): dtype, "device_map": device}
    try:
        model = AutoModelForCausalLM.from_pretrained(name, **kw)
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(name, **kw)
    return tok, model.eval()


def _dtype_kwarg() -> str:
    """Name of the dtype argument for the installed transformers version.

    transformers 4.56 renamed ``torch_dtype`` to ``dtype``; an older install
    would silently ignore ``dtype`` and load full-precision weights.

    Returns
    -------
    str
        ``"dtype"`` or ``"torch_dtype"``.
    """
    import transformers
    from packaging.version import Version
    return "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"


def gpu_report(model_names=()) -> None:
    """Print the GPU and warn about configurations that will fail or mislead.

    Parameters
    ----------
    model_names : iterable of str
        Models about to be run, used to target the warnings.

    Returns
    -------
    None
    """
    import torch
    if not torch.cuda.is_available():
        print("no CUDA GPU; running on", pick_device()[0])
        return
    p = torch.cuda.get_device_properties(0)
    mem, cc = p.total_memory / 1e9, p.major
    print(f"GPU {p.name}: {mem:.0f} GB, compute capability {p.major}.{p.minor}")
    names = " ".join(model_names).lower()
    if mem < 20 and any(k in names for k in ("7b", "8b", "9b")):
        print("  WARNING: 7-8B models in 16-bit need ~17 GB; this GPU will run out "
              "of memory. Use an L4 or A100 runtime. Quantising instead would "
              "change the confidence scores being measured.")
    if cc < 8 and "gemma" in names:
        print("  WARNING: no native bf16 on this GPU; Gemma-family models can "
              "overflow in fp16. Use an L4 or A100 runtime.")


# Models that open every reply with a reasoning trace whatever the template
# says. Their turn is pre-filled with an empty trace so the answer comes first,
# matching the other models (as enable_thinking=False does for Qwen-style
# templates). Keyed by a substring of the model id.
NO_THINK_PREFIX = {
    "medgemma-1.5": "<unused94>thought\n<unused95>",
}


def _no_think_prefix(tok) -> str:
    name = str(getattr(tok, "name_or_path", "")).lower()
    return next((p for k, p in NO_THINK_PREFIX.items() if k in name), "")


def local_provenance(tok, method: str) -> dict:
    """How a local record is produced, stored with it so that records made by
    older code (no reasoning pre-fill, single precision) are run again.

    Returns
    -------
    dict
        Fields written into, and required of, every record of this method.
    """
    how = {f"{method}_think_prefix": _no_think_prefix(tok)}
    if method == "tlp":
        how["tlp_precision"] = "fp64"
    return how


def chat_text(tok, system: str, user: str) -> str:
    """Render a system + user turn with the model's chat template.

    Falls back, in order, to folding the system text into the user turn (some
    templates reject a system role), to list-of-parts content (image-text
    templates), and finally to plain concatenation for models without a
    template.

    Parameters
    ----------
    tok : tokenizer
        Model tokenizer.
    system, user : str
        Turn contents.

    Returns
    -------
    str
        Prompt text ending where the assistant's reply begins.
    """
    attempts = [
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        [{"role": "user", "content": system + "\n\n" + user}],
        [{"role": "user", "content": [{"type": "text", "text": system + "\n\n" + user}]}],
    ]
    if getattr(tok, "chat_template", None):
        for msgs in attempts:
            try:
                # enable_thinking=False: thinking models otherwise open with a
                # reasoning block, so the first token is not an option letter
                # and TLP is meaningless. Templates without the switch ignore it.
                return tok.apply_chat_template(msgs, add_generation_prompt=True,
                                               tokenize=False, enable_thinking=False) \
                    + _no_think_prefix(tok)
            except Exception:
                continue
    return f"{system}\n\n{user}\n\nAnswer: "


def letter_token_ids(tok, letters=LETTERS) -> dict:
    """Single-token encodings of each option letter.

    Parameters
    ----------
    tok : tokenizer
        Model tokenizer.
    letters : sequence of str
        Option letters.

    Returns
    -------
    dict
        Letter to sorted list of token ids.

    Raises
    ------
    AssertionError
        If a letter has no single-token encoding.
    """
    table = {}
    for L in letters:
        ids = set()
        for form in (L, " " + L, "▁" + L):
            enc = tok.encode(form, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(enc[0])
        assert ids, f"option letter {L!r} has no single-token encoding"
        table[L] = sorted(ids)
    return table


def tlp_batch(model, tok, tok_ids: dict, texts: list, letters) -> list:
    """Token log-probability confidence for a batch of prompts.

    Parameters
    ----------
    model, tok
        Loaded model and left-padding tokenizer.
    tok_ids : dict
        Output of :func:`letter_token_ids`.
    texts : list of str
        Rendered prompts.
    letters : sequence of str
        Valid option letters for these items.

    Returns
    -------
    list of dict
        Per item: ``probs`` (renormalised over letters), ``pred``, ``conf``,
        and ``letter_mass`` (unnormalised probability on any option letter).

    Raises
    ------
    FloatingPointError
        If the logits contain NaN or inf, which usually means the dtype is
        unsuitable for this model.
    """
    import torch
    enc = tok(texts, return_tensors="pt", padding=True,
              add_special_tokens=False).to(model.device)
    # Explicit positions: without them a left-padded sequence is shifted and
    # its logits differ from the unbatched run.
    pos = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
    with torch.no_grad():
        try:
            logits = model(**enc, position_ids=pos).logits[:, -1, :].float()
        except TypeError:
            logits = model(**enc).logits[:, -1, :].float()
    if torch.isnan(logits).any():
        raise FloatingPointError("NaN logits; the dtype is probably unsuitable "
                                 "for this model (try an L4/A100 runtime or float32)")
    # double precision: confident answers sit within 1e-7 of p = 1, where float32
    # cannot tell them apart, and the referral threshold ranks exactly there
    lp = torch.log_softmax(logits.cpu().double(), dim=-1)      # MPS has no float64
    cols = torch.stack([lp[:, tok_ids[L]].max(dim=1).values for L in letters], dim=1)
    mass = cols.exp().sum(dim=1)
    probs = torch.softmax(cols, dim=1)
    out = []
    for i in range(len(texts)):
        p = {L: float(probs[i, j]) for j, L in enumerate(letters)}
        pred = max(p, key=p.get)
        out.append({"probs": p, "pred": pred, "conf": p[pred],
                    "letter_mass": float(mass[i])})
    return out


def top_next_tokens(model, tok, text: str, k: int = 8) -> list:
    """The model's most likely next tokens after a prompt, for diagnosing low
    letter mass (what does it want to write instead of an option letter?).

    Returns
    -------
    list of tuple
        ``(token string, probability)``, most likely first.
    """
    import torch
    enc = tok([text], return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        p = torch.softmax(model(**enc).logits[0, -1].float(), -1)
    v, i = p.topk(k)
    return [(tok.decode([int(t)]), round(float(x), 3)) for x, t in zip(v, i)]


def run_local_tlp(model, tok, model_name: str, model_dir: str, pairs: dict,
                  n_options: dict, system_prompts: dict, conditions, results_root,
                  batch_size: int = 8, log_every: int = 400):
    """Run TLP for every dataset and condition, resuming where it stopped.

    Items are length-sorted within each batch run to minimise padding.

    Parameters
    ----------
    model, tok
        Loaded model and tokenizer.
    model_name, model_dir : str
        Identifier stored in records, and results directory name.
    pairs : dict
        Dataset name to list of pair records.
    n_options : dict
        Dataset name to number of options.
    system_prompts : dict
        Condition to system text.
    conditions : sequence of str
        Conditions to run.
    results_root : str or Path
        Results root.
    batch_size : int
        Items per forward pass.
    log_every : int
        Progress interval.

    Returns
    -------
    None
    """
    tok_ids = letter_token_ids(tok)
    how = local_provenance(tok, "tlp")
    for ds, recs in pairs.items():
        letters = LETTERS[:n_options[ds]]
        for cond in conditions:
            path = out_path(results_root, ds, model_dir, cond)
            seen = done_keys(path, "tlp", require=how)
            todo = [r for r in recs if (str(r["question_id"]), r["variant"]) not in seen]
            if not todo:
                print(f"  {ds}/{cond}: complete")
                continue
            texts = [chat_text(tok, system_prompts[cond], user_prompt(r)) for r in todo]
            order = sorted(range(len(todo)), key=lambda i: len(texts[i]))
            print(f"  {ds}/{cond}: {len(todo)} to run")
            t0, n = time.time(), 0
            with open_append(path) as fh:
                for b in range(0, len(order), batch_size):
                    idx = order[b:b + batch_size]
                    res = tlp_batch(model, tok, tok_ids, [texts[i] for i in idx], letters)
                    for i, r in zip(idx, res):
                        rec = todo[i]
                        fh.write(json.dumps({
                            **base_record(rec, cond, model_name),
                            "tlp_answer": r["pred"], "tlp_conf": r["conf"],
                            "tlp_correct": int(r["pred"] == rec["answer_idx"]),
                            "tlp_option_probs": json.dumps(r["probs"]),
                            "tlp_letter_mass": r["letter_mass"], **how}) + "\n")
                    n += len(idx)
                    if n % log_every < batch_size:
                        fh.flush()
                        rate = (time.time() - t0) / n
                        print(f"    {n}/{len(todo)}  {rate:.3f}s/item  "
                              f"~{rate * (len(todo) - n) / 60:.0f} min left")


def ce_batch(model, tok, texts: list, letters, max_new_tokens: int = 24) -> list:
    """Elicited confidence for a batch of prompts, by greedy generation.

    Parameters
    ----------
    model, tok
        Loaded model and left-padding tokenizer.
    texts : list of str
        Rendered prompts built with the CE instruction.
    letters : sequence of str
        Valid option letters.
    max_new_tokens : int
        Output budget; "Answer=[B]\nConfidence=[85]" needs about 15 tokens.

    Returns
    -------
    list of dict
        Per item: ``answer``, ``conf`` (both None if unparsable) and ``raw``.
    """
    import torch
    enc = tok(texts, return_tensors="pt", padding=True,
              add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    cut = enc["input_ids"].shape[1]
    res = []
    for row in out:
        raw = tok.decode(row[cut:], skip_special_tokens=True).strip()
        a, c = parse_ce(raw, letters)
        res.append({"answer": a, "conf": c, "raw": raw})
    return res


def run_local_ce(model, tok, model_name: str, model_dir: str, pairs: dict,
                 n_options: dict, system_prompts: dict, conditions, results_root,
                 batch_size: int = 8, log_every: int = 400):
    """Run elicited confidence for every dataset and condition, resuming.

    Records are appended to the same cache files as TLP; the ``ce_*`` fields
    sit alongside the ``tlp_*`` ones. Unparsable replies are recorded with
    ``ce_parse_ok = False`` so they count as done and their rate can be checked
    per group.

    Parameters
    ----------
    model, tok
        Loaded model and tokenizer.
    model_name, model_dir : str
        Identifier stored in records, and results directory name.
    pairs, n_options, system_prompts, conditions, results_root
        As for :func:`run_local_tlp`.
    batch_size : int
        Items per generate call.
    log_every : int
        Progress interval.

    Returns
    -------
    None
    """
    how = local_provenance(tok, "ce")
    for ds, recs in pairs.items():
        letters = LETTERS[:n_options[ds]]
        for cond in conditions:
            path = out_path(results_root, ds, model_dir, cond)
            seen = done_keys(path, "ce", require=how)
            todo = [r for r in recs if (str(r["question_id"]), r["variant"]) not in seen]
            if not todo:
                print(f"  {ds}/{cond}/ce: complete")
                continue
            texts = [chat_text(tok, system_prompts[cond], user_prompt(r, "ce")) for r in todo]
            order = sorted(range(len(todo)), key=lambda i: len(texts[i]))
            print(f"  {ds}/{cond}/ce: {len(todo)} to run")
            t0, n = time.time(), 0
            with open_append(path) as fh:
                for b in range(0, len(order), batch_size):
                    idx = order[b:b + batch_size]
                    res = ce_batch(model, tok, [texts[i] for i in idx], letters)
                    for i, r in zip(idx, res):
                        rec = todo[i]
                        fh.write(json.dumps({
                            **base_record(rec, cond, model_name),
                            "ce_answer": r["answer"], "ce_conf": r["conf"],
                            "ce_correct": int(r["answer"] == rec["answer_idx"]) if r["answer"] else 0,
                            "ce_parse_ok": r["answer"] is not None,
                            "ce_raw_text": r["raw"], "ce_version": CE_VERSION, **how}) + "\n")
                    n += len(idx)
                    if n % log_every < batch_size:
                        fh.flush()
                        rate = (time.time() - t0) / n
                        print(f"    {n}/{len(todo)}  {rate:.3f}s/item  "
                              f"~{rate * (len(todo) - n) / 60:.0f} min left")


# --------------------------------------------------------------------------- #
# API models
# --------------------------------------------------------------------------- #

TRANSIENT_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}
TRANSIENT_TYPES = {"RateLimitError", "APITimeoutError", "APIConnectionError",
                   "InternalServerError", "OverloadedError", "ServiceUnavailableError"}
PARAM_WORDS = ("max_tokens", "max_completion_tokens", "max_output_tokens", "temperature",
               "reasoning", "thinking", "logprobs", "top_logprobs", "unsupported", "not supported",
               "does not support", "unrecognized", "unexpected keyword",
               "responses", "chat/completions")


def _is_transient(e) -> bool:
    """Whether an API error is worth retrying (rate limit, timeout, server error).

    Parameters
    ----------
    e : Exception
        The error.

    Returns
    -------
    bool
        True for transient failures; False for errors that will recur, such as
        a wrong model id or an invalid key, which must fail fast rather than
        burn retries on every item.
    """
    msg = str(e).lower()
    # An exhausted balance arrives as HTTP 429 but will not clear by waiting.
    if any(w in msg for w in ("insufficient_quota", "exceeded your current quota",
                              "credit balance", "billing")):
        return False
    return (getattr(e, "status_code", None) in TRANSIENT_STATUS
            or type(e).__name__ in TRANSIENT_TYPES)


class BudgetExhausted(Exception):
    """The reply was cut off before any answer text (hidden reasoning used up
    the output budget). A request-style problem, not a model answer."""


def _is_param_error(e) -> bool:
    """Whether an error says one of our request parameters was rejected.

    Parameters
    ----------
    e : Exception
        The error.

    Returns
    -------
    bool
        True if a different parameter style should be tried.
    """
    if isinstance(e, BudgetExhausted):
        return True
    if _is_transient(e):
        return False
    msg = str(e).lower()
    # "use the Responses API" is a routing instruction whatever the error class
    if "v1/responses" in msg or "responses api" in msg or "chat/completions" in msg:
        return True
    return (isinstance(e, TypeError) or getattr(e, "status_code", None) in (400, 404, 422)
            or type(e).__name__ == "BadRequestError") and any(w in msg for w in PARAM_WORDS)


class APIModel:
    """One API model behind a uniform generate() call.

    Providers and model generations disagree on request parameters: newer
    OpenAI reasoning models reject ``temperature`` and ``max_tokens``, some are
    served only on the Responses API, and a Claude model may reject
    ``temperature``. The first request *style* the model accepts is found once
    and reused; the values in it (output budget, temperature) are rebuilt on
    every call, so a style found during a one-token TLP call does not truncate
    later CE replies or pin self-consistency to temperature 0.

    Only transient errors are retried. Permanent ones (wrong model id, invalid
    key) fail immediately, and :func:`run_api` stops a run whose first calls all
    fail.

    Parameters
    ----------
    backend : {"openai", "anthropic"}
        Provider.
    model : str
        Model identifier.
    """

    def __init__(self, backend: str, model: str):
        self.backend, self.model = backend, model
        self.style = None          # index of the accepted request style
        self.params = None         # parameters of the most recent successful call
        self.resolved_model = None
        self.last_error = None
        if backend == "openai":
            from openai import OpenAI
            self.client = OpenAI()
        elif backend == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic()
        else:
            raise ValueError(backend)

    def _styles(self):
        """Candidate request styles, most conventional first.

        Returns
        -------
        list
            Callables ``(max_tokens, temperature) -> dict``, or ``"responses"``.
        """
        if self.backend == "anthropic":
            # Thinking models spend output tokens before the answer, so a small
            # budget returns nothing. Prefer thinking off (same footing as the
            # other models); otherwise give it room.
            off = {"type": "disabled"}
            return [lambda mt, t: {"max_tokens": mt, "temperature": t, "thinking": off},
                    lambda mt, t: {"max_tokens": mt, "thinking": off},
                    lambda mt, t: {"max_tokens": mt, "temperature": t},
                    lambda mt, t: {"max_tokens": max(mt, 2048)}]
        big = lambda mt: max(mt, 2048)          # room for hidden reasoning
        return [lambda mt, t: {"max_tokens": mt, "temperature": t},
                lambda mt, t: {"max_tokens": mt},
                lambda mt, t: {"max_completion_tokens": big(mt), "temperature": t},
                lambda mt, t: {"max_completion_tokens": big(mt), "reasoning_effort": "minimal"},
                lambda mt, t: {"max_completion_tokens": big(mt), "reasoning_effort": "low"},
                lambda mt, t: {"max_completion_tokens": big(mt)},
                "responses"]

    def _invoke(self, style, system, user, n, max_tokens, temperature, logprobs):
        """Issue one request in the given style.

        Returns
        -------
        tuple of (list of str, list or None, dict)
            Completions, first-token top log-probabilities, parameters used.
        """
        if style == "responses":
            p = {"api": "responses", "max_output_tokens": max(max_tokens, 2048)}
            outs = []
            for _ in range(n):
                r = self.client.responses.create(model=self.model, instructions=system,
                                                 input=user,
                                                 max_output_tokens=p["max_output_tokens"])
                self.resolved_model = getattr(r, "model", self.model)
                outs.append(getattr(r, "output_text", "") or "")
            return outs, None, p
        p = style(max_tokens, temperature)
        if self.backend == "anthropic":
            outs = []
            for _ in range(n):
                r = self.client.messages.create(model=self.model, system=system,
                                                messages=[{"role": "user", "content": user}],
                                                **p)
                self.resolved_model = getattr(r, "model", self.model)
                text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
                if not text.strip() and getattr(r, "stop_reason", None) == "max_tokens":
                    raise BudgetExhausted(f"empty reply at max_tokens={p['max_tokens']}")
                outs.append(text)
            return outs, None, p
        extra = {"logprobs": True, "top_logprobs": 20} if logprobs else {}
        r = self.client.chat.completions.create(
            model=self.model, n=n, **p, **extra,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}])
        self.resolved_model = getattr(r, "model", self.model)
        outs = [c.message.content or "" for c in r.choices]
        if not any(o.strip() for o in outs) and not logprobs and \
                any(getattr(c, "finish_reason", None) == "length" for c in r.choices):
            raise BudgetExhausted(f"empty reply with {p}")
        top = None
        if logprobs and r.choices and r.choices[0].logprobs and r.choices[0].logprobs.content:
            top = r.choices[0].logprobs.content[0].top_logprobs
        return outs, top, p

    def _call(self, system, user, n, max_tokens, temperature, logprobs):
        """One request, negotiating the style on the first call.

        Raises
        ------
        Exception
            The last error if no style is accepted, or any non-parameter error.
        """
        styles = self._styles()
        order = list(range(len(styles)))
        if self.style is not None:
            order.remove(self.style)
            order.insert(0, self.style)
        last = None
        for i in order:
            try:
                outs, top, p = self._invoke(styles[i], system, user, n, max_tokens,
                                            temperature, logprobs)
                self.style, self.params = i, p
                return outs, top
            except Exception as e:
                if _is_param_error(e):
                    last = e
                    continue
                raise
        raise last or RuntimeError("no request style accepted")

    def generate(self, system: str, user: str, n: int = 1, temperature: float = 0.0,
                 max_tokens: int = 32, logprobs: bool = False, tries: int = 5):
        """Return completions (and first-token logprobs if requested).

        Parameters
        ----------
        system, user : str
            Turn contents.
        n : int
            Number of completions.
        temperature : float
            Requested temperature; dropped for models with a fixed one.
        max_tokens : int
            Output budget; raised for reasoning models.
        logprobs : bool
            Request first-token top log-probabilities (OpenAI only).
        tries : int
            Attempts for transient errors.

        Returns
        -------
        tuple of (list of str, list or None)
            Completions and first-token ``top_logprobs``; ``([], None)`` on
            failure, with the error kept in ``last_error``.
        """
        for attempt in range(tries):
            try:
                out = self._call(system, user, n, max_tokens, temperature, logprobs)
                self.last_error = None
                return out
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {str(e)[:200]}"
                if not _is_transient(e) or attempt == tries - 1:
                    return [], None
                time.sleep(min(2 ** attempt, 20) + random.random())
        return [], None

    def supports_logprobs(self) -> tuple:
        """Probe whether first-token log-probabilities come back.

        Returns
        -------
        tuple of (bool, str)
            Availability and a short diagnostic.
        """
        if self.backend != "openai":
            return False, "API returns no token log-probabilities"
        outs, top = self.generate("You answer with one letter.",
                                  "Answer with one letter: A, B, or C?",
                                  max_tokens=1, logprobs=True, tries=2)
        if top:
            return True, f"{len(top)} candidates, e.g. {top[0].token!r} {top[0].logprob:.2f}"
        if outs:
            return False, "call succeeded but no logprobs returned"
        return False, f"call failed: {self.last_error}"


def _openai_tlp(top, letters):
    """Renormalised option-letter distribution from ``top_logprobs``.

    Parameters
    ----------
    top : list
        First-token top log-probabilities.
    letters : sequence of str
        Valid option letters.

    Returns
    -------
    tuple of (dict or None, float)
        Probabilities over letters (None if no letter appears) and the
        unnormalised mass on letters.
    """
    lp = {}
    for t in top or []:
        s = t.token.strip().upper()
        if s in letters:
            lp[s] = max(lp.get(s, -1e9), t.logprob)
    if not lp:
        return None, 0.0
    mass = sum(math.exp(v) for v in lp.values())
    floor = min(lp.values()) - 10.0             # letters outside the top-20
    v = [lp.get(L, floor) for L in letters]
    m = max(v)
    e = [math.exp(x - m) for x in v]
    Z = sum(e)
    return {L: x / Z for L, x in zip(letters, e)}, mass


def run_api(api: APIModel, model_dir: str, pairs: dict, n_options: dict,
            system_prompts: dict, conditions, methods, results_root,
            max_workers: int = 8, log_every: int = 400, fail_fast_after: int = 20):
    """Run API inference concurrently, resuming where it stopped.

    If the first ``fail_fast_after`` calls of a batch all fail -- a wrong model
    id, an invalid key, a parameter the model rejects -- the run stops with the
    error instead of spending hours on thousands of doomed requests.

    Parameters
    ----------
    api : APIModel
        Model wrapper.
    model_dir : str
        Results directory name.
    pairs, n_options, system_prompts
        As for :func:`run_local_tlp`.
    conditions, methods : sequence of str
        Conditions and confidence methods to run.
    results_root : str or Path
        Results root.
    max_workers : int
        Concurrent requests.
    log_every : int
        Progress interval.
    fail_fast_after : int
        Consecutive failures at the start of a batch that abort the run.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If every one of the first ``fail_fast_after`` calls fails.
    """
    lock = threading.Lock()
    for ds, recs in pairs.items():
        letters = LETTERS[:n_options[ds]]
        for cond in conditions:
            path = out_path(results_root, ds, model_dir, cond)
            for method in methods:
                items = recs
                seen = done_keys(path, method)
                todo = [r for r in items if (str(r["question_id"]), r["variant"]) not in seen]
                if not todo:
                    print(f"  {ds}/{cond}/{method}: complete")
                    continue
                print(f"  {ds}/{cond}/{method}: {len(todo)} to run")
                stop = threading.Event()
                state = {"ok": 0, "fail": 0, "done": 0, "streak": 0}

                def work(rec):
                    if stop.is_set():
                        return "skipped"
                    try:
                        row = compute(rec)
                    except Exception as e:       # malformed response: skip, never crash
                        api.last_error = f"{type(e).__name__}: {str(e)[:200]}"
                        row = "failed"
                    # Counted in the worker, so the stop takes effect for work
                    # still queued even when calls fail faster than results are
                    # collected.
                    with lock:
                        if row == "failed":
                            state["fail"] += 1
                            state["streak"] += 1
                            # All early calls failed, or a long unbroken run of
                            # failures later (quota or credit exhausted mid-run).
                            if (state["ok"] == 0 and state["fail"] >= fail_fast_after) \
                                    or state["streak"] >= 5 * fail_fast_after:
                                stop.set()
                        elif isinstance(row, dict):
                            state["ok"] += 1
                            state["streak"] = 0
                    return row

                def compute(rec):
                    sysmsg = system_prompts[cond]
                    if method == "tlp":
                        outs, top = api.generate(sysmsg, user_prompt(rec, "tlp"),
                                                 max_tokens=1, logprobs=True)
                        if not outs and top is None:
                            return "failed"
                        p, mass = _openai_tlp(top, letters)
                        if p is None:
                            # Recorded, not dropped: the case has no usable
                            # confidence and is therefore referred.
                            return {"tlp_answer": None, "tlp_conf": None, "tlp_correct": 0,
                                    "tlp_option_probs": None, "tlp_parse_ok": False,
                                    "tlp_letter_mass": 0.0, "tlp_raw_text": outs[0] if outs else ""}
                        pred = max(p, key=p.get)
                        return {"tlp_answer": pred, "tlp_conf": p[pred],
                                "tlp_correct": int(pred == rec["answer_idx"]),
                                "tlp_option_probs": json.dumps(p),
                                "tlp_letter_mass": mass}
                    if method == "ce":
                        outs, _ = api.generate(sysmsg, user_prompt(rec, "ce"),
                                               max_tokens=CE_API_MAX_TOKENS)
                        if not outs or not outs[0].strip():
                            return "failed"
                        a, c = parse_ce(outs[0], letters)
                        return {"ce_answer": a, "ce_conf": c,
                                "ce_correct": int(a == rec["answer_idx"]) if a else 0,
                                "ce_parse_ok": a is not None, "ce_raw_text": outs[0],
                                "ce_version": CE_VERSION, "ce_max_tokens": CE_API_MAX_TOKENS}
                    raise ValueError(f"unknown confidence method {method!r}")

                t0 = time.time()
                with open_append(path) as fh, \
                        ThreadPoolExecutor(max_workers=max_workers) as ex:
                    futs = {ex.submit(work, r): r for r in todo}
                    for fut in as_completed(futs):
                        rec, row = futs[fut], fut.result()
                        with lock:
                            state["done"] += 1
                            if not isinstance(row, dict):
                                continue
                            fh.write(json.dumps({
                                **base_record(rec, cond, api.model), **row,
                                "api_params": json.dumps(api.params),
                                "resolved_model": api.resolved_model}) + "\n")
                            if state["done"] % log_every == 0:
                                fh.flush()
                                rate = (time.time() - t0) / state["done"]
                                print(f"    {state['done']}/{len(todo)}  ok={state['ok']} "
                                      f"failed={state['fail']}  "
                                      f"~{rate * (len(todo) - state['done']) / 60:.0f} min left")
                if stop.is_set():
                    raise RuntimeError(
                        f"{api.model}: stopped after {state['fail']} failed calls "
                        f"({state['ok']} succeeded); re-run to resume. "
                        f"Last error: {api.last_error}")
                if state["fail"]:
                    print(f"    {state['fail']} call(s) failed; re-run to retry them. "
                          f"Last error: {api.last_error}")


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #

def sha256_file(path) -> str:
    """SHA-256 of a file's bytes.

    Parameters
    ----------
    path : str or Path
        File to hash.

    Returns
    -------
    str
        Hex digest.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(path, pair_files: dict, system_prompts: dict, models: dict,
                   extra: dict | None = None) -> str:
    """Record what was run, with hashes, so the results are reproducible.

    Parameters
    ----------
    path : str or Path
        Destination JSON.
    pair_files : dict
        Dataset name to pair-file path.
    system_prompts : dict
        Condition to system text.
    models : dict
        Model name to metadata (backend, resolved id, parameters used).
    extra : dict, optional
        Any additional fields.

    Returns
    -------
    str
        The path written.
    """
    import importlib.metadata as md

    def ver(pkg):
        try:
            return md.version(pkg)
        except md.PackageNotFoundError:
            return None

    info = {
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {p: ver(p) for p in ("torch", "transformers", "openai",
                                         "anthropic", "numpy", "pandas", "scipy")},
        "pair_files": {k: {"path": str(v), "sha256": sha256_file(v)}
                       for k, v in pair_files.items()},
        "system_prompts": {k: {"sha256": hashlib.sha256(v.encode()).hexdigest(),
                               "text": v} for k, v in system_prompts.items()},
        "models": models,
        **(extra or {}),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(info, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    return str(path)
