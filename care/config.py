"""Paths, constants and registries.

Every path can be overridden with an environment variable, so the same code runs
locally, on Colab (with results on Google Drive), or on a server.
"""

from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Source corpora (MedQA, MedMCQA) and the matched pairs built from them.
SOURCE_DIR = os.environ.get("CARE_SOURCE_DIR", os.path.join(ROOT, "data", "source"))
PAIRS_DIR = os.environ.get("CARE_PAIRS_DIR", os.path.join(ROOT, "data", "pairs"))

# Cached model responses: <DATA_ROOT>/<dataset>/<model>/calibration_cache/<prompt>.jsonl
DATA_ROOT = os.environ.get("CARE_DATA_ROOT", os.path.join(ROOT, "results", "inference"))

# Analysis outputs.
OUTPUT_ROOT = os.environ.get("CARE_OUTPUT_ROOT", os.path.join(ROOT, "results"))
RESULTS_DIR = os.path.join(OUTPUT_ROOT, "csv")
FIGS_DIR = os.path.join(OUTPUT_ROOT, "figs")
TABLES_DIR = os.path.join(OUTPUT_ROOT, "tables")

SEED = 42
N_BOOT = 1000          # paired bootstrap resamples
MIN_ACCEPTED = 10      # smallest accepted set for which a budget counts as met

# The four matched-pair sets. ref/dem: reference and demographic group.
DATASETS = {
    "pairs_medqa_race":     {"paired": True, "n_options": 5, "groups": ("white", "black"),
                             "ref": "white", "dem": "black"},
    "pairs_medqa_gender":   {"paired": True, "n_options": 5, "groups": ("male", "female"),
                             "ref": "male", "dem": "female"},
    "pairs_medmcqa_race":   {"paired": True, "n_options": 4, "groups": ("white", "black"),
                             "ref": "white", "dem": "black"},
    "pairs_medmcqa_gender": {"paired": True, "n_options": 4, "groups": ("male", "female"),
                             "ref": "male", "dem": "female"},
}

# The four sets analysed in the submitted paper. The paper's tables and figures
# are always built from these, whatever else is registered below.
MAIN_DATASETS = tuple(DATASETS)

# --------------------------------------------------------------------------- #
# Revision (R1) control sets, built by ``python -m care.controls``
# --------------------------------------------------------------------------- #
# ``parent`` is the demographic set a control is read against; ``kind`` says what
# the control isolates.
PLACEBO_TERMS = ("right-handed", "left-handed")     # reference first
for _src, _k in (("medqa", 5), ("medmcqa", 4)):
    for _axis, _groups in (("race", ("white", "black")), ("gender", ("male", "female"))):
        # every pair the rules can build, not a sample of 1,000
        DATASETS[f"pairs_{_src}_{_axis}_full"] = {
            "paired": True, "n_options": _k, "groups": _groups,
            "ref": _groups[0], "dem": _groups[1],
            "kind": "full", "parent": f"pairs_{_src}_{_axis}"}
    # a clinically irrelevant descriptor at the race insertion point
    DATASETS[f"pairs_{_src}_placebo"] = {
        "paired": True, "n_options": _k, "groups": PLACEBO_TERMS,
        "ref": PLACEBO_TERMS[0], "dem": PLACEBO_TERMS[1],
        "kind": "placebo", "parent": f"pairs_{_src}_race"}
    DATASETS[f"pairs_{_src}_placebo_full"] = {
        **DATASETS[f"pairs_{_src}_placebo"], "kind": "placebo_full",
        "parent": f"pairs_{_src}_race_full"}
    # sex pairs in which BOTH versions carry an inserted neutral word, so
    # neither is the verbatim benchmark text
    DATASETS[f"pairs_{_src}_gender_edited"] = {
        "paired": True, "n_options": _k, "groups": ("male", "female"),
        "ref": "male", "dem": "female",
        "kind": "both_edited", "parent": f"pairs_{_src}_gender"}
    # sex pairs built from female-coded originals: the FEMALE version is the
    # verbatim benchmark text and the male version is the substituted one
    DATASETS[f"pairs_{_src}_gender_reverse"] = {
        "paired": True, "n_options": _k, "groups": ("male", "female"),
        "ref": "male", "dem": "female",
        "kind": "reverse", "parent": f"pairs_{_src}_gender"}
CONTROL_DATASETS = tuple(d for d in DATASETS if d not in MAIN_DATASETS)


def source_of(dataset: str) -> str:
    """Source corpus of a pair set (``"medqa"`` or ``"medmcqa"``)."""
    return dataset.split("_")[1]


def axis_of(dataset: str) -> str:
    """Attribute a pair set varies: ``race``, ``gender`` or ``placebo``."""
    return dataset.split("_")[2]


# Models in the paper, by results directory name -> short label.
LOCAL_MODELS = ["google_medgemma-4b-it", "m42-health_Llama3-Med42-8B",
                "google_gemma-4-E4B-it", "meta-llama_Llama-3.1-8B-Instruct"]
API_MODELS = ["gpt-6-sol", "gpt-5.6-terra", "gpt-4.1-mini-2025-04-14", "claude-sonnet-5"]
MODELS = {
    "google_medgemma-4b-it": "MedGemma-4B",
    "m42-health_Llama3-Med42-8B": "Med42-8B",
    "google_gemma-4-E4B-it": "Gemma-4-E4B",
    "meta-llama_Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "gpt-6-sol": "GPT-6-Sol",
    "gpt-5.6-terra": "GPT-5.6-Terra",
    "claude-sonnet-5": "Claude-Sonnet-5",
    "gpt-4.1-mini-2025-04-14": "GPT-4.1-mini",
}


def short(model: str) -> str:
    """Short label for a model directory name (the name itself if unregistered)."""
    return MODELS.get(model, model)
