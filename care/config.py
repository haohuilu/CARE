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
