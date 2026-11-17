# Safe automation and demographic fairness in LLM referral systems

Code for the paper *Who gets referred? Demographic asymmetry in the safe
automation of medical language model referral systems*.

A referral system accepts a language model's answer to a clinical question when
the model is sufficiently confident, and refers the case to a clinician
otherwise. For an error budget α, the **safe automation rate (SAR)** of a
demographic group is the largest share of its cases that a confidence threshold
can accept while keeping the error rate among accepted cases at or below α. This
repository builds matched demographic pairs from MedQA and MedMCQA, runs the
models, and computes SAR per group, the effect of fairness-aware prompts, and
the out-of-sample checks reported in the paper.

## Repository layout

```
care/
  config.py      paths (overridable by environment variables), seeds, datasets, model names
  prompts.py     the three system prompts (P1 bare, P2 bias-aware, P3 equity-aware)
  pairgen.py     rule-based construction of race and sex minimal pairs
  infer.py       inference: local models (token probabilities, elicited confidence) and APIs
  data.py        loading and validation of cached responses
  analysis.py    SAR, paired bootstrap, prompt contrasts, cross-fitting, group-blind threshold
  figures.py     figures in the paper
  tables.py      LaTeX tables in the paper
notebooks/
  colab_local_models.ipynb   open-weight models on a Colab GPU, then the full analysis
  api_models.ipynb           commercial models, run on a laptop
tests/
  test_core.py   pytest suite (no network, no model calls)
```

## Setup

```bash
git clone https://github.com/haohuilu/CARE.git
cd CARE
pip install -r requirements.txt
pytest
```

Two tests need the source corpora and are skipped when they are absent.

## Data

The data are not included. Place the source corpora under `data/source/` (or
point `CARE_SOURCE_DIR` elsewhere):

```
data/source/MedQA/{train,dev,test}.json       MedQA, US English, 5 options (MIT licence)
data/source/medmcqa/{train,dev,test}.json     MedMCQA, JSON lines as released on Hugging Face
```

`MedQA/*.json` holds `{"data": [...]}` with fields `id`, `question` (options
inline as `A: ... B: ...`) and `answer`, as in the widely used reformatting of
the US subset of [MedQA](https://github.com/jind11/MedQA). MedMCQA files are the
[openlifescienceai/medmcqa](https://huggingface.co/datasets/openlifescienceai/medmcqa)
splits with fields `id`, `question`, `opa`–`opd`, `cop` and `subject_name`.

Build the matched pairs (1,000 per dataset and attribute, fixed seed):

```bash
python -m care.pairgen
```

The pair files used in the paper have these SHA-256 hashes (first 16 characters):

| File | SHA-256 |
|---|---|
| `pairs_medqa_race.jsonl` | `d32c84414db706f4` |
| `pairs_medqa_gender.jsonl` | `084ad9446b00e11d` |
| `pairs_medmcqa_race.jsonl` | `b2058037d8772b85` |
| `pairs_medmcqa_gender.jsonl` | `e12d901caa9ff4a7` |

## Running the study

1. **Open-weight models** — `notebooks/colab_local_models.ipynb` on a Colab L4 or
   A100 GPU. Add an `HF_TOKEN` secret; the first cell clones this repository.
   Source data, pairs, cached responses and outputs are kept on Google Drive, and
   interrupted runs resume where they stopped.
2. **Commercial models** — `notebooks/api_models.ipynb` on a laptop. API keys are
   requested at run time and never written to disk. Upload the resulting
   `api_results.zip` to the same Drive folder.
3. **Analysis** — the last sections of the Colab notebook, or from a shell once the
   cached responses are in place:

```bash
python -m care.analysis   # SAR, contrasts and checks  -> results/csv/
python -m care.tables     # LaTeX tables               -> results/tables/
python -m care.figures    # figures                    -> results/figs/
```

Cached responses are read from
`<CARE_DATA_ROOT>/<dataset>/<model>/calibration_cache/<prompt>.jsonl`
(default `results/inference/`).

## Revision analyses and control experiments

Added in response to peer review. The submitted analysis is unchanged:
`python -m care.analysis` is pinned to the paper's four pair sets and three prompts,
and the four pair files above are checked by hash and never rewritten.

```
care/
  controls.py    control pair sets, per-pair screens, source split of every pair
  revision.py    the added analyses (CPU only, from cached responses)
  prompts.py     + control prompts P4-P6
notebooks/
  r1_revision_colab.ipynb    the new inference runs, then the analysis
tests/
  test_revision.py
```

**Control pair sets** (`python -m care.controls`, after `python -m care.pairgen`):

| Set | What it is | What it controls for |
|---|---|---|
| `pairs_<src>_placebo` | `right-handed` / `left-handed` at the race insertion point, same vignettes as the race set | the effect of one changed word that carries no demographic information |
| `pairs_<src>_gender_edited` | the sex pairs with one neutral word inserted into both versions | the male version being the verbatim benchmark text |
| `pairs_<src>_gender_reverse` | sex pairs built from female-coded vignettes, so the female version is verbatim and the male version is substituted | direction of substitution |
| `pairs_<src>_{race,gender}_full` | every pair the rules can build | sample size |

The command also writes `pair_flags.csv` (source split and screens per pair),
`excluded_<set>.csv`, and `review_<set>.md` with 40 sampled pairs per set for reading.

| File | SHA-256 |
|---|---|
| `pairs_medqa_placebo.jsonl` | `0697c458768d84cc` |
| `pairs_medqa_gender_edited.jsonl` | `d6a3d8ea99e4ad56` |
| `pairs_medqa_gender_reverse.jsonl` | `bb0532e911e26fa3` |
| `pairs_medqa_race_full.jsonl` | `6e93930dd192d13f` |
| `pairs_medqa_gender_full.jsonl` | `ee692c1d53a602cd` |
| `pairs_medmcqa_placebo.jsonl` | `0b3b9b89f8edd19a` |
| `pairs_medmcqa_gender_edited.jsonl` | `bc424f5791c77b51` |
| `pairs_medmcqa_gender_reverse.jsonl` | `a0a40b986353d884` |
| `pairs_medmcqa_race_full.jsonl` | `098a2567be174910` |
| `pairs_medmcqa_gender_full.jsonl` | `7714d0a567d65d09` |

**Control prompts.** `P4` physician persona only; `P5` length-matched neutral
instructions without persona; `P6` persona and length-matched instructions without
fairness wording. `P4 - P1` isolates the persona, and `P2 - P6`, `P3 - P6` the fairness
guidance with persona and length held fixed.

**Analyses** (`python -m care.revision`; outputs `results/csv/r1_*.csv` and `r1_summary.md`):

```bash
python -m care.revision --stage cache      # needs only the responses cached for the paper
python -m care.revision --stage controls   # needs the runs of notebooks/r1_revision_colab.ipynb
python -m care.revision --stage cache --tag full --datasets pairs_medqa_race_full pairs_medqa_gender_full pairs_medmcqa_race_full pairs_medmcqa_gender_full
```

| Output | Content |
|---|---|
| `r1_pair_level` | within-pair difference in confidence and in the probability of the keyed answer; rate at which the selected answer changes; discordant correctness |
| `r1_ranking` | AUROC and area under the risk-coverage curve per group, with paired differences |
| `r1_common` | held-out coverage and accepted-case error per group under one cross-fitted threshold, under group-specific thresholds, and under thresholds with a safety margin (upper confidence bound on the error) |
| `r1_directional`, `r1_permutation`, `r1_pooled`, `r1_mixed` | the directional pattern: counts with non-estimable comparisons set aside, a within-pair label-swap test, a cluster bootstrap, a mixed model |
| `r1_bca` | percentile and BCa intervals for the SAR difference |
| `r1_by_split`, `r1_sensitivity` | results by source split and after removing screened pairs |
| `r1_tlp_vs_ce` | agreement of answers between confidence procedures; the two sources on the same answers |
| `r1_power` | pairs needed to detect a two-point difference in held-out coverage |
| `r1_controls`, `r1_edit_effect`, `r1_prompt_controls` | the control experiments |

One correction to the submitted code: the group-blind threshold compared the error
rate as `1 - mean(correct)`, which rejects a group that sits exactly on the tolerance
through floating-point rounding. It now uses errors / accepted, as the SAR code does.
No value reported at the 10% tolerance changes.

## Citation

If you use this code, please cite the paper (reference to be added on publication).

## Licence

The code is released under the [MIT Licence](LICENSE). MedQA and MedMCQA are
distributed by their authors under their own licences and are not included.
