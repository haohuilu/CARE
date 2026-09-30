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

## Citation

If you use this code, please cite the paper (reference to be added on publication).

## Licence

The code is released under the [MIT Licence](LICENSE). MedQA and MedMCQA are
distributed by their authors under their own licences and are not included.
