# Auto Align-VLM: experiment record (v3 run)

This repository holds the code and the complete executed record of the Auto Align-VLM experiments on
LLaVA-1.5-7B. It is the data and code companion to the manuscript *Auto Align-VLM: A Reinforcement
Learning Framework for Mitigating Object Hallucination in Vision–Language Models* (under review).

The record covers the following:

* the executed Colab notebook, with its cell outputs and logs;
* every sampled caption, detector call and reward score;
* every preference pair and trained adapter;
* every raw evaluation output.

The summary files that the reported tables and figures are computed from are included too. Every
reported number can therefore be re-derived from the files here, and checked, without retraining
anything.

---

## 1. Contents

```
autoalignvlm/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── notebooks/
│   └── auto_align_vlm_v3_q1_pipeline_full_executed.ipynb   # full run, outputs kept
└── auto-align-vlm_v3/                # copy of the run folder from Google Drive
    ├── cache/                        # training-side intermediate files
    │   ├── selections.json           # fixed image ids for training and evaluation (data seed 2024)
    │   ├── samples_r{1,2}.jsonl      # k = 4 sampled captions per training image, per round
    │   ├── detections_*_r{1,2}.jsonl # Grounding DINO output for each sampled caption
    │   ├── scores_r*_*.jsonl         # reward terms per sample, for each reward variant
    │   ├── pairs/                    # chosen/rejected pairs used for each trained model
    │   ├── detections_cc_eval_base.jsonl
    │   └── chair_synonyms.txt        # synonym list used by the CHAIR scorer
    ├── runs/                         # one folder per trained model
    │   └── <model>/                  # LoRA adapter (adapter_config.json + weights), training log, run metadata
    ├── eval/                         # one folder per evaluated model
    │   └── <model>/
    │       ├── chair_captions.jsonl  # greedy caption for each of the 500 evaluation images
    │       ├── chair_per_image.json  # per-image CHAIR record
    │       ├── pope_{random,popular,adversarial}.jsonl
    │       ├── mme.jsonl
    │       └── metrics.json
    ├── logs/
    │   └── compute_ledger.jsonl      # wall-clock time and peak GPU memory of every step
    └── publication/
        ├── figures/                  # fig01–fig13, PDF + PNG
        ├── tables/                   # tab01–tab09, LaTeX + CSV
        └── data/                     # machine-readable summaries (see Section 3)
```

## 2. Models in the record

Each model name is `<group>_s<training seed>`. Both the `runs/` folders and the `eval/` folders use
these names.

| Name | What it is |
|---|---|
| `base` | LLaVA-1.5-7B with no training |
| `har_main_s42`, `har_main_s123`, `har_main_s456` | Main method, first round, three training seeds |
| `iter2_s42` | Second round, started from `har_main_s42` and trained on its own new samples |
| `ctrl_length_s42` | Control: the shorter caption of each pair is preferred |
| `ctrl_random_s42` | Control: chosen and rejected captions picked at random |
| `abl_h_only_s42` | Ablation: hallucination term only, no coverage term |
| `abl_coco80_s42` | Ablation: detector asked only about the 80 COCO class names |
| `lam1_s42`, `lam4_s42` | Hallucination-penalty weight 1 and 4 (main setting: 2) |
| `base_vcd`, `ours_vcd` | `base` and `har_main_s42` decoded with visual contrastive decoding |

All models share the same images, samples, training schedule and evaluation. They differ only in the
property named in the table.

## 3. Where each number comes from

| Quantity | File | Key or field |
|---|---|---|
| CHAIR_S, CHAIR_I, recall, caption length | `publication/data/results_master.json` | `policies.<model>.metrics` (`CHAIRs`, `CHAIRi`, `Recall`, `avg_words`) |
| POPE accuracy, precision, recall, F1, yes-ratio | same file | `policies.<model>.metrics` (`POPE_<setting>_<metric>`) |
| MME scores | same file | `policies.<model>.mme_detail` |
| Seed means and standard deviations | same file | `groups` |
| Paired tests (bootstrap CI, Wilcoxon, McNemar, Holm) | same file | `tests`, `pope_tests` |
| Bootstrap resamples behind the confidence intervals | `publication/data/bootstrap_distributions.json` | |
| CHAIR within caption-length bins | `publication/data/length_matched_chair.json` | one list of bins per model |
| Caption-level reward check | `publication/data/reward_validity.json` | Spearman ρ, AUC |
| Object-level reward check against COCO labels | `publication/data/reward_validity_object_level.json` | `summary` (AUC, κ, confusion counts); `rows` (one per object) |
| Preference-pair statistics | `publication/data/pair_stats.json` | |
| GPU hours | `publication/data/results_master.json`; `logs/compute_ledger.jsonl` | `compute_gpu_hours`, `compute_gpu_hours_total` |
| Full configuration and library versions | `publication/data/run_config.json` | `config`, `env` |

`figure_index.json` and `table_index.json` record which inputs each figure and each table was
drawn from.

A quick check, which needs nothing beyond Python:

```python
import json
r = json.load(open("auto-align-vlm_v3/publication/data/results_master.json"))
for m in ["base", "har_main_s42", "iter2_s42", "ctrl_length_s42"]:
    x = r["policies"][m]["metrics"]
    print(f"{m:16s} CHAIR_S={x['CHAIRs']:.1f}  CHAIR_I={x['CHAIRi']:.2f}  recall={x['Recall']:.2f}")
```

## 4. Running the pipeline

The notebook is written for Google Colab with one NVIDIA A100 GPU (the record was produced on an
A100-SXM4-80GB). Some steps need more than 40 GB of GPU memory.

1. Open `notebooks/auto_align_vlm_v3_q1_pipeline_full_executed.ipynb` in Colab and mount Google
   Drive (cell 0).
2. In cell 1, set `DRIVE_BASE` to the Drive folder that will hold the run. Set `PROFILE`:
   * `"smoke"` runs a few images through every step, in roughly 25–40 minutes. It writes to a
     separate `_smoke` folder and never touches the full run.
   * `"full"` is the complete run.
3. Run the cells in order. Cell 2 installs pinned versions of `transformers`, `trl` and `peft`
   without replacing Colab's PyTorch build.

All intermediate stores are resumable. Copy this repository's `auto-align-vlm_v3/` folder to
`DRIVE_BASE` first, and the notebook will find the completed samples, detections, pairs, adapters
and evaluations and skip those steps. That way it rebuilds the summary files, tables and figures
from the stored outputs. To repeat the experiment from scratch, start with an empty folder; the
full run took about 21.6 GPU-hours.

Main settings, as stored in `run_config.json`:

| Setting | Value |
|---|---|
| Policy model | `llava-hf/llava-1.5-7b-hf` |
| Detector | `IDEA-Research/grounding-dino-tiny` (box 0.30, text 0.25, keep 0.35, ≤ 40 phrases) |
| Phrase matcher | `sentence-transformers/all-MiniLM-L6-v2` (τ = 0.5) |
| Reward weights | α = 1.0, λ = 2.0; minimum pair margin 0.25 |
| Sampling | k = 4, temperature 0.9, top-p 0.95 |
| Training | LoRA r = 16, α = 32, dropout 0.05 on the language model; DPO β = 0.3; learning rate 1e-5; 2 epochs |
| Data | 5,000 training images; 500 evaluation images (COCO val2014); data seed 2024 |
| Evaluation | greedy decoding, up to 512 new tokens; POPE random/popular/adversarial; MME |
| Software | Python 3.13, PyTorch 2.14 (CUDA 13.0), transformers 4.57.6, trl 0.29.1, peft 0.19.1, datasets 3.6.0, spaCy 3.8, NLTK 3.9 |

## 5. Third-party data that is not included

The notebook downloads these datasets from their original sources. They are not redistributed here,
for licensing reasons and because of their size:

* **COCO 2014 images and annotations:** <https://cocodataset.org>. The notebook fetches only the
  images the study uses.
* **Karpathy caption split:** the standard `dataset_coco.json`.
* **POPE question files** for COCO, from the POPE authors' release.
* **MME:** the `lmms-lab/MME` dataset on Hugging Face.

Captions, detections and scores in this repository refer to COCO images by their image id.

## 6. Licence and citation

All rights reserved. While the manuscript is under review, the material in this repository may be
viewed, downloaded and run only to evaluate or verify the reported results; any other use needs
written permission (see `LICENSE`). An open licence will replace this one when the article is
published. Third-party models and datasets remain under their own licences. To cite this
repository, use `CITATION.cff`.

Contact: Md Mynoddin, Department of Computer Science and Engineering, Rangamati Science and
Technology University, Bangladesh. Email: mynoddin at rmstu .ac .bd  or mmynoddin at gmail com
