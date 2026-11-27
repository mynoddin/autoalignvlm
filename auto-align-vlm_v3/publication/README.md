# Auto Align-VLM v3 — publication bundle

Profile: full | config hash 8553c4b43c | generated 2026-11-12 23:29

All numbers come from `data/results_master.json`, produced by the notebook from cached raw
model outputs (`../eval/<policy>/`). Figures are vector PDF + PNG; tables are LaTeX + CSV.

## Figures
- `fig01_pipeline.pdf`
- `fig02_preference_pairs.pdf`
- `fig03_training_dynamics.pdf`
- `fig04_chair_main.pdf`
- `fig05_tradeoff.pdf`
- `fig06_pope.pdf`
- `fig07_length_analysis.pdf`
- `fig08_lambda_sensitivity.pdf`
- `fig09_categories.pdf`
- `fig10_mme.pdf`
- `fig11_bootstrap.pdf`
- `fig12_reward_validity.pdf`
- `fig13_qualitative.pdf`

## Tables
- `tab01_setup.tex` — Experimental configuration....
- `tab02_chair.tex` — Object hallucination on 500 COCO val2014 images (official CHAIR; greedy decoding, max 512 ...
- `tab03_pope.tex` — POPE results on COCO (3{,}000 questions per setting)....
- `tab04_mme.tex` — MME scores (acc + acc+). Hallucination subset = existence, count, position, colour (max 80...
- `tab05_ablations.tex` — Ablations and matched controls (training seed 42; identical samples, schedule and evaluati...
- `tab06_significance.tex` — Paired comparison with the base model on the same images: change with 95\% bootstrap CI (1...
- `tab07_pairs.tex` — Preference-pair statistics. $H$ is always the caption-conditioned hallucination rate, so v...
- `tab08_compute.tex` — Measured wall-clock cost on one NVIDIA A100-SXM4-80GB....
- `tab09_per_seed.tex` — Per-seed results of the main model (supplementary)....

## Data
- `results_master.json` — every metric, test and compute figure
- `run_config.json` — configuration + library versions
- `reward_validity.json`, `bootstrap_distributions.json`, `pair_stats.json`, `figure_index.json`, `table_index.json`
