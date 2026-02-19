# OpenClaw_V2

This repository contains the analysis code, input datasets, generated figures, and derived results used for the OpenClaw project.

## Repository structure

- `codes/` — Python scripts for analysis and figure generation
- `Datasets/` — input CSV files required by the analysis scripts
- `Figures/` — generated figure files
- `results/` — derived outputs (CSV/TXT) produced by the analysis scripts

## Reproducing outputs

The scripts in `codes/` read from `Datasets/` and write outputs to `results/` and `Figures/`.

A typical run order is:

1. `codes/label_comments_response_type.py`  
   Creates labeled comments (if needed), including `Datasets/comments_labeled.csv`.

2. `codes/figure19_risk_vs_policing.py`  
   Generates the main DI vs corrective figure and related summary outputs.

3. `codes/stepX_permutation_null_test.py`  
   Runs the permutation null test and generates its figure and summaries.

4. `codes/di_negative_feedback_event_aligned.py`  
   Creates event-aligned thread-level outputs.

5. `codes/event_aligned_bootstrap_ci.py`  
   Computes bootstrap confidence intervals for event-aligned summaries.

6. `codes/figure_event_aligned_negative_feedback.py`  
   Generates the event-aligned figure.

7. `codes/mixed_effects_di_corrective.py`  
   Produces mixed-effects model outputs.

8. `codes/di_lexicon_summary.py`  
   Exports lexicon summary and pattern list outputs.

9. `codes/di_robustness_variant.py`  
   Runs DI lexicon robustness checks.

10. `codes/step2_within_agent_fe_regression.py`  
    Produces within-agent fixed-effects analysis outputs.

11. `codes/step3_stratified_early_corrected.py`  
    Produces stratified early-correction analysis outputs.

## Python environment

A typical environment for running the scripts:

```bash
conda create -n openclaw python=3.10 -y
conda activate openclaw
pip install numpy pandas scipy matplotlib statsmodels scikit-learn
