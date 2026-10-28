# Swin-T Pediatric Pneumonia Paper: Project README

LaTeX source and supporting code for a research paper on fine-tuning an ImageNet-pretrained **Swin-T** to classify pediatric chest X-rays (NORMAL vs. PNEUMONIA) on the Kermany dataset, using a patient-level split and a recall-oriented threshold.

Source notebook: `swin_pneumonia (1).ipynb`

## Files

| File | Purpose |
|---|---|
| `swin_pneumonia_paper_overleaf.zip` | The Overleaf project (upload this) |
| `main.tex` | Paper source (also inside the zip) |
| `references.bib` | 17 BibTeX entries (inside the zip) |
| `figures/` | `fig_training.png`, `fig_confusion.png`, `fig_threshold.png`, `fig_gradcam.png`, taken from the notebook's saved outputs (inside the zip) |
| `swin_pneumonia_paper_preview.pdf` | Compiled preview of the current draft |
| `swin_extra_experiments.py` | Code for the experiments the paper still needs |

## Using the project in Overleaf

1. **New Project → Upload Project**, then select the zip.
2. Compiler: **pdfLaTeX** (default). Main document: `main.tex`.
3. Bibliography uses `natbib` + BibTeX (`unsrtnat`); Overleaf builds it automatically.

## Status of the draft

Real, from your notebook run: dataset counts, patient-level split (2,650 groups; 4,489 train / 743 val; zero overlap), training setup, threshold (0.363), and test results (sensitivity 0.995, specificity 0.709, precision 0.851, NPV 0.988, accuracy 0.888, F1 0.917, ROC-AUC 0.981; confusion matrix 166/68/2/388).

Red **[TBD]** marks mean the result needs the extra experiments. Nothing there is invented. Search `main.tex` for `\tbd` and `\TBD` to find them all. When everything is filled in, redefine both macros in the preamble to print nothing.

## Running the extra experiments

`swin_extra_experiments.py` is written as `# %%` blocks to paste at the **end of your original notebook**, in the same session. It relies on variables defined there (`train_df`, `val_df`, `test_df`, `XrayDataset`, `train_tf`, `eval_tf`, `device`, `SEED`, `class_w`). Block 6 also needs `pip install grad-cam`. Reduce `SEEDS` or `MODELS` first if time is short; the full grid takes several GPU-hours.

| Block | What it does | Output files | Where it goes in the paper |
|---|---|---|---|
| 1 | Baselines (ResNet-50, DenseNet-121, EfficientNet-B0, ViT-B/16, Swin-T), 5 seeds | `table4_baselines.csv`, `results_baselines_all_seeds.csv` | Baseline table (Sec. 6.2) |
| 2 | Ablations: pretrained vs. scratch, weighted vs. unweighted | `table5_ablation.csv` | Ablation table (Sec. 6.3) |
| 3 | Bootstrap 95% CIs, validation-vs-test AUC gap | `table2_ci.csv` | Main-results table (Sec. 6.1, 6.5) |
| 4 | ROC/PR curves, calibration (ECE, Brier) | `fig_roc_pr.png`, `fig_calibration.png` | Sec. 6.4: upload to `figures/` and uncomment the two figure blocks |
| 5 | False-positive and false-negative analysis | `table6_fp_vs_tn.csv`, `false_positives.csv`, `false_negatives.csv`, `fig_fp_gallery.png` | Sec. 6.7: the text references these, but you still need to add the table and figure |
| 6 | Shortcut check: Grad-CAM mass on borders/corners, occlusion test | `shortcut_check.csv` | Sec. 6.6 |
| 7 | Optional lung-centred-crop experiment (stub only; not implemented) | none | Extra ablation row |

Block 1 also saves `swin_t_seed0.pt`, which block 6 loads.

## Important caveats

- **Shortcut cues:** your Grad-CAM panel shows strong activation on the "R" marker, borders and corners, even for correct predictions. The paper reports this as a limitation. Run block 6 before deciding how strongly to word it.
- **Validation-test gap:** validation AUC is 0.999 vs. 0.981 on test. The paper hypothesises distribution shift; that claim is marked for verification.
- **Patient IDs are inferred from filenames**, not from official metadata, so residual leakage is possible.
- **Baseline fairness:** torchvision's `DEFAULT` weights come from different training recipes across models (for example ResNet-50 uses ImageNet1K_V2). State this in the paper.
- **References:** check every entry in `references.bib` against the original source, and add 4-6 prior papers on this dataset with their reported metrics (marked `\tbd` in Related Work).
- **Not for clinical use:** the model has not been clinically validated.

## Placeholders to complete

Author names and affiliations, GPU model, repository link, and (if targeting a venue) the venue's LaTeX template.
