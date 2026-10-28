# Pneumonia Detection from Chest X-rays with a Swin Transformer

Fine-tunes an ImageNet-pretrained **Swin-T** (Swin Transformer, tiny) to classify pediatric chest X-rays as **NORMAL** or **PNEUMONIA**. The pipeline uses a **patient-level** train/validation split to avoid data leakage, a class-weighted loss for the imbalanced data, and a **recall-oriented decision threshold** chosen on validation data. The held-out test set is evaluated once, and Grad-CAM heatmaps are produced for inspection.

> **Not a medical device.** This is a research and educational project. It has not been clinically validated and must not be used for diagnosis or treatment decisions.

## Results (held-out test set, 624 images)

| Metric | Value |
|---|---|
| Recall (sensitivity) | 0.995 (388 / 390) |
| Specificity | 0.709 (166 / 234) |
| Precision | 0.851 |
| NPV | 0.988 |
| Accuracy | 0.888 |
| F1 | 0.917 |
| ROC-AUC | 0.981 |
| Decision threshold | 0.363 (chosen on validation) |
| Training time | about 16.9 min |

Confusion matrix (`[[TN, FP], [FN, TP]]`): `[[166, 68], [2, 388]]`

These come from a single run with seed 42. Validation ROC-AUC (0.999) is noticeably higher than test ROC-AUC (0.981), so treat the numbers as indicative, not definitive.

## Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) from Kaggle (Kermany et al., *Cell*, 2018): 5,856 pediatric (ages 1-5) anterior-posterior chest radiographs from one medical centre, labelled NORMAL (0) or PNEUMONIA (1).

| Official split | NORMAL | PNEUMONIA |
|---|---|---|
| train | 1,341 | 3,875 |
| val | 8 | 8 |
| test | 234 | 390 |

The notebook downloads the dataset automatically through the Kaggle API (see Setup).

## Pipeline

The notebook (`swin_pneumonia.ipynb`) runs these steps in order:

1. **Dataset setup:** download the data via the Kaggle API.
2. **Data loading:** build `data.csv` (path, label, class, split).
3. **EDA:** class balance per split and image-size statistics.
4. **Visualisation:** sample X-rays and pixel-intensity histograms.
5. **Bad-image removal:** drop unreadable, tiny (<64 px) or blank images (none were removed).
6. **Data selection:** merge the 16-image official `val` into `train` to form a development pool; keep `test` held out.
7. **Normalisation:** resize to 224×224, ImageNet mean/std.
8. **Patch and window illustration:** 4×4 patches, 7×7-token attention windows.
9. **Patient-level split:** patient IDs parsed from filenames; `StratifiedGroupKFold` (7 folds, one used for validation); asserts zero patient overlap.
10. **Class-imbalance handling:** class-weighted cross-entropy (weights 2.84 : 1.00); augmentation on training images only.
11. **Model:** torchvision `swin_t` with ImageNet weights, new `Linear(768, 2)` head (27.5 M parameters).
12. **Training:** AdamW, cosine schedule, mixed precision; the best-validation-AUC checkpoint is saved.
13. **Prediction scores:** pneumonia probability histograms.
14. **Threshold selection:** the highest validation threshold that still reaches 95% recall.
15. **Test evaluation:** run once at the fixed threshold.
16. **Confusion matrix.**
17. **Grad-CAM:** heatmaps for 4 correct and 4 incorrect test predictions.

## Setup

**Recommended: Google Colab with a GPU runtime** (the notebook uses `/content/data` paths).

1. Open `swin_pneumonia.ipynb` in Colab and select **Runtime → Change runtime type → GPU**.
2. Create a Kaggle API token (Kaggle → Settings → API). Paste it when the dataset cell prompts you. It is read with `getpass`, so it is not stored in the notebook.
3. Run all cells top to bottom.

**Dependencies** (the first cell installs `kaggle` and `grad-cam`; everything else is preinstalled on Colab):
`torch`, `torchvision`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `pillow`, `kaggle`, `grad-cam`

**Running locally:** change `DATA_DIR` in the dataset cell to a local folder and install the packages above. A CUDA GPU is strongly recommended; on CPU, training will be very slow.

If you hit an out-of-memory error, set `BS = 16` in the data-loader cell.

## Outputs

| File | Description |
|---|---|
| `data.csv` | Image path, label, class and official split for every image |
| `swin_t_best.pt` | Best checkpoint (by validation ROC-AUC) |
| `results.csv` | Test metrics for the final model |
| `confusion_matrix.png` | Test confusion matrix |
| `gradcam_swin.png` | Grad-CAM panel (4 correct, 4 incorrect predictions) |

## Key settings

| Setting | Value |
|---|---|
| Input size | 224 × 224 |
| Optimiser | AdamW, lr 3e-5, weight decay 0.05 |
| Schedule | Cosine annealing, 10 epochs |
| Batch size | 32 (training), 64 (evaluation) |
| Loss | Cross-entropy, class weights (2.84, 1.00) |
| Augmentation | Rotation ≤10°, random resized crop (0.9-1.0), brightness 0.1; no horizontal flip |
| Seed | 42 |
| Validation split | 1 of 7 `StratifiedGroupKFold` folds: 4,489 train / 743 val |

## Known limitations

- **Possible shortcut learning:** the Grad-CAM panel shows strong activation on the "R" laterality marker, image borders and corners, even for correct predictions. The model may partly rely on non-lung cues.
- **Validation-test gap** (AUC 0.999 vs. 0.981; validation precision 1.0 vs. test 0.851): the 68 false positives suggest the threshold does not transfer perfectly.
- **Patient IDs are inferred from filenames** (`personNNN`, `IM-NNNN`); residual leakage is possible if these do not uniquely identify patients.
- **Single run:** one seed, one validation fold, no confidence intervals.
- **Narrow data:** single-centre, pediatric, anterior-posterior images only; results will not generalise to adults or other hospitals without external validation.
- **Binary task:** bacterial and viral pneumonia are not distinguished.

## Suggested extensions

`swin_extra_experiments.py` adds blocks (paste at the end of the notebook, same session) for CNN/ViT baselines, pretrained-vs-scratch and class-weight ablations, multi-seed runs, bootstrap confidence intervals, ROC/PR and calibration plots, false-positive analysis and a quantitative border/corner shortcut check. It needs the notebook's variables (`train_df`, `val_df`, `test_df`, `XrayDataset`, `train_tf`, `eval_tf`, `device`, `SEED`, `class_w`) and `grad-cam` installed.

## References

- D. S. Kermany et al., *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning*, Cell 172(5), 2018.
- Z. Liu et al., *Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows*, ICCV 2021.
- R. R. Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization*, ICCV 2017.

## License and acknowledgements

Add your license here. The dataset is subject to its own terms on Kaggle (Kermany et al.). Model weights come from torchvision's ImageNet-pretrained Swin-T.
