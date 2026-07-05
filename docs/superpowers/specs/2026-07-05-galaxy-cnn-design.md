# Design: Galaxy10 DECaLS: morphological classification with CNN

Date: 2026-07-05. Deadline: 2026-07-06 (evening). Sprint ~1 day.
Group (5): Patrick Yvan Eyike, Philémon Vittet, Sidi Hamadi, Abdelrahman Dahy, Ines Lylia Zerdani.

## 1. Goal

Classify galaxy morphology (10 classes) from color images, and
tell a clear story in three levels: **what do we gain with convolution?**
Baseline without convolution (logistic regression, MLP) < custom CNN < transfer learning
(pretrained ResNet18). Interpretation with Grad-CAM. Physical link: the translation
equivariance of the CNN is a symmetry imposed on the architecture.

## 2. Data

`matthieulel/galaxy10_decals` (Hugging Face). 17 736 images 256x256 RGB (g, r, z bands),
10 classes. Galaxy Zoo labels, DESI Legacy Imaging Surveys survey. Schema: `image` (PIL),
`label` (uint8 0-9).

Class counts (strong imbalance, factor ~8):

| id | class | n |
|----|--------|---|
| 0 | Disturbed | 1081 |
| 1 | Merging | 1853 |
| 2 | Round Smooth | 2645 |
| 3 | In-between Round Smooth | 2027 |
| 4 | Cigar Shaped Smooth | 334 |
| 5 | Barred Spiral | 2043 |
| 6 | Unbarred Tight Spiral | 1829 |
| 7 | Unbarred Loose Spiral | 2628 |
| 8 | Edge-on without Bulge | 1423 |
| 9 | Edge-on with Bulge | 1873 |

**Split.** The dataset already provides `train` (16 000) and `test` (1 800). We keep the **official
test as the final test, never touched**. We carve out a **stratified 15 % validation** in
the train (fixed seed). All tuning decisions are made on the validation.

**Preprocessing.** Resize 256 -> 64 px (iteration, baselines, first CNNs) then
128 px (final CNN + ResNet runs). Normalization by mean/std computed **on the train
only** (per channel). Cache the decoded tensors as `.pt` to avoid re-decoding 2.3 GB
on each run.

**Augmentation (train only).** Random rotation 0-360°, H and V flips, slight brightness/contrast
jitter. Physical justification: a galaxy has no preferred orientation,
these transforms are real invariances of the problem. Applied after the split, never
on val/test (no data leakage).

## 3. Models

**Baseline 1: Logistic regression** on flattened pixels (3x64x64 = 12288 features).
Multinomial softmax. Deliberately poor: no notion of spatial structure.

**Baseline 2: MLP** 2 hidden layers (512, 256) + ReLU + dropout on flattened pixels.
A bit better, still blind to locality.

**Custom CNN** (input 3x64x64):

```
Block 1: Conv(3->32, 3x3, pad 1)  -> BN -> ReLU -> MaxPool 2   => 32 x 32 x 32
Block 2: Conv(32->64, 3x3, pad 1) -> BN -> ReLU -> MaxPool 2   => 64 x 16 x 16
Block 3: Conv(64->128, 3x3, pad 1)-> BN -> ReLU -> MaxPool 2   => 128 x 8 x 8
Block 4: Conv(128->256,3x3, pad 1)-> BN -> ReLU -> MaxPool 2   => 256 x 4 x 4
AdaptiveAvgPool(4x4) (fixes the head whatever the input resolution)
Head   : Flatten(4096) -> Dropout -> Linear(4096->256) -> ReLU -> Dropout -> Linear(256->10)
```

Parameter count (to redo by hand for the defense, verified by the code = 1 440 298):
- Convs **without bias** (BatchNorm provides the shift): 864 + 18 432 + 73 728 + 294 912 = 387 936
- BatchNorm (2 params per channel): 64 + 128 + 256 + 512 = 960
- FC: (4096x256 + 256) + (256x10 + 10) = 1 048 832 + 2 570 = 1 051 402
- **Total = 1 440 298 (~1.44 M)**, dominated by the first FC layer (73 %).

`AdaptiveAvgPool(4x4)` keeps ~1.44 M params at 64 as at 128 px. Without adaptive pooling,
moving to 128 px would blow up FC1 (16384x256 ≈ 4,2 M): a good teaching point to mention.

**Transfer learning: ResNet18** pretrained on ImageNet. Two regimes:
1. *Feature extraction*: freeze the backbone, train only the new `fc` (512->10).
2. *Fine-tuning*: unfreeze everything (or `layer4` + `fc`) with a lower LR.

3-channel input compatible, resize to 128 px, ImageNet-stats normalization for
the pretrained backbone.

## 4. Training and tuning

- Adam optimizer, starting LR 1e-3 (the most important to tune), cosine scheduler.
- `CrossEntropyLoss(weight=...)` loss weighted by inverse frequencies (imbalance).
- Early stopping on the validation **macro F1** (not accuracy).
- Light grid: LR, batch size, dropout, aug on/off, depth. Runs table in pandas.
- `model.eval()` at evaluation (BN and dropout change behavior).

## 5. Evaluation (on the official test, never touched before)

Accuracy, per-class precision/recall/F1 + macro and weighted, confusion matrix, train/val
learning curves. Analysis of confusions between neighboring classes (tight vs loose
spirals, round vs cigar, edge-on with/without bulge).

## 6. Interpretation: Grad-CAM

Implemented by hand (~30 lines): forward/backward hook on the last conv (custom CNN)
and on `layer4` (ResNet18). Heat map = ReLU(weighted sum of the feature maps by the
class mean gradient). Heatmaps on correct predictions AND errors: does the network look at
the spiral arms, the bar, the merging companion, or artifacts?

## 7. Ablations

- Without augmentation.
- Without BatchNorm.
Quantify the macro F1 loss to isolate the contribution of each ingredient.

## 8. Deliverables (mirror of `../cifar/`)

```
final/
  pyproject.toml            uv: torch, torchvision, datasets, scikit-learn, matplotlib, pandas
  galaxy_cnn.ipynb          sections 0-8
  galaxy/                   testable package (data, models, train, metrics, gradcam)
  figures/                  samples, augment, courbes, confusion, perclass, echelle, gradcam
  report/rapport.tex        + rapport.pdf (LaTeX)
  slides/presentation.tex   + script.md + presentation.pdf (Beamer)
  data/                     gitignored (local .pt cache / Drive on Colab)
```

Notebook sections: 0 Loading, 1 EDA, 2 Preprocessing and augmentation, 3 Baselines,
4 Custom CNN and tuning, 5 Transfer learning, 6 Test evaluation, 7 Grad-CAM, 8 Ablations.

The notebook detects Colab vs local (CUDA/MPS device, data path) and runs in both places.

## 9. Execution plan (1 day, parallel sprint)

1. Scaffold repo + uv env + `galaxy/` package. Check: `uv sync` OK, imports OK.
2. Local smoke test (streaming ~600 images, CNN 64px, 1-2 epochs on MPS). Check: train
   loop runs, one figure produced. Guarantees a runnable notebook.
3. Notebook sections 0-8 orchestrating the package. Check: low-scale local run.
4. In parallel: LaTeX report + Beamer slides + script.md with placeholders for the numbers.
5. Full run on Colab T4 (128px, ResNet, tuning). Collect figures + `results.json`.
6. Inject the real numbers into report/slides. Compile the PDFs.

## 10. Pitfalls (reminder)

Augmentation after the split, train only. Test never touched during tuning. Macro F1,
not just accuracy. Normalize with train stats. LR 1e-3 to start. Iterate at 64/128,
not 256 right away. `model.eval()` at evaluation. Regular checkpoints on Colab.
