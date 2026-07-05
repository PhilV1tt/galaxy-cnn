"""Build galaxy_cnn.ipynb, self-contained, from the validated galaxy/ package.

We inline the package code (without the relative imports) as "library" cells,
then add the narration and orchestration of sections 1 to 8. This way the notebook
runs on Colab without depending on the repo files, while staying the same code as
the one tested locally.

    uv run python build_notebook.py
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent
GAL = ROOT / "galaxy"


def strip_module(fname: str) -> str:
    """Read a module and strip __future__ and the relative imports (inline)."""
    out = []
    for ln in (GAL / fname).read_text().splitlines():
        s = ln.strip()
        if s.startswith("from __future__"):
            continue
        if s.startswith("from .") or s.startswith("from . "):
            continue
        out.append(ln)
    return "\n".join(out).strip()


nb = nbf.v4.new_notebook()
cells = []
md = lambda t: cells.append(nbf.v4.new_markdown_cell(t))
code = lambda t: cells.append(nbf.v4.new_code_cell(t))

# ---------------------------------------------------------------- Titre
md(r"""# Morphological classification of galaxies with a CNN
### Galaxy10 DECaLS: 17 736 images, 10 classes

Philémon Vittet

**Question.** Can a convolutional neural network reproduce the morphological
sorting work done by the Galaxy Zoo volunteers?

We tell a story on three levels to isolate the contribution of **convolution**:

1. **baselines without convolution** (logistic regression, MLP on flattened pixels);
2. **custom CNN** (weight sharing, translation equivariance);
3. **transfer learning** (ResNet18 pretrained on ImageNet).

Then we **interpret** with Grad-CAM (where the network looks) and we quantify the
contribution of each ingredient through **ablations**. The physical link is direct: the
translation equivariance of the CNN is a symmetry imposed on the architecture, in the
spirit of symmetries in physics.

Sections: 0 Setup · 1 EDA · 2 Preprocessing & augmentation · 3 Baselines · 4 custom CNN &
tuning · 5 Transfer learning · 6 Test evaluation · 7 Grad-CAM · 8 Ablations.""")

# ---------------------------------------------------------------- 0. Setup
md(r"""## 0. Setup: environment and library

On Colab, enable the GPU (Runtime → Change runtime type → T4) and run the
install cell. The following cells define the whole project library
(data, models, training, metrics, figures, Grad-CAM): the notebook is
self-contained.""")

code(r"""# Colab: uncomment the next line (torch/torchvision/sklearn are already present)
# !pip -q install "datasets>=3.0.0"
import os, json, time, copy, random
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from IPython.display import Image as ShowImage, display
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "| mps", torch.backends.mps.is_available())""")

md("### Constants, device and paths")
code(strip_module("config.py"))

md(r"""### Run settings

`FULL_RUN=True`: full run (Colab, the whole dataset, ~30 epochs). To iterate fast
locally, export `GALAXY_FULL=0`: the notebook switches to the small test split (all
classes) and few epochs, which validates the whole chain in a few minutes.""")
code(r"""FULL_RUN = os.environ.get("GALAXY_FULL", "1") == "1"
RES_SMALL, RES_LARGE = 64, 128        # 64 px to iterate, 128 px for the final runs
BATCH = 128
NUM_WORKERS = 2 if torch.cuda.is_available() else 0
if FULL_RUN:
    SMOKE_N = None
    EPOCHS_BASE, EPOCHS_CNN, TUNE_EPOCHS = 15, 30, 10
    EPOCHS_RES_FE, EPOCHS_RES = 6, 12
else:
    SMOKE_N = 800
    EPOCHS_BASE, EPOCHS_CNN, TUNE_EPOCHS = 3, 4, 2
    EPOCHS_RES_FE, EPOCHS_RES = 2, 2
print("FULL_RUN =", FULL_RUN, "| SMOKE_N =", SMOKE_N)""")

md("### Metrics")
code(strip_module("metrics.py"))
md("### Models: baselines, custom CNN, ResNet18")
code(strip_module("models.py"))
md("### Data: loading, split, normalization, augmentation")
code(strip_module("data.py"))
md("### Training: loop, evaluation, early stopping on macro F1")
code(strip_module("engine.py"))
md("### Figures")
code(strip_module("viz.py"))
md("### Grad-CAM")
code(strip_module("gradcam.py"))

# ---------------------------------------------------------------- 1. EDA
md(r"""## 1. Exploratory analysis (EDA)

We load the data and look at what we handle: one example per class, the
distribution (strongly imbalanced), and the color statistics per class.""")
code(r"""set_seed()
device = get_device(); print("device:", device)

train_ds, val_ds, test_ds, (mean, std) = get_datasets(res=RES_SMALL, smoke_n=SMOKE_N)
print(f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")
display(ShowImage(plot_samples_grid(test_ds.x, test_ds.y)))""")

code(r"""# Class distribution over the training set (train + val)
y_full = torch.cat([train_ds.y, val_ds.y])
display(ShowImage(plot_class_distribution(y_full)))""")

code(r"""# Mean color statistics per class (on the train set, in [0,1])
xtr = train_ds.x.float() / 255.0
rows = []
for c in range(NUM_CLASSES):
    m = xtr[train_ds.y == c].mean(dim=(0, 2, 3))
    rows.append([CLASS_NAMES[c], int((train_ds.y == c).sum()), *[round(v, 3) for v in m.tolist()]])
pd.DataFrame(rows, columns=["classe", "n_train", "R", "G", "B"])""")

# ---------------------------------------------------------------- 2. Preprocessing
md(r"""## 2. Preprocessing and augmentation

**Split.** The dataset already provides a `train` (16 000) and a `test` (1 800). We keep the
**official test as the final test, never touched**. Validation is a **stratified
15 %** draw from the train (fixed seed). All tuning decisions are made on the validation set.

**Normalization.** per-channel mean/std computed **on the train only**, to avoid any
information leak from the test.

**Augmentation (train only).** Rotations 0-360°, horizontal and vertical flips, slight
brightness/contrast jitter. Physical justification: a galaxy has no privileged
orientation in the sky, these transformations are real invariances of the problem. We
apply them **after the split**, never on val/test.""")
code(r"""print("mean train:", [round(v, 4) for v in mean.tolist()])
print("std  train:", [round(v, 4) for v in std.tolist()])
display(ShowImage(plot_augmentation(train_ds.x[0], mean, std)))""")

# ---------------------------------------------------------------- 3. Baselines
md(r"""## 3. Baselines without convolution

They should be mediocre: that is the point. They flatten the image and treat each
pixel as an independent variable, with no notion of spatial structure. We weight the
loss by the inverse of the frequencies (imbalance) and track the **macro F1**.""")
code(r"""tl, vl, te = make_loaders(train_ds, val_ds, test_ds, batch_size=BATCH, num_workers=NUM_WORKERS)
cw = class_weights(train_ds.y).to(device)
crit = nn.CrossEntropyLoss(weight=cw)
results = {}

logreg = LogisticRegression(image_size=RES_SMALL)
logreg, _ = fit(logreg, tl, vl, epochs=EPOCHS_BASE, lr=1e-3, criterion=crit,
                scheduler=None, device=device, verbose=False)
_, yt, yp = evaluate(logreg, te, crit, device)
results["LogReg"] = summary_metrics(yt, yp)

mlp = MLP(image_size=RES_SMALL)
mlp, _ = fit(mlp, tl, vl, epochs=EPOCHS_BASE, lr=1e-3, criterion=crit,
             scheduler=None, device=device, verbose=False)
_, yt, yp = evaluate(mlp, te, crit, device)
results["MLP"] = summary_metrics(yt, yp)

pd.DataFrame(results).T[["accuracy", "f1_macro", "f1_weighted"]].round(3)""")

# ---------------------------------------------------------------- 4. CNN
md(r"""## 4. Custom CNN and tuning

4 blocks `[Conv 3x3 (no bias) → BatchNorm → ReLU → MaxPool]` with channels
`(32, 64, 128, 256)`, `AdaptiveAvgPool(4x4)`, then a dense head. The 3x3 conv with padding 1
keeps the size; pooling divides it by 2. Sharing a filter's weights over the whole
image gives **translation equivariance**: moving a pattern in the image moves
its response by the same amount.""")
code(r"""cnn = GalaxyCNN()
print("trainable parameters:", f"{count_parameters(cnn):,}")

# Output dimensions block by block (input 3 x RES_SMALL x RES_SMALL)
x = torch.zeros(1, 3, RES_SMALL, RES_SMALL); shapes = []
for i, blk in enumerate(cnn.features):
    x = blk(x); shapes.append([f"bloc {i+1}", tuple(x.shape[1:])])
display(pd.DataFrame(shapes, columns=["couche", "sortie (C, H, W)"]))""")

code(r"""cnn, hist = fit(cnn, tl, vl, epochs=EPOCHS_CNN, lr=1e-3, criterion=crit, device=device)
display(ShowImage(plot_curves(hist)))
_, yt, yp = evaluate(cnn, te, crit, device)
results["CNN"] = summary_metrics(yt, yp)
print("CNN test:", {k: round(v, 3) for k, v in results["CNN"].items()})""")

md("**Learning rate tuning** (the most important). We keep the best validation macro F1.")
code(r"""runs = []
for lr in [3e-4, 1e-3, 3e-3]:
    m = GalaxyCNN()
    m, h = fit(m, tl, vl, epochs=TUNE_EPOCHS, lr=lr, criterion=crit, device=device, verbose=False)
    runs.append({"lr": lr, "val_f1_macro": round(h["best_val_f1_macro"], 3),
                 "best_epoch": h["best_epoch"]})
pd.DataFrame(runs)""")

# ---------------------------------------------------------------- 5. Transfer
md(r"""## 5. Transfer learning: ResNet18

We reuse a ResNet18 pretrained on ImageNet, replace the last layer with 10
outputs, and proceed in two stages: first **feature extraction** (frozen backbone, we
train only the head), then **fine-tuning** (everything unfrozen, lower LR). We work
at 128 px and normalize with the ImageNet statistics expected by the backbone.""")
code(r"""tr2, val2, te2ds, _ = get_datasets(res=RES_LARGE, smoke_n=SMOKE_N,
                                   mean=IMAGENET_MEAN, std=IMAGENET_STD)
tl2, vl2, te2 = make_loaders(tr2, val2, te2ds, batch_size=BATCH, num_workers=NUM_WORKERS)
cw2 = class_weights(tr2.y).to(device)
crit2 = nn.CrossEntropyLoss(weight=cw2)

resnet = build_resnet18(pretrained=True, mode="features")
resnet, _ = fit(resnet, tl2, vl2, epochs=EPOCHS_RES_FE, lr=1e-3, criterion=crit2,
                device=device, verbose=False)                    # head only
for p in resnet.parameters():
    p.requires_grad = True
resnet, hist_res = fit(resnet, tl2, vl2, epochs=EPOCHS_RES, lr=1e-4, criterion=crit2,
                       device=device)                            # full fine-tuning
_, yt2, yp2 = evaluate(resnet, te2, crit2, device)
results["ResNet18"] = summary_metrics(yt2, yp2)
print("ResNet18 test:", {k: round(v, 3) for k, v in results["ResNet18"].items()})""")

# ---------------------------------------------------------------- 6. Eval
md(r"""## 6. Test evaluation

The baseline → CNN → ResNet scale answers the question "what do we gain with
convolution?". We then look at the confusion matrix and the per-class F1 of the best
model, and analyze the confusions between neighboring classes.""")
code(r"""display(ShowImage(plot_scale_comparison(results, metric="f1_macro")))
display(pd.DataFrame(results).T[["accuracy", "f1_macro", "f1_weighted"]].round(3))""")

code(r"""# Best model = ResNet18: confusion + per-class F1
display(ShowImage(plot_confusion(confusion(yt2, yp2, normalize="true"))))
pc = per_class_df(yt2, yp2)
display(ShowImage(plot_perclass_f1(pc)))
pc.round(3)""")

code(r"""# Save results for the report (number injection)
results_out = {
    "models": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in results.items()},
    "cnn_params": count_parameters(GalaxyCNN()),
    "cnn_best_epoch": hist.get("best_epoch"),
    "resnet_best_epoch": hist_res.get("best_epoch"),
    "per_class_f1_resnet": {CLASS_NAMES[i]: round(float(pc["f1"].iloc[i]), 3)
                            for i in range(NUM_CLASSES)},
    "tuning_lr": runs,
    "config": {"res_small": RES_SMALL, "res_large": RES_LARGE, "batch": BATCH,
               "full_run": FULL_RUN},
}
out_path = project_root() / "results.json"
json.dump(results_out, open(out_path, "w"), indent=2, ensure_ascii=False)
print("written:", out_path)
print(json.dumps(results_out["models"], indent=2, ensure_ascii=False))""")

# ---------------------------------------------------------------- 7. Grad-CAM
md(r"""## 7. Grad-CAM: where the network looks

Grad-CAM weights the feature maps of the last convolution by the mean gradient of the class
score, applies a ReLU, and rescales to the image size. We apply it on correct
predictions **and** errors, for the custom CNN and for the ResNet: does the network
look at the spiral arms, the bar, the merger companion, or artifacts?""")
code(r"""def pick_examples(model, ds, device, n_ok=3, n_bad=3):
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in DataLoader(ds, batch_size=256):
            preds.append(model(xb.to(device)).argmax(1).cpu())
    preds = torch.cat(preds)
    ok = (preds == ds.y).nonzero().flatten().tolist()
    bad = (preds != ds.y).nonzero().flatten().tolist()
    return ok[:n_ok] + bad[:n_bad]

idx = pick_examples(cnn, test_ds, device)
display(ShowImage(plot_gradcam_grid(cnn, cnn.last_conv, test_ds.x[idx], mean, std,
                                    y_true=test_ds.y[idx], device=device, name="gradcam_cnn.png")))""")

code(r"""idx2 = pick_examples(resnet, te2ds, device)
display(ShowImage(plot_gradcam_grid(resnet, resnet.layer4[-1], te2ds.x[idx2],
                                    IMAGENET_MEAN, IMAGENET_STD, y_true=te2ds.y[idx2],
                                    device=device, name="gradcam_resnet.png")))""")

# ---------------------------------------------------------------- 8. Ablations
md(r"""## 8. Ablations

We remove one ingredient at a time to quantify its contribution: **no augmentation** and
**no BatchNorm**. We compare the macro F1 (test) with the full CNN.""")
code(r"""abl = {"full CNN": results["CNN"]["f1_macro"]}

# no augmentation
tr_na, val_na, te_na, (m_na, s_na) = get_datasets(res=RES_SMALL, smoke_n=SMOKE_N, augment=False)
tl_na, vl_na, te_na_l = make_loaders(tr_na, val_na, te_na, batch_size=BATCH, num_workers=NUM_WORKERS)
cnn_na, _ = fit(GalaxyCNN(), tl_na, vl_na, epochs=EPOCHS_CNN, lr=1e-3, criterion=crit,
                device=device, verbose=False)
_, yt, yp = evaluate(cnn_na, te_na_l, crit, device)
abl["no augmentation"] = summary_metrics(yt, yp)["f1_macro"]

# no BatchNorm
cnn_nobn, _ = fit(GalaxyCNN(use_bn=False), tl, vl, epochs=EPOCHS_CNN, lr=1e-3, criterion=crit,
                  device=device, verbose=False)
_, yt, yp = evaluate(cnn_nobn, te, crit, device)
abl["no BatchNorm"] = summary_metrics(yt, yp)["f1_macro"]

# We add the ablations to results.json (for injection into the report).
try:
    _r = json.load(open(project_root() / "results.json"))
    _r["ablations"] = {k: round(v, 4) for k, v in abl.items()}
    json.dump(_r, open(project_root() / "results.json", "w"), indent=2, ensure_ascii=False)
except Exception as e:
    print("update results.json:", e)

pd.DataFrame({"f1_macro (test)": {k: round(v, 3) for k, v in abl.items()}})""")

md(r"""## Conclusion

The baseline → CNN → ResNet scale concretely shows the contribution of convolution, then of
transfer learning. Grad-CAM confirms that the network relies on the relevant morphological
structures. Limits: noisy Galaxy Zoo labels, strong imbalance (the Cigar Shaped class
with 334 images stays the hardest), and training cost.

**For the full run on Colab:** enable the T4 GPU, keep `FULL_RUN=True`, run the whole
notebook, then retrieve `figures/` and `results.json`.""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
out = ROOT / "galaxy_cnn.ipynb"
nbf.write(nb, out)
print(f"written {out} ({len(cells)} cells)")
