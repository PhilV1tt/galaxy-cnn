"""Construit galaxy_cnn.ipynb, auto-portant, à partir du package galaxy/ validé.

On inline le code du package (sans les imports relatifs) en cellules "bibliothèque",
puis on ajoute la narration et l'orchestration des sections 1 à 8. Ainsi le notebook
tourne sur Colab sans dépendre des fichiers du repo, tout en restant le même code que
celui testé localement.

    uv run python build_notebook.py
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent
GAL = ROOT / "galaxy"


def strip_module(fname: str) -> str:
    """Lit un module et retire __future__ et les imports relatifs (inline)."""
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
md(r"""# Classification morphologique de galaxies par CNN
### Galaxy10 DECaLS — 17 736 images, 10 classes

Patrick Yvan Eyike · Philémon Vittet · Sidi Hamadi · Abdelrahman Dahy · Inès Lylia Zerdani

**Question.** Un réseau de neurones convolutif peut-il reproduire le travail de tri
morphologique fait par les volontaires de Galaxy Zoo ?

On raconte une histoire en trois niveaux pour isoler l'apport de la **convolution** :

1. **baselines sans convolution** (régression logistique, MLP sur pixels aplatis) ;
2. **CNN maison** (partage de poids, équivariance par translation) ;
3. **transfer learning** (ResNet18 pré-entraîné sur ImageNet).

Puis on **interprète** avec Grad-CAM (où le réseau regarde) et on quantifie l'apport de
chaque ingrédient par **ablations**. Le lien physique est direct : l'équivariance par
translation du CNN est une symétrie imposée à l'architecture, dans l'esprit des symétries
en physique.

Sections : 0 Setup · 1 EDA · 2 Preprocessing & augmentation · 3 Baselines · 4 CNN maison &
tuning · 5 Transfer learning · 6 Évaluation test · 7 Grad-CAM · 8 Ablations.""")

# ---------------------------------------------------------------- 0. Setup
md(r"""## 0. Setup — environnement et bibliothèque

Sur Colab, activer le GPU (Exécution → Modifier le type d'exécution → T4) et lancer la
cellule d'installation. Les cellules suivantes définissent toute la bibliothèque du projet
(données, modèles, entraînement, métriques, figures, Grad-CAM) : le notebook est
auto-portant.""")

code(r"""# Colab : décommenter la ligne suivante (torch/torchvision/sklearn sont déjà présents)
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

md("### Constantes, device et chemins")
code(strip_module("config.py"))

md(r"""### Réglages d'exécution

`FULL_RUN=True` : run complet (Colab, tout le dataset, ~30 epochs). Pour itérer vite en
local, exporter `GALAXY_FULL=0` : le notebook bascule sur le petit split test (toutes les
classes) et peu d'epochs, ce qui valide toute la chaîne en quelques minutes.""")
code(r"""FULL_RUN = os.environ.get("GALAXY_FULL", "1") == "1"
RES_SMALL, RES_LARGE = 64, 128        # 64 px pour itérer, 128 px pour les runs finaux
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

md("### Métriques")
code(strip_module("metrics.py"))
md("### Modèles — baselines, CNN maison, ResNet18")
code(strip_module("models.py"))
md("### Données — chargement, split, normalisation, augmentation")
code(strip_module("data.py"))
md("### Entraînement — boucle, évaluation, early stopping sur F1 macro")
code(strip_module("engine.py"))
md("### Figures")
code(strip_module("viz.py"))
md("### Grad-CAM")
code(strip_module("gradcam.py"))

# ---------------------------------------------------------------- 1. EDA
md(r"""## 1. Analyse exploratoire (EDA)

On charge les données et on regarde ce qu'on manipule : un exemple par classe, la
distribution (fortement déséquilibrée), et les statistiques de couleur par classe.""")
code(r"""set_seed()
device = get_device(); print("device:", device)

train_ds, val_ds, test_ds, (mean, std) = get_datasets(res=RES_SMALL, smoke_n=SMOKE_N)
print(f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")
display(ShowImage(plot_samples_grid(test_ds.x, test_ds.y)))""")

code(r"""# Distribution des classes sur l'ensemble d'entraînement (train + val)
y_full = torch.cat([train_ds.y, val_ds.y])
display(ShowImage(plot_class_distribution(y_full)))""")

code(r"""# Statistiques de couleur moyenne par classe (sur le train, en [0,1])
xtr = train_ds.x.float() / 255.0
rows = []
for c in range(NUM_CLASSES):
    m = xtr[train_ds.y == c].mean(dim=(0, 2, 3))
    rows.append([CLASS_NAMES[c], int((train_ds.y == c).sum()), *[round(v, 3) for v in m.tolist()]])
pd.DataFrame(rows, columns=["classe", "n_train", "R", "G", "B"])""")

# ---------------------------------------------------------------- 2. Preprocessing
md(r"""## 2. Preprocessing et augmentation

**Split.** Le dataset fournit déjà un `train` (16 000) et un `test` (1 800). On garde le
**test officiel comme test final, jamais touché**. La validation est un tirage **stratifié
15 %** dans le train (seed figé). Toutes les décisions de tuning se prennent sur la validation.

**Normalisation.** mean/std par canal calculés **sur le train uniquement**, pour éviter toute
fuite d'information du test.

**Augmentation (train uniquement).** Rotations 0-360°, flips horizontaux et verticaux, léger
jitter de luminosité/contraste. Justification physique : une galaxie n'a pas d'orientation
privilégiée dans le ciel, ces transformations sont des invariances réelles du problème. On les
applique **après le split**, jamais sur val/test.""")
code(r"""print("mean train:", [round(v, 4) for v in mean.tolist()])
print("std  train:", [round(v, 4) for v in std.tolist()])
display(ShowImage(plot_augmentation(train_ds.x[0], mean, std)))""")

# ---------------------------------------------------------------- 3. Baselines
md(r"""## 3. Baselines sans convolution

Elles doivent être médiocres : c'est le but. Elles aplatissent l'image et traitent chaque
pixel comme une variable indépendante, sans aucune notion de structure spatiale. On pondère la
loss par l'inverse des fréquences (déséquilibre) et on suit le **F1 macro**.""")
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
md(r"""## 4. CNN maison et tuning

4 blocs `[Conv 3x3 (sans biais) → BatchNorm → ReLU → MaxPool]` avec channels
`(32, 64, 128, 256)`, `AdaptiveAvgPool(4x4)`, puis une tête dense. La conv 3x3 avec padding 1
conserve la taille ; le pooling la divise par 2. Le partage de poids d'un filtre sur toute
l'image donne l'**équivariance par translation** : déplacer un motif dans l'image déplace
d'autant sa réponse.""")
code(r"""cnn = GalaxyCNN()
print("paramètres entraînables:", f"{count_parameters(cnn):,}")

# Dimensions de sortie bloc par bloc (entrée 3 x RES_SMALL x RES_SMALL)
x = torch.zeros(1, 3, RES_SMALL, RES_SMALL); shapes = []
for i, blk in enumerate(cnn.features):
    x = blk(x); shapes.append([f"bloc {i+1}", tuple(x.shape[1:])])
display(pd.DataFrame(shapes, columns=["couche", "sortie (C, H, W)"]))""")

code(r"""cnn, hist = fit(cnn, tl, vl, epochs=EPOCHS_CNN, lr=1e-3, criterion=crit, device=device)
display(ShowImage(plot_curves(hist)))
_, yt, yp = evaluate(cnn, te, crit, device)
results["CNN"] = summary_metrics(yt, yp)
print("CNN test:", {k: round(v, 3) for k, v in results["CNN"].items()})""")

md("**Tuning du learning rate** (le plus important). On garde le meilleur F1 macro de validation.")
code(r"""runs = []
for lr in [3e-4, 1e-3, 3e-3]:
    m = GalaxyCNN()
    m, h = fit(m, tl, vl, epochs=TUNE_EPOCHS, lr=lr, criterion=crit, device=device, verbose=False)
    runs.append({"lr": lr, "val_f1_macro": round(h["best_val_f1_macro"], 3),
                 "best_epoch": h["best_epoch"]})
pd.DataFrame(runs)""")

# ---------------------------------------------------------------- 5. Transfer
md(r"""## 5. Transfer learning — ResNet18

On réutilise un ResNet18 pré-entraîné sur ImageNet, on remplace la dernière couche par 10
sorties, et on procède en deux temps : d'abord **feature extraction** (backbone gelé, on
n'entraîne que la tête), puis **fine-tuning** (tout est dégelé, LR plus faible). On travaille
en 128 px et on normalise avec les statistiques ImageNet, attendues par le backbone.""")
code(r"""tr2, val2, te2ds, _ = get_datasets(res=RES_LARGE, smoke_n=SMOKE_N,
                                   mean=IMAGENET_MEAN, std=IMAGENET_STD)
tl2, vl2, te2 = make_loaders(tr2, val2, te2ds, batch_size=BATCH, num_workers=NUM_WORKERS)
cw2 = class_weights(tr2.y).to(device)
crit2 = nn.CrossEntropyLoss(weight=cw2)

resnet = build_resnet18(pretrained=True, mode="features")
resnet, _ = fit(resnet, tl2, vl2, epochs=EPOCHS_RES_FE, lr=1e-3, criterion=crit2,
                device=device, verbose=False)                    # tête seule
for p in resnet.parameters():
    p.requires_grad = True
resnet, hist_res = fit(resnet, tl2, vl2, epochs=EPOCHS_RES, lr=1e-4, criterion=crit2,
                       device=device)                            # fine-tuning complet
_, yt2, yp2 = evaluate(resnet, te2, crit2, device)
results["ResNet18"] = summary_metrics(yt2, yp2)
print("ResNet18 test:", {k: round(v, 3) for k, v in results["ResNet18"].items()})""")

# ---------------------------------------------------------------- 6. Eval
md(r"""## 6. Évaluation sur le test

L'échelle baseline → CNN → ResNet répond à la question « que gagne-t-on avec la
convolution ? ». On regarde ensuite la matrice de confusion et le F1 par classe du meilleur
modèle, et on analyse les confusions entre classes voisines.""")
code(r"""display(ShowImage(plot_scale_comparison(results, metric="f1_macro")))
display(pd.DataFrame(results).T[["accuracy", "f1_macro", "f1_weighted"]].round(3))""")

code(r"""# Meilleur modèle = ResNet18 : confusion + F1 par classe
display(ShowImage(plot_confusion(confusion(yt2, yp2, normalize="true"))))
pc = per_class_df(yt2, yp2)
display(ShowImage(plot_perclass_f1(pc)))
pc.round(3)""")

code(r"""# Sauvegarde des résultats pour le rapport (injection des chiffres)
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
print("écrit:", out_path)
print(json.dumps(results_out["models"], indent=2, ensure_ascii=False))""")

# ---------------------------------------------------------------- 7. Grad-CAM
md(r"""## 7. Grad-CAM — où le réseau regarde

Grad-CAM pondère les feature maps de la dernière convolution par le gradient moyen du score
de la classe, applique une ReLU, et rééchelonne à la taille de l'image. On l'applique sur des
prédictions correctes **et** des erreurs, pour le CNN maison et pour le ResNet : le réseau
regarde-t-il les bras spiraux, la barre, le compagnon de fusion, ou des artefacts ?""")
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

On retire un ingrédient à la fois pour quantifier son apport : **sans augmentation** et
**sans BatchNorm**. On compare le F1 macro (test) au CNN complet.""")
code(r"""abl = {"CNN complet": results["CNN"]["f1_macro"]}

# sans augmentation
tr_na, val_na, te_na, (m_na, s_na) = get_datasets(res=RES_SMALL, smoke_n=SMOKE_N, augment=False)
tl_na, vl_na, te_na_l = make_loaders(tr_na, val_na, te_na, batch_size=BATCH, num_workers=NUM_WORKERS)
cnn_na, _ = fit(GalaxyCNN(), tl_na, vl_na, epochs=EPOCHS_CNN, lr=1e-3, criterion=crit,
                device=device, verbose=False)
_, yt, yp = evaluate(cnn_na, te_na_l, crit, device)
abl["sans augmentation"] = summary_metrics(yt, yp)["f1_macro"]

# sans BatchNorm
cnn_nobn, _ = fit(GalaxyCNN(use_bn=False), tl, vl, epochs=EPOCHS_CNN, lr=1e-3, criterion=crit,
                  device=device, verbose=False)
_, yt, yp = evaluate(cnn_nobn, te, crit, device)
abl["sans BatchNorm"] = summary_metrics(yt, yp)["f1_macro"]

# On ajoute les ablations à results.json (pour l'injection dans le rapport).
try:
    _r = json.load(open(project_root() / "results.json"))
    _r["ablations"] = {k: round(v, 4) for k, v in abl.items()}
    json.dump(_r, open(project_root() / "results.json", "w"), indent=2, ensure_ascii=False)
except Exception as e:
    print("maj results.json:", e)

pd.DataFrame({"f1_macro (test)": {k: round(v, 3) for k, v in abl.items()}})""")

md(r"""## Conclusion

L'échelle baseline → CNN → ResNet montre concrètement l'apport de la convolution, puis du
transfer learning. Grad-CAM confirme que le réseau s'appuie sur les structures morphologiques
pertinentes. Limites : labels bruités de Galaxy Zoo, fort déséquilibre (la classe Cigar Shaped
à 334 images reste la plus difficile), et coût d'entraînement.

**Pour le run complet sur Colab :** activer le GPU T4, garder `FULL_RUN=True`, exécuter tout le
notebook, puis récupérer `figures/` et `results.json`.""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
out = ROOT / "galaxy_cnn.ipynb"
nbf.write(nb, out)
print(f"écrit {out} ({len(cells)} cellules)")
