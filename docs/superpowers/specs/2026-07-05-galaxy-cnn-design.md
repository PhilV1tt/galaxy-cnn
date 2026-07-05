# Design — Galaxy10 DECaLS : classification morphologique par CNN

Date : 2026-07-05. Échéance : 2026-07-06 (soir). Sprint ~1 jour.
Groupe (5) : Patrick Yvan Eyike, Philémon Vittet, Sidi Hamadi, Abdelrahman Dahy, Ines Lylia Zerdani.

## 1. Objectif

Classifier la morphologie de galaxies (10 classes) à partir d'images couleur, et
raconter une histoire claire en trois niveaux : **que gagne-t-on avec la convolution ?**
Baseline sans convolution (régression logistique, MLP) < CNN maison < transfer learning
(ResNet18 pré-entraîné). Interprétation par Grad-CAM. Lien physique : l'équivariance par
translation du CNN est une symétrie imposée à l'architecture.

## 2. Données

`matthieulel/galaxy10_decals` (Hugging Face). 17 736 images 256x256 RGB (bandes g, r, z),
10 classes. Labels Galaxy Zoo, relevé DESI Legacy Imaging Surveys. Schéma : `image` (PIL),
`label` (uint8 0-9).

Effectifs par classe (déséquilibre marqué, facteur ~8) :

| id | classe | n |
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

**Split.** Le dataset fournit déjà `train` (16 000) et `test` (1 800). On garde le **test
officiel comme test final, jamais touché**. On découpe un **validation stratifié 15 %** dans
le train (seed figé). Toutes les décisions de tuning se prennent sur la validation.

**Préprocessing.** Redimensionnement 256 -> 64 px (itération, baselines, premiers CNN) puis
128 px (runs finaux CNN + ResNet). Normalisation par mean/std calculés **sur le train
uniquement** (par canal). Cache des tenseurs décodés en `.pt` pour éviter de re-décoder 2,3 Go
à chaque run.

**Augmentation (train uniquement).** Rotation aléatoire 0-360°, flips H et V, léger jitter
luminosité/contraste. Justification physique : une galaxie n'a pas d'orientation privilégiée,
ces transformations sont des invariances réelles du problème. Appliquée après le split, jamais
sur val/test (pas de fuite de données).

## 3. Modèles

**Baseline 1 — Régression logistique** sur pixels aplatis (3x64x64 = 12288 features).
softmax multinomiale. Volontairement médiocre : aucune notion de structure spatiale.

**Baseline 2 — MLP** 2 couches cachées (512, 256) + ReLU + dropout sur pixels aplatis.
Un peu mieux, toujours aveugle à la localité.

**CNN maison** (entrée 3x64x64) :

```
Bloc 1 : Conv(3->32, 3x3, pad 1)  -> BN -> ReLU -> MaxPool 2   => 32 x 32 x 32
Bloc 2 : Conv(32->64, 3x3, pad 1) -> BN -> ReLU -> MaxPool 2   => 64 x 16 x 16
Bloc 3 : Conv(64->128, 3x3, pad 1)-> BN -> ReLU -> MaxPool 2   => 128 x 8 x 8
Bloc 4 : Conv(128->256,3x3, pad 1)-> BN -> ReLU -> MaxPool 2   => 256 x 4 x 4
AdaptiveAvgPool(4x4) (fixe la tête quelle que soit la résolution d'entrée)
Tête   : Flatten(4096) -> Dropout -> Linear(4096->256) -> ReLU -> Dropout -> Linear(256->10)
```

Nombre de paramètres (à refaire à la main pour la soutenance) :
- Convs : 896 + 18 496 + 73 856 + 295 168 = 388 416
- BatchNorm : 64 + 128 + 256 + 512 = 960
- FC : (4096x256 + 256) + (256x10 + 10) = 1 048 832 + 2 570 = 1 051 402
- **Total ≈ 1 440 778 (~1,44 M)**, dominé par la première couche FC.

L'`AdaptiveAvgPool(4x4)` garde ~1,44 M params en 64 comme en 128 px. Sans pooling adaptatif,
passer à 128 px ferait exploser FC1 (16384x256 ≈ 4,2 M) : bon point pédagogique à mentionner.

**Transfer learning — ResNet18** pré-entraîné ImageNet. Deux régimes :
1. *Feature extraction* : on gèle le backbone, on n'entraîne que la nouvelle `fc` (512->10).
2. *Fine-tuning* : on dégèle tout (ou `layer4` + `fc`) avec un LR plus faible.

Entrée 3 canaux compatible, redimensionnement à 128 px, normalisation stats ImageNet pour
le backbone pré-entraîné.

## 4. Entraînement et tuning

- Optimiseur Adam, LR de départ 1e-3 (le plus important à régler), scheduler cosine.
- Loss `CrossEntropyLoss(weight=...)` pondérée par l'inverse des fréquences (déséquilibre).
- Early stopping sur **F1 macro** de la validation (pas l'accuracy).
- Grille légère : LR, batch size, dropout, aug on/off, profondeur. Tableau de runs en pandas.
- `model.eval()` en évaluation (BN et dropout changent de comportement).

## 5. Évaluation (sur le test officiel, jamais touché avant)

Accuracy, precision/recall/F1 par classe + macro et pondéré, matrice de confusion, courbes
d'apprentissage train/val. Analyse des confusions entre classes voisines (spirales serrées vs
lâches, rondes vs cigare, edge-on avec/sans bulbe).

## 6. Interprétation — Grad-CAM

Implémenté à la main (~30 lignes) : hook forward/backward sur la dernière conv (CNN maison)
et sur `layer4` (ResNet18). Carte de chaleur = ReLU(somme pondérée des feature maps par le
gradient moyen de la classe). Heatmaps sur bonnes prédictions ET erreurs : le réseau regarde-t-il
les bras spiraux, la barre, le compagnon de fusion, ou des artefacts ?

## 7. Ablations

- Sans augmentation.
- Sans BatchNorm.
Quantifier la perte de F1 macro pour isoler l'apport de chaque ingrédient.

## 8. Livrables (miroir de `../cifar/`)

```
final/
  pyproject.toml            uv : torch, torchvision, datasets, scikit-learn, matplotlib, pandas
  galaxy_cnn.ipynb          sections 0-8
  galaxy/                   package testable (data, models, train, metrics, gradcam)
  figures/                  samples, augment, courbes, confusion, perclass, echelle, gradcam
  report/rapport.tex        + rapport.pdf (LaTeX)
  slides/presentation.tex   + script.md + presentation.pdf (Beamer)
  data/                     gitignoré (cache .pt local / Drive sur Colab)
```

Sections du notebook : 0 Chargement, 1 EDA, 2 Preprocessing et augmentation, 3 Baselines,
4 CNN maison et tuning, 5 Transfer learning, 6 Évaluation test, 7 Grad-CAM, 8 Ablations.

Le notebook détecte Colab vs local (device CUDA/MPS, chemin data) et tourne aux deux endroits.

## 9. Plan d'exécution (1 jour, sprint parallèle)

1. Scaffold repo + env uv + package `galaxy/`. Verif : `uv sync` OK, imports OK.
2. Smoke test local (streaming ~600 images, CNN 64px, 1-2 epochs sur MPS). Verif : boucle
   train tourne, une figure produite. Garantit un notebook runnable.
3. Notebook sections 0-8 orchestrant le package. Verif : exécution locale basse échelle.
4. En parallèle : rapport LaTeX + slides Beamer + script.md avec placeholders pour les chiffres.
5. Run complet sur Colab T4 (128px, ResNet, tuning). Récupérer figures + `results.json`.
6. Injecter les vrais chiffres dans rapport/slides. Compiler les PDF.

## 10. Pièges (rappel)

Augmentation après le split, train uniquement. Test jamais touché pendant le tuning. F1 macro,
pas seulement accuracy. Normaliser avec stats train. LR 1e-3 pour démarrer. Itérer en 64/128,
pas 256 d'emblée. `model.eval()` à l'évaluation. Checkpoints réguliers sur Colab.
