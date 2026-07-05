# Classification morphologique de galaxies par CNN

Galaxy10 DECaLS (17 736 images, 10 classes). Trois niveaux de modèles pour isoler
l'apport de la convolution : baselines sans convolution (régression logistique, MLP),
CNN maison (~1,44 M paramètres), et ResNet18 pré-entraîné. Interprétation par Grad-CAM,
ablations sans augmentation / sans BatchNorm.

Projet : Patrick Yvan Eyike, Philémon Vittet, Sidi Hamadi, Abdelrahman Dahy, Inès Lylia Zerdani.

## Contenu

```
galaxy/            package (data, models, engine, metrics, viz, gradcam)
galaxy_cnn.ipynb   notebook auto-portant, sections 0-8 (le rendu à exécuter)
run_full.py        run complet headless -> figures/ + results.json
smoke_test.py      vérif rapide du pipeline (petit split, quelques epochs)
inject_results.py  injecte results.json dans les .tex
report/rapport.tex rapport LaTeX
slides/            presentation.tex (Beamer) + script.md
figures/           figures générées
docs/…/*-design.md design doc
```

## Lancer en local (Mac, GPU MPS)

```bash
uv sync
uv run python smoke_test.py          # vérif ~1 min (petit split test)
uv run python run_full.py            # run complet -> figures/ + results.json
```

## Lancer sur Colab (GPU T4)

1. Activer le GPU : Exécution → Modifier le type d'exécution → T4.
2. Ouvrir `galaxy_cnn.ipynb`, décommenter `!pip install datasets` dans la première cellule.
3. Exécuter tout (`FULL_RUN=True`). Récupérer ensuite `figures/` et `results.json`.

Pour itérer vite en local sans télécharger les 2,3 Go, exporter `GALAXY_FULL=0` avant de
lancer le notebook : il bascule sur le petit split test.

## Rapport et slides

```bash
uv run python inject_results.py                    # chiffres -> rapport.tex + presentation.tex
latexmk -pdf -cd report/rapport.tex
latexmk -pdf -cd slides/presentation.tex
```
