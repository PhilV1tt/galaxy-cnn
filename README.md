# Galaxy morphological classification with CNN

Galaxy10 DECaLS (17 736 images, 10 classes). Three model levels to isolate the gain from
convolution: baselines without convolution (logistic regression, MLP), a custom CNN
(~1,44 M parameters), and a pretrained ResNet18. Interpretation with Grad-CAM, ablations
without augmentation / without BatchNorm.

Project: Patrick Yvan Eyike, Philémon Vittet, Sidi Hamadi, Abdelrahman Dahy, Inès Lylia Zerdani.

## Contents

```
galaxy/            package (data, models, engine, metrics, viz, gradcam)
galaxy_cnn.ipynb   self-contained notebook, sections 0-8 (the deliverable to run)
run_full.py        full headless run -> figures/ + results.json
smoke_test.py      quick pipeline check (small split, a few epochs)
inject_results.py  injects results.json into the .tex files
report/rapport.tex LaTeX report
slides/            presentation.tex (Beamer) + script.md
figures/           generated figures
docs/…/*-design.md design doc
```

## Run locally (Mac, MPS GPU)

```bash
uv sync
uv run python smoke_test.py          # check ~1 min (small test split)
uv run python run_full.py            # full run -> figures/ + results.json
```

## Run on Colab (T4 GPU)

1. Enable the GPU: Runtime → Change runtime type → T4.
2. Open `galaxy_cnn.ipynb`, uncomment `!pip install datasets` in the first cell.
3. Run all (`FULL_RUN=True`). Then collect `figures/` and `results.json`.

To iterate fast locally without downloading the 2,3 GB, export `GALAXY_FULL=0` before
running the notebook: it switches to the small test split.

## Report and slides

```bash
uv run python inject_results.py                    # numbers -> rapport.tex + presentation.tex
latexmk -pdf -cd report/rapport.tex
latexmk -pdf -cd slides/presentation.tex
```
