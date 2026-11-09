# Galaxy morphological classification with a CNN

Galaxy10 DECaLS (17 736 images, 10 classes). Three model levels to isolate the gain from
convolution: baselines without convolution (logistic regression, MLP), a custom CNN
(~1.44 M parameters), and a pretrained ResNet18. Interpretation with Grad-CAM, ablations
without augmentation and without BatchNorm.

Project: Philémon Vittet.

## Contents

```
galaxy/            package (data, models, engine, metrics, viz, gradcam)
galaxy_cnn.ipynb   self-contained notebook, sections 0-8 (the deliverable to run)
report/            rapport.tex + rapport.pdf
slides/            presentation.tex + presentation.pdf + script.md
figures/           figures used by the report and slides
```

## Run

The notebook is self-contained. On Colab (T4 GPU):

1. Runtime → Change runtime type → T4.
2. Open `galaxy_cnn.ipynb`, uncomment `!pip install datasets` in the first cell.
3. Run all.

Locally (Mac, MPS GPU):

```bash
uv sync
uv run jupyter lab galaxy_cnn.ipynb
```

To iterate fast without downloading the 2.3 GB dataset, set `GALAXY_FULL=0` before running:
it switches to the small test split.

## Report and slides

```bash
latexmk -pdf -cd report/rapport.tex
latexmk -pdf -cd slides/presentation.tex
```
