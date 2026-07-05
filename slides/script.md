# Speaker script — Galaxy morphology with a CNN

About 6 minutes, 10 slides, five speakers (~2 slides each). Numbers in brackets are
filled from the final run.

## Slide 1 — Title  (Patrick)

Our project classifies the morphology of galaxies with a convolutional neural network.
The data is Galaxy10 DECaLS: real astrophysics images, ten morphological classes. The
question we ask is whether a CNN can reproduce the sorting that a hundred thousand Galaxy
Zoo volunteers did by hand.

## Slide 2 — The problem  (Patrick)

Galaxy Zoo is a citizen-science project: volunteers looked at images and voted on shape.
We have about seventeen thousand color images, 256 by 256, from the DESI Legacy Imaging
Surveys, sorted into ten classes: smooth galaxies, spirals that are barred, tight or
loose, edge-on galaxies, and mergers. Our goal is to automate that labeling.

## Slide 3 — The data, and why it is hard  (Philémon)

Two things make it hard. First, strong class imbalance: the rarest class, cigar-shaped,
has 334 images, the most common has more than two and a half thousand, a factor of eight.
Second, neighboring classes really look alike: tight versus loose spirals, round versus
cigar. Labels are also noisy, since they are volunteer votes. We keep the official test
set untouched and take our validation from fifteen percent of the training set.

## Slide 4 — Convolution in one slide  (Philémon)

A convolution slides the same small filter over the whole image and computes a local
weighted sum. Two consequences. Weight sharing: one filter everywhere, so very few
parameters, and each filter becomes a motif detector. And translation equivariance: if
the pattern moves, its response moves with it. That is a symmetry built into the
architecture, exactly the spirit of symmetries in physics. Since a galaxy has no preferred
orientation, we also augment with rotations and flips.

## Slide 5 — Our custom CNN  (Sidi)

The network is four blocks of convolution, batch norm, ReLU and max-pooling, with channels
going 32, 64, 128, 256, then adaptive pooling and a dense head. The input is 3 by 64 by 64,
halved at every pooling. It has about 1.4 million parameters, and interestingly almost
three quarters of them are in the first dense layer. We train with a class-weighted loss,
Adam, a cosine schedule, and early stopping on macro-F1.

## Slide 6 — What convolution buys us  (Sidi)

This is the core result. The two baselines, logistic regression and an MLP on raw pixels,
are blind to spatial structure and stay weak [F1 logreg, F1 mlp]. The custom CNN jumps to
[F1 cnn], and the fine-tuned ResNet18 adds another step to [F1 resnet]. Convolution, then
pretraining, explain almost all of the gain.

## Slide 7 — Where it fails  (Abdelrahman)

The confusion matrix shows the mistakes are not random: they concentrate between
neighboring morphologies, tight versus loose spirals, round versus cigar, edge-on with or
without a bulge. The rare cigar-shaped class stays the hardest, which matches its small
size and visual ambiguity.

## Slide 8 — Grad-CAM  (Abdelrahman)

To see what the network looks at, we use Grad-CAM, which weights the last convolution's
feature maps by the class gradient and gives a heatmap. On correct predictions the heat
sits on the spiral arms, the central bar, or the merging companion, which is reassuring.
On errors it sometimes latches onto a neighboring object or an artifact.

## Slide 9 — Ablations and limits  (Inès)

We removed one ingredient at a time. The full CNN reaches [abl base] macro-F1; without
augmentation it drops to [abl no-aug], and without batch norm to [abl no-bn], so each
piece measurably helps. The main limits are the noisy Galaxy Zoo labels, which cap
accuracy, the class imbalance, and the training cost, which kept us at 64 and 128 pixels.

## Slide 10 — Conclusion  (Inès)

So yes, a CNN can reproduce the morphological sorting: the ResNet reaches accuracy
[acc resnet] and macro-F1 [F1 resnet]. The scale from baseline to CNN to ResNet isolates
the value of convolution, and Grad-CAM confirms the network uses real morphological
structures. The idea to remember is that convolution works on images because it builds in
a symmetry, translation equivariance, the same way physics builds in symmetries. Thank you.
