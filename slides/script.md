# Speaker script: Galaxy morphology with a CNN

About 6 minutes, 10 slides. Short notes per slide.

## Slide 1: Title

This project classifies the morphology of galaxies with a convolutional neural network.
The data is Galaxy10 DECaLS: real astrophysics images, ten morphological classes. The
question is whether a CNN can reproduce the sorting that a hundred thousand Galaxy Zoo
volunteers did by hand.

## Slide 2: The problem

Galaxy Zoo is a citizen-science project. Volunteers looked at images and voted on shape.
We have about seventeen thousand color images, 256 by 256, from the DESI Legacy Imaging
Surveys, in ten classes: smooth galaxies, spirals that are barred, tight or loose, edge-on
galaxies, and mergers. The goal is to automate that labeling.

## Slide 3: The data, and why it is hard

Two things make it hard. First, strong class imbalance. The rarest class, cigar-shaped, has
334 images. The most common has more than two and a half thousand, a factor of eight.
Second, neighboring classes look alike: tight versus loose spirals, round versus cigar.
Labels are also noisy, since they are volunteer votes. I keep the official test set
untouched and take the validation from fifteen percent of the training set.

## Slide 4: Convolution in one slide

A convolution slides the same small filter over the whole image and computes a local
weighted sum. Two consequences. Weight sharing: one filter everywhere, so very few
parameters, and each filter becomes a motif detector. And translation equivariance: if the
pattern moves, its response moves with it. That is a symmetry built into the architecture,
exactly the spirit of symmetries in physics. Since a galaxy has no preferred orientation, I
also augment with rotations and flips.

## Slide 5: Our custom CNN

The network is four blocks of convolution, batch norm, ReLU and max-pooling, with channels
32, 64, 128, 256, then adaptive pooling and a dense head. The input is 3 by 64 by 64, halved
at every pooling. It has about 1.4 million parameters, and almost three quarters of them are
in the first dense layer. I train with a class-weighted loss, Adam, a cosine schedule, and
early stopping on macro-F1.

## Slide 6: What convolution buys us

This is the core result. The two baselines, logistic regression and an MLP on raw pixels,
are blind to spatial structure and stay weak, macro-F1 0.26 and 0.40. The custom CNN jumps
to 0.68, and the fine-tuned ResNet18 adds another step to 0.78. Convolution, then
pretraining, explain almost all of the gain.

## Slide 7: Where it fails

The confusion matrix shows the mistakes are not random. They concentrate between neighboring
morphologies: loose versus tight spirals, and disturbed galaxies that spread across several
classes. Disturbed and loose spirals are the hardest, with recall around 0.49 and 0.56. The
smooth and edge-on classes are almost perfect.

## Slide 8: Grad-CAM

To see what the network looks at, I use Grad-CAM. It weights the last convolution's feature
maps by the class gradient and gives a heatmap. On correct predictions the heat sits on the
spiral arms, the central bar, or the merging companion, which is reassuring. On errors it
sometimes latches onto a neighboring object or an artifact.

## Slide 9: Ablations and limits

I removed one ingredient at a time. The full CNN reaches 0.68 macro-F1, without augmentation
0.69, and without batch norm 0.72. So, against intuition, augmentation barely helps here and
removing batch norm even nudges the score up. At 64 pixels and 30 epochs the model does not
overfit much, and a single run hides small gaps in the noise. The main limits are the noisy
Galaxy Zoo labels, the class imbalance, and the training cost, which kept me at 64 and 128
pixels.

## Slide 10: Conclusion

So yes, a CNN can reproduce the morphological sorting. The ResNet reaches accuracy 0.80 and
macro-F1 0.78. The scale from baseline to CNN to ResNet isolates the value of convolution,
and Grad-CAM confirms the network uses real morphological structures. The idea to remember
is that convolution works on images because it builds in a symmetry, translation
equivariance, the same way physics builds in symmetries. Thank you.
