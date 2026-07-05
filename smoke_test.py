"""Quick pipeline check (without downloading the 2.3 GB train set).

Loads the small official test split, trains 3 epochs of a mini-CNN 64px, evaluates,
and produces all figures + a Grad-CAM map. Used to validate the environment.

    uv run python smoke_test.py
"""
import matplotlib
matplotlib.use("Agg")

import torch

from galaxy import config, data, engine, gradcam, metrics, models, viz

config.set_seed()
dev = config.get_device()
print(f"[env] device={dev} torch={torch.__version__}")

train_ds, val_ds, test_ds, (mean, std) = data.get_datasets(res=64, smoke_n=800)
print(f"[data] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
      f"mean={[round(m, 3) for m in mean.tolist()]}")
tl, vl, te = data.make_loaders(train_ds, val_ds, test_ds, batch_size=64)

xb, _ = next(iter(tl))
for name, m in [
    ("logreg", models.LogisticRegression(image_size=64)),
    ("mlp", models.MLP(image_size=64)),
    ("cnn", models.GalaxyCNN()),
    ("resnet18", models.build_resnet18(pretrained=False, mode="finetune")),
]:
    out = m(xb)
    print(f"[model] {name:9s} params={models.count_parameters(m):>10,d} out={tuple(out.shape)}")

cnn = models.GalaxyCNN()
crit = torch.nn.CrossEntropyLoss(weight=engine.class_weights(train_ds.y).to(dev))
cnn, hist = engine.fit(cnn, tl, vl, epochs=3, patience=5, device=dev, criterion=crit)

_, yt, yp = engine.evaluate(cnn, te, crit, dev)
print("[test]", {k: round(v, 3) for k, v in metrics.summary_metrics(yt, yp).items()})

viz.plot_samples_grid(test_ds.x, test_ds.y)
viz.plot_class_distribution(train_ds.y)
viz.plot_augmentation(train_ds.x[0], mean, std)
viz.plot_curves(hist)
viz.plot_confusion(metrics.confusion(yt, yp, normalize="true"))
viz.plot_perclass_f1(metrics.per_class_df(yt, yp))
gradcam.plot_gradcam_grid(cnn, cnn.last_conv, test_ds.x[:4], mean, std,
                          y_true=test_ds.y[:4], device=dev)
print("[figures] samples, distribution, augment, curves, confusion, perclass, gradcam -> figures/")
print("SMOKE OK")
