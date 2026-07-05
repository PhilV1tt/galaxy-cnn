"""Run complet headless (le même pipeline que le notebook) pour produire les vrais
chiffres et figures. Streame la progression par epoch. Écrit figures/ + results.json.

    uv run python run_full.py
"""
import json

import matplotlib
matplotlib.use("Agg")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from galaxy import config, data, engine, gradcam, metrics, models, viz
from galaxy.config import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES, project_root

config.set_seed()
dev = config.get_device()
NW = 2 if torch.cuda.is_available() else 0
RES_S, RES_L, BATCH = 64, 128, 128
print(f"[full] device={dev} num_workers={NW}", flush=True)

# --- Données 64 px ---
tr, va, teds, (mean, std) = data.get_datasets(res=RES_S)
tl, vl, te = data.make_loaders(tr, va, teds, BATCH, NW)
cw = engine.class_weights(tr.y).to(dev)
crit = nn.CrossEntropyLoss(weight=cw)
print(f"[full] train={len(tr)} val={len(va)} test={len(teds)}", flush=True)

results = {}

print("== baselines ==", flush=True)
for name, mk in [("LogReg", lambda: models.LogisticRegression(image_size=RES_S)),
                 ("MLP", lambda: models.MLP(image_size=RES_S))]:
    m, _ = engine.fit(mk(), tl, vl, epochs=15, lr=1e-3, criterion=crit,
                      scheduler=None, device=dev, verbose=False)
    _, yt, yp = engine.evaluate(m, te, crit, dev)
    results[name] = metrics.summary_metrics(yt, yp)
    print(f"[full] {name} test F1={results[name]['f1_macro']:.3f}", flush=True)

print("== CNN maison ==", flush=True)
cnn = models.GalaxyCNN()
cnn, hist = engine.fit(cnn, tl, vl, epochs=30, lr=1e-3, criterion=crit, device=dev)
_, yt, yp = engine.evaluate(cnn, te, crit, dev)
results["CNN"] = metrics.summary_metrics(yt, yp)
viz.plot_curves(hist)

print("== tuning LR ==", flush=True)
runs = []
for lr in [3e-4, 1e-3, 3e-3]:
    m, h = engine.fit(models.GalaxyCNN(), tl, vl, epochs=10, lr=lr, criterion=crit,
                      device=dev, verbose=False)
    runs.append({"lr": lr, "val_f1_macro": round(h["best_val_f1_macro"], 3),
                 "best_epoch": h["best_epoch"]})
    print(f"[full] lr={lr} val F1={h['best_val_f1_macro']:.3f}", flush=True)

print("== ResNet18 (128 px) ==", flush=True)
tr2, va2, te2ds, _ = data.get_datasets(res=RES_L, mean=IMAGENET_MEAN, std=IMAGENET_STD)
tl2, vl2, te2 = data.make_loaders(tr2, va2, te2ds, BATCH, NW)
cw2 = engine.class_weights(tr2.y).to(dev)
crit2 = nn.CrossEntropyLoss(weight=cw2)
res = models.build_resnet18(pretrained=True, mode="features")
res, _ = engine.fit(res, tl2, vl2, epochs=6, lr=1e-3, criterion=crit2, device=dev, verbose=False)
for p in res.parameters():
    p.requires_grad = True
res, hres = engine.fit(res, tl2, vl2, epochs=12, lr=1e-4, criterion=crit2, device=dev)
_, yt2, yp2 = engine.evaluate(res, te2, crit2, dev)
results["ResNet18"] = metrics.summary_metrics(yt2, yp2)
print(f"[full] ResNet18 test F1={results['ResNet18']['f1_macro']:.3f}", flush=True)

print("== figures ==", flush=True)
viz.plot_samples_grid(teds.x, teds.y)
viz.plot_class_distribution(torch.cat([tr.y, va.y]))
viz.plot_augmentation(tr.x[0], mean, std)
viz.plot_scale_comparison(results, "f1_macro")
viz.plot_confusion(metrics.confusion(yt2, yp2, normalize="true"))
pc = metrics.per_class_df(yt2, yp2)
viz.plot_perclass_f1(pc)


def pick(model, ds, nok=3, nbad=3):
    model.eval()
    ps = []
    with torch.no_grad():
        for xb, _ in DataLoader(ds, batch_size=256):
            ps.append(model(xb.to(dev)).argmax(1).cpu())
    ps = torch.cat(ps)
    ok = (ps == ds.y).nonzero().flatten().tolist()
    bad = (ps != ds.y).nonzero().flatten().tolist()
    return ok[:nok] + bad[:nbad]


print("== grad-cam ==", flush=True)
idx = pick(cnn, teds)
gradcam.plot_gradcam_grid(cnn, cnn.last_conv, teds.x[idx], mean, std,
                          y_true=teds.y[idx], device=dev, name="gradcam_cnn.png")
idx2 = pick(res, te2ds)
gradcam.plot_gradcam_grid(res, res.layer4[-1], te2ds.x[idx2], IMAGENET_MEAN, IMAGENET_STD,
                          y_true=te2ds.y[idx2], device=dev, name="gradcam_resnet.png")

print("== ablations ==", flush=True)
abl = {"CNN complet": results["CNN"]["f1_macro"]}
trna, vana, tenads, _ = data.get_datasets(res=RES_S, augment=False)
tlna, vlna, tena = data.make_loaders(trna, vana, tenads, BATCH, NW)
m, _ = engine.fit(models.GalaxyCNN(), tlna, vlna, epochs=30, lr=1e-3, criterion=crit,
                  device=dev, verbose=False)
_, yt, yp = engine.evaluate(m, tena, crit, dev)
abl["sans augmentation"] = metrics.summary_metrics(yt, yp)["f1_macro"]
m, _ = engine.fit(models.GalaxyCNN(use_bn=False), tl, vl, epochs=30, lr=1e-3, criterion=crit,
                  device=dev, verbose=False)
_, yt, yp = engine.evaluate(m, te, crit, dev)
abl["sans BatchNorm"] = metrics.summary_metrics(yt, yp)["f1_macro"]
print(f"[full] ablations {abl}", flush=True)

out = {
    "models": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in results.items()},
    "cnn_params": models.count_parameters(models.GalaxyCNN()),
    "cnn_best_epoch": hist["best_epoch"],
    "resnet_best_epoch": hres["best_epoch"],
    "per_class_f1_resnet": {CLASS_NAMES[i]: round(float(pc["f1"].iloc[i]), 3)
                            for i in range(NUM_CLASSES)},
    "tuning_lr": runs,
    "ablations": {k: round(v, 4) for k, v in abl.items()},
    "config": {"res_small": RES_S, "res_large": RES_L, "batch": BATCH, "full_run": True},
}
json.dump(out, open(project_root() / "results.json", "w"), indent=2, ensure_ascii=False)
print("DONE", flush=True)
print(json.dumps(out["models"], indent=2, ensure_ascii=False), flush=True)
