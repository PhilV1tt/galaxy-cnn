"""Inject the numbers from results.json into the RESULTS blocks of rapport.tex and
presentation.tex, then (optionally) leave compilation to the Makefile / latexmk.

    uv run python inject_results.py
"""
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
res = json.load(open(ROOT / "results.json"))
M = res["models"]


def g(model, key):
    return M.get(model, {}).get(key)


def fr(x, d=3):
    return "--" if x is None else f"{x:.{d}f}".replace(".", "{,}")


def en(x, d=2):
    return "--" if x is None else f"{x:.{d}f}"


def lr_tex(sci_dot=False):
    runs = res.get("tuning_lr", [])
    if not runs:
        return "1\\times10^{-3}" if not sci_dot else "1e{-}3"
    best = max(runs, key=lambda r: r["val_f1_macro"])["lr"]
    e = round(math.log10(best))
    m = int(round(best / 10 ** e))
    return f"{m}e{{{e}}}" if sci_dot else f"{m}\\times10^{{{e}}}"


pc = res.get("per_class_f1_resnet", {})
best_class = max(pc, key=pc.get) if pc else "--"
worst_class = min(pc, key=pc.get) if pc else "--"
abl = res.get("ablations", {})
cnnp = res.get("cnn_params", 1440298)

# Report and slides are in English: decimal points everywhere (en()).
report_block = "\n".join([
    rf"\newcommand{{\accLogReg}}{{{en(g('LogReg','accuracy'), 3)}}}",
    rf"\newcommand{{\fLogReg}}{{{en(g('LogReg','f1_macro'), 3)}}}",
    rf"\newcommand{{\fwLogReg}}{{{en(g('LogReg','f1_weighted'), 3)}}}",
    rf"\newcommand{{\accMLP}}{{{en(g('MLP','accuracy'), 3)}}}",
    rf"\newcommand{{\fMLP}}{{{en(g('MLP','f1_macro'), 3)}}}",
    rf"\newcommand{{\fwMLP}}{{{en(g('MLP','f1_weighted'), 3)}}}",
    rf"\newcommand{{\accCNN}}{{{en(g('CNN','accuracy'), 3)}}}",
    rf"\newcommand{{\fCNN}}{{{en(g('CNN','f1_macro'), 3)}}}",
    rf"\newcommand{{\fwCNN}}{{{en(g('CNN','f1_weighted'), 3)}}}",
    rf"\newcommand{{\accResnet}}{{{en(g('ResNet18','accuracy'), 3)}}}",
    rf"\newcommand{{\fResnet}}{{{en(g('ResNet18','f1_macro'), 3)}}}",
    rf"\newcommand{{\fwResnet}}{{{en(g('ResNet18','f1_weighted'), 3)}}}",
    rf"\newcommand{{\bestlr}}{{{lr_tex()}}}",
    rf"\newcommand{{\cnnparams}}{{{cnnp:,}}}".replace(",", "\\,"),
    rf"\newcommand{{\bestclass}}{{{best_class}}}",
    rf"\newcommand{{\worstclass}}{{{worst_class}}}",
    rf"\newcommand{{\ablbase}}{{{en(abl.get('full CNN'), 3)}}}",
    rf"\newcommand{{\ablnoaug}}{{{en(abl.get('no augmentation'), 3)}}}",
    rf"\newcommand{{\ablnobn}}{{{en(abl.get('no BatchNorm'), 3)}}}",
]) + "\n"

slides_block = "\n".join([
    rf"\newcommand{{\fLogReg}}{{{en(g('LogReg','f1_macro'))}}}",
    rf"\newcommand{{\fMLP}}{{{en(g('MLP','f1_macro'))}}}",
    rf"\newcommand{{\accCNN}}{{{en(g('CNN','accuracy'))}}}",
    rf"\newcommand{{\fCNN}}{{{en(g('CNN','f1_macro'))}}}",
    rf"\newcommand{{\accResnet}}{{{en(g('ResNet18','accuracy'))}}}",
    rf"\newcommand{{\fResnet}}{{{en(g('ResNet18','f1_macro'))}}}",
    rf"\newcommand{{\bestlr}}{{{lr_tex(sci_dot=True)}}}",
    rf"\newcommand{{\cnnparams}}{{{cnnp/1e6:.2f}\,M}}",
    rf"\newcommand{{\ablbase}}{{{en(abl.get('full CNN'))}}}",
    rf"\newcommand{{\ablnoaug}}{{{en(abl.get('no augmentation'))}}}",
    rf"\newcommand{{\ablnobn}}{{{en(abl.get('no BatchNorm'))}}}",
]) + "\n"

pat = re.compile(r"(% ==== BEGIN RESULTS.*?\n)(.*?)(% ==== END RESULTS ====)", re.DOTALL)
for path, block in [(ROOT / "report/rapport.tex", report_block),
                    (ROOT / "slides/presentation.tex", slides_block)]:
    txt = path.read_text()
    new = pat.sub(lambda m: m.group(1) + block + m.group(3), txt)
    path.write_text(new)
    print("injected:", path.name)
