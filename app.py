"""Office item classifier (PDE3802 Assessment 1, Part B).

Commands:
    python app.py split                       # raw_data/ -> data/{train,val,test}
    python app.py train                       # trains, saves model.pt
    python app.py evaluate                    # accuracy, macro-F1, confusion matrix
    python app.py predict --image a.jpg       # files, --folder, or --camera
    python app.py robustness                  # perturbation tests
    python app.py download                    # optional public images (4 classes)
"""
import argparse
import csv
import json
import random
import shutil
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix, f1_score)
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

MODEL_PATH = "model.pt"
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMG = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

COMMANDS = {}


def command(name, help_text, args=()):
    """Register a CLI command. args = [((flags...), {argparse kwargs}), ...]."""
    def deco(fn):
        COMMANDS[name] = (fn, help_text, args)
        return fn
    return deco


# ------------------------------------------------------------------- split
@command("split", "raw_data/ -> data/{train,val,test} (resized, seeded 70/15/15)", [
    (("--seed",), dict(type=int, default=42)),
    (("--max-side",), dict(type=int, default=640)),
])
def cmd_split(a):
    raw, out = Path("raw_data"), Path("data")
    classes = sorted(d.name for d in raw.iterdir() if d.is_dir()) if raw.exists() else []
    if not classes:
        sys.exit("Put images in raw_data/<class_name>/ first.")
    if out.exists():
        shutil.rmtree(out)
    rng, summary = random.Random(a.seed), {}
    for cls in classes:
        files = sorted(f for f in (raw / cls).iterdir() if f.suffix.lower() in EXTS)
        rng.shuffle(files)
        n = len(files)
        n_tr, n_va = int(n * 0.70), int(n * 0.15)
        parts = {"train": files[:n_tr], "val": files[n_tr:n_tr + n_va], "test": files[n_tr + n_va:]}
        summary[cls] = {k: len(v) for k, v in parts.items()} | {"total": n}
        for split, fl in parts.items():
            dest = out / split / cls
            dest.mkdir(parents=True, exist_ok=True)
            for i, f in enumerate(fl):
                try:
                    img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
                except Exception as e:
                    print(f"skipped {f}: {e}")
                    continue
                img.thumbnail((a.max_side, a.max_side))
                img.save(dest / f"{cls}_{split}_{i:04d}.jpg", quality=90)
    (out / "split_summary.json").write_text(json.dumps({"seed": a.seed, "counts": summary}, indent=2))
    print(f"{'class':<12}{'train':>7}{'val':>6}{'test':>6}{'total':>7}")
    for cls, c in summary.items():
        print(f"{cls:<12}{c['train']:>7}{c['val']:>6}{c['test']:>6}{c['total']:>7}")


# Create pretrained MobileNetV3 Small classifier
def create_model(num_classes=6):

    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)

    # Freeze pretrained layers
    for param in model.parameters():
        param.requires_grad = False

    # Adapt the classifier to the six office item classes
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    return model


# ==================== PASTE NEW COMMANDS ABOVE THIS LINE ====================
def main():
    p = argparse.ArgumentParser(description="Office item classifier")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, (fn, help_text, args) in COMMANDS.items():
        s = sub.add_parser(name, help=help_text)
        for flags, kw in args:
            s.add_argument(*flags, **kw)
        s.set_defaults(func=fn)
    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()