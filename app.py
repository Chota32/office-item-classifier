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


# Load training and validation datasets
def load_datasets(batch_size=32):
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
        ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
        ])

    train_dataset = datasets.ImageFolder("data/train", transform=train_transforms)
    val_dataset = datasets.ImageFolder("data/val", transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


@command("train", "Train and validate the classifier",
[
    (("--epochs",), dict(type=int, default=5)),
    (("--lr",), dict(type=float, default=0.001))
])

def cmd_train(args):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader = load_datasets()

    model = create_model().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier[-1].parameters(), lr=args.lr)

    best_accuracy = 0.0

    for epoch in range(args.epochs):
        model.eval()
        model.classifier[-1].train()
        train_loss, train_correct = 0, 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(images)
            train_correct += (outputs.argmax(1) == labels).sum().item()

        model.eval()
        val_loss, val_correct = 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = loss_fn(outputs, labels)

                val_loss += loss.item() * len(images)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        val_accuracy = val_correct / len(val_loader.dataset)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy

            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": train_loader.dataset.classes,
                "model_name": "mobilenet_v3_small",
                "image_size": 224
                }, MODEL_PATH)

        print(f"Model saved to {MODEL_PATH}")

        train_count = len(train_loader.dataset)
        val_count = len(val_loader.dataset)

        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"Training loss: {train_loss / train_count:.4f}, accuracy: {train_correct / train_count:.2%}")
        print(f"Validation loss: {val_loss / val_count:.4f}, accuracy: {val_correct / val_count:.2%}")


@command("evaluate", "Evaluate model on the held-out test set", [
    (("--checkpoint",), dict(default=MODEL_PATH)),
    (("--test-dir",), dict(default="data/test")),
    (("--output-dir",), dict(default="evaluation_results")),
    (("--batch-size",), dict(type=int, default=32)),
])
def cmd_evaluate(args):
    from evaluate import run_evaluation

    run_evaluation(
        checkpoint_path=args.checkpoint,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )


@command("robustness", "Test model robustness to image perturbations", [
    (("--checkpoint",), dict(default=MODEL_PATH)),
    (("--test-dir",), dict(default="data/test")),
    (("--output",), dict(default="robustness_results.json")),
    (("--batch-size",), dict(type=int, default=32)),
])
def cmd_robustness(args):
    from robustness import run_robustness

    run_robustness(
        checkpoint_path=args.checkpoint,
        test_dir=args.test_dir,
        output_path=args.output,
        batch_size=args.batch_size,
    )


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