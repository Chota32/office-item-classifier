 
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
MODEL_NAME = "mobilenet_v3_small"


def load_checkpoint_model(checkpoint_path, device=None):

    """Create the checkpoint's model architecture and load its saved weights."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid checkpoint format: {checkpoint_path}")

    required = {"model_state_dict", "class_names", "model_name", "image_size"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint is missing required fields: {', '.join(sorted(missing))}")
    if checkpoint["model_name"] != MODEL_NAME:
        raise ValueError(f"Unsupported model: {checkpoint['model_name']}")

    class_names = checkpoint["class_names"]
    if (not isinstance(class_names, list) or not class_names
            or not all(isinstance(name, str) for name in class_names)
            or len(set(class_names)) != len(class_names)):
        raise ValueError("Checkpoint class_names must be a non-empty list of unique strings.")

    image_size = checkpoint["image_size"]
    if not isinstance(image_size, int) or image_size <= 0:
        raise ValueError("Checkpoint image_size must be a positive integer.")

    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_names, image_size, device


def make_test_loader(test_dir, class_names, image_size=224, batch_size=32,
                     perturbation=None):
    
    """Build a deterministic test loader, optionally perturbing source images."""
    steps = []
    if perturbation is not None:
        steps.append(perturbation)
    steps.extend([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    dataset = datasets.ImageFolder(test_dir, transform=transforms.Compose(steps))
    if dataset.classes != class_names:
        raise ValueError(
            "Test dataset classes do not match checkpoint class order. "
            f"Checkpoint: {class_names}; dataset: {dataset.classes}"
        )
    if not dataset:
        raise ValueError(f"No test images found in {test_dir}")
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def calculate_metrics(y_true, y_pred, class_names):

    """Return aggregate and per-class metrics using the checkpoint class order."""
    labels = list(range(len(class_names)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": report,
    }


def evaluate_model(model, data_loader, class_names, device):

    """Run inference and return metrics plus paths of incorrectly classified images."""
    y_true, y_pred = [], []
    model.eval()
    with torch.inference_mode():
        for images, labels in data_loader:
            outputs = model(images.to(device))
            y_true.extend(labels.tolist())
            y_pred.extend(outputs.argmax(dim=1).cpu().tolist())

    metrics = calculate_metrics(y_true, y_pred, class_names)
    samples = getattr(data_loader.dataset, "samples", [])
    misclassified = [
        {
            "image": str(samples[index][0]),
            "true": class_names[actual],
            "predicted": class_names[predicted],
        }
        for index, (actual, predicted) in enumerate(zip(y_true, y_pred))
        if actual != predicted and index < len(samples)
    ]
    return metrics, misclassified


def save_evaluation(metrics, misclassified, output_dir, class_names):
    
    """Write machine-readable metrics, a text report, a matrix plot, and error list."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    report = metrics["classification_report"]
    report_text = ( 
        f"{'':<20}{'precision':>10}{'recall':>10}{'f1-score':>10}{'support':>10}\n\n"
        + "".join(
            f"{name:<20}{report[name]['precision']:>10.2f}"
            f"{report[name]['recall']:>10.2f}{report[name]['f1-score']:>10.2f}"
            f"{report[name]['support']:>10.0f}\n"
            for name in class_names
        )
        + f"\naccuracy{'':<12}{metrics['accuracy']:>10.2f}\n"
        + f"macro avg{'':<11}{report['macro avg']['precision']:>10.2f}"
        f"{report['macro avg']['recall']:>10.2f}{report['macro avg']['f1-score']:>10.2f}"
        f"{report['macro avg']['support']:>10.0f}\n"
        + f"weighted avg{'':<8}{report['weighted avg']['precision']:>10.2f}"
        f"{report['weighted avg']['recall']:>10.2f}{report['weighted avg']['f1-score']:>10.2f}"
        f"{report['weighted avg']['support']:>10.0f}\n"
    )
    (output_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")

    matrix = np.asarray(metrics["confusion_matrix"])
    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["true/predicted", *class_names])
        writer.writerows(
            [class_names[index], *row] for index, row in enumerate(matrix.tolist())
        )

    figure, axis = plt.subplots(figsize=(8, 7))
    axis.imshow(matrix, cmap="Blues")
    axis.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title="Test-set confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png")
    plt.close(figure)

    with (output_dir / "misclassified.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=["image", "true", "predicted"])
        writer.writeheader()
        writer.writerows(misclassified)


def run_evaluation(checkpoint_path="model.pt", test_dir="data/test",
                   output_dir="evaluation_results", batch_size=32):
    model, class_names, image_size, device = load_checkpoint_model(checkpoint_path)
    data_loader = make_test_loader(test_dir, class_names, image_size, batch_size)
    metrics, misclassified = evaluate_model(model, data_loader, class_names, device)
    save_evaluation(metrics, misclassified, output_dir, class_names)

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro precision: {metrics['precision_macro']:.4f}")
    print(f"Macro recall: {metrics['recall_macro']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"Misclassified images: {len(misclassified)}")
    print(f"Results saved to {output_dir}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="model.pt")
    parser.add_argument("--test-dir", default="data/test")
    parser.add_argument("--output-dir", default="evaluation_results")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    run_evaluation(args.checkpoint, args.test_dir, args.output_dir, args.batch_size)


if __name__ == "__main__":
    main()