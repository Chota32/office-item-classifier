"""Measure test-set performance under common image perturbations."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

from evaluate import load_checkpoint_model, make_test_loader, evaluate_model


def gaussian_blur(image):
    return image.filter(ImageFilter.GaussianBlur(radius=2))


def rotate(image):
    return image.rotate(
        15,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=(255, 255, 255),
    )


def darken(image):
    return ImageEnhance.Brightness(image).enhance(0.5)


def brighten(image):
    return ImageEnhance.Brightness(image).enhance(1.5)


PERTURBATIONS = {
    "blur_radius_2": gaussian_blur,
    "rotation_15_degrees": rotate,
    "brightness_50_percent": darken,
    "brightness_150_percent": brighten,
}


def run_robustness(checkpoint_path="model.pt", test_dir="data/test",
                   output_path="robustness_results.json", batch_size=32):
    model, class_names, image_size, device = load_checkpoint_model(checkpoint_path)
    results = {}
    conditions = {"clean": None, **PERTURBATIONS}
    for name, perturbation in conditions.items():
        data_loader = make_test_loader(
            test_dir, class_names, image_size, batch_size, perturbation=perturbation
        )
        metrics, _ = evaluate_model(model, data_loader, class_names, device)
        results[name] = {
            "accuracy": metrics["accuracy"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
            "macro_f1": metrics["macro_f1"],
            "confusion_matrix": metrics["confusion_matrix"],
        }

    baseline = results["clean"]
    for name, metrics in results.items():
        if name != "clean":
            metrics["accuracy_delta"] = metrics["accuracy"] - baseline["accuracy"]
            metrics["macro_f1_delta"] = metrics["macro_f1"] - baseline["macro_f1"]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for name, metrics in results.items():
        delta = "" if name == "clean" else (
            f", accuracy delta={metrics['accuracy_delta']:+.4f}, "
            f"macro-F1 delta={metrics['macro_f1_delta']:+.4f}"
        )
        print(
            f"{name}: accuracy={metrics['accuracy']:.4f}, "
            f"macro-F1={metrics['macro_f1']:.4f}{delta}"
        )
    print(f"Results saved to {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="model.pt")
    parser.add_argument("--test-dir", default="data/test")
    parser.add_argument("--output", default="robustness_results.json")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    run_robustness(args.checkpoint, args.test_dir, args.output, args.batch_size)


if __name__ == "__main__":
    main()
