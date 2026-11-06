"""Classify office items from images, a folder, or a webcam.

    python predict.py --image photo.jpg
    python predict.py --folder data/test
    python predict.py --camera
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms

MODEL_PATH = "model.pt"
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def build_model(num_classes):
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def load_model(path=MODEL_PATH):
    if not Path(path).is_file():
        raise FileNotFoundError(f"Model '{path}' not found. Run: python app.py train")
    ckpt = torch.load(path, map_location="cpu")
    classes = ckpt["class_names"]
    model = build_model(len(classes))
    model.load_state_dict(ckpt["model_state_dict"])
    size = ckpt.get("image_size", 224)
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return model.eval(), classes, transform


def predict(model, classes, transform, image):
    """Return (class_name, confidence) for a file path or PIL image."""
    if not isinstance(image, Image.Image):
        image = ImageOps.exif_transpose(Image.open(image))
    x = transform(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    conf, idx = probs.max(0)
    return classes[idx.item()], conf.item()


def run_camera(model, classes, transform, index=0):
    import cv2

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Error: could not open camera {index}.", file=sys.stderr)
        return 1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        label, conf = predict(model, classes, transform, rgb)
        cv2.putText(frame, f"{label}: {conf:.0%}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 2)
        cv2.imshow("Office Item Classifier (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="Classify office items.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", nargs="+", help="one or more image files")
    src.add_argument("--folder", help="folder of images")
    src.add_argument("--camera", type=int, nargs="?", const=0, help="camera index (default 0)")
    p.add_argument("--model", default=MODEL_PATH)
    a = p.parse_args(argv)

    try:
        model, classes, transform = load_model(a.model)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if a.camera is not None:
        return run_camera(model, classes, transform, a.camera)

    if a.folder:
        if not Path(a.folder).is_dir():
            print(f"Error: folder not found: {a.folder}", file=sys.stderr)
            return 1
        files = sorted(f for f in Path(a.folder).rglob("*") if f.suffix.lower() in EXTS)
    else:
        files = [Path(f) for f in a.image]

    status = 1 if not files else 0
    for f in files:
        try:
            label, conf = predict(model, classes, transform, f)
        except OSError as e:
            print(f"Skipped {f}: {e}", file=sys.stderr)
            status = 1
            continue
        print(f"{f.name} -> {label} ({conf:.1%})")
    return status


if __name__ == "__main__":
    sys.exit(main())