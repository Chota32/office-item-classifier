"""Run from the project root with:  python -m pytest -v"""
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import predict  # noqa: E402

CLASSES = ["keyboard", "mouse", "mug", "notebook", "stapler", "waste_bin"]


def save_checkpoint(path, image_size=224):
    torch.manual_seed(0)
    torch.save({
        "model_state_dict": predict.build_model(len(CLASSES)).state_dict(),
        "class_names": CLASSES,
        "image_size": image_size,
    }, path)


def make_image(path):
    Image.new("RGB", (120, 80), (200, 50, 50)).save(path)
    return path


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("ckpt") / "model.pt"
    save_checkpoint(path)
    return path


def test_load_model(model_path):
    model, classes, _ = predict.load_model(model_path)
    assert classes == CLASSES
    assert not model.training


def test_image_size_comes_from_checkpoint(tmp_path):
    path = tmp_path / "model.pt"
    save_checkpoint(path, image_size=160)
    _, _, transform = predict.load_model(path)
    assert transform(Image.new("RGB", (300, 200))).shape == (3, 160, 160)


def test_missing_model_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        predict.load_model(tmp_path / "none.pt")


def test_predict_returns_class_and_confidence(model_path, tmp_path):
    model, classes, transform = predict.load_model(model_path)
    label, conf = predict.predict(model, classes, transform, make_image(tmp_path / "a.jpg"))
    assert label in CLASSES
    assert 0.0 <= conf <= 1.0


def test_predict_accepts_pil_image(model_path):
    model, classes, transform = predict.load_model(model_path)
    label, _ = predict.predict(model, classes, transform, Image.new("RGB", (50, 50)))
    assert label in CLASSES


def test_missing_image_raises(model_path, tmp_path):
    model, classes, transform = predict.load_model(model_path)
    with pytest.raises(OSError):
        predict.predict(model, classes, transform, tmp_path / "nope.jpg")


def test_corrupt_image_raises(model_path, tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    model, classes, transform = predict.load_model(model_path)
    with pytest.raises(OSError):
        predict.predict(model, classes, transform, bad)


def test_cli_image(model_path, tmp_path, capsys):
    img = make_image(tmp_path / "desk.jpg")
    assert predict.main(["--image", str(img), "--model", str(model_path)]) == 0
    assert "desk.jpg ->" in capsys.readouterr().out


def test_cli_folder_skips_bad_files(model_path, tmp_path, capsys):
    make_image(tmp_path / "a.jpg")
    make_image(tmp_path / "b.png")
    (tmp_path / "notes.txt").write_text("ignore me")
    (tmp_path / "broken.jpg").write_bytes(b"garbage")
    predict.main(["--folder", str(tmp_path), "--model", str(model_path)])
    out = capsys.readouterr().out
    assert "a.jpg ->" in out and "b.png ->" in out
    assert "broken" not in out and "notes" not in out


def test_cli_missing_folder_returns_error(model_path, tmp_path):
    assert predict.main(["--folder", str(tmp_path / "x"), "--model", str(model_path)]) == 1


def test_cli_missing_model_returns_error(tmp_path):
    img = make_image(tmp_path / "a.jpg")
    assert predict.main(["--image", str(img), "--model", str(tmp_path / "no.pt")]) == 1