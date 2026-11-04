from pathlib import Path
import torch

def test_checkpoint():
    assert Path("model.pt").exists()

    checkpoint = torch.load("model.pt", map_location="cpu", weights_only=True)

    assert "model_state_dict" in checkpoint
    assert checkpoint["model_name"] == "mobilenet_v3_small"
    assert checkpoint["image_size"] == 224
    assert checkpoint["class_names"] == ["ballpoint", "file","monitor", "pencil_sharpener", "printer", "rubber_eraser"]

    print("Checkpoint test passed!")

if __name__ == "__main__":
    test_checkpoint()