import torch
from app import create_model

def test_model():
    model = create_model()
    model.eval()

    assert model.classifier[-1].out_features == 6

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable == 6150

    with torch.no_grad():
        output = model(torch.randn(1, 3, 224, 224))

    assert output.shape == (1, 6)

    print("All model tests passed!")

if __name__ == "__main__":
    test_model()