from app import load_datasets

def test_dataset():
    train_loader, val_loader = load_datasets()
    images, labels = next(iter(train_loader))

    assert len(train_loader.dataset) == 5574
    assert len(val_loader.dataset) == 1193
    assert len(train_loader.dataset.classes) == 6
    assert images.shape == (32, 3, 224, 224)
    assert labels.shape == (32,)

    print("All dataset tests passed!")

if __name__ == "__main__":
    test_dataset()