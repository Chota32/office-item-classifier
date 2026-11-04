# office-item-classifier

## Testing
test_training.py will train the model for one epoch and may overwrite the existing model.pt checkpoint.

## Dataset Sources
Item: Ballpoint
Source: https://images.cv/dataset/ballpoint-image-classification-dataset

Item: File
Source: https://images.cv/dataset/file-image-classification-dataset

Item: Monitor
Source: https://images.cv/dataset/monitor-image-classification-dataset

Item: Pencil sharpener
Source: https://images.cv/dataset/pencil-sharpener-image-classification-dataset

Item: Printer
Source: https://images.cv/dataset/printer-image-classification-dataset

Item: Rubber eraser:
Source: https://images.cv/dataset/rubber-eraser-image-classification-dataset

## Prediction

```bash
python predict.py --image photo.jpg       # one or more image files
python predict.py --folder data/test      # every image in a folder
python predict.py --camera                # live webcam, press q to quit
```

Example output: `printer_test_0000.jpg -> printer (99.9%)`

Troubleshooting:
- `Model 'model.pt' not found`: run from the project folder, or train with `python app.py train`.
- Every image `Skipped`: the images are Git LFS placeholders, so run `git lfs install` then `git lfs pull`.

Tests (these do not overwrite `model.pt`): `python -m pytest tests/test_predict.py -v`