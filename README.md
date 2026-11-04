# office-item-classifier

## Testing
test_training.py will train the model for one epoch and may overwrite the existing model.pt checkpoint.

## Dataset

The dataset contains 7,969 images across six office-item classes: ballpoint, file, monitor, pencil sharpener, printer and rubber eraser. Images were obtained from Images.cv.

The dataset was divided into:
- Training: 5,574 images (approximately 70%)
- Validation: 1,193 images (approximately 15%)
- Testing: 1,202 images (approximately 15%)

During dataset preparation, images are converted to RGB, resized to a maximum side length of 640 pixels while preserving aspect ratio, and saved as JPEG. For model input, images are resized to 224 x 224 pixels and normalised using ImageNet mean and standard deviation values. Random horizontal flipping and rotation are applied only during training.

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