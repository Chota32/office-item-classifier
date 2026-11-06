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

## Model and Training

The classifier uses MobileNetV3 Small with pretrained ImageNet weights. The pretrained layers are frozen, and the final classification layer is replaced to recognise the six office-item categories.

Training images are resized to 224x224, normalised, and augmented using random horizontal flipping and rotation.

To train the model:
```bash
python app.py train --epochs 5
```

The model uses cross-entropy loss and the Adam optimiser. The checkpoint with the highest validation accuracy is saved as 'model.pt'.

In our five epoch experiment, the best validation accuracy was 89.02% at epoch 2.

Training results are recorded in 'training_history.csv', with accuracy and loss graphs saved as 'training_accuracy.png' and 'training_loss.png'.


## Installation

Install the required Python packages:

```bash
python -m pip install -r requirements.txt
```

The trained model is stored in `model.pt`. Git LFS is used to track
the model file. After cloning the repository, run:

```bash
git lfs pull
```

## Prediction

Predict a single image:

```bash
python predict.py --image path/to/image.jpg
```

Predict all images in a folder:

```bash
python predict.py --folder path/to/folder
```

Use a webcam:

```bash
python predict.py --camera 0
```

The program displays the predicted class and confidence score.

## Tests

Run the automated tests:

```bash
python -m pytest -v
```

All 20 tests passed during the final test run.

Note: The training test may overwrite `model.pt` and
`training_history.csv`. Back up these files before running all tests.


Troubleshooting:
- `Model 'model.pt' not found`: run from the project folder, or train with `python app.py train`.
- Every image `Skipped`: the images are Git LFS placeholders, so run `git lfs install` then `git lfs pull`.

Tests (these do not overwrite `model.pt`): `python -m pytest tests/test_predict.py -v`


## Evaluation

To evaluate the trained model on the test dataset:

```bash
python evaluate.py
```

The results are saved in `evaluation_results/`, including the
classification report, confusion matrix and misclassified images.

Test results on 1,202 images:
- Accuracy: 87.52%
- Macro-F1: 87.34%

## Robustness Testing

To run the robustness tests:

```bash
python robustness.py
```

The tests apply Gaussian blur, 15-degree rotation, and brightness
changes of 50% and 150%.

The results are saved in `robustness_results.json`.

Accuracy under different conditions:
- Original images: 87.52%
- Blur: 84.44%
- Rotation: 81.95%
- Darkened: 86.69%
- Brightened: 84.94%

