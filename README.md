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

## Evaluation
After training a model that saves `model.pt`, evaluate it on the held-out test
set with:

```powershell
python app.py evaluate
```

The command reports accuracy, macro precision, macro recall, macro-F1, a
per-class classification report, and incorrectly classified image paths. It
writes `evaluation_results/metrics.json`, `classification_report.txt`,
`confusion_matrix.csv`, `confusion_matrix.png`, and `misclassified.csv`.
Use `--checkpoint`, `--test-dir`, `--output-dir`, and `--batch-size` to change
the defaults.

To measure performance under blur, 15-degree rotation, and darkened or
brightened lighting:

```powershell
python app.py robustness
```

This writes per-perturbation metrics and confusion matrices to
`robustness_results.json`, including a clean baseline and metric deltas for
each perturbation. Perturbation strengths are fixed (blur radius 2, rotation
15 degrees, brightness 50% and 150%); results measure this dataset and these
specific transformations, not all real-world image conditions. A trained
`model.pt` checkpoint is required for both commands. Final performance
summaries and limitations should be based on the generated metrics and manual
review of `misclassified.csv`; no evaluation numbers are available before
training.