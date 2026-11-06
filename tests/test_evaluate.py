import torch
from torch.utils.data import DataLoader, TensorDataset

from evaluate import calculate_metrics, evaluate_model, save_evaluation


class FixedClassifier(torch.nn.Module):
    def forward(self, images):
        return images


def test_calculate_metrics_returns_six_class_outputs():
    classes = ["a", "b", "c", "d", "e", "f"]
    y_true = [0, 0, 1, 1, 2, 5]
    y_pred = [0, 1, 1, 2, 2, 5]

    metrics = calculate_metrics(y_true, y_pred, classes)

    assert metrics["accuracy"] == 4 / 6
    assert len(metrics["confusion_matrix"]) == 6
    assert metrics["confusion_matrix"][0] == [1, 1, 0, 0, 0, 0]
    assert "a" in metrics["classification_report"]
    assert "macro avg" in metrics["classification_report"]
    assert 0 <= metrics["precision_macro"] <= 1
    assert 0 <= metrics["recall_macro"] <= 1
    assert 0 <= metrics["macro_f1"] <= 1


def test_evaluate_model_collects_predictions():
    images = torch.tensor([[5.0, 0.0], [0.0, 5.0], [5.0, 0.0]])
    labels = torch.tensor([0, 1, 1])
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)

    metrics, misclassified = evaluate_model(
        FixedClassifier(), loader, ["first", "second"], torch.device("cpu")
    )

    assert metrics["accuracy"] == 2 / 3
    assert misclassified == []


def test_save_evaluation_writes_report_matrix_and_errors(tmp_path):
    class_names = ["first", "second"]
    metrics = calculate_metrics([0, 1], [1, 1], class_names)
    mistakes = [{"image": "sample.jpg", "true": "first", "predicted": "second"}]

    save_evaluation(metrics, mistakes, tmp_path, class_names)

    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "classification_report.txt").exists()
    assert (tmp_path / "confusion_matrix.csv").exists()
    assert (tmp_path / "confusion_matrix.png").exists()
    assert "sample.jpg" in (tmp_path / "misclassified.csv").read_text(encoding="utf-8")
