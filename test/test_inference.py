import pytest
import torch
import numpy as np
from unittest.mock import patch
from evaluation.inference import run_inference


# === Simulated Batch Class ===
class EvaluationBatch:
    def __init__(self, labels):
        self.y = torch.tensor(labels)

# === Model for Testing ===
class EvaluationModel(torch.nn.Module):
    def forward(self, batch):
        # Simulated logits for two classes
        logits = torch.tensor([[2.0, 1.0], [0.5, 1.5]])
        probabilities = torch.softmax(logits, dim=-1)
        return logits, probabilities, None, None, None

# === Sample Criterion Function ===
def mock_criterion(logits, targets):
    return torch.tensor(0.25)  # Constant loss

# === Pytest Fixture for Test Dataloader ===
@pytest.fixture
def test_dataloader():
    batch1 = EvaluationBatch([0, 1])
    batch2 = EvaluationBatch([1, 0])
    return [batch1, batch2]

# === Test 1: Output Structure Test ===
def test_model_evaluation_output_format(test_dataloader):
    model = EvaluationModel()
    criterion = mock_criterion

    eval_result, probabilities, predictions = run_inference(test_dataloader, model, criterion)

    assert isinstance(eval_result, dict)
    assert "loss" in eval_result and "acc" in eval_result
    assert isinstance(probabilities, np.ndarray)
    assert isinstance(predictions, np.ndarray)

# === Test 2: Loss and Accuracy Aggregation Test ===
def test_batch_loss_and_accuracy_aggregation(test_dataloader):
    model = EvaluationModel()
    criterion = mock_criterion

    eval_result, _, _ = run_inference(test_dataloader, model, criterion)

    assert 0.0 <= eval_result["loss"] <= 1.0
    assert 0.0 <= eval_result["acc"] <= 1.0

# === Test 3: Concatenation of Outputs Test ===
def test_probability_and_prediction_concatenation(test_dataloader):
    model = EvaluationModel()
    criterion = mock_criterion

    _, probabilities, predictions = run_inference(test_dataloader, model, criterion)

    assert probabilities.shape[0] == 4  # 2 batches * 2 samples each
    assert predictions.shape[0] == 4

# === Test 4: Logging Function Call Test ===
def test_logging_called_once(test_dataloader):
    model = EvaluationModel()
    criterion = mock_criterion

    with patch("evaluation.inference.append_record") as mock_logger:
        run_inference(test_dataloader, model, criterion)
        mock_logger.assert_called_once()



"""
In terminal (project's root), run tests with:

export PYTHONPATH=./src
pytest --cov=evaluation.inference --cov-report=term --cov-report=html tests/test_inference.py
"""