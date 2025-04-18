import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import MagicMock, patch, mock_open
from models import train
from utils import Configures
Configures.model_args.device = "cpu"



@pytest.fixture
def dummy_dataset():
    return [MagicMock(x=torch.ones(3, 2), edge_index=torch.tensor([[0, 1], [1, 0]]), y=torch.tensor(0)) for _ in range(20)]


@pytest.fixture
def dummy_dataloader():
    mock_batch = MagicMock()
    mock_batch.y = torch.tensor([0, 1])
    dataloader = MagicMock()
    dataloader.__iter__.return_value = iter([mock_batch])
    dataloader.dataset.indices = list(range(20))
    return {'train': dataloader, 'eval': dataloader}


@pytest.fixture
def dummy_model():
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = MagicMock()
            # 2 classes, 5 prototypes each → (10, 2) for identity
            self.model.prototype_class_identity = torch.eye(2).repeat(5, 1)
            self.model.prototype_vectors = nn.Parameter(torch.rand(10, 10))
            self.model.last_layer = nn.Linear(10, 2)
            self.model.gnn_layers = nn.Linear(10, 10)

        def forward(self, batch, protgnn_plus=False):
            return (
                torch.rand(2, 2), torch.rand(2, 2),
                None, None, torch.rand(2, 10)
            )

        def train(self): pass
        def eval(self): pass
        def parameters(self): return [torch.rand(1, requires_grad=True)]
        def to(self, device): return self

    return DummyModel()


def test_compute_total_loss(dummy_model):
    logits = torch.rand(2, 2)
    labels = torch.tensor([0, 1])
    min_dist = torch.rand(2, 10)
    crit = nn.CrossEntropyLoss()
    loss = train._compute_total_loss(logits, labels, min_dist, dummy_model, 2, crit, clst=1.0, sep=1.0)
    assert isinstance(loss, torch.Tensor)


def test_log_dataset_stats(dummy_dataset, capsys):
    train._log_dataset_stats(dummy_dataset)
    captured = capsys.readouterr()
    assert "Avg Nodes" in captured.out


def test_set_training_mode(dummy_model):
    train._set_training_mode(dummy_model, warm_only=True)
    train._set_training_mode(dummy_model, warm_only=False)


@patch("evaluation.explanation.get_explanation", return_value=([0, 1], 0.9, torch.rand(1, 10)))
def test_project_prototypes(mock_exp, dummy_model, dummy_dataset):
    dummy_model.eval = lambda: None
    dummy_model.model.prototype_vectors = nn.Parameter(torch.rand(10, 10))
    dummy_model.model.prototype_class_identity = torch.eye(2).repeat(5, 1)
    train._project_prototypes(dummy_model, dummy_dataset, list(range(20)), output_dim=2)


def test_evaluate(dummy_model, dummy_dataloader):
    dummy_model.eval = lambda: None
    dummy_model.forward = lambda batch: (torch.rand(2, 2), torch.rand(2, 2), None, None, None)
    result = train._evaluate(dummy_dataloader['eval'], dummy_model, nn.CrossEntropyLoss())
    assert 'loss' in result and 'acc' in result


"""
In terminal (project's root), run tests with:

PYTHONPATH=./src pytest --cov=models --cov-report=term-missing tests/test_train_module.py

"""