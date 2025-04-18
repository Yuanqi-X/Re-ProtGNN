import os
import torch
import pytest
from torch.nn import CrossEntropyLoss
import numpy as np

from utils.Configures import model_args, data_args
from models import setup_model
from data_processing.dataUtils import load_dataset
from evaluation.inference import run_inference


def test_inference_on_test_split():
    # Load real dataset and dataloader
    dataset, input_dim, output_dim, dataloader = load_dataset()
    test_loader = dataloader['test']

    # Setup model and loss
    model, criterion = setup_model(input_dim, output_dim, model_args)

    # Load the best checkpoint
    ckpt_dir = f"./src/checkpoint/{data_args.dataset_name}/"
    ckpt_path = os.path.join(ckpt_dir, f"{model_args.model_name}_best.pth")
    assert os.path.exists(ckpt_path), f"Missing checkpoint: {ckpt_path}"
    state_dict = torch.load(ckpt_path)
    model.update_state_dict(state_dict['net'])

    # Run inference
    test_state, all_probs, all_preds = run_inference(test_loader, model, criterion)

    # Assertions
    assert "acc" in test_state and "loss" in test_state
    assert test_state["acc"] > 0.5, "Expected accuracy > 50%"
    assert all_preds.shape[0] == len(test_loader.dataset)
    assert len(all_probs.shape) == 2  # Expecting (N, C)
    assert all_preds.shape[0] == all_probs.shape[0]
    print(f"System Inference Test Passed | Accuracy: {test_state['acc']:.3f}")


"""
In terminal (project's root), run tests with:

export PYTHONPATH=./src
pytest tests/sys_test_inference.py
"""