import pytest
import torch
import argparse
from main import main  # Your actual entry point
import models.train as train_module  # So we patch what train.py actually uses

from utils.Configures import train_args

def test_loss_converges(monkeypatch):
    loss_history = []

    # Patch append_record inside train.py
    def capture_log(line):
        if "Epoch" in line and "loss:" in line:
            try:
                loss = float(line.split("loss:")[1].split(",")[0].strip())
                loss_history.append(loss)
            except ValueError:
                pass

    monkeypatch.setattr(train_module, "append_record", capture_log)
    monkeypatch.setattr(train_module, "save_best", lambda *args, **kwargs: None)


    args = argparse.Namespace(clst=0.0, sep=0.0)
    main(args)

    # Print losses for debugging
    print("\nLoss history across epochs:", loss_history)

    assert len(loss_history) >= 2, "Not enough loss values captured"
    assert loss_history[-1] <= loss_history[0], "Loss did not decrease during training"


"""
In terminal (project's root), run tests with:

PYTHONPATH=./src pytest tests/sys_test_loss_converge.py
"""