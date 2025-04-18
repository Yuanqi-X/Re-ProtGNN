import os
import torch
import networkx as nx
import matplotlib.pyplot as plt
from unittest.mock import patch, mock_open, MagicMock
from utils.outputUtils import append_record, save_best, ExpPlot


# ========== Test append_record ==========

@patch("builtins.open", new_callable=mock_open)
def test_append_record(mock_file):
    append_record("test log line")
    mock_file.assert_called_once_with('./results/log/hyper_search', 'a')
    mock_file().write.assert_called_once_with("test log line\n")


# ========== Dummy Model for save_best ==========
class DummyModel:
    def __init__(self):
        self._moved_to = None
    def to(self, device):
        self._moved_to = device
    def state_dict(self):
        return {"dummy": "weights"}


# ========== Test save_best ==========

@patch("utils.outputUtils.torch.save")
@patch("utils.outputUtils.shutil.copy")
def test_save_best_true(mock_copy, mock_save):
    model = DummyModel()
    save_best("ckpt_dir", 5, model, "GNN", 0.88, is_best=True)
    mock_save.assert_called()
    mock_copy.assert_called_once()


@patch("utils.outputUtils.torch.save")
@patch("utils.outputUtils.shutil.copy")
def test_save_best_false(mock_copy, mock_save):
    model = DummyModel()
    save_best("ckpt_dir", 5, model, "GNN", 0.88, is_best=False)
    mock_save.assert_called()
    mock_copy.assert_not_called()


# ========== Test ExpPlot.draw() ==========

@patch.object(ExpPlot, '_draw_ba2motifs')
def test_draw_ba2motifs(mock_draw):
    plot = ExpPlot('ba_2motifs')
    plot.draw(nx.Graph(), [0, 1], figname='temp.png')
    mock_draw.assert_called_once()


@patch.object(ExpPlot, '_draw_molecule')
def test_draw_mutag(mock_draw):
    plot = ExpPlot('mutag')
    plot.draw(nx.Graph(), [0, 1], figname='temp.png', x=torch.eye(3))
    mock_draw.assert_called_once()


@patch.object(ExpPlot, '_draw_molecule')
def test_draw_bbbp(mock_draw):
    plot = ExpPlot('bbbp')
    plot.draw(nx.Graph(), [0, 1], figname='temp.png', x=torch.tensor([[6], [8], [1]]))
    mock_draw.assert_called_once()


# ========== Test _draw_subgraph ==========

def test_draw_subgraph_runs(tmp_path):
    plot = ExpPlot('ba_2motifs')
    g = nx.path_graph(3)
    fig_path = tmp_path / "subgraph.png"
    plot._draw_subgraph(g, [0, 1], figname=str(fig_path))
    assert fig_path.exists()


# ========== Test ExpPlot.draw() fallback ==========

def test_draw_invalid_dataset_raises():
    plot = ExpPlot('unknown')
    with patch("matplotlib.pyplot.savefig"):  # suppress actual plotting
        try:
            plot.draw(nx.Graph(), [0, 1], figname='test.png')
        except NotImplementedError:
            assert True
        else:
            assert False, "Expected NotImplementedError"

"""
In terminal (project's root), run tests with:

pytest --cov=utils.outputUtils --cov-report=term-missing tests/test_output_utils.py
"""