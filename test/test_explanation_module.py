import pytest
import torch
import networkx as nx
import os
import numpy as np
import torch
from unittest.mock import patch, MagicMock
from torch_geometric.data import Data
from evaluation.explanation import (
    exp_visualize,
    _MCTSNode,
    _prot_similarity_scores,
    _mcts_scores,
    _mcts_rollout,
    _mcts,
    get_explanation
)

from torch_geometric.data import Data, Batch

# === Dummy GNN that returns fixed embeddings ===
class DummyGNN(torch.nn.Module):
    def forward(self, batch, protgnn_plus=False):
        return None, None, None, torch.tensor([[1.0, 2.0]]), None

# === Shared tiny graph data ===
def tiny_batch():
    x = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    data = Data(x=x, edge_index=edge_index)
    return Batch.from_data_list([data])

# === Test: MCTSNode scoring (already covered) ===
def test_mcts_node_scoring():
    graph = nx.path_graph(3)
    node = _MCTSNode(coalition=[0, 1, 2], data=None, ori_graph=graph, W=9, N=3, P=0.5)
    assert node.Q() == 3.0
    assert node.U(10) > 0

# === Test: Prototype similarity score (already covered) ===
def test_prototype_similarity_score():
    gnn = DummyGNN()
    prototype = torch.tensor([[0.5, 0.5]])
    score = _prot_similarity_scores([0, 1, 2], data=tiny_batch(), gnnNet=gnn, prototype=prototype)
    assert isinstance(score, float)

# === Test: MCTS prior scoring ===
def test_mcts_scores():
    gnn = DummyGNN()
    prototype = torch.tensor([[0.5, 0.5]])
    data = tiny_batch()
    score_fn = lambda c: _prot_similarity_scores(c, data, gnn, prototype)

    node1 = _MCTSNode([0, 1], data, nx.path_graph(3), P=0)
    node2 = _MCTSNode([1, 2], data, nx.path_graph(3), P=0.2)
    scores = _mcts_scores(score_fn, [node1, node2])
    assert len(scores) == 2

# === Test: MCTS rollout (with mocked expansion conditions) ===
def test_mcts_rollout():
    gnn = DummyGNN()
    prototype = torch.tensor([[0.5, 0.5]])
    data = tiny_batch()
    graph = nx.path_graph(3)

    root = _MCTSNode([0, 1, 2], data, graph)
    state_map = {str(root.coalition): root}
    score_fn = lambda c: _prot_similarity_scores(c, data, gnn, prototype)

    _mcts_rollout(root, state_map, data, graph, score_fn)
    assert isinstance(root.children, list)

# === Test: Full MCTS explanation generation ===
def test_mcts_subgraph_explanation():
    gnn = DummyGNN()
    prototype = torch.tensor([[0.5, 0.5]])
    data = Data(x=torch.tensor([[1., 0.], [0., 1.]]), edge_index=torch.tensor([[0, 1], [1, 0]]))

    coalition, score, emb = get_explanation(data, gnn, prototype)
    assert isinstance(coalition, list)
    assert isinstance(score, (float, int))  # <-- fixed here
    assert isinstance(emb, torch.Tensor)

# === Test: exp_visualize without writing files ===

@patch("evaluation.explanation.ExpPlot")
@patch("evaluation.explanation.os.makedirs")
@patch("evaluation.explanation.to_networkx")
@patch("evaluation.explanation.get_explanation")
@patch("evaluation.explanation.np.random.choice")
def test_exp_visualize_mocked(
    mock_choice, mock_get_explanation, mock_to_networkx, mock_makedirs, mock_plot
):
    # Fake graph data
    dataset = [Data(x=torch.ones(3, 2), edge_index=torch.tensor([[0, 1], [1, 2]])) for _ in range(20)]

    # Patch data_args and model_args
    from evaluation import explanation
    explanation.data_args.dataset_name = "DummyDataset"
    explanation.model_args.model_name = "DummyModel"
    explanation.model_args.num_prototypes_per_class = 10
    explanation.model_args.device = "cpu"
    explanation.mcts_args.dataset_name = "DummyDS"

    # Return 16 fixed indices for visualization
    mock_choice.return_value = np.arange(16)
    mock_get_explanation.return_value = ([0, 1], 0.9, torch.tensor([[1.0]]))
    mock_to_networkx.return_value = MagicMock()

    # Dummy model
    class DummyModel:
        def to_device(self): pass
        def eval(self): pass

    dataloader = {'train': MagicMock()}
    dataloader['train'].dataset.indices = list(range(20))

    exp_visualize(dataset, dataloader, DummyModel(), output_dim=1)

    # Check that plotting was triggered
    assert mock_plot.return_value.draw.called


# === Test: backup logic in rollout is hit ===

def test_mcts_rollout_backpropagation():
    graph = nx.path_graph(4)
    coalition = [0, 1, 2, 3]
    root = _MCTSNode(coalition=coalition, data=None, ori_graph=graph)

    # Setup a fake child to trigger backup
    child = _MCTSNode(coalition=[0, 1, 2], data=None, ori_graph=graph)
    root.children.append(child)

    # Provide dummy prior so root can backpropagate it
    child.P = 0.75
    score_fn = lambda c: 0.75  # Not used, child already has P

    total_N = 0
    v = _mcts_rollout(root, {}, None, graph, score_fn)
    assert isinstance(v, (float, int))
