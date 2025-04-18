import os
import math
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
from functools import partial
from collections import Counter

from torch_geometric.utils import to_networkx
from torch_geometric.data import Data, Batch
from utils.Configures import data_args, model_args, mcts_args
from utils.outputUtils import ExpPlot


# ========== Exported Functions ==========

# Main function to visualize explanations using learned prototype vectors and MCTS.
# Saves visualizations for 16 randomly sampled training graphs.
def exp_visualize(dataset, dataloader, gnnNets, output_dim):
    data_indices = dataloader['train'].dataset.indices

    # Randomly initialized prototype vectors (fixed for visualization)
    prototype_shape = (output_dim * model_args.num_prototypes_per_class, 128)
    prototype_vectors = nn.Parameter(
        torch.rand(prototype_shape), requires_grad=False
    ).to(model_args.device)

    gnnNets.to_device()
    gnnNets.eval()

    # Create output directory for saving explanation plots
    save_dir = os.path.join('./results/plots',
                            f"{mcts_args.dataset_name}_{model_args.model_name}_")
    os.makedirs(save_dir, exist_ok=True)
    expPlot = ExpPlot(dataset_name=data_args.dataset_name)

    # Visualize explanations for 16 random graphs and top 10 prototypes
    batch_indices = np.random.choice(data_indices, 16, replace=False)
    for i in batch_indices:
        data = dataset[i.item()]
        for j in range(10):
            coalition, _, _ = get_explanation(data, gnnNets, prototype_vectors[j])
            print(coalition)
            graph = to_networkx(data, to_undirected=True)
            expPlot.draw(graph, coalition, x=data.x,
                         figname=os.path.join(save_dir, f"example_{i*10+j}.png"))


# Main explanation interface using MCTS to identify most relevant subgraph
# Returns: node indices in subgraph, score, and projected embedding
def get_explanation(data, gnnNet, prototype):
    return _mcts(data, gnnNet, prototype)


# ========== Internal Utilities ==========

# MCTS search node storing subgraph coalition and rollout metadata
class _MCTSNode:
    def __init__(self, coalition, data, ori_graph, c_puct=10.0, W=0, N=0, P=0):
        self.data = data
        self.coalition = coalition
        self.ori_graph = ori_graph
        self.c_puct = c_puct
        self.children = []
        self.W = W  # total reward
        self.N = N  # visit count
        self.P = P  # prior score

    def Q(self):  # Exploitation
        return self.W / self.N if self.N > 0 else 0

    def U(self, n):  # Exploration
        return self.c_puct * self.P * math.sqrt(n) / (1 + self.N)


# Performs a single rollout (graph pruning → node expansion → value backup)
def _mcts_rollout(tree_node, state_map, data, graph, score_func):
    cur_coalition = tree_node.coalition

    # Stop expanding if coalition is too small
    if len(cur_coalition) <= mcts_args.min_atoms:
        return tree_node.P

    # Expand unexplored node
    if not tree_node.children:
        node_degrees = sorted(
            graph.subgraph(cur_coalition).degree, 
            key=lambda x: x[1], 
            reverse=mcts_args.high2low
        )
        all_nodes = [n for n, _ in node_degrees]
        expand_nodes = all_nodes[:mcts_args.expand_atoms] if len(all_nodes) >= mcts_args.expand_atoms else all_nodes

        for node in expand_nodes:
            subgraph_nodes = [n for n in all_nodes if n != node]
            subgraphs = [graph.subgraph(c) for c in nx.connected_components(graph.subgraph(subgraph_nodes))]
            main_subgraph = max(subgraphs, key=lambda sg: sg.number_of_nodes())
            new_coalition = sorted(list(main_subgraph.nodes()))

            # Check for duplicate in state_map
            node_key = str(new_coalition)
            if node_key not in state_map:
                new_node = _MCTSNode(new_coalition, data, graph)
                state_map[node_key] = new_node
            else:
                new_node = state_map[node_key]

            # Avoid duplicate children
            if all(Counter(c.coalition) != Counter(new_node.coalition) for c in tree_node.children):
                tree_node.children.append(new_node)

        # Compute prior scores for new children
        scores = _mcts_scores(score_func, tree_node.children)
        for child, score in zip(tree_node.children, scores):
            child.P = score

    # Select best child via UCT and backpropagate value
    total_N = sum(c.N for c in tree_node.children)
    selected = max(tree_node.children, key=lambda c: c.Q() + c.U(total_N))
    v = _mcts_rollout(selected, state_map, data, graph, score_func)
    selected.W += v
    selected.N += 1
    return v


# Runs MCTS search to select best subgraph explanation based on prototype similarity
def _mcts(data, gnnNet, prototype):
    data = Data(x=data.x, edge_index=data.edge_index)
    graph = to_networkx(data, to_undirected=True)
    data = Batch.from_data_list([data])

    num_nodes = graph.number_of_nodes()
    root = _MCTSNode(coalition=list(range(num_nodes)), data=data, ori_graph=graph)
    state_map = {str(root.coalition): root}
    score_func = partial(_prot_similarity_scores, data=data, gnnNet=gnnNet, prototype=prototype)

    for _ in range(mcts_args.rollout):
        _mcts_rollout(root, state_map, data, graph, score_func)

    # Rank candidate subgraphs by priority (score then size)
    explanations = sorted(state_map.values(), key=lambda x: (len(x.coalition), -x.P))
    result = explanations[0]
    for candidate in explanations:
        if len(candidate.coalition) <= mcts_args.max_atoms and candidate.P > result.P:
            result = candidate

    # Compute final embedding for selected explanation
    mask = torch.zeros(data.num_nodes).float()
    mask[result.coalition] = 1.0
    ret_x = data.x * mask.unsqueeze(1)
    masked_data = Batch.from_data_list([Data(x=ret_x, edge_index=data.edge_index)])
    _, _, _, emb, _ = gnnNet(masked_data, protgnn_plus=False)
    return result.coalition, result.P, emb


# Computes prior scores for child nodes if not already assigned
def _mcts_scores(score_func, children):
    scores = []
    for child in children:
        scores.append(score_func(child.coalition) if child.P == 0 else child.P)
    return scores


# Computes similarity score between input subgraph embedding and given prototype
def _prot_similarity_scores(coalition, data, gnnNet, prototype):
    epsilon = 1e-4
    mask = torch.zeros(data.num_nodes).float()
    mask[coalition] = 1.0
    ret_x = data.x * mask.unsqueeze(1)

    masked_data = Batch.from_data_list([Data(x=ret_x, edge_index=data.edge_index)])
    _, _, _, emb, _ = gnnNet(masked_data, protgnn_plus=False)

    # Smaller distance = higher similarity
    distance = torch.norm(emb - prototype) ** 2
    similarity = torch.log((distance + 1) / (distance + epsilon))
    return similarity.item()
