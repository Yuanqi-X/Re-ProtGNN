import os
import torch
import shutil
import numpy as np
import rdkit.Chem as Chem
import networkx as nx
import matplotlib.pyplot as plt
from textwrap import wrap

from utils.Configures import model_args


# ========== Exported Functions ==========

# Appends a text record (e.g., logs, metrics) to the output log file.
def append_record(info):
    with open('./results/log/hyper_search', 'a') as f:
        f.write(info + '\n')


# Saves the current model checkpoint.
# If 'is_best=True', it also updates the best model file.
def save_best(ckpt_dir, epoch, gnnNets, model_name, eval_acc, is_best):
    print('saving....')
    gnnNets.to('cpu')
    state = {
        'net': gnnNets.state_dict(),
        'epoch': epoch,
        'acc': eval_acc
    }

    pth_name = f"{model_name}_latest.pth"
    best_pth_name = f"{model_name}_best.pth"
    ckpt_path = os.path.join(ckpt_dir, pth_name)

    torch.save(state, ckpt_path)
    if is_best:
        shutil.copy(ckpt_path, os.path.join(ckpt_dir, best_pth_name))
    gnnNets.to(model_args.device)


# Visualization helper for subgraph explanation.
# Supports BA_2Motifs, MUTAG, and BBBP datasets.
class ExpPlot:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    # Public interface to draw explanation depending on dataset type
    def draw(self, graph, nodelist, figname, **kwargs):
        if self.dataset_name.lower() == 'ba_2motifs':
            self._draw_ba2motifs(graph, nodelist, figname=figname)
        elif self.dataset_name.lower() in ['bbbp', 'mutag']:
            x = kwargs.get('x')
            self._draw_molecule(graph, nodelist, x, figname=figname)
        else:
            raise NotImplementedError

    # ========== Internal Drawing Methods ==========

    # Base subgraph drawing utility using NetworkX
    def _draw_subgraph(self, graph, nodelist, colors='green', labels=None, edge_color='gray',
                       edgelist=None, subgraph_edge_color='black', title_sentence=None, figname=None):

        if edgelist is None:
            edgelist = [(u, v) for (u, v) in graph.edges() if u in nodelist and v in nodelist]

        pos = nx.kamada_kawai_layout(graph)
        pos_nodelist = {k: v for k, v in pos.items() if k in nodelist}

        nx.draw_networkx_nodes(graph, pos,
                               nodelist=list(graph.nodes()),
                               node_color=colors,
                               node_size=300)

        nx.draw_networkx_edges(graph, pos, width=3, edge_color=edge_color, arrows=False)

        nx.draw_networkx_edges(graph, pos=pos_nodelist,
                               edgelist=edgelist, width=6,
                               edge_color=subgraph_edge_color, arrows=False)

        if labels:
            nx.draw_networkx_labels(graph, pos, labels)

        plt.axis('off')
        if title_sentence:
            plt.title('\n'.join(wrap(title_sentence, width=60)))

        if figname:
            plt.savefig(figname)
        plt.close('all')

    # Visualization for BA_2Motifs graphs (no atom features)
    def _draw_ba2motifs(self, graph, nodelist, edgelist=None, figname=None):
        return self._draw_subgraph(graph, nodelist, edgelist=edgelist, figname=figname)

    # Visualization for molecule datasets (MUTAG, BBBP)
    def _draw_molecule(self, graph, nodelist, x, edgelist=None, figname=None):
        if self.dataset_name.lower() == 'mutag':
            node_dict = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4: 'I', 5: 'Cl', 6: 'Br'}
            node_idxs = {k: int(v) for k, v in enumerate(np.where(x.cpu().numpy() == 1)[1])}
            node_labels = {k: node_dict[v] for k, v in node_idxs.items()}
            node_color = [
                            '#228B22',  # C - Forest Green
                            '#1E90FF',  # N - Dodger Blue
                            '#8A2BE2',  # O - Blue Violet
                            '#5F9EA0',  # F - Cadet Blue
                            '#4169E1',  # I - Royal Blue
                            '#6A5ACD',  # Cl - Slate Blue
                            '#00CED1',  # Br - Dark Turquoise
                        ]
            colors = [node_color[v % len(node_color)] for v in node_idxs.values()]

        elif self.dataset_name.lower() == 'bbbp':
            element_idxs = {k: int(v) for k, v in enumerate(x[:, 0])}
            node_labels = {
                k: Chem.PeriodicTable.GetElementSymbol(Chem.GetPeriodicTable(), int(v))
                for k, v in element_idxs.items()
            }
            node_color = [
                            '#FF8C00',  # H - Dark Orange
                            '#F4A460',  # C - Sandy Brown
                            '#CD5C5C',  # N - Indian Red
                            '#FFD700',  # O - Gold
                            '#A0522D',  # F - Sienna
                            '#DEB887',  # Cl - BurlyWood
                            '#B22222',  # Br - Firebrick
                            '#DA70D6',  # I - Orchid
                        ]
            colors = [node_color[(v - 1) % len(node_color)] for v in element_idxs.values()]

        else:
            raise NotImplementedError

        self._draw_subgraph(graph, nodelist, colors=colors, labels=node_labels,
                            edgelist=edgelist, edge_color='gray',
                            subgraph_edge_color='black', title_sentence=None,
                            figname=figname)

