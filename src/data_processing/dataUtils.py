import os
import torch
import pickle
import numpy as np
import os.path as osp

from torch.utils.data import random_split, Subset
from torch_geometric.utils import dense_to_sparse
from torch_geometric.datasets import MoleculeNet
from torch_geometric.data import Data, InMemoryDataset, DataLoader

from utils.Configures import data_args, train_args
from utils.outputUtils import append_record

# ========== Exported Functions ==========

# Main function to load dataset, extract dimensions, and construct data loaders.
def load_dataset():
    dataset = _get_dataset(data_args.dataset_dir, data_args.dataset_name, task=data_args.task)
    input_dim = dataset.num_node_features
    output_dim = int(dataset.num_classes)
    dataloader = _get_dataloader(dataset, train_args.batch_size, data_split_ratio=data_args.data_split_ratio)
    append_record(f"dataset: {data_args.dataset_name}")
    return dataset, input_dim, output_dim, dataloader


# ========== Internal Utilities ==========

# Selects and loads dataset by name.
def _get_dataset(dataset_dir, dataset_name, task=None):
    name = dataset_name.lower()
    molecule_net_dataset_names = [name.lower() for name in MoleculeNet.names.keys()]

    if name == 'mutag':
        return _load_MUTAG(dataset_dir, 'MUTAG')
    elif name == 'ba_2motifs':
        return _load_syn_data(dataset_dir, 'BA_2Motifs')
    elif dataset_name.lower() in molecule_net_dataset_names:
        return _load_MolecueNet(dataset_dir, dataset_name, task)
    else:
        raise NotImplementedError


# Builds train/val/test data loaders either from split indices or by random split.
def _get_dataloader(dataset, batch_size, random_split_flag=True, data_split_ratio=None, seed=2):
    if not random_split_flag and hasattr(dataset, 'supplement'):
        assert 'split_indices' in dataset.supplement.keys(), "Missing split indices in dataset"
        split_indices = dataset.supplement['split_indices']
        train = Subset(dataset, torch.where(split_indices == 0)[0].tolist())
        eval = Subset(dataset, torch.where(split_indices == 1)[0].tolist())
        test = Subset(dataset, torch.where(split_indices == 2)[0].tolist())
    else:
        print("dataloader used")
        num_train = int(data_split_ratio[0] * len(dataset))
        num_eval = int(data_split_ratio[1] * len(dataset))
        num_test = len(dataset) - num_train - num_eval
        train, eval, test = random_split(dataset, [num_train, num_eval, num_test],
                                         generator=torch.Generator().manual_seed(seed))

    return {
        'train': DataLoader(train, batch_size=batch_size, shuffle=True),
        'eval': DataLoader(eval, batch_size=batch_size, shuffle=False),
        'test': DataLoader(test, batch_size=batch_size, shuffle=False)
    }


# Loads synthetic BA_2Motif dataset from a pickle file.
# Each graph includes dense edge matrix, node features, and graph labels.
def _read_ba2motif_data(folder: str, prefix):
    with open(os.path.join(folder, f"{prefix}.pkl"), 'rb') as f:
        dense_edges, node_features, graph_labels = pickle.load(f)

    data_list = []
    for graph_idx in range(dense_edges.shape[0]):
        data_list.append(Data(
            x=torch.from_numpy(node_features[graph_idx]).float(),
            edge_index=dense_to_sparse(torch.from_numpy(dense_edges[graph_idx]))[0],
            y=torch.from_numpy(np.where(graph_labels[graph_idx])[0])
        ))
    return data_list


# PyG-compatible wrapper for the BA_2Motif dataset.
# Applies optional transforms and saves processed data.
class _BA2MotifDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self): return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self): return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self): return [f"{self.name}.pkl"]

    @property
    def processed_file_names(self): return ['data.pt']

    def process(self):
        data_list = _read_ba2motif_data(self.raw_dir, self.name)
        if self.pre_filter:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform:
            data_list = [self.pre_transform(d) for d in data_list]
        torch.save(self.collate(data_list), self.processed_paths[0])


# PyG-compatible wrapper for MUTAG dataset.
# Reads and parses raw text files, builds graph objects, and saves them.
class _MUTAGDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.root = root
        self.name = name.upper()
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def __len__(self):
        return len(self.slices['x']) - 1

    @property
    def raw_dir(self): return os.path.join(self.root, self.name, 'raw')

    @property
    def raw_file_names(self):
        return ['MUTAG_A', 'MUTAG_graph_labels', 'MUTAG_graph_indicator', 'MUTAG_node_labels']

    @property
    def processed_dir(self): return os.path.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self): return ['data.pt']

    def process(self):
        # Check for required raw files
        for fname in ['MUTAG_A.txt', 'MUTAG_graph_labels.txt', 'MUTAG_graph_indicator.txt', 'MUTAG_node_labels.txt']:
            fpath = os.path.join(self.raw_dir, fname)
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"Required raw file '{fname}' is missing in {self.raw_dir}")

        # Read node labels
        try:
            with open(os.path.join(self.raw_dir, 'MUTAG_node_labels.txt')) as f:
                nodes_all = list(map(int, f.read().splitlines()))
        except Exception as e:
            raise ValueError(f"Failed to parse node labels: {e}")
        if len(nodes_all) == 0:
            raise ValueError("No node labels found: file appears to be empty.")

        # Build adjacency matrix from edge list
        try:
            adj_all = np.zeros((len(nodes_all), len(nodes_all)))
            with open(os.path.join(self.raw_dir, 'MUTAG_A.txt')) as f:
                for line in f.read().splitlines():
                    l, r = map(int, line.split(', '))
                    adj_all[l - 1, r - 1] = 1
        except Exception as e:
            raise ValueError(f"Failed to parse edge list: {e}")

        # Read graph indicator
        try:
            with open(os.path.join(self.raw_dir, 'MUTAG_graph_indicator.txt')) as f:
                graph_indicator = np.array(list(map(int, f.read().splitlines())))
        except Exception as e:
            raise ValueError(f"Failed to parse graph indicator: {e}")

        # Read graph labels
        try:
            with open(os.path.join(self.raw_dir, 'MUTAG_graph_labels.txt')) as f:
                graph_labels = list(map(int, f.read().splitlines()))
        except Exception as e:
            raise ValueError(f"Failed to parse graph labels: {e}")

        if len(graph_indicator) != len(nodes_all):
            raise ValueError("Mismatch between number of nodes and graph indicators.")

        # Construct each graph
        data_list = []
        for i in range(1, 189):
            idx = np.where(graph_indicator == i)
            if len(idx[0]) == 0:
                raise ValueError(f"Graph {i} is empty: no nodes assigned to it.")
            start, length = idx[0][0], len(idx[0])
            adj = adj_all[start:start + length, start:start + length]
            label = int(graph_labels[i - 1] == 1)
            feature = nodes_all[start:start + length]
            try:
                one_hot_feature = np.eye(7)[np.array(feature).reshape(-1)]
            except Exception as e:
                raise ValueError(f"One-hot encoding failed for graph {i}: {e}")

            data_list.append(Data(
                x=torch.from_numpy(one_hot_feature).float(),
                edge_index=dense_to_sparse(torch.from_numpy(adj))[0],
                y=label
            ))

        torch.save(self.collate(data_list), self.processed_paths[0])



# Wrapper for loading the MUTAG dataset.
def _load_MUTAG(dataset_dir, dataset_name):
    return _MUTAGDataset(root=dataset_dir, name=dataset_name)


# Wrapper for loading the BA_2Motifs synthetic dataset.
def _load_syn_data(dataset_dir, dataset_name):
    if dataset_name.lower() != 'ba_2motifs':
        raise NotImplementedError
    dataset = _BA2MotifDataset(root=dataset_dir, name=dataset_name)
    dataset.node_type_dict = {k: v for k, v in enumerate(range(dataset.num_classes))}
    dataset.node_color = None
    return dataset


# Wrapper for loading the BBBP dataset or any dataset from MoleculeNet.
def _load_MolecueNet(dataset_dir, dataset_name, task=None):
    # Map lowercase dataset name to its original case-sensitive form
    molecule_net_dataset_names = {name.lower(): name for name in MoleculeNet.names.keys()}

    # Load dataset with proper name
    dataset = MoleculeNet(root=dataset_dir, name=molecule_net_dataset_names[dataset_name.lower()])

    # Ensure node features are float tensors
    dataset.data.x = dataset.data.x.float()

    # Process labels: handle both single-task and multi-task formats
    if task is None:
        dataset.data.y = dataset.data.y.squeeze().long()     # Single task: flatten
    else:
        dataset.data.y = dataset.data.y[:, 0].long()         # Multi-task: select column 0 (TODO: generalize)

    # Compatibility hooks (optional for visual explanation)
    dataset.node_type_dict = None
    dataset.node_color = None

    return dataset