import os
import torch
from typing import List
import random
import numpy as np

# ========== Configuration Classes ==========

# Configuration for dataset loading and preprocessing
class DataParser():
    def __init__(self):
        super().__init__()
        self.dataset_name = 'mutag'                     # Dataset name
        self.dataset_dir = './data'                     # Root directory for dataset
        self.task = None                                # Optional: classification/regression
        self.random_split: bool = True                  # Whether to split randomly or use predefined indices
        self.data_split_ratio: List = [0.8, 0.1, 0.1]    # Train / Val / Test split ratios
        self.seed = 1                                   # Random seed for reproducibility


# Configuration for model architecture and behavior
class ModelParser():
    def __init__(self):
        super().__init__()
        self.device: int = 0                             # GPU device index
        self.model_name: str = 'gat'                     # Model type: 'gcn', 'gat', 'gin'
        self.checkpoint: str = './checkpoint'            # Directory for saving checkpoints
        self.concate: bool = False                       # Whether to concatenate GNN layer outputs
        self.latent_dim: List[int] = [128, 128, 128]     # Hidden dimensions for GNN layers
        self.readout: str = 'max'                        # Readout type: 'mean', 'sum', 'max'
        self.mlp_hidden: List[int] = []                  # Hidden dims for MLP head
        self.gnn_dropout: float = 0.0                    # Dropout after GNN layers
        self.dropout: float = 0.5                        # Dropout after MLP layers
        self.adj_normlize: bool = True                   # Normalize adjacency matrix (GCN)
        self.emb_normlize: bool = False                  # L2-normalize GNN output embeddings
        self.enable_prot = True                          # Enable prototype layer
        self.num_prototypes_per_class = 5                # Number of prototypes per class

        # GAT-specific arguments
        self.gat_dropout = 0.6
        self.gat_heads = 10
        self.gat_hidden = 10
        self.gat_concate = True
        self.num_gat_layer = 3

    # Optional post-processing (currently unused)
    def process_args(self) -> None:
        if torch.cuda.is_available():
            self.device = torch.device('cuda', self.device_id)
        else:
            pass


# Configuration for MCTS-based explanation (inherits model/data)
class MCTSParser(DataParser, ModelParser):
    rollout: int = 10                         # Number of rollouts per explanation
    high2low: bool = False                    # Node selection order: high-degree first or not
    c_puct: float = 5                         # MCTS exploration constant
    min_atoms: int = 5                        # Minimum subgraph size
    max_atoms: int = 10                       # Maximum subgraph size
    expand_atoms: int = 10                    # Number of nodes to expand per rollout

    def process_args(self) -> None:
        # Set path to the pretrained model for explanation
        self.explain_model_path = os.path.join(
            self.checkpoint,
            self.dataset_name,
            f"{self.model_name}_best.pth"
        )


# Configuration for training behavior and optimization
class TrainParser():
    def __init__(self):
        super().__init__()
        self.learning_rate = 0.005
        self.batch_size = 24
        self.weight_decay = 0.0
        self.max_epochs = 800
        self.save_epoch = 10
        self.early_stopping = 80

        # Learning rates for specialized optimizers
        self.last_layer_optimizer_lr = 1e-4
        self.joint_optimizer_lrs = {
            'features': 1e-4,
            'add_on_layers': 3e-3,
            'prototype_vectors': 3e-3
        }

        self.warm_epochs = 10                  # Epochs before enabling full training
        #self.proj_epochs = 100                 # Epoch to start prototype projection
        #self.sampling_epochs = 100             # Epoch to start edge sampling
        self.proj_epochs = 90                 
        self.sampling_epochs = 90             
        self.nearest_graphs = 10               # number of graphs for prototype projection


# ========== Instantiate Global Configurations ==========

data_args = DataParser()
model_args = ModelParser()
mcts_args = MCTSParser()
train_args = TrainParser()


# ========== Global Random Seed ==========
# random_seed = 1000
random_seed = 1234
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
