import torch.nn as nn
from models.GCN import GCNNet
from models.GAT import GATNet
from models.GIN import GINNet

from utils.outputUtils import append_record

# ========== Exported Function ==========

# Main setup function to initialize GNN model and loss function.
# Returns the wrapped GNN model (_GnnNets) and criterion.
def setup_model(input_dim, output_dim, model_args):
    gnnNets = _GnnNets(input_dim, output_dim, model_args)
    gnnNets.to_device()
    criterion = nn.CrossEntropyLoss()
    append_record(f"model: {model_args.model_name}")
    return gnnNets, criterion


# ========== Internal Utilities ==========

# Base wrapper for GNN models. Adds device management and forward logic.
class _GnnBase(nn.Module):
    def __init__(self):
        super(_GnnBase, self).__init__()

    def forward(self, data):
        data = data.to(self.device)
        logits, prob, emb1, emb2, min_distances = self.model(data)
        return logits, prob, emb1, emb2, min_distances

    # Load matching parameters into model
    def update_state_dict(self, state_dict):
        current_state = self.state_dict()
        filtered_state = {k: v for k, v in state_dict.items() if k in current_state}
        self.load_state_dict(filtered_state)

    # Move model to device
    def to_device(self):
        self.to(self.device)


# Final GNN wrapper combining architecture + device binding + enhanced forward
class _GnnNets(_GnnBase):
    def __init__(self, input_dim, output_dim, model_args):
        super(_GnnNets, self).__init__()
        self.model = _get_model(input_dim, output_dim, model_args)
        self.device = model_args.device

    # Extended forward with support for prototype similarity control
    def forward(self, data, protgnn_plus=False, similarity=None):
        data = data.to(self.device)
        return self.model(data, protgnn_plus, similarity)


# Selects model architecture based on config (gcn, gat, gin).
def _get_model(input_dim, output_dim, model_args):
    model_name = model_args.model_name.lower()
    if model_name == 'gcn':
        return GCNNet(input_dim, output_dim, model_args)
    elif model_name == 'gat':
        return GATNet(input_dim, output_dim, model_args)
    elif model_name == 'gin':
        return GINNet(input_dim, output_dim, model_args)
    else:
        raise NotImplementedError(f"Unsupported model: {model_args.model_name}")

