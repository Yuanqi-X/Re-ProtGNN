import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam

from utils.Configures import data_args, train_args, model_args
from utils.outputUtils import append_record, save_best
from evaluation.explanation import get_explanation


# ========== Exported Functions ==========

# Main training function that performs warm-up training, prototype projection, and joint training.
def train(clst, sep, dataset, dataloader, gnnNets, output_dim, criterion, ckpt_dir):
    # Initialize optimizer
    optimizer = Adam(
        gnnNets.parameters(), 
        lr=train_args.learning_rate, 
        weight_decay=train_args.weight_decay
    )

    # Log average node/edge stats for dataset
    _log_dataset_stats(dataset)
    best_acc, early_stop_count = 0.0, 0
    data_indices = dataloader['train'].dataset.indices

    os.makedirs(ckpt_dir, exist_ok=True)

    # Start training loop
    for epoch in range(train_args.max_epochs):
        acc, loss_list, ld_loss_list = [], [], []

        # Project prototypes periodically (every 10 epochs after projection phase starts)
        if epoch >= train_args.proj_epochs and epoch % 15 == 0:
            _project_prototypes(gnnNets, dataset, data_indices, output_dim)

        gnnNets.train()
        # Enable warm-up mode or full joint training
        _set_training_mode(gnnNets, epoch < train_args.warm_epochs)

        # Train on all batches in the current epoch
        for batch in dataloader['train']:
            logits, probs, _, _, min_distances = gnnNets(batch)
            loss = _compute_total_loss(logits, batch.y, min_distances, gnnNets, output_dim, criterion, clst, sep)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(gnnNets.parameters(), 2.0)
            optimizer.step()

            _, prediction = torch.max(logits, -1)
            loss_list.append(loss.item())
            acc.append(prediction.eq(batch.y).cpu().numpy())

        # Logging training progress
        train_acc = np.concatenate(acc).mean()
        append_record(f"Epoch {epoch:2d}, loss: {np.mean(loss_list):.3f}, acc: {train_acc:.3f}")
        print(f"Train Epoch: {epoch} | Loss: {np.mean(loss_list):.3f} | Acc: {train_acc:.3f}")

        # Evaluate on validation set
        eval_state = _evaluate(dataloader['eval'], gnnNets, criterion)
        append_record(f"Eval epoch {epoch:2d}, loss: {eval_state['loss']:.3f}, acc: {eval_state['acc']:.3f}")
        # print(f"Eval Epoch: {epoch} | Loss: {eval_state['loss']:.3f} | Acc: {eval_state['acc']:.3f}")

        # Early stopping and checkpoint saving
        if eval_state['acc'] > best_acc:
            best_acc = eval_state['acc']
            early_stop_count = 0
            is_best = True
        else:
            early_stop_count += 1
            is_best = False

        if is_best or epoch % train_args.save_epoch == 0:
            save_best(ckpt_dir, epoch, gnnNets, model_args.model_name, eval_state['acc'], is_best)

        if early_stop_count > train_args.early_stopping:
            break

    print(f"The best validation accuracy is {best_acc:.4f}")


# ========== Internal Utilities ==========

# Logs average node and edge statistics of the dataset
def _log_dataset_stats(dataset):
    avg_nodes = sum(data.x.size(0) for data in dataset) / len(dataset)
    avg_edges = sum(data.edge_index.size(1) for data in dataset) / len(dataset) / 2
    print(f"Graphs: {len(dataset)}, Avg Nodes: {avg_nodes:.2f}, Avg Edges: {avg_edges:.2f}")


# Configures the model's training mode:
# If warm_only=True, only GNN layers and prototypes are trainable (for warm-up);
# Otherwise, the final classification layer is also trained (joint training).
def _set_training_mode(model, warm_only=True):
    for p in model.model.gnn_layers.parameters():
        p.requires_grad = True
    model.model.prototype_vectors.requires_grad = True
    for p in model.model.last_layer.parameters():
        p.requires_grad = not warm_only


# Computes the full loss function, including:
# - Classification loss
# - Cluster loss (encourage prototypes to be close to samples of same class)
# - Separation loss (encourage prototypes to be far from other classes)
# - L1 regularization loss (for sparsity)
# - Diversity loss (discourage redundant prototypes within each class)
def _compute_total_loss(logits, labels, min_distances, model, output_dim, criterion, clst, sep):
    loss = criterion(logits, labels)

    proto_identity = torch.t(model.model.prototype_class_identity[:, labels].bool()).to(model_args.device)
    correct_dist = torch.min(min_distances[proto_identity].reshape(-1, model_args.num_prototypes_per_class), dim=1)[0]
    wrong_dist = torch.min(min_distances[~proto_identity].reshape(-1, (output_dim - 1) * model_args.num_prototypes_per_class), dim=1)[0]

    cluster_cost = correct_dist.mean()
    separation_cost = -wrong_dist.mean()

    l1_mask = 1 - torch.t(model.model.prototype_class_identity).to(model_args.device)
    l1 = (model.model.last_layer.weight * l1_mask).norm(p=1)

    diversity = 0
    for k in range(output_dim):
        p = model.model.prototype_vectors[k * model_args.num_prototypes_per_class:(k + 1) * model_args.num_prototypes_per_class]
        p = F.normalize(p, p=2, dim=1)
        sim = torch.mm(p, p.T) - torch.eye(p.shape[0], device=p.device) - 0.3
        diversity += torch.sum(torch.where(sim > 0, sim, torch.zeros_like(sim)))

    return loss + clst * cluster_cost + sep * separation_cost + 5e-4 * l1 + 0.0 * diversity


# Projects learned prototype vectors to closest matching real data embeddings from training set.
# Helps align prototypes with actual input features.
"""
def _project_prototypes(model, dataset, indices, output_dim):
    model.eval()
    for i in range(output_dim * model_args.num_prototypes_per_class):
        label = i // model_args.num_prototypes_per_class
        best_sim, count = 0.0, 0
        for j in range(i * 10, len(indices)):
            data = dataset[indices[j]]
            if data.y != label:
                continue
            count += 1
            _, sim, prot = get_explanation(data, model, model.model.prototype_vectors[i])
            if sim > best_sim:
                best_sim = sim
                best_proj = prot
            if count >= 10:
                model.model.prototype_vectors.data[i] = best_proj
                print(f'Prototype {i} projected.')
                break
"""
def _project_prototypes(model, dataset, indices, output_dim):
    model.eval()
    for i in range(output_dim * model_args.num_prototypes_per_class):
        label = i // model_args.num_prototypes_per_class
        best_sim, best_proj, count = 0.0, None, 0
        for j in range(i * 10, len(indices)):
            data = dataset[indices[j]]
            if data.y != label:
                continue
            count += 1
            _, sim, prot = get_explanation(data, model, model.model.prototype_vectors[i])
            if sim > best_sim:
                best_sim = sim
                best_proj = prot
            if count >= 10:
                break
        if best_proj is not None:
            model.model.prototype_vectors.data[i] = best_proj
            print(f'Prototype {i} projected.')
        else:
            print(f'Warning: No projection found for prototype {i}')


# Evaluates the model on the validation (or test) set.
# Returns average loss and accuracy across all batches.
def _evaluate(eval_dataloader, model, criterion):
    model.eval()
    total_loss, total_acc = [], []

    with torch.no_grad():
        for batch in eval_dataloader:
            logits, probs, _, _, _ = model(batch)
            loss = criterion(logits, batch.y)

            _, predictions = torch.max(logits, dim=-1)
            total_loss.append(loss.item())
            total_acc.append(predictions.eq(batch.y).cpu().numpy())

    return {
        'loss': np.mean(total_loss),
        'acc': np.concatenate(total_acc, axis=0).mean()
    }
