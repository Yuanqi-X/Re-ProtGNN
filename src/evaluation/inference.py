import torch
import numpy as np

from utils.outputUtils import append_record


# ========== Exported Function ==========

# Function to evaluate the model on a test dataset
def run_inference(test_dataloader, model, criterion):
    model.eval()  # Set model to evaluation mode

    all_losses = []        # List to store batch losses
    all_accuracies = []    # List to store batch accuracy arrays
    all_probs = []         # List to store prediction probabilities
    all_preds = []         # List to store prediction class labels

    with torch.no_grad():  # Disable gradient tracking for evaluation
        for batch in test_dataloader:
            logits, probs, _, _, _ = model(batch)        # Forward pass
            loss = criterion(logits, batch.y)            # Compute loss

            _, predicted = torch.max(logits, dim=-1)     # Get predicted class

            # Store per-batch results
            all_losses.append(loss.item())
            all_accuracies.append(predicted.eq(batch.y).cpu().numpy())
            all_preds.append(predicted)
            all_probs.append(probs)

    # Calculate average loss and accuracy over entire test set
    test_loss = np.mean(all_losses)
    test_acc = np.concatenate(all_accuracies).mean()
    test_state = {'loss': test_loss, 'acc': test_acc}

    # Print and log the results
    print(f"Test: | Loss: {test_loss:.3f} | Acc: {test_acc:.3f}")
    append_record(f"loss: {test_loss:.3f}, acc: {test_acc:.3f}")

    # Concatenate all batches into final prediction and probability arrays
    all_probs = torch.cat(all_probs, dim=0).cpu().detach().numpy()
    all_preds = torch.cat(all_preds, dim=0).cpu().detach().numpy()

    return test_state, all_probs, all_preds
