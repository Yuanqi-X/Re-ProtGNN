import os
import torch
import argparse

from utils.Configures import data_args, train_args, model_args, mcts_args
from data_processing.dataUtils import load_dataset
from models import setup_model
from models.train import train
from evaluation.inference import run_inference as test
from evaluation.explanation import exp_visualize


# ========== Main Experiment Pipeline ==========
def main(args):
    # Step 1: Load dataset and construct dataloaders
    print('====================Loading Data====================')
    dataset, input_dim, output_dim, dataloader = load_dataset()
    print(f"Dataset Name: {data_args.dataset_name}")
    print(f"Dataset Length: {len(dataset)}")

    # Step 2: Initialize model and loss function
    print('====================Setting Up Model====================')
    gnnNets, criterion = setup_model(input_dim, output_dim, model_args)
    print(f"gnnNets: {gnnNets}")

    # Step 3: Train the model with prototype alignment and joint optimization
    print('====================Training Model==================')
    ckpt_dir = f"./src/checkpoint/{data_args.dataset_name}/"
    train(args.clst, args.sep, dataset, dataloader, gnnNets, output_dim, criterion, ckpt_dir)

    # Step 4: Evaluate the best model on the test set
    print('====================Testing==================')
    best_checkpoint = torch.load(os.path.join(ckpt_dir, f'{model_args.model_name}_best.pth'))
    gnnNets.update_state_dict(best_checkpoint['net'])
    test(dataloader['test'], gnnNets, criterion)

    # Step 5: Output explanations for selected graphs
    print('====================Generating Explanations====================')
    exp_visualize(dataset, dataloader, gnnNets, output_dim)


# ========== Entry Point ==========
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Implementation of Re-ProtGNN')

    # Command-line argument for cluster loss weight
    parser.add_argument('--clst', type=float, default=0.0,
                        help='cluster')

    # Command-line argument for separation loss weight
    parser.add_argument('--sep', type=float, default=0.0,
                        help='separation')

    args = parser.parse_args()
    main(args)
