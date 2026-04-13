"""
Example usage of Federated LFF with Local Reweighting.
This script demonstrates how to use the federated framework programmatically.
"""

import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
import os

# Import federated components
from federated.client import FederatedLFFClient
from federated.server import FederatedLFFServer
from federated.utils import get_client_data_distribution
from local_reweighting import AdaptiveLocalReweighting, ConflictDetectionReweighting

# Import existing components
from data.util import get_dataset, IdxDataset
from module.loss import GeneralizedCELoss
from module.util import get_backbone
import torch.nn as nn


class SimpleFederatedLFFExample:
    """Simple example of federated LFF training"""
    
    def __init__(self, num_clients=3, dataset='cmnist', model='MLP'):
        self.num_clients = num_clients
        self.dataset_name = dataset
        self.model_name = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = 10
        
    def load_data(self):
        """Load and partition dataset"""
        print(f"Loading {self.dataset_name} dataset...")
        
        # Load full training dataset
        self.train_dataset = get_dataset(
            self.dataset_name,
            data_dir='./data',
            dataset_split='train',
            transform_split='train',
            percent='1pct',
            use_preprocess=None
        )
        
        # Load test dataset
        self.test_dataset = get_dataset(
            self.dataset_name,
            data_dir='./data',
            dataset_split='test',
            transform_split='train',
            percent='1pct',
            use_preprocess=None
        )
        
        # Partition among clients
        client_indices = get_client_data_distribution(
            self.train_dataset,
            self.num_clients,
            distribution='non-iid'  # Non-IID for more realistic scenario
        )
        
        # Create subset datasets for each client
        self.client_datasets = []
        for client_id, indices in enumerate(client_indices):
            from torch.utils.data import Subset
            client_dataset = Subset(self.train_dataset, indices)
            client_dataset = IdxDataset(client_dataset)
            self.client_datasets.append(client_dataset)
            print(f"  Client {client_id}: {len(indices)} samples")
        
        # Create test loader
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=256,
            shuffle=False,
            num_workers=4
        )
    
    def create_models(self):
        """Create global and client models"""
        print("Creating models...")
        
        # Global models
        self.global_model_b = get_backbone(self.model_name, self.num_classes)
        self.global_model_d = get_backbone(self.model_name, self.num_classes)
        
        # Create server
        self.server = FederatedLFFServer(
            self.global_model_b,
            self.global_model_d,
            self.num_clients,
            device=self.device,
            aggregation_strategy='fedavg'
        )
        
        # Loss functions
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.bias_criterion = GeneralizedCELoss(q=0.7)
        
        # Create clients
        self.clients = []
        for client_id in range(self.num_clients):
            client_model_b = get_backbone(self.model_name, self.num_classes)
            client_model_d = get_backbone(self.model_name, self.num_classes)
            
            client = FederatedLFFClient(
                client_id=client_id,
                model_b=client_model_b,
                model_d=client_model_d,
                local_dataset=self.client_datasets[client_id],
                criterion=self.criterion,
                bias_criterion=self.bias_criterion,
                lr=1e-3,
                device=self.device,
                num_classes=self.num_classes
            )
            
            self.server.register_client(client_id, len(self.client_datasets[client_id]))
            self.clients.append(client)
        
        print(f"  Global Model B: {sum(p.numel() for p in self.global_model_b.parameters())} params")
        print(f"  Global Model D: {sum(p.numel() for p in self.global_model_d.parameters())} params")
    
    def federated_round(self, round_num, local_epochs=1):
        """Execute one federated learning round"""
        print(f"\n--- Communication Round {round_num + 1} ---")
        
        # Send global models to clients
        for client_id, client in enumerate(self.clients):
            package = self.server.send_model_to_client(client_id)
            client.load_model_state(
                package['model_b_state'],
                package['model_d_state']
            )
        
        # Local training
        client_updates = []
        for client_id, client in enumerate(self.clients):
            print(f"Client {client_id} training...")
            metrics = client.local_train(
                num_epochs=local_epochs,
                batch_size=32,
                verbose=False
            )
            
            state_b, state_d = client.get_model_state()
            update = {
                'client_id': client_id,
                'model_b_state': state_b,
                'model_d_state': state_d,
                'data_size': len(client.local_dataset),
                'metrics': metrics
            }
            client_updates.append(update)
            
            print(f"  Loss B: {metrics['loss_b']:.4f}, Loss D: {metrics['loss_d']:.4f}")
        
        # Server aggregation
        print("Server aggregating models...")
        self.server.aggregate_client_models(client_updates)
        
        # Evaluation
        print("Evaluating global models...")
        test_metrics = self.server.evaluate_global_models(self.test_loader)
        print(f"Test Acc D: {test_metrics['acc_d_total']:.2f}% | "
              f"Conflict: {test_metrics['acc_d_conflict']:.2f}%")
    
    def run(self, num_rounds=3):
        """Run federated training"""
        print("="*60)
        print("Federated LFF with Local Reweighting")
        print("="*60)
        
        self.load_data()
        self.create_models()
        
        for round_num in range(num_rounds):
            self.federated_round(round_num, local_epochs=1)
        
        print("\n" + "="*60)
        print("Training completed!")
        print("="*60)


def demo_local_reweighting():
    """Demonstrate local reweighting functionality"""
    print("\n" + "="*60)
    print("Local Reweighting Demo")
    print("="*60)
    
    num_samples = 100
    num_classes = 10
    
    # Create dummy data
    logits_b = torch.randn(num_samples, num_classes)
    logits_d = torch.randn(num_samples, num_classes)
    losses = torch.rand(num_samples)
    labels = torch.randint(0, num_classes, (num_samples,))
    indices = torch.arange(num_samples)
    
    # 1. Conflict Detection Reweighting
    print("\n1. Conflict Detection Reweighting:")
    conflict_reweighter = ConflictDetectionReweighting()
    conflict_reweighter.initialize(num_samples)
    weights, conflict_mask = conflict_reweighter.detect_and_reweight(
        indices, logits_b, logits_d, labels
    )
    print(f"   Conflict samples: {conflict_mask.sum().item()}/{num_samples}")
    print(f"   Conflict weight range: [{weights[conflict_mask].min():.4f}, {weights[conflict_mask].max():.4f}]")
    
    # 2. Adaptive Local Reweighting
    print("\n2. Adaptive Local Reweighting:")
    adaptive_reweighter = AdaptiveLocalReweighting(num_classes)
    adaptive_reweighter.initialize(num_samples)
    
    for progress in [0.0, 0.5, 1.0]:
        weight_dict = adaptive_reweighter.compute_adaptive_weights(
            indices, logits_b, logits_d, losses, losses, labels,
            progress_ratio=progress
        )
        print(f"   Progress {progress*100:.0f}%:")
        print(f"     Alpha conflict: {weight_dict['alpha_conflict']:.4f}")
        print(f"     Alpha loss: {weight_dict['alpha_loss']:.4f}")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_clients', type=int, default=3)
    parser.add_argument('--num_rounds', type=int, default=3)
    parser.add_argument('--dataset', type=str, default='cmnist')
    parser.add_argument('--model', type=str, default='MLP')
    parser.add_argument('--demo_reweighting', action='store_true')
    args = parser.parse_args()
    
    # Run federated training example
    example = SimpleFederatedLFFExample(
        num_clients=args.num_clients,
        dataset=args.dataset,
        model=args.model
    )
    try:
        example.run(num_rounds=args.num_rounds)
    except Exception as e:
        print(f"\nNote: Example requires proper data setup. Error: {e}")
        print("Please ensure CMNIST dataset is available in ./data directory")
    
    # Demo local reweighting
    if args.demo_reweighting:
        demo_local_reweighting()
