"""
Federated Learning with Local Reweighting for LFF (Learning From Failure)

This script implements a federated learning approach for bias mitigation where:
1. Multiple clients train locally with LFF and local reweighting
2. A server aggregates model updates from clients
3. Both biased and debiased models are maintained at global and local levels
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import argparse
import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# Import local modules
from data.util import get_dataset, IdxDataset
from module.loss import GeneralizedCELoss
from module.util import get_model, get_backbone
from federated.client import FederatedLFFClient
from federated.server import FederatedLFFServer
from federated.utils import aggregate_models, get_client_data_distribution


class FederatedLFFTrainer:
    """
    Main trainer for federated LFF with local reweighting.
    """
    
    def __init__(self, args):
        """Initialize federated trainer"""
        self.args = args
        self.device = torch.device(args.device)
        
        # Setup logging
        self.log_dir = os.path.join(args.log_dir, args.dataset, args.exp)
        self.result_dir = os.path.join(self.log_dir, "result")
        os.makedirs(self.result_dir, exist_ok=True)
        
        print(f"[Federated LFF] Dataset: {args.dataset}")
        print(f"[Federated LFF] Model: {args.model}")
        print(f"[Federated LFF] Number of clients: {args.num_clients}")
        print(f"[Federated LFF] Aggregation strategy: {args.aggregation_strategy}")
        print(f"[Federated LFF] Reweighting method: {args.reweighting_method}")
        
        # Load datasets
        self._load_datasets()
        
        # Create servers and clients
        self._create_server_and_clients()
    
    def _load_datasets(self):
        """Load and partition datasets for federated setting"""
        
        # Determine model and batch size
        data2model = {'cmnist': self.args.model, 'bar': "ResNet18", 'bffhq': "ResNet18"}
        data2batch_size = {'cmnist': 256, 'bar': 64, 'bffhq': 64}
        data2preprocess = {'cmnist': None, 'bar': True, 'bffhq': True}
        
        self.model_name = data2model.get(self.args.dataset, self.args.model)
        self.batch_size = data2batch_size.get(self.args.dataset, 256)
        preprocess = data2preprocess.get(self.args.dataset, None)
        
        # Load full datasets
        self.train_dataset = get_dataset(
            self.args.dataset,
            data_dir=self.args.data_dir,
            dataset_split="train",
            transform_split="train",
            percent=self.args.percent,
            use_preprocess=preprocess,
        )
        
        self.valid_dataset = get_dataset(
            self.args.dataset,
            data_dir=self.args.data_dir,
            dataset_split="valid",
            transform_split="valid",
            percent=self.args.percent,
            use_preprocess=preprocess,
        )
        
        self.test_dataset = get_dataset(
            self.args.dataset,
            data_dir=self.args.data_dir,
            dataset_split="test",
            transform_split="valid",
            percent=self.args.percent,
            use_preprocess=preprocess,
        )
        
        # Get number of classes
        train_target_attr = []
        for data in self.train_dataset.data:
            train_target_attr.append(int(data.split('_')[-2]))
        train_target_attr = torch.LongTensor(train_target_attr)
        self.num_classes = torch.max(train_target_attr).item() + 1
        
        # Partition train dataset among clients
        client_indices = get_client_data_distribution(
            self.train_dataset,
            self.args.num_clients,
            distribution=self.args.data_distribution
        )
        
        # Create client-specific datasets
        self.client_datasets = []
        for client_id, indices in enumerate(client_indices):
            client_dataset = Subset(self.train_dataset, indices)
            client_dataset = IdxDataset(client_dataset)
            self.client_datasets.append(client_dataset)
            print(f"[Dataset] Client {client_id} has {len(indices)} samples")
        
        # Test and validation loaders (global)
        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
    
    def _create_server_and_clients(self):
        """Create federated server and clients"""
        
        # Create global models
        global_model_b = get_backbone(self.model_name, self.num_classes, 
                                      pretrained=self.args.resnet_pretrained)
        global_model_d = get_backbone(self.model_name, self.num_classes,
                                      pretrained=self.args.resnet_pretrained)
        
        # Create server
        self.server = FederatedLFFServer(
            global_model_b, global_model_d,
            self.args.num_clients,
            device=self.args.device,
            aggregation_strategy=self.args.aggregation_strategy,
            compression_enabled=self.args.compression_enabled,
            compression_ratio=self.args.compression_ratio
        )
        
        # Create loss functions
        criterion = nn.CrossEntropyLoss(reduction='none')
        bias_criterion = GeneralizedCELoss(q=self.args.q)
        
        # Create clients
        self.clients = []
        for client_id in range(self.args.num_clients):
            # Create client models (copies of global models)
            client_model_b = get_backbone(self.model_name, self.num_classes,
                                         pretrained=self.args.resnet_pretrained)
            client_model_d = get_backbone(self.model_name, self.num_classes,
                                         pretrained=self.args.resnet_pretrained)
            
            # Create client
            client = FederatedLFFClient(
                client_id=client_id,
                model_b=client_model_b,
                model_d=client_model_d,
                local_dataset=self.client_datasets[client_id],
                criterion=criterion,
                bias_criterion=bias_criterion,
                lr=self.args.lr,
                device=self.args.device,
                num_classes=self.num_classes
            )
            
            # Register with server
            self.server.register_client(client_id, len(self.client_datasets[client_id]))
            
            self.clients.append(client)
        
        print(f"[Server] Created server with {self.args.num_clients} clients")
    
    def train(self, num_rounds: int = 10, local_epochs: int = 1):
        """
        Train federated LFF model.
        
        Args:
            num_rounds: Number of communication rounds
            local_epochs: Number of local training epochs per round
        """
        
        best_acc_d = 0.0
        best_acc_b = 0.0
        
        # Create log file
        log_path = os.path.join(self.result_dir, "federated_training_log.csv")
        with open(log_path, "w") as f:
            f.write("round,client_id,loss_b,loss_d,val_acc_d,val_acc_b,test_acc_d,test_acc_b\n")
        
        for round_num in range(num_rounds):
            print(f"\n{'='*60}")
            print(f"Communication Round {round_num + 1}/{num_rounds}")
            print(f"{'='*60}")
            
            # Send global models to all clients
            print("[Server] Sending global models to clients...")
            for client_id, client in enumerate(self.clients):
                package = self.server.send_model_to_client(client_id)
                client.load_model_state(
                    package['model_b_state'],
                    package['model_d_state']
                )
            
            # Local training on clients
            client_updates = []
            print("[Clients] Starting local training...")
            
            for client_id, client in enumerate(self.clients):
                # Train locally
                metrics = client.local_train(
                    num_epochs=local_epochs,
                    batch_size=self.args.batch_size,
                    verbose=False
                )
                
                # Get model updates
                state_b, state_d = client.get_model_state()
                
                # Prepare update package
                update = {
                    'client_id': client_id,
                    'model_b_state': state_b,
                    'model_d_state': state_d,
                    'data_size': len(client.local_dataset),
                    'metrics': metrics
                }
                
                client_updates.append(update)
                
                print(f"[Client {client_id}] Loss_B: {metrics['loss_b']:.4f}, "
                      f"Loss_D: {metrics['loss_d']:.4f}")
            
            # Server aggregates client models
            print("[Server] Aggregating client models...")
            self.server.aggregate_client_models(client_updates)
            
            # Evaluate global models
            print("[Server] Evaluating global models...")
            val_metrics = self.server.evaluate_global_models(self.valid_loader)
            test_metrics = self.server.evaluate_global_models(self.test_loader)
            
            print(f"[Validation] Acc_D: {val_metrics['acc_d_total']:.2f}%, "
                  f"Acc_B: {val_metrics['acc_b_total']:.2f}%")
            print(f"[Test] Acc_D: {test_metrics['acc_d_total']:.2f}%, "
                  f"Acc_B: {test_metrics['acc_b_total']:.2f}%")
            print(f"       Conflict D: {test_metrics['acc_d_conflict']:.2f}%, "
                  f"Conflict B: {test_metrics['acc_b_conflict']:.2f}%")
            
            # Save best models
            if test_metrics['acc_d_total'] > best_acc_d:
                best_acc_d = test_metrics['acc_d_total']
                self.server.save_global_models(self.result_dir, round_num)
            
            # Log results
            with open(log_path, "a") as f:
                for update in client_updates:
                    client_id = update['client_id']
                    metrics = update['metrics']
                    f.write(f"{round_num},{client_id},{metrics['loss_b']:.4f},"
                           f"{metrics['loss_d']:.4f},{val_metrics['acc_d_total']:.2f},"
                           f"{val_metrics['acc_b_total']:.2f},"
                           f"{test_metrics['acc_d_total']:.2f},"
                           f"{test_metrics['acc_b_total']:.2f}\n")
        
        print("\n" + "="*60)
        print("Federated Training Completed!")
        print("="*60)
        print(f"Best Test Accuracy (Model D): {best_acc_d:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='Federated LFF with Local Reweighting')
    
    # Federated learning args
    parser.add_argument("--num_clients", type=int, default=5,
                       help="Number of federated clients")
    parser.add_argument("--aggregation_strategy", type=str, default='fedavg',
                       choices=['fedavg', 'fedprox'],
                       help="Model aggregation strategy")
    parser.add_argument("--data_distribution", type=str, default='iid',
                       choices=['iid', 'non-iid'],
                       help="Data distribution among clients")
    parser.add_argument("--reweighting_method", type=str, 
                       default='adaptive_local',
                       choices=['loss_based', 'importance', 'uncertainty', 'adaptive_local'],
                       help="Local reweighting method")
    parser.add_argument("--compression_enabled", action='store_true',
                       help="Enable communication compression")
    parser.add_argument("--compression_ratio", type=float, default=0.1,
                       help="Compression ratio (0-1)")
    parser.add_argument("--num_rounds", type=int, default=10,
                       help="Number of communication rounds")
    parser.add_argument("--local_epochs", type=int, default=1,
                       help="Number of local training epochs per round")
    
    # Training args
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--q", type=float, default=0.7,
                       help="GCE parameter q")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loading workers")
    
    # Dataset args
    parser.add_argument("--dataset", type=str, default='cmnist',
                       choices=['cmnist', 'bar', 'bffhq'],
                       help="Dataset")
    parser.add_argument("--percent", type=str, default='1pct',
                       help="Percentage of conflicting samples")
    parser.add_argument("--model", type=str, default='MLP',
                       help="Model architecture")
    
    # Path args
    parser.add_argument("--data_dir", type=str, default='./data',
                       help="Data directory")
    parser.add_argument("--log_dir", type=str, default='./log',
                       help="Log directory")
    parser.add_argument("--exp", type=str, default='federated_lff',
                       help="Experiment name")
    parser.add_argument("--device", type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help="Device")
    parser.add_argument("--resnet_pretrained", action='store_true',
                       help="Use pretrained ResNet")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create trainer and train
    trainer = FederatedLFFTrainer(args)
    trainer.train(num_rounds=args.num_rounds, local_epochs=args.local_epochs)


if __name__ == '__main__':
    main()
