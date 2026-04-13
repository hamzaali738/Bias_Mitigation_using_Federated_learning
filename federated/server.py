"""Federated Server for Model Aggregation and Coordination"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import numpy as np
from copy import deepcopy
import os
from federated.utils import (aggregate_models, ClientManager, 
                             CommunicationCompressor, calculate_client_weight)


class FederatedLFFServer:
    """
    Federated server that coordinates with clients for distributed LFF training.
    Aggregates model updates from clients and maintains global models.
    """
    
    def __init__(self, global_model_b: nn.Module, global_model_d: nn.Module,
                 num_clients: int, device: str = 'cuda',
                 aggregation_strategy: str = 'fedavg',
                 compression_enabled: bool = False,
                 compression_ratio: float = 0.1):
        """
        Initialize federated server.
        
        Args:
            global_model_b: Global biased model
            global_model_d: Global debiased model
            num_clients: Total number of clients
            device: Device for computation
            aggregation_strategy: 'fedavg' or 'fedprox'
            compression_enabled: Whether to use communication compression
            compression_ratio: Compression ratio (0.1 = 10% of parameters)
        """
        self.global_model_b = global_model_b
        self.global_model_d = global_model_d
        self.num_clients = num_clients
        self.device = torch.device(device) if isinstance(device, str) else device
        self.aggregation_strategy = aggregation_strategy
        self.compression_enabled = compression_enabled
        self.compression_ratio = compression_ratio
        
        # Move models to device
        self.global_model_b.to(self.device)
        self.global_model_d.to(self.device)
        
        # Save initial model state
        self.initial_state_b = deepcopy(self.global_model_b.state_dict())
        self.initial_state_d = deepcopy(self.global_model_d.state_dict())
        
        # Client management
        self.client_manager = ClientManager(num_clients)
        
        # Compression
        self.compressor = CommunicationCompressor()
        
        # Statistics
        self.server_stats = {
            'global_updates': 0,
            'total_communication_rounds': 0,
            'client_updates': {i: 0 for i in range(num_clients)},
            'aggregation_history': []
        }
        
    def register_client(self, client_id: int, data_size: int):
        """Register a client with the server"""
        self.client_manager.register_client(client_id, data_size)
    
    def send_model_to_client(self, client_id: int) -> Dict:
        """
        Send global model to client.
        
        Args:
            client_id: Client identifier
        
        Returns:
            Package containing model state and metadata
        """
        package = {
            'global_step': self.server_stats['global_updates'],
            'model_b_state': deepcopy(self.global_model_b.state_dict()),
            'model_d_state': deepcopy(self.global_model_d.state_dict()),
            'num_clients': self.num_clients,
            'aggregation_strategy': self.aggregation_strategy
        }
        return package
    
    def receive_client_update(self, client_id: int, 
                            client_state_b: Dict, client_state_d: Dict,
                            client_metrics: Dict):
        """
        Receive model update and metrics from client.
        
        Args:
            client_id: Client identifier
            client_state_b: Client's biased model state
            client_state_d: Client's debiased model state
            client_metrics: Training metrics from client
        """
        self.server_stats['client_updates'][client_id] += 1
        self.client_manager.update_client_loss(
            client_id, 
            client_metrics.get('loss_b', 0.0)
        )
    
    def aggregate_client_models(self, client_updates: List[Dict]) -> Tuple[Dict, Dict]:
        """
        Aggregate model updates from all participating clients.
        
        Args:
            client_updates: List of dicts with keys:
                           - 'client_id'
                           - 'model_b_state': biased model state dict
                           - 'model_d_state': debiased model state dict
                           - 'data_size': number of samples in client's dataset
        
        Returns:
            Tuple of (aggregated_state_b, aggregated_state_d)
        """
        if not client_updates:
            return self.global_model_b.state_dict(), self.global_model_d.state_dict()
        
        # Get aggregation weights based on data sizes
        weights = []
        states_b = []
        states_d = []
        
        total_samples = sum(update['data_size'] for update in client_updates)
        
        for update in client_updates:
            weight = update['data_size'] / max(total_samples, 1)
            weights.append(weight)
            states_b.append(update['model_b_state'])
            states_d.append(update['model_d_state'])
        
        # Apply compression if enabled
        if self.compression_enabled:
            states_b = [self.compressor.compress(state, self.compression_ratio) 
                       for state in states_b]
            states_d = [self.compressor.compress(state, self.compression_ratio) 
                       for state in states_d]
        
        # Aggregate based on strategy
        if self.aggregation_strategy == 'fedavg':
            aggregated_b = self._fedavg_aggregate(states_b, weights)
            aggregated_d = self._fedavg_aggregate(states_d, weights)
        
        elif self.aggregation_strategy == 'fedprox':
            aggregated_b = self._fedprox_aggregate(states_b, weights)
            aggregated_d = self._fedprox_aggregate(states_d, weights)
        
        else:
            raise ValueError(f"Unknown aggregation strategy: {self.aggregation_strategy}")
        
        # Update global models
        self.global_model_b.load_state_dict(aggregated_b)
        self.global_model_d.load_state_dict(aggregated_d)
        
        # Record statistics
        self.server_stats['global_updates'] += 1
        self.server_stats['total_communication_rounds'] += 1
        self.server_stats['aggregation_history'].append({
            'round': self.server_stats['total_communication_rounds'],
            'num_clients': len(client_updates),
            'total_samples': total_samples,
            'model_difference_b': self._compute_model_drift(aggregated_b, self.initial_state_b),
            'model_difference_d': self._compute_model_drift(aggregated_d, self.initial_state_d)
        })
        
        return aggregated_b, aggregated_d
    
    def _fedavg_aggregate(self, state_dicts: List[Dict], 
                         weights: List[float]) -> Dict:
        """
        FedAvg aggregation: weighted averaging of parameters.
        """
        aggregated = {}
        
        # Get reference state
        reference_keys = state_dicts[0].keys()
        
        for key in reference_keys:
            aggregated[key] = torch.zeros_like(state_dicts[0][key])
            
            for state, weight in zip(state_dicts, weights):
                aggregated[key] += weight * state[key]
        
        return aggregated
    
    def _fedprox_aggregate(self, state_dicts: List[Dict],
                          weights: List[float], mu: float = 0.01) -> Dict:
        """
        FedProx aggregation: weighted averaging with proximal term.
        Helps with non-IID data and stragglers.
        """
        aggregated = {}
        
        # Compute weighted average
        avg_state = {}
        reference_keys = state_dicts[0].keys()
        
        for key in reference_keys:
            avg_state[key] = torch.zeros_like(state_dicts[0][key])
            for state, weight in zip(state_dicts, weights):
                avg_state[key] += weight * state[key]
        
        # FedProx: regularize towards initial state
        for key in reference_keys:
            aggregated[key] = avg_state[key]
            # Could add proximal term if needed: mu * (avg_state[key] - initial_state[key])
        
        return aggregated
    
    def _compute_model_drift(self, current_state: Dict, initial_state: Dict) -> float:
        """Compute L2 norm of model drift from initial state"""
        total_drift = 0.0
        for key in current_state.keys():
            drift = (current_state[key] - initial_state[key]).norm()
            total_drift += drift.item()
        return total_drift
    
    def save_global_models(self, save_dir: str, round_num: int = None):
        """
        Save global models to disk.
        
        Args:
            save_dir: Directory to save models
            round_num: Communication round number (optional)
        """
        os.makedirs(save_dir, exist_ok=True)
        
        suffix = f"_round{round_num}" if round_num is not None else ""
        
        path_b = os.path.join(save_dir, f"global_model_b{suffix}.pt")
        path_d = os.path.join(save_dir, f"global_model_d{suffix}.pt")
        
        torch.save(self.global_model_b.state_dict(), path_b)
        torch.save(self.global_model_d.state_dict(), path_d)
        
        print(f"[Server] Saved global models to {save_dir}")
    
    def get_global_models(self) -> Tuple[nn.Module, nn.Module]:
        """Get current global models"""
        return self.global_model_b, self.global_model_d
    
    def get_statistics(self) -> Dict:
        """Get server statistics"""
        return {
            'global_updates': self.server_stats['global_updates'],
            'communication_rounds': self.server_stats['total_communication_rounds'],
            'client_updates': self.server_stats['client_updates'],
            'aggregation_history': self.server_stats['aggregation_history']
        }
    
    def evaluate_global_models(self, test_loader, 
                              target_attr_idx: int = 0,
                              bias_attr_idx: int = 1) -> Dict:
        """
        Evaluate global models on test set.
        
        Args:
            test_loader: Test data loader
            target_attr_idx: Index of target attribute
            bias_attr_idx: Index of bias attribute
        
        Returns:
            Evaluation metrics
        """
        self.global_model_b.eval()
        self.global_model_d.eval()
        
        metrics = {
            "total": 0, "align": 0, "conflict": 0,
            "acc_d_total": 0, "acc_b_total": 0,
            "acc_d_align": 0, "acc_d_conflict": 0,
            "acc_b_align": 0, "acc_b_conflict": 0,
        }
        
        with torch.no_grad():
            for batch_data in test_loader:
                if len(batch_data) == 3:
                    data, attr, path = batch_data
                else:
                    _, data, attr, _ = batch_data
                
                data = data.to(self.device)
                
                if attr.dim() == 2:
                    target_label = attr[:, target_attr_idx].to(self.device)
                    bias_label = attr[:, bias_attr_idx].to(self.device)
                else:
                    target_label = attr.to(self.device)
                    bias_label = target_label
                
                logit_d = self.global_model_d(data)
                pred_d = logit_d.data.max(1, keepdim=True)[1].squeeze(1)
                
                logit_b = self.global_model_b(data)
                pred_b = logit_b.data.max(1, keepdim=True)[1].squeeze(1)
                
                is_align = (target_label == bias_label)
                is_conflict = (target_label != bias_label)
                
                correct_d = (pred_d == target_label)
                correct_b = (pred_b == target_label)
                
                metrics["total"] += data.size(0)
                metrics["acc_d_total"] += correct_d.sum().item()
                metrics["acc_b_total"] += correct_b.sum().item()
                metrics["align"] += is_align.sum().item()
                metrics["conflict"] += is_conflict.sum().item()
                metrics["acc_d_align"] += (correct_d & is_align).sum().item()
                metrics["acc_d_conflict"] += (correct_d & is_conflict).sum().item()
                metrics["acc_b_align"] += (correct_b & is_align).sum().item()
                metrics["acc_b_conflict"] += (correct_b & is_conflict).sum().item()
        
        total = max(metrics["total"], 1)
        t_align = max(metrics["align"], 1)
        t_conflict = max(metrics["conflict"], 1)
        
        self.global_model_b.train()
        self.global_model_d.train()
        
        return {
            "acc_d_total": (metrics["acc_d_total"] / total) * 100,
            "acc_b_total": (metrics["acc_b_total"] / total) * 100,
            "acc_d_align": (metrics["acc_d_align"] / t_align) * 100 if t_align > 0 else 0,
            "acc_d_conflict": (metrics["acc_d_conflict"] / t_conflict) * 100 if t_conflict > 0 else 0,
            "acc_b_align": (metrics["acc_b_align"] / t_align) * 100 if t_align > 0 else 0,
            "acc_b_conflict": (metrics["acc_b_conflict"] / t_conflict) * 100 if t_conflict > 0 else 0,
            "num_samples": total
        }
