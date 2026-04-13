"""Utility functions for Federated Learning"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import numpy as np


def aggregate_models(models: List[Dict], weights: List[float] = None) -> Dict:
    """
    Aggregate model parameters from multiple clients using weighted averaging.
    
    Args:
        models: List of state_dicts from client models
        weights: Weights for each model (e.g., based on data size)
                If None, uniform averaging is used.
    
    Returns:
        Aggregated state_dict
    """
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    
    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    aggregated_state = {}
    
    # Get reference model to initialize aggregated state
    for key in models[0].keys():
        aggregated_state[key] = torch.zeros_like(models[0][key])
    
    # Weighted average
    for model, weight in zip(models, weights):
        for key in model.keys():
            aggregated_state[key] += weight * model[key]
    
    return aggregated_state


def calculate_client_weight(data_size: int, total_size: int) -> float:
    """Calculate weight for a client based on data size"""
    return data_size / total_size


def compute_model_difference(model1_state: Dict, model2_state: Dict) -> float:
    """
    Compute L2 norm of parameter differences between two models.
    """
    total_diff = 0.0
    for key in model1_state.keys():
        diff = (model1_state[key] - model2_state[key]).norm()
        total_diff += diff.item()
    return total_diff


def apply_differential_privacy(state_dict: Dict, noise_scale: float = 0.01) -> Dict:
    """
    Apply Gaussian noise for differential privacy to model parameters.
    """
    dp_state = {}
    for key, param in state_dict.items():
        noise = torch.randn_like(param) * noise_scale
        dp_state[key] = param + noise
    return dp_state


class ClientManager:
    """Manages multiple federated clients"""
    
    def __init__(self, num_clients: int):
        self.num_clients = num_clients
        self.client_data_sizes = {}
        self.client_losses = {}
        
    def register_client(self, client_id: int, data_size: int):
        """Register a client with its data size"""
        self.client_data_sizes[client_id] = data_size
        
    def get_aggregation_weights(self) -> Dict[int, float]:
        """Get aggregation weights based on data sizes"""
        total_size = sum(self.client_data_sizes.values())
        weights = {}
        for client_id, size in self.client_data_sizes.items():
            weights[client_id] = calculate_client_weight(size, total_size)
        return weights
    
    def update_client_loss(self, client_id: int, loss: float):
        """Track client's loss for monitoring"""
        self.client_losses[client_id] = loss


class CommunicationCompressor:
    """Compresses model updates for efficient communication"""
    
    @staticmethod
    def compress(state_dict: Dict, top_k: float = 0.1) -> Dict:
        """
        Keep only top-k% of parameters by magnitude.
        
        Args:
            state_dict: Model state dictionary
            top_k: Fraction of parameters to keep (0-1)
        
        Returns:
            Compressed state dictionary
        """
        compressed = {}
        
        for key, param in state_dict.items():
            if param.dtype in [torch.float32, torch.float64]:
                # Flatten and compute magnitude
                flat_param = param.flatten()
                magnitude = torch.abs(flat_param)
                
                # Calculate threshold
                k = max(1, int(len(flat_param) * top_k))
                threshold = torch.kthvalue(magnitude, len(flat_param) - k + 1)[0]
                
                # Create mask
                mask = magnitude >= threshold
                
                # Apply mask
                compressed_param = param.clone()
                compressed_param[~mask.view_as(param)] = 0
                compressed[key] = compressed_param
            else:
                compressed[key] = param.clone()
        
        return compressed


def get_client_data_distribution(train_dataset, num_clients: int, 
                                 distribution: str = 'iid') -> List[List[int]]:
    """
    Split dataset indices among clients.
    
    Args:
        train_dataset: Training dataset
        num_clients: Number of clients
        distribution: 'iid' for non-biased split, 'non-iid' for class-skewed
    
    Returns:
        List of client data indices
    """
    data_size = len(train_dataset)
    indices = list(range(data_size))
    
    if distribution == 'iid':
        # Shuffle and split equally
        np.random.shuffle(indices)
        client_indices = [indices[i::num_clients] for i in range(num_clients)]
    
    else:  # non-iid
        # Assign labels to clients to create class imbalance
        labels = []
        for idx in indices:
            try:
                # Try to get label from dataset
                _, attr, _ = train_dataset[idx]
                labels.append(attr[0].item())
            except:
                labels.append(0)
        
        # Sort by label
        sorted_indices = [idx for _, idx in sorted(zip(labels, indices))]
        client_indices = [sorted_indices[i::num_clients] for i in range(num_clients)]
    
    return client_indices
