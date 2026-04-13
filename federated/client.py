"""Federated Client for Local Training with Bias Mitigation"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
from typing import Dict, Tuple, List
import numpy as np
from copy import deepcopy
from local_reweighting import AdaptiveLocalReweighting


class FederatedLFFClient:
    """
    A federated client that performs local LFF training with reweighting.
    Maintains two models (biased and debiased) locally and communicates with server.
    """
    
    def __init__(self, client_id: int, model_b, model_d, 
                 local_dataset, criterion, bias_criterion,
                 lr: float = 1e-3, device: str = 'cuda',
                 num_classes: int = 10):
        """
        Initialize federated client.
        
        Args:
            client_id: Unique client identifier
            model_b: Biased model instance
            model_d: Debiased model instance
            local_dataset: Local training dataset
            criterion: Standard CE loss
            bias_criterion: GeneralizedCELoss
            lr: Learning rate
            device: Device ('cuda' or 'cpu')
            num_classes: Number of classes
        """
        self.client_id = client_id
        self.model_b = model_b
        self.model_d = model_d
        self.local_dataset = local_dataset
        self.criterion = criterion
        self.bias_criterion = bias_criterion
        self.device = torch.device(device) if isinstance(device, str) else device
        self.lr = lr
        self.num_classes = num_classes
        
        # Optimizers
        self.optimizer_b = optim.Adam(self.model_b.parameters(), lr=lr)
        self.optimizer_d = optim.Adam(self.model_d.parameters(), lr=lr)
        
        # Setup reweighting manager
        self.reweighting = AdaptiveLocalReweighting(num_classes=num_classes)
        self.reweighting.initialize(len(self.local_dataset))
        
        # Training statistics
        self.local_steps = 0
        self.global_steps = 0
        self.training_history = {
            'loss_b': [],
            'loss_d': [],
            'local_avg_loss': []
        }
        
    def load_model_state(self, state_dict_b: Dict, state_dict_d: Dict):
        """Load global model state from server"""
        self.model_b.load_state_dict(state_dict_b)
        self.model_d.load_state_dict(state_dict_d)
    
    def get_model_state(self) -> Tuple[Dict, Dict]:
        """Get current model state for sending to server"""
        return self.model_b.state_dict(), self.model_d.state_dict()
    
    def local_train(self, num_epochs: int = 1, batch_size: int = 32, 
                   verbose: bool = False) -> Dict:
        """
        Perform local training for specified number of epochs.
        
        Args:
            num_epochs: Number of local training epochs
            batch_size: Batch size for local training
            verbose: Whether to print training info
        
        Returns:
            Dictionary with training metrics
        """
        self.model_b.to(self.device)
        self.model_d.to(self.device)
        
        # Create data loader
        train_loader = DataLoader(
            self.local_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )
        
        total_loss_b = 0.0
        total_loss_d = 0.0
        total_samples = 0
        
        for epoch in range(num_epochs):
            self.model_b.train()
            self.model_d.train()
            
            for batch_idx, batch_data in enumerate(train_loader):
                # Handle different data formats
                if len(batch_data) == 3:
                    data, attr, path = batch_data
                elif len(batch_data) == 4:
                    index, data, attr, _ = batch_data
                else:
                    continue
                
                data = data.to(self.device)
                attr = attr.to(self.device)
                
                # Get labels
                if attr.dim() == 2:
                    label = attr[:, 0]
                else:
                    label = attr
                
                label = label.to(self.device)
                
                # Forward pass
                logit_b = self.model_b(data)
                logit_d = self.model_d(data)
                
                # Compute raw losses
                loss_b_raw = self.criterion(logit_b, label).detach()
                loss_d_raw = self.criterion(logit_d, label).detach()
                
                # Get sample indices for reweighting
                try:
                    batch_indices = index if len(batch_data) == 4 else torch.arange(
                        batch_idx * batch_size, 
                        min((batch_idx + 1) * batch_size, len(self.local_dataset))
                    )
                except:
                    batch_indices = torch.arange(len(data))
                
                batch_indices = batch_indices.to(self.device)
                
                # Compute adaptive weights
                progress_ratio = self.global_steps / max(1, self.global_steps + 1000)  # Rough estimate
                weight_dict = self.reweighting.compute_adaptive_weights(
                    batch_indices,
                    logit_b=logit_b,
                    logits_biased=logit_b,
                    logits_debiased=logit_d,
                    losses_biased=loss_b_raw,
                    losses_debiased=loss_d_raw,
                    labels=label,
                    progress_ratio=progress_ratio
                )
                
                weights = weight_dict['weights'].to(self.device)
                
                # Weighted loss
                loss_b_update = self.bias_criterion(logit_b, label)
                loss_d_update = self.criterion(logit_d, label) * weights
                
                loss = loss_b_update.mean() + loss_d_update.mean()
                
                # Backward pass
                self.optimizer_b.zero_grad()
                self.optimizer_d.zero_grad()
                loss.backward()
                self.optimizer_b.step()
                self.optimizer_d.step()
                
                # Track losses
                total_loss_b += loss_b_raw.mean().item() * data.size(0)
                total_loss_d += loss_d_raw.mean().item() * data.size(0)
                total_samples += data.size(0)
                
                self.local_steps += 1
        
        avg_loss_b = total_loss_b / max(1, total_samples)
        avg_loss_d = total_loss_d / max(1, total_samples)
        
        # Store history
        self.training_history['loss_b'].append(avg_loss_b)
        self.training_history['loss_d'].append(avg_loss_d)
        self.training_history['local_avg_loss'].append((avg_loss_b + avg_loss_d) / 2)
        
        metrics = {
            'client_id': self.client_id,
            'loss_b': avg_loss_b,
            'loss_d': avg_loss_d,
            'local_samples': len(self.local_dataset),
            'local_steps': num_epochs * len(train_loader)
        }
        
        if verbose:
            print(f"[Client {self.client_id}] Epoch {epoch+1}/{num_epochs} - "
                  f"Loss_B: {avg_loss_b:.4f}, Loss_D: {avg_loss_d:.4f}")
        
        return metrics
    
    def validate(self, val_loader: DataLoader, 
                target_attr_idx: int = 0, bias_attr_idx: int = 1) -> Dict:
        """
        Validate on local validation set.
        
        Args:
            val_loader: Validation data loader
            target_attr_idx: Index of target attribute
            bias_attr_idx: Index of bias attribute
        
        Returns:
            Dictionary with validation metrics
        """
        self.model_b.eval()
        self.model_d.eval()
        
        metrics = {
            "total": 0, "align": 0, "conflict": 0,
            "acc_d_total": 0, "acc_b_total": 0,
            "acc_d_align": 0, "acc_d_conflict": 0,
            "acc_b_align": 0, "acc_b_conflict": 0,
        }
        
        with torch.no_grad():
            for batch_data in val_loader:
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
                
                logit_d = self.model_d(data)
                pred_d = logit_d.data.max(1, keepdim=True)[1].squeeze(1)
                
                logit_b = self.model_b(data)
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
        
        # Compute percentages
        total = max(metrics["total"], 1)
        t_align = max(metrics["align"], 1)
        t_conflict = max(metrics["conflict"], 1)
        
        self.model_b.train()
        self.model_d.train()
        
        return {
            "client_id": self.client_id,
            "acc_d_total": (metrics["acc_d_total"] / total) * 100,
            "acc_b_total": (metrics["acc_b_total"] / total) * 100,
            "acc_d_align": (metrics["acc_d_align"] / t_align) * 100 if t_align > 0 else 0,
            "acc_d_conflict": (metrics["acc_d_conflict"] / t_conflict) * 100 if t_conflict > 0 else 0,
            "acc_b_align": (metrics["acc_b_align"] / t_align) * 100 if t_align > 0 else 0,
            "acc_b_conflict": (metrics["acc_b_conflict"] / t_conflict) * 100 if t_conflict > 0 else 0,
            "num_samples": total
        }
    
    def compute_local_model_size(self) -> int:
        """Get total number of parameters in local models"""
        total_params = sum(p.numel() for p in self.model_b.parameters())
        total_params += sum(p.numel() for p in self.model_d.parameters())
        return total_params
