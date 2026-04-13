"""Local Reweighting Methods for Bias Mitigation in Federated Learning"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict
import copy


class LocalReweightingManager:
    """Manages local sample reweighting on client side"""
    
    def __init__(self, reweighting_method: str = 'loss_based', 
                 alpha: float = 0.9, num_classes: int = 10):
        """
        Initialize reweighting manager.
        
        Args:
            reweighting_method: Type of reweighting ('loss_based', 'importance', 'uncertainty')
            alpha: EMA smoothing factor
            num_classes: Number of classes
        """
        self.reweighting_method = reweighting_method
        self.alpha = alpha
        self.num_classes = num_classes
        self.sample_weights = None
        self.loss_ema = None
        
    def initialize(self, dataset_size: int):
        """Initialize reweighting for a dataset"""
        self.sample_weights = torch.ones(dataset_size) / dataset_size
        self.loss_ema = torch.zeros(dataset_size)
        
    def update_weights(self, indices: torch.Tensor, losses: torch.Tensor, 
                      labels: torch.Tensor = None) -> torch.Tensor:
        """
        Update sample weights based on losses.
        
        Args:
            indices: Sample indices
            losses: Sample losses
            labels: Sample labels (optional)
        
        Returns:
            Updated sample weights
        """
        if self.reweighting_method == 'loss_based':
            return self._loss_based_reweighting(indices, losses, labels)
        elif self.reweighting_method == 'importance':
            return self._importance_reweighting(indices, losses, labels)
        elif self.reweighting_method == 'uncertainty':
            return self._uncertainty_reweighting(indices, losses, labels)
        else:
            raise ValueError(f"Unknown reweighting method: {self.reweighting_method}")
    
    def _loss_based_reweighting(self, indices: torch.Tensor, losses: torch.Tensor,
                               labels: torch.Tensor = None) -> torch.Tensor:
        """
        Reweight samples inversely proportional to their loss.
        Samples with higher loss get lower weight.
        """
        # Update EMA of losses
        indices = indices.cpu()
        losses = losses.detach().cpu()
        
        self.loss_ema[indices] = (self.alpha * self.loss_ema[indices] + 
                                   (1 - self.alpha) * losses)
        
        # Get normalized weights (inverse of loss)
        weights = torch.zeros_like(self.sample_weights)
        min_loss = self.loss_ema.min() + 1e-8
        max_loss = self.loss_ema.max() + 1e-8
        
        # Normalize to [0, 1] range
        normalized_loss = (self.loss_ema - min_loss) / (max_loss - min_loss)
        weights = 1.0 - normalized_loss  # Inverse relationship
        
        # Ensure positive weights
        weights = torch.clamp(weights, min=1e-8)
        
        # Normalize to sum to 1
        weights = weights / weights.sum()
        
        self.sample_weights = weights
        return weights[indices]
    
    def _importance_reweighting(self, indices: torch.Tensor, losses: torch.Tensor,
                               labels: torch.Tensor = None) -> torch.Tensor:
        """
        Reweight based on importance: samples contributing more to gradient updates.
        """
        # Use gradient magnitude as importance measure
        losses = losses.detach().cpu()
        
        # Compute importance as gradient magnitude
        importance = torch.abs(losses)
        importance = importance / (importance.sum() + 1e-8)
        
        self.sample_weights[indices] = importance
        return self.sample_weights[indices]
    
    def _uncertainty_reweighting(self, indices: torch.Tensor, losses: torch.Tensor,
                                labels: torch.Tensor = None) -> torch.Tensor:
        """
        Reweight based on model uncertainty. 
        High confidence samples get lower weight if they're already correct.
        """
        losses = losses.detach().cpu()
        
        # Higher loss = higher uncertainty = higher weight
        weights = torch.softmax(losses * 2, dim=0)  # Temperature scaling
        
        self.sample_weights[indices] = weights
        return self.sample_weights[indices]
    
    def get_weights(self, indices: torch.Tensor = None) -> torch.Tensor:
        """Get current sample weights"""
        if indices is None:
            return self.sample_weights
        return self.sample_weights[indices.cpu()]


class ConflictDetectionReweighting:
    """
    Reweight samples based on conflict detection between biased and debiased models.
    This is specific to bias mitigation (LFF style).
    """
    
    def __init__(self, alpha: float = 0.9):
        self.alpha = alpha
        self.conflict_scores = None
        
    def initialize(self, dataset_size: int):
        """Initialize conflict detection"""
        self.conflict_scores = torch.zeros(dataset_size)
    
    def detect_and_reweight(self, indices: torch.Tensor,
                           logits_biased: torch.Tensor,
                           logits_debiased: torch.Tensor,
                           labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect conflicting samples and compute reweighting.
        
        Args:
            indices: Sample indices
            logits_biased: Predictions from biased model
            logits_debiased: Predictions from debiased model
            labels: Ground truth labels
        
        Returns:
            Tuple of (weights, conflict_mask)
        """
        indices = indices.cpu()
        
        # Get predictions
        pred_biased = logits_biased.detach().argmax(dim=1).cpu()
        pred_debiased = logits_debiased.detach().argmax(dim=1).cpu()
        labels = labels.cpu()
        
        # Detect conflicts: when biased and debiased models disagree
        in_conflict = pred_biased != pred_debiased
        
        # Get confidence from both models
        probs_biased = torch.softmax(logits_biased.detach(), dim=1).cpu()
        probs_debiased = torch.softmax(logits_debiased.detach(), dim=1).cpu()
        
        # Confidence in predictions
        conf_biased = probs_biased.max(dim=1)[0]
        conf_debiased = probs_debiased.max(dim=1)[0]
        
        # Compute conflict scores
        conflict_score = torch.zeros(len(indices))
        
        for i, idx in enumerate(indices):
            if in_conflict[i]:
                # In conflict: weight based on confidence of debiased model
                conflict_score[i] = conf_debiased[i]
            else:
                # No conflict: weight based on agreement
                if pred_biased[i] == labels[i]:
                    conflict_score[i] = conf_biased[i] * 0.5  # Lower weight for aligned easy samples
                else:
                    conflict_score[i] = 1.0  # Higher weight for hard samples
        
        # Update EMA of conflict scores
        self.conflict_scores[indices] = (self.alpha * self.conflict_scores[indices] + 
                                         (1 - self.alpha) * conflict_score)
        
        # Normalize weights
        weights = self.conflict_scores / (self.conflict_scores.sum() + 1e-8)
        
        return weights[indices], in_conflict


class AdaptiveLocalReweighting:
    """
    Adaptive reweighting that combines multiple strategies based on global iteration.
    """
    
    def __init__(self, num_classes: int = 10):
        self.num_classes = num_classes
        self.loss_manager = LocalReweightingManager('loss_based', num_classes=num_classes)
        self.conflict_manager = ConflictDetectionReweighting()
        self.current_iteration = 0
        self.local_iterations = 0
        
    def initialize(self, dataset_size: int):
        self.loss_manager.initialize(dataset_size)
        self.conflict_manager.initialize(dataset_size)
    
    def compute_adaptive_weights(self, indices: torch.Tensor, 
                                logits_biased: torch.Tensor,
                                logits_debiased: torch.Tensor,
                                losses_biased: torch.Tensor,
                                losses_debiased: torch.Tensor,
                                labels: torch.Tensor,
                                progress_ratio: float = 0.0) -> Dict[str, torch.Tensor]:
        """
        Compute weights adaptively based on training progress.
        Early training: emphasize conflict detection
        Late training: emphasize loss-based reweighting
        
        Args:
            indices: Sample indices
            logits_biased: Biased model logits
            logits_debiased: Debiased model logits
            losses_biased: Biased model losses
            losses_debiased: Debiased model losses
            labels: Ground truth labels
            progress_ratio: Training progress (0 to 1)
        
        Returns:
            Dictionary with weights and metadata
        """
        # Get loss-based weights
        loss_weights = self.loss_manager.update_weights(indices, losses_debiased, labels)
        
        # Get conflict-based weights
        conflict_weights, conflict_mask = self.conflict_manager.detect_and_reweight(
            indices, logits_biased, logits_debiased, labels
        )
        
        # Adaptive combination: early phase emphasizes conflict, late phase emphasizes loss
        alpha_conflict = max(0.1, 1.0 - progress_ratio)  # Decreases from 1 to 0.1
        alpha_loss = progress_ratio + 0.1  # Increases from 0.1 to 1.0
        
        # Normalize alphas
        total = alpha_conflict + alpha_loss
        alpha_conflict /= total
        alpha_loss /= total
        
        # Combine weights
        combined_weights = (alpha_conflict * conflict_weights + alpha_loss * loss_weights)
        combined_weights = combined_weights / (combined_weights.sum() + 1e-8)
        
        return {
            'weights': combined_weights,
            'conflict_weights': conflict_weights,
            'loss_weights': loss_weights,
            'conflict_mask': conflict_mask,
            'alpha_conflict': alpha_conflict,
            'alpha_loss': alpha_loss
        }


class DataDrivenReweighting:
    """
    Reweighting based on data properties and sample characteristics.
    """
    
    def __init__(self, num_classes: int = 10):
        self.num_classes = num_classes
        self.sample_frequencies = None
        
    def initialize(self, labels: torch.Tensor):
        """Initialize based on label distribution"""
        self.sample_frequencies = torch.bincount(labels, minlength=self.num_classes)
        
    def compute_class_weights(self) -> torch.Tensor:
        """
        Compute inverse class frequencies as weights.
        Underrepresented classes get higher weight.
        """
        freqs = self.sample_frequencies.float()
        freqs = freqs / freqs.sum()
        
        # Inverse frequency weighting
        weights = 1.0 / (freqs + 1e-8)
        weights = weights / weights.sum()
        
        return weights
    
    def apply_class_weights(self, sample_weights: torch.Tensor, 
                           labels: torch.Tensor) -> torch.Tensor:
        """Apply class-specific weights to sample weights"""
        class_weights = self.compute_class_weights()
        
        # Scale sample weights by class weight
        weighted_samples = sample_weights.clone()
        for label in range(self.num_classes):
            mask = labels == label
            weighted_samples[mask] *= class_weights[label]
        
        weighted_samples = weighted_samples / (weighted_samples.sum() + 1e-8)
        
        return weighted_samples
