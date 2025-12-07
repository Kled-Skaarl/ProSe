"""Copyright(c) 2024. ProSe-RTDETRv2: Integration of ProSe with RT-DETRv2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
import copy

from ...core import register
from .prose import (
    PrototypeLearningModule,
    AttentionSelector,
    MultiHeadCrossAttention,
)


__all__ = ['ProSeRTDETRv2']


@register()
class ProSeRTDETRv2(nn.Module):
    """ProSe-RTDETRv2: Data-Incremental Object Detection with Prototype-based Selection
    
    Combines RT-DETRv2 with ProSe for incremental learning:
    - Training: Uses PrototypeLearningModule to learn prototypes for current increment
    - Inference: Uses AttentionSelector to route to best branch based on prototype similarity
    """
    
    __inject__ = ['backbone', 'encoder', 'decoder']
    
    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        hidden_dim: int = 256,
        num_prototypes: int = 1200,
        num_heads: int = 8,
        alpha: float = 1.0,
        lambda_weight: float = 0.5,
        use_gumbel_softmax: bool = True,
        use_prose: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        
        self.hidden_dim = hidden_dim
        self.num_prototypes = num_prototypes
        self.use_prose = use_prose
        
        if use_prose:
            # Prototype learning module (training only)
            self.prototype_learning = PrototypeLearningModule(
                hidden_dim=hidden_dim,
                num_prototypes=num_prototypes,
                num_heads=num_heads,
                alpha=alpha,
                use_gumbel_softmax=use_gumbel_softmax,
            )
            
            # Attention selector (inference only)
            self.attention_selector = AttentionSelector(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                lambda_weight=lambda_weight,
            )
            
            # Multi-branch support (for incremental learning)
            self.num_branches = 1
            self.branch_encoders = nn.ModuleList([encoder])
            self.branch_decoders = nn.ModuleList([decoder])
            self.branch_prototypes = nn.ParameterList([
                nn.Parameter(torch.randn(num_prototypes, hidden_dim))
            ])
            nn.init.xavier_uniform_(self.branch_prototypes[0])
            
            # Current active branch
            self.active_branch = 0
    
    def add_branch(self, encoder: nn.Module, decoder: nn.Module):
        """Add a new branch for new increment
        
        Args:
            encoder: encoder module for new branch
            decoder: decoder module for new branch
        """
        if not self.use_prose:
            raise RuntimeError("Cannot add branch when use_prose=False")
        
        # Copy encoder and decoder from previous branch as initialization
        self.branch_encoders.append(copy.deepcopy(self.branch_encoders[-1]))
        self.branch_decoders.append(copy.deepcopy(self.branch_decoders[-1]))
        
        # Initialize new prototype pool
        new_prototypes = nn.Parameter(torch.randn(self.num_prototypes, self.hidden_dim))
        nn.init.xavier_uniform_(new_prototypes)
        self.branch_prototypes.append(new_prototypes)
        
        self.num_branches += 1
        self.active_branch = self.num_branches - 1
    
    def set_active_branch(self, branch_idx: int):
        """Set which branch to use for training
        
        Args:
            branch_idx: index of branch to activate
        """
        if not self.use_prose:
            raise RuntimeError("Cannot set branch when use_prose=False")
        
        if branch_idx >= self.num_branches:
            raise ValueError(f"Branch {branch_idx} does not exist (total: {self.num_branches})")
        
        self.active_branch = branch_idx
    
    def forward(self, x, targets=None, return_branch_idx=False):
        """Forward pass
        
        Args:
            x: input images
            targets: training targets (optional)
            return_branch_idx: whether to return selected branch index (inference only)
        
        Returns:
            output: detection results
            branch_idx: selected branch index (only if return_branch_idx=True and not training)
        """
        # Backbone
        x = self.backbone(x)
        
        if self.training:
            # Training: use active branch with prototype learning
            encoder = self.branch_encoders[self.active_branch]
            decoder = self.branch_decoders[self.active_branch]
            
            # Encode
            encoded_features = encoder(x)
            
            # Apply prototype learning module
            enhanced_features = self.prototype_learning(encoded_features)
            
            # Decode
            output = decoder(enhanced_features, targets)
            
            return output
        
        else:
            # Inference: use attention selector to choose branch
            if self.use_prose and self.num_branches > 1:
                # Parallel encoding with all branches
                encoded_list = []
                for encoder in self.branch_encoders:
                    encoded_list.append(encoder(x))
                
                # Prepare inputs for selector
                # For simplicity, use first decoder's query embeddings
                # In practice, each branch should have its own queries
                queries_list = [
                    self.branch_decoders[i].query_pos_head.weight.unsqueeze(0).expand(x.shape[0], -1, -1)
                    for i in range(self.num_branches)
                ]
                
                prototypes_list = [
                    self.branch_prototypes[i].unsqueeze(0).expand(x.shape[0], -1, -1)
                    for i in range(self.num_branches)
                ]
                
                features_list = encoded_list
                
                # Select best branch
                selected_branch = self.attention_selector(
                    queries_list,
                    prototypes_list,
                    features_list,
                )
            else:
                # Single branch or ProSe disabled
                selected_branch = self.active_branch
            
            # Use selected branch for decoding
            encoder = self.branch_encoders[selected_branch]
            decoder = self.branch_decoders[selected_branch]
            
            encoded_features = encoder(x)
            output = decoder(encoded_features, targets)
            
            if return_branch_idx:
                return output, selected_branch
            else:
                return output
    
    def deploy(self):
        """Prepare model for deployment"""
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self


__all__ = ['ProSeRTDETRv2']
