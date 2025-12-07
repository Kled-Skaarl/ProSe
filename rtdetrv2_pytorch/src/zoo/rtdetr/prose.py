"""Copyright(c) 2024. ProSe: Decoupling Knowledge via Prototype-based Selection for Data-Incremental Object Detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Optional, Tuple
from collections import OrderedDict


class GumbelSoftmax(nn.Module):
    """Gumbel-Softmax for hard attention assignment"""
    
    def __init__(self, tau: float = 1.0, hard: bool = True):
        super().__init__()
        self.tau = tau
        self.hard = hard
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, K] or [B, N, K] attention logits
        Returns:
            weights: [B, K] or [B, N, K] hard attention weights
        """
        if self.training:
            # Gumbel-Softmax: add Gumbel noise for hard assignment
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
            y = logits + gumbel_noise
            y_soft = F.softmax(y / self.tau, dim=-1)
            
            if self.hard:
                # Straight-through estimator: hard assignment in forward, soft in backward
                index = y_soft.argmax(dim=-1)
                y_hard = torch.zeros_like(y_soft).scatter_(-1, index.unsqueeze(-1), 1.0)
                y = y_hard - y_soft.detach() + y_soft
            else:
                y = y_soft
        else:
            # Inference: use hard assignment
            y = F.softmax(logits, dim=-1)
            index = y.argmax(dim=-1)
            y = torch.zeros_like(y).scatter_(-1, index.unsqueeze(-1), 1.0)
        
        return y


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross attention for prototype interaction"""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [B, N_q, C] - prototype queries
            key: [B, N_kv, C] - feature keys
            value: [B, N_kv, C] - feature values
            attn_mask: optional attention mask
        
        Returns:
            output: [B, N_q, C] - attended features
            attn_weights: [B, num_heads, N_q, N_kv] - attention weights
        """
        B, N_q, C = query.shape
        _, N_kv, _ = key.shape
        
        # Project to multi-head
        q = self.q_proj(query).reshape(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N_q, D]
        k = self.k_proj(key).reshape(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)    # [B, H, N_kv, D]
        v = self.v_proj(value).reshape(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N_kv, D]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, N_q, N_kv]
        
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)  # [B, H, N_q, D]
        output = output.transpose(1, 2).reshape(B, N_q, C)  # [B, N_q, C]
        output = self.out_proj(output)
        
        return output, attn_weights


class PrototypeLearningModule(nn.Module):
    """Prototype Learning Module for training phase
    
    Learns a set of prototypes that capture semantic distribution of current increment.
    Inserted between encoder and decoder during training.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_prototypes: int = 1200,
        num_heads: int = 8,
        dropout: float = 0.0,
        alpha: float = 1.0,
        use_gumbel_softmax: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_prototypes = num_prototypes
        self.alpha = alpha
        self.use_gumbel_softmax = use_gumbel_softmax
        
        # Learnable prototypes [K, C]
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, hidden_dim))
        nn.init.xavier_uniform_(self.prototypes)
        
        # Multi-head cross attention
        self.mhca = MultiHeadCrossAttention(hidden_dim, num_heads, dropout)
        
        # Gumbel-Softmax for hard assignment
        if use_gumbel_softmax:
            self.gumbel_softmax = GumbelSoftmax(tau=1.0, hard=True)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, L, C] - encoder output features (flattened multi-scale)
        
        Returns:
            enhanced_features: [B, L, C] - features enhanced with prototype knowledge
        """
        B, L, C = features.shape
        
        # Compute attention between prototypes and features
        # prototypes as query, features as key/value
        proto_attended, attn_weights = self.mhca(
            self.prototypes.unsqueeze(0).expand(B, -1, -1),  # [B, K, C]
            features,  # [B, L, C]
            features   # [B, L, C]
        )  # proto_attended: [B, K, C], attn_weights: [B, H, K, L]
        
        # Compute prototype assignment weights using Gumbel-Softmax
        # Average attention across heads: [B, K, L]
        attn_weights_avg = attn_weights.mean(dim=1)  # [B, K, L]
        
        # Get assignment logits: [B, K]
        assignment_logits = attn_weights_avg.mean(dim=-1)  # [B, K]
        
        if self.use_gumbel_softmax:
            assignment_weights = self.gumbel_softmax(assignment_logits)  # [B, K]
        else:
            assignment_weights = F.softmax(assignment_logits, dim=-1)  # [B, K]
        
        # Weighted combination of prototype knowledge
        # [B, K, C] @ [B, K, 1] -> [B, C]
        proto_knowledge = (proto_attended * assignment_weights.unsqueeze(-1)).sum(dim=1)  # [B, C]
        
        # Residual injection: enhance features with prototype knowledge
        # Broadcast proto_knowledge to all positions and add residually
        enhanced_features = features + self.alpha * proto_knowledge.unsqueeze(1)  # [B, L, C]
        
        return enhanced_features


class AttentionSelector(nn.Module):
    """Attention-based Selector for inference phase
    
    Routes input to the most suitable branch based on:
    1. Prototype similarity (JS divergence)
    2. Semantic response consistency (JS divergence)
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.0,
        lambda_weight: float = 0.5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.lambda_weight = lambda_weight
        
        # Multi-head cross attention for scoring
        self.mhca = MultiHeadCrossAttention(hidden_dim, num_heads, dropout)
    
    def _js_divergence(self, p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Compute Jensen-Shannon divergence between two distributions
        
        Args:
            p: [B, N] - probability distribution
            q: [B, N] - probability distribution
        
        Returns:
            js_div: [B] - JS divergence values
        """
        # Normalize to ensure valid probability distributions
        p = F.softmax(p, dim=-1)
        q = F.softmax(q, dim=-1)
        
        # JS divergence = 0.5 * KL(p || m) + 0.5 * KL(q || m)
        # where m = 0.5 * (p + q)
        m = 0.5 * (p + q)
        
        kl_pm = F.kl_div(torch.log(m + eps), p, reduction='batchmean')
        kl_qm = F.kl_div(torch.log(m + eps), q, reduction='batchmean')
        
        js_div = 0.5 * kl_pm + 0.5 * kl_qm
        
        return js_div
    
    def compute_prototype_similarity(
        self,
        query: torch.Tensor,
        prototypes: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute prototype similarity score (S_proto)
        
        Args:
            query: [B, N_q, C] - decoder queries
            prototypes: [K, C] - prototypes
            features: [B, L, C] - multi-scale features
        
        Returns:
            proto_score: [B] - prototype similarity score per batch
        """
        B = query.shape[0]
        
        # Attention from query to prototypes
        query_to_proto, _ = self.mhca(
            query,  # [B, N_q, C]
            prototypes.unsqueeze(0).expand(B, -1, -1),  # [B, K, C]
            prototypes.unsqueeze(0).expand(B, -1, -1)   # [B, K, C]
        )  # [B, N_q, C]
        
        # Attention from query to features
        query_to_feat, _ = self.mhca(
            query,  # [B, N_q, C]
            features,  # [B, L, C]
            features   # [B, L, C]
        )  # [B, N_q, C]
        
        # Compute JS divergence between the two attention outputs
        # Flatten to [B, N_q * C] for divergence computation
        proto_sim = self._js_divergence(
            query_to_proto.reshape(B, -1),
            query_to_feat.reshape(B, -1)
        )
        
        return proto_sim
    
    def compute_response_consistency(
        self,
        query: torch.Tensor,
        prototypes: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute semantic response consistency (S_resp)
        
        Args:
            query: [B, N_q, C] - decoder queries
            prototypes: [K, C] - prototypes
            features: [B, L, C] - multi-scale features
        
        Returns:
            resp_score: [B] - response consistency score per batch
        """
        B = query.shape[0]
        
        # Response from query
        query_response, _ = self.mhca(
            query,  # [B, N_q, C]
            features,  # [B, L, C]
            features   # [B, L, C]
        )  # [B, N_q, C]
        
        # Response from prototypes
        proto_response, _ = self.mhca(
            prototypes.unsqueeze(0).expand(B, -1, -1),  # [B, K, C]
            features,  # [B, L, C]
            features   # [B, L, C]
        )  # [B, K, C]
        
        # Compute JS divergence between responses
        resp_consistency = self._js_divergence(
            query_response.reshape(B, -1),
            proto_response.reshape(B, -1)
        )
        
        return resp_consistency
    
    def forward(
        self,
        queries_list: List[torch.Tensor],
        prototypes_list: List[torch.Tensor],
        features_list: List[torch.Tensor],
    ) -> int:
        """Select the best branch based on prototype and response scores
        
        Args:
            queries_list: List of [B, N_q, C] - queries from each branch
            prototypes_list: List of [K, C] - prototypes from each branch
            features_list: List of [B, L, C] - features from each branch
        
        Returns:
            selected_branch: int - index of selected branch
        """
        num_branches = len(queries_list)
        scores = []
        
        for t in range(num_branches):
            # Compute prototype similarity
            proto_sim = self.compute_prototype_similarity(
                queries_list[t],
                prototypes_list[t],
                features_list[t]
            )
            
            # Compute response consistency
            resp_cons = self.compute_response_consistency(
                queries_list[t],
                prototypes_list[t],
                features_list[t]
            )
            
            # Combine scores: S = λ * S_proto + (1-λ) * S_resp
            combined_score = self.lambda_weight * proto_sim + (1 - self.lambda_weight) * resp_cons
            scores.append(combined_score)
        
        # Select branch with minimum score (lower is better)
        scores = torch.stack(scores)  # [num_branches]
        selected_branch = torch.argmin(scores).item()
        
        return selected_branch


class BranchManager(nn.Module):
    """Manages multiple branches for data-incremental learning
    
    Each branch maintains:
    - Independent encoder parameters
    - Independent decoder parameters
    - Prototype pool
    - Query embeddings
    """
    
    def __init__(
        self,
        num_branches: int,
        encoder_factory,
        decoder_factory,
        hidden_dim: int = 256,
        num_prototypes: int = 1200,
        num_queries: int = 300,
    ):
        super().__init__()
        self.num_branches = num_branches
        self.hidden_dim = hidden_dim
        self.num_prototypes = num_prototypes
        self.num_queries = num_queries
        
        # Create branches
        self.encoders = nn.ModuleList([encoder_factory() for _ in range(num_branches)])
        self.decoders = nn.ModuleList([decoder_factory() for _ in range(num_branches)])
        
        # Prototype pools for each branch
        self.prototype_pools = nn.ParameterList([
            nn.Parameter(torch.randn(num_prototypes, hidden_dim))
            for _ in range(num_branches)
        ])
        
        # Initialize prototypes
        for proto_pool in self.prototype_pools:
            nn.init.xavier_uniform_(proto_pool)
        
        # Query embeddings for each branch
        self.query_embeds = nn.ParameterList([
            nn.Parameter(torch.randn(num_queries, hidden_dim))
            for _ in range(num_branches)
        ])
        
        # Initialize query embeddings
        for query_embed in self.query_embeds:
            nn.init.xavier_uniform_(query_embed)
    
    def get_branch_encoder(self, branch_idx: int) -> nn.Module:
        """Get encoder for specific branch"""
        return self.encoders[branch_idx]
    
    def get_branch_decoder(self, branch_idx: int) -> nn.Module:
        """Get decoder for specific branch"""
        return self.decoders[branch_idx]
    
    def get_branch_prototypes(self, branch_idx: int) -> torch.Tensor:
        """Get prototypes for specific branch"""
        return self.prototype_pools[branch_idx]
    
    def get_branch_queries(self, branch_idx: int) -> torch.Tensor:
        """Get query embeddings for specific branch"""
        return self.query_embeds[branch_idx]
    
    def forward(self, x: torch.Tensor, branch_idx: int, targets=None):
        """Forward pass through specific branch
        
        Args:
            x: input features
            branch_idx: which branch to use
            targets: optional training targets
        
        Returns:
            output from decoder
        """
        # Encode
        encoded = self.encoders[branch_idx](x)
        
        # Decode
        output = self.decoders[branch_idx](encoded, targets)
        
        return output


__all__ = [
    'GumbelSoftmax',
    'MultiHeadCrossAttention',
    'PrototypeLearningModule',
    'AttentionSelector',
    'BranchManager',
]
