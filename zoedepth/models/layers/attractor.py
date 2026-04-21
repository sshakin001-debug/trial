"""
Attractor layers for depth binning (ZoeDepth core mechanism).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttractorLayer(nn.Module):
    """
    Attractor-based depth binning layer.
    Uses softbinning mechanism from ZoeDepth paper.
    """
    
    def __init__(self, in_channels, n_bins=64, embed_dim=128, 
                 alpha=1000, gamma=2, attractor_type="mean"):
        super().__init__()
        self.n_bins = n_bins
        self.alpha = alpha
        self.gamma = gamma
        self.attractor_type = attractor_type
        
        self.bin_embedding = nn.Embedding(n_bins, embed_dim)
        
        self.depth_refine = nn.Sequential(
            nn.Conv2d(in_channels + 1, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, 1)
        )
    
    def forward(self, features, depth_prior):
        B, C, H, W = features.shape
        
        bin_centers = self.bin_embedding.weight
        
        bin_centers_expanded = bin_centers.view(1, self.n_bins, 1, 1).expand(B, -1, H, W)
        
        depth_prior_expanded = depth_prior.unsqueeze(1)
        
        distances = torch.abs(bin_centers_expanded - depth_prior_expanded)
        
        attention = F.softmax(-self.alpha * distances, dim=1)
        
        weighted_centers = (attention * bin_centers_expanded).sum(dim=1, keepdim=True)
        
        refined_depth = self.depth_refine(torch.cat([features, weighted_centers], dim=1))
        
        return refined_depth, bin_centers[:self.n_bins], attention


class AttractorBlock(nn.Module):
    """
    Multi-scale attractor block.
    """
    
    def __init__(self, in_channels, n_bins_list, embed_dim, alpha=1000, gamma=2):
        super().__init__()
        self.attractors = nn.ModuleList([
            AttractorLayer(in_channels, n_bins, embed_dim, alpha, gamma)
            for n_bins in n_bins_list
        ])
    
    def forward(self, features, depth_prior):
        outputs = []
        for attractor in self.attractors:
            depth, bins, attn = attractor(features, depth_prior)
            outputs.append((depth, bins, attn))
        return outputs