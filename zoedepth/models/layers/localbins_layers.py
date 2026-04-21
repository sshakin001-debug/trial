"""
Local bins layers for depth distribution estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalBinsLayer(nn.Module):
    """
    Local bins layer for adaptive depth binning.
    """
    
    def __init__(self, in_channels, n_bins=64, min_depth=0.1, max_depth=10.0):
        super().__init__()
        self.n_bins = n_bins
        self.min_depth = min_depth
        self.max_depth = max_depth
        
        self.log_bin_centers = nn.Parameter(
            torch.linspace(torch.log(torch.tensor(min_depth)), 
                         torch.log(torch.tensor(max_depth)), 
                         n_bins),
            requires_grad=False
        )
        
        self.bin_weights = nn.Sequential(
            nn.Conv2d(in_channels, n_bins, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, features, depth_prior):
        bin_centers = torch.exp(self.log_bin_centers).view(1, -1, 1, 1)
        
        weights = self.bin_weights(features)
        
        depth = (weights * bin_centers).sum(dim=1, keepdim=True)
        
        return depth, bin_centers.squeeze(0), weights


class LocalBinsDecoder(nn.Module):
    """
    Decoder with local bins for depth estimation.
    """
    
    def __init__(self, in_channels, n_bins=64, min_depth=0.1, max_depth=10.0):
        super().__init__()
        self.bins_layer = LocalBinsLayer(in_channels, n_bins, min_depth, max_depth)
        
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels + 1, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, 1)
        )
    
    def forward(self, features, depth_prior):
        depth, bin_centers, weights = self.bins_layer(features, depth_prior)
        
        refined = self.refine(torch.cat([features, depth], dim=1))
        
        return refined, bin_centers, weights