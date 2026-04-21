"""
Distance transformation layers for depth binning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistanceTransformer(nn.Module):
    """
    Transforms depth predictions using distance-based features.
    """
    
    def __init__(self, in_channels, hidden_dim=128):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        self.depth_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, 1)
        )
    
    def forward(self, features):
        x = self.transform(features)
        depth = self.depth_head(x)
        return depth, x