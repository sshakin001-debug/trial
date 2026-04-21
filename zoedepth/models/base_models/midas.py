"""
MiDaS-based decoder for depth estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MidasDecoder(nn.Module):
    """
    MiDaS-style decoder for depth estimation.
    """
    
    def __init__(self, features, num_channels=1):
        super().__init__()
        self.num_channels = num_channels
        
        self.refinenet = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(features // (4 ** i), features // (4 ** (i + 1)), 3, padding=1),
                nn.BatchNorm2d(features // (4 ** (i + 1))),
                nn.ReLU(inplace=True)
            )
            for i in range(4)
        ])
        
        self.scratch = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(features // 16, features // 16, 3, padding=1),
                nn.BatchNorm2d(features // 16),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(features // 8, num_channels, 1),
            )
        ])
    
    def forward(self, features):
        """Forward pass through decoder."""
        x = features[-1]
        
        for i, layer in enumerate(self.refinenet):
            if i > 0:
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
                skip = F.interpolate(features[-(i + 1)], size=x.shape[2:], mode='bilinear', align_corners=False)
                x = x + skip
            x = layer(x)
        
        x = self.scratch[0](x)
        depth = self.scratch[1](x)
        
        return depth