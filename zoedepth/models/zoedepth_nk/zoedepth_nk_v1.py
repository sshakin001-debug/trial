"""
ZoeDepth NK (NYU Depth v2 + KITTI) variant.
"""

import torch
import torch.nn as nn

from zoedepth.models.depth_model import DepthModel


class ZoeDepthNKV1(DepthModel):
    """
    ZoeDepth NK model for indoor/outdoor depth estimation.
    Supports both NYU (indoor) and KITTI (outdoor) datasets.
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        model_config = config.model
        
        self.n_bins = model_config.get('n_bins', 64)
        self.bin_embedding_dim = model_config.get('bin_embedding_dim', 128)
        
        self.min_depth = model_config.get('min_temp', 0.1)
        self.max_depth = model_config.get('max_temp', 10.0)
        
        self.encoder = nn.Identity()
        self.decoder = nn.Sequential(
            nn.Conv2d(1024, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1)
        )
    
    def forward(self, x, dataset=None, **kwargs):
        features = self.encoder(x)
        depth = self.decoder(features)
        return depth.squeeze(1)
    
    def infer(self, x, dataset=None, **kwargs):
        with torch.no_grad():
            return self.forward(x, dataset, **kwargs)