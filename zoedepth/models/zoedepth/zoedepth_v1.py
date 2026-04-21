"""
ZoeDepth v1 model implementation.
Uses depth binning with attractors for metric depth estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from zoedepth.models.depth_model import DepthModel
from zoedepth.models.layers.attractor import AttractorBlock


class ZoeDepthV1(DepthModel):
    """
    ZoeDepth v1 model with attractor-based depth binning.
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        model_config = config.model
        
        self.n_bins = model_config.get('n_bins', 64)
        self.bin_embedding_dim = model_config.get('bin_embedding_dim', 128)
        self.n_attractors = model_config.get('n_attractors', [16, 8, 4, 1])
        self.alpha = model_config.get('attractor_alpha', 1000)
        self.gamma = model_config.get('attractor_gamma', 2)
        
        self.min_depth = model_config.get('min_temp', 0.0212)
        self.max_depth = model_config.get('max_temp', 50.0)
        
        encoder_dim = 1024
        self.encoder_dim = encoder_dim
        
        self.attractors = AttractorBlock(
            encoder_dim, 
            self.n_attractors, 
            self.bin_embedding_dim,
            self.alpha, 
            self.gamma
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(encoder_dim, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1)
        )
        
        bin_centers = torch.linspace(self.min_depth, self.max_depth, self.n_bins)
        self.register_buffer('bin_centers', bin_centers.view(1, -1, 1, 1))
    
    def forward(self, x, dataset=None, **kwargs):
        features = self._encode(x)
        
        depth_init = self.decoder(features)
        
        attractor_outputs = self.attractors(features, depth_init)
        
        depth_final = attractor_outputs[-1][0]
        
        if torch.is_tensor(depth_final):
            return depth_final.squeeze(1)
        
        return depth_init.squeeze(1)
    
    def _encode(self, x):
        return torch.zeros(x.shape[0], self.encoder_dim, x.shape[2] // 16, x.shape[3] // 16, device=x.device)
    
    def infer(self, x, dataset=None, **kwargs):
        with torch.no_grad():
            return self.forward(x, dataset, **kwargs)


class ZoeDepthAnythingV1(ZoeDepthV1):
    """
    ZoeDepth with Depth Anything encoder.
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        model_type = self.config.model.get('depth_anything_model_type', 'vitl')
        
        from zoedepth.models.base_models.depth_anything import DepthAnythingEncoder
        self.depth_encoder = DepthAnythingEncoder(model_type=model_type)
        
        self.encoder_dim = self.depth_encoder.embed_dim
    
    def _encode(self, x):
        return self.depth_encoder(x)