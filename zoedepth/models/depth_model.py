"""
Base depth model interface.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class DepthModel(nn.Module, ABC):
    """
    Abstract base class for depth estimation models.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dataset = config.model.get('dataset', 'kitti')
    
    @abstractmethod
    def forward(self, x, dataset=None, **kwargs):
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, 3, H, W)
            dataset: Optional dataset name for conditioning
            
        Returns:
            Depth prediction
        """
        pass
    
    @abstractmethod
    def infer(self, x, dataset=None, **kwargs):
        """
        Inference mode forward pass.
        """
        pass