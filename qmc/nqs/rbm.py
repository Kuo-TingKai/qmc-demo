"""
Restricted Boltzmann Machine (RBM) Quantum State
Based on Carleo & Troyer (2017) - Neural Network Quantum States
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from qmc.core.base import QuantumState


class RBMQuantumState(QuantumState):
    """
    Restricted Boltzmann Machine Quantum State
    
    Wave function: psi(sigma) = exp(sum_i a_i sigma_i) * 
                   product_j cosh(sum_i W_ij sigma_i + b_j)
    """
    
    def __init__(self, n_sites: int, hidden_dim: int = 16, 
                 dtype: torch.dtype = torch.float32, 
                 init_scale: float = 0.01):
        """
        Initialize RBM quantum state
        
        Args:
            n_sites: Number of visible units (lattice sites)
            hidden_dim: Number of hidden units
            dtype: Data type for tensors
            init_scale: Initialization scale for weights
        """
        super().__init__(n_sites)
        self.hidden_dim = hidden_dim
        self.dtype = dtype
        
        # Visible bias (a_i)
        self.visible_bias = nn.Parameter(
            torch.randn(n_sites, dtype=dtype) * init_scale
        )
        
        # Hidden bias (b_j)
        self.hidden_bias = nn.Parameter(
            torch.randn(hidden_dim, dtype=dtype) * init_scale
        )
        
        # Weight matrix (W_ij)
        self.weight = nn.Parameter(
            torch.randn(n_sites, hidden_dim, dtype=dtype) * init_scale
        )
        
        # Initialize parameters
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Reset parameters with proper initialization"""
        nn.init.normal_(self.visible_bias, std=0.01)
        nn.init.normal_(self.hidden_bias, std=0.01)
        nn.init.normal_(self.weight, std=0.01)
    
    def log_amplitude(self, config: np.ndarray) -> complex:
        """
        Compute log wave function amplitude
        
        Args:
            config: Configuration array (shape: n_sites)
            
        Returns:
            Log amplitude (complex)
        """
        # Convert to tensor
        if isinstance(config, np.ndarray):
            config_tensor = torch.from_numpy(config.astype(np.float32))
        else:
            config_tensor = config
        
        # Ensure correct shape
        if config_tensor.dim() == 1:
            config_tensor = config_tensor.unsqueeze(0)
        
        # Visible layer contribution
        visible_term = torch.sum(self.visible_bias * config_tensor, dim=-1)
        
        # Hidden layer contribution
        hidden_input = torch.matmul(config_tensor, self.weight) + self.hidden_bias
        hidden_term = torch.sum(torch.log(torch.cosh(hidden_input)), dim=-1)
        
        log_psi = visible_term + hidden_term
        
        # Convert to numpy complex
        if isinstance(log_psi, torch.Tensor):
            log_psi = log_psi.item() if log_psi.numel() == 1 else log_psi.detach().numpy()
        
        return log_psi
    
    def amplitude(self, config: np.ndarray) -> complex:
        """
        Compute wave function amplitude
        
        Args:
            config: Configuration array
            
        Returns:
            Complex amplitude
        """
        log_psi = self.log_amplitude(config)
        return np.exp(log_psi)
    
    def sample(self, n_samples: int, n_gibbs_steps: int = 10) -> np.ndarray:
        """
        Sample configurations using Gibbs sampling
        
        Args:
            n_samples: Number of samples
            n_gibbs_steps: Number of Gibbs sampling steps
            
        Returns:
            Array of sampled configurations
        """
        # Initialize random configurations
        samples = []
        config = torch.randint(0, 2, (1, self.n_sites), dtype=torch.float32) * 2 - 1
        
        for _ in range(n_samples):
            # Gibbs sampling
            for _ in range(n_gibbs_steps):
                # Sample hidden units
                hidden_input = torch.matmul(config, self.weight) + self.hidden_bias
                hidden_probs = torch.sigmoid(2 * hidden_input)
                hidden = (torch.rand_like(hidden_probs) < hidden_probs).float() * 2 - 1
                
                # Sample visible units
                visible_input = torch.matmul(hidden, self.weight.t()) + self.visible_bias
                visible_probs = torch.sigmoid(2 * visible_input)
                config = (torch.rand_like(visible_probs) < visible_probs).float() * 2 - 1
            
            samples.append(config.clone().squeeze().numpy())
        
        return np.array(samples)
    
    def get_parameters(self) -> dict:
        """Get all parameters as dictionary"""
        return {
            'visible_bias': self.visible_bias.detach().numpy(),
            'hidden_bias': self.hidden_bias.detach().numpy(),
            'weight': self.weight.detach().numpy(),
        }
    
    def set_parameters(self, params: dict):
        """Set parameters from dictionary"""
        if 'visible_bias' in params:
            self.visible_bias.data = torch.from_numpy(params['visible_bias'])
        if 'hidden_bias' in params:
            self.hidden_bias.data = torch.from_numpy(params['hidden_bias'])
        if 'weight' in params:
            self.weight.data = torch.from_numpy(params['weight'])

