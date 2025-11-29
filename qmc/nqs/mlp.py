"""
Multi-Layer Perceptron (MLP) Quantum State
"""

import numpy as np
import torch
import torch.nn as nn
from qmc.core.base import QuantumState


class MLPQuantumState(QuantumState):
    """
    Multi-Layer Perceptron Quantum State
    
    Uses a feedforward neural network to represent the wave function
    """
    
    def __init__(self, n_sites: int, hidden_dims: list = [64, 64], 
                 activation: str = "tanh", dtype: torch.dtype = torch.float32):
        """
        Initialize MLP quantum state
        
        Args:
            n_sites: Number of lattice sites
            hidden_dims: List of hidden layer dimensions
            activation: Activation function ("tanh", "relu", "gelu")
            dtype: Data type for tensors
        """
        super().__init__(n_sites)
        self.hidden_dims = hidden_dims
        self.activation_name = activation
        self.dtype = dtype
        
        # Build network layers
        layers = []
        input_dim = n_sites
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(self._get_activation())
            input_dim = hidden_dim
        
        # Output layer (real and imaginary parts)
        layers.append(nn.Linear(input_dim, 2))
        
        self.network = nn.Sequential(*layers)
    
    def _get_activation(self):
        """Get activation function"""
        if self.activation_name == "tanh":
            return nn.Tanh()
        elif self.activation_name == "relu":
            return nn.ReLU()
        elif self.activation_name == "gelu":
            return nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {self.activation_name}")
    
    def log_amplitude(self, config: np.ndarray) -> complex:
        """
        Compute log wave function amplitude
        
        Args:
            config: Configuration array
            
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
        
        # Forward pass
        output = self.network(config_tensor)
        
        # Split into real and imaginary parts
        real_part = output[:, 0]
        imag_part = output[:, 1]
        
        # Compute log amplitude
        log_psi = real_part + 1j * imag_part
        
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
    
    def sample(self, n_samples: int) -> np.ndarray:
        """
        Sample configurations (simple random sampling)
        
        Args:
            n_samples: Number of samples
            
        Returns:
            Array of sampled configurations
        """
        # For now, use uniform random sampling
        # In practice, would use MCMC with this wave function
        samples = np.random.choice([-1, 1], size=(n_samples, self.n_sites))
        return samples

