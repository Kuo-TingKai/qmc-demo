"""
Convolutional Neural Network (CNN) Quantum State
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from qmc.core.base import QuantumState


class CNNQuantumState(QuantumState):
    """
    Convolutional Neural Network Quantum State
    
    Uses CNN to capture spatial correlations in lattice systems
    """
    
    def __init__(self, n_sites: int, Lx: int, Ly: int,
                 n_channels: list = [16, 32], kernel_size: int = 3,
                 dtype: torch.dtype = torch.float32):
        """
        Initialize CNN quantum state
        
        Args:
            n_sites: Number of lattice sites
            Lx: Lattice size in x direction
            Ly: Lattice size in y direction
            n_channels: List of channel numbers for each conv layer
            kernel_size: Convolution kernel size
            dtype: Data type for tensors
        """
        super().__init__(n_sites)
        self.Lx = Lx
        self.Ly = Ly
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.dtype = dtype
        
        # Build CNN layers
        layers = []
        in_channels = 1
        
        for out_channels in n_channels:
            layers.append(nn.Conv2d(in_channels, out_channels, 
                                   kernel_size=kernel_size, padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Flatten and fully connected layers
        # Compute flattened size
        conv_output_size = Lx * Ly * n_channels[-1]
        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Real and imaginary parts
        )
    
    def log_amplitude(self, config: np.ndarray) -> complex:
        """
        Compute log wave function amplitude
        
        Args:
            config: Configuration array (flattened)
            
        Returns:
            Log amplitude (complex)
        """
        # Convert to tensor
        if isinstance(config, np.ndarray):
            config_tensor = torch.from_numpy(config.astype(np.float32))
        else:
            config_tensor = config
        
        # Reshape to 2D lattice
        if config_tensor.dim() == 1:
            config_tensor = config_tensor.unsqueeze(0)
        
        # Reshape to (batch, 1, Lx, Ly)
        config_2d = config_tensor.view(-1, 1, self.Lx, self.Ly)
        
        # Convolutional layers
        x = self.conv_layers(config_2d)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        output = self.fc(x)
        
        # Split into real and imaginary parts
        real_part = output[:, 0]
        imag_part = output[:, 1]
        
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
        Sample configurations
        
        Args:
            n_samples: Number of samples
            
        Returns:
            Array of sampled configurations
        """
        # For now, use uniform random sampling
        samples = np.random.choice([-1, 1], size=(n_samples, self.n_sites))
        return samples

