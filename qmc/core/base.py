"""
Base classes for quantum models and quantum states
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional


class QuantumState(ABC):
    """
    Abstract base class for quantum states
    """
    
    def __init__(self, n_sites: int):
        """
        Initialize quantum state
        
        Args:
            n_sites: Number of lattice sites
        """
        self.n_sites = n_sites
    
    @abstractmethod
    def amplitude(self, config: np.ndarray) -> complex:
        """
        Compute wave function amplitude for a given configuration
        
        Args:
            config: Configuration array (e.g., occupation numbers or spins)
            
        Returns:
            Complex amplitude
        """
        pass
    
    @abstractmethod
    def log_amplitude(self, config: np.ndarray) -> complex:
        """
        Compute log of wave function amplitude
        
        Args:
            config: Configuration array
            
        Returns:
            Complex log amplitude
        """
        pass
    
    @abstractmethod
    def sample(self, n_samples: int) -> np.ndarray:
        """
        Sample configurations from the quantum state
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Array of sampled configurations
        """
        pass


class QuantumModel(ABC):
    """
    Abstract base class for quantum many-body models
    """
    
    def __init__(self, lattice, **params):
        """
        Initialize quantum model
        
        Args:
            lattice: Lattice object
            **params: Model parameters
        """
        self.lattice = lattice
        self.n_sites = lattice.n_sites
        self.params = params
    
    @abstractmethod
    def hamiltonian(self, config: np.ndarray) -> float:
        """
        Compute Hamiltonian expectation for a configuration
        
        Args:
            config: Configuration array
            
        Returns:
            Energy value
        """
        pass
    
    @abstractmethod
    def local_energy(self, config: np.ndarray, state: QuantumState) -> complex:
        """
        Compute local energy for a configuration
        
        Args:
            config: Configuration array
            state: Quantum state object
            
        Returns:
            Local energy (complex)
        """
        pass
    
    @abstractmethod
    def generate_excited_states(self, config: np.ndarray) -> list:
        """
        Generate excited states (neighbors) from current configuration
        
        Args:
            config: Current configuration
            
        Returns:
            List of excited configurations
        """
        pass

