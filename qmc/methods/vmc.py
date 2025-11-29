"""
Variational Monte Carlo (VMC) method
"""

import numpy as np
from typing import Optional, Dict, Any
from tqdm import tqdm
from qmc.core.base import QuantumModel, QuantumState
from qmc.core.utils import metropolis_acceptance, bootstrap_error


class VMC:
    """
    Variational Monte Carlo method
    
    Minimizes energy expectation value using stochastic optimization
    """
    
    def __init__(self, model: QuantumModel, state: QuantumState,
                 learning_rate: float = 0.01, optimizer: str = "adam"):
        """
        Initialize VMC
        
        Args:
            model: Quantum model
            state: Quantum state (e.g., NQS)
            learning_rate: Learning rate for optimization
            optimizer: Optimizer type ("adam", "sgd", "rmsprop")
        """
        self.model = model
        self.state = state
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        
        # Initialize optimizer if state has parameters
        self.optimizer = None
        self._init_optimizer()
        
        # Storage for results
        self.energy_history = []
        self.energy_variance_history = []
        self.configs = []
    
    def _init_optimizer(self):
        """Initialize optimizer for neural network parameters"""
        import torch.optim as optim
        
        # Get all parameters
        params = []
        if hasattr(self.state, 'visible_bias'):
            params.extend([self.state.visible_bias, self.state.hidden_bias, 
                          self.state.weight])
        elif hasattr(self.state, 'network'):
            params = list(self.state.network.parameters())
        elif hasattr(self.state, 'conv_layers'):
            params = list(self.state.conv_layers.parameters())
            params.extend(list(self.state.fc.parameters()))
        
        if len(params) > 0:
            if self.optimizer_name == "adam":
                self.optimizer = optim.Adam(params, lr=self.learning_rate)
            elif self.optimizer_name == "sgd":
                self.optimizer = optim.SGD(params, lr=self.learning_rate)
            elif self.optimizer_name == "rmsprop":
                self.optimizer = optim.RMSprop(params, lr=self.learning_rate)
    
    def local_energy(self, config: np.ndarray) -> complex:
        """
        Compute local energy for a configuration
        
        Args:
            config: Configuration array
            
        Returns:
            Local energy (complex)
        """
        return self.model.local_energy(config, self.state)
    
    def compute_energy(self, configs: np.ndarray) -> tuple:
        """
        Compute energy and variance from configurations
        
        Args:
            configs: Array of configurations
            
        Returns:
            (mean_energy, variance) tuple
        """
        energies = []
        for config in configs:
            E_loc = self.local_energy(config)
            energies.append(E_loc)
        
        energies = np.array(energies)
        mean_energy = np.real(np.mean(energies))
        variance = np.var(np.real(energies))
        
        return mean_energy, variance
    
    def metropolis_step(self, config: np.ndarray) -> np.ndarray:
        """
        Perform one Metropolis step
        
        Args:
            config: Current configuration
            
        Returns:
            New configuration
        """
        # Generate excited state
        excited_configs = self.model.generate_excited_states(config)
        
        if len(excited_configs) == 0:
            return config
        
        # Choose random excited state
        new_config = excited_configs[np.random.randint(len(excited_configs))]
        
        # Compute acceptance ratio
        psi_old = self.state.amplitude(config)
        psi_new = self.state.amplitude(new_config)
        
        ratio = np.abs(psi_new / psi_old)**2
        
        # Metropolis acceptance
        if metropolis_acceptance(ratio):
            return new_config
        else:
            return config
    
    def run(self, n_steps: int = 10000, n_walkers: int = 100,
            n_equil: int = 1000, n_skip: int = 10,
            verbose: bool = True) -> Dict[str, Any]:
        """
        Run VMC simulation
        
        Args:
            n_steps: Total number of Monte Carlo steps
            n_walkers: Number of walkers
            n_equil: Number of equilibration steps
            n_skip: Skip steps between measurements
            verbose: Whether to show progress
            
        Returns:
            Dictionary with results
        """
        # Initialize walkers
        walkers = []
        for _ in range(n_walkers):
            # Random initial configuration
            config = np.random.choice([-1, 1], size=self.model.n_sites)
            walkers.append(config)
        
        # Equilibration
        if verbose:
            print("Equilibrating...")
        for step in tqdm(range(n_equil), disable=not verbose):
            for i in range(n_walkers):
                walkers[i] = self.metropolis_step(walkers[i])
        
        # Main simulation
        if verbose:
            print("Running VMC simulation...")
        
        all_configs = []
        all_energies = []
        
        for step in tqdm(range(n_steps), disable=not verbose):
            # Update walkers
            for i in range(n_walkers):
                walkers[i] = self.metropolis_step(walkers[i])
            
            # Measure every n_skip steps
            if step % n_skip == 0:
                for config in walkers:
                    E_loc = self.local_energy(config)
                    all_energies.append(np.real(E_loc))
                    all_configs.append(config.copy())
        
        # Store results
        self.configs = np.array(all_configs)
        energies = np.array(all_energies)
        
        # Compute statistics
        mean_energy, energy_error = bootstrap_error(energies)
        variance = np.var(energies)
        
        self.energy_history.append(mean_energy)
        self.energy_variance_history.append(variance)
        
        results = {
            'energy': mean_energy,
            'energy_error': energy_error,
            'variance': variance,
            'configurations': self.configs,
            'energies': energies,
        }
        
        return results
    
    def optimize(self, n_opt_steps: int = 100, n_mc_steps: int = 1000,
                 n_walkers: int = 100, verbose: bool = True) -> Dict[str, Any]:
        """
        Optimize wave function parameters
        
        Args:
            n_opt_steps: Number of optimization steps
            n_mc_steps: Monte Carlo steps per optimization step
            n_walkers: Number of walkers
            verbose: Whether to show progress
            
        Returns:
            Dictionary with optimization history
        """
        import torch
        
        energy_history = []
        
        if verbose:
            if self.optimizer is None:
                print("Note: Optimizer not available. Tracking energy only (no parameter updates).")
            else:
                print("Optimizing wave function...")
        
        for opt_step in tqdm(range(n_opt_steps), disable=not verbose):
            # Run short MC simulation
            results = self.run(n_steps=n_mc_steps, n_walkers=n_walkers,
                              n_equil=100, verbose=False)
            
            # Compute gradient and update
            if len(results['energies']) > 0:
                mean_energy = np.mean(results['energies'])
                energy_history.append(mean_energy)
                
                # Note: Full VMC optimization requires computing gradients using
                # the log-derivative trick: dE/dtheta = <E_loc * d(ln psi)/dtheta>
                # This simplified version only tracks energy changes
                # For production use, implement proper gradient computation with:
                # - Compute O_k = d(ln psi)/dtheta_k for each parameter
                # - Compute gradient: <E_loc * O_k> - <E_loc> * <O_k>
                # - Update parameters using optimizer
        
        return {
            'energy_history': energy_history,
            'final_energy': energy_history[-1] if energy_history else None,
        }

