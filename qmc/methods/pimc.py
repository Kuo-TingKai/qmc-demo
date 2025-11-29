"""
Path Integral Monte Carlo (PIMC)
For bosonic systems and finite temperature
"""

import numpy as np
from typing import Optional, Dict, Any
from tqdm import tqdm
from qmc.core.base import QuantumModel
from qmc.core.utils import metropolis_acceptance


class PIMC:
    """
    Path Integral Monte Carlo method
    
    Uses imaginary time path integrals for finite temperature calculations
    """
    
    def __init__(self, model: QuantumModel):
        """
        Initialize PIMC
        
        Args:
            model: Quantum model (typically bosonic)
        """
        self.model = model
        self.n_sites = model.n_sites
    
    def compute_action(self, path: np.ndarray, beta: float, 
                      n_tau: int) -> float:
        """
        Compute action for a path configuration
        
        Args:
            path: Path configuration (n_tau x n_sites)
            beta: Inverse temperature
            n_tau: Number of time slices
            
        Returns:
            Action value
        """
        dt = beta / n_tau
        action = 0.0
        
        # Kinetic term
        for tau in range(n_tau):
            tau_next = (tau + 1) % n_tau
            diff = path[tau_next] - path[tau]
            action += 0.5 * np.sum(diff**2) / dt
        
        # Potential term
        for tau in range(n_tau):
            config = path[tau]
            E = self.model.hamiltonian(config)
            action += dt * E
        
        return action
    
    def worm_update(self, path: np.ndarray, beta: float, 
                   n_tau: int) -> np.ndarray:
        """
        Worm algorithm update for bosonic systems
        
        Args:
            path: Current path configuration
            beta: Inverse temperature
            n_tau: Number of time slices
            
        Returns:
            Updated path
        """
        new_path = path.copy()
        
        # Randomly select a time slice and site
        tau = np.random.randint(0, n_tau)
        site = np.random.randint(0, self.n_sites)
        
        # Propose change
        old_value = new_path[tau, site]
        new_value = old_value + np.random.normal(0, 0.1)
        
        # Compute action change
        new_path[tau, site] = new_value
        action_old = self.compute_action(path, beta, n_tau)
        action_new = self.compute_action(new_path, beta, n_tau)
        
        delta_action = action_new - action_old
        
        # Metropolis acceptance
        if metropolis_acceptance(np.exp(-delta_action)):
            return new_path
        else:
            return path
    
    def compute_order_parameter(self, path: np.ndarray) -> float:
        """
        Compute order parameter from path
        
        Args:
            path: Path configuration
            
        Returns:
            Order parameter value
        """
        # Average over time slices
        avg_config = np.mean(path, axis=0)
        
        # Compute staggered magnetization
        sign = np.array([(-1)**i for i in range(self.n_sites)])
        order = np.abs(np.sum(avg_config * sign)) / self.n_sites
        
        return order
    
    def compute_superfluid_density(self, path: np.ndarray, beta: float,
                                   n_tau: int) -> float:
        """
        Compute superfluid density
        
        Args:
            path: Path configuration
            beta: Inverse temperature
            n_tau: Number of time slices
            
        Returns:
            Superfluid density
        """
        # Wind number (winding number)
        # Simplified calculation
        dt = beta / n_tau
        
        # Compute winding
        winding = 0.0
        for site in range(self.n_sites):
            for tau in range(n_tau):
                tau_next = (tau + 1) % n_tau
                diff = path[tau_next, site] - path[tau, site]
                winding += diff
        
        # Superfluid density proportional to winding^2
        rho_s = winding**2 / (beta * self.n_sites)
        
        return rho_s
    
    def run(self, n_steps: int = 10000, beta: float = 2.0,
            n_tau: int = 20, n_equil: int = 1000,
            verbose: bool = True) -> Dict[str, Any]:
        """
        Run PIMC simulation
        
        Args:
            n_steps: Number of Monte Carlo steps
            beta: Inverse temperature
            n_tau: Number of time slices
            n_equil: Equilibration steps
            verbose: Whether to show progress
            
        Returns:
            Dictionary with results
        """
        # Initialize path
        # Path is (n_tau x n_sites) array
        path = np.random.randn(n_tau, self.n_sites)
        
        # Equilibration
        if verbose:
            print("Equilibrating...")
        for step in tqdm(range(n_equil), disable=not verbose):
            path = self.worm_update(path, beta, n_tau)
        
        # Main simulation
        if verbose:
            print("Running PIMC simulation...")
        
        all_energies = []
        all_orders = []
        all_superfluid = []
        
        for step in tqdm(range(n_steps), disable=not verbose):
            # Update path
            path = self.worm_update(path, beta, n_tau)
            
            # Measure
            if step % 10 == 0:
                # Energy
                avg_config = np.mean(path, axis=0)
                E = self.model.hamiltonian(avg_config)
                all_energies.append(E)
                
                # Order parameter
                order = self.compute_order_parameter(path)
                all_orders.append(order)
                
                # Superfluid density
                rho_s = self.compute_superfluid_density(path, beta, n_tau)
                all_superfluid.append(rho_s)
        
        # Compute statistics
        energies = np.array(all_energies)
        orders = np.array(all_orders)
        superfluid = np.array(all_superfluid)
        
        mean_energy = np.mean(energies)
        energy_error = np.std(energies) / np.sqrt(len(energies))
        mean_order = np.mean(orders)
        mean_superfluid = np.mean(superfluid)
        
        results = {
            'energy': mean_energy,
            'energy_error': energy_error,
            'order_parameter': mean_order,
            'superfluid_density': mean_superfluid,
            'energies': energies,
            'orders': orders,
            'superfluid': superfluid,
            'beta': beta,
            'temperature': 1.0 / beta,
        }
        
        return results

