"""
Bose-Hubbard Model
For studying superfluid-insulator phase transitions
"""

import numpy as np
from qmc.core.base import QuantumModel, QuantumState
from qmc.core.lattice import SquareLattice


class BoseHubbardModel(QuantumModel):
    """
    Bose-Hubbard Model
    
    H = -t * sum_<ij> (b^dagger_i b_j + h.c.)
        + (U/2) * sum_i n_i * (n_i - 1)
        - mu * sum_i n_i
    
    where:
        t: Hopping amplitude
        U: On-site interaction
        mu: Chemical potential
        n_i: Boson number at site i
    """
    
    def __init__(self, L: int = 6, U: float = 1.0, t: float = 1.0,
                 mu: float = 0.5, max_occupation: int = 3,
                 lattice_type: str = "square"):
        """
        Initialize Bose-Hubbard model
        
        Args:
            L: Linear system size
            U: On-site interaction strength
            t: Hopping amplitude
            mu: Chemical potential
            max_occupation: Maximum boson occupation per site
            lattice_type: Type of lattice
        """
        # Create lattice
        if lattice_type == "square":
            lattice = SquareLattice(L, L, periodic=True)
        else:
            raise ValueError(f"Unknown lattice type: {lattice_type}")
        
        super().__init__(lattice, U=U, t=t, mu=mu, max_occupation=max_occupation)
        
        self.U = U
        self.t = t
        self.mu = mu
        self.max_occupation = max_occupation
        
        # Build hopping matrix
        self._build_hopping_matrix()
    
    def _build_hopping_matrix(self):
        """Build hopping matrix"""
        self.hopping_matrix = np.zeros((self.n_sites, self.n_sites))
        
        for i in range(self.n_sites):
            neighbors = self.lattice.get_neighbors(i)
            for j in neighbors:
                self.hopping_matrix[i, j] = -self.t
    
    def hamiltonian(self, config: np.ndarray) -> float:
        """
        Compute Hamiltonian expectation
        
        Args:
            config: Configuration array (occupation numbers)
            
        Returns:
            Energy value
        """
        # Kinetic energy (hopping)
        kinetic = 0.0
        for i in range(self.n_sites):
            neighbors = self.lattice.get_neighbors(i)
            for j in neighbors:
                # Hopping term: -t * sqrt(n_i * (n_j + 1))
                n_i = max(0, int(config[i]))
                n_j = max(0, int(config[j]))
                if n_i > 0 and n_j < self.max_occupation:
                    kinetic += self.hopping_matrix[i, j] * np.sqrt(n_i * (n_j + 1))
        
        # Potential energy (on-site interaction)
        potential = 0.0
        for i in range(self.n_sites):
            n_i = max(0, int(config[i]))
            potential += 0.5 * self.U * n_i * (n_i - 1)
        
        # Chemical potential
        chemical = -self.mu * np.sum(config)
        
        return kinetic + potential + chemical
    
    def local_energy(self, config: np.ndarray, state: QuantumState) -> complex:
        """
        Compute local energy
        
        Args:
            config: Configuration array
            state: Quantum state
            
        Returns:
            Local energy (complex)
        """
        # Diagonal part
        E_diag = self.hamiltonian(config)
        
        # Off-diagonal part (hopping)
        E_off = 0.0
        psi_0 = state.amplitude(config)
        
        if abs(psi_0) < 1e-10:
            return complex(E_diag, 0)
        
        # Generate excited states
        excited_configs = self.generate_excited_states(config)
        
        for idx, new_config in enumerate(excited_configs):
            psi_new = state.amplitude(new_config)
            # Matrix element for bosonic hopping
            # Find which sites were involved in the hop
            diff = new_config - config
            changed_sites = np.where(np.abs(diff) > 0.5)[0]
            if len(changed_sites) >= 2:
                i, j = changed_sites[0], changed_sites[1]
            else:
                i, j = 0, 0
            n_i = max(0, int(config[i]))
            n_j = max(0, int(config[j]))
            matrix_element = -self.t * np.sqrt(n_i * (n_j + 1))
            E_off += matrix_element * (psi_new / psi_0)
        
        return E_diag + E_off
    
    def generate_excited_states(self, config: np.ndarray) -> list:
        """
        Generate excited states (boson hops)
        
        Args:
            config: Current configuration
            
        Returns:
            List of excited configurations
        """
        excited = []
        
        # Generate single boson hops
        for i in range(self.n_sites):
            neighbors = self.lattice.get_neighbors(i)
            for j in neighbors:
                n_i = max(0, int(config[i]))
                n_j = max(0, int(config[j]))
                
                # Can hop if site i has bosons and site j is not full
                if n_i > 0 and n_j < self.max_occupation:
                    new_config = config.copy()
                    new_config[i] = n_i - 1
                    new_config[j] = n_j + 1
                    excited.append(new_config)
        
        return excited
    
    def compute_superfluid_order(self, config: np.ndarray) -> float:
        """
        Compute superfluid order parameter
        
        Args:
            config: Configuration array
            
        Returns:
            Superfluid order parameter
        """
        # Order parameter: |<b_i>|
        # Simplified: use average occupation
        avg_occupation = np.mean(config)
        return avg_occupation
    
    def compute_mott_lobe(self, config: np.ndarray) -> float:
        """
        Compute Mott insulator indicator
        
        Args:
            config: Configuration array
            
        Returns:
            Mott lobe indicator (variance of occupation)
        """
        # Mott phase: integer filling with no fluctuations
        variance = np.var(config)
        return variance
    
    def compute_winding_number(self, path: np.ndarray) -> float:
        """
        Compute winding number for superfluid phase
        
        Args:
            path: Path configuration (n_tau x n_sites)
            
        Returns:
            Winding number
        """
        # Compute winding in each direction
        n_tau = path.shape[0]
        winding = 0.0
        
        for site in range(self.n_sites):
            neighbors = self.lattice.get_neighbors(site)
            for neighbor in neighbors:
                # Compute phase difference
                phase_diff = 0.0
                for tau in range(n_tau):
                    tau_next = (tau + 1) % n_tau
                    diff = path[tau_next, neighbor] - path[tau_next, site] - \
                           (path[tau, neighbor] - path[tau, site])
                    phase_diff += diff
                winding += phase_diff
        
        return winding / (n_tau * self.n_sites)

