"""
Hubbard Model
Strongly correlated fermionic system
"""

import numpy as np
from qmc.core.base import QuantumModel, QuantumState
from qmc.core.lattice import SquareLattice


class HubbardModel(QuantumModel):
    """
    Hubbard Model
    
    H = -t * sum_<ij>,sigma (c^dagger_i,sigma c_j,sigma + h.c.)
        + U * sum_i n_i,up * n_i,down
        - mu * sum_i,sigma n_i,sigma
    
    where:
        t: Hopping amplitude
        U: On-site interaction
        mu: Chemical potential
    """
    
    def __init__(self, L: int = 4, U: float = 4.0, t: float = 1.0,
                 mu: float = 0.0, lattice_type: str = "square"):
        """
        Initialize Hubbard model
        
        Args:
            L: Linear system size
            U: On-site interaction strength
            t: Hopping amplitude
            mu: Chemical potential
            lattice_type: Type of lattice ("square", "triangular")
        """
        # Create lattice
        if lattice_type == "square":
            lattice = SquareLattice(L, L, periodic=True)
        else:
            raise ValueError(f"Unknown lattice type: {lattice_type}")
        
        super().__init__(lattice, U=U, t=t, mu=mu)
        
        self.U = U
        self.t = t
        self.mu = mu
        
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
            config: Configuration array (spin-1/2: -1 or 1)
            
        Returns:
            Energy value
        """
        # Kinetic energy (hopping)
        kinetic = 0.0
        for i in range(self.n_sites):
            neighbors = self.lattice.get_neighbors(i)
            for j in neighbors:
                # Simplified: assume config represents occupation
                kinetic += self.hopping_matrix[i, j] * config[i] * config[j]
        
        # Potential energy (on-site interaction)
        # For spin-1/2, U term when both spins are present
        potential = 0.0
        for i in range(self.n_sites):
            # Simplified interaction
            if abs(config[i]) > 0.5:
                potential += self.U * 0.25  # Simplified
        
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
        
        # Generate excited states (single particle hops)
        excited_configs = self.generate_excited_states(config)
        
        for new_config in excited_configs:
            psi_new = state.amplitude(new_config)
            # Matrix element for hopping
            matrix_element = -self.t
            E_off += matrix_element * (psi_new / psi_0)
        
        return E_diag + E_off
    
    def generate_excited_states(self, config: np.ndarray) -> list:
        """
        Generate excited states (neighbors) from current configuration
        
        Args:
            config: Current configuration
            
        Returns:
            List of excited configurations
        """
        excited = []
        
        # Generate single particle hops
        for i in range(self.n_sites):
            neighbors = self.lattice.get_neighbors(i)
            for j in neighbors:
                if i != j:
                    new_config = config.copy()
                    # Swap or move particle
                    if abs(new_config[i]) > 0.5 and abs(new_config[j]) < 0.5:
                        new_config[j] = new_config[i]
                        new_config[i] = 0.0
                        excited.append(new_config)
        
        return excited
    
    def compute_double_occupancy(self, config: np.ndarray) -> float:
        """
        Compute double occupancy (n_up * n_down)
        
        Args:
            config: Configuration array
            
        Returns:
            Double occupancy value
        """
        # Simplified: for spin-1/2 representation
        double_occ = 0.0
        for i in range(self.n_sites):
            if abs(config[i]) > 0.5:
                double_occ += 0.25  # Simplified
        
        return double_occ / self.n_sites
    
    def compute_momentum_distribution(self, config: np.ndarray) -> np.ndarray:
        """
        Compute momentum distribution n(k)
        
        Args:
            config: Configuration array
            
        Returns:
            Momentum distribution array
        """
        # Fourier transform
        n_k = np.fft.fft(config)
        return np.abs(n_k)**2 / self.n_sites

