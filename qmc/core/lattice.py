"""
Lattice structures for quantum models
"""

import numpy as np
from typing import List, Tuple, Optional


class Lattice:
    """
    Base class for lattice structures
    """
    
    def __init__(self, n_sites: int, dimension: int = 2):
        """
        Initialize lattice
        
        Args:
            n_sites: Number of sites
            dimension: Spatial dimension
        """
        self.n_sites = n_sites
        self.dimension = dimension
        self.coordinates = None
        self.neighbors = None
    
    def get_neighbors(self, site: int) -> List[int]:
        """
        Get neighboring sites
        
        Args:
            site: Site index
            
        Returns:
            List of neighbor indices
        """
        if self.neighbors is None:
            self._build_neighbors()
        return self.neighbors[site]
    
    def _build_neighbors(self):
        """Build neighbor list"""
        raise NotImplementedError


class SquareLattice(Lattice):
    """
    2D square lattice
    """
    
    def __init__(self, Lx: int, Ly: Optional[int] = None, periodic: bool = True):
        """
        Initialize square lattice
        
        Args:
            Lx: Size in x direction
            Ly: Size in y direction (default: Lx)
            periodic: Whether to use periodic boundary conditions
        """
        if Ly is None:
            Ly = Lx
        n_sites = Lx * Ly
        super().__init__(n_sites, dimension=2)
        self.Lx = Lx
        self.Ly = Ly
        self.periodic = periodic
        self._build_coordinates()
        self._build_neighbors()
    
    def _build_coordinates(self):
        """Build coordinate array"""
        self.coordinates = np.zeros((self.n_sites, 2), dtype=int)
        for i in range(self.n_sites):
            self.coordinates[i] = [i % self.Lx, i // self.Lx]
    
    def _build_neighbors(self):
        """Build neighbor list for square lattice"""
        self.neighbors = [[] for _ in range(self.n_sites)]
        
        for i in range(self.n_sites):
            x, y = self.coordinates[i]
            
            # Right neighbor
            if self.periodic:
                x_right = (x + 1) % self.Lx
            else:
                x_right = x + 1 if x + 1 < self.Lx else None
            
            if x_right is not None:
                j = y * self.Lx + x_right
                self.neighbors[i].append(j)
            
            # Left neighbor
            if self.periodic:
                x_left = (x - 1) % self.Lx
            else:
                x_left = x - 1 if x >= 1 else None
            
            if x_left is not None:
                j = y * self.Lx + x_left
                self.neighbors[i].append(j)
            
            # Up neighbor
            if self.periodic:
                y_up = (y + 1) % self.Ly
            else:
                y_up = y + 1 if y + 1 < self.Ly else None
            
            if y_up is not None:
                j = y_up * self.Lx + x
                self.neighbors[i].append(j)
            
            # Down neighbor
            if self.periodic:
                y_down = (y - 1) % self.Ly
            else:
                y_down = y - 1 if y >= 1 else None
            
            if y_down is not None:
                j = y_down * self.Lx + x
                self.neighbors[i].append(j)


class TriangularLattice(Lattice):
    """
    2D triangular lattice
    """
    
    def __init__(self, Lx: int, Ly: Optional[int] = None, periodic: bool = True):
        """
        Initialize triangular lattice
        
        Args:
            Lx: Size in x direction
            Ly: Size in y direction (default: Lx)
            periodic: Whether to use periodic boundary conditions
        """
        if Ly is None:
            Ly = Lx
        n_sites = Lx * Ly
        super().__init__(n_sites, dimension=2)
        self.Lx = Lx
        self.Ly = Ly
        self.periodic = periodic
        self._build_coordinates()
        self._build_neighbors()
    
    def _build_coordinates(self):
        """Build coordinate array"""
        self.coordinates = np.zeros((self.n_sites, 2), dtype=int)
        for i in range(self.n_sites):
            self.coordinates[i] = [i % self.Lx, i // self.Lx]
    
    def _build_neighbors(self):
        """Build neighbor list for triangular lattice (6 neighbors)"""
        self.neighbors = [[] for _ in range(self.n_sites)]
        
        for i in range(self.n_sites):
            x, y = self.coordinates[i]
            
            # Standard square neighbors
            neighbors_coords = [
                ((x + 1) % self.Lx if self.periodic else x + 1, y),
                ((x - 1) % self.Lx if self.periodic else x - 1, y),
                (x, (y + 1) % self.Ly if self.periodic else y + 1),
                (x, (y - 1) % self.Ly if self.periodic else y - 1),
            ]
            
            # Additional diagonal neighbors for triangular lattice
            if y % 2 == 0:  # Even rows
                neighbors_coords.extend([
                    ((x + 1) % self.Lx if self.periodic else x + 1, 
                     (y + 1) % self.Ly if self.periodic else y + 1),
                    ((x + 1) % self.Lx if self.periodic else x + 1,
                     (y - 1) % self.Ly if self.periodic else y - 1),
                ])
            else:  # Odd rows
                neighbors_coords.extend([
                    ((x - 1) % self.Lx if self.periodic else x - 1,
                     (y + 1) % self.Ly if self.periodic else y + 1),
                    ((x - 1) % self.Lx if self.periodic else x - 1,
                     (y - 1) % self.Ly if self.periodic else y - 1),
                ])
            
            for nx, ny in neighbors_coords:
                if (0 <= nx < self.Lx or self.periodic) and \
                   (0 <= ny < self.Ly or self.periodic):
                    j = ny * self.Lx + (nx % self.Lx)
                    if j not in self.neighbors[i]:
                        self.neighbors[i].append(j)

