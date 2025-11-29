"""
Basic tests for QMC package
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qmc.core.lattice import SquareLattice, TriangularLattice
from qmc.models import HubbardModel, BoseHubbardModel
from qmc.nqs import RBMQuantumState


def test_square_lattice():
    """Test square lattice creation"""
    print("Testing SquareLattice...")
    lattice = SquareLattice(4, 4, periodic=True)
    assert lattice.n_sites == 16
    assert len(lattice.get_neighbors(0)) > 0
    print("  ✓ SquareLattice works correctly")


def test_hubbard_model():
    """Test Hubbard model creation"""
    print("Testing HubbardModel...")
    model = HubbardModel(L=4, U=4.0, t=1.0)
    assert model.n_sites == 16
    assert model.U == 4.0
    assert model.t == 1.0
    
    # Test Hamiltonian
    config = np.random.choice([-1, 1], size=16)
    energy = model.hamiltonian(config)
    assert isinstance(energy, (float, np.floating))
    print("  ✓ HubbardModel works correctly")


def test_bose_hubbard_model():
    """Test Bose-Hubbard model creation"""
    print("Testing BoseHubbardModel...")
    model = BoseHubbardModel(L=4, U=1.0, t=1.0, mu=0.5)
    assert model.n_sites == 16
    assert model.U == 1.0
    assert model.t == 1.0
    
    # Test Hamiltonian
    config = np.random.randint(0, 3, size=16).astype(float)
    energy = model.hamiltonian(config)
    assert isinstance(energy, (float, np.floating))
    print("  ✓ BoseHubbardModel works correctly")


def test_rbm_quantum_state():
    """Test RBM quantum state"""
    print("Testing RBMQuantumState...")
    nqs = RBMQuantumState(n_sites=16, hidden_dim=8)
    assert nqs.n_sites == 16
    assert nqs.hidden_dim == 8
    
    # Test amplitude
    config = np.random.choice([-1, 1], size=16)
    amp = nqs.amplitude(config)
    assert isinstance(amp, (complex, np.complexfloating))
    print("  ✓ RBMQuantumState works correctly")


def run_all_tests():
    """Run all basic tests"""
    print("=" * 60)
    print("Running Basic Tests")
    print("=" * 60)
    
    test_square_lattice()
    test_hubbard_model()
    test_bose_hubbard_model()
    test_rbm_quantum_state()
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()

