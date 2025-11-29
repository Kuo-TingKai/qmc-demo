"""
Example: Hubbard Model with Variational Monte Carlo using Neural Network Quantum States

This example demonstrates:
1. Setting up a Hubbard model
2. Using RBM quantum state
3. Running VMC optimization
4. Computing ground state energy
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qmc.models import HubbardModel
from qmc.methods import VMC
from qmc.nqs import RBMQuantumState


def main():
    """Main example function"""
    
    print("=" * 60)
    print("Hubbard Model - Variational Monte Carlo with NQS")
    print("=" * 60)
    
    # Model parameters
    L = 4  # Linear system size (4x4 = 16 sites)
    U = 4.0  # Interaction strength
    t = 1.0  # Hopping amplitude
    mu = 0.0  # Chemical potential
    
    print(f"\nModel Parameters:")
    print(f"  System size: {L}x{L} = {L*L} sites")
    print(f"  U/t = {U/t}")
    print(f"  U = {U}, t = {t}, mu = {mu}")
    
    # Create model
    print("\nCreating Hubbard model...")
    model = HubbardModel(L=L, U=U, t=t, mu=mu)
    
    # Create neural network quantum state
    print("Initializing RBM quantum state...")
    hidden_dim = 16
    nqs = RBMQuantumState(model.n_sites, hidden_dim=hidden_dim)
    print(f"  Hidden units: {hidden_dim}")
    
    # Create VMC
    print("Setting up VMC...")
    vmc = VMC(model, nqs, learning_rate=0.01, optimizer="adam")
    
    # Run VMC simulation
    print("\nRunning VMC simulation...")
    print("  This may take a few minutes...")
    
    results = vmc.run(
        n_steps=5000,
        n_walkers=100,
        n_equil=1000,
        n_skip=10,
        verbose=True
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"Ground state energy: {results['energy']:.6f} ± {results['energy_error']:.6f}")
    print(f"Energy variance: {results['variance']:.6f}")
    print(f"Number of measurements: {len(results['energies'])}")
    
    # Plot energy history
    if len(results['energies']) > 0:
        plt.figure(figsize=(10, 6))
        
        # Energy vs step
        plt.subplot(1, 2, 1)
        steps = np.arange(len(results['energies'])) * 10
        plt.plot(steps, results['energies'], alpha=0.6, linewidth=0.5)
        plt.axhline(results['energy'], color='r', linestyle='--', 
                   label=f'Mean: {results["energy"]:.4f}')
        plt.xlabel('Monte Carlo Step')
        plt.ylabel('Local Energy')
        plt.title('Energy vs MC Step')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Energy histogram
        plt.subplot(1, 2, 2)
        plt.hist(results['energies'], bins=50, density=True, alpha=0.7)
        plt.axvline(results['energy'], color='r', linestyle='--', 
                 label=f'Mean: {results["energy"]:.4f}')
        plt.xlabel('Energy')
        plt.ylabel('Probability Density')
        plt.title('Energy Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('hubbard_vmc_results.png', dpi=150)
        print(f"\nPlot saved to: hubbard_vmc_results.png")
    
    # Optional: Optimize wave function
    print("\n" + "=" * 60)
    print("Optimizing wave function parameters...")
    print("=" * 60)
    
    try:
        opt_results = vmc.optimize(
            n_opt_steps=20,
            n_mc_steps=500,
            n_walkers=50,
            verbose=True
        )
        
        if opt_results['energy_history']:
            print(f"\nOptimization Results:")
            print(f"  Initial energy: {opt_results['energy_history'][0]:.6f}")
            print(f"  Final energy: {opt_results['energy_history'][-1]:.6f}")
            print(f"  Improvement: {opt_results['energy_history'][0] - opt_results['energy_history'][-1]:.6f}")
            
            # Plot optimization history
            plt.figure(figsize=(8, 5))
            plt.plot(opt_results['energy_history'], 'o-', linewidth=2, markersize=4)
            plt.xlabel('Optimization Step')
            plt.ylabel('Energy')
            plt.title('VMC Optimization History')
            plt.grid(True, alpha=0.3)
            plt.savefig('hubbard_vmc_optimization.png', dpi=150)
            print(f"  Optimization plot saved to: hubbard_vmc_optimization.png")
    except ValueError as e:
        print(f"\nNote: Wave function optimization skipped: {e}")
        print("  (This requires trainable parameters in the quantum state)")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

