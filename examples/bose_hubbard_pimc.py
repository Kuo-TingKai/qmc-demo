"""
Example: Bose-Hubbard Model with Path Integral Monte Carlo

This example demonstrates:
1. Setting up a Bose-Hubbard model
2. Using PIMC for finite temperature calculations
3. Studying superfluid-insulator phase transition
4. Computing order parameters
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qmc.models import BoseHubbardModel
from qmc.methods import PIMC


def main():
    """Main example function"""
    
    print("=" * 60)
    print("Bose-Hubbard Model - Path Integral Monte Carlo")
    print("=" * 60)
    
    # Model parameters
    L = 6  # Linear system size
    U = 1.0  # Interaction strength
    t = 1.0  # Hopping amplitude
    mu = 0.5  # Chemical potential
    
    print(f"\nModel Parameters:")
    print(f"  System size: {L}x{L} = {L*L} sites")
    print(f"  U/t = {U/t}")
    print(f"  U = {U}, t = {t}, mu = {mu}")
    
    # Create model
    print("\nCreating Bose-Hubbard model...")
    model = BoseHubbardModel(L=L, U=U, t=t, mu=mu, max_occupation=3)
    
    # Study phase transition by varying temperature
    print("\n" + "=" * 60)
    print("Studying Phase Transition vs Temperature")
    print("=" * 60)
    
    temperatures = np.linspace(0.1, 2.0, 10)
    energies = []
    orders = []
    superfluid_densities = []
    
    for T in temperatures:
        beta = 1.0 / T
        print(f"\nTemperature: T = {T:.3f} (beta = {beta:.3f})")
        
        # Create PIMC
        pimc = PIMC(model)
        
        # Run simulation
        results = pimc.run(
            n_steps=3000,
            beta=beta,
            n_tau=20,
            n_equil=500,
            verbose=False
        )
        
        energies.append(results['energy'])
        orders.append(results['order_parameter'])
        superfluid_densities.append(results['superfluid_density'])
        
        print(f"  Energy: {results['energy']:.4f}")
        print(f"  Order parameter: {results['order_parameter']:.4f}")
        print(f"  Superfluid density: {results['superfluid_density']:.4f}")
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Energy vs temperature
    plt.subplot(1, 3, 1)
    plt.plot(temperatures, energies, 'o-', linewidth=2, markersize=6)
    plt.xlabel('Temperature')
    plt.ylabel('Energy')
    plt.title('Energy vs Temperature')
    plt.grid(True, alpha=0.3)
    
    # Order parameter vs temperature
    plt.subplot(1, 3, 2)
    plt.plot(temperatures, orders, 'o-', linewidth=2, markersize=6, color='green')
    plt.xlabel('Temperature')
    plt.ylabel('Order Parameter')
    plt.title('Order Parameter vs Temperature')
    plt.grid(True, alpha=0.3)
    
    # Superfluid density vs temperature
    plt.subplot(1, 3, 3)
    plt.plot(temperatures, superfluid_densities, 'o-', linewidth=2, markersize=6, color='red')
    plt.xlabel('Temperature')
    plt.ylabel('Superfluid Density')
    plt.title('Superfluid Density vs Temperature')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bose_hubbard_phase_transition.png', dpi=150)
    print(f"\nPhase transition plot saved to: bose_hubbard_phase_transition.png")
    
    # Study at fixed temperature, varying U/t
    print("\n" + "=" * 60)
    print("Studying Phase Transition vs U/t")
    print("=" * 60)
    
    U_over_t_values = np.linspace(0.1, 5.0, 10)
    beta = 2.0  # Fixed temperature
    
    energies_ut = []
    orders_ut = []
    superfluid_ut = []
    
    for U_over_t in U_over_t_values:
        U_val = U_over_t * t
        print(f"\nU/t = {U_over_t:.2f} (U = {U_val:.2f})")
        
        # Create new model with different U
        model_ut = BoseHubbardModel(L=L, U=U_val, t=t, mu=mu, max_occupation=3)
        pimc_ut = PIMC(model_ut)
        
        # Run simulation
        results = pimc_ut.run(
            n_steps=3000,
            beta=beta,
            n_tau=20,
            n_equil=500,
            verbose=False
        )
        
        energies_ut.append(results['energy'])
        orders_ut.append(results['order_parameter'])
        superfluid_ut.append(results['superfluid_density'])
        
        print(f"  Energy: {results['energy']:.4f}")
        print(f"  Order parameter: {results['order_parameter']:.4f}")
        print(f"  Superfluid density: {results['superfluid_density']:.4f}")
    
    # Plot U/t dependence
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(U_over_t_values, energies_ut, 'o-', linewidth=2, markersize=6)
    plt.xlabel('U/t')
    plt.ylabel('Energy')
    plt.title('Energy vs U/t')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(U_over_t_values, orders_ut, 'o-', linewidth=2, markersize=6, color='green')
    plt.xlabel('U/t')
    plt.ylabel('Order Parameter')
    plt.title('Order Parameter vs U/t')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(U_over_t_values, superfluid_ut, 'o-', linewidth=2, markersize=6, color='red')
    plt.xlabel('U/t')
    plt.ylabel('Superfluid Density')
    plt.title('Superfluid Density vs U/t')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bose_hubbard_ut_dependence.png', dpi=150)
    print(f"\nU/t dependence plot saved to: bose_hubbard_ut_dependence.png")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

