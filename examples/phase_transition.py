"""
Example: Comprehensive Phase Transition Analysis

This example demonstrates:
1. Systematic phase transition studies
2. Critical point identification
3. Finite-size scaling
4. Order parameter analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy.optimize import curve_fit

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qmc.models import BoseHubbardModel, HubbardModel
from qmc.methods import PIMC, VMC
from qmc.nqs import RBMQuantumState


def fit_power_law(x, a, b, c):
    """Power law fitting function"""
    return a * (x - c)**b


def analyze_phase_transition():
    """Analyze phase transition in Bose-Hubbard model"""
    
    print("=" * 60)
    print("Comprehensive Phase Transition Analysis")
    print("=" * 60)
    
    # Parameters
    L_values = [4, 6, 8]  # Different system sizes for finite-size scaling
    U_over_t_values = np.linspace(0.5, 4.0, 15)
    t = 1.0
    mu = 0.5
    beta = 2.0
    
    all_results = {}
    
    for L in L_values:
        print(f"\n{'='*60}")
        print(f"System size: {L}x{L}")
        print(f"{'='*60}")
        
        energies = []
        orders = []
        superfluid = []
        
        for U_over_t in U_over_t_values:
            U = U_over_t * t
            print(f"  U/t = {U_over_t:.2f}...", end=" ", flush=True)
            
            # Create model
            model = BoseHubbardModel(L=L, U=U, t=t, mu=mu, max_occupation=3)
            pimc = PIMC(model)
            
            # Run simulation
            results = pimc.run(
                n_steps=2000,
                beta=beta,
                n_tau=20,
                n_equil=500,
                verbose=False
            )
            
            energies.append(results['energy'])
            orders.append(results['order_parameter'])
            superfluid.append(results['superfluid_density'])
            
            print(f"E={results['energy']:.3f}, O={results['order_parameter']:.3f}")
        
        all_results[L] = {
            'U_over_t': U_over_t_values,
            'energies': np.array(energies),
            'orders': np.array(orders),
            'superfluid': np.array(superfluid),
        }
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    colors = ['blue', 'green', 'red']
    for i, L in enumerate(L_values):
        results = all_results[L]
        axes[0].plot(results['U_over_t'], results['energies'], 
                    'o-', label=f'L={L}', color=colors[i], linewidth=2, markersize=4)
        axes[1].plot(results['U_over_t'], results['orders'], 
                    'o-', label=f'L={L}', color=colors[i], linewidth=2, markersize=4)
        axes[2].plot(results['U_over_t'], results['superfluid'], 
                    'o-', label=f'L={L}', color=colors[i], linewidth=2, markersize=4)
    
    axes[0].set_xlabel('U/t')
    axes[0].set_ylabel('Energy')
    axes[0].set_title('Energy vs U/t (Finite-Size Scaling)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('U/t')
    axes[1].set_ylabel('Order Parameter')
    axes[1].set_title('Order Parameter vs U/t')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_xlabel('U/t')
    axes[2].set_ylabel('Superfluid Density')
    axes[2].set_title('Superfluid Density vs U/t')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('phase_transition_finite_size.png', dpi=150)
    print(f"\nFinite-size scaling plot saved to: phase_transition_finite_size.png")
    
    # Try to identify critical point
    print("\n" + "=" * 60)
    print("Critical Point Analysis")
    print("=" * 60)
    
    # Find where order parameter drops significantly
    L_large = max(L_values)
    results_large = all_results[L_large]
    orders_large = results_large['orders']
    U_over_t_large = results_large['U_over_t']
    
    # Find inflection point (maximum derivative)
    d_order = np.diff(orders_large)
    dU = np.diff(U_over_t_large)
    derivative = d_order / dU
    
    max_derivative_idx = np.argmax(np.abs(derivative))
    U_critical = U_over_t_large[max_derivative_idx]
    
    print(f"Estimated critical point: U_c/t ≈ {U_critical:.3f}")
    print(f"  (based on maximum order parameter derivative)")
    
    # Plot with critical point
    plt.figure(figsize=(10, 6))
    for i, L in enumerate(L_values):
        results = all_results[L]
        plt.plot(results['U_over_t'], results['orders'], 
                'o-', label=f'L={L}', color=colors[i], linewidth=2, markersize=5)
    
    plt.axvline(U_critical, color='black', linestyle='--', 
               linewidth=2, label=f'U_c/t ≈ {U_critical:.2f}')
    plt.xlabel('U/t')
    plt.ylabel('Order Parameter')
    plt.title('Phase Transition - Critical Point Identification')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('phase_transition_critical_point.png', dpi=150)
    print(f"Critical point plot saved to: phase_transition_critical_point.png")
    
    print("\n" + "=" * 60)
    print("Analysis completed!")
    print("=" * 60)


def analyze_hubbard_phase_transition():
    """Analyze phase transition in Hubbard model using VMC"""
    
    print("\n" + "=" * 60)
    print("Hubbard Model Phase Transition (VMC)")
    print("=" * 60)
    
    L = 4
    t = 1.0
    U_values = np.linspace(1.0, 8.0, 10)
    
    energies = []
    double_occupancies = []
    
    for U in U_values:
        print(f"\nU/t = {U/t:.2f}...", end=" ", flush=True)
        
        # Create model
        model = HubbardModel(L=L, U=U, t=t, mu=0.0)
        
        # Create NQS
        nqs = RBMQuantumState(model.n_sites, hidden_dim=16)
        
        # Run VMC
        vmc = VMC(model, nqs, learning_rate=0.01)
        results = vmc.run(
            n_steps=3000,
            n_walkers=50,
            n_equil=500,
            verbose=False
        )
        
        # Compute double occupancy
        configs = results['configurations']
        double_occ = np.mean([model.compute_double_occupancy(c) for c in configs])
        
        energies.append(results['energy'])
        double_occupancies.append(double_occ)
        
        print(f"E={results['energy']:.3f}, D={double_occ:.3f}")
    
    # Plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(U_values / t, energies, 'o-', linewidth=2, markersize=6, color='blue')
    plt.xlabel('U/t')
    plt.ylabel('Energy')
    plt.title('Hubbard Model: Energy vs U/t')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(U_values / t, double_occupancies, 'o-', linewidth=2, markersize=6, color='red')
    plt.xlabel('U/t')
    plt.ylabel('Double Occupancy')
    plt.title('Hubbard Model: Double Occupancy vs U/t')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hubbard_phase_transition.png', dpi=150)
    print(f"\nHubbard phase transition plot saved to: hubbard_phase_transition.png")


if __name__ == "__main__":
    # Analyze Bose-Hubbard phase transition
    analyze_phase_transition()
    
    # Analyze Hubbard phase transition
    analyze_hubbard_phase_transition()
    
    print("\n" + "=" * 60)
    print("All analyses completed!")
    print("=" * 60)

