"""
Utility functions for QMC calculations
"""

import numpy as np
from typing import Tuple


def metropolis_acceptance(ratio: float) -> bool:
    """
    Metropolis acceptance criterion
    
    Args:
        ratio: Acceptance ratio (probability ratio)
        
    Returns:
        True if accepted, False otherwise
    """
    if ratio >= 1.0:
        return True
    return np.random.random() < ratio


def compute_correlation(config: np.ndarray, i: int, j: int) -> float:
    """
    Compute correlation function between sites i and j
    
    Args:
        config: Configuration array
        i: Site index i
        j: Site index j
        
    Returns:
        Correlation value
    """
    return config[i] * config[j]


def compute_structure_factor(config: np.ndarray, q: np.ndarray, 
                             lattice_coords: np.ndarray) -> complex:
    """
    Compute structure factor S(q)
    
    Args:
        config: Configuration array
        q: Momentum vector
        lattice_coords: Lattice coordinates (n_sites x dimension)
        
    Returns:
        Structure factor (complex)
    """
    n_sites = len(config)
    phase = np.exp(1j * np.dot(lattice_coords, q))
    return np.abs(np.sum(config * phase))**2 / n_sites


def compute_order_parameter(config: np.ndarray, order_type: str = "staggered") -> float:
    """
    Compute order parameter
    
    Args:
        config: Configuration array
        order_type: Type of order ("staggered", "ferromagnetic", etc.)
        
    Returns:
        Order parameter value
    """
    if order_type == "staggered":
        # Staggered magnetization
        n_sites = len(config)
        sign = np.array([(-1)**i for i in range(n_sites)])
        return np.abs(np.sum(config * sign)) / n_sites
    elif order_type == "ferromagnetic":
        return np.abs(np.sum(config)) / len(config)
    else:
        raise ValueError(f"Unknown order type: {order_type}")


def bootstrap_error(data: np.ndarray, n_bootstrap: int = 1000) -> Tuple[float, float]:
    """
    Compute bootstrap error estimate
    
    Args:
        data: Data array
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        (mean, error) tuple
    """
    n = len(data)
    means = []
    for _ in range(n_bootstrap):
        indices = np.random.randint(0, n, size=n)
        means.append(np.mean(data[indices]))
    means = np.array(means)
    return np.mean(means), np.std(means)


def autocorrelation_time(data: np.ndarray, max_lag: int = 100) -> float:
    """
    Estimate autocorrelation time
    
    Args:
        data: Time series data
        max_lag: Maximum lag to consider
        
    Returns:
        Autocorrelation time
    """
    n = len(data)
    mean = np.mean(data)
    data_centered = data - mean
    
    autocorr = []
    for lag in range(1, min(max_lag, n // 2)):
        corr = np.mean(data_centered[:-lag] * data_centered[lag:])
        autocorr.append(corr)
    
    autocorr = np.array(autocorr)
    autocorr = autocorr / autocorr[0] if len(autocorr) > 0 and autocorr[0] != 0 else autocorr
    
    # Find where autocorrelation drops to 1/e
    tau = 1.0
    for i, c in enumerate(autocorr):
        if c < 1.0 / np.e:
            tau = i + 1
            break
    
    return tau

