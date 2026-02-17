# src/pde/initial_conditions.py

import numpy as np

def gaussian_initial_condition(x, mu0, sigma0):
    """
    Returns a 1D Gaussian with mean mu0 and standard deviation sigma0.
    """
    return (1.0 / np.sqrt(2 * np.pi * sigma0**2)) * np.exp(- (x - mu0)**2 / (2 * sigma0**2))
