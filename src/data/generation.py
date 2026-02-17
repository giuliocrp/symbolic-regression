# src/data/generation.py

import numpy as np
from src.pde.initial_conditions import gaussian_initial_condition
from src.pde.solver import advection_diffusion_step

def generate_advection_diffusion_data(mu0_list, sigma0_list, a, D, x, dx, dt, Nt, time_stride):
    """
    Generates advection-diffusion solutions for each (mu0, sigma0) pair.

    Params:
    -------
    mu0_list, sigma0_list: lists of initial-mean and initial-std-dev.
    a, D: scalar advection speed and diffusion coefficient.
    x: NumPy array of spatial grid points (length Nx).
    dx: grid spacing.
    dt: time-step.
    Nt: number of time steps.
    store_all_time: if True, store the solution at all Nt time steps;
                    otherwise store only the final snapshot.

    Returns:
    --------
    solutions: dict
      - key = f"mu0_{mu0}_sigma0_{sigma0}"
      - value = array of shape (Nt, Nx) if store_all_time=True,
                or shape (1, Nx) if store_all_time=False.
    """
    solutions = {}

    for mu0 in mu0_list:
        for sigma0 in sigma0_list:
            # Initial condition
            u = gaussian_initial_condition(x, mu0, sigma0)
            
            # We'll store snapshots in a list first, then convert to NumPy at the end.
            snapshots = []
            
            # Store initial snapshot at t=0
            snapshots.append(u.copy())

            # Time stepping
            for n in range(1, Nt):
                u = advection_diffusion_step(u, a, D, dx, dt)

                # Check if we are at a stride step or at the final step
                if n % time_stride == 0 or n == Nt - 1:
                    snapshots.append(u.copy())

            # Convert the list of snapshots to a NumPy array of shape (num_stored, Nx)
            u_stored = np.array(snapshots)
            
            key = f"mu0_{mu0}_sigma0_{sigma0}"
            solutions[key] = u_stored

    return solutions