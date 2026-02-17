# src/pde/solver.py

import numpy as np

def advection_diffusion_step(u, a, D, dx, dt):
    """
    Performs one time-step using:
      - Upwind scheme for advection (assuming a>0)
      - Centered difference for diffusion
      - Explicit Euler in time
    Dirichlet boundary conditions: u=0 at x=0 and x=L.
    """
    N = len(u)
    u_new = np.zeros_like(u)
    
    for i in range(1, N-1):
        # Upwind term (a>0):
        advection_term = -a * (u[i] - u[i-1]) / dx
        # Diffusion term:
        diffusion_term = D * (u[i+1] - 2*u[i] + u[i-1]) / dx**2
        u_new[i] = u[i] + dt*(advection_term + diffusion_term)
    
    # Dirichlet BCs
    u_new[0] = 0.0
    u_new[-1] = 0.0
    
    return u_new