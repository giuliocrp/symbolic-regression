# src/models/symbolic_decoder.py

import torch
import torch.nn as nn
import numpy as np

class SymbolicDecoder(nn.Module):
    def __init__(self, latent_dim=2, output_dim=96, library_size=16):
        super(SymbolicDecoder, self).__init__()
        """
        Symbolic Decoder inspired by ADAM-SINDy.
        This decoder constructs a symbolic expression using a predefined library of candidate functions.
        
        Args:
            latent_dim (int): Dimension of the latent space.
            output_dim (int): Dimension of the output space (original data dimension).
            library_size (int): Number of candidate functions in the library.
        """
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.library_size = library_size
        
        # Initialize coefficients for the candidate functions -> shape=(output_dim, library_size)
        self.coefficients = nn.Parameter(torch.randn(output_dim, library_size) * 0.01)
        
        # Initialize nonlinear parameters (frequencies, exponents, etc.) -> shape=(library_size,)
        self.nonlinear_params = nn.Parameter(torch.randn(library_size) * 0.1)
        
        # Define the candidate library functions
        # For illustration, we'll use a combination of polynomials and trigonometric functions
        self.library_functions = [
            lambda z: torch.ones_like(z[:, 0]),                      # 0) Constant
            lambda z: z[:, 0],                                       # 1) Linear z1
            lambda z: z[:, 1],                                       # 2) Linear z2
            lambda z: z[:, 0] ** 2,                                  # 3) Quadratic z1^2
            lambda z: z[:, 1] ** 2,                                  # 4) Quadratic z2^2
            lambda z: torch.sin(self.nonlinear_params[0] * z[:, 0]), # 5) sin(a * z1)
            lambda z: torch.cos(self.nonlinear_params[1] * z[:, 1]), # 6) cos(b * z2)
            lambda z: z[:, 0] * z[:, 1],                             # 7) Interaction z1*z2
            lambda z: z[:, 0] ** 3,                                  # 8) Cubic z1^3
            lambda z: z[:, 1] ** 3,                                  # 9) Cubic z2^3
            lambda z: torch.exp(self.nonlinear_params[2] * z[:, 0]), # 10) exp(c * z1)
            lambda z: torch.exp(self.nonlinear_params[3] * z[:, 1]), # 11) exp(d * z2)
        ]
        
        # If library_size > number of explicitly defined functions, fill the remainder
        while len(self.library_functions) < self.library_size:
            # Ensure any additional placeholders are device-aware
            self.library_functions.append(lambda z: torch.zeros(z.size(0), device=z.device))

    def forward(self, z):
        """
        Forward pass to compute the symbolic reconstruction.
        
        Args:
            z (torch.Tensor): Latent representations, shape (batch_size, latent_dim)
        
        Returns:
            torch.Tensor: Reconstructed data, shape (batch_size, output_dim)
        """
        library_matrix = []
        for func in self.library_functions[:self.library_size]:
            library_matrix.append(func(z))

        # Shape of each element in library_matrix: (batch_size,)
        # Stack along dim=1 -> shape: (batch_size, library_size)
        Theta = torch.stack(library_matrix, dim=1)  # (batch_size, library_size)

        # Multiply with coefficients -> output shape: (batch_size, output_dim)
        x_recon = Theta @ self.coefficients.t()
        return x_recon
