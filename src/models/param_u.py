# src/models/param_u.py

import torch
import torch.nn as nn
import math

class ParamU(nn.Module):
    """
    A small parametric model that directly fits a PDE snapshot
    u(x) ~ <some symbolic formula in x>
    with a handful of free parameters.
    """
    def __init__(self):
        super().__init__()
        # For illustration: t1 + t2*x + t3*x^2 + t4*sin(a*x) + t5*cos(b*x)
        # We'll have 7 trainable parameters:
        #   t1, t2, t3, t4, a, t5, b
        self.t1 = nn.Parameter(torch.zeros(1))
        self.t2 = nn.Parameter(torch.zeros(1))
        self.t3 = nn.Parameter(torch.zeros(1))
        self.t4 = nn.Parameter(torch.zeros(1))
        self.a  = nn.Parameter(torch.ones(1)*math.pi)  # freq for sin
        self.t5 = nn.Parameter(torch.zeros(1))
        self.b  = nn.Parameter(torch.ones(1)*math.pi)  # freq for cos

    def forward(self, x):
        """
        x: shape=(N_points,) containing spatial coordinates in [0,1].
        Returns a vector of shape=(N_points,) with the predicted u(x).
        """
        # Polynomials
        poly_part = self.t1 + self.t2*x + self.t3*x**2
        
        # Sine
        sin_part = self.t4 * torch.sin(self.a * x)

        # Cosine
        cos_part = self.t5 * torch.cos(self.b * x)

        return poly_part + sin_part + cos_part
