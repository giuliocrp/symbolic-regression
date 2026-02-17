# src/data/preprocessing.py

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from src.models.autoencoder import ConvAutoencoder

def create_training_tensor(solutions_dict):
    """
    Converts PDE solution dictionary into a single Torch tensor
    of shape [num_samples, 1, Nx].
    Here we only take the stored snapshot(s).
    """
    X_list = []
    for key, arr in solutions_dict.items():
        # arr shape can be either (Nt, Nx) or (1, Nx), depending on store_all_time
        # If store_all_time=False, arr.shape = (1, Nx)
        # We'll flatten across the first dimension (Nt or 1).
        for i in range(arr.shape[0]):
            snapshot = arr[i, :]  # shape (Nx,)
            X_list.append(snapshot)
    
    X_np = np.array(X_list)  # shape (num_samples, Nx)
    X_tensor = torch.from_numpy(X_np).float().unsqueeze(1)  # -> [num_samples, 1, Nx]
    
    return TensorDataset(X_tensor, X_tensor)

def encode_snapshot(snapshot_1d, encoder):
    """
    snapshot_1d: NumPy array of shape (Nx,)
    encoder: the encoder part of the autoencoder (nn.Module)

    Returns:
    --------
    z: a NumPy array of shape (latent_dim,) containing the latent code
    """
    # Convert to torch tensor of shape [batch=1, channels=1, Nx]
    ten_in = torch.from_numpy(snapshot_1d).float().unsqueeze(0).unsqueeze(0)

    # 2) Move the input to the same device as the encoder
    device = next(encoder.parameters()).device
    ten_in = ten_in.to(device)
    
    with torch.no_grad():
        z_out = encoder(ten_in)  # shape => [1, latent_dim]
    return z_out.squeeze().cpu().numpy()  # shape => (latent_dim,)

def build_parameterized_dataset(solutions_dict, encoder, x_grid, time_idx=-1, latent_dim_idx=0):
    """
    Builds a dataset of (x, mu) -> u(x, mu) using the autoencoder to extract mu.

    solutions_dict: dict of PDE solutions
        - key = e.g. "mu0_0.3_sigma0_0.1"
        - val = array shape (Nt, Nx)
    encoder: the trained encoder (nn.Module)
    x_grid: 1D NumPy array of length Nx (the spatial domain)
    time_idx: which time index to pick (-1 = final time)
    latent_dim_idx: which dimension of the latent code is interpreted as 'mu'

    Returns:
    --------
    X_full: numpy array of shape (num_samples, 2), columns = [x, mu]
    y_full: numpy array of shape (num_samples,)
    """
    X_list = []
    y_list = []

    for key, arr in solutions_dict.items():
        # arr: shape (Nt, Nx)
        snapshot = arr[time_idx]  # pick the desired time slice

        # Encode to get latent code z
        z = encode_snapshot(snapshot, encoder)  # shape = (latent_dim,)

        # Let's interpret z[latent_dim_idx] as 'mu'
        mu_val = z[latent_dim_idx]

        # Now for each spatial point x_i, we form (x_i, mu_val) => u_i
        for i, x_val in enumerate(x_grid):
            X_list.append([x_val, mu_val])
            y_list.append(snapshot[i])

    X_full = np.array(X_list, dtype=np.float32)
    y_full = np.array(y_list, dtype=np.float32)
    return X_full, y_full
