# src/visualization/plots.py

import matplotlib.pyplot as plt
import os
import torch
import numpy as np

from src.data.preprocessing import build_parameterized_dataset, encode_snapshot
from src.models.param_decoder import ParamDecoderMLP

def plot_reconstructions(autoencoder, solutions_dict, x, device, save_dir="autoencoder_reconstructions_plots"):
    autoencoder.eval()
    keys_sorted = sorted(solutions_dict.keys(), key=lambda k: (float(k.split('_')[1]), float(k.split('_')[3])))
    
    os.makedirs(save_dir, exist_ok=True)
    
    for key in keys_sorted:
        sim_data = solutions_dict[key]  # shape: (Nt, Nx)
    
        time_idx_0 = 0
        time_idx_mid = int((sim_data.shape[0] - 1) / 2)
        time_idx_end = sim_data.shape[0] - 1  
        
        parts = key.split('_')
        mu0 = parts[1]
        sigma0 = parts[3]
        
        snapshots = [sim_data[time_idx_0, :],
                    sim_data[time_idx_mid, :],
                    sim_data[time_idx_end, :]]
        titles = ["t = 0", "t = T/2", "t = T"]
        reconstructions = []
        
        with torch.no_grad():
            for snap in snapshots:
                snap_tensor = torch.from_numpy(snap).float().unsqueeze(0).unsqueeze(0).to(device)
                recon = autoencoder(snap_tensor).cpu().numpy()[0, 0, :]
                reconstructions.append(recon)
        
        fig, axes = plt.subplots(ncols=3, figsize=(15, 4), sharex=True, sharey=True)
        
        for ax, original, recon, title in zip(axes, snapshots, reconstructions, titles):
            ax.plot(x, original, color='blue', label='Original')
            ax.plot(x, recon, '--', color='red', label='Reconstructed')
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("u(x, t)" if ax == axes[0] else "")
            ax.legend()
        
        fig.suptitle(f"Simulation: mu0 = {mu0}, sigma0 = {sigma0}", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        output_path = os.path.join(save_dir, f"sim_{mu0}_{sigma0}.png")
        plt.savefig(output_path)
        plt.close(fig)
    
    print(f"\nAll plots have been saved in the '{save_dir}' folder.\n")


def plot_symbolic_reconstructions(
    x_recon, x_true, 
    num_samples_to_plot=5, 
    save_dir="sindy_reconstructions_plots"
):
    """
    Plots and saves a comparison of reconstructed PDE snapshots vs. ground truth.

    Args:
        x_recon (torch.Tensor or np.ndarray): Reconstructed PDE snapshots, shape (N_samples, Nx).
        x_true  (torch.Tensor or np.ndarray): Ground-truth PDE snapshots, shape (N_samples, Nx).
        num_samples_to_plot (int): Number of random samples to visualize.
        save_dir (str): Directory to store the generated plots.
    """

    # Create folder if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to numpy if using torch tensors
    if isinstance(x_recon, torch.Tensor):
        x_recon = x_recon.detach().cpu().numpy()
    if isinstance(x_true, torch.Tensor):
        x_true = x_true.detach().cpu().numpy()

    # Randomly select 'num_samples_to_plot' from range [0, N_samples)
    N_samples = x_true.shape[0]
    Nx = x_true.shape[1]
    indices = np.random.choice(N_samples, num_samples_to_plot, replace=False)

    # Plot each chosen sample
    for i, idx in enumerate(indices, start=1):
        plt.figure(figsize=(10, 4))
        plt.plot(x_true[idx], label="Original PDE Snapshot", color="blue")
        plt.plot(x_recon[idx], label="Symbolic Decoder Reconstruction", color="red", linestyle="--")
        plt.title(f"Sample {i} Reconstruction (index={idx})")
        plt.xlabel("Grid Index")
        plt.ylabel("u(x)")
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        filename = f"reconstruction_sample_{i}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=150)
        plt.close()
        print(f"[INFO] Reconstruction plot saved as '{filepath}'.")



def plot_decoder_results(model_param, encoder, x_grid, solutions_dict, device, time_idx=-1):
    """
    For each PDE solution in solutions_dict, compare:
      1) The final-time PDE snapshot,
      2) The decoder MLP's approximation.

    model_param: the trained MLP, mapping (x, mu) -> u
    encoder: the autoencoder's encoder
    x_grid: 1D array of shape (Nx,)
    solutions_dict: PDE solutions, shape = (Nt, Nx) each
    device: torch.device
    time_idx: which time index to evaluate (default = -1 => final time)
    """
    model_param.eval()

    # Create a folder for param-decoder plots
    os.makedirs("decoder_plots", exist_ok=True)

    for key, arr in solutions_dict.items():
        # PDE snapshot at time_idx => shape (Nx,)
        snapshot = arr[time_idx]

        # Encode to get latent code => shape (latent_dim,)
        # We interpret z[0] as "mu" below. If you need z[1] for sigma, adapt as needed.
        z = encode_snapshot(snapshot, encoder)  # returns np array, shape (latent_dim,)
        mu_val = z[0]

        # Build the input for the MLP: (x, mu) for each x in x_grid
        # shape => [Nx, 2]
        X_test = []
        for i, x_val in enumerate(x_grid):
            X_test.append([x_val, mu_val])
        X_test = np.array(X_test, dtype=np.float32)
        
        # Convert to torch and run forward pass
        X_torch = torch.from_numpy(X_test).to(device)  # shape => [Nx, 2]
        with torch.no_grad():
            u_pred = model_param(X_torch).cpu().numpy().squeeze()  # shape => (Nx,)

        # Now we have PDE snapshot "snapshot" and decoder approx "u_pred"
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x_grid, snapshot, 'b-', label='PDE snapshot')
        ax.plot(x_grid, u_pred, 'r--', label='Decoder MLP')
        ax.set_xlabel('x')
        ax.set_ylabel('u(x)')
        ax.set_title(f"Simulation = {key} | mu ~ {mu_val:.3f}")
        ax.legend()
        plt.tight_layout()

        # Save figure
        outname = f"decoder_plots/compare_{key}.png"
        plt.savefig(outname)
        plt.close(fig)

    print("Decoder comparison plots saved in 'decoder_plots' folder.")


def autoencoder_loss(ae_loss_history, save_dir="loss_plots"):
    # Ensure output directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Create the figure
    plt.figure()
    plt.plot(ae_loss_history, label='Autoencoder Training Loss', color='blue')
    plt.title('Autoencoder Loss vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    filename = os.path.join(save_dir, "autoencoder_loss.png")
    plt.savefig(filename, dpi=150)
    plt.close()  # Close figure to free memory
    print(f"[INFO] Autoencoder loss plot saved to '{filename}'.")


def SINDy_loss(symdec_loss_history, save_dir="loss_plots"):
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure()
    plt.plot(symdec_loss_history, label='Symbolic Decoder Training Loss', color='red')
    plt.title('Symbolic Decoder Loss vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    filename = os.path.join(save_dir, "symbolic_decoder_loss.png")
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"[INFO] Symbolic Decoder loss plot saved to '{filename}'.")


def plot_evolution_over_time(solutions_dict, x, mu0=0.3, sigma0=0.05, max_plots=6, time_stride=5, dt=0.0001):
    """
    Plots the evolution over time (for a few time snapshots) of the PDE solution
    corresponding to a specific mu0 and sigma0.

    Args:
        solutions_dict (dict): Dictionary returned by generate_advection_diffusion_data.
            The keys are strings like "mu0_0.3_sigma0_0.05" and values are arrays of shape (num_stored, Nx).
        x (numpy.ndarray): The spatial grid (length Nx).
        mu0 (float): The chosen initial mean to plot.
        sigma0 (float): The chosen initial std. dev. to plot.
        max_plots (int): Maximum number of time snapshots to overlay in the same plot 
                         (to avoid cluttering).
    """
    # Build the dictionary key
    key = f"mu0_{mu0}_sigma0_{sigma0}"

    if key not in solutions_dict:
        raise KeyError(f"Key '{key}' not found in the solutions dictionary.")

    # Extract the stored snapshots
    # shape: (num_stored, Nx)
    snapshots = solutions_dict[key]
    num_stored = snapshots.shape[0]

    # Decide which snapshots to plot (equally spaced in the stored data)
    # e.g., if max_plots=6 and num_stored=20 => we pick 6 snapshots evenly spaced in range(num_stored)
    indices_to_plot = np.linspace(0, num_stored - 1, min(num_stored, max_plots), dtype=int)

    for idx in indices_to_plot:
        # Convert "stored index" => PDE time step => actual time
        pde_step = idx * time_stride  # PDE step number
        t_val = pde_step * dt         # physical time

        plt.plot(
            x, 
            snapshots[idx], 
            label=f"t = {t_val:.5f}"
        )

    plt.title(f"Evolution Over Time for mu0={mu0}, sigma0={sigma0}")
    plt.xlabel("x")
    plt.ylabel("u(x, t)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("pde_evolution_mu0_03_sigma0_005.png", dpi=150)
    plt.close()