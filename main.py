# scripts/main.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.data.generation import generate_advection_diffusion_data
from src.data.preprocessing import create_training_tensor
from src.models.autoencoder import ConvAutoencoder
from src.training.autoencoder_trainer import train_autoencoder
from src.models.symbolic_decoder import SymbolicDecoder
from src.training.symbolic_decoder_trainer import train_symbolic_decoder
from src.models.param_u import ParamU
from src.training.param_u_trainer import train_param_u
from src.visualization.plots import plot_reconstructions, plot_symbolic_reconstructions, autoencoder_loss, SINDy_loss


def main():
    # -------------------------
    # Device Selection: CUDA > MPS > CPU
    # -------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device.")
    else:
        device = torch.device("cpu")
        print("Using CPU.")
    
    # -------------------------
    # Simulation and Physical Parameters
    # -------------------------
    a = 1.0       # Advection speed (a > 0 for upwind scheme)
    D = 0.1       # Diffusion coefficient
    x_min, x_max = 0.0, 1.0
    Nx = 96
    x = np.linspace(x_min, x_max, Nx)
    dx = x[1] - x[0]
    
    T = 0.3     # Final time
    dt = 0.0001 # Time step (must satisfy the CFL conditions)
    Nt = int(T / dt)
    
    print("Simulation parameters:")
    print(f"Advection: a = {a}, Diffusion: D = {D}")
    print(f"Spatial domain: [{x_min}, {x_max}], Nx = {Nx}, dx = {dx:.5f}")
    print(f"Time: T = {T}, dt = {dt}, Nt = {Nt}")
    
    # Check approximate CFL conditions
    CFL_a = a * dt / dx
    CFL_D = D * dt / dx**2
    print("CFL_advection =", CFL_a, "| Should be <= 1")
    print("CFL_diffusion  =", CFL_D, "| Should be <= 0.5\n")
    
    # -------------------------
    # Define Parameter Sets for PDE Simulations
    # -------------------------
    mu0_list = [0.3, 0.5, 0.7]
    sigma0_list = [0.05, 0.1, 0.15]
    
    # -------------------------
    # Generate PDE Solutions
    # -------------------------
    time_stride = 5
    solutions_dict = generate_advection_diffusion_data(
        mu0_list, sigma0_list, a, D, x, dx, dt, Nt, time_stride,
    )
    print("Number of simulations stored:", len(solutions_dict))
    
    # -------------------------
    # Data Preparation: Create Dataset
    # -------------------------
    dataset = create_training_tensor(solutions_dict)
    print("Dataset shape (num_samples, 1, Nx):", dataset.tensors[0].shape)
    
    # -------------------------
    # Train the Autoencoder
    # -------------------------
    latent_dim = 2
    autoencoder_path = "autoencoder_weights.pth"

    if os.path.exists(autoencoder_path):
        # Load
        autoencoder = ConvAutoencoder(input_dim=Nx, latent_dim=2).to(device)
        autoencoder.load_state_dict(torch.load(autoencoder_path, map_location=device, weights_only=True))
        autoencoder.eval()
        print("Autoencoder loaded from disk.")
    else:
        # Train
        autoencoder, ae_loss_history = train_autoencoder(
            dataset=dataset,
            device=device,
            input_dim=Nx,
            latent_dim=latent_dim,
            batch_size=16,
            num_epochs=100,
            learning_rate=1e-3,
            save_path=autoencoder_path
        )
        print("Autoencoder trained from scratch and saved.")
    
    # -------------------------
    # Testing and Visualization
    # -------------------------
    plot_reconstructions(
        autoencoder=autoencoder,
        solutions_dict=solutions_dict,
        x=x,
        device=device,
        save_dir="autoencoder_plots"
    )

    autoencoder_loss(ae_loss_history, save_dir="loss_plots")

    encoder = autoencoder.encoder
    encoder.eval()
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
    latent_reps = []
    target_data = []

    with torch.no_grad():
        for batch_in, _ in dataloader:
            batch_in = batch_in.to(device)
            z = encoder(batch_in)  # Shape: (batch_size, 2)
            latent_reps.append(z.cpu())
            target_data.append(batch_in.cpu().squeeze(1))  # Shape: (batch_size, Nx)
    
    latent_reps = torch.cat(latent_reps, dim=0)  # Shape: (N_samples, 2)
    target_data = torch.cat(target_data, dim=0)  # Shape: (N_samples, Nx)
    
    print("Latent representations shape:", latent_reps.shape)
    print("Target data shape:", target_data.shape)
    
    # -------------------------
    # Initialize Symbolic Decoder
    # -------------------------
    library_size = 16
    symbolic_dec = SymbolicDecoder(latent_dim=2, output_dim=Nx, library_size=library_size).to(device)
    
    # -------------------------
    # Train Symbolic Decoder
    # -------------------------
    symbolic_decoder_path = "SINDy_weights.pth"
    symbolic_dec, SINDy_loss_history = train_symbolic_decoder(
        symbolic_dec,
        latent_reps,
        target_data,
        device=device,
        lr=1e-3,
        num_epochs=10000,
        alpha_l1=1e-4,
        threshold=1e-3,
        save_path=symbolic_decoder_path
    )

    SINDy_loss(SINDy_loss_history, save_dir="loss_plots")

    
    # -------------------------
    # Visualization: Reconstructed PDE Snapshots
    # -------------------------
    # Select a few samples to visualize
    num_samples_to_plot = 5
   
    num_samples_to_plot = 5
    with torch.no_grad():
        z_sample = latent_reps[:].to(device)      # for example, reconstruct all
        x_recon = symbolic_dec(z_sample).cpu()    # (N_samples, Nx)

    # Now call your new plotting function to handle the visualization:
    plot_symbolic_reconstructions(
        x_recon=x_recon, 
        x_true=target_data, 
        num_samples_to_plot=num_samples_to_plot,
        save_dir="SINDy_plots"
    )
    # -------------------------
    # Save the Symbolic Decoder Model
    # -------------------------
    torch.save(symbolic_dec.state_dict(), symbolic_decoder_path)
    print(f"[INFO] Symbolic Decoder model saved to '{symbolic_decoder_path}'.")

    key = "mu0_0.3_sigma0_0.05"
    pde_array = solutions_dict[key]  # shape=(N_snap, Nx)
    chosen_snapshot = pde_array[-1]

    x_t = torch.from_numpy(x).float().to(device)
    u_t = torch.from_numpy(chosen_snapshot).float().to(device)

    param_u_model = ParamU().to(device)

    train_param_u(
        model=param_u_model,
        x_data=x_t,
        y_data=u_t,
        lr=1e-2,
        n_epochs=2000,
        alpha_l1=1e-4,
        threshold=1e-3
    )

    # Evaluate final fit
    x_plot = torch.linspace(0, 1, 200).to(device)
    with torch.no_grad():
        u_pred = param_u_model(x_plot)

    # Convert to numpy for plotting
    x_plot_np = x_plot.cpu().numpy()
    u_pred_np = u_pred.cpu().numpy()

    # Plot
    plt.figure(figsize=(8,4))
    plt.plot(x, chosen_snapshot, 'bo-', label="PDE snapshot (original)", markersize=3)
    plt.plot(x_plot_np, u_pred_np, 'r--', label="ParamU fit")
    plt.title(f"ParamU Fit => PDE snapshot for {key}")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.grid(True)
    plt.legend()
    plt.savefig("param_u_snapshot_fit.png", dpi=150)
    plt.show()

    # Inspect learned parameters
    print("\n[INFO] Learned parameters from ParamU:")
    for name, p in param_u_model.named_parameters():
        print(f"{name} = {p.item():.6f}")

    
    # -------------------------
    # Extract and Display Symbolic Equations
    # -------------------------
    # For each output dimension, extract the symbolic equation based on the coefficients and library
    print("\nExtracted Symbolic Equations for Each Output Dimension:")
    for output_idx in range(Nx):
        coeffs = symbolic_dec.coefficients.data[output_idx].cpu().numpy()
        nonlinear_params = symbolic_dec.nonlinear_params.data.cpu().numpy()
        equation = ""
        for func_idx, (coeff, param) in enumerate(zip(coeffs, nonlinear_params)):
            if coeff == 0:
                continue
            func = symbolic_dec.library_functions[func_idx]
            if func_idx < len(symbolic_dec.library_functions):
                if func_idx == 0:
                    term = f"{coeff:.4f}"
                elif func_idx == 1:
                    term = f"{coeff:.4f}*z1"
                elif func_idx == 2:
                    term = f"{coeff:.4f}*z2"
                elif func_idx == 3:
                    term = f"{coeff:.4f}*z1^2"
                elif func_idx == 4:
                    term = f"{coeff:.4f}*z2^2"
                elif func_idx == 5:
                    term = f"{coeff:.4f}*sin({param:.4f}*z1)"
                elif func_idx == 6:
                    term = f"{coeff:.4f}*cos({param:.4f}*z2)"
                else:
                    term = f"{coeff:.4f}*func{func_idx}(z)"
                equation += term + " + "
        equation = equation.rstrip(" + ")
        print(f"Output {output_idx+1}: {equation}")
    
    print("\n[INFO] Symbolic equations extraction completed.")

if __name__ == "__main__":
    main()