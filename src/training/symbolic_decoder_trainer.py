# src/training/symbolic_decoder_trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from src.models.symbolic_decoder import SymbolicDecoder

def train_symbolic_decoder(symbolic_dec, latent_data, target_data, 
                           device, lr=1e-3, num_epochs=5000, 
                           alpha_l1=1e-4, threshold=1e-3, save_path="SINDy_weights.pth"):
    """
    Trains the SymbolicDecoder using ADAM optimizer with L1 regularization.
    
    Args:
        symbolic_dec (SymbolicDecoder): The symbolic decoder model.
        latent_data (torch.Tensor): Latent representations, shape (N_samples, latent_dim)
        target_data (torch.Tensor): Original data, shape (N_samples, output_dim)
        device (str): Device to train on ('cpu' or 'cuda').
        lr (float): Learning rate for the optimizer.
        num_epochs (int): Number of training epochs.
        alpha_l1 (float): Weight for L1 regularization.
        threshold (float): Threshold for coefficient pruning.
        
    Returns:
        SymbolicDecoder: Trained symbolic decoder.
        list: Loss history.
    """
    symbolic_dec.to(device)
    symbolic_dec.train()
    
    optimizer = optim.Adam(symbolic_dec.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    loss_history = []

    print("SINDy training ...")
    
    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()

        x_recon = symbolic_dec(latent_data.to(device))
        mse_loss = criterion(x_recon, target_data.to(device))

        # L1 on both symbolic_dec.coefficients + symbolic_dec.nonlinear_params
        l1_loss = symbolic_dec.coefficients.abs().sum() + symbolic_dec.nonlinear_params.abs().sum()
        loss = mse_loss + alpha_l1 * l1_loss

        loss.backward()
        optimizer.step()

        # Thresholding small coefficients for SINDy-like sparsity
        with torch.no_grad():
            symbolic_dec.coefficients.abs_().masked_fill_(symbolic_dec.coefficients.abs()<threshold, 0.0)
            symbolic_dec.nonlinear_params.abs_().masked_fill_(symbolic_dec.nonlinear_params.abs()<threshold, 0.0)

        loss_history.append(loss.item())

        # Logging
        if epoch == 1 or epoch % 500 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.6f}, "
                  f"MSE: {mse_loss.item():.6f}, L1: {l1_loss.item():.6f}")
            
    torch.save(symbolic_dec.state_dict(), save_path)
    print(f"Saved symbolic decoder weights => {save_path}")

    return symbolic_dec, loss_history
