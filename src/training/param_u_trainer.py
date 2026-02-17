# src/training/param_u_trainer.py

import torch
import torch.nn.functional as F
import torch.optim as optim

def train_param_u(model, x_data, y_data,
                  lr=1e-3, n_epochs=1000, alpha_l1=1e-4, threshold=1e-3):
    """
    Train a ParamU model with MSE + L1 regularization + thresholding.
    
    Args:
        model: an instance of ParamU
        x_data: torch.Tensor shape=(N_points,)
        y_data: torch.Tensor shape=(N_points,)
        lr (float): learning rate
        n_epochs (int): number of training epochs
        alpha_l1 (float): weight of L1 penalty
        threshold (float): threshold for zeroing small parameters
    Returns:
        loss_history (list of float): record of total losses
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_history = []

    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()

        y_pred = model(x_data)  # => shape=(N_points,)
        mse = F.mse_loss(y_pred, y_data)

        # L1 penalty on the *model's parameters*
        # Sum of absolute values
        l1_reg = 0.0
        for p in model.parameters():
            l1_reg += p.abs().sum()

        loss = mse + alpha_l1*l1_reg
        loss.backward()
        optimizer.step()

        # Threshold small parameters => param[|param|<threshold]=0
        with torch.no_grad():
            for p in model.parameters():
                mask = (p.abs() < threshold)
                p[mask] = 0.0

        loss_history.append(loss.item())
        
        # Optional debug print
        if epoch == 1 or epoch % 200 == 0:
            print(f"[Epoch {epoch}/{n_epochs}] Loss={loss.item():.6e}, MSE={mse.item():.6e}, L1={l1_reg.item():.6e}")

    return loss_history
