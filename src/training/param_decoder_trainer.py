# src/training/param_decoder_trainer.py

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from src.models.param_decoder import ParamDecoderMLP

def train_param_decoder(X_full, y_full, device, num_epochs=2000, batch_size=128, lr=1e-3):
    """
    Trains the ParamDecoderMLP on the dataset (x, mu) -> u.

    Parameters:
    -----------
    X_full : np.ndarray
        Input features of shape (num_samples, 2).
    y_full : np.ndarray
        Target values of shape (num_samples,).
    device : torch.device
        Device to train the model on (e.g., CPU, CUDA).
    num_epochs : int, optional
        Number of training epochs (default is 2000).
    batch_size : int, optional
        Batch size for training (default is 128).
    lr : float, optional
        Learning rate for the optimizer (default is 1e-3).

    Returns:
    --------
    model : ParamDecoderMLP
        The trained parameter decoder model.
    """
    # Create dataset & data loader
    dataset = TensorDataset(
        torch.from_numpy(X_full),
        torch.from_numpy(y_full)
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define model, loss, optimizer
    model = ParamDecoderMLP(hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for Xb, yb in dataloader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(Xb).squeeze(1)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * Xb.size(0)
        
        epoch_loss /= len(dataloader.dataset)

        # Print occasional progress
        if epoch % 500 == 0 or epoch == 1:
            print(f"[Epoch {epoch:4d}/{num_epochs}] Loss: {epoch_loss:.6e}")

    return model
