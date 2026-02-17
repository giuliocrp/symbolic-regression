# src/training/autoencoder_trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.models.autoencoder import ConvAutoencoder

def train_autoencoder(dataset, device, input_dim, latent_dim=2, batch_size=16, num_epochs=10, learning_rate=1e-3, save_path="autoencoder_weights.pth"):
    """
    Trains the ConvAutoencoder on the provided dataset.

    Parameters:
    -----------
    dataset : torch.utils.data.Dataset
        The training dataset.
    device : torch.device
        Device to train the model on.
    input_dim : int
        The size of the input dimension (Nx).
    latent_dim : int, optional
        The size of the latent dimension (default is 2).
    batch_size : int, optional
        Batch size for training (default is 16).
    num_epochs : int, optional
        Number of training epochs (default is 10).
    learning_rate : float, optional
        Learning rate for the optimizer (default is 1e-3).
    save_path : str, optional
        Path to save the trained model weights (default is "autoencoder_weights.pth").

    Returns:
    --------
    autoencoder : ConvAutoencoder
        The trained autoencoder model.
    """
    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    
    autoencoder = ConvAutoencoder(input_dim, latent_dim).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    loss_history = []
    
    print("Autoencoder training ...")

    for epoch in range(1, num_epochs + 1):
        autoencoder.train()
        running_loss = 0.0
        for x_in, _ in dataloader:
            x_in = x_in.to(device)
            optimizer.zero_grad()

            x_recon = autoencoder(x_in)
            loss = criterion(x_recon, x_in)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x_in.size(0)

        scheduler.step()

        epoch_loss = running_loss / len(dataloader.dataset)
        loss_history.append(epoch_loss)
        
        # Logging
        if (epoch == 1) or (epoch % 2 == 0):
            print(f"[Epoch {epoch}/{num_epochs}] Loss={epoch_loss:.6e}")

    torch.save(autoencoder.state_dict(), save_path)
    print(f"Saved autoencoder weights => {save_path}")

    return autoencoder, loss_history