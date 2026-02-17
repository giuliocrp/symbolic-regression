# src/models/autoencoder.py

import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self, input_dim=100, latent_dim=2):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2),  # Nx -> Nx/2
            nn.ELU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),  # Nx/2 -> Nx/4
            nn.ELU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2), # Nx/4 -> Nx/8
            nn.ELU(),
            nn.Flatten(),  # shape = [batch, 128*(input_dim//8)]
            nn.Linear(128*(input_dim//8), 128),
            nn.ELU(),
            nn.Linear(128, latent_dim),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ELU(),
            nn.Linear(128, 128*(input_dim//8)),
            nn.ELU(),
            nn.Unflatten(1, (128, input_dim//8)),  # shape => [batch, 128, Nx/8]
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2,
                               padding=2, output_padding=1), # Nx/8->Nx/4
            nn.ELU(),
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2,
                               padding=2, output_padding=1), # Nx/4->Nx/2
            nn.ELU(),
            nn.ConvTranspose1d(32, 1, kernel_size=5, stride=2,
                               padding=2, output_padding=1), # Nx/2->Nx
            # Final shape => [batch, 1, Nx]
        )

    def forward(self, x):
        z = self.encoder(x)     # [batch, latent_dim]
        x_recon = self.decoder(z)  # [batch, 1, Nx]
        return x_recon