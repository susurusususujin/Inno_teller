"""Models module for InnoWhitespaceExtractor.

This module contains the PyTorch model definitions for the Autoencoder and Variational Autoencoder (VAE)
used for dimensionality reduction, as well as the EarlyStopping utility for training.
"""

import numpy as np
import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Encoder network for the Autoencoder."""

    def __init__(self, input_dim: int = 1024) -> None:
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, 2),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the encoder.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Encoded tensor of shape (batch_size, 2).
        """
        return self.encoder(x)


class Decoder(nn.Module):
    """Decoder network for the Autoencoder and VAE."""

    def __init__(self, output_dim: int = 1024) -> None:
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 16),
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, 1024),
            nn.Tanh(),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the decoder.

        Args:
            x: Input tensor of shape (batch_size, 2).

        Returns:
            Decoded tensor of shape (batch_size, output_dim), normalized to unit length.
        """
        x = self.decoder(x)
        return torch.nn.functional.normalize(x, p=2, dim=1)


class Autoencoder(nn.Module):
    """Full Autoencoder model combining Encoder and Decoder."""

    def __init__(self, embedding_dim: int = 1024) -> None:
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim=embedding_dim)
        self.decoder = Decoder(output_dim=embedding_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the Autoencoder.

        Args:
            x: Input tensor of shape (batch_size, embedding_dim).

        Returns:
            A tuple containing:
                - decoded: Reconstructed tensor of shape (batch_size, embedding_dim).
                - encoded: Latent representation of shape (batch_size, 2).
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded




class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience: int = 30, verbose: bool = False, delta: float = 0,
                 path: str = 'checkpoint.pt', trace_func=print) -> None:
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss amendment.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
            trace_func (function): Trace print function.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: nn.Module) -> None:
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss