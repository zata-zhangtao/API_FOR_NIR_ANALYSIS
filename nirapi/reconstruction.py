"""
Spectral reconstruction utilities for NIR spectroscopy.

This module provides neural network-based and other methods for
reconstructing high-resolution spectra from lower-resolution measurements.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple
import os


__all__ = [
    'SpectralReconstructor',
    'spectral_reconstruction_train',
    'PD_reduce_noise'
]


class SpectralReconstructor(nn.Module):
    """
    Neural network for spectral reconstruction.
    
    This network takes photodiode (PD) measurements and reconstructs
    full spectral information using learned mappings.
    """
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list = None):
        """
        Initialize the spectral reconstructor network.
        
        Args:
            input_dim: Number of input PD channels
            output_dim: Number of output spectral points
            hidden_dims: List of hidden layer dimensions
        """
        super(SpectralReconstructor, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input PD measurements [batch_size, input_dim]
            
        Returns:
            Reconstructed spectra [batch_size, output_dim]
        """
        return self.network(x)


def spectral_reconstruction_train(
    PD_values: np.ndarray, 
    Spectra_values: np.ndarray, 
    epochs: int = 50, 
    lr: float = 1e-3,
    save_dir: Optional[str] = None
) -> SpectralReconstructor:
    """
    Train a neural network for spectral reconstruction.
    
    Args:
        PD_values: Photodiode measurements [n_samples, n_pd_channels]
        Spectra_values: Target spectra [n_samples, n_spectral_points]
        epochs: Number of training epochs
        lr: Learning rate
        save_dir: Directory to save trained model
        
    Returns:
        Trained SpectralReconstructor model
        
    Example:
        >>> pd_data = np.random.randn(100, 10)  # 100 samples, 10 PD channels
        >>> spectra_data = np.random.randn(100, 1000)  # 100 samples, 1000 wavelengths
        >>> model = spectral_reconstruction_train(pd_data, spectra_data)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(PD_values).to(device)
    y_tensor = torch.FloatTensor(Spectra_values).to(device)
    
    # Initialize model
    input_dim = PD_values.shape[1]
    output_dim = Spectra_values.shape[1]
    model = SpectralReconstructor(input_dim, output_dim).to(device)
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')
    
    # Save model if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, 'spectral_reconstructor.pth')
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    return model


def PD_reduce_noise(
    PD_samples: np.ndarray, 
    PD_noise: np.ndarray, 
    ratio: float = 9,
    base_noise: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Reduce noise in photodiode measurements.
    
    This function subtracts scaled noise measurements from sample measurements
    to improve signal quality.
    
    Args:
        PD_samples: Sample photodiode measurements
        PD_noise: Noise photodiode measurements  
        ratio: Scaling ratio for noise subtraction
        base_noise: Optional baseline noise to subtract
        
    Returns:
        Noise-reduced photodiode measurements
        
    Example:
        >>> samples = np.random.randn(100, 10) + 1.0  # Signal + noise
        >>> noise = np.random.randn(100, 10) * 0.1    # Pure noise
        >>> clean_samples = PD_reduce_noise(samples, noise, ratio=10)
    """
    # Scale noise according to ratio
    scaled_noise = PD_noise / ratio
    
    # Subtract scaled noise from samples
    denoised_samples = PD_samples - scaled_noise
    
    # Optionally subtract baseline noise
    if base_noise is not None:
        denoised_samples = denoised_samples - base_noise
    
    return denoised_samples