"""
PyTorch Neural Architectures for Acoustic Anomaly Detection.
Includes:
1. Baseline Single-Branch Convolutional Autoencoder (Log-Mel).
2. Proposed Adaptive Dual-Branch Feature Fusion Autoencoder (Log-Mel + Inverse-Mel).
"""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DeconvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2, output_padding: int = 1):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            output_padding=output_padding,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.deconv(x)))


class BaselineConvolutionalAutoencoder(nn.Module):
    """
    Baseline Deep Convolutional Autoencoder for Acoustic Anomaly Detection.
    Operates on 2D Log-Mel spectrograms: (Batch, 1, n_mels, time_frames).
    Trained on normal operating sounds using Mean Squared Error reconstruction loss.
    """
    def __init__(self, in_channels: int = 1, latent_channels: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, 32, stride=2),   # 64 -> 32
            ConvBlock(32, 64, stride=2),           # 32 -> 16
            ConvBlock(64, latent_channels, stride=2) # 16 -> 8
        )
        self.decoder = nn.Sequential(
            DeconvBlock(latent_channels, 64, stride=2, output_padding=1),
            DeconvBlock(64, 32, stride=2, output_padding=1),
            nn.ConvTranspose2d(32, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            reconstructed (torch.Tensor): Shape matching input x.
            latent (torch.Tensor): Bottleneck latent representation.
        """
        orig_shape = x.shape
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        
        # Ensure exact spatial alignment with input in case of odd time dimensions
        if reconstructed.shape != orig_shape:
            reconstructed = F.interpolate(
                reconstructed,
                size=(orig_shape[2], orig_shape[3]),
                mode="bilinear",
                align_corners=False
            )
        return reconstructed, latent

    def compute_anomaly_score(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes per-sample anomaly score as Mean Squared Error over time and frequency.
        Returns:
            scores (torch.Tensor): 1D tensor of anomaly scores (Batch,).
            recon_error_map (torch.Tensor): 2D spatial error map (Batch, n_mels, time_frames).
        """
        self.eval()
        with torch.no_grad():
            reconstructed, _ = self.forward(x)
            error_map = (x - reconstructed) ** 2
            # Mean error across channels, frequency bins, and time frames
            scores = error_map.mean(dim=[1, 2, 3])
            return scores, error_map.squeeze(1)


class AdaptiveGatingNetwork(nn.Module):
    """
    Learns dynamic reliability weights alpha in [0, 1] conditioned on the latent representations
    of both Mel and Inverse-Mel acoustic branches.
    """
    def __init__(self, latent_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim * 2, latent_dim // 2)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(latent_dim // 2, 1)

    def forward(self, z_mel: torch.Tensor, z_inv: torch.Tensor) -> torch.Tensor:
        # Global average pool spatial dimensions to obtain latent descriptors
        gap_mel = F.adaptive_avg_pool2d(z_mel, (1, 1)).flatten(1)
        gap_inv = F.adaptive_avg_pool2d(z_inv, (1, 1)).flatten(1)
        
        joint = torch.cat([gap_mel, gap_inv], dim=1)
        hidden = self.act(self.fc1(joint))
        alpha = torch.sigmoid(self.fc2(hidden)) # shape (Batch, 1)
        return alpha


class ProposedAdaptiveFusionAutoencoder(nn.Module):
    """
    Proposed Research Model: Dual-Branch Adaptive Feature Fusion Autoencoder.
    
    Architecture:
    - Mel-spectrogram branch encoder and decoder.
    - Inverse-Mel complementary branch encoder and decoder.
    - Adaptive Gating Fusion module that assigns dynamic weights alpha to Mel
      and (1 - alpha) to Inverse-Mel based on latent noise and harmonic content.
    """
    def __init__(self, in_channels: int = 1, latent_channels: int = 128):
        super().__init__()
        # Branch 1: Mel encoder
        self.mel_encoder = nn.Sequential(
            ConvBlock(in_channels, 32, stride=2),
            ConvBlock(32, 64, stride=2),
            ConvBlock(64, latent_channels, stride=2)
        )
        # Branch 2: Inverse-Mel encoder
        self.inv_encoder = nn.Sequential(
            ConvBlock(in_channels, 32, stride=2),
            ConvBlock(32, 64, stride=2),
            ConvBlock(64, latent_channels, stride=2)
        )
        
        # Adaptive Fusion Gate
        self.gating = AdaptiveGatingNetwork(latent_dim=latent_channels)
        
        # Decoders for each branch
        self.mel_decoder = nn.Sequential(
            DeconvBlock(latent_channels, 64, stride=2, output_padding=1),
            DeconvBlock(64, 32, stride=2, output_padding=1),
            nn.ConvTranspose2d(32, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        self.inv_decoder = nn.Sequential(
            DeconvBlock(latent_channels, 64, stride=2, output_padding=1),
            DeconvBlock(64, 32, stride=2, output_padding=1),
            nn.ConvTranspose2d(32, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

    def forward(
        self, x_mel: torch.Tensor, x_inv: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        orig_shape = x_mel.shape
        z_mel = self.mel_encoder(x_mel)
        z_inv = self.inv_encoder(x_inv)
        
        # Compute dynamic gating coefficient alpha
        alpha = self.gating(z_mel, z_inv) # (Batch, 1)
        alpha_spatial = alpha.unsqueeze(-1).unsqueeze(-1) # (Batch, 1, 1, 1)
        
        # Adaptive latent fusion
        z_fused = alpha_spatial * z_mel + (1.0 - alpha_spatial) * z_inv
        
        recon_mel = self.mel_decoder(z_fused)
        recon_inv = self.inv_decoder(z_fused)
        
        if recon_mel.shape != orig_shape:
            recon_mel = F.interpolate(
                recon_mel,
                size=(orig_shape[2], orig_shape[3]),
                mode="bilinear",
                align_corners=False
            )
            recon_inv = F.interpolate(
                recon_inv,
                size=(orig_shape[2], orig_shape[3]),
                mode="bilinear",
                align_corners=False
            )
            
        return recon_mel, recon_inv, alpha.squeeze(1)

    def compute_anomaly_score(
        self, x_mel: torch.Tensor, x_inv: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes composite anomaly score weighted by the learned adaptive gate alpha:
        S(x) = alpha * MSE_mel + (1 - alpha) * MSE_inv
        """
        self.eval()
        with torch.no_grad():
            recon_mel, recon_inv, alpha = self.forward(x_mel, x_inv)
            err_mel = ((x_mel - recon_mel) ** 2).mean(dim=[1, 2, 3])
            err_inv = ((x_inv - recon_inv) ** 2).mean(dim=[1, 2, 3])
            
            # Weighted anomaly score
            scores = alpha * err_mel + (1.0 - alpha) * err_inv
            
            # Composite error map for UI display
            alpha_spatial = alpha.view(-1, 1, 1)
            err_map = alpha_spatial * ((x_mel - recon_mel) ** 2).squeeze(1) + \
                      (1.0 - alpha_spatial) * ((x_inv - recon_inv) ** 2).squeeze(1)
                      
            return scores, err_map, alpha
