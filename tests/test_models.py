"""
Unit tests for Machine Learning Models (Baseline & Proposed Fusion).
"""

import torch
from ml.models.autoencoder import (
    BaselineConvolutionalAutoencoder,
    ProposedAdaptiveFusionAutoencoder,
    AdaptiveGatingNetwork,
)


def test_baseline_autoencoder_forward():
    model = BaselineConvolutionalAutoencoder()
    model.eval()

    # Batch of 2, 1 channel, 64 mel bins, 128 time frames
    x = torch.randn(2, 1, 64, 128)
    recon, latent = model(x)

    assert recon.shape == x.shape
    assert latent.shape[1] == 128 # latent channels


def test_baseline_anomaly_score_computation():
    model = BaselineConvolutionalAutoencoder()
    x = torch.randn(4, 1, 64, 128)
    scores, error_map = model.compute_anomaly_score(x)

    assert scores.shape == (4,)
    assert (scores >= 0).all()
    assert error_map.shape == (4, 64, 128)


def test_proposed_fusion_autoencoder_forward():
    model = ProposedAdaptiveFusionAutoencoder()
    model.eval()

    x_mel = torch.randn(2, 1, 64, 128)
    x_inv = torch.randn(2, 1, 64, 128)

    recon_mel, recon_inv, alpha = model(x_mel, x_inv)

    assert recon_mel.shape == x_mel.shape
    assert recon_inv.shape == x_inv.shape
    assert alpha.shape == (2,)
    # Alpha must be in [0, 1] due to sigmoid
    assert (alpha >= 0.0).all() and (alpha <= 1.0).all()


def test_proposed_anomaly_score_computation():
    model = ProposedAdaptiveFusionAutoencoder()
    x_mel = torch.randn(3, 1, 64, 128)
    x_inv = torch.randn(3, 1, 64, 128)

    scores, error_map, alpha = model.compute_anomaly_score(x_mel, x_inv)

    assert scores.shape == (3,)
    assert (scores >= 0).all()
    assert error_map.shape == (3, 64, 128)
    assert alpha.shape == (3,)
