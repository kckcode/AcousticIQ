"""
Model Training and Checkpoint Management for AcousticIQ.
Implements reproducible unsupervised training for Baseline and Proposed Autoencoders.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ml.models.autoencoder import BaselineConvolutionalAutoencoder, ProposedAdaptiveFusionAutoencoder
from ml.features.extractor import DualFeatureExtractor, FeatureConfig
from ml.preprocessing.audio import AudioPreprocessor, PreprocessingConfig


class AudioSpectrogramDataset(Dataset):
    """PyTorch Dataset for audio spectrogram pairs."""
    def __init__(self, mel_list: List[np.ndarray], inv_list: Optional[List[np.ndarray]] = None, labels: Optional[List[int]] = None):
        self.mel_list = mel_list
        self.inv_list = inv_list
        self.labels = labels

    def __len__(self):
        return len(self.mel_list)

    def __getitem__(self, idx):
        # Shape: (1, n_mels, time_frames)
        mel = torch.from_numpy(self.mel_list[idx]).unsqueeze(0)
        item = {"mel": mel}
        if self.inv_list is not None:
            item["inv"] = torch.from_numpy(self.inv_list[idx]).unsqueeze(0)
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


class ModelTrainer:
    def __init__(
        self,
        machine_category: str,
        checkpoint_dir: Path,
        device: Optional[str] = None,
        seed: int = 42,
    ):
        self.machine_category = machine_category
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.set_seed(seed)
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    @staticmethod
    def set_seed(seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def train_baseline(
        self,
        train_mels: List[np.ndarray],
        val_mels: List[np.ndarray],
        val_labels: List[int], # 0 = normal, 1 = anomaly
        epochs: int = 25,
        batch_size: int = 16,
        lr: float = 1e-3,
    ) -> Tuple[BaselineConvolutionalAutoencoder, Dict[str, Any]]:
        """
        Trains the Baseline Convolutional Autoencoder on normal samples.
        """
        model = BaselineConvolutionalAutoencoder().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.MSELoss()

        train_dataset = AudioSpectrogramDataset(train_mels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        history = {"train_loss": []}
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            for batch in train_loader:
                x = batch["mel"].to(self.device)
                optimizer.zero_grad()
                recon, _ = model(x)
                loss = criterion(recon, x)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(x)

            epoch_loss /= len(train_dataset)
            history["train_loss"].append(round(epoch_loss, 6))

        training_duration = time.time() - start_time

        # Validation & threshold determination
        model.eval()
        val_scores = []
        with torch.no_grad():
            for mel in val_mels:
                x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(self.device)
                scores, _ = model.compute_anomaly_score(x)
                val_scores.append(float(scores.cpu().item()))

        # Threshold calibrated on normal validation samples (95th percentile)
        val_normal_scores = [s for s, lab in zip(val_scores, val_labels) if lab == 0]
        if not val_normal_scores:
            val_normal_scores = val_scores
        threshold = float(np.percentile(val_normal_scores, 95.0))

        # Checkpoint metadata
        ckpt_filename = f"{self.machine_category}_baseline_ae.pt"
        ckpt_path = self.checkpoint_dir / ckpt_filename
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_type": "baseline",
            "architecture": "BaselineConvolutionalAutoencoder",
            "machine_category": self.machine_category,
            "threshold": threshold,
            "seed": self.seed,
            "epochs": epochs,
            "training_duration_seconds": round(training_duration, 2),
            "history": history,
        }, ckpt_path)

        metadata = {
            "model_type": "baseline",
            "architecture": "BaselineConvolutionalAutoencoder",
            "machine_category": self.machine_category,
            "checkpoint_path": str(ckpt_path),
            "threshold": threshold,
            "final_train_loss": history["train_loss"][-1],
            "training_duration_seconds": round(training_duration, 2),
            "history": history,
            "parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }

        # Save sidecar metadata JSON
        meta_json_path = self.checkpoint_dir / f"{self.machine_category}_baseline_meta.json"
        with open(meta_json_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return model, metadata

    def train_proposed(
        self,
        train_mels: List[np.ndarray],
        train_invs: List[np.ndarray],
        val_mels: List[np.ndarray],
        val_invs: List[np.ndarray],
        val_labels: List[int],
        epochs: int = 25,
        batch_size: int = 16,
        lr: float = 1e-3,
    ) -> Tuple[ProposedAdaptiveFusionAutoencoder, Dict[str, Any]]:
        """
        Trains the Proposed Dual-Branch Adaptive Fusion Autoencoder.
        """
        model = ProposedAdaptiveFusionAutoencoder().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

        train_dataset = AudioSpectrogramDataset(train_mels, train_invs)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        history = {"train_loss": [], "mean_alpha": []}
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            alphas = []
            for batch in train_loader:
                x_mel = batch["mel"].to(self.device)
                x_inv = batch["inv"].to(self.device)
                optimizer.zero_grad()

                recon_mel, recon_inv, alpha = model(x_mel, x_inv)
                
                # Dynamic loss formulation
                err_mel = ((x_mel - recon_mel) ** 2).mean(dim=[1, 2, 3])
                err_inv = ((x_inv - recon_inv) ** 2).mean(dim=[1, 2, 3])
                batch_loss = (alpha * err_mel + (1.0 - alpha) * err_inv).mean()

                batch_loss.backward()
                optimizer.step()

                epoch_loss += batch_loss.item() * len(x_mel)
                alphas.extend(alpha.detach().cpu().numpy().tolist())

            epoch_loss /= len(train_dataset)
            history["train_loss"].append(round(epoch_loss, 6))
            history["mean_alpha"].append(round(float(np.mean(alphas)), 4))

        training_duration = time.time() - start_time

        # Validation & threshold determination
        model.eval()
        val_scores = []
        with torch.no_grad():
            for mel, inv in zip(val_mels, val_invs):
                x_m = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(self.device)
                x_i = torch.from_numpy(inv).unsqueeze(0).unsqueeze(0).to(self.device)
                scores, _, _ = model.compute_anomaly_score(x_m, x_i)
                val_scores.append(float(scores.cpu().item()))

        val_normal_scores = [s for s, lab in zip(val_scores, val_labels) if lab == 0]
        if not val_normal_scores:
            val_normal_scores = val_scores
        threshold = float(np.percentile(val_normal_scores, 95.0))

        # Checkpoint metadata
        ckpt_filename = f"{self.machine_category}_proposed_fusion.pt"
        ckpt_path = self.checkpoint_dir / ckpt_filename
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_type": "proposed",
            "architecture": "ProposedAdaptiveFusionAutoencoder",
            "machine_category": self.machine_category,
            "threshold": threshold,
            "seed": self.seed,
            "epochs": epochs,
            "training_duration_seconds": round(training_duration, 2),
            "history": history,
        }, ckpt_path)

        metadata = {
            "model_type": "proposed",
            "architecture": "ProposedAdaptiveFusionAutoencoder",
            "machine_category": self.machine_category,
            "checkpoint_path": str(ckpt_path),
            "threshold": threshold,
            "final_train_loss": history["train_loss"][-1],
            "final_mean_alpha": history["mean_alpha"][-1],
            "training_duration_seconds": round(training_duration, 2),
            "history": history,
            "parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }

        meta_json_path = self.checkpoint_dir / f"{self.machine_category}_proposed_meta.json"
        with open(meta_json_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return model, metadata
