"""
Utilities for denoising multi-channel traces with a residual 1D CNN or PCA
projection. Traces are expected in channel-first format:
    shape = (batch, n_channels, n_samples)
"""

from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset


class TraceDenoiseDataset(Dataset):
    """
    Lightweight Dataset for traces shaped (N, C, T). Optionally normalizes each
    sample by its per-channel standard deviation to stabilize training.
    """

    def __init__(
        self,
        noisy_traces: np.ndarray,
        clean_traces: Optional[np.ndarray] = None,
        normalize: bool = False,
    ) -> None:
        noisy = np.asarray(noisy_traces)
        clean = noisy if clean_traces is None else np.asarray(clean_traces)

        if noisy.shape != clean.shape:
            raise ValueError(f"noisy and clean traces must have same shape, got {noisy.shape} vs {clean.shape}")

        if noisy.ndim == 2:
            noisy = noisy[:, None, :]
            clean = clean[:, None, :]
        elif noisy.ndim != 3:
            raise ValueError(f"Expected traces with 2 or 3 dims, got {noisy.ndim}")

        self.x = torch.as_tensor(noisy, dtype=torch.float32)
        self.y = torch.as_tensor(clean, dtype=torch.float32)

        self._scale: Optional[torch.Tensor] = None
        if normalize:
            # per-sample, per-channel std prevents division by zero
            std = self.x.std(dim=-1, keepdim=True).clamp(min=1e-6)
            self.x = self.x / std
            self.y = self.y / std
            self._scale = std

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]

    def denormalize(self, preds: torch.Tensor, idxs: Optional[slice] = None) -> torch.Tensor:
        if self._scale is None:
            return preds
        scale = self._scale if idxs is None else self._scale[idxs]
        return preds * scale.to(preds.device)


class CNNDenoiser1DResidual(nn.Module):
    """
    U-Net style 1D CNN with skips and residual output: learns a noise estimate
    and adds it back to the input, which helps preserve signal amplitude.
    """

    def __init__(self, in_channels: int = 1):
        super().__init__()

        def block(cin, cout):
            gn_groups = max(1, min(8, cout // 2))
            return nn.Sequential(
                nn.Conv1d(cin, cout, kernel_size=5, stride=1, padding=2),
                nn.GroupNorm(gn_groups, cout),
                nn.ReLU(inplace=True),
            )

        self.enc1 = block(in_channels, 32)
        self.enc2 = block(32, 64)
        self.down1 = nn.Conv1d(64, 64, kernel_size=4, stride=2, padding=1)
        self.enc3 = block(64, 128)
        self.down2 = nn.Conv1d(128, 128, kernel_size=4, stride=2, padding=1)

        self.bottleneck = block(128, 128)

        self.up2 = nn.ConvTranspose1d(128, 128, kernel_size=4, stride=2, padding=1)
        self.dec2 = block(128 + 128, 64)
        self.up1 = nn.ConvTranspose1d(64, 64, kernel_size=4, stride=2, padding=1)
        self.dec1 = block(64 + 64, 32)
        self.out_conv = nn.Conv1d(32, in_channels, kernel_size=5, padding=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_len = x.shape[-1]

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        d1 = self.down1(e2)
        e3 = self.enc3(d1)
        d2 = self.down2(e3)

        b = self.bottleneck(d2)

        u2 = self.up2(b)
        if u2.shape[-1] != e3.shape[-1]:
            u2 = F.interpolate(u2, size=e3.shape[-1], mode="linear", align_corners=False)
        dcat2 = torch.cat([u2, e3], dim=1)
        d2_out = self.dec2(dcat2)

        u1 = self.up1(d2_out)
        if u1.shape[-1] != e2.shape[-1]:
            u1 = F.interpolate(u1, size=e2.shape[-1], mode="linear", align_corners=False)
        dcat1 = torch.cat([u1, e2], dim=1)
        d1_out = self.dec1(dcat1)

        res = self.out_conv(d1_out)
        if res.shape[-1] != original_len:
            res = F.interpolate(res, size=original_len, mode="linear", align_corners=False)

        return x + res


def train_cnn_denoiser(
    noisy_traces: np.ndarray,
    clean_traces: Optional[np.ndarray] = None,
    batch_size: int = 8,
    num_epochs: int = 5,
    lr: float = 1e-3,
    device: Optional[str] = None,
    normalize: bool = True,
) -> CNNDenoiser1DResidual:
    """
    Train the residual CNN denoiser on paired noisy/clean traces.
    """
    dataset = TraceDenoiseDataset(noisy_traces, clean_traces, normalize=normalize)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    in_channels = dataset.x.shape[1]
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = CNNDenoiser1DResidual(in_channels=in_channels).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for x_noisy, x_clean in loader:
            x_noisy = x_noisy.to(device)
            x_clean = x_clean.to(device)

            optimizer.zero_grad()
            x_pred = model(x_noisy)
            loss = criterion(x_pred, x_clean)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x_noisy.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}  Loss: {epoch_loss:.4e}")

    return model


def run_cnn_denoise(
    model: CNNDenoiser1DResidual,
    noisy_traces: np.ndarray,
    batch_size: int = 8,
    device: Optional[str] = None,
    normalize: bool = True,
) -> np.ndarray:
    """
    Run a trained CNN denoiser on numpy traces.
    """
    dataset = TraceDenoiseDataset(noisy_traces, noisy_traces, normalize=normalize)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    model.eval()

    denoised_batches: Sequence[np.ndarray] = []
    start = 0
    with torch.no_grad():
        for x_noisy, _ in loader:
            bs = x_noisy.size(0)
            x_noisy = x_noisy.to(device)
            preds = model(x_noisy)
            preds = dataset.denormalize(preds, slice(start, start + bs)).cpu()
            start += bs
            denoised_batches.append(preds.numpy())

    return np.concatenate(denoised_batches, axis=0)


def fit_pca(traces: np.ndarray, n_components: int = 10) -> PCA:
    """
    Fit PCA on traces. For 3D input (N, C, T), the channels and samples are
    flattened so PCA learns across channels as well.
    """
    arr = np.asarray(traces)
    if arr.ndim == 3:
        arr_2d = arr.reshape(arr.shape[0], -1)
    elif arr.ndim == 2:
        arr_2d = arr
    else:
        raise ValueError(f"Expected traces with 2 or 3 dims, got {arr.ndim}")

    pca = PCA(n_components=n_components, svd_solver="randomized")
    pca.fit(arr_2d)
    return pca


def pca_denoise(pca: PCA, noisy_traces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project traces onto the leading PCA components and reconstruct.
    """
    arr = np.asarray(noisy_traces)
    original_shape = arr.shape

    if arr.ndim == 3:
        arr_2d = arr.reshape(arr.shape[0], -1)
    elif arr.ndim == 2:
        arr_2d = arr
    else:
        raise ValueError(f"Expected traces with 2 or 3 dims, got {arr.ndim}")

    coeffs = pca.transform(arr_2d)
    recon = pca.inverse_transform(coeffs).reshape(original_shape)
    return recon.astype(np.float32), coeffs
