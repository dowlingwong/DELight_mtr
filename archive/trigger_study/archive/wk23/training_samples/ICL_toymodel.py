#!/usr/bin/env python3
"""
ictool_pytorch_memopt.py

Memory-optimized PyTorch reimplementation of the LSTM -> Transformer architecture
(compatible with the thesis + Keras reference), including three task heads:
  - PR   : position regression (x,y,z)
  - PRC  : position + classification
  - PRCE : position + classification + energy

Key memory savers (without changing the model's high-level design):
  - Channel chunking (process subsets of detectors through the LSTM)
  - Optional truncated BPTT (process long time series in windows)
  - Optional gradient checkpointing for Transformer layers
  - AMP (fp16/bf16), smaller batch + grad accumulation

Run a quick synthetic test:
  python ictool_pytorch_memopt.py --model prce --batch-size 1 --seq-len 16384 --steps 3 --precision bf16 --ch-chunk 8 --t-chunk 1024 --use-checkpoint

Author: ChatGPT (PyTorch 2.x)
"""
import os
import math
import argparse
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint as _checkpoint


# --------------------------
# Config dataclass
# --------------------------
@dataclass
class ICLConfig:
    # data
    n_channels: int = 56
    seq_len: int = 32768

    # LSTM (per-channel time encoder)
    lstm_input_size: int = 1
    lstm_hidden: int = 64
    lstm_layers: int = 3
    lstm_dropout: float = 0.3  # between layers
    # Memory: process channels in chunks (e.g., 8)
    ch_chunk: int = 8
    # Memory: truncated BPTT window (None = full sequence)
    t_chunk: Optional[int] = None

    # Transformer (across channels)
    proj_dim: int = 24  # must be divisible by tfm_heads in PyTorch
    tfm_layers: int = 5
    tfm_heads: int = 3
    tfm_ff: int = 128
    tfm_dropout: float = 0.2
    tfm_checkpoint: bool = False  # gradient checkpointing for transformer

    # MLP heads
    mlp_hidden: int = 256
    mlp_dropout: float = 0.0


# --------------------------
# Backbone: LSTM -> Transformer
# --------------------------
class ICLBackbone(nn.Module):
    def __init__(self, cfg: ICLConfig):
        super().__init__()
        self.cfg = cfg

        self.lstm = nn.LSTM(
            input_size=cfg.lstm_input_size,
            hidden_size=cfg.lstm_hidden,
            num_layers=cfg.lstm_layers,
            dropout=cfg.lstm_dropout if cfg.lstm_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )

        self.proj = nn.Linear(2 * cfg.lstm_hidden, cfg.proj_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.proj_dim,
            nhead=cfg.tfm_heads,
            dim_feedforward=cfg.tfm_ff,
            dropout=cfg.tfm_dropout,
            activation="gelu",
            batch_first=False,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=cfg.tfm_layers)
        self.layernorm = nn.LayerNorm(cfg.proj_dim)

        self.out_dim = cfg.n_channels * cfg.proj_dim

    def _run_lstm_full(self, x_bc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x_bc: (B*c_chunk, T, 1)
        lstm_out, (hn, cn) = self.lstm(x_bc)
        return hn, cn

    def _run_lstm_tbptt(self, x_bc: torch.Tensor, T: int, t_chunk: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Process in time windows, carrying hidden; detach between windows.
        h = None
        c = None
        for t0 in range(0, T, t_chunk):
            t1 = min(T, t0 + t_chunk)
            _, (h, c) = self.lstm(x_bc[:, t0:t1, :], (h, c) if h is not None else None)
            # Detach to avoid backprop across chunk boundaries
            h = h.detach()
            c = c.detach()
        return h, c

    def _transformer_forward(self, tfm_in: torch.Tensor) -> torch.Tensor:
        # tfm_in: (C, B, E)
        if not self.cfg.tfm_checkpoint:
            out = self.transformer(tfm_in)
            return self.layernorm(out)

        # Gradient checkpoint each layer
        out = tfm_in
        for layer in self.transformer.layers:
            out = _checkpoint(layer, out)
        if self.transformer.norm is not None:
            out = self.transformer.norm(out)
        out = self.layernorm(out)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        returns: (B, C*proj_dim)
        """
        B, C, T = x.shape
        assert C == self.cfg.n_channels, f"Expected {self.cfg.n_channels} channels, got {C}"

        ch_chunk = max(1, int(self.cfg.ch_chunk))
        t_chunk = self.cfg.t_chunk

        embs = []
        for c0 in range(0, C, ch_chunk):
            c1 = min(C, c0 + ch_chunk)
            # (B, c_chunk, T) -> (B*c_chunk, T, 1)
            x_bc = x[:, c0:c1, :].reshape(B * (c1 - c0), T).unsqueeze(-1)

            if t_chunk is None:
                hn, cn = self._run_lstm_full(x_bc)
            else:
                hn, cn = self._run_lstm_tbptt(x_bc, T, t_chunk)

            L = self.cfg.lstm_layers
            last_f = hn[2 * (L - 1) + 0]  # (B*c_chunk, hidden)
            last_b = hn[2 * (L - 1) + 1]  # (B*c_chunk, hidden)
            h_cat = torch.cat([last_f, last_b], dim=-1)  # (B*c_chunk, 2*hidden)

            emb_chunk = self.proj(h_cat).view(B, (c1 - c0), -1)  # (B, c_chunk, proj_dim)
            embs.append(emb_chunk)

        emb = torch.cat(embs, dim=1)  # (B, C, E)
        tfm_in = emb.permute(1, 0, 2).contiguous()  # (C, B, E)
        tfm_out = self._transformer_forward(tfm_in)  # (C, B, E)

        tfm_out = tfm_out.permute(1, 0, 2).contiguous()  # (B, C, E)
        return tfm_out.flatten(start_dim=1)  # (B, C*E)


# --------------------------
# Heads
# --------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout=0.0, activation="gelu"):
        super().__init__()
        act = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            act,
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        return self.net(x)


class PR(nn.Module):
    def __init__(self, cfg: Optional[ICLConfig] = None):
        super().__init__()
        self.cfg = cfg or ICLConfig()
        self.backbone = ICLBackbone(self.cfg)
        self.pos_head = MLP(self.backbone.out_dim, self.cfg.mlp_hidden, 3, self.cfg.mlp_dropout, "gelu")
    def forward(self, x):
        feats = self.backbone(x)
        pos = self.pos_head(feats)
        return {"pos": pos}


class PRC(nn.Module):
    def __init__(self, cfg: Optional[ICLConfig] = None):
        super().__init__()
        self.cfg = cfg or ICLConfig()
        self.backbone = ICLBackbone(self.cfg)
        self.shared = nn.Sequential(
            nn.Linear(self.backbone.out_dim, self.cfg.mlp_hidden),
            nn.GELU(),
            nn.Dropout(self.cfg.mlp_dropout) if self.cfg.mlp_dropout > 0 else nn.Identity(),
        )
        self.pos_out = nn.Linear(self.cfg.mlp_hidden, 3)
        self.cls_out = nn.Linear(self.cfg.mlp_hidden, 1)
    def forward(self, x):
        feats = self.backbone(x)
        h = self.shared(feats)
        pos = self.pos_out(h)
        logit = self.cls_out(h)
        return {"pos": pos, "logit": logit}


class PRCE(nn.Module):
    def __init__(self, cfg: Optional[ICLConfig] = None):
        super().__init__()
        self.cfg = cfg or ICLConfig()
        self.backbone = ICLBackbone(self.cfg)
        self.pos_head = MLP(self.backbone.out_dim, self.cfg.mlp_hidden, 3, self.cfg.mlp_dropout, "gelu")
        self.cls_head = MLP(self.backbone.out_dim, self.cfg.mlp_hidden, 1, self.cfg.mlp_dropout, "gelu")
        self.energy_head = MLP(self.backbone.out_dim, self.cfg.mlp_hidden, 1, self.cfg.mlp_dropout, "relu")
    def forward(self, x):
        feats = self.backbone(x)
        pos = self.pos_head(feats)
        logit = self.cls_head(feats)
        energy = torch.relu(self.energy_head(feats))
        return {"pos": pos, "logit": logit, "energy": energy}


# --------------------------
# Loss helpers
# --------------------------
def pr_loss(pred: Dict[str, torch.Tensor], target_pos: torch.Tensor):
    return nn.functional.mse_loss(pred["pos"], target_pos)

def prc_loss(pred: Dict[str, torch.Tensor], target_pos: torch.Tensor, target_cls: torch.Tensor):
    mse = nn.functional.mse_loss(pred["pos"], target_pos)
    bce = nn.functional.binary_cross_entropy_with_logits(pred["logit"].squeeze(-1), target_cls.float())
    return mse + bce, {"mse": mse.item(), "bce": bce.item()}

def prce_loss(pred: Dict[str, torch.Tensor],
              target_pos: torch.Tensor,
              target_cls: torch.Tensor,
              target_energy: torch.Tensor):
    mse_pos = nn.functional.mse_loss(pred["pos"], target_pos)
    bce = nn.functional.binary_cross_entropy_with_logits(pred["logit"].squeeze(-1), target_cls.float())
    mse_e = nn.functional.mse_loss(pred["energy"].squeeze(-1), target_energy.float())
    total = mse_pos + bce + mse_e
    return total, {"mse_pos": mse_pos.item(), "bce": bce.item(), "mse_energy": mse_e.item()}


# --------------------------
# Optimizer & scheduler
# --------------------------
def make_optimizer_and_sched(model: nn.Module,
                             lr: float = 3e-4,
                             weight_decay: float = 4e-2,
                             T_0: int = 30,
                             T_mult: int = 1,
                             eta_min: float = 1e-8):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=T_0, T_mult=T_mult, eta_min=eta_min)
    return opt, sched


# --------------------------
# Synthetic dataset for quick test
# --------------------------
class SyntheticWaveDataset(Dataset):
    def __init__(self, n_samples: int, n_channels: int, seq_len: int, task: str = "prce"):
        self.n = n_samples
        self.C = n_channels
        self.T = seq_len
        self.task = task
    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        # Random waveforms
        x = torch.randn(self.C, self.T)
        pos = torch.randn(3)
        if self.task == "pr":
            return x, pos, None, None
        cls = torch.randint(0, 2, (1,)).item()
        if self.task == "prc":
            return x, pos, torch.tensor(cls), None
        energy = torch.rand(1).item() * 1e5
        return x, pos, torch.tensor(cls), torch.tensor(energy, dtype=torch.float32)


# --------------------------
# Training / eval utilities
# --------------------------
def get_model(name: str, cfg: ICLConfig) -> nn.Module:
    name = name.lower()
    if name == "pr":
        return PR(cfg)
    if name == "prc":
        return PRC(cfg)
    if name == "prce":
        return PRCE(cfg)
    raise ValueError(f"Unknown model: {name}")


def run_step(model, batch, task, device, scaler, accum_steps, precision):
    x, pos, cls, energy = batch
    x = x.to(device, non_blocking=True)
    pos = pos.to(device, non_blocking=True)

    # dtype for autocast
    if precision == "fp16":
        amp_dtype = torch.float16
    elif precision == "bf16":
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = None

    with autocast(enabled=(amp_dtype is not None), dtype=amp_dtype):
        out = model(x)
        if task == "pr":
            loss = pr_loss(out, pos) / accum_steps
        elif task == "prc":
            cls = cls.to(device, non_blocking=True).float()
            loss, _ = prc_loss(out, pos, cls)  # returns summed loss
            loss = loss / accum_steps
        else:  # prce
            cls = cls.to(device, non_blocking=True).float()
            energy = energy.to(device, non_blocking=True).float()
            loss, _ = prce_loss(out, pos, cls, energy)
            loss = loss / accum_steps

    scaler.scale(loss).backward()
    return loss


