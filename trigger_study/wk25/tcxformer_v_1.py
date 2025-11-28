from __future__ import annotations

"""
TCX-Former (Time-then-Channel Transformer) — fixed & 65,536-ready
-----------------------------------------------------------------
Shapes
------
Input:  x: (B, C, L)
Front:  (B, C, L) -> (B, C, E, N)   # E: per-channel embed size, N: tokens after stride
Proj:   (B, C, E, N) -> (B, C, N, d)
Time:   (B*C, N, d)  -> (B, C, N, d)
Chan:   (B, N, C, d) -> (B*N, C, d) -> (B, N, C, d)
Pool:   (B, d)

"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


# -------------------------------
# Utilities: RoPE
# -------------------------------
def precompute_rope_angles(max_seq_len: int, d_head: int, base: float = 10000.0, device=None) -> Tensor:
    """Return (Nmax, d_head//2, 2) cache of (cos, sin)."""
    assert d_head % 2 == 0, "d_head must be even for RoPE"
    # classic RoPE frequency schedule
    freqs = 1.0 / (base ** (torch.arange(0, d_head, 2, device=device) / d_head))
    t = torch.arange(max_seq_len, device=device)  # (Nmax,)
    angles = torch.outer(t, freqs)                # (Nmax, d_head//2)
    return torch.stack([angles.cos(), angles.sin()], dim=-1)  # (Nmax, d_head//2, 2)


def apply_rotary_embeddings(x: Tensor, rope_slice: Tensor) -> Tensor:
    """Apply RoPE to q/k.
    x:          (B*, N, n_h, d_h)
    rope_slice: (N, d_h//2, 2)
    return:     (B*, N, n_h, d_h)
    """
    Bstar, N, n_h, d_h = x.shape
    assert d_h % 2 == 0
    x = x.view(Bstar, N, n_h, d_h // 2, 2)
    cos, sin = rope_slice[..., 0], rope_slice[..., 1]    # (N, d_h//2)
    cos = cos.view(1, N, 1, -1)
    sin = sin.view(1, N, 1, -1)
    x1, x2 = x[..., 0], x[..., 1]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.stack([y1, y2], dim=-1).reshape(Bstar, N, n_h, d_h)


# -------------------------------
# Front-end: Depthwise + Grouped Pointwise
# -------------------------------
class DepthwisePointwise1D(nn.Module):
    """Preserve per-channel identity and compress time via stride.

    In:  (B, C, L)
    Out: (B, C, E, N)  where N ≈ ceil(L/stride)
    """

    def __init__(self, in_ch: int, embed: int, kernel: int = 9, stride: int = 128):
        super().__init__()
        assert kernel % 2 == 1, "Use odd kernel for SAME padding."
        pad = kernel // 2
        self.in_ch = in_ch
        self.embed = embed
        self.dw = nn.Conv1d(in_ch, in_ch, kernel_size=kernel, stride=stride, padding=pad, groups=in_ch, bias=False)
        # grouped pointwise: no cross-channel mixing
        self.pw = nn.Conv1d(in_ch, in_ch * embed, kernel_size=1, groups=in_ch, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, L)
        y = self.dw(x)                 # (B, C, N)
        y = self.act(y)
        y = self.pw(y)                 # (B, C*E, N)
        B, CE, N = y.shape
        C = self.in_ch
        E = self.embed
        y = y.view(B, C, E, N)         # (B, C, E, N)
        return y


# -------------------------------
# Core blocks
# -------------------------------
class MHA(nn.Module):
    def __init__(self, d_model: int, n_head: int, use_rope: bool):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.use_rope = use_rope
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor, rope_slice: Optional[Tensor] = None) -> Tensor:
        # x: (B*, N, d)
        Bstar, N, d = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(Bstar, N, self.n_head, self.d_head)
        k = k.view(Bstar, N, self.n_head, self.d_head)
        v = v.view(Bstar, N, self.n_head, self.d_head).transpose(1, 2)  # (B*, n_h, N, d_h)
        if self.use_rope:
            assert rope_slice is not None and rope_slice.shape[0] == N
            q = apply_rotary_embeddings(q, rope_slice)
            k = apply_rotary_embeddings(k, rope_slice)
        q = q.transpose(1, 2)  # (B*, n_h, N, d_h)
        k = k.transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v)  # (B*, n_h, N, d_h)
        attn = attn.transpose(1, 2).reshape(Bstar, N, d)
        return self.o(attn)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_head: int, use_rope: bool, eps: float = 1e-6):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model, eps=eps)
        self.attn = MHA(d_model, n_head, use_rope=use_rope)
        self.norm2 = nn.RMSNorm(d_model, eps=eps)
        self.ffn_up = nn.Linear(d_model, d_ff, bias=False)
        self.ffn_gate = nn.Linear(d_model, d_ff, bias=False)
        self.ffn_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: Tensor, rope_slice: Optional[Tensor] = None) -> Tensor:
        x = x + self.attn(self.norm1(x), rope_slice)
        y = self.norm2(x)
        x = x + self.ffn_down(F.silu(self.ffn_up(y)) * self.ffn_gate(y))
        return x


class ChannelPositionalEmbedding(nn.Module):
    def __init__(self, max_channels: int, d_model: int):
        super().__init__()
        self.pos = nn.Parameter(torch.empty(1, max_channels, d_model))
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B*, C, d)
        return x + self.pos[:, : x.size(1), :]


# -------------------------------
# Branch encoder
# -------------------------------
class BranchEncoder(nn.Module):
    def __init__(
        self,
        n_channels: int,
        d_model: int,
        d_ff: int,
        n_head: int,
        n_time_layers: int,
        n_chan_layers: int,
        per_ch_embed: int = 64,
        stride: int = 128,
        kernel: int = 9,
        rope_base: float = 10000.0,
        max_seq_len: int = 65536,
        max_channels: Optional[int] = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.C = n_channels
        self.d_model = d_model
        self.d_head = d_model // n_head

        # Front-end
        self.front = DepthwisePointwise1D(in_ch=n_channels, embed=per_ch_embed, kernel=kernel, stride=stride)
        self.patch_proj = nn.Linear(per_ch_embed, d_model, bias=True)

        # Temporal stack
        self.time_blocks = nn.ModuleList([
            TransformerBlock(d_model, d_ff, n_head, use_rope=True, eps=eps)
            for _ in range(n_time_layers)
        ])

        # Channel stack
        self.chan_pos = ChannelPositionalEmbedding(max_channels or n_channels, d_model)
        self.chan_blocks = nn.ModuleList([
            TransformerBlock(d_model, d_ff, n_head, use_rope=False, eps=eps)
            for _ in range(n_chan_layers)
        ])

        # RoPE cache sized by post-stride tokens: Nmax = ceil(max_seq_len/stride)
        Nmax = (max_seq_len + stride - 1) // stride
        self.register_buffer(
            "rope_cache",
            precompute_rope_angles(Nmax, self.d_head, base=rope_base, device=None),
            persistent=False,
        )

    def forward(self, x: Tensor, chan_mask: Optional[Tensor] = None) -> Tensor:
        """x: (B, C, L) -> returns y: (B, N, C, d)"""
        B, C, L = x.shape
        assert C == self.C, f"Expected {self.C} channels, got {C}"

        # (B, C, L) -> (B, C, E, N)
        y = self.front(x)
        B, C, E, N = y.shape

        # (B, C, E, N) -> (B, C, N, d)
        y = y.permute(0, 1, 3, 2).contiguous()
        y = self.patch_proj(y)

        # Temporal transformer over N, per channel
        yt = y.view(B * C, N, self.d_model)
        rope_slice = self.rope_cache[:N]
        for blk in self.time_blocks:
            yt = blk(yt, rope_slice=rope_slice)
        y = yt.view(B, C, N, self.d_model)

        # Channel transformer over C, per time step
        y = y.permute(0, 2, 1, 3).contiguous()  # (B, N, C, d)
        y_flat = y.view(B * N, C, self.d_model)
        y_flat = self.chan_pos(y_flat)

        # Optional: channel mask could be integrated into attention; skipped for brevity.
        for blk in self.chan_blocks:
            y_flat = blk(y_flat, rope_slice=None)

        y = y_flat.view(B, N, C, self.d_model)
        return y


# -------------------------------
# Full model
# -------------------------------
@dataclass
class TCXConfig:
    # Model dims
    d_model: int = 256
    d_ff: int = 1024
    n_head: int = 8
    # Depth
    n_time_layers: int = 2
    n_chan_layers: int = 1
    # Front-end
    per_ch_embed: int = 64
    stride_photon: int = 128
    stride_phonon: int = 128
    kernel: int = 9
    rope_base: float = 10000.0
    max_seq_len: int = 65536
    # Channel counts
    n_ch_photon: int = 19
    n_ch_phonon: int = 37
    # Heads
    classify_er_nr: bool = True
    # Numerics
    eps: float = 1e-6


class TCXFormer(nn.Module):
    def __init__(self, cfg: TCXConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        # Two branches (customize counts/strides)
        self.photon = BranchEncoder(
            n_channels=cfg.n_ch_photon,
            d_model=d,
            d_ff=cfg.d_ff,
            n_head=cfg.n_head,
            n_time_layers=cfg.n_time_layers,
            n_chan_layers=cfg.n_chan_layers,
            per_ch_embed=cfg.per_ch_embed,
            stride=cfg.stride_photon,
            kernel=cfg.kernel,
            rope_base=cfg.rope_base,
            max_seq_len=cfg.max_seq_len,
            max_channels=cfg.n_ch_photon,
            eps=cfg.eps,
        )
        self.phonon = BranchEncoder(
            n_channels=cfg.n_ch_phonon,
            d_model=d,
            d_ff=cfg.d_ff,
            n_head=cfg.n_head,
            n_time_layers=cfg.n_time_layers,
            n_chan_layers=cfg.n_chan_layers,
            per_ch_embed=cfg.per_ch_embed,
            stride=cfg.stride_phonon,
            kernel=cfg.kernel,
            rope_base=cfg.rope_base,
            max_seq_len=cfg.max_seq_len,
            max_channels=cfg.n_ch_phonon,
            eps=cfg.eps,
        )

        # Heads
        self.final_norm = nn.RMSNorm(d, eps=cfg.eps)
        self.xyz_head = nn.Linear(d, 3)
        self.energy_head = nn.Linear(d, 1)
        self.cls_head = nn.Linear(d, 2) if cfg.classify_er_nr else None

        self._init_weights()

    def _init_weights(self):
        def init_linear(m: nn.Module):
            if isinstance(m, nn.Linear):
                d_out, d_in = m.weight.shape
                sigma = min(1.0, math.sqrt(d_out / d_in)) / math.sqrt(d_in)
                nn.init.normal_(m.weight, mean=0.0, std=sigma)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(init_linear)
        # start heads near-zero for stable early training
        nn.init.zeros_(self.xyz_head.weight); nn.init.zeros_(self.xyz_head.bias)
        nn.init.zeros_(self.energy_head.weight); nn.init.zeros_(self.energy_head.bias)
        if self.cls_head is not None:
            nn.init.zeros_(self.cls_head.weight); nn.init.zeros_(self.cls_head.bias)

    @torch.no_grad()
    def _split_modalities(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Split input channels into photon + phonon tensors.
        x: (B, C_total, L)
        return: (x_photon: (B, Cp, L), x_phonon: (B, Cn, L))
        """
        Cp, Cn = self.cfg.n_ch_photon, self.cfg.n_ch_phonon
        assert x.size(1) == Cp + Cn, f"Expected {Cp + Cn} channels, got {x.size(1)}"
        return x[:, :Cp, :], x[:, Cp:, :]

    def forward(self, x: Tensor, chan_mask: Optional[Tensor] = None):
        """x: (B, C_total, L=cfg.max_seq_len)
        chan_mask: optional (B, C_total) boolean mask (True = valid)
        returns: spatial_pred (B,3), energy_pred (B,1), cls_logits (B,2 or None)
        """
        B, C, L = x.shape
        xp, xn = self._split_modalities(x)
        mp = mn = None
        if chan_mask is not None:
            Cp = self.cfg.n_ch_photon
            mp = chan_mask[:, :Cp]
            mn = chan_mask[:, Cp:]

        # Encode branches -> (B, Np, Cp, d), (B, Nn, Cn, d)
        yp = self.photon(xp, mp)
        yn = self.phonon(xn, mn)

        # Simple late fusion: concat channels dimension
        # Align N by simple interpolate or crop if different (usually same if strides equal)
        if yp.size(1) != yn.size(1):
            # interpolate along N to the max length
            Nmax = max(yp.size(1), yn.size(1))
            if yp.size(1) != Nmax:
                yp = F.interpolate(yp.permute(0,3,2,1), size=Nmax, mode="linear", align_corners=False).permute(0,3,2,1)
            if yn.size(1) != Nmax:
                yn = F.interpolate(yn.permute(0,3,2,1), size=Nmax, mode="linear", align_corners=False).permute(0,3,2,1)
        y = torch.cat([yp, yn], dim=2)  # (B, N, C_total, d)

        # Global pooling
        z = self.final_norm(y).mean(dim=(1, 2))  # (B, d)
        spatial = self.xyz_head(z)
        energy = self.energy_head(z)
        cls = self.cls_head(z) if self.cls_head is not None else None
        return spatial, energy, cls


# -------------------------------
# Quick smoke test
# -------------------------------
if __name__ == "__main__":
    cfg = TCXConfig(
        d_model=256,
        d_ff=1024,
        n_head=8,
        n_time_layers=2,
        n_chan_layers=1,
        per_ch_embed=64,
        stride_photon=128,
        stride_phonon=128,
        kernel=9,
        max_seq_len=65536,
        n_ch_photon=19,
        n_ch_phonon=37,
        classify_er_nr=True,
    )

    model = TCXFormer(cfg)
    model.eval()

    B = 2
    C = cfg.n_ch_photon + cfg.n_ch_phonon
    L = cfg.max_seq_len
    x = torch.randn(B, C, L)
    with torch.no_grad():
        spatial, energy, cls = model(x)
    print("spatial:", spatial.shape, "energy:", energy.shape, "cls:", None if cls is None else cls.shape)
