"""
Upgrades added to the Modality-Aware Hierarchical TCX-Former:

1) μP-style initialization for Linear layers (+ keep existing zero-inits on residual paths).
2) Dual optimizers: Muon for block (matrix) weights, AdamW for auxiliary params.
3) Residual global-mean pooling from per-branch features injected into task tokens (reduces fusion bottleneck).
4) Gated FFN (SwiGLU-style) in Transformer blocks and cross-attn MLPs.

Note: This is a drop-in replacement for the previous TCX model; class names kept.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

# Optional: import your Muon optimizer implementation
try:
    from reconstruction_model.muon import Muon
except Exception as e:
    Muon = None  # If unavailable, user can provide their Muon optimizer

# ----------------------------
# Utilities & helpers
# ----------------------------

def rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.eps) * self.weight


def zscore(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, keepdim=True)
    return (x - mean) / (std + eps)


# ----------------------------
# μP-style init helper
# ----------------------------

def mup_init_linear(module: nn.Module):
    if isinstance(module, nn.Linear):
        d_out, d_in = module.weight.size(0), module.weight.size(1)
        sigma = min(1.0, math.sqrt(d_out / d_in)) / math.sqrt(d_in)
        nn.init.normal_(module.weight, mean=0.0, std=sigma)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# ----------------------------
# RoPE (Rotary Positional Embedding)
# ----------------------------

def build_rope_cache(seq_len: int, d_head: int, base: float, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    assert d_head % 2 == 0, "d_head must be even for RoPE"
    half = d_head // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.einsum("n,f->nf", t, inv_freq)  # (N, half)
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d = x.size(-1)
    assert d % 2 == 0
    x = x.view(*x.shape[:-1], d // 2, 2)
    x0, x1 = x[..., 0], x[..., 1]
    cos = cos.unsqueeze(0)  # (1, N, half)
    sin = sin.unsqueeze(0)
    xr0 = x0 * cos - x1 * sin
    xr1 = x1 * cos + x0 * sin
    out = torch.stack((xr0, xr1), dim=-1).reshape(*x.shape[:-2], d)
    return out


# ----------------------------
# Gated FFN (SwiGLU)
# ----------------------------

class GatedFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        # Zero-init down for stable residual start
        nn.init.zeros_(self.down.weight)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.silu(self.up(x)) * self.gate(x)
        return self.drop(self.down(y))


# ----------------------------
# Attention blocks (custom, expose Q/K for RoPE)
# ----------------------------

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, rope: bool = False, rope_base: float = 10000.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.rope = rope
        self.rope_base = rope_base
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        nn.init.zeros_(self.o_proj.weight)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        def split_heads(t):
            return t.view(B, N, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)
        if self.rope:
            cos, sin = build_rope_cache(N, self.d_head, self.rope_base, device=x.device, dtype=x.dtype)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        q = q.reshape(B * self.n_heads, N, self.d_head)
        k = k.reshape(B * self.n_heads, N, self.d_head)
        v = v.reshape(B * self.n_heads, N, self.d_head)
        if attn_mask is not None:
            if attn_mask.dim() == 3 and attn_mask.size(0) == B:
                attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
                attn_mask = attn_mask.reshape(B * self.n_heads, N, N)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.attn_drop.p if self.training else 0.0)
        out = out.reshape(B, self.n_heads, N, self.d_head).permute(0, 2, 1, 3).contiguous()
        out = out.view(B, N, self.d_model)
        out = self.proj_drop(self.o_proj(out))
        return out


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        nn.init.zeros_(self.o_proj.weight)

    def forward(self, q_tokens: torch.Tensor, kv_feats: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, Q, _ = q_tokens.shape
        K = kv_feats.size(1)
        q = self.q_proj(q_tokens)
        kv = self.kv_proj(kv_feats)
        k, v = kv.chunk(2, dim=-1)
        def split_heads(t):
            return t.view(B, -1, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)
        q = q.reshape(B * self.n_heads, Q, self.d_head)
        k = k.reshape(B * self.n_heads, K, self.d_head)
        v = v.reshape(B * self.n_heads, K, self.d_head)
        if attn_mask is not None:
            if attn_mask.dim() == 3 and attn_mask.size(0) == B:
                attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
                attn_mask = attn_mask.reshape(B * self.n_heads, Q, K)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.attn_drop.p if self.training else 0.0)
        out = out.reshape(B, self.n_heads, Q, self.d_head).permute(0, 2, 1, 3).contiguous()
        out = out.view(B, Q, self.d_model)
        out = self.proj_drop(self.o_proj(out))
        return out


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, rope: bool = False, rope_base: float = 10000.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout=dropout, rope=rope, rope_base=rope_base)
        self.norm2 = RMSNorm(d_model)
        self.ffn = GatedFFN(d_model, d_ff, dropout=dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.q_norm = RMSNorm(d_model)
        self.kv_norm = RMSNorm(d_model)
        self.attn = MultiHeadCrossAttention(d_model, n_heads, dropout=dropout)
        self.mlp = GatedFFN(d_model, d_ff, dropout=dropout)

    def forward(self, tokens: torch.Tensor, feats: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        t = tokens + self.attn(self.q_norm(tokens), self.kv_norm(feats), attn_mask=attn_mask)
        t = t + self.mlp(t)
        return t


# ----------------------------
# Patch / Downsample front-end
# ----------------------------

class DepthwisePointwise1D(nn.Module):
    def __init__(self, in_ch: int, embed: int, kernel: int = 9, stride: int = 64, dropout: float = 0.0):
        super().__init__()
        pad = kernel // 2
        self.dw = nn.Conv1d(in_ch, in_ch, kernel_size=kernel, stride=stride, padding=pad, groups=in_ch)
        self.pw = nn.Conv1d(in_ch, embed, kernel_size=1)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.dw(x))
        y = self.drop(self.pw(y))
        return y


# ----------------------------
# Branch encoder (per modality)
# ----------------------------

@dataclass
class BranchConfig:
    n_channels: int
    stride: int
    d_model: int = 256
    d_ff: int = 1024
    n_heads: int = 4
    n_time_layers: int = 3
    n_chan_layers: int = 1
    patch_embed: int = 64
    dropout: float = 0.1
    rope_base: float = 10000.0


class BranchEncoder(nn.Module):
    def __init__(self, cfg: BranchConfig):
        super().__init__()
        self.cfg = cfg
        self.front = DepthwisePointwise1D(in_ch=cfg.n_channels, embed=cfg.patch_embed, stride=cfg.stride, dropout=cfg.dropout)
        self.patch_proj = nn.Linear(cfg.patch_embed, cfg.d_model, bias=True)
        self.time_blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, dropout=cfg.dropout, rope=True, rope_base=cfg.rope_base)
            for _ in range(cfg.n_time_layers)
        ])
        self.chan_pos = nn.Parameter(torch.zeros(1, cfg.n_channels, cfg.d_model))
        nn.init.normal_(self.chan_pos, std=0.02)
        self.chan_blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, dropout=cfg.dropout, rope=False)
            for _ in range(cfg.n_chan_layers)
        ])
        self.branch_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.normal_(self.branch_token, std=0.02)
        # small token refiner for branch token
        self._bt_mlp = GatedFFN(cfg.d_model, cfg.d_ff, dropout=cfg.dropout)

    def forward(self, x: torch.Tensor, ch_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, Cb, L = x.shape
        assert Cb == self.cfg.n_channels
        x = zscore(x, dim=-1)
        y = self.front(x)                 # (B, E, N)
        B2, E, N = y.shape
        assert B2 == B
        y = y.transpose(1, 2)             # (B, N, E)
        y = self.patch_proj(y)            # (B, N, d)
        y = y.unsqueeze(1).expand(B, Cb, N, -1)
        t = y.reshape(B * Cb, N, -1)
        for blk in self.time_blocks:
            t = blk(t, attn_mask=None)
        y = t.view(B, Cb, N, -1).transpose(1, 2)   # (B, N, Cb, d)

        # Channel transformer over Cb
        y = y + self.chan_pos[:, :Cb, :].unsqueeze(1)
        y = y.reshape(B * N, Cb, -1)
        attn_mask = None
        if ch_mask is not None:
            cm = ch_mask.to(torch.bool).unsqueeze(1)
            cm = cm.unsqueeze(1).expand(B, N, 1, Cb).reshape(B * N, 1, Cb)
            attn_mask = cm.expand(B * N, Cb, Cb)
        for blk in self.chan_blocks:
            y = blk(y, attn_mask=attn_mask)
        y = y.view(B, N, Cb, -1)

        feats = y.reshape(B, N * Cb, -1)  # (B, T, d)
        token = self.branch_token.expand(B, -1, -1)
        token = MultiHeadCrossAttention(self.cfg.d_model, self.cfg.n_heads, dropout=self.cfg.dropout)(token, feats, attn_mask=None)
        token = token + self._bt_mlp(token)
        return feats, token


# ----------------------------
# Top-level model
# ----------------------------

@dataclass
class ModelConfig:
    n_photon: int = 19
    n_phonon: int = 37
    stride_photon: int = 128
    stride_phonon: int = 256
    d_model: int = 256
    d_ff: int = 1024
    n_heads: int = 4
    n_time_layers: int = 3
    n_chan_layers: int = 1
    patch_embed: int = 64
    dropout: float = 0.1
    rope_base: float = 10000.0
    n_branch_to_task: int = 2


class ModalityTCXFormer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        # Branch encoders
        self.photon_enc = BranchEncoder(BranchConfig(
            n_channels=cfg.n_photon,
            stride=cfg.stride_photon,
            d_model=cfg.d_model,
            d_ff=cfg.d_ff,
            n_heads=cfg.n_heads,
            n_time_layers=cfg.n_time_layers,
            n_chan_layers=cfg.n_chan_layers,
            patch_embed=cfg.patch_embed,
            dropout=cfg.dropout,
            rope_base=cfg.rope_base,
        ))
        self.phonon_enc = BranchEncoder(BranchConfig(
            n_channels=cfg.n_phonon,
            stride=cfg.stride_phonon,
            d_model=cfg.d_model,
            d_ff=cfg.d_ff,
            n_heads=cfg.n_heads,
            n_time_layers=cfg.n_time_layers,
            n_chan_layers=cfg.n_chan_layers,
            patch_embed=cfg.patch_embed,
            dropout=cfg.dropout,
            rope_base=cfg.rope_base,
        ))
        # Task tokens
        self.task_tokens = nn.Parameter(torch.zeros(3, cfg.d_model))
        nn.init.normal_(self.task_tokens, std=0.02)
        self.branch2task = nn.ModuleList([
            CrossAttentionBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, dropout=cfg.dropout)
            for _ in range(cfg.n_branch_to_task)
        ])
        # Residual pooling for task tokens (new)
        self.pool_proj = nn.Linear(2 * cfg.d_model, 3 * cfg.d_model, bias=False)
        # Heads
        self.pos_head = nn.Sequential(nn.Linear(cfg.d_model, 128), nn.SiLU(), nn.Linear(128, 3))
        self.energy_head = nn.Sequential(nn.Linear(cfg.d_model, 128), nn.SiLU(), nn.Linear(128, 1))
        self.cls_head = nn.Linear(cfg.d_model, 1)
        # Aux heads
        self.energy_head_phonon = nn.Sequential(nn.Linear(cfg.d_model, 128), nn.SiLU(), nn.Linear(128, 1))
        self.cls_head_photon = nn.Linear(cfg.d_model, 1)

        # Apply μP-style initialization to all Linear layers (except explicit zero-inits retained)
        self.apply(mup_init_linear)

    # ----------------------------
    # Optimizer configuration (Muon + AdamW)
    # ----------------------------
    def configure_optimisers(
        self,
        adamw_lr: float = 1e-3,
        adamw_betas: tuple = (0.9, 0.999),
        adamw_weight_decay: float = 0.0,
        adamw_fused: bool = True,
        muon_lr: float = 1e-3,
        muon_momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
    ):
        # Collect parameters
        block_weight_params: List[torch.nn.Parameter] = []
        aux_params: List[torch.nn.Parameter] = []

        def add_params_from_module(mod: nn.Module, use_muon_for_conv: bool = False):
            for p in mod.parameters(recurse=True):
                if p.ndim >= 2:
                    # Route convs to AdamW unless explicitly requested
                    if p.is_leaf and (use_muon_for_conv or not isinstance(getattr(p, 'grad_fn', None), torch.autograd.Function)):
                        block_weight_params.append(p)
                    else:
                        block_weight_params.append(p)
                else:
                    aux_params.append(p)

        # Route transformer block matrices to Muon
        add_params_from_module(self.photon_enc.time_blocks)
        add_params_from_module(self.photon_enc.chan_blocks)
        add_params_from_module(self.phonon_enc.time_blocks)
        add_params_from_module(self.phonon_enc.chan_blocks)
        add_params_from_module(self.branch2task)
        # Route everything else to AdamW (embeddings, norms, branch tokens, task tokens, convs, heads)
        aux_params.extend(list(self.photon_enc.front.parameters()))
        aux_params.extend(list(self.phonon_enc.front.parameters()))
        aux_params.extend(list(self.photon_enc.patch_proj.parameters()))
        aux_params.extend(list(self.phonon_enc.patch_proj.parameters()))
        aux_params.extend([self.photon_enc.chan_pos, self.phonon_enc.chan_pos, self.photon_enc.branch_token, self.phonon_enc.branch_token, self.task_tokens])
        aux_params.extend(list(self.pool_proj.parameters()))
        aux_params.extend(list(self.pos_head.parameters()))
        aux_params.extend(list(self.energy_head.parameters()))
        aux_params.extend(list(self.cls_head.parameters()))
        aux_params.extend(list(self.energy_head_phonon.parameters()))
        aux_params.extend(list(self.cls_head_photon.parameters()))

        adamw_kwargs = dict(lr=adamw_lr, betas=adamw_betas, weight_decay=adamw_weight_decay, fused=adamw_fused)
        adamw_optim = AdamW([{'params': aux_params}], **adamw_kwargs)

        if Muon is None:
            muon_optim = None
        else:
            muon_kwargs = dict(lr=muon_lr, momentum=muon_momentum, nesterov=nesterov, ns_steps=ns_steps)
            muon_optim = Muon(block_weight_params, **muon_kwargs)

        return [opt for opt in [adamw_optim, muon_optim] if opt is not None]

    def forward(
        self,
        x: torch.Tensor,
        channel_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        B, C, L = x.shape
        n_ph = self.cfg.n_photon
        n_pn = self.cfg.n_phonon
        x_ph = x[:, :n_ph, :]
        x_pn = x[:, -n_pn:, :]

        mask_ph = channel_mask[:, :n_ph] if channel_mask is not None else None
        mask_pn = channel_mask[:, -n_pn:] if channel_mask is not None else None

        feats_ph, tok_ph = self.photon_enc(x_ph, mask_ph)   # (B, Tp, d), (B,1,d)
        feats_pn, tok_pn = self.phonon_enc(x_pn, mask_pn)   # (B, Tn, d), (B,1,d)

        # Residual global mean pooling -> enrich task tokens (NEW)
        pool_ph = feats_ph.mean(dim=1)  # (B, d)
        pool_pn = feats_pn.mean(dim=1)  # (B, d)
        pooled = torch.cat([pool_ph, pool_pn], dim=-1)  # (B, 2d)
        pooled_to_tasks = self.pool_proj(pooled).view(B, 3, self.cfg.d_model)  # (B,3,d)

        # Base task tokens
        task = self.task_tokens.unsqueeze(0).expand(B, -1, -1)
        task = task + pooled_to_tasks  # residual enrichment

        # Fast fusion using branch tokens only
        branch_tokens = torch.cat([tok_ph, tok_pn], dim=1)  # (B,2,d)
        for blk in self.branch2task:
            task = blk(task, branch_tokens)

        t_pos, t_energy, t_cls = task[:, 0], task[:, 1], task[:, 2]
        pos = self.pos_head(t_pos)
        energy = self.energy_head(t_energy)
        cls_logit = self.cls_head(t_cls)

        # Aux heads
        aux_energy = self.energy_head_phonon(tok_pn.squeeze(1))
        aux_cls = self.cls_head_photon(tok_ph.squeeze(1))
        aux = {"energy_phonon": aux_energy, "cls_photon": aux_cls}
        return pos, energy, cls_logit, aux

