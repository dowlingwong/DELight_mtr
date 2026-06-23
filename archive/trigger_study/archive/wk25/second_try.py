from pathlib import Path
import os, io, gc
import numpy as np
import pandas as pd
import h5py
import zstandard as zstd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# =============================
# (0) OOM-friendly runtime knobs
# =============================
# ---- After device selection ----
gpu_index = 0
if torch.cuda.is_available():
    device = torch.device(f"cuda:{gpu_index}")
    torch.cuda.set_device(gpu_index)
    try:
        torch.set_default_device(device)
    except Exception:
        pass
else:
    device = torch.device("cpu")
print("Using device:", device)

# ---- Define generator now that device exists ----
gen = torch.Generator(device=device)

gen = torch.Generator(device=device if device.type == "cuda" else torch.device("cpu"))

os.environ.setdefault("PYTORCH_ALLOC_CONF", "cuda_expandable_segments:true")  # new name
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("medium")  # PyTorch 2.x
except Exception:
    pass

# =============================
# (1) Helpers (unchanged)
# =============================
def read_meta_h5(meta_path: Path):
    meta_path = Path(meta_path)
    with h5py.File(meta_path, "r") as f:
        attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}
        data = f["events"][:]   # structured array
    df = pd.DataFrame({
        "x": data["x"], "y": data["y"], "z": data["z"],
        "energy": data["energy"],
        "type_recoil": [s.decode("utf-8") if isinstance(s, (bytes, bytearray)) else str(s)
                        for s in data["type_recoil"]],
        "no_noise": data["no_noise"],
        "quantize": data["quantize"],
    })
    return attrs, df

def _unshuffle_batch(block: bytes, batch_events: int, n_channels: int, trace_samples: int,
                     dtype=np.float16) -> np.ndarray:
    dtype = np.dtype(dtype)
    itemsize = dtype.itemsize
    num_elements = n_channels * trace_samples
    u8 = np.frombuffer(block, dtype=np.uint8)
    expected = batch_events * itemsize * num_elements
    if u8.size != expected:
        raise ValueError(f"Unexpected batch size: got {u8.size} bytes, expected {expected}")
    u8 = (u8.reshape(batch_events, itemsize, num_elements)
             .swapaxes(1, 2)
             .reshape(batch_events, num_elements * itemsize))
    arr = u8.view(dtype)  # (B, N)
    return arr.reshape(batch_events, n_channels, trace_samples)

# =============================
# (2) Decompress → memmap (Option A)
# =============================
def decompress_to_memmap(
    traces_path: Path,
    out_mm: Path,
    n_events: int,
    n_channels: int,
    trace_samples: int,
    dtype=np.float16,
    read_batch: int = 200,
    tolerant_padding: bool = True,      # ignore trailing non-event-aligned bytes
):
    """
    Stream-decompress traces.zst and write to a (n_events, C, T) memmap file.
    If the .zst has trailing padded bytes, optionally ignore them.
    """
    dtype = np.dtype(dtype)
    shape = (n_events, n_channels, trace_samples)
    ev_sz = int(n_channels * trace_samples * dtype.itemsize)

    out_mm.parent.mkdir(parents=True, exist_ok=True)
    mm = np.memmap(out_mm, mode="w+", dtype=dtype, shape=shape)

    emitted = 0
    dctx = zstd.ZstdDecompressor()
    with open(traces_path, "rb") as fin, dctx.stream_reader(fin) as reader:
        buf = io.BufferedReader(reader)
        stash = bytearray()

        while emitted < n_events:
            if len(stash) < ev_sz:
                chunk = buf.read(1 << 20)  # ~1 MiB at a time
                if not chunk:
                    break  # EOF
                stash += chunk

            full = len(stash) // ev_sz
            if full == 0:
                break

            bsz = min(read_batch, n_events - emitted, full)
            need = bsz * ev_sz
            block = bytes(stash[:need]); del stash[:need]

            mm[emitted:emitted+bsz] = _unshuffle_batch(block, bsz, n_channels, trace_samples, dtype=dtype)
            emitted += bsz

        # leftover handling
        rem = len(stash) % ev_sz
        if rem != 0 and not tolerant_padding:
            raise EOFError(
                f"Stream ended with {len(stash)} leftover bytes "
                f"({rem} not forming a full event of {ev_sz})."
            )

    if emitted < n_events:
        mm[emitted:] = 0  # zero-fill tail if fewer events present

    mm.flush()
    del mm  # close mapping

def open_memmap(mm_path: Path, n_events: int, n_channels: int, trace_samples: int, dtype=np.float16) -> np.memmap:
    dtype = np.dtype(dtype)
    return np.memmap(mm_path, mode="r", dtype=dtype, shape=(n_events, n_channels, trace_samples))

# =============================
# (3) Dataset using memmap (map-style)
# =============================
class MemmapEventDataset(Dataset):
    """
    Random-access dataset over a (N, C, T) memmap + meta DataFrame (aligned by row).
    Returns CPU tensors; DataLoader(pin_memory=True) can pin them; training loop moves to GPU.
    """
    def __init__(self, mm_path: Path, meta_df: pd.DataFrame,
                 n_channels: int, trace_samples: int, trace_dtype=np.float16):
        super().__init__()
        self.mm_path = Path(mm_path)
        self.meta = meta_df.reset_index(drop=True)
        self.n = len(self.meta)
        self.C = int(n_channels)
        self.T = int(trace_samples)
        self.dtype = np.dtype(trace_dtype)

        self.mm = open_memmap(self.mm_path, self.n, self.C, self.T, self.dtype)

        classes = sorted(self.meta["type_recoil"].unique())
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.num_classes = len(classes)

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int):
        # Keep EVERYTHING on CPU explicitly.
        x = torch.from_numpy(self.mm[idx])  # (C, T), float16 CPU
        pos = torch.tensor([self.meta.at[idx, "x"],
                            self.meta.at[idx, "y"],
                            self.meta.at[idx, "z"]],
                           dtype=torch.float32, device="cpu")
        energy = torch.tensor(self.meta.at[idx, "energy"], dtype=torch.float32, device="cpu")
        cls = torch.tensor(self.class_to_idx[self.meta.at[idx, "type_recoil"]],
                           dtype=torch.long, device="cpu")
        return {"x": x, "pos": pos, "energy": energy, "cls": cls}

def collate_events(batch, n_channels: int):
    B = len(batch)
    x = torch.stack([b["x"] for b in batch], dim=0)                # (B,C,T) float16 CPU
    pos = torch.stack([b["pos"] for b in batch], dim=0)            # (B,3)   float32 CPU
    energy = torch.stack([b["energy"] for b in batch], dim=0)      # (B,)    float32 CPU
    cls = torch.stack([b["cls"] for b in batch], dim=0)            # (B,)    long    CPU
    mask = torch.ones((B, n_channels), dtype=torch.float32)        # CPU
    return {"x": x, "mask": mask, "pos": pos, "energy": energy, "cls": cls}

# =============================
# (4) Training utils
# =============================
def _trim_cache_every(n, step):
    if step % n == 0 and torch.cuda.is_available():
        torch.cuda.empty_cache()

def assert_model_on(device, model: nn.Module):
    bad = []
    for n, p in model.named_parameters(recurse=True):
        if p is not None and p.data.numel() and p.device != device:
            bad.append(("param", n, p.device))
    for n, b in model.named_buffers(recurse=True):
        if b is not None and b.data.numel() and b.device != device:
            bad.append(("buffer", n, b.device))
    if bad:
        msg = "Found modules not on target device:\n" + "\n".join([f"{k}: {n} @ {d}" for k,n,d in bad])
        raise RuntimeError(msg)

def _auto_to_input_device(module, inputs):
    # Forward pre-hook: ensure model sits on the same device as the input tensor
    if not inputs:
        return
    x = inputs[0]
    if isinstance(x, torch.Tensor) and x.is_cuda:
        # cheap no-op if already correct
        module.to(x.device)

def train_one_epoch(model, loader, optimizer, scaler, device,
                    w_pos=1.0, w_energy=1.0, w_cls=1.0,
                    accum_steps=4, cache_trim_every=50):
    model.train()
    loss_pos_fn = nn.SmoothL1Loss()
    loss_energy_fn = nn.MSELoss()
    loss_cls_fn = nn.CrossEntropyLoss()

    optimizer.zero_grad(set_to_none=True)
    steps = 0

    for batch in loader:
        x_cpu = batch["x"]           # (B,C,T) float16 CPU
        m_cpu = batch["mask"]        # (B,C)   float32 CPU
        yp_cpu = batch["pos"]        # (B,3)   float32 CPU
        ye_cpu = batch["energy"]     # (B,)    float32 CPU
        yc_cpu = batch["cls"]        # (B,)    long    CPU

        B = x_cpu.shape[0]
        mb = max(1, B // accum_steps)
        if B % mb != 0:
            mb = 1

        for i in range(0, B, mb):
            x = x_cpu[i:i+mb].to(device, dtype=torch.float32, non_blocking=True)
            m = m_cpu[i:i+mb].to(device, non_blocking=True)
            yp = yp_cpu[i:i+mb].to(device, non_blocking=True)
            ye = ye_cpu[i:i+mb].to(device, non_blocking=True)
            yc = yc_cpu[i:i+mb].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                # model has a pre-hook to auto-move to x.device if any laziness exists
                pos_pred, energy_pred, cls_logit, aux = model(x, channel_mask=m)
                if energy_pred.dim() == 2 and energy_pred.shape[1] == 1:
                    energy_pred = energy_pred.squeeze(1)

                loss_pos = loss_pos_fn(pos_pred, yp)
                loss_energy = loss_energy_fn(energy_pred, ye)
                loss_cls = loss_cls_fn(cls_logit, yc)
                loss = (w_pos * loss_pos + w_energy * loss_energy + w_cls * loss_cls) * (mb / B)

            scaler.scale(loss).backward()

            del x, m, yp, ye, yc, pos_pred, energy_pred, cls_logit, aux, loss_pos, loss_energy, loss_cls, loss

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        steps += 1
        _trim_cache_every(cache_trim_every, steps)

    return {}

# =============================
# (5) Main
# =============================
if __name__ == "__main__":
    from archive.trigger_study.archive.wk25.tcxformer_v2 import ModelConfig, ModalityTCXFormer  # your file

    torch.manual_seed(0)

    # ---- Choose GPU explicitly and set default device for NEW modules/tensors
    gpu_index = 0
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_index}")
        torch.cuda.set_device(gpu_index)
        try:
            # Ensures any tensors/modules created AFTER this line default to this GPU
            torch.set_default_device(device)
        except Exception:
            pass
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    # ---- Paths & meta
    BASE = Path("/ceph/dwong/work/training_samples/ER/small")
    ENERGY_BIN = 1000
    meta_path = BASE / f"meta_energy_{ENERGY_BIN}.h5"
    traces_path = BASE / f"traces_energy_{ENERGY_BIN}.zst"
    mm_path = BASE / f"traces_energy_{ENERGY_BIN}.mm"

    attrs, meta_df = read_meta_h5(meta_path)
    n_events = len(meta_df)
    n_channels = int(attrs["n_channels"])
    trace_samples = int(attrs["trace_samples"])
    trace_dtype = np.dtype(attrs.get("trace_dtype", np.float16))

    # ---- Build memmap once (idempotent)
    if not mm_path.exists():
        print(f"[memmap] Creating {mm_path} from {traces_path} ...")
        decompress_to_memmap(
            traces_path, mm_path,
            n_events=n_events,
            n_channels=n_channels,
            trace_samples=trace_samples,
            dtype=trace_dtype,
            read_batch=200,
            tolerant_padding=True,  # ignore padded trailing bytes
        )
        print("[memmap] Done.")

    # ---- Dataset / DataLoader
    ds = MemmapEventDataset(mm_path, meta_df, n_channels, trace_samples, trace_dtype)
    print(f"Classes: {ds.class_to_idx}  (num={ds.num_classes})")

    def _collate(batch):
        return collate_events(batch, n_channels)

    loader = DataLoader(
        ds,
        batch_size=64,          # DataLoader batches events (map-style dataset)
        shuffle=True,
        num_workers=0,          # set >0 later if desired (no CUDA in workers)
        pin_memory=False,        # CPU -> pinned; moved in loop
        persistent_workers=False,
        prefetch_factor=None,   # must be None when num_workers=0
        collate_fn=_collate,
        generator=gen,             # <--- important

    )

    # ---- Model (created AFTER set_default_device -> lazy modules default to CUDA)
    cfg = ModelConfig(
        n_photon=19, n_phonon=37,
        stride_photon=128, stride_phonon=256,
        d_model=256, d_ff=1024, n_heads=4,
        n_time_layers=3, n_chan_layers=1,
        patch_embed=64, dropout=0.1,
        rope_base=10000.0, n_branch_to_task=2,
    )
    model = ModalityTCXFormer(cfg).to(device)

    # Safety: move-to-input-device pre-hook (covers rare lazy-in-forward cases)
    model.register_forward_pre_hook(_auto_to_input_device)

    # Sanity: all existing params/buffers should be on device
    assert_model_on(device, model)

    # ---- Optimizer / AMP
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    try:
        for epoch in range(1, 4):
            _ = train_one_epoch(
                model, loader, optimizer, scaler, device,
                w_pos=1.0, w_energy=1.0, w_cls=1.0,
                accum_steps=8,              # increase if you still see OOM
                cache_trim_every=50
            )
            print(f"[epoch {epoch}] done")
    finally:
        del model, optimizer, scaler, loader, ds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
