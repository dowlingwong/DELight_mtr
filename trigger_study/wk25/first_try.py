from pathlib import Path
import os, io, gc
import numpy as np
import pandas as pd
import h5py
import zstandard as zstd
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

# -----------------------------
# (0) OOM-friendly runtime knobs
# -----------------------------
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True           # Ampere+
try:
    torch.set_float32_matmul_precision("medium")       # PyTorch 2.x
except Exception:
    pass

# -----------------------------
# (1) Helpers (unchanged)
# -----------------------------
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

def iter_traces_zst_batched_merged(traces_path: Path,
                                   n_events: int,
                                   n_channels: int,
                                   trace_samples: int,
                                   batch_size: int = 200,
                                   dtype=np.float16,
                                   max_events: int | None = None):
    dtype = np.dtype(dtype)
    per_event_bytes = int(n_channels * trace_samples * dtype.itemsize)
    to_read = n_events if max_events is None else min(max_events, n_events)

    dctx = zstd.ZstdDecompressor()
    with open(traces_path, "rb") as fin, dctx.stream_reader(fin) as reader:
        buf = io.BufferedReader(reader)
        remaining = to_read
        while remaining > 0:
            bsz = int(min(batch_size, remaining))
            need = per_event_bytes * bsz
            got, chunks = 0, []
            while got < need:
                chunk = buf.read(need - got)
                if not chunk:
                    raise EOFError(f"Unexpected end of stream: need {need} bytes, got {got} bytes")
                chunks.append(chunk)
                got += len(chunk)
            block = b"".join(chunks)
            yield _unshuffle_batch(block, bsz, n_channels, trace_samples, dtype=dtype)
            remaining -= bsz

def open_merged_dataset(base_dir: Path, energy: int):
    base = Path(base_dir)
    meta_path = base / f"meta_energy_{energy}.h5"
    traces_path = base / f"traces_energy_{energy}.zst"
    attrs, meta_df = read_meta_h5(meta_path)
    n_events = len(meta_df)
    n_channels = int(attrs["n_channels"])
    trace_samples = int(attrs["trace_samples"])
    trace_dtype = np.dtype(attrs.get("trace_dtype", np.float16))

    def get_iter(batch_size=200, dtype=trace_dtype, max_events=None):
        return iter_traces_zst_batched_merged(traces_path, n_events, n_channels, trace_samples,
                                              batch_size=batch_size, dtype=dtype, max_events=max_events)
    return attrs, meta_df, get_iter

# -----------------------------
# (2) IterableDataset
# -----------------------------
class MergedTrainingDataset(IterableDataset):
    """
    Streams batches from a single merged .zst file and aligns with metadata rows.
    Yields dicts:
      - "x": (B, C, T) float32
      - "mask": (B, C) float32
      - "pos": (B, 3) float32
      - "energy": (B,) float32
      - "cls": (B,) int64
    """
    def __init__(self, base_dir: Path, energy: int, batch_size: int = 64,
                 out_dtype=np.float32, max_events: int | None = None):
        super().__init__()
        self.base_dir = Path(base_dir)
        self.energy = energy
        self.batch_size = int(batch_size)
        self.out_dtype = np.dtype(out_dtype)
        self.max_events = max_events

        attrs, meta_df, get_iter = open_merged_dataset(self.base_dir, self.energy)
        self.attrs = attrs
        self.meta_df = meta_df.reset_index(drop=True)
        self.get_iter = get_iter

        self.n_events = len(self.meta_df)
        self.n_channels = int(attrs["n_channels"])
        self.trace_samples = int(attrs["trace_samples"])
        self.in_dtype = np.dtype(attrs.get("trace_dtype", np.float16))

        classes = sorted(self.meta_df["type_recoil"].unique())
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.num_classes = len(classes)

    def __iter__(self):
        wi = get_worker_info()
        if wi is not None and wi.num_workers > 1 and wi.id != 0:
            return iter(())  # only worker 0 reads the single stream

        meta_idx = 0
        for np_batch in self.get_iter(batch_size=self.batch_size, dtype=self.in_dtype, max_events=self.max_events):
            B = np_batch.shape[0]

            if self.out_dtype != np_batch.dtype:
                np_batch = np_batch.astype(self.out_dtype, copy=False)
            x = torch.from_numpy(np_batch)  # (B, C, T)

            mask = torch.ones((B, self.n_channels), dtype=torch.float32)

            meta_slice = self.meta_df.iloc[meta_idx: meta_idx + B]
            meta_idx += B

            pos = torch.tensor(meta_slice[["x", "y", "z"]].to_numpy(), dtype=torch.float32)
            energy = torch.tensor(meta_slice["energy"].to_numpy(), dtype=torch.float32)
            cls = torch.tensor([self.class_to_idx[s] for s in meta_slice["type_recoil"]], dtype=torch.long)

            yield {"x": x, "mask": mask, "pos": pos, "energy": energy, "cls": cls}

# -----------------------------
# (3) Training utils
# -----------------------------
def human_mb(t: torch.Tensor) -> float:
    return (t.element_size() * t.nelement()) / (1024**2)

@torch.no_grad()
def _trim_cache_every(n, step):
    if step % n == 0 and torch.cuda.is_available():
        torch.cuda.empty_cache()

def assert_model_on(device, model: nn.Module):
    for n, p in model.named_parameters(recurse=True):
        if p is not None and p.data.numel() and p.device != device:
            raise RuntimeError(f"Param {n} on {p.device}, expected {device}")
    for n, b in model.named_buffers(recurse=True):
        if b is not None and b.data.numel() and b.device != device:
            raise RuntimeError(f"Buffer {n} on {b.device}, expected {device}")

def train_one_epoch(model, loader, optimizer, scaler, device,
                    w_pos=1.0, w_energy=1.0, w_cls=1.0,
                    accum_steps=4, cache_trim_every=50):
    """Gradient accumulation to reduce peak VRAM."""
    model.train()
    loss_pos_fn = nn.SmoothL1Loss()
    loss_energy_fn = nn.MSELoss()
    loss_cls_fn = nn.CrossEntropyLoss()

    optimizer.zero_grad(set_to_none=True)
    steps = 0

    for batch in loader:
        # CPU tensors
        x_cpu = batch["x"]; m_cpu = batch["mask"]
        yp_cpu = batch["pos"]; ye_cpu = batch["energy"]; yc_cpu = batch["cls"]
        del batch

        B = x_cpu.shape[0]
        mb = max(1, B // accum_steps)
        if B % mb != 0:
            mb = 1

        for i in range(0, B, mb):
            x = x_cpu[i:i+mb].to(device, non_blocking=True)
            m = m_cpu[i:i+mb].to(device, non_blocking=True)
            yp = yp_cpu[i:i+mb].to(device, non_blocking=True)
            ye = ye_cpu[i:i+mb].to(device, non_blocking=True)
            yc = yc_cpu[i:i+mb].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                pos_pred, energy_pred, cls_logit, aux = model(x, channel_mask=m)
                if energy_pred.dim() == 2 and energy_pred.shape[1] == 1:
                    energy_pred = energy_pred.squeeze(1)

                loss_pos = loss_pos_fn(pos_pred, yp)
                loss_energy = loss_energy_fn(energy_pred, ye)
                loss_cls = loss_cls_fn(cls_logit, yc)
                loss = w_pos * loss_pos + w_energy * loss_energy + w_cls * loss_cls
                loss = loss * (mb / B)  # scale for accumulation

            scaler.scale(loss).backward()

            # free microbatch temporaries
            del x, m, yp, ye, yc, pos_pred, energy_pred, cls_logit, aux, loss_pos, loss_energy, loss_cls, loss

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        steps += 1
        _trim_cache_every(cache_trim_every, steps)

        # free CPU batch tensors
        del x_cpu, m_cpu, yp_cpu, ye_cpu, yc_cpu

    return {}

# -----------------------------
# (4) Main
# -----------------------------
if __name__ == "__main__":
    from tcxformer_v2 import ModelConfig, ModalityTCXFormer  # your file

    torch.manual_seed(0)

    # ---- Choose GPU explicitly and make it the default tensor device
    gpu_index = 0
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_index)                          # set default CUDA device
        device = torch.device(f"cuda:{gpu_index}")
        try:
            # PyTorch >= 2.0: make all new tensors default to this GPU
            torch.set_default_device(device)
        except Exception:
            pass
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    # ---- Model
    cfg = ModelConfig(
        n_photon=19, n_phonon=37,
        stride_photon=128, stride_phonon=256,
        d_model=256, d_ff=1024, n_heads=4,
        n_time_layers=3, n_chan_layers=1,
        patch_embed=64, dropout=0.1,
        rope_base=10000.0, n_branch_to_task=2,
    )
    model = ModalityTCXFormer(cfg).to(device)

    # sanity check: params & buffers are on the right device
    assert_model_on(device, model)

    # ---- Data
    BASE = Path("/ceph/dwong/work/training_samples/ER/small")
    ENERGY_BIN = 1000
    ds = MergedTrainingDataset(BASE, ENERGY_BIN, batch_size=64, out_dtype=np.float32, max_events=None)
    print(f"Classes: {ds.class_to_idx}  (num={ds.num_classes})")

    loader = DataLoader(
        ds,
        batch_size=None,       # dataset yields ready-made batches
        num_workers=0,         # keep 0/1 for single merged .zst
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=None,
    )

    # ---- Optimizer / AMP
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    try:
        for epoch in range(1, 4):
            stats = train_one_epoch(
                model, loader, optimizer, scaler, device,
                w_pos=1.0, w_energy=1.0, w_cls=1.0,
                accum_steps=16,
                cache_trim_every=50
            )
            print(f"[epoch {epoch}] done")
    finally:
        # Aggressive cleanup (useful in notebooks/REPL)
        del model, optimizer, scaler, loader, ds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
