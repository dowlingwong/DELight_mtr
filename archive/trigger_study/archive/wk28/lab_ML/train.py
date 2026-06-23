import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from archive.trigger_study.archive.wk28.lab_ML.denoiser import CNNDenoiser1DResidual


class H5PairDataset(Dataset):
    """
    Lazy loader over multiple H5 files with paired noisy/clean traces.
    """

    def __init__(self, files: List[Path]):
        self.files = list(files)
        if not self.files:
            raise ValueError("No H5 files found for training.")

        self.file_lengths = []
        for fp in self.files:
            with h5py.File(fp, "r") as f:
                self.file_lengths.append(len(f["traces"]))
        self.cum = np.cumsum([0] + self.file_lengths)

    def __len__(self) -> int:
        return int(self.cum[-1])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_idx = np.searchsorted(self.cum, idx, side="right") - 1
        inner_idx = idx - self.cum[file_idx]
        fp = self.files[file_idx]
        with h5py.File(fp, "r") as f:
            noisy = np.asarray(f["traces"][inner_idx], dtype=np.float32)
            clean = np.asarray(f["traces_clean"][inner_idx], dtype=np.float32)
        return torch.from_numpy(noisy), torch.from_numpy(clean)


def _freeze_encoder_blocks(model: torch.nn.Module) -> None:
    """
    Freeze encoder/downsample blocks on the residual model for fine-tuning.
    """
    for name in ("enc1", "enc2", "down1", "enc3", "down2", "bottleneck"):
        if hasattr(model, name):
            for p in getattr(model, name).parameters():
                p.requires_grad = False


def train_model(
    model_name: str,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    ckpt_dir: Path,
    *,
    resume: bool = True,
    load_weights_from: Optional[Path] = None,
    freeze_encoder: bool = False,
):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{model_name}_latest.pt"
    final_path = ckpt_dir / f"{model_name}_final.pt"

    if load_weights_from is not None:
        state = torch.load(load_weights_from, map_location=device)
        model.load_state_dict(state["model"])
        print(f"[{model_name}] Loaded weights from {load_weights_from}")

    if freeze_encoder:
        _freeze_encoder_blocks(model)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found after freezing.")

    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    criterion = nn.L1Loss()

    start_epoch = 0
    if resume and load_weights_from is None and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state.get("epoch", 0)
        print(f"[{model_name}] Resumed from {ckpt_path} at epoch {start_epoch}")

    model.to(device)
    model.train()
    for epoch in range(start_epoch, epochs):
        running = 0.0
        for noisy, clean in loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            optimizer.zero_grad()
            pred = model(noisy)
            loss = criterion(pred, clean)
            loss.backward()
            optimizer.step()

            running += loss.item() * noisy.size(0)

        epoch_loss = running / len(loader.dataset)
        print(f"[{model_name}] Epoch {epoch + 1}/{epochs} | loss {epoch_loss:.4e}")

        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch + 1}, ckpt_path)

    torch.save({"model": model.state_dict()}, final_path)
    print(f"[{model_name}] Saved final model to {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Train CNN denoisers on paired H5 traces.")
    parser.add_argument("--data-dir", type=Path, default=Path("/ceph/srv/dwong/training_samples_h5/wk28_pairs"), help="Folder with *pair_batch_*.h5")
    parser.add_argument("--max-files", type=int, default=None, help="Limit number of H5 files for quick runs.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=2, help="Epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (keep low for h5py).")
    parser.add_argument("--mode", type=str, default="pretrain", choices=["pretrain", "finetune"], help="Training mode.")
    parser.add_argument("--finetune-from", type=Path, default=None, help="Checkpoint to load weights from when fine-tuning.")
    parser.add_argument("--freeze-encoder", action="store_true", help="Freeze encoder/downsample blocks during fine-tuning.")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from the latest checkpoint if present (ignored for fine-tune).",
    )
    parser.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints"), help="Checkpoint directory.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    files = sorted(args.data_dir.glob("*pair_batch_*.h5"))
    if args.max_files is not None:
        files = files[: args.max_files]
    dataset = H5PairDataset(files)
    sample_noisy, _ = dataset[0]
    in_channels = sample_noisy.shape[0]
    trace_len = sample_noisy.shape[-1]
    print(f"found {len(dataset)} events | files={len(files)} | channels={in_channels} len={trace_len}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=args.num_workers > 0,
    )

    model = CNNDenoiser1DResidual(in_channels=in_channels)

    load_weights_from = None
    resume = args.resume
    freeze_encoder = False
    if args.mode == "finetune":
        resume = False  # fine-tune should start fresh optimizer even when loading weights
        load_weights_from = args.finetune_from
        freeze_encoder = args.freeze_encoder
        if load_weights_from is None:
            fallback = args.ckpt_dir / "cnn_residual_final.pt"
            if fallback.exists():
                load_weights_from = fallback
                print(f"[cnn_residual] No --finetune-from provided, using {fallback}")
            else:
                raise ValueError("Fine-tune mode requires --finetune-from or an existing final checkpoint.")

    train_model(
        "cnn_residual",
        model,
        loader,
        device,
        args.epochs,
        args.lr,
        args.ckpt_dir,
        resume=resume,
        load_weights_from=load_weights_from,
        freeze_encoder=freeze_encoder,
    )


if __name__ == "__main__":
    main()
