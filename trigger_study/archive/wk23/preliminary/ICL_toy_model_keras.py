"""
Keras implementation of the LSTM+Transformer backbone for position reconstruction (PR),
position+classification (PRC), and position+classification+energy (PRCE).

This file defines:
- build_backbone(): shared feature extractor (per-detector BiLSTM -> 20-D features -> Transformer)
- build_pr_model(): PR head (x,y,z)
- build_prc_model(): PR + classifier (ER/NR logit)
- build_prce_model(): PR + classifier + energy head (non-negative)
- make_datasets(): Example tf.data pipelines (replace the dummy loader with yours)
- train(): One-stop training function with mixed precision, AdamW, cosine restarts, clipnorm

Input shape expected by the models: (time, detectors) or (detectors, time)?
We use (detectors, time). If your data is (time, detectors), transpose before feeding.

Why? We apply a shared per-detector submodel to each detector's time series. We wrap the
per-detector submodel in TimeDistributed over the detector axis.

Notes:
- Long sequences (e.g., time=16384) are heavy; ensure you use tf.data and GPU.
- CosineDecayRestarts is step-based; we compute steps_per_epoch from the dataset for the schedule.
- Mixed precision is enabled by default (set policy at the start of train()).
"""
from __future__ import annotations
import math
from typing import Tuple, Optional, List, Dict

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers


# ---------------------------
# Utility blocks
# ---------------------------

def gelu(x):
    return tf.keras.activations.gelu(x)


class TransformerEncoder(layers.Layer):
    """Stackable Transformer encoder block.

    Args:
        num_heads: number of attention heads (e.g., 3)
        d_model: model width (feature dim, e.g., 20)
        d_ff: hidden width in the feed-forward (e.g., 128)
        dropout: dropout rate (e.g., 0.2)
    Input:
        (batch, seq_len, d_model)
    Output:
        (batch, seq_len, d_model)
    """
    def __init__(self, num_heads: int, d_model: int, d_ff: int, dropout: float, **kwargs):
        super().__init__(**kwargs)
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model, dropout=dropout)
        self.dropout1 = layers.Dropout(dropout)
        self.norm1 = layers.LayerNormalization(epsilon=1e-5)

        self.ffn = keras.Sequential([
            layers.Dense(d_ff, activation=gelu),
            layers.Dropout(dropout),
            layers.Dense(d_model),
        ])
        self.dropout2 = layers.Dropout(dropout)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)

    def call(self, x, training=False):
        # Self-attention
        attn_output = self.mha(x, x, training=training)  # (B, S, D)
        attn_output = self.dropout1(attn_output, training=training)
        x = self.norm1(x + attn_output)
        # Feed-forward
        ffn_output = self.ffn(x, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        x = self.norm2(x + ffn_output)
        return x


def build_per_detector_lstm(time_steps: int, lstm_units: int = 64, dropout: float = 0.3, proj_dim: int = 20) -> keras.Model:
    """Builds a shared submodel that processes a single detector's time series.

    Input:  (time_steps, 1)
    Output: (proj_dim,) feature vector
    """
    inp = keras.Input(shape=(time_steps, 1))
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True, dropout=dropout))(inp)
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True, dropout=dropout))(x)
    # Final LSTM returns the last output (summary over time)
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=False, dropout=dropout))(x)
    # Project to Transformer feature dim
    out = layers.Dense(proj_dim, activation=None)(x)
    return keras.Model(inp, out, name="per_detector_lstm")


def build_backbone(
    time_steps: int,
    num_detectors: int = 56,
    lstm_units: int = 64,
    lstm_dropout: float = 0.3,
    proj_dim: int = 20,
    num_transformer_layers: int = 5,
    num_heads: int = 3,
    d_ff: int = 128,
    attn_dropout: float = 0.2,
) -> keras.Model:
    """Builds the shared backbone: TimeDistributed(per-detector BiLSTM) -> Transformer stack.

    Input:  (num_detectors, time_steps)
    Output: (num_detectors, proj_dim) features after Transformer
    """
    inp = keras.Input(shape=(num_detectors, time_steps))

    # Expand a channel dim for per-detector series: (B, D, T, 1)
    x = layers.Lambda(lambda t: tf.expand_dims(t, axis=-1))(inp)

    # Shared per-detector LSTM submodel
    per_det = build_per_detector_lstm(time_steps, lstm_units, lstm_dropout, proj_dim)

    # Apply to each detector: (B, D, proj_dim)
    x = layers.TimeDistributed(per_det, name="per_detector_encoder")(x)

    # Transformer expects (B, seq_len=D, d_model=proj_dim)
    for i in range(num_transformer_layers):
        x = TransformerEncoder(num_heads=num_heads, d_model=proj_dim, d_ff=d_ff, dropout=attn_dropout, name=f"tx_enc_{i}")(x)

    return keras.Model(inp, x, name="backbone_lstm_transformer")


# ---------------------------
# Heads and full models
# ---------------------------

def _mlp_head(x, width=256, act=gelu, dropout=0.0, name_prefix: str = "mlp"):
    x = layers.Dense(width, activation=act, name=f"{name_prefix}_dense")(x)
    if dropout > 0:
        x = layers.Dropout(dropout, name=f"{name_prefix}_dropout")(x)
    return x


def build_pr_model(time_steps: int, num_detectors: int = 56) -> keras.Model:
    backbone = build_backbone(time_steps=time_steps, num_detectors=num_detectors)
    inp = backbone.inputs[0]
    x = backbone.outputs[0]
    x = layers.Flatten()(x)
    x = _mlp_head(x, width=256, act=gelu, dropout=0.0, name_prefix="pos")
    pos = layers.Dense(3, activation=None, name="position_xyz")(x)
    return keras.Model(inp, pos, name="PR")


def build_prc_model(time_steps: int, num_detectors: int = 56) -> keras.Model:
    backbone = build_backbone(time_steps=time_steps, num_detectors=num_detectors)
    inp = backbone.inputs[0]
    x = backbone.outputs[0]
    x = layers.Flatten()(x)
    shared = _mlp_head(x, width=256, act=gelu, dropout=0.0, name_prefix="shared")
    pos = layers.Dense(3, activation=None, name="position_xyz")(shared)
    clf = layers.Dense(1, activation=None, name="er_nr_logit")(shared)  # logits
    return keras.Model(inp, {"position_xyz": pos, "er_nr_logit": clf}, name="PRC")


def build_prce_model(time_steps: int, num_detectors: int = 56) -> keras.Model:
    backbone = build_backbone(time_steps=time_steps, num_detectors=num_detectors)
    inp = backbone.inputs[0]
    x = backbone.outputs[0]
    x = layers.Flatten()(x)

    # Position head
    pos_h = _mlp_head(x, width=256, act=gelu, dropout=0.0, name_prefix="pos")
    pos = layers.Dense(3, activation=None, name="position_xyz")(pos_h)

    # Classification head (separate MLP; ends with logits)
    clf_h = _mlp_head(x, width=256, act=gelu, dropout=0.0, name_prefix="clf")
    clf = layers.Dense(1, activation=None, name="er_nr_logit")(clf_h)

    # Energy head (non-negative output)
    en_h = _mlp_head(x, width=256, act="relu", dropout=0.0, name_prefix="energy")
    energy = layers.Dense(1, activation="relu", name="energy_eV")(en_h)

    return keras.Model(inp, {"position_xyz": pos, "er_nr_logit": clf, "energy_eV": energy}, name="PRCE")


# ---------------------------
# Data pipeline (example)
# ---------------------------

def _dummy_generator(n_samples: int, num_detectors: int, time_steps: int):
    """Replace with your real loader. Yields (x, y_dict) for PRCE example.

    Input X shape: (num_detectors, time_steps)
    position_xyz: (3,)  | er_nr_logit target: (1,) -> labels 0/1 | energy_eV: (1,)
    """
    import numpy as np
    rng = np.random.default_rng(0)
    for _ in range(n_samples):
        x = rng.standard_normal((num_detectors, time_steps)).astype("float32")
        pos = rng.uniform(low=[-140, -140, -120], high=[140, 140, 120]).astype("float32")
        er = rng.integers(0, 2, size=(1,), dtype=np.int32).astype("float32")
        energy = rng.lognormal(mean=2.0, sigma=1.0, size=(1,)).astype("float32")
        yield x, {"position_xyz": pos, "er_nr_logit": er, "energy_eV": energy}


def make_datasets(
    num_detectors: int,
    time_steps: int,
    batch_size: int,
    n_train: int,
    n_val: int,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    train = tf.data.Dataset.from_generator(
        lambda: _dummy_generator(n_train, num_detectors, time_steps),
        output_signature=(
            tf.TensorSpec(shape=(num_detectors, time_steps), dtype=tf.float32),
            {
                "position_xyz": tf.TensorSpec(shape=(3,), dtype=tf.float32),
                "er_nr_logit": tf.TensorSpec(shape=(1,), dtype=tf.float32),
                "energy_eV": tf.TensorSpec(shape=(1,), dtype=tf.float32),
            },
        ),
    )
    val = tf.data.Dataset.from_generator(
        lambda: _dummy_generator(n_val, num_detectors, time_steps),
        output_signature=(
            tf.TensorSpec(shape=(num_detectors, time_steps), dtype=tf.float32),
            {
                "position_xyz": tf.TensorSpec(shape=(3,), dtype=tf.float32),
                "er_nr_logit": tf.TensorSpec(shape=(1,), dtype=tf.float32),
                "energy_eV": tf.TensorSpec(shape=(1,), dtype=tf.float32),
            },
        ),
    )

    def _prep(batch):
        x, y = batch
        # Center Z if needed, log-normalize energy target, etc. (hook for real preprocessing)
        return x, y

    train = train.shuffle(4 * batch_size).batch(batch_size).map(_prep).prefetch(tf.data.AUTOTUNE)
    val = val.batch(batch_size).map(_prep).prefetch(tf.data.AUTOTUNE)
    return train, val


# ---------------------------
# Training helper
# ---------------------------

def compile_model(model: keras.Model, steps_per_epoch: int, initial_lr: float = 3e-4, weight_decay: float = 0.04, clipnorm: float = 1.0):
    # Cosine restarts every 30 epochs -> in steps
    first_decay_steps = max(1, 30 * steps_per_epoch)
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=initial_lr,
        first_decay_steps=first_decay_steps,
        t_mul=1.0,
        m_mul=1.0,
        alpha=1e-8 / initial_lr,
    )

    opt = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=weight_decay, clipnorm=clipnorm)

    # Losses
    losses = {}
    metrics = {}
    if "position_xyz" in model.output_names:
        losses["position_xyz"] = tf.keras.losses.MeanSquaredError()
        metrics["position_xyz"] = [tf.keras.metrics.MeanAbsoluteError(name="mae")]
    if "er_nr_logit" in model.output_names:
        losses["er_nr_logit"] = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        metrics["er_nr_logit"] = [tf.keras.metrics.AUC(from_logits=True, name="auc")]
    if "energy_eV" in model.output_names:
        losses["energy_eV"] = tf.keras.losses.MeanSquaredError()
        metrics["energy_eV"] = [tf.keras.metrics.MeanAbsoluteError(name="mae")]

    if not losses:  # PR model single output named 'position_xyz'
        loss = tf.keras.losses.MeanSquaredError()
        metrics = [tf.keras.metrics.MeanAbsoluteError(name="mae")]
        model.compile(optimizer=opt, loss=loss, metrics=metrics)
    else:
        model.compile(optimizer=opt, loss=losses, metrics=metrics)


def train(
    task: str = "PRCE",
    time_steps: int = 16384,
    num_detectors: int = 56,
    batch_size: int = 128,
    epochs: int = 300,
    n_train: int = 1024,
    n_val: int = 256,
    mixed_precision: bool = True,
    outdir: Optional[str] = None,
):
    """Train one of the three tasks on dummy data. Replace make_datasets with your real loader.

    Args mirror the paper's defaults where applicable:
    - batch_size: 128 for PR/PRC; 256 for PRCE (adjust here)
    - epochs: 600 for PR, 300 for PRC/PRCE (adjust here)
    """
    if mixed_precision:
        from tensorflow.keras import mixed_precision as mp
        mp.set_global_policy("mixed_float16")

    # Choose model
    task = task.upper()
    if task == "PR":
        model = build_pr_model(time_steps, num_detectors)
    elif task == "PRC":
        model = build_prc_model(time_steps, num_detectors)
    elif task == "PRCE":
        model = build_prce_model(time_steps, num_detectors)
    else:
        raise ValueError("task must be one of: PR, PRC, PRCE")

    # Data
    train_ds, val_ds = make_datasets(num_detectors, time_steps, batch_size, n_train, n_val)

    # Steps for scheduler
    steps_per_epoch = math.ceil(n_train / batch_size)
    compile_model(model, steps_per_epoch)

    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath=(outdir+"/weights.keras" if outdir else "weights.keras"), save_weights_only=True, save_best_only=True, monitor="val_loss"),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True),
        keras.callbacks.CSVLogger((outdir+"/history.csv" if outdir else "history.csv"), append=True),
        keras.callbacks.TerminateOnNaN(),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    return model, history


