# Agent Instruction: Build a <5 Minute Training Smoke Test for 1D ResNet Triggering

## Objective

Implement a single training smoke test around the existing `ResNet1D` model and the existing notebook that already imports the traces. The goal is not to maximize physics performance yet. The goal is to make one end-to-end training run complete in under 5 minutes on a MacBook Pro M1 Pro 16 GB, while producing enough metrics to judge whether the setup is viable for the autoresearch harness.

The current model is a real 1D ResNet with:
- stem conv: `Conv1d(..., kernel_size=7, stride=2)` + maxpool
- residual stages built from `BasicBlock`
- adaptive average pooling
- final linear head

This means the model is meaningful, but the first training setup must be intentionally conservative to fit the time budget.

---

# 1. What you must do first

## Step 1 — Find the existing trace import code in the notebook
Open the notebook where the H5 traces are already imported.

### Your job
1. Find the exact cells that:
   - open the H5 file
   - load the 4000 signal traces
   - identify the array/dataset name
   - show the trace shape
2. Reuse that loading path instead of inventing a new data source.
3. Document:
   - file path
   - H5 dataset key
   - resulting loaded shape
   - dtype
4. If the notebook already preprocesses traces, note exactly what preprocessing is already done.

### Deliverable
Create a short markdown block in the notebook or script comments saying:
- where the traces are loaded
- what array is used for training
- what the final tensor shape is before cropping

Do not guess the dataset key. Read it from the notebook.

---

# 2. Build the smallest meaningful training dataset

## Step 2 — Use the crop window `[3000:7000]`
For each signal trace:
- crop sample indices `3000:7000`
- resulting length must be `4000`

Assume single-channel input for now.

## Step 3 — Build binary labels
Use:
- positive class = cropped signal traces from the H5 file
- negative class = pure noise traces generated separately

### Important
The first training run should use a balanced dataset:
- 4000 signal windows
- 4000 pure noise windows

Do not start with 2x, 5x, or 10x more negatives. Balanced is best for the first smoke test because:
- easier to debug
- simpler metrics
- shorter runtime
- cleaner interpretation

---

# 3. Data split rules

## Step 4 — Split train / val / test by trace, not after augmentation
Use:
- train: 70%
- val: 15%
- test: 15%

For the signal set:
- train: 2800
- val: 600
- test: 600

For the noise set:
- train: 2800
- val: 600
- test: 600

### Requirements
- shuffle with a fixed random seed
- do not leak the same original trace across splits
- store the split indices so later runs are reproducible

### Deliverable
Save split metadata in a small file, for example:
`artifacts/splits_v1.json`

---

# 4. Model choice for the first smoke test

## Step 5 — Use the smallest ResNet that is still meaningful
Instantiate:

```python
ResNet1D(
    in_channels=1,
    layers=[1, 1, 1],
    classes=1,
    kernel_size=7,
)
```

### Why this config
- meaningful architecture
- small enough for a fast benchmark
- less likely to exceed memory/time on M1 Pro
- good enough for a first autoresearch loop

Do not start with deeper variants such as `[2,2,2]` until timing is measured.

---

# 5. Training configuration for the <5 minute target

## Step 6 — Use this exact first training configuration
Use the following as the baseline smoke test:

```yaml
model:
  in_channels: 1
  layers: [1, 1, 1]
  classes: 1
  kernel_size: 7

data:
  crop_start: 2000
  crop_end: 8000
  normalize: per-trace zscore

train:
  batch_size: 32
  epochs: 3
  learning_rate: 0.001
  weight_decay: 0.0001
  optimizer: adamw
  loss: bce_with_logits
  pos_weight: 1.0
  device: mps
  num_workers: 0
```

### Notes
- `num_workers=0` is safer initially on macOS
- use `mps` if available, otherwise CPU fallback
- `epochs=3` is intentional for timing control
- `batch_size=32` is the safest starting point

---

# 6. Required preprocessing

## Step 7 — Keep preprocessing minimal
For the first run:
1. crop `[2000:8000]`
2. cast to `float32`
3. normalize per trace:
   - subtract mean
   - divide by std + epsilon
4. add channel dimension:
   - shape `(N, 1, 6000)`

Do not add heavy preprocessing yet:
- no augmentation
- no whitening
- no frequency transforms
- no random cropping
- no waveform alignment tricks

The goal is timing + pipeline validation.

---

# 7. Loss, optimizer, and metrics

## Step 8 — Use the correct training ingredients
Use:

```python
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

### Do not add scheduler in the first smoke test
A scheduler is not necessary for the first benchmark run. Add it only later if training is stable and time budget allows.

## Step 9 — Report these metrics
At the end of each epoch, compute:
- train loss
- val loss
- val ROC-AUC
- val PR-AUC

At end of training, compute on test set:
- test ROC-AUC
- test PR-AUC
- confusion matrix at threshold 0.5

Also report:
- train wall time in seconds
- mean inference latency per batch
- total parameter count

---

# 8. Timing protocol

## Step 10 — Measure time carefully
You must measure:
1. dataset preparation time
2. training time only
3. validation time
4. end-to-end wall time

### Required output
Print a compact timing summary like:
```text
dataset_prep_seconds: ...
training_seconds: ...
validation_seconds: ...
total_wall_seconds: ...
```

### Success criterion
The first smoke test is considered successful if:
- `total_wall_seconds < 300`

If it exceeds 300 seconds, do not guess why. Diagnose it.

---

# 9. If training is too slow, reduce in this order

## Step 11 — Use this fallback ladder, in order
If the first run exceeds 5 minutes, apply changes in this exact order:

### Fallback A
Reduce epochs:
- `3 -> 2`

### Fallback B
Reduce batch size search:
- try `16` and `32`
- keep whichever is faster on MPS

### Fallback C
Reduce training subset only for smoke test:
- use 2000 signal + 2000 noise
- keep val/test unchanged if possible, or scale proportionally

### Fallback D
Reduce crop length for timing-only benchmark
Only if necessary:
- crop a shorter window around the signal, e.g. `2500:6500` or another centered 4000-sample window

### Fallback E
Use CPU if MPS is unexpectedly slower or unstable
Benchmark both once if needed.

### Important
Do not deepen the model or add augmentations until the timing target is met.

---

# 10. What hyperparameters autoresearch should tune first

## Step 12 — Keep the first search space small
After the smoke test works, expose only these tunable parameters:

### Architecture
- `layers`: `[[1,1,1], [2,2,2]]`
- `kernel_size`: `[5, 7, 9]`

### Optimization
- `learning_rate`: `[3e-4, 1e-3, 3e-3]`
- `weight_decay`: `[1e-5, 1e-4, 1e-3]`
- `batch_size`: `[16, 32, 64]`

### Training policy
- `epochs`: `[2, 3, 5]`
- `pos_weight`: `[1.0, 1.5, 2.0]`

### Do not tune yet
Do not tune in stage 1:
- stride_type
- dilation
- norm type
- deep architecture families
- augmentation policy
- scheduler settings

That search space is too large for the current goal.

---

# 11. What the agent must actually run

## Step 13 — Execute one real training run
After implementing the data path and training loop, run exactly one baseline smoke test with:

- 4000 signal traces
- 4000 noise traces
- crop `[2000:8000]`
- `layers=[1,1,1]`
- `kernel_size=7`
- `batch_size=32`
- `epochs=3`
- `lr=1e-3`
- `weight_decay=1e-4`
- device = `mps` if available

### Required outputs
Save:
- `metrics.json`
- `history.json`
- `timing.json`

And print a concise terminal summary containing:
- train size / val size / test size
- model parameter count
- training time
- validation metrics
- test metrics

---

# 12. What to inspect after the first run

## Step 14 — Provide a diagnosis, not just numbers
After the first run, write a short assessment answering:

1. Did it finish under 5 minutes?
2. Which stage consumed most time?
   - data loading
   - preprocessing
   - training
   - validation
3. Was MPS actually used?
4. Was memory usage stable?
5. Did the model learn anything at all?
   - compare val ROC-AUC / PR-AUC against chance
6. Is this configuration suitable as the baseline for autoresearch loops?

### Deliverable
Write a short markdown summary:
`reports/first_training_assessment.md`

---

# 13. Minimal implementation requirements

## Step 15 — Implement only what is necessary
The training script must:
1. load traces from the same source as the notebook
2. crop `[2000:8000]`
3. combine signal + noise
4. split reproducibly
5. normalize per trace
6. instantiate `ResNet1D`
7. train with BCEWithLogitsLoss + AdamW
8. evaluate
9. save metrics and timing

That is enough.

Do not spend time on:
- fancy logging
- full harness integration
- advanced plotting
- packaging
- multi-run sweeps

The first target is a single, timed, real training run.

---

# 14. Expected conclusion target

The outcome we want is one of these two:

## Best case
A real training run completes in under 5 minutes on M1 Pro and produces nontrivial validation metrics. Then this becomes the baseline workload for the autoresearch harness.

## Acceptable case
The first run exceeds 5 minutes, but timing breakdown clearly shows how to reduce it using the fallback ladder. Then revise config and rerun once.

What is not acceptable:
- guessing timing without running
- inventing a data path instead of reading the notebook
- using a deeper model before benchmarking
- changing multiple factors at once and making diagnosis impossible

---

# 15. Final reminder to the agent

Your job is not to make the best classifier yet.

Your job is to:
- reuse the existing trace-import notebook logic
- build the smallest meaningful binary training dataset
- run one real training benchmark
- determine whether this workload can serve as the baseline autoresearch loop under a 5-minute budget
