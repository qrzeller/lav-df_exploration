# Training Issues & Fixes Report (Run 02)

Date: 2025-12-07
Branch: mps
Workspace: lav-df_exploration

## Summary
During attempts to train `batfd` with LAV-DF, the run initially failed due to API incompatibilities and later due to CUDA OOM and data/shape mismatches. We implemented targeted fixes in model and dataset code and adjusted loss functions for variable temporal lengths. Training now starts and progresses with `num_frames=256` and AMP (16-mixed).

## Environment & Commands
- GPU: NVIDIA GeForce RTX 4090 (Tensor Cores)
- Python: 3.11 (venv)
- Precision: 16-mixed recommended
- Example command:
```
python train.py --config config/batfd_default_c.toml \
  --data_root dataset/LAV-DF/LAV-DF \
  --gpus 1 --precision 16-mixed --batch_size 1 --num_workers 8
```

## Issues Encountered
1. ReduceLROnPlateau TypeError: unsupported `verbose` arg.
2. Lightning detected multiple optimizers due to `training_step` signature.
3. CUDA OOM in 3D conv with `num_frames=512`, even at batch size 1.
4. Audio spectrogram assertion: expected `(64, 2048)` fixed to adapt to config.
5. Static `512` length frame labels causing BCE mismatch when `num_frames != 512`.
6. Masked loss slicing size mismatches when `n_frames > model T`.
7. Boundary module assertion when audio/video temporal dims misaligned.

## Fixes Implemented
- `model/batfd.py`:
  - Removed `verbose=True` from `ReduceLROnPlateau`.
  - `training_step(batch, batch_idx)` signature standardized (removed `optimizer_idx`, `hiddens`).
  - Frame labels sized dynamically: `frame_len = label.size(1)`.
- `model/batfd_plus.py`:
  - `training_step` signature standardized.
  - Frame labels sized dynamically: `frame_len = fusion_bm_label.size(1)`.
- `dataset/lavdf.py`:
  - `_get_log_mel_spectrogram` now instance method.
  - Uses `hop_length=160` and pads/crops temporal bins to `expected_bins = 4 * self.video_padding`.
- `loss.py`:
  - In `MaskedBMLoss`, `MaskedFrameLoss`, `MaskedContrastLoss`, `MaskedMSE`, clamp per-sample `frame` to `max_t = min(pred_T, true_T, frame)` to avoid mismatches.

## Current Status
- Sanity check and training start succeed with `config/batfd_default_c.toml` (`num_frames = 256`) and AMP `16-mixed`.
- Training iterates; run was manually interrupted during testing.

## Recommendations
- Memory/Compute:
  - Keep `num_frames` ≤ 256 for C3D; reduce further if needed.
  - Maintain resize to 96×96; for higher detail, add early spatial downsampling or center-crop.
  - Use `--precision 16-mixed`.
  - Set allocator: `PYTORCH_ALLOC_CONF=expandable_segments:true` (new var replacing deprecated `PYTORCH_CUDA_ALLOC_CONF`).
- Config Convenience:
  - Create small-footprint preset (e.g., `batfd_small.toml`: `num_frames=256`, smaller `hidden_dims`).
  - Optional flag to disable resize with center-crop in `dataset/lavdf.py`.

## Try-It Commands
- Recommended run:
```
export PYTORCH_ALLOC_CONF=expandable_segments:true
python train.py --config config/batfd_default_c.toml \
  --data_root dataset/LAV-DF/LAV-DF \
  --gpus 1 --precision 16-mixed --batch_size 1 --num_workers 8
```

## Next Steps (Optional)
- Add config flag to toggle resize and crop size.
- Tune `C3DVideoEncoder` pooling/stride for no-resize scenario.
- Add `config/batfd_small.toml` preset.

## Files Modified
- `model/batfd.py`
- `model/batfd_plus.py`
- `dataset/lavdf.py`
- `loss.py`

## Logs Snapshot
- Initial failures: TypeError (scheduler), OOM, assertion on audio shape, BCE/MSE mismatches, boundary module assertions.
- After fixes: training progresses with `num_frames=256` and AMP.
