Problem: running evaluate.py on macOS with MPS hit Lightning errors and very slow throughput (~0.5 it/s).
Root causes we investigated: device/accelerator config, Lightning callback signature mismatch, high IO/preprocessing cost, and heavy model compute (3D convs).
What I changed / added (actions taken)

Fixed Trainer config in inference.py to detect MPS and set accelerator/devices correctly (avoid passing None for GPU devices).
Fixed SaveToCsvCallback.on_predict_batch_end signature to be compatible with Lightning.
Added lightweight profiler instrumentation and dataset timing while iterating:
Profiling callback (measured model-predict timing).
Dataset per-item timing (to measure IO + preprocess).
Added a small profiling tool: profile_quick.py (cProfile-based) to:
Profile dataset __getitem__ (single-process, num_workers=0) and save io.prof.
Profile model forward (with option to pre-move batches to device) and save model.prof.
Ran the profiler and inspected results (you ran the script locally).

___

Main hotspot: torch.conv3d inside video_encoder (the C3D/MViT encoder). This is the dominant cost.