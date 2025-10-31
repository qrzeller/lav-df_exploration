# Simple Video Inference Pipeline

This pipeline allows you to process new video files for deepfake detection without requiring metadata JSON files or labels.

## Overview

The pipeline consists of:

1. **SimpleVideoDataset** (`dataset/simple_video.py`): A PyTorch Dataset class that handles video loading and preprocessing
2. **simple_inference.py**: A standalone script for running inference on videos
3. **Helper functions**: Utilities for creating dataloaders and processing results

## Quick Start

### 1. Basic Usage - Single Video

```python
from dataset import create_simple_dataloader
from model.batfd_plus import BatfdPlus
import torch

# Load your trained model
model = BatfdPlus.load_from_checkpoint("path/to/checkpoint.ckpt")
model.eval()

# Create dataloader for a single video
dataloader = create_simple_dataloader("path/to/video.mp4", batch_size=1)

# Run inference
with torch.no_grad():
    for video, audio, n_frames, file_name in dataloader:
        outputs = model(video, audio)
        # Process outputs...
```

### 2. Process Multiple Videos from a Directory

```python
from dataset import create_simple_dataloader

# Automatically loads all video files from directory
dataloader = create_simple_dataloader(
    "path/to/videos/",
    batch_size=4,
    num_workers=2
)

for batch in dataloader:
    video, audio, n_frames, file_names = batch
    # Process batch...
```

### 3. Process Specific Video Files

```python
from dataset import SimpleVideoDataset
from torch.utils.data import DataLoader

# Create dataset with specific files
video_list = [
    "video1.mp4",
    "video2.mp4",
    "video3.mp4"
]

dataset = SimpleVideoDataset(
    video_paths=video_list,
    frame_padding=512,  # Default
    fps=25,  # Default
    return_file_name=True
)

dataloader = DataLoader(dataset, batch_size=1)
```

## Command Line Usage

### Process a single video
```bash
python simple_inference.py \
    --video path/to/video.mp4 \
    --model checkpoints/model.ckpt \
    --model-type batfd_plus \
    --output-dir output/results
```

### Process a directory of videos
```bash
python simple_inference.py \
    --video path/to/videos/ \
    --model checkpoints/model.ckpt \
    --output-dir output/results \
    --batch-size 4 \
    --num-workers 2
```

### Process specific videos
```bash
python simple_inference.py \
    --videos video1.mp4 video2.mp4 video3.mp4 \
    --model checkpoints/model.ckpt \
    --device cuda
```

## Programmatic API

### Using SimpleVideoDataset Directly

```python
from dataset import SimpleVideoDataset
import torch

# Create dataset
dataset = SimpleVideoDataset(
    video_paths="path/to/videos/",
    frame_padding=512,
    fps=25,
    return_file_name=True
)

# Access a single video
video, audio, n_frames, file_name = dataset[0]

print(f"Video shape: {video.shape}")  # (C, T, H, W) = (3, 512, 96, 96)
print(f"Audio shape: {audio.shape}")  # (64, 2048) - log mel spectrogram
print(f"Number of frames: {n_frames}")
print(f"File: {file_name}")
```

### Custom Processing

```python
from dataset import SimpleVideoDataset
from torch.utils.data import DataLoader

# Custom transforms
def my_video_transform(video):
    # Your custom processing
    return video

dataset = SimpleVideoDataset(
    video_paths=["video1.mp4", "video2.mp4"],
    video_transform=my_video_transform,
    return_file_name=True
)

# Create dataloader with custom settings
dataloader = DataLoader(
    dataset,
    batch_size=2,
    num_workers=4,
    pin_memory=True
)
```

## Output Format

The inference script saves results as PyTorch tensor files (`.pt`) containing:

### For BATFD model:
```python
{
    'fusion_boundary_map': Tensor,  # Shape: (max_duration, n_frames)
    'visual_boundary_map': Tensor,
    'audio_boundary_map': Tensor
}
```

### For BATFD+ model:
```python
{
    'fusion_boundary_map': Tensor,
    'fusion_start': Tensor,  # Start boundary scores
    'fusion_end': Tensor,    # End boundary scores
    'visual_boundary_map': Tensor,
    'visual_start': Tensor,
    'visual_end': Tensor,
    'audio_boundary_map': Tensor,
    'audio_start': Tensor,
    'audio_end': Tensor
}
```

## Advanced Examples

### 1. Batch Processing with Progress Tracking

```python
from dataset import create_simple_dataloader
from tqdm import tqdm
import torch

dataloader = create_simple_dataloader("videos/", batch_size=4)

model = load_model("model.ckpt")
model.eval()
model = model.cuda()

results = []
with torch.no_grad():
    for batch in tqdm(dataloader, desc="Processing videos"):
        video, audio, n_frames, file_names = batch
        video, audio = video.cuda(), audio.cuda()
        
        outputs = model(video, audio)
        
        # Store results
        for i in range(len(file_names)):
            results.append({
                'file': file_names[i],
                'n_frames': n_frames[i].item(),
                'boundary_map': outputs[0][i].cpu()
            })
```

### 2. Integration with Existing Inference Code

The SimpleVideoDataset produces outputs compatible with the existing inference pipeline:

```python
from dataset import SimpleVideoDataset
from inference import inference_batfd
from model.batfd_plus import BatfdPlus

# Load model
model = BatfdPlus.load_from_checkpoint("checkpoint.ckpt")

# Create dataset (mimics the structure expected by inference code)
dataset = SimpleVideoDataset(
    video_paths="videos/",
    frame_padding=512,
    return_file_name=True
)

# The outputs match the expected format:
# video: (C, T, H, W)
# audio: (64, 2048) 
# n_frames: int
# file_name: str
```

### 3. Processing with Post-Processing

```python
from dataset import create_simple_dataloader
from post_process import boundary_map_to_proposals
import torch

dataloader = create_simple_dataloader("videos/")
model = load_model("model.ckpt").cuda()

for video, audio, n_frames, file_names in dataloader:
    video, audio = video.cuda(), audio.cuda()
    
    # Get predictions
    fusion_bm, _, _, _, _, _, _, _, _ = model(video, audio)
    
    # Convert to proposals
    for i in range(len(file_names)):
        proposals = boundary_map_to_proposals(
            fusion_bm[i].cpu(),
            n_frames[i].item()
        )
        
        print(f"{file_names[i]}: Found {len(proposals)} potential manipulations")
```

## Supported Video Formats

The pipeline supports all video formats handled by `torchvision.io.read_video`:
- MP4
- AVI
- MOV
- MKV
- FLV
- WMV
- WebM

## Configuration

### Key Parameters

- **frame_padding** (default: 512): Number of frames to pad/truncate videos to
- **fps** (default: 25): Frames per second, used to calculate audio padding
- **batch_size** (default: 1): Number of videos to process simultaneously
- **num_workers** (default: 0): Number of parallel data loading workers

### Adjusting for Different Video Lengths

```python
# For longer videos (up to 40 seconds at 25fps)
dataset = SimpleVideoDataset(
    video_paths="long_videos/",
    frame_padding=1000,  # 40 seconds * 25 fps
    fps=25
)

# For shorter videos
dataset = SimpleVideoDataset(
    video_paths="short_videos/",
    frame_padding=250,  # 10 seconds * 25 fps
    fps=25
)
```

## Differences from LAV-DF Dataset

| Feature | LAV-DF Dataset | SimpleVideoDataset |
|---------|----------------|-------------------|
| Metadata Required | Yes (JSON file) | No |
| Labels Required | Yes | No |
| Use Case | Training/Evaluation | Inference only |
| Input | Dataset directory structure | Video files directly |
| Output | Video, Audio, Labels, Metadata | Video, Audio, Filename |

## Troubleshooting

### Issue: "Video file not found"
Make sure video paths are absolute or relative to the current working directory.

### Issue: "Wrong log mel-spectrogram setup"
The audio is expected to be at 16kHz sample rate. If your videos have different sample rates, they will be resampled automatically by `torchvision.io.read_video`.

### Issue: Out of memory
Reduce batch_size or frame_padding:
```python
dataloader = create_simple_dataloader(
    "videos/",
    batch_size=1,  # Process one at a time
    frame_padding=256  # Reduce length
)
```

## Next Steps

- Check `inference.py` for the full evaluation pipeline
- See `post_process.py` for converting boundary maps to temporal proposals
- Review model code in `model/` for understanding the outputs
