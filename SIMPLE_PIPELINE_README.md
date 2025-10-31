# Simple Video Inference Pipeline - Quick Reference

## What is this?

A simple pipeline for running deepfake detection on **new videos** without needing metadata JSON files or labels. Perfect for inference on videos from any source.

## 🚀 Quick Start (3 steps)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run inference on your videos

**Single video:**
```bash
python simple_inference.py \
    --video my_video.mp4 \
    --model checkpoints/model.ckpt
```

**Folder of videos:**
```bash
python simple_inference.py \
    --video videos/ \
    --model checkpoints/model.ckpt \
    --batch-size 4
```

**Specific videos:**
```bash
python simple_inference.py \
    --videos video1.mp4 video2.mp4 video3.mp4 \
    --model checkpoints/model.ckpt
```

### 3. Get results

Results are saved in `output/simple_inference/` as `.pt` files containing boundary maps.

## 📝 Python API

### Minimal Example (5 lines)

```python
from dataset import create_simple_dataloader
from model.batfd_plus import BatfdPlus

model = BatfdPlus.load_from_checkpoint("checkpoint.ckpt")
dataloader = create_simple_dataloader("video.mp4")

for video, audio, n_frames, filename in dataloader:
    outputs = model(video, audio)
```

### Process a Directory

```python
from dataset import create_simple_dataloader

# Automatically finds all videos in the directory
dataloader = create_simple_dataloader("path/to/videos/", batch_size=4)

for batch in dataloader:
    # Process videos...
```

### Custom Processing

```python
from dataset import SimpleVideoDataset
from torch.utils.data import DataLoader

dataset = SimpleVideoDataset(
    video_paths=["video1.mp4", "video2.mp4"],
    frame_padding=512,
    fps=25,
    return_file_name=True
)

dataloader = DataLoader(dataset, batch_size=2)
```

## 📁 Files Added

- **`dataset/simple_video.py`**: Main dataset class
- **`simple_inference.py`**: Command-line inference script  
- **`SIMPLE_INFERENCE_GUIDE.md`**: Detailed documentation
- **`examples/simple_video_inference_example.py`**: Code examples

## 🔧 Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `frame_padding` | 512 | Max frames to process |
| `fps` | 25 | Frames per second |
| `batch_size` | 1 | Videos per batch |
| `num_workers` | 0 | Data loading workers |

## 📊 Input/Output

**Input:** 
- Single video file (`.mp4`, `.avi`, `.mov`, etc.)
- Directory containing videos
- List of video paths

**Output:**
```python
{
    'fusion_boundary_map': Tensor,    # (max_duration, n_frames)
    'visual_boundary_map': Tensor,
    'audio_boundary_map': Tensor,
    'fusion_start': Tensor,           # For BATFD+ only
    'fusion_end': Tensor,             # For BATFD+ only
    # ... (visual/audio start/end for BATFD+)
}
```

## ⚙️ Command Line Options

```bash
python simple_inference.py --help

Required:
  --video PATH         Single video or directory
  --videos PATH [...]  List of videos
  --model PATH         Model checkpoint

Optional:
  --model-type TYPE    batfd or batfd_plus (default: batfd_plus)
  --output-dir PATH    Where to save results (default: output/simple_inference)
  --batch-size N       Batch size (default: 1)
  --num-workers N      Data loading workers (default: 0)
  --device DEVICE      auto/cpu/cuda/mps (default: auto)
```

## 🎯 Use Cases

✅ **Good for:**
- Running inference on new, unlabeled videos
- Testing models on real-world data
- Batch processing video collections
- Quick experiments without dataset setup

❌ **Not for:**
- Training models (use `LavdfDataModule`)
- Evaluation with metrics (use original `inference.py`)

## 🔍 Differences from Original Dataset

| Feature | LAV-DF Dataset | SimpleVideoDataset |
|---------|----------------|-------------------|
| Requires metadata.json | ✅ Yes | ❌ No |
| Requires labels | ✅ Yes | ❌ No |
| Use case | Training/Eval | Inference |
| Input | Dataset structure | Any videos |

## 📚 More Information

- Full guide: [`SIMPLE_INFERENCE_GUIDE.md`](SIMPLE_INFERENCE_GUIDE.md)
- Examples: [`examples/simple_video_inference_example.py`](examples/simple_video_inference_example.py)
- Original dataset: [`dataset/lavdf.py`](dataset/lavdf.py)

## 💡 Tips

**Speed up processing:**
```bash
python simple_inference.py --video videos/ --batch-size 8 --num-workers 4
```

**GPU acceleration:**
```bash
python simple_inference.py --video videos/ --device cuda
```

**Handle longer videos:**
```python
dataset = SimpleVideoDataset(video_paths="long_videos/", frame_padding=1000)
```

## 🐛 Troubleshooting

**Out of memory?**
- Reduce `batch_size` to 1
- Reduce `frame_padding` (e.g., to 256)

**Can't find videos?**
- Use absolute paths
- Check file extensions (`.mp4`, `.avi`, `.mov`, etc.)

**Import errors?**
```bash
pip install -r requirements.txt
```

## 📄 License

Same as the main LAV-DF project. See LICENSE and TERMS_AND_CONDITIONS.md.
