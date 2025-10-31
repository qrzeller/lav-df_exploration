# New Video Inference Pipeline - Summary

## Overview

A complete pipeline has been created for processing new video files for deepfake detection **without requiring metadata JSON files or labels**. This is ideal for inference on videos from any source.

## What Was Created

### Core Components

1. **`dataset/simple_video.py`** - Main dataset class
   - `SimpleVideoDataset`: PyTorch Dataset for loading and preprocessing videos
   - `create_simple_dataloader()`: Helper function to create DataLoaders easily
   - Handles single files, directories, or lists of video paths
   - Compatible with existing model architectures

2. **`simple_inference.py`** - Command-line inference script
   - Run inference from the command line
   - Supports BATFD and BATFD+ models
   - Batch processing with configurable options
   - Saves results as PyTorch tensor files

3. **Documentation**
   - `SIMPLE_PIPELINE_README.md`: Quick reference guide
   - `SIMPLE_INFERENCE_GUIDE.md`: Comprehensive documentation with examples
   - `examples/simple_video_inference_example.py`: Code examples

4. **Testing**
   - `test_simple_pipeline.py`: Automated tests for the pipeline

### Updated Files

- `dataset/__init__.py`: Added exports for new classes

## Key Features

✅ **No Metadata Required**: Process any video file directly
✅ **Flexible Input**: Single file, directory, or list of files
✅ **Compatible**: Works with existing BATFD/BATFD+ models
✅ **Easy to Use**: Simple API and command-line interface
✅ **Batch Processing**: Process multiple videos efficiently
✅ **Well Documented**: Complete examples and guides

## Usage Examples

### Command Line

```bash
# Single video
python simple_inference.py --video video.mp4 --model checkpoint.ckpt

# Directory
python simple_inference.py --video videos/ --model checkpoint.ckpt --batch-size 4

# Multiple specific videos
python simple_inference.py --videos v1.mp4 v2.mp4 v3.mp4 --model checkpoint.ckpt
```

### Python API

```python
# Minimal example
from dataset import create_simple_dataloader
from model.batfd_plus import BatfdPlus

model = BatfdPlus.load_from_checkpoint("checkpoint.ckpt")
dataloader = create_simple_dataloader("videos/")

for video, audio, n_frames, filename in dataloader:
    outputs = model(video, audio)
    # Process outputs...
```

## Architecture

```
Input Videos
    ↓
SimpleVideoDataset
    ├─ Read video with torchvision
    ├─ Pad/truncate to fixed length (default: 512 frames)
    ├─ Resize frames to 96x96
    ├─ Convert audio to log mel spectrogram
    └─ Return: (video, audio, n_frames, filename)
    ↓
DataLoader (batch processing)
    ↓
Model (BATFD/BATFD+)
    ↓
Boundary Maps & Scores
    ↓
Save Results (.pt files)
```

## Comparison with Original Dataset

| Aspect | LAV-DF Dataset | SimpleVideoDataset |
|--------|----------------|-------------------|
| **Purpose** | Training & Evaluation | Inference Only |
| **Input Required** | metadata.min.json + labels | Just video files |
| **Output** | Video, Audio, Labels, Metadata | Video, Audio, Filename |
| **Use Case** | Supervised learning | Inference on new data |
| **Setup** | Complex (dataset structure) | Simple (any videos) |

## File Structure

```
LAV-DF/
├── dataset/
│   ├── __init__.py (updated)
│   ├── lavdf.py (original)
│   └── simple_video.py (NEW)
├── examples/
│   └── simple_video_inference_example.py (NEW)
├── simple_inference.py (NEW)
├── test_simple_pipeline.py (NEW)
├── SIMPLE_PIPELINE_README.md (NEW)
└── SIMPLE_INFERENCE_GUIDE.md (NEW)
```

## Integration Points

The pipeline integrates seamlessly with existing code:

1. **Models**: Works with BATFD (`model/batfd.py`) and BATFD+ (`model/batfd_plus.py`)
2. **Utils**: Uses existing `read_video()`, `padding_video()`, etc.
3. **Post-processing**: Compatible with `post_process.py` for converting boundary maps to proposals
4. **Output Format**: Matches expected tensor shapes for model input

## Parameters & Configuration

### SimpleVideoDataset

```python
SimpleVideoDataset(
    video_paths: str | List[str],  # Path(s) to video(s)
    frame_padding: int = 512,       # Target frame count
    fps: int = 25,                  # Frames per second
    video_transform: Callable = Identity(),
    audio_transform: Callable = Identity(),
    return_file_name: bool = True
)
```

### create_simple_dataloader

```python
create_simple_dataloader(
    video_paths: str | List[str],
    batch_size: int = 1,
    num_workers: int = 0,
    frame_padding: int = 512,
    fps: int = 25,
    **dataset_kwargs
)
```

### simple_inference.py

```bash
--video PATH              # Single video or directory
--videos PATH [...]       # List of videos
--model PATH              # Model checkpoint (required)
--model-type TYPE         # batfd or batfd_plus (default: batfd_plus)
--output-dir PATH         # Output directory (default: output/simple_inference)
--batch-size N            # Batch size (default: 1)
--num-workers N           # Workers (default: 0)
--device DEVICE           # auto/cpu/cuda/mps (default: auto)
```

## Output Format

Results are saved as `.pt` files containing:

### BATFD Model
```python
{
    'fusion_boundary_map': Tensor,   # (max_duration, n_frames)
    'visual_boundary_map': Tensor,
    'audio_boundary_map': Tensor
}
```

### BATFD+ Model
```python
{
    'fusion_boundary_map': Tensor,
    'fusion_start': Tensor,
    'fusion_end': Tensor,
    'visual_boundary_map': Tensor,
    'visual_start': Tensor,
    'visual_end': Tensor,
    'audio_boundary_map': Tensor,
    'audio_start': Tensor,
    'audio_end': Tensor
}
```

## Testing

Run the test suite to verify the pipeline:

```bash
python test_simple_pipeline.py
```

Tests cover:
- Single video loading
- Directory processing
- DataLoader creation
- Batch iteration
- Video preprocessing

## Common Use Cases

1. **Quick Test on New Videos**
   ```bash
   python simple_inference.py --video test_video.mp4 --model model.ckpt
   ```

2. **Batch Process a Collection**
   ```bash
   python simple_inference.py --video dataset/ --model model.ckpt --batch-size 8
   ```

3. **Custom Processing in Python**
   ```python
   from dataset import SimpleVideoDataset
   dataset = SimpleVideoDataset("videos/", frame_padding=1000)
   # Custom processing loop...
   ```

4. **Integration with Existing Code**
   ```python
   from dataset import create_simple_dataloader
   from post_process import boundary_map_to_proposals
   
   dataloader = create_simple_dataloader("videos/")
   # Use with existing inference/post-processing pipelines
   ```

## Next Steps

1. **Run Tests**: `python test_simple_pipeline.py`
2. **Try Examples**: See `examples/simple_video_inference_example.py`
3. **Read Documentation**: Check `SIMPLE_INFERENCE_GUIDE.md`
4. **Run Inference**: Use `simple_inference.py` on your videos

## Support & Troubleshooting

- **Full Documentation**: `SIMPLE_INFERENCE_GUIDE.md`
- **Quick Reference**: `SIMPLE_PIPELINE_README.md`
- **Code Examples**: `examples/simple_video_inference_example.py`
- **Test Suite**: `test_simple_pipeline.py`

## Supported Video Formats

All formats supported by torchvision:
- MP4, AVI, MOV, MKV, FLV, WMV, WebM

## Performance Tips

- Use `--batch-size` > 1 for faster processing
- Set `--num-workers` > 0 for parallel data loading
- Use `--device cuda` for GPU acceleration
- Reduce `frame_padding` if running out of memory

## License

Same as LAV-DF project. See LICENSE and TERMS_AND_CONDITIONS.md.
