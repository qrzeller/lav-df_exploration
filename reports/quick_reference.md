# Quick Reference Card - Simple Video Inference Pipeline

## 🎯 One-Line Commands

```bash
# Single video
python simple_inference.py --video video.mp4 --model checkpoint.ckpt

# Directory
python simple_inference.py --video videos/ --model checkpoint.ckpt

# With GPU
python simple_inference.py --video videos/ --model checkpoint.ckpt --device cuda
```

## 🐍 Python Code (Minimal)

```python
from dataset import create_simple_dataloader
from model.batfd_plus import BatfdPlus

model = BatfdPlus.load_from_checkpoint("checkpoint.ckpt")
dataloader = create_simple_dataloader("videos/")

for video, audio, n_frames, filename in dataloader:
    outputs = model(video, audio)
```

## 📊 Input/Output

**Input:** Video files (MP4, AVI, MOV, MKV, etc.)

**Output:** `.pt` files containing boundary maps
```python
{
    'fusion_boundary_map': Tensor,  # Detection scores
    'visual_boundary_map': Tensor,
    'audio_boundary_map': Tensor
}
```

## ⚙️ Common Options

| Option | Example | Description |
|--------|---------|-------------|
| `--batch-size` | `4` | Process 4 videos at once |
| `--num-workers` | `2` | Use 2 data loading workers |
| `--device` | `cuda` | Use GPU |
| `--output-dir` | `results/` | Save to custom directory |
| `--model-type` | `batfd` | Use BATFD model |

## 🚀 Performance Tips

- Use `--batch-size 4-8` for faster processing
- Add `--device cuda` for GPU acceleration
- Set `--num-workers 2-4` for parallel loading
- Reduce `--batch-size 1` if out of memory

## 📁 File Locations

| File | Description |
|------|-------------|
| `simple_inference.py` | CLI script |
| `dataset/simple_video.py` | Dataset class |
| `examples/` | Code examples |
| `reports/` | This report |

## 📚 Documentation

- **Quick Start**: `QUICKSTART.md`
- **Full Guide**: `SIMPLE_INFERENCE_GUIDE.md`
- **Report**: `reports/simple_video_pipeline_report.md`

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of memory | `--batch-size 1` |
| Slow | `--device cuda --batch-size 8` |
| Can't find video | Use absolute path |

## ✅ Test Pipeline

```bash
python test_simple_pipeline.py
```

---

**Need Help?** Read `reports/simple_video_pipeline_report.md`
