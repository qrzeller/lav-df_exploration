# Quick Start: Video Inference Pipeline

Get started with video inference in 60 seconds!

## 📦 Installation

```bash
pip install -r requirements.txt
```

## 🚀 Three Ways to Use

### 1. Command Line (Easiest)

```bash
# Single video
python simple_inference.py --video my_video.mp4 --model checkpoint.ckpt

# Folder
python simple_inference.py --video videos/ --model checkpoint.ckpt

# Multiple files
python simple_inference.py --videos v1.mp4 v2.mp4 --model checkpoint.ckpt
```

### 2. Python Script (5 lines)

```python
from dataset import create_simple_dataloader
from model.batfd_plus import BatfdPlus

model = BatfdPlus.load_from_checkpoint("checkpoint.ckpt")
dataloader = create_simple_dataloader("videos/")

for video, audio, n_frames, filename in dataloader:
    outputs = model(video, audio)
```

### 3. Jupyter Notebook (Interactive)

Open `examples/simple_video_inference.ipynb` and run the cells.

## 📊 What You Get

Results saved in `output/simple_inference/` as `.pt` files:

```python
{
    'fusion_boundary_map': Tensor,  # Deepfake detection scores
    'visual_boundary_map': Tensor,
    'audio_boundary_map': Tensor,
    # ... (more for BATFD+ model)
}
```

## 🎯 Common Commands

```bash
# Process with GPU
python simple_inference.py --video videos/ --model model.ckpt --device cuda

# Batch processing (faster)
python simple_inference.py --video videos/ --model model.ckpt --batch-size 4

# Different model
python simple_inference.py --video videos/ --model model.ckpt --model-type batfd

# Custom output location
python simple_inference.py --video videos/ --model model.ckpt --output-dir results/
```

## 📁 Example Structure

```
your-project/
├── videos/           # Your video files
│   ├── video1.mp4
│   ├── video2.mp4
│   └── video3.mp4
├── checkpoint.ckpt   # Your trained model
└── LAV-DF/          # This repository
```

## 💡 Tips

- **Supported formats**: MP4, AVI, MOV, MKV, FLV, WMV, WebM
- **GPU acceleration**: Add `--device cuda`
- **Faster processing**: Increase `--batch-size`
- **Out of memory?**: Decrease `--batch-size` to 1

## 🔍 Need Help?

- **Quick Reference**: `SIMPLE_PIPELINE_README.md`
- **Full Guide**: `SIMPLE_INFERENCE_GUIDE.md`
- **Code Examples**: `examples/simple_video_inference_example.py`
- **Notebook**: `examples/simple_video_inference.ipynb`
- **Test**: `python test_simple_pipeline.py`

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Import errors | `pip install -r requirements.txt` |
| Can't find videos | Use absolute paths |
| Out of memory | Use `--batch-size 1` |
| Slow processing | Use `--device cuda` and `--batch-size 4` |

## Next Steps

1. ✅ Install dependencies
2. ✅ Run: `python simple_inference.py --video YOUR_VIDEO.mp4 --model YOUR_MODEL.ckpt`
3. ✅ Check results in `output/simple_inference/`
4. ✅ Read `SIMPLE_PIPELINE_README.md` for more options

---

**That's it!** You're ready to process videos. 🎉
