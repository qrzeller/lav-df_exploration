# Visual Workflow Guide - Simple Video Inference Pipeline

## Complete Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        START HERE                                │
│                                                                  │
│  Do you have videos to analyze for deepfakes?                   │
│                          ↓ YES                                   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
        ┌───────────────────┴────────────────────┐
        │                                        │
        ▼                                        ▼
┌──────────────────┐                   ┌──────────────────┐
│  Command Line    │                   │  Python Script   │
│  (Easiest)       │                   │  (Flexible)      │
└────────┬─────────┘                   └────────┬─────────┘
         │                                      │
         │                                      │
         ▼                                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  python simple_inference.py \                                   │
│      --video YOUR_VIDEO.mp4 \                                   │
│      --model YOUR_MODEL.ckpt                                    │
│                                                                  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Pipeline Processing                           │
│                                                                  │
│  1. Load Video  ───────────────────────┐                        │
│     └─ Read with torchvision            │                        │
│                                         │                        │
│  2. Preprocess Video  ──────────────────┤                        │
│     ├─ Pad/truncate to 512 frames       │                        │
│     ├─ Resize to 96x96                  │                        │
│     └─ Normalize pixels                 │                        │
│                                         │                        │
│  3. Preprocess Audio  ──────────────────┤                        │
│     ├─ Pad to match video length        │                        │
│     ├─ Convert to mel spectrogram       │                        │
│     └─ Apply log scaling                │                        │
│                                         │                        │
│  4. Run Model (BATFD/BATFD+)  ──────────┤                        │
│     ├─ Process video stream             │                        │
│     ├─ Process audio stream             │                        │
│     └─ Fuse multimodal features         │                        │
│                                         │                        │
│  5. Generate Results  ───────────────────┘                        │
│     ├─ Fusion boundary map                                       │
│     ├─ Visual boundary map                                       │
│     └─ Audio boundary map                                        │
│                                                                  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Output Results                              │
│                                                                  │
│  📁 output/simple_inference/                                     │
│      ├─ video1.pt  ← Boundary maps                             │
│      ├─ video2.pt                                               │
│      └─ video3.pt                                               │
│                                                                  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Analyze Results                               │
│                                                                  │
│  Option A: Load in Python                                       │
│  ─────────────────────────                                       │
│  import torch                                                    │
│  result = torch.load("output/simple_inference/video1.pt")       │
│  max_score = result['fusion_boundary_map'].max()                │
│                                                                  │
│  Option B: Use Post-Processing                                  │
│  ──────────────────────────                                      │
│  Convert boundary maps to temporal proposals                     │
│  Apply non-maximum suppression                                   │
│  Generate detection report                                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Decision Tree: Which Method to Use?

```
                    ┌─────────────────────┐
                    │  What do you need?  │
                    └──────────┬──────────┘
                               │
           ┌───────────────────┼───────────────────┐
           │                   │                   │
           ▼                   ▼                   ▼
    ┌─────────────┐    ┌──────────────┐    ┌────────────┐
    │ Quick test  │    │ Integration  │    │ Learning/  │
    │ on videos   │    │ with code    │    │ Exploring  │
    └──────┬──────┘    └──────┬───────┘    └─────┬──────┘
           │                  │                   │
           ▼                  ▼                   ▼
    ┌─────────────┐    ┌──────────────┐    ┌────────────┐
    │ Use CLI:    │    │ Use Python   │    │ Use Jupyter│
    │ simple_     │    │ API with     │    │ Notebook   │
    │ inference   │    │ Simple       │    │ Tutorial   │
    │ .py         │    │ VideoDataset │    │            │
    └─────────────┘    └──────────────┘    └────────────┘
```

## Input Options Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                    Input Methods                              │
└───────────────────┬──────────────────────────────────────────┘
                    │
        ┌───────────┼───────────┐
        │           │           │
        ▼           ▼           ▼
┌──────────┐  ┌──────────┐  ┌──────────────┐
│  Single  │  │Directory │  │ List of      │
│  Video   │  │ of Videos│  │ Specific     │
│          │  │          │  │ Videos       │
└────┬─────┘  └────┬─────┘  └─────┬────────┘
     │             │              │
     │             │              │
     ▼             ▼              ▼
┌────────────────────────────────────────────────────────┐
│                                                        │
│  --video         --video        --videos               │
│  video.mp4       videos/        v1.mp4 v2.mp4 v3.mp4  │
│                                                        │
└───────────────────┬────────────────────────────────────┘
                    │
                    ▼
            ┌───────────────┐
            │ SimpleVideo   │
            │ Dataset       │
            │ Processes All │
            └───────────────┘
```

## Data Flow Visualization

```
┌─────────────┐
│  Video File │  (any format: MP4, AVI, MOV, etc.)
└──────┬──────┘
       │
       ▼
┌──────────────────────────────────────────┐
│  torchvision.io.read_video()             │
├──────────────┬───────────────────────────┤
│   Video      │      Audio                │
│   Stream     │      Stream               │
└──────┬───────┴───────┬───────────────────┘
       │               │
       ▼               ▼
┌─────────────┐  ┌─────────────────────┐
│ Pad to 512  │  │ Pad to 327680       │
│ frames      │  │ samples (16kHz)     │
└──────┬──────┘  └──────┬──────────────┘
       │                │
       ▼                ▼
┌─────────────┐  ┌─────────────────────┐
│ Resize to   │  │ Mel Spectrogram     │
│ 96x96       │  │ + Log Transform     │
└──────┬──────┘  └──────┬──────────────┘
       │                │
       │                │
       └────────┬───────┘
                ▼
        ┌───────────────┐
        │ (C,T,H,W)     │ Video: (3, 512, 96, 96)
        │ (F,T)         │ Audio: (64, 2048)
        └───────┬───────┘
                │
                ▼
        ┌───────────────────┐
        │  Model            │
        │  (BATFD/BATFD+)   │
        └────────┬──────────┘
                 │
                 ▼
        ┌────────────────────┐
        │  Boundary Maps     │
        │  • Fusion          │
        │  • Visual          │
        │  • Audio           │
        └────────┬───────────┘
                 │
                 ▼
        ┌────────────────────┐
        │  Save as .pt file  │
        └────────────────────┘
```

## Batch Processing Flow

```
┌──────────────────────────────────────────────────────┐
│         Directory with 100 videos                     │
│  ├─ video_001.mp4                                     │
│  ├─ video_002.mp4                                     │
│  ├─ ...                                               │
│  └─ video_100.mp4                                     │
└────────────────┬─────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────┐
│         SimpleVideoDataset                            │
│  • Discovers all 100 videos                          │
│  • Sorts them alphabetically                         │
│  • Prepares for loading                              │
└────────────────┬─────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────┐
│         DataLoader (batch_size=4)                     │
│                                                       │
│  Batch 1: [video_001, video_002, video_003, video_004]│
│  Batch 2: [video_005, video_006, video_007, video_008]│
│  ...                                                  │
│  Batch 25: [video_097, video_098, video_099, video_100]│
└────────────────┬─────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────┐
│         Model Inference (parallel)                    │
│                                                       │
│  GPU processes 4 videos simultaneously                │
│  ├─ Much faster than sequential                      │
│  └─ Efficient use of GPU resources                   │
└────────────────┬─────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────┐
│         Results Saved                                 │
│  ├─ video_001.pt                                      │
│  ├─ video_002.pt                                      │
│  ├─ ...                                               │
│  └─ video_100.pt                                      │
└──────────────────────────────────────────────────────┘
```

## Component Interaction

```
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│                    User Interface Layer                      │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ CLI Script   │  │ Python API   │  │ Jupyter      │      │
│  │ (simple_     │  │              │  │ Notebook     │      │
│  │ inference.py)│  │              │  │              │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                 │                 │               │
└─────────┼─────────────────┼─────────────────┼───────────────┘
          │                 │                 │
          └─────────────────┼─────────────────┘
                            │
┌───────────────────────────┼───────────────────────────────────┐
│                           ▼                                    │
│                 Dataset Layer                                  │
│                                                                │
│         ┌───────────────────────────┐                         │
│         │  SimpleVideoDataset       │                         │
│         │  • Video loading          │                         │
│         │  • Preprocessing          │                         │
│         │  • Batching               │                         │
│         └─────────────┬─────────────┘                         │
│                       │                                        │
└───────────────────────┼────────────────────────────────────────┘
                        │
┌───────────────────────┼────────────────────────────────────────┐
│                       ▼                                         │
│                 Utility Layer                                   │
│                                                                 │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│   │ read_video() │  │padding_video()│  │resize_video()│        │
│   │              │  │padding_audio()│  │              │        │
│   └──────────────┘  └──────────────┘  └──────────────┘        │
│                                                                 │
└───────────────────────┼────────────────────────────────────────┘
                        │
┌───────────────────────┼────────────────────────────────────────┐
│                       ▼                                         │
│                 Model Layer                                     │
│                                                                 │
│   ┌──────────────────┐        ┌──────────────────┐            │
│   │  BATFD Model     │        │  BATFD+ Model    │            │
│   │  • Video Encoder │        │  • Video Encoder │            │
│   │  • Audio Encoder │        │  • Audio Encoder │            │
│   │  • Fusion Module │        │  • Fusion Module │            │
│   │  • Boundary Map  │        │  • Boundary Map  │            │
│   │                  │        │  • Start/End Map │            │
│   └──────────────────┘        └──────────────────┘            │
│                                                                 │
└───────────────────────┼────────────────────────────────────────┘
                        │
                        ▼
                ┌───────────────┐
                │   Results     │
                │  • .pt files  │
                └───────────────┘
```

## Example Usage Paths

### Path 1: Quick Single Video Test
```
User → CLI → SimpleVideoDataset → Model → Result
  ↓
python simple_inference.py --video test.mp4 --model model.ckpt
  ↓
output/simple_inference/test.pt
```

### Path 2: Batch Processing
```
User → CLI (batch mode) → DataLoader → Model → Results
  ↓
python simple_inference.py --video videos/ --batch-size 4
  ↓
output/simple_inference/*.pt (multiple files)
```

### Path 3: Python Integration
```
User Script → SimpleVideoDataset → Custom Processing → Model → Custom Output
  ↓
Custom analysis, visualization, or integration with other tools
```

---

**This workflow guide provides visual representations of how the pipeline works.**
For detailed instructions, see `reports/simple_video_pipeline_report.md`
