# Localized Audio Visual DeepFake Dataset (LAV-DF)
Link for the personal repo : [link](https://github.com/qrzeller/lav-df_exploration)

<div align="center">
    <img src="assets/overview.svg">
    <p></p>
</div>

---

## 🚀 Quick Start: FakeAVCeleb Cross-Dataset Evaluation

### Complete Pipeline for Cross-Dataset Testing

This guide shows how to run the full evaluation pipeline on FakeAVCeleb dataset with temporal feature analysis.

#### Prerequisites

1. **Download FakeAVCeleb v1.2 dataset** and place it in:
   ```
   dataset/multimodal-time-localisation/FakeAVCeleb_v1.2/FakeAVCeleb_v1.2/
   ```

2. **Setup environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   pip install matplotlib seaborn  # For plotting
   ```

3. **Download pre-trained model** (see Models section below)

#### Step 1: Run Evaluation (Inference + Post-Processing)

```bash
# Run LAV-DF model evaluation on FakeAVCeleb videos
# This performs inference AND post-processing in one step
python evaluate.py --config config/batfd_fakeavceleb.toml \
    --data_root dataset/multimodal-time-localisation/FakeAVCeleb_v1.2/FakeAVCeleb_v1.2 \
    --checkpoint output/ckpt/batfd_default.ckpt
```

**What this does**:
- Runs inference on all FakeAVCeleb videos
- Applies soft-NMS post-processing
- Generates boundary predictions

**Output**: 
- CSV files with boundary scores: `output/results/batfd_default_fakeavceleb/`
- JSON with temporal proposals: `output/results/batfd_default_fakeavceleb.json`

⚠️ **Known Issue**: CSV filename collisions cause only ~158/500 real videos to be processed (non-unique filenames like `00028.mp4`). Results remain valid but based on a subset.

#### Step 2: Extract Temporal Features & Train Classifier

```bash
# Extract 17 temporal features and train Random Forest
python temporal_feature_extraction.py --config config/batfd_fakeavceleb.toml
```

**Output**: 
- Trained model: `output/temporal_classifier.pkl`
- Test set results: Baseline AUC=0.7732 → Temporal AUC=0.8265 (+5.3% improvement)
- Cross-validation: 5-fold AUC=0.8154 ± 0.1131
- Demographic fairness analysis (race/gender)

**Key Features Extracted**: std_score, kurtosis, entropy, score_range, max_score, temporal_diff, etc.

#### Step 3: Generate Evaluation Plots

```bash
# Create comprehensive visualizations
python generate_evaluation_plots.py
```

**Output**: High-resolution plots in `output/plots/`:
1. `roc_curves_comparison.png/pdf` - ROC curves (baseline vs temporal classifier)
2. `method_comparison.png/pdf` - Bar chart comparing methods
3. `demographic_fairness.png/pdf` - Performance by race and gender
4. `feature_importance.png/pdf` - Top 10 most important features
5. `threshold_analysis.png/pdf` - Metrics vs classification threshold
6. `score_distributions.png/pdf` - Real vs fake score distributions
7. `summary_dashboard.png/pdf` - Comprehensive single-page overview

#### Step 4: Run Inference on New Videos (Optional)

```bash
# Predict on single video
python temporal_classifier_inference.py \
    --csv output/results/batfd_default_fakeavceleb/00001.csv

# Batch prediction on directory
python temporal_classifier_inference.py \
    --csv-dir output/results/batfd_default_fakeavceleb/ \
    --output predictions.csv
```

### Performance Summary

**Test Set (20% held-out, 3,679 videos)**:
- Baseline (max boundary score): **AUC = 0.7732**
- Temporal Classifier (17 features): **AUC = 0.8265**
- Absolute improvement: **+0.0533** (+6.9% relative)
- Cross-validation: **0.8154 ± 0.1131**

**Demographic Fairness** (no bias detected):
- Race groups: AUC 0.75-0.85 (variance ±0.03)
- Gender: Men 0.81, Women 0.79 (variance ±0.01)

### Documentation

- **Temporal Feature Analysis**: `Documentation/temporal_distribution_analysis.md`
- **Training Results**: `Documentation/temporal_classifier_training_results.md`
- **Usage Guide**: `Documentation/temporal_feature_scripts_guide.md`
- **Binary Classification Report**: `Documentation/fakeavceleb_evaluation_results.md`

---

<div align="center">
    <a href="https://github.com/ControlNet/LAV-DF/issues">
        <img src="https://img.shields.io/github/issues/ControlNet/LAV-DF?style=flat-square">
    </a>
    <a href="https://github.com/ControlNet/LAV-DF/network/members">
        <img src="https://img.shields.io/github/forks/ControlNet/LAV-DF?style=flat-square">
    </a>
    <a href="https://github.com/ControlNet/LAV-DF/stargazers">
        <img src="https://img.shields.io/github/stars/ControlNet/LAV-DF?style=flat-square">
    </a>
    <a href="https://github.com/ControlNet/LAV-DF/blob/master/LICENSE">
        <img src="https://img.shields.io/badge/license-CC%20BY--NC%204.0-97ca00?style=flat-square">
    </a>
    <a href="https://arxiv.org/abs/2204.06228">
        <img src="https://img.shields.io/badge/arXiv-2204.06228-b31b1b.svg?style=flat-square">
    </a>
    <a href="https://arxiv.org/abs/2305.01979">
        <img src="https://img.shields.io/badge/arXiv-2305.01979-b31b1b.svg?style=flat-square">
    </a>
    <a href="https://huggingface.co/datasets/ControlNet/LAV-DF">
        <img src="https://img.shields.io/badge/huggingface-dataset-FFD21E?style=flat-square&logo=huggingface">
    </a>
    <a href="https://huggingface.co/ControlNet/LAV-DF">
        <img src="https://img.shields.io/badge/huggingface-model-FFD21E?style=flat-square&logo=huggingface">
    </a>
    <a href="https://paperswithcode.com/sota/temporal-forgery-localization-on-lav-df?p=glitch-in-the-matrix-a-large-scale-benchmark">
        <img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/glitch-in-the-matrix-a-large-scale-benchmark/temporal-forgery-localization-on-lav-df&style=flat-square">
    </a>
</div>

This repo is the official PyTorch implementation for the DICTA paper [Do You Really Mean That? Content Driven Audio-Visual 
Deepfake Dataset and Multimodal Method for Temporal Forgery Localization](https://ieeexplore.ieee.org/document/10034605)
(Best Award), and the journal paper [_Glitch in the Matrix_: A Large Scale Benchmark for Content Driven Audio-Visual 
Forgery Detection and Localization](https://www.sciencedirect.com/science/article/pii/S1077314223001984) accepted by CVIU.

## LAV-DF Dataset

### Download

To use this LAV-DF dataset, you should agree the [terms and conditions](https://github.com/ControlNet/LAV-DF/blob/master/TERMS_AND_CONDITIONS.md).

Download link: [OneDrive](https://monashuni-my.sharepoint.com/:f:/g/personal/zhixi_cai_monash_edu/EklD-8lD_GRNl0yyJJ-cF3kBWEiHRmH4U5Dtg7eJjAOUlg?e=wowDpd), [Google Drive](https://drive.google.com/drive/folders/1U8asIMb0bpH6-zMR_5FaJmPnC53lomq7?usp=sharing), [HuggingFace](https://huggingface.co/datasets/ControlNet/LAV-DF).

### Baseline Benchmark

| Method  | AP@0.5 | AP@0.75 | AP@0.95 | AR@100 | AR@50 | AR@20 | AR@10 |
|---------|--------|---------|---------|--------|-------|-------|-------|
| BA-TFD  | 79.15  | 38.57   | 00.24   | 67.03  | 64.18 | 60.89 | 58.51 |
| BA-TFD+ | 96.30  | 84.96   | 04.44   | 81.62  | 80.48 | 79.40 | 78.75 |

Please note this result of BA-TFD is slightly better than the one reported in the paper. 
This is because we have used the better hyperparameters in this repository.

## Baseline Models

### Requirements

The main versions are,
- Python >= 3.7, < 3.11
- PyTorch >= 1.13
- torchvision >= 0.14
- lightning >= 2.0.0

Run the following command to install the required packages.

```bash
pip install -r requirements.txt
```

### Training BA-TFD

Train the BA-TFD introduced in paper [Do You Really Mean That? Content Driven Audio-Visual 
Deepfake Dataset and Multimodal Method for Temporal Forgery Localization](https://ieeexplore.ieee.org/document/10034605) with default hyperparameter on LAV-DF dataset.

```bash
python train.py \
  --config ./config/batfd_default.toml \
  --data_root <DATASET_PATH> \
  --batch_size 4 --num_workers 8 --gpus 1 --precision 16
```

The checkpoint will be saved in `ckpt` directory, and the tensorboard log will be saved in `lighntning_logs` directory. If you meet the NaN issue when training BA-TFD+, that might be caused by the bug in PyTorch self attention ops, upgrading or changing the PyTorch version can solve it.

### Training BA-TFD+

Train the BA-TFD+ introduced in paper [_Glitch in the Matrix_: A Large Scale Benchmark for Content Driven Audio-Visual Forgery Detection and Localization](https://www.sciencedirect.com/science/article/pii/S1077314223001984) with default hyperparameter on LAV-DF dataset.

```bash
python train.py \
  --config ./config/batfd_plus_default.toml \
  --data_root <DATASET_PATH> \
  --batch_size 4 --num_workers 8 --gpus 2 --precision 32
```

Please use `FP32` for training BA-TFD+ as `FP16` will cause inf and nan.

The checkpoint will be saved in `ckpt` directory, and the tensorboard log will be saved in `lighntning_logs` directory.


### Evaluation

Please run the following command to evaluate the model with the checkpoint saved in `ckpt` directory.

Besides, you can also download the [BA-TFD](https://github.com/ControlNet/LAV-DF/releases/download/pretrained_model/batfd_default.ckpt) and [BA-TFD+](https://github.com/ControlNet/LAV-DF/releases/download/pretrained_model_v2/batfd_plus_default.ckpt) pretrained models.

```bash
python evaluate.py \
  --config <CONFIG_PATH> \
  --data_root <DATASET_PATH> \
  --checkpoint <CHECKPOINT_PATH> \
  --batch_size 1 --num_workers 4
```

In the script, there will be a temporal inference results generated in `output` directory, and the AP and AR scores will
be printed in the console.

Note please make sure only one GPU is visible to the evaluation script.

## License

This project is under the CC BY-NC 4.0 license. See [LICENSE](LICENSE) for details.

## References

If you find this work useful in your research, please cite them.

The conference paper,
```bibtex
@inproceedings{cai2022you,
  title = {Do You Really Mean That? Content Driven Audio-Visual Deepfake Dataset and Multimodal Method for Temporal Forgery Localization},
  author = {Cai, Zhixi and Stefanov, Kalin and Dhall, Abhinav and Hayat, Munawar},
  booktitle = {2022 International Conference on Digital Image Computing: Techniques and Applications (DICTA)},
  year = {2022},
  doi = {10.1109/DICTA56598.2022.10034605},
  pages = {1--10},
  address = {Sydney, Australia},
}
```

The extended journal version is accepted by CVIU,
```bibtex
@article{cai2023glitch,
  title = {Glitch in the Matrix: A Large Scale Benchmark for Content Driven Audio-Visual Forgery Detection and Localization},
  author = {Cai, Zhixi and Ghosh, Shreya and Dhall, Abhinav and Gedeon, Tom and Stefanov, Kalin and Hayat, Munawar},
  journal = {Computer Vision and Image Understanding},
  year = {2023},
  volume = {236},
  pages = {103818},
  issn = {1077-3142},
  doi = {10.1016/j.cviu.2023.103818},
}
```

## Acknowledgements

Some code related to boundary matching mechanism is borrowed from 
[JJBOY/BMN-Boundary-Matching-Network](https://github.com/JJBOY/BMN-Boundary-Matching-Network) and 
[xxcheng0708/BSNPlusPlus-boundary-sensitive-network](https://github.com/xxcheng0708/BSNPlusPlus-boundary-sensitive-network).
