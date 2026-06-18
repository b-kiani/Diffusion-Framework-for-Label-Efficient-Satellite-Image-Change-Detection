# ObjChange-MambaDiff

**Object-Aware Foundation–Mamba Diffusion Framework for Label-Efficient Satellite Image Change Detection**

ObjChange-MambaDiff is a research pipeline for multi-dataset satellite and aerial image change detection. It supports binary building change, general/seasonal change, semantic land-cover change, off-nadir/viewpoint robustness, and disaster-related change detection.

The project combines strong baseline change-detection models with an experimental object-aware architecture using foundation-style features, object-token reasoning, temporal/state-space modelling, pseudo-change learning, uncertainty estimation, and reproducible benchmarking.

---

## Research Goal

The aim of this project is to develop and evaluate a robust, label-efficient, and explainable AI framework for satellite image change detection across multiple public datasets.

The framework is designed to answer the following questions:

- Can object-aware features improve satellite change detection compared with pixel-only Siamese models?
- Can stronger pretrained encoders and object-level reasoning improve robustness to shadows, illumination, seasonal variation, misregistration, and off-nadir viewing?
- Can pseudo-change learning improve performance when only a small percentage of labelled data is available?
- How well do models generalise across different datasets, sensors, geographic regions, and disaster domains?

---

## Main Features

- Unified dataset audit and discovery under a single root folder
- Support for multiple public change-detection datasets
- Paired before/after image loading
- Binary mask generation and sanity checking
- xView2 JSON-to-mask conversion support
- SECOND semantic-mask verification utilities
- Siamese U-Net baseline
- Pretrained ResNet-style Siamese baseline
- ObjChange-MambaDiffLite / improved experimental architecture
- BCE + Dice + Focal loss
- Validation threshold tuning
- Label-efficiency experiments
- Cross-dataset generalisation testing
- Ablation experiments
- Qualitative prediction export
- Checkpoint resume support
- Consolidated CSV reports and plots

---

## Supported Datasets

| Dataset | Task Type | Role in This Project |
|---|---|---|
| LEVIR-CD | Binary building change | Main building change benchmark |
| WHU-CD | Binary building change | Additional building-domain validation |
| DSIFN-CD | Binary/general change | General remote-sensing change benchmark |
| CDD | General/seasonal change | Seasonal and illumination robustness |
| SECOND | Semantic change detection | Semantic “what changed” benchmark |
| HRSCD | Semantic change + land cover | High-resolution semantic validation |
| xBD / xView2 | Disaster damage change | Disaster and building-damage assessment |
| S2Looking | Off-nadir building change | Viewpoint and rural-scene robustness |

Datasets are **not redistributed** in this repository. Users must download each dataset from its official source and comply with the relevant licence or terms of use.

---

## Suggested Dataset Folder Structure

The current notebook expects the dataset root to be:

```text
D:\Datasets\Satillate
```

Recommended structure:

```text
D:\Datasets\Satillate
│
├── LEVIR CD
├── WHU-CD
├── DSIFN Train Test
├── CDD
├── SECOND
├── second_dataset
├── HRSCD
├── xView2
├── S2Looking
│
└── _objchange_mambadiff_full_experiment
```

The code includes folder-discovery logic, so folder names do not need to be perfectly identical, but standardising the folder names is recommended for reproducibility.

---

## Repository Structure

Suggested GitHub structure:

```text
ObjChange-MambaDiff/
│
├── notebooks/
│   ├── ObjChange_MambaDiff_Improved_All_In_One.ipynb
│   ├── ObjChange_MambaDiff_Full_Experiment_Run.ipynb
│   └── ObjChange_MambaDiff_Finalise_From_Restart.ipynb
│
├── src/
│   ├── datasets/
│   ├── models/
│   ├── losses/
│   ├── training/
│   ├── evaluation/
│   ├── preprocessing/
│   └── utils/
│
├── configs/
│   ├── levir_cd.yaml
│   ├── dsifn_cd.yaml
│   ├── s2looking.yaml
│   ├── second.yaml
│   └── xview2.yaml
│
├── outputs/
│   ├── checkpoints/
│   ├── reports/
│   ├── figures/
│   └── qualitative_predictions/
│
├── manuscripts/
│
├── requirements.txt
├── environment.yml
├── README.md
└── LICENSE
```

---

## Installation

### 1. Create a Python environment

Using conda:

```bash
conda create -n objchange python=3.10 -y
conda activate objchange
```

Or using venv:

```bash
python -m venv objchange_env
objchange_env\Scripts\activate
```

### 2. Install PyTorch

Install the correct PyTorch version for your CUDA setup from the official PyTorch website.

Example for CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install required packages

```bash
pip install numpy pandas matplotlib pillow opencv-python scikit-learn tqdm
pip install albumentations rasterio shapely geopandas
pip install timm segmentation-models-pytorch
pip install jupyter notebook ipykernel
```

Optional packages:

```bash
pip install wandb tensorboard einops
```

---

## Hardware Requirements

Initial experiments can run on a single consumer GPU.

Recommended minimum:

```text
GPU: NVIDIA RTX 4060 or better
VRAM: 8 GB minimum
RAM: 32 GB recommended
Storage: 200 GB+ depending on datasets
```

For larger foundation-model experiments, xView2, and full multi-dataset benchmarking:

```text
GPU: RTX 4090 / A5000 / A6000 / A100 recommended
VRAM: 24 GB+ recommended
```

The notebooks support mixed precision where possible.

---

## Running the Notebook

Open Jupyter:

```bash
jupyter notebook
```

Then run:

```text
notebooks/ObjChange_MambaDiff_Improved_All_In_One.ipynb
```

Set the dataset root:

```python
DATA_ROOT = r"D:\Datasets\Satillate"
```

For a quick test:

```python
QUICK_RUN = True
EPOCHS = 2
```

For a full experiment:

```python
QUICK_RUN = False
EPOCHS = 50
BATCH_SIZE = 4
IMAGE_SIZE = 256
```

Then run all cells from top to bottom.

---

## Resume After Restart

If the computer restarts or training is interrupted, use:

```text
notebooks/ObjChange_MambaDiff_Finalise_From_Restart.ipynb
```

This notebook checks saved CSV reports and checkpoints, skips completed rows, and resumes incomplete experiments where possible.

Expected output folder:

```text
D:\Datasets\Satillate\_objchange_mambadiff_full_experiment
```

---

## Main Experiment Types

### 1. Dataset Audit

The pipeline checks:

- Available datasets
- Top-level folders
- File counts
- Image/mask extensions
- Train/validation/test split detection
- Pairing between image time 1, image time 2, and mask
- Possible missing datasets
- Possible duplicate folders
- SECOND mask validity
- xView2 JSON annotation availability

### 2. Main Benchmark

The main benchmark compares:

- Siamese U-Net baseline
- Pretrained Siamese encoder baseline
- ObjChange-MambaDiffLite / improved model

Across available datasets such as:

```text
LEVIR-CD
DSIFN-CD
S2Looking
SECOND
xView2
```

### 3. Label-Efficiency Experiment

The pipeline trains with different labelled-data fractions:

```text
1%, 5%, 10%, 25%, 50%, 100%
```

This is used to test whether the proposed method is effective when annotated data are limited.

### 4. Cross-Dataset Generalisation

Example cross-dataset tests:

```text
Train on LEVIR-CD → Test on S2Looking
Train on S2Looking → Test on LEVIR-CD
Train on DSIFN-CD → Test on CDD
```

This measures domain shift and robustness.

### 5. Ablation Study

The ablation study evaluates the contribution of each model component:

| Ablation | Purpose |
|---|---|
| Baseline Siamese U-Net | Lower-bound comparison |
| + Pretrained encoder | Test foundation-style feature benefit |
| + Object branch | Test object-level reasoning |
| + Mamba/temporal block | Test temporal/state-space contribution |
| + Pseudo-change learning | Test low-label improvement |
| + Uncertainty head | Test reliability and calibration |

---

## Evaluation Metrics

The pipeline reports:

| Metric | Meaning |
|---|---|
| Precision | Fraction of predicted changes that are correct |
| Recall | Fraction of true changes detected |
| F1-score | Harmonic mean of precision and recall |
| IoU | Intersection-over-union for change masks |
| Accuracy | Overall pixel accuracy |
| Kappa | Agreement beyond chance |
| Loss | Validation or test loss |
| ECE | Expected calibration error |
| Threshold | Best tuned validation threshold |

For semantic experiments, future versions may also include:

```text
mIoU
per-class F1
transition-class accuracy
semantic confusion matrices
```

---

## Output Files

The full run saves outputs under:

```text
D:\Datasets\Satillate\_objchange_mambadiff_full_experiment
```

Typical output structure:

```text
_objchange_mambadiff_full_experiment/
│
├── checkpoints/
├── reports/
│   ├── main_benchmark_results.csv
│   ├── label_efficiency_results.csv
│   ├── cross_dataset_results.csv
│   ├── ablation_results.csv
│   └── ALL_RESULTS_CONSOLIDATED.csv
│
├── figures/
└── qualitative_predictions/
```

The most important final file is:

```text
reports/ALL_RESULTS_CONSOLIDATED.csv
```

---

## Current Experimental Notes

Early experiments showed that the pipeline runs successfully across multiple datasets. However, the improved model should be evaluated carefully against strong baselines before making publication claims.

Important checks before manuscript submission:

- Verify SECOND masks visually before using semantic results.
- Treat xView2 as a disaster-damage dataset, not only a simple binary change dataset.
- Use threshold tuning rather than a fixed 0.5 threshold.
- Report dataset licences and terms of use.
- Run at least three seeds for reliable confidence intervals.
- Include qualitative examples and failure cases.
- Avoid claiming state-of-the-art performance unless verified against official or reproduced baselines.

---

## Reproducibility Checklist

Before releasing results, record:

- Dataset source and version
- Dataset licence/terms
- Train/validation/test split files
- Random seed
- GPU model
- CUDA version
- PyTorch version
- Python version
- Batch size
- Image size
- Number of epochs
- Optimiser and learning rate
- Loss function
- Augmentations
- Threshold-selection method
- Checkpoint path
- Final metrics CSV

---

## Example Result Table Format

Use this format in the manuscript:

| Dataset | Model | Precision | Recall | F1 | IoU | Accuracy | Kappa | ECE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LEVIR-CD | Siamese U-Net | - | - | - | - | - | - | - |
| LEVIR-CD | ObjChange-MambaDiff | - | - | - | - | - | - | - |
| DSIFN-CD | Siamese U-Net | - | - | - | - | - | - | - |
| DSIFN-CD | ObjChange-MambaDiff | - | - | - | - | - | - | - |

Replace the placeholders with values from:

```text
reports/ALL_RESULTS_CONSOLIDATED.csv
```

---

## Troubleshooting

### CUDA out-of-memory

Reduce:

```python
BATCH_SIZE = 2
IMAGE_SIZE = 224
```

Enable mixed precision if not already enabled.

### Dataset not detected

Check:

- Folder name
- Image extensions
- Whether ZIP/RAR files were extracted
- Whether train/test folders are nested deeper than expected
- Whether masks are stored under unusual names

### SECOND gives perfect scores

This may indicate leakage, mask conversion error, or an invalid binary conversion. Run the SECOND mask sanity-check cells and visually inspect several samples before using the result.

### xView2 gives very low F1

xView2 uses JSON building/damage annotations. Make sure JSON files are correctly converted into segmentation masks. Also consider evaluating damage categories separately instead of using a simple binary mask.

### Training interrupted

Use the finalisation notebook:

```text
ObjChange_MambaDiff_Finalise_From_Restart.ipynb
```

It skips completed experiments and resumes from saved outputs where possible.

---

## Citation

If you use this repository, please cite:

```bibtex
@misc{objchange_mambadiff_2026,
  title  = {ObjChange-MambaDiff: An Object-Aware Foundation--Mamba Diffusion Framework for Label-Efficient Satellite Image Change Detection},
  author = {Aghaei, Pedram and Collaborators},
  year   = {2026},
  note   = {Research code and experimental pipeline}
}
```

Please also cite the original dataset papers and official dataset sources for LEVIR-CD, WHU-CD, DSIFN-CD, CDD, SECOND, HRSCD, xBD/xView2, and S2Looking.

---

## Licence

This repository is intended for academic and research use.

Recommended licence for code:

```text
MIT Licence
```

Dataset licences are separate and must be checked from the official dataset providers. This repository should not redistribute datasets unless the dataset licence explicitly allows redistribution.

---

## Disclaimer

This is a research pipeline for experimental satellite image change detection. It is not intended for operational disaster response, legal land-boundary decisions, insurance assessment, or safety-critical monitoring without independent validation and domain-expert review.
