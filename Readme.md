# Quad-ViM: Efficient Vision Mamba for High-Resolution Satellite Image Modeling

Quad-ViM is a deep learning research project focused on **memory-efficient image modeling** for high-resolution remote sensing data (e.g., DOTA/EuroSAT).  
The project compares three model families:

- **ViT (Vision Transformer)**: strong baseline, but expensive due to quadratic attention.
- **Vim (Vision Mamba)**: replaces attention with linear-complexity state-space scanning.
- **Quad-ViM (proposed variant)**: extends Vim with **quad-directional scanning** to better capture 2D spatial context.

---

## 1) Motivation

Satellite images are often large (up to 1024–2048 px), making standard self-attention prohibitively expensive.

- ViT self-attention scales as:
	$$
	\mathcal{O}(N^2)
	$$
- Mamba-style sequence modeling scales as:
	$$
	\mathcal{O}(N)
	$$

Where $N$ is the number of patches/tokens.  
Quad-ViM targets this bottleneck by preserving long-range modeling with significantly better memory behavior at high resolution.

---

## 2) Key Features

- **Linear complexity backbone** (Mamba-based) for scalable high-res inputs.
- **Supports large image resolutions** (tested up to 2048 in benchmarking scripts).
- **Lower VRAM pressure** than ViT in stress scenarios.
- **Quad-directional context modeling** via multi-directional scan blocks.
- **Research-friendly codebase** with separate scripts for training, inference, visualization, and benchmarking.

---

## 3) Architecture Overview

### ViT vs Vim vs Quad-ViM

| Model | Core Operator | Complexity | Spatial Context Strategy |
|---|---|---:|---|
| ViT | Self-Attention | $\mathcal{O}(N^2)$ | Global pairwise token interactions |
| Vim | Mamba SSM | $\mathcal{O}(N)$ | Bidirectional sequence scanning |
| Quad-ViM | Mamba SSM + Quad Scan | $\mathcal{O}(N)$ | 4-way directional scanning over patch grid |

### Quad-ViM Intuition

Quad-ViM processes patch tokens in multiple directional orders to preserve richer 2D spatial dependencies without quadratic attention cost.

```text
Image -> Patch Embed -> Token Grid -> Quad Scan Block(s) -> Pool -> Classifier
																↘  top-left -> bottom-right
																 ↘ top-right -> bottom-left
																 ↘ bottom-left -> top-right
																 ↘ bottom-right -> top-left
```

### High-level Comparison Diagram

```text
ViT:
Tokens -> [Self-Attention (N x N)] -> MLP -> ...

Vim:
Tokens -> [Mamba Forward] + [Mamba Backward] -> ...

Quad-ViM:
Tokens(Grid) -> [Mamba Scan 1]
						 -> [Mamba Scan 2]
						 -> [Mamba Scan 3]
						 -> [Mamba Scan 4]
						 -> Merge -> Head
```

---

## 4) Installation

### Requirements

- Python 
- CUDA-enabled PyTorch environment for GPU training/benchmarking

### Core Libraries

- `torch`, `torchvision`
- `timm`
- `mamba-ssm`
- `einops`
- `matplotlib`
- `opencv-python`

### Setup

```bash
git clone https://github.com/Krish2005tech/DL_PROJECT_PHASE2.git
cd DL_PROJECT_PHASE2

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install torch torchvision timm mamba-ssm einops matplotlib opencv-python
```

---

## 5) Dataset Setup

This project uses two dataset paths/pipelines in scripts:

- **EuroSAT** helper in `download_dataset.py` / `dataset.py`
- **DOTA-style** training paths in training scripts

### A) Download EuroSAT (quick start)

```bash
python download_dataset.py
```

Expected location:
```text
datasets/
└── EuroSAT/
		└── 27500/
				├── AnnualCrop/
				├── Forest/
				└── ...
```

### B) DOTA Folder Layout (for DOTA scripts)

```text
datasets/
└── DOTAv1/
		└── DOTAv1/
				├── images/
				│   ├── train/
				│   └── test/
				└── labels/
						└── train/
```

> Ensure paths in scripts match your local filesystem before training.

---

## 6) Training

Current training scripts are configuration-driven (hyperparameters are mostly defined **inside** each file).

### Train Vim

```bash
python train_vim.py
```

### Train Quad-ViM (research baseline)

```bash
python train_quad_vim.py
```

### Train ViT baseline

```bash
python train_vit.py
```

### Train DOTA pipeline (QuadVisionMamba variant)

```bash
python train_dota.py
```

### Key Hyperparameters (from scripts)

- **Batch size**: `16`
- **Learning rate**: `1e-4`
- **Epochs**: `5`
- **Resize**:
	- Vim/ViT high-res scripts: `1024 x 1024`
	- Quad-ViM script variant: `224 x 224`
- **Patch size**: typically `16` (model-dependent)

Outputs are saved to `results/*.json`, and trained weights can be saved as `model.pth`.

---

### Expected Output Format

```text
Res        | ViT (GB)    | Vim (GB)    | QuadViM (GB)  | Saving %
--------------------------------------------------------------------
224        | ...         | ...         | ...           | ...
512        | ...         | ...         | ...           | ...
...
```

---

## 9) Results

### Sample Logged Results (`results/*.json`)

| Experiment | Model  | Time (s) | Notes |
|---|---|---:|---:|---|
| `vim.json` | Vim  | 941.20 | Baseline run |
| `quad_vim.json` | Quad-ViM | 955.88 | Quad scan variant |
| `vit.json` | ViT  | 952.80 | ViT baseline |
| `vim_highres.json` | Vim | 1153.42 | High-res run |
| `vit_highres_log.json` | ViT | 2.04 | `OOM_CRASHED` |

### Interpretation

- ViT remains a strong reference model but struggles with high-res memory scaling.
- Vim/Quad-ViM are designed for **better scaling efficiency** under large token counts.
- Quad-ViM introduces directional inductive bias for 2D context while keeping linear complexity.

---

## 10) Project Structure

```text
PHASE 2/
├── train_vim.py
├── train_quad_vim.py
├── train_vit.py
├── train_dota.py
├── train_vim_er.py
├── train_vim_rotaion.py
├── train_quad_vim_er.py
├── train_quad_vim_rotaion.py
├── dataset.py
├── download_dataset.py
├── inference.py
├── visual_inference.py
├── stress_test.py
├── stress2.py
├── plot_results.py
├── utils.py
└── datasets/
├── models/
│   ├── vim_model.py
│   ├── quad_vim_model.py
│   ├── quad_vim_block.py
│   └── vim.py
└── results/
		├── vim.json
		├── quad_vim.json
		├── vit.json
		├── vim_highres.json
		└── vit_highres_log.json
```

---

## 11) Known Issues / Limitations

- Some scripts assume different dataset classes/paths (`DotaDataset` vs `DOTADataset`); path harmonization may be required.
- Several hyperparameters are hardcoded per script (no unified CLI yet).
- Metric logs are from short research runs and may vary across hardware/seeds.
- `mamba-ssm` installation can be environment-sensitive.

---

---

## 13) Credits / References

- **Mamba (State Space Models)**  
	Gu, A., & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*.  
	https://arxiv.org/abs/2312.00752

- **Vision Transformer (ViT)**  
	Dosovitskiy, A., et al. (2020). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*.  
	https://arxiv.org/abs/2010.11929

- **DOTA Dataset**  
	Xia, G.-S., et al. (2018). *DOTA: A Large-Scale Dataset for Object Detection in Aerial Images*.  
	https://captain-whu.github.io/DOTA/

