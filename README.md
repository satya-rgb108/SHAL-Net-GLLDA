# SHAL-Net-GLLDA: GLL-DA for Low-Light Image Enhancement

## Overview

This repository contains my experimental modification of **SHAL-Net**, an existing low-light image enhancement method. I propose a module called **GLL-DA (Global–Local–Latent Degradation-Aware)** to investigate global-context modulation and local-feature refinement for low-light image enhancement.

This is an independent research modification, not the official SHAL-Net repository.

## Original Work and Acknowledgment

This project builds on the original research:

- **Paper:** *Degraded Structure and Hue Guided Auxiliary Learning for Low-Light Image Enhancement*
- **Original repository:** https://github.com/hmx-harry/SHAL-Net
- **Credit:** The original SHAL-Net authors retain credit for their original research, architecture, and implementation.

## My Contributions

My work focuses on modifying and experimenting with the SHAL-Net architecture:

- Proposing the GLL-DA module.
- Integrating global-context modulation and local-feature refinement into DecomNet.
- Modifying supporting code for training, enhancement, and evaluation where required.
- Evaluating the modified model against the original baseline.

**Research status:** This is an experimental implementation. Performance improvements have not been established until supported by reproducible experiments. The initial implementation covers global and local branches; planned latent-detail and degradation-aware components should not be considered complete until implemented and verified.

## Architecture

The initial GLL-DA design includes:

- **Global branch:** Uses pooled features to generate channel-wise modulation parameters, including FiLM-style modulation.
- **Local branch:** Uses convolutional features and spatial attention to refine spatial information.
- **Feature fusion:** Combines global and local features through a learnable fusion layer and residual connection.

The broader research direction includes latent-detail and degradation-aware processing.

## Repository Structure

- `model.py` — Model architecture and GLL-DA modifications.
- `train.py` — Training procedure.
- `dataloader.py` — Dataset loading and preprocessing.
- `loss.py` — Training loss functions.
- `detect.py` — Image enhancement.
- `eval_metrics.py` — Image quality evaluation.
- `util.py` — Utility functions.
- `requirements.txt` — Python dependencies.
- `imgs/` — Architecture and result illustrations.

## Setup

Install the dependencies:

```bash
pip install -r requirements.txt
```

A compatible PyTorch installation is also required. Check the installed PyTorch and CUDA versions before training on a GPU.

## Dataset

Prepare the dataset in the directory structure expected by the scripts. Dataset files and trained checkpoints are not included in this repository.

Useful resources:

- Original SHAL-Net repository: https://github.com/hmx-harry/SHAL-Net
- LOL dataset: https://daooshee.github.io/BMVC2018website/
- SICE dataset: https://github.com/csjcai/SICE

Check dataset licensing and usage conditions before using the data.

## Training

Configure the dataset paths and training parameters in `train.py`, then run:

```bash
python train.py
```

## Testing

Configure the test-image and output paths in `detect.py`, then run:

```bash
python detect.py
```

## Evaluation

Configure the enhanced-image and reference-image paths in `eval_metrics.py`, then run:

```bash
python eval_metrics.py
```

For a fair comparison, evaluate the baseline and modified model using the same test images and metrics.

## Research Disclaimer

GLL-DA is a proposed experimental modification. Its effectiveness must be established through controlled experiments and reproducible comparisons. This repository is intended for research and educational purposes.

## Citation

Please cite the original SHAL-Net publication when using its research or implementation:

```bibtex
@article{xu2024degraded,
  title={Degraded Structure and Hue Guided Auxiliary Learning for Low-Light Image Enhancement},
  author={Xu, Heming and Liu, Xintong and Zhang, Hanwen and Wu, Xiaohe and Zuo, Wangmeng},
  journal={Knowledge-Based Systems},
  pages={111779},
  year={2024},
  publisher={Elsevier}
}
```

Original research and implementation credit belongs to the SHAL-Net authors. Any future publication about GLL-DA should clearly distinguish the original contributions from the new modifications and experimental findings.
