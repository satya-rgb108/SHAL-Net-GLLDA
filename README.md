# SHAL-Net-GLLDA: GLL-DA for Low-Light Image Enhancement

## Overview

This repository contains my experimental modification of **SHAL-Net**, an existing low-light image enhancement method, with a proposed module called **GLL-DA (Global–Local–Latent Degradation-Aware)**.

The project investigates whether adding global-context modulation and local spatial-feature refinement to the original SHAL-Net architecture can improve low-light image enhancement.

**This is an independent research modification of the original implementation, not the official SHAL-Net repository.**

## Original Work and Acknowledgment

This project is based on the original SHAL-Net research:

- **Paper:** *Degraded Structure and Hue Guided Auxiliary Learning for Low-Light Image Enhancement*
- **Original repository:** https://github.com/hmx-harry/SHAL-Net
- **Original implementation and research:** Credit to the original SHAL-Net authors.

The original architecture, methodology, and associated research belong to their respective authors. This repository builds on their publicly available implementation and acknowledges their contribution.

## My Contributions

My work focuses on modifying and experimenting with the SHAL-Net architecture.

- Proposing the GLL-DA module for global-context modulation and local-feature refinement.
- Integrating the module into the original DecomNet architecture.
- Modifying supporting code where required for model execution, image enhancement, and evaluation.
- Investigating the effect of these changes through controlled experiments.

**Research status:** This is an experimental implementation. Performance improvements must be established through reproducible experiments and comparison with the original SHAL-Net baseline. The initial implementation does not yet necessarily include every planned GLL-DA component.

## Architecture

The modified architecture builds on SHAL-Net and introduces a GLL-DA refinement module.

The initial design includes:
- **Global branch:** Uses pooled feature information to generate channel-wise modulation parameters, including FiLM-style modulation.
- **Local branch:** Uses convolutional features and spatial attention to refine spatially important information.
- **Feature fusion:** Combines global and local features using a learnable fusion layer and residual connection.

The latent-detail and degradation-aware components remain part of the broader research direction and should only be described as implemented once they have been added and verified.

## Repository Structure

- `model.py` — SHAL-Net architecture and GLL-DA modifications.
- `train.py` — Model training.
- `dataloader.py` — Dataset loading and preprocessing.
- `loss.py` — Training loss functions.
- `detect.py` — Low-light image enhancement.
- `eval_metrics.py` — Image quality evaluation.
- `util.py` — Utility functions.
- `requirements.txt` — Python dependencies.
- `imgs/` — Architecture and result illustrations.

## Setup

Install the dependencies:

```bash
pip install -r requirements.txt
```

A compatible PyTorch installation and GPU are recommended for training. The exact environment should be checked against the dependencies and available CUDA runtime.

## Dataset

The model requires low-light images for training and appropriate test images for evaluation.

The original SHAL-Net project provides information about its datasets:

- Original repository: https://github.com/hmx-harry/SHAL-Net
- LOL dataset: https://daooshee.github.io/BMVC2018website/
- SICE dataset: https://github.com/csjcai/SICE

Dataset availability, directory structure, and usage conditions should be checked before use. Dataset files and trained weights are not included in this repository.

## Training

Check the dataset paths and training configuration in `train.py` before running:

```bash
python train.py
```

Training settings, checkpoints, and results should be documented for reproducibility.

## Testing

Configure the test-image paths and output directory in `detect.py`, then run:

```bash
python detect.py
```

## Evaluation

Configure the enhanced-image and reference-image paths in `eval_metrics.py`, then run:

```bash
python eval_metrics.py
```

Evaluation should use the same test images and metrics for the original baseline and the modified model.

## Research Disclaimer

GLL-DA is a proposed modification developed for experimentation. Its effectiveness is not established until supported by controlled experiments. This repository is intended for research and educational purposes.

## Citation

If you use the original SHAL-Net method, please acknowledge and cite its original publication:

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

Please also cite the original repository when appropriate. Any future publication describing GLL-DA should clearly distinguish the original SHAL-Net contributions from the new modifications and experimental findings.
