# US-JEPA: Self-Supervised Representation Learning for Carotid Ultrasound Denoising

## Overview

This repository contains the work from a summer research internship focused on **self-supervised learning for medical ultrasound image analysis**. The project develops **US-JEPA**, a Vision Transformer encoder pretrained without labels on carotid ultrasound images (CUBS dataset), and evaluates its usefulness as a shared backbone for a downstream **image denoising** task.

Ultrasound images are inherently degraded by speckle noise, which harms both visual readability and downstream clinical analysis. Since paired clean/noisy ultrasound data does not exist, this project:

1. Pretrains a self-supervised encoder directly on real (unlabeled) ultrasound data, so it learns anatomically meaningful features without needing any annotations.
2. Rigorously verifies that the encoder actually learned useful representations (rather than a degenerate shortcut) before using it downstream.
3. Uses that shared encoder as a frozen/fine-tunable backbone for **four different decoder architectures**, trained on synthetically noised ultrasound images, and compares them on standard image-quality metrics.

## Pipeline

The project is organized as four sequential notebooks:

| Notebook | Stage | Description |
|---|---|---|
| `01_usjepa_cubs_pretrain_roi.ipynb` | Pretraining | Self-supervised JEPA-style pretraining of a ViT-S/16 encoder on the CUBS carotid ultrasound dataset. No annotations required. Uses ROI-aware block masking (morphological operations + convex hull) so masked patches stay within the ultrasound cone rather than the black background. |
| `02_usjepa_encoder_evaluation.ipynb` | Representation evaluation | Answers the key question: *did the encoder learn transferable, anatomically meaningful features, or did it find a low-loss shortcut?* Includes embedding visualization (t-SNE, PCA, UMAP), nearest-neighbour retrieval, embedding statistics (variance, cosine similarity, covariance, norms), and a comparison against a randomly-initialized encoder baseline. Also includes a ready-to-run linear probing scaffold for when annotations become available, plus an unsupervised proxy (k-means + silhouette score) usable without labels. |
| `03_usjepa_onevideo_evaluation.ipynb` | Qualitative check | Single-video evaluation of encoder behavior as a focused sanity check. |
| `04_denoising_decoders_comparison.ipynb` | Downstream task | Uses the shared US-JEPA encoder (frozen or fine-tuned) as backbone for four different decoder architectures, trained to denoise synthetically corrupted ultrasound images (speckle, gaussian, and mixed noise). Compares all four on PSNR, SSIM, and LPIPS. |

### Decoder architectures compared (notebook 04)

- **Model A: Visual Mamba**: selective-scan (state-space) based decoder
- **Model B: Physics-Informed**: speckle-gating mechanism with a physics-based loss term
- **Model C: U-Net-style**: residual convolutional decoder with progressive upsampling
- **Model D: Restormer-style**: transformer decoder using multi-head transposed attention (MDTA) and gated-Dconv feed-forward networks (GDFN)

## Dataset

- **CUBS** (Carotid Ultrasound Boundary Study): carotid ultrasound images used for both self-supervised pretraining and (synthetically noised) downstream denoising training.
- Dataset / checkpoints stored at: `(https://drive.google.com/drive/folders/1ta_YtvyfFwvZVhN9ZHeOFQSMSSHL6qlE?usp=drive_link)`

## Requirements

Core dependencies: `torch`, `torchvision`, `einops`, `scipy`, `scikit-image`, `matplotlib`, `tqdm`, `pillow`.

## How to run

Each notebook was developed and run on Google Colab / Kaggle and includes its own dependency installation cell. Open in Colab/Kaggle or run locally, in the numbered order above (01 → 02 → 03 → 04).

## Results

See `04_denoising_decoders_comparison.ipynb` for the full PSNR/SSIM/LPIPS comparison table and qualitative noisy/clean/denoised image grids across all four decoder architectures.


