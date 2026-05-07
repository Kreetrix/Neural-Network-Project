# Image Colorization — Model Comparison Report

**Project:** Neural Network Colorization  
**Date:** May 2026

---

## Overview

Over the course of this project we trained five different models for grayscale image colorization. All of them work in the LAB color space — the network takes the L (lightness) channel as input and tries to predict the A and B (color) channels. They range from a small U-Net trained on 4,000 images to a two-stage conditional GAN with perceptual loss trained on 8,000, all the way up to architectures trained on the full 128k COCO dataset.

The five models, roughly in order of complexity:

- **Baseline U-Net** — simple two-level encoder, L1 loss only, 4k images
- **Single-Stage cGAN** — U-Net generator + PatchGAN discriminator, 2k images
- **Two-Stage cGAN** — same as above but pretrained with L1 first, adds VGG perceptual loss, 8k images
- **Large U-Net** — three-level encoder, L1 loss, trained on the full 128k COCO dataset
- **Pix2Pix GAN** — U-Net + PatchGAN with TTUR, also on 128k images

---

## Architecture

Every model uses a U-Net as its backbone. The encoder halves the spatial resolution at each step while doubling the number of channels; the decoder mirrors that structure and brings in skip connections from the encoder to preserve spatial detail. The main difference between models is how deep that encoder goes and what loss they train against.

The Baseline U-Net goes two levels deep (64 → 128 channels, 256 bottleneck). Every other model goes three levels deep (64 → 128 → 256, 512 bottleneck), which roughly doubles the parameter count to around 7.7M.

The GAN models add a PatchGAN discriminator on top. Rather than judging the whole image real or fake, PatchGAN classifies overlapping 70×70 patches — this keeps the adversarial signal focused on local texture and color, which is exactly where colorization tends to go wrong.

The Two-Stage cGAN is the only model that uses VGG perceptual loss. A frozen VGG16 (up to `relu2_2`) extracts features from both the predicted image and the ground truth, and L1 distance in that feature space is added to the generator's loss. This pushes the model toward perceptually correct colors rather than just pixel-correct ones.

---

## Training

The models were trained with quite different setups, which makes direct comparison tricky.

The Baseline U-Net trained for 10 epochs with a 1e-3 learning rate, no augmentation, and no learning rate scheduling. It's the simplest setup in the project.

The Single-Stage cGAN used a lower learning rate (2e-4), horizontal flip augmentation, and early stopping with a patience of 8. It stopped at epoch 9 — the adversarial training never really stabilized.

The Two-Stage cGAN is the most carefully designed. The generator first trains on L1 loss alone for 20 epochs (learning rate 1e-4, StepLR halving every 10 epochs), giving it a solid starting point before the discriminator is introduced. Stage 2 then fine-tunes with the full adversarial + VGG objective. This two-stage approach is the single biggest difference between this model and the Single-Stage cGAN.

The Large U-Net and Pix2Pix GAN both trained on the full 128k COCO dataset with batch size 64. The Large U-Net used ReduceLROnPlateau (patience 5) and early stopping (patience 12), stopping at epoch 26. The Pix2Pix GAN ran for a fixed 50 epochs with TTUR — the discriminator learning rate was set 10× lower than the generator's, which is a well-known trick for keeping GAN training stable.

---

## Results

### Visual comparison

The grid below shows the same six test images run through each available model. Left column is the grayscale input, middle is the ground truth, and the remaining columns show model outputs with PSNR scores.

![Comparison Grid](report_images/comparison_grid.png)

---

### Best and worst colorizations — Large U-Net

The top row shows the three images where the Large U-Net performed best; the bottom row shows where it struggled most.

![Large U-Net Best and Worst](report_images/best_worst_Large_U-Net.png)

---

### PSNR across models

![PSNR Comparison](report_images/psnr_comparison.png)

The quantitative results, with the important caveat that these models trained on different datasets:

- **Baseline U-Net** — 23.42 dB PSNR, 0.9160 SSIM (10 epochs, 4k images)
- **Single-Stage cGAN** — 20.57 dB PSNR, 0.9281 SSIM (9 epochs, 2k images)
- **Two-Stage cGAN** — 27.44 dB PSNR, 0.8756 SSIM (20+20 epochs, 8k images)
- **Large U-Net** — best val L1 of 0.0727 (26 epochs, 128k images) — PSNR/SSIM not yet evaluated
- **Pix2Pix GAN** — best val L1 of 0.0875 at epoch 31 (50 epochs, 128k images) — PSNR/SSIM not yet evaluated

SSIM tends to be high across all models because the L channel (lightness and structure) is passed through unchanged — the shape is always right, only the color is predicted. The Two-Stage cGAN's lower SSIM despite higher PSNR makes sense: VGG perceptual loss pushes toward richer, more saturated colors that can differ structurally from the exact ground-truth hue, which is actually what you want for colorization.

---

## What we learned

**Two-stage training is by far the most impactful change.** Going from Single-Stage to Two-Stage cGAN jumps PSNR by 7 dB. The pre-training phase gives the generator a stable starting point so the adversarial training doesn't waste early epochs fighting random initialization.

**More data doesn't automatically beat better training.** The Pix2Pix GAN trains on 16× more images than the Two-Stage cGAN but gets a worse val L1 (0.0875 vs 0.0788). Without the VGG loss and the two-stage warmup, raw data volume doesn't close the gap.

**The Single-Stage cGAN is the weakest approach here.** Training a GAN from scratch on only 2,000 images with no warmup produces classic instability: the discriminator swings between overwhelming the generator (D loss → 0.35) and being fooled by it (D loss → 0.54), never settling near the ideal 0.5 equilibrium.

**The Large U-Net generalizes best among the pure U-Net models.** Its best val L1 of 0.0727 on 128k images is solid, but without adversarial training the colors tend to be muted and conservative — L1 alone pushes toward the mean.

---

## Verdict

The **Two-Stage cGAN** is the strongest model on measurable quality right now, with a test PSNR of 27.44 dB — the highest of any model we evaluated. Its combination of L1 pretraining, adversarial fine-tuning, and VGG perceptual loss produces more vivid and perceptually correct colors than anything else in the project.

The **Large U-Net** is the best bet for generalization — it's seen far more diverse content than any other model. Its weakness is color vibrancy; without adversarial training it plays it safe.

The ideal model doesn't exist yet: the Two-Stage cGAN architecture trained on the full 128k COCO dataset. That combination would be expected to clearly outperform everything here. The infrastructure is already in place — it's just a matter of pointing `colorization_final.ipynb` at the larger dataset and running it.

---
