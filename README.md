---
title: ISIC Skin Lesion Segmentation
emoji: 🔬
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "6.9.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# ISIC 2018 Skin Lesion Segmentation — U-Net

Upload a dermoscopy image to get an instant binary segmentation mask from a trained U-Net.

## Results

| Metric | Test Set |
|--------|----------|
| Dice   | **0.9301 ± 0.0621** |
| IoU    | **0.8744 ± 0.0891** |

Trained on **ISIC 2018 Task 1** (568 images, 70/15/15 train/val/test split).  
Best checkpoint: epoch **45**, val Dice **0.9207**.

## Model Architecture

Classic **U-Net** with skip connections.  
Channel progression: 3 → 64 → 128 → 256 → 512 → 1024 → 512 → 256 → 128 → 64 → 1

## Usage

1. Upload any dermoscopy / skin lesion image
2. Click **Segment 🔍**
3. View the predicted binary mask and overlay
