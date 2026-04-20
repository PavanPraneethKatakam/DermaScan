---
title: DermaScan AI
emoji: 🔬
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: "1.44.1"
python_version: "3.11"
app_file: dermascan_app.py
pinned: false
---

# DermaScan AI — Clinical Skin Lesion Analysis

Upload a dermoscopy image for **full clinical ABCDE analysis**, risk scoring,
measurements, Grad-CAM explainability, and a downloadable report — all powered
by a trained **U-Net** (ISIC 2018, Dice 0.854).

## Features

| Feature | Description |
|---|---|
| 🎯 Segmentation | U-Net binary mask with green overlay |
| 🔬 ABCDE Analysis | Asymmetry, Border, Color, Diameter — all computed from the mask |
| 📊 Risk Score | Weighted 0–10 gauge with LOW / MEDIUM / HIGH level |
| 📐 Measurements | Area (mm²), Perimeter, Coverage, Bounding box |
| 🧠 Grad-CAM | Model explainability heatmap |
| 📅 Evolution | Upload a previous scan to track lesion growth |
| 📄 Report | Downloadable PDF + text clinical report |

## Model

- Architecture: **U-Net** with skip connections
- Dataset: **ISIC 2018 Task 1** (568 images, 70/15/15 split)
- Loss: **BCE + Dice** (50/50)
- Test Dice: **0.8543 ± 0.0821**
- Weights hosted at: `pavanpraneeth/isic-unet`

> ⚠️ **Disclaimer:** DermaScan AI is a research/screening tool only.
> It does NOT constitute a medical diagnosis. Always consult a qualified dermatologist.
