# DermaScan - Clinical Skin Lesion Analysis

DermaScan is an automated skin lesion analysis tool designed for clinical screening. It provides an instant binary segmentation mask, ABCDE analysis, risk scoring, measurements, and explainability heatmaps, all powered by a trained U-Net model.

## Features

- **Segmentation**: U-Net binary mask with green overlay to isolate the lesion.
- **ABCDE Analysis**: Computes Asymmetry, Border irregularity, Color variegation, and Diameter directly from the mask.
- **Risk Score**: A weighted 0-10 gauge indicating LOW, MEDIUM, or HIGH risk levels based on clinical metrics and optional patient demographics.
- **Measurements**: Calculates physical properties including Area (mm²), Perimeter, Coverage percentage, and Bounding Box.
- **Explainability**: Generates Grad-CAM heatmaps showing the model's focus regions during prediction.
- **Evolution Tracking**: Upload a previous scan to compare and track lesion growth over time.
- **Clinical Report**: Exports a downloadable PDF + text report summarizing the findings.

## Model Details

- **Architecture**: U-Net with skip connections for precise feature extraction and localization.
- **Dataset**: ISIC 2018 Task 1 (568 dermoscopy images, 70/15/15 split for train/val/test).
- **Loss Function**: Binary Cross-Entropy (BCE) + Dice Loss (50/50 weighting).
- **Test Performance**: Dice Score: 0.8543 ± 0.0821.
- **Weights Location**: The model automatically downloads its weights from `pavanpraneeth/isic-unet` upon the first run.

## Setup & Installation

Follow these instructions to run the application locally.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/PavanPraneethKatakam/DermaScan.git
   cd DermaScan
   ```

2. **Install dependencies:**
   It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application:**
   This project includes two web interfaces. The primary, feature-rich clinical dashboard is built with Streamlit.

   - **To run the Streamlit App (Recommended):**
     ```bash
     streamlit run dermascan_app.py
     ```

   - **To run the simple Gradio Interface:**
     ```bash
     python app.py
     ```

## Project Structure

- `dermascan_app.py`: The main Streamlit application, containing the advanced user interface, tabs, and interactive visualizations.
- `app.py`: A simpler Gradio-based interface for fast segmentation checks.
- `app_analysis.py`: Contains the core logic for running the ABCDE analysis, risk scoring algorithms, Grad-CAM generation, and image processing.
- `model.py`: Defines the PyTorch U-Net architecture.
- `requirements.txt`: Python package dependencies.

> **Disclaimer:** DermaScan is a research and screening tool only. It does NOT constitute a medical diagnosis. Always consult a qualified dermatologist for clinical evaluation.
