"""
app.py
------
Gradio demo for ISIC 2018 Skin Lesion Segmentation using a trained U-Net.

Hosted on Hugging Face Spaces.
Model weights are downloaded from the HF Hub model repo on first run.
"""

import os
import numpy as np
import torch
import gradio as gr
from PIL import Image
from huggingface_hub import hf_hub_download

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_REPO   = "pavanpraneeth/isic-unet"
MODEL_FILE   = "best_model.pth"
IMAGE_SIZE   = 256
THRESHOLD    = 0.5
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)

# ---------------------------------------------------------------------------
# Load model (once at startup)
# ---------------------------------------------------------------------------
from model import UNet  # model.py is alongside app.py in the Space repo

def load_model() -> torch.nn.Module:
    ckpt_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    model = UNet(in_channels=3, out_channels=1)
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state["model_state_dict"])
    model.eval().to(DEVICE)
    print(f"[app] Model loaded from {MODEL_REPO} on {DEVICE}")
    return model

MODEL = load_model()

# ---------------------------------------------------------------------------
# Preprocessing / postprocessing helpers
# ---------------------------------------------------------------------------

def preprocess(img: np.ndarray) -> torch.Tensor:
    """Resize, normalise (ImageNet), convert to tensor."""
    pil = Image.fromarray(img).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(pil, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD          # (H, W, 3)
    tensor = torch.from_numpy(arr.transpose(2, 0, 1))   # (3, H, W)
    return tensor.unsqueeze(0).to(DEVICE)               # (1, 3, H, W)


def postprocess_mask(pred: torch.Tensor) -> np.ndarray:
    """Convert raw sigmoid output → uint8 mask image (0 or 255)."""
    mask = (pred.squeeze().cpu().numpy() > THRESHOLD).astype(np.uint8) * 255
    return mask


def make_overlay(original_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Overlay mask boundary on original image in red."""
    h, w = mask.shape
    orig_resized = np.array(
        Image.fromarray(original_rgb).resize((w, h))
    ).copy()

    # Draw red where mask == 255
    overlay = orig_resized.copy()
    overlay[mask > 0] = (
        overlay[mask > 0] * 0.4 + np.array([255, 0, 0]) * 0.6
    ).astype(np.uint8)
    return overlay


# ---------------------------------------------------------------------------
# Inference function (called by Gradio)
# ---------------------------------------------------------------------------

def segment(image: np.ndarray):
    """Run inference and return (mask_image, overlay_image)."""
    if image is None:
        return None, None

    tensor = preprocess(image)
    with torch.no_grad():
        pred = MODEL(tensor)         # (1, 1, 256, 256)

    mask   = postprocess_mask(pred)  # (256, 256) uint8
    overlay = make_overlay(image, mask)

    mask_rgb = np.stack([mask, mask, mask], axis=-1)  # grey → RGB for display
    return mask_rgb, overlay


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

DESCRIPTION = """
## 🔬 ISIC 2018 Skin Lesion Segmentation

Upload a dermoscopy image to get an instant binary segmentation mask from a trained **U-Net**.

| Metric | Test Set Score |
|--------|---------------|
| Dice   | **0.9301 ± 0.0621** |
| IoU    | **0.8744 ± 0.0891** |

*Trained on ISIC 2018 Task 1 (568 images, 70/15/15 split).*
"""

with gr.Blocks(theme=gr.themes.Soft(), title="ISIC Skin Lesion Segmentation") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column():
            inp = gr.Image(label="Input Image", type="numpy")
            btn = gr.Button("Segment 🔍", variant="primary")
        with gr.Column():
            out_mask    = gr.Image(label="Predicted Mask")
            out_overlay = gr.Image(label="Overlay on Original")

    btn.click(fn=segment, inputs=inp, outputs=[out_mask, out_overlay])

    gr.Examples(
        examples=[],
        inputs=inp,
    )

if __name__ == "__main__":
    demo.launch()
