"""
app_analysis.py
---------------
Clinical analysis functions for DermaScan AI.

All functions operate on numpy arrays.  No Streamlit imports here —
this module is pure logic, importable and testable independently.
"""

import io
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ── Constants ──────────────────────────────────────────────────────────────────
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_SIZE    = 256
THRESHOLD     = 0.5
PIXELS_PER_MM = 25.0          # typical dermoscope calibration

# ── Preprocessing ──────────────────────────────────────────────────────────────

def preprocess(image_rgb: np.ndarray) -> torch.Tensor:
    """H×W×3 uint8 numpy array → 1×3×256×256 float32 tensor (ImageNet-normalised)."""
    img = cv2.resize(image_rgb, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor


# ── Segmentation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def run_segmentation(model, device, image_rgb: np.ndarray):
    """
    Run U-Net inference on a single RGB image.

    Returns
    -------
    mask_orig : np.ndarray  bool H×W mask in original image resolution
    prob_256  : np.ndarray  float32 probability map at 256×256
    """
    model.eval()
    tensor = preprocess(image_rgb).to(device)
    prob   = model(tensor).squeeze().cpu().numpy()        # (256, 256) float32
    mask_256 = (prob > THRESHOLD).astype(np.uint8)

    h, w = image_rgb.shape[:2]
    mask_orig = cv2.resize(mask_256, (w, h),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
    return mask_orig, prob


# ── Image Quality Check ────────────────────────────────────────────────────────

def check_image_quality(image_rgb: np.ndarray) -> dict:
    """
    Assess uploaded image quality before running the model.

    Returns a dict with 'ok' bool, numeric scores, and a list of issues.
    """
    gray       = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blur       = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast   = float(gray.std())

    issues = []
    if blur < 80:
        issues.append("Image may be blurry — results may be unreliable.")
    if brightness < 40:
        issues.append("Image is too dark.")
    if brightness > 230:
        issues.append("Image is overexposed.")
    if contrast < 15:
        issues.append("Very low contrast — lesion boundaries may be invisible.")

    quality_score = min(100.0,
        (min(blur, 500) / 500) * 40 +
        (contrast / 80)        * 40 +
        (1 - abs(brightness - 128) / 128) * 20
    )

    return {
        "ok":         len(issues) == 0,
        "score":      round(quality_score, 1),
        "blur":       round(blur, 1),
        "brightness": round(brightness, 1),
        "contrast":   round(contrast, 1),
        "issues":     issues,
    }


# ── ABCDE: A — Asymmetry ───────────────────────────────────────────────────────

def abcde_asymmetry(mask: np.ndarray) -> dict:
    """
    A criterion: measure asymmetry on the two principal lesion axes.
    Score 0 (symmetric) | 1 (one axis) | 2 (both axes asymmetric).
    """
    m = mask.astype(np.uint8)
    if m.sum() == 0:
        return {"score": 0, "axis_h": 0.0, "axis_v": 0.0, "label": "Symmetric", "risk": False}

    # Find centroid
    M   = cv2.moments(m)
    cy  = int(M["m01"] / (M["m00"] + 1e-6))
    cx  = int(M["m10"] / (M["m00"] + 1e-6))

    def _half_overlap(a, b):
        """Overlap ratio of two halves (flipped to align)."""
        min_r = min(a.shape[0], b.shape[0])
        min_c = min(a.shape[1], b.shape[1])
        a, b  = a[:min_r, :min_c], b[:min_r, :min_c]
        union = (a | b).sum()
        inter = (a & b).sum()
        return float(inter) / float(union + 1e-6)

    # Horizontal split (top vs bottom)
    top    = m[:cy, :]
    bottom = np.flipud(m[cy:, :])
    asym_h = 1.0 - _half_overlap(top.astype(bool), bottom.astype(bool))

    # Vertical split (left vs right)
    left  = m[:, :cx]
    right = np.fliplr(m[:, cx:])
    asym_v = 1.0 - _half_overlap(left.astype(bool), right.astype(bool))

    thr   = 0.18
    score = int(asym_h > thr) + int(asym_v > thr)
    labels = {0: "Symmetric", 1: "Mildly Asymmetric", 2: "Highly Asymmetric"}

    return {
        "score": score,
        "axis_h": round(asym_h, 3),
        "axis_v": round(asym_v, 3),
        "label": labels[score],
        "risk": score >= 1,
    }


# ── ABCDE: B — Border ─────────────────────────────────────────────────────────

def abcde_border(mask: np.ndarray) -> dict:
    """
    B criterion: border irregularity via the circularity index.
    irregularity = 1 − (4π·Area / Perimeter²)
    0  = perfect circle (smooth border, low risk)
    1  = maximally irregular (high risk)
    """
    m = mask.astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or m.sum() == 0:
        return {"score": 0.0, "circularity": 1.0, "label": "Regular", "risk": False}

    cnt  = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)

    if peri < 1 or area < 1:
        return {"score": 0.0, "circularity": 1.0, "label": "Regular", "risk": False}

    circularity   = (4 * np.pi * area) / (peri ** 2)
    irregularity  = round(float(np.clip(1 - circularity, 0, 1)), 3)

    if irregularity < 0.25:
        label, risk = "Regular",           False
    elif irregularity < 0.50:
        label, risk = "Mildly Irregular",  True
    else:
        label, risk = "Irregular",         True

    return {
        "score":        irregularity,
        "circularity":  round(float(circularity), 3),
        "label":        label,
        "risk":         risk,
    }


# ── ABCDE: C — Color ──────────────────────────────────────────────────────────

def abcde_color(image_rgb: np.ndarray, mask: np.ndarray) -> dict:
    """
    C criterion: count distinct color clusters inside the lesion via k-Means.
    ≥ 3 distinct clusters = higher clinical concern.
    """
    from sklearn.cluster import KMeans  # lazy import

    pixels = image_rgb[mask > 0]
    if len(pixels) < 20:
        return {"count": 1, "label": "Single Color", "hex_colors": ["#888888"], "risk": False}

    n_k = min(6, max(2, len(pixels) // 100))
    km  = KMeans(n_clusters=n_k, n_init=5, random_state=42)
    km.fit(pixels)

    labels_arr, counts = np.unique(km.labels_, return_counts=True)
    min_pop  = 0.03 * len(pixels)   # cluster must hold ≥ 3 % of pixels
    order    = counts.argsort()[::-1]

    hex_colors = []
    significant = 0
    for i in order:
        if counts[i] >= min_pop:
            r, g, b = km.cluster_centers_[i].astype(int)
            hex_colors.append(f"#{int(r):02x}{int(g):02x}{int(b):02x}")
            significant += 1

    if significant <= 2:
        label, risk = "Uniform Color",     False
    elif significant == 3:
        label, risk = "Moderate Variation", True
    else:
        label, risk = "High Variation",     True

    return {
        "count":      significant,
        "label":      label,
        "hex_colors": hex_colors,
        "risk":       risk,
    }


# ── ABCDE: D — Diameter ───────────────────────────────────────────────────────

def abcde_diameter(mask: np.ndarray) -> dict:
    """
    D criterion: estimate real-world diameter from pixel area.
    Clinical threshold: > 6 mm is a warning sign.
    """
    area_px = int(mask.astype(bool).sum())
    if area_px == 0:
        return {"area_px": 0, "area_mm2": 0.0, "diameter_mm": 0.0,
                "label": "No lesion", "risk": False, "coverage_pct": 0.0}

    area_mm2    = area_px / (PIXELS_PER_MM ** 2)
    diameter_mm = 2 * np.sqrt(area_mm2 / np.pi)
    total_px    = mask.shape[0] * mask.shape[1]
    coverage    = 100.0 * area_px / total_px
    flag        = diameter_mm > 6.0

    return {
        "area_px":     area_px,
        "area_mm2":    round(area_mm2, 2),
        "diameter_mm": round(diameter_mm, 2),
        "coverage_pct": round(coverage, 1),
        "label":       "Large (>6 mm)" if flag else "Small (<6 mm)",
        "risk":        flag,
    }


# ── Risk Score ────────────────────────────────────────────────────────────────

def compute_risk(A: dict, B: dict, C: dict, D: dict,
                 demographics: dict | None = None) -> dict:
    """
    Combined ABCD weighted risk score (0–10 scale).
    Demographic multipliers optionally applied.
    """
    raw = (A["score"] * 1.3 +
           B["score"] * 2.0 +
           C["count"] * 0.4 +
           float(D["risk"]) * 1.0)

    if demographics:
        mod = 1.0
        if demographics.get("age_over_50"):     mod *= 1.2
        if demographics.get("fair_skin"):        mod *= 1.2
        if demographics.get("family_history"):   mod *= 1.4
        if demographics.get("prev_melanoma"):    mod *= 1.8
        if demographics.get("high_sun_exposure"):mod *= 1.1
        raw = min(raw * mod, 10.0)

    score = round(float(np.clip(raw, 0, 10)), 2)

    if score < 2.5:
        level, label, color = "LOW",    "Likely Benign",          "#22c55e"
    elif score < 5.0:
        level, label, color = "MEDIUM", "Monitor — Seek Advice",  "#f59e0b"
    else:
        level, label, color = "HIGH",   "Consult Dermatologist",  "#ef4444"

    return {
        "score": score,
        "level": level,
        "label": label,
        "color": color,
    }


# ── Clinical Measurements ─────────────────────────────────────────────────────

def compute_measurements(image_rgb: np.ndarray, mask: np.ndarray) -> dict:
    """Full set of clinical measurements derived from the mask and image."""
    m = mask.astype(np.uint8)
    area_px = int(m.sum())
    if area_px == 0:
        return {}

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt          = max(contours, key=cv2.contourArea) if contours else None
    peri_px      = float(cv2.arcLength(cnt, True)) if cnt is not None else 0.0

    # Bounding box
    x, y, bw, bh = cv2.boundingRect(cnt) if cnt is not None else (0, 0, 0, 0)

    # Mean & std color inside lesion
    lesion_px = image_rgb[mask > 0].astype(float)
    mean_rgb  = lesion_px.mean(axis=0).astype(int) if len(lesion_px) else np.array([0, 0, 0])
    std_rgb   = lesion_px.std(axis=0)              if len(lesion_px) else np.array([0, 0, 0])

    mean_hex  = f"#{int(mean_rgb[0]):02x}{int(mean_rgb[1]):02x}{int(mean_rgb[2]):02x}"

    return {
        "area_px":        area_px,
        "area_mm2":       round(area_px / PIXELS_PER_MM**2, 2),
        "perimeter_px":   round(peri_px, 1),
        "perimeter_mm":   round(peri_px / PIXELS_PER_MM, 2),
        "bbox":           (x, y, bw, bh),
        "bbox_mm":        (round(x/PIXELS_PER_MM,1), round(y/PIXELS_PER_MM,1),
                           round(bw/PIXELS_PER_MM,1), round(bh/PIXELS_PER_MM,1)),
        "mean_color_rgb": mean_rgb.tolist(),
        "mean_color_hex": mean_hex,
        "std_color_rgb":  [round(v, 1) for v in std_rgb.tolist()],
        "coverage_pct":   round(100 * area_px / (mask.shape[0] * mask.shape[1]), 2),
    }


# ── Overlay ───────────────────────────────────────────────────────────────────

def make_overlay(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Returns a copy of image_rgb with:
      - semi-transparent green tint over the lesion
      - bright red contour on the lesion boundary
    """
    overlay = image_rgb.copy().astype(np.float32)
    m       = mask.astype(bool)

    # Green tint inside lesion
    tint            = np.array([0, 220, 150], dtype=np.float32)
    overlay[m]      = overlay[m] * 0.55 + tint * 0.45

    result = np.clip(overlay, 0, 255).astype(np.uint8)

    # Red contour
    cnt_mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(cnt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (255, 70, 70), 2)
    return result


# ── Grad-CAM ──────────────────────────────────────────────────────────────────

def make_gradcam(model, device, image_rgb: np.ndarray) -> np.ndarray:
    """
    Manual Grad-CAM targeting the last encoder DoubleConv block.
    Returns an H×W×3 uint8 heatmap blended with the original image.
    """
    model.eval()
    tensor = preprocess(image_rgb).to(device)

    activations: dict = {}
    gradients:   dict = {}

    def _fwd(module, inp, out):
        activations["v"] = out

    def _bwd(module, gin, gout):
        gradients["v"] = gout[0]

    target = model.encoders[-1]
    fh = target.register_forward_hook(_fwd)
    bh = target.register_full_backward_hook(_bwd)

    output = model(tensor)
    model.zero_grad()
    output.mean().backward()

    fh.remove()
    bh.remove()

    act  = activations["v"].detach().squeeze(0)   # C×H×W
    grad = gradients["v"].detach().squeeze(0)     # C×H×W
    w    = grad.mean(dim=(1, 2), keepdim=True)
    cam  = F.relu((w * act).sum(dim=0)).cpu().numpy()  # H×W

    if cam.max() > 0:
        cam = cam / cam.max()

    h, w_img = image_rgb.shape[:2]
    cam_up   = cv2.resize(cam, (w_img, h))
    heatmap  = cv2.applyColorMap((cam_up * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap  = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    blended  = (0.55 * image_rgb.astype(np.float32) +
                0.45 * heatmap.astype(np.float32)).clip(0, 255).astype(np.uint8)
    return blended


# ── Lesion Change Map ─────────────────────────────────────────────────────────

def compare_masks(mask_old: np.ndarray, mask_new: np.ndarray) -> dict:
    """
    Compare two binary masks from different timepoints.
    Returns growth stats and a three-channel change map (RGB).
    """
    # Resize new mask to match old
    if mask_old.shape != mask_new.shape:
        mask_new = cv2.resize(
            mask_new.astype(np.uint8), (mask_old.shape[1], mask_old.shape[0]),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)

    old = mask_old.astype(bool)
    new = mask_new.astype(bool)

    area_old  = int(old.sum())
    area_new  = int(new.sum())
    growth    = round((area_new - area_old) / (area_old + 1e-6) * 100, 1)
    iou       = float((old & new).sum()) / float((old | new).sum() + 1e-6)

    # Change map image
    h, w = old.shape
    change_img = np.ones((h, w, 3), dtype=np.uint8) * 30  # dark background
    # Stable (both masks)
    change_img[old & new] = [100, 200, 120]   # green
    # New growth
    change_img[~old & new] = [255, 80, 80]    # red
    # Regression
    change_img[old & ~new] = [80, 130, 255]   # blue

    return {
        "area_old_mm2":  round(area_old / PIXELS_PER_MM**2, 2),
        "area_new_mm2":  round(area_new / PIXELS_PER_MM**2, 2),
        "growth_pct":    growth,
        "iou":           round(iou, 3),
        "warning":       growth > 20 or iou < 0.70,
        "change_image":  change_img,
    }


# ── PDF Report ────────────────────────────────────────────────────────────────

def generate_pdf(image_rgb: np.ndarray, mask: np.ndarray,
                 quality: dict, abcde: dict,
                 risk: dict, meas: dict) -> bytes:
    """
    Generate a PDF clinical report and return as bytes.
    Requires: pip install fpdf2
    NOTE: Helvetica is Latin-1 only. All text passes through safe() first.
    """
    try:
        from fpdf import FPDF
    except ImportError:
        return b""

    # ── Sanitise any non-Latin-1 characters to ASCII equivalents ──────────────
    def safe(text: str) -> str:
        return (
            str(text)
            .replace("\u2014", " - ")   # em dash
            .replace("\u2013", " - ")   # en dash
            .replace("\u00b2", "2")     # superscript 2  (mm²)
            .replace("\u00b0", " deg")  # degree sign
            .replace("\u2265", ">=")    # >=
            .replace("\u2264", "<=")    # <=
            .replace("\u00d7", "x")     # multiplication sign
            .replace("\u03c0", "pi")    # Greek pi
            .replace("\u2192", "->")    # right arrow
            .encode("latin-1", errors="replace").decode("latin-1")
        )

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ── Header ──
    pdf.set_fill_color(20, 20, 50)
    pdf.rect(0, 0, 210, 28, "F")
    pdf.set_text_color(200, 180, 255)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_y(8)
    pdf.cell(0, 12, safe("DermaScan AI - Clinical Skin Lesion Report"), align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.set_y(34)

    # ── Images ──
    def _np_to_tmpfile(arr, suffix=".jpg"):
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        Image.fromarray(arr).save(tmp.name)
        return tmp.name

    orig_path    = _np_to_tmpfile(image_rgb)
    overlay_path = _np_to_tmpfile(make_overlay(image_rgb, mask))

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(95, 8, "Original Image", align="C")
    pdf.cell(95, 8, "Segmentation Overlay", align="C")
    pdf.ln(2)
    pdf.image(orig_path,    x=10,  y=pdf.get_y(), w=88, h=66)
    pdf.image(overlay_path, x=112, y=pdf.get_y(), w=88, h=66)
    pdf.ln(70)

    # ── Body helper ──
    def section(title):
        pdf.set_fill_color(230, 225, 255)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, safe(f"  {title}"), fill=True, ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.ln(1)

    def row(label, value, indent=8):
        pdf.set_x(indent)
        pdf.cell(72, 6, safe(str(label)))
        pdf.cell(0,  6, safe(str(value)), ln=True)

    # ── Quality ──
    section("Image Quality")
    row("Quality Score",  f"{quality['score']} / 100")
    row("Blur Score",     quality['blur'])
    row("Brightness",     quality['brightness'])
    row("Contrast",       quality['contrast'])
    pdf.ln(3)

    # ── ABCDE ──
    section("ABCDE Clinical Analysis")
    A, B, C, D = abcde["A"], abcde["B"], abcde["C"], abcde["D"]
    row("A  Asymmetry",  f"{A['score']} / 2 - {A['label']}")
    row("B  Border",     f"{B['score']:.2f} irregularity - {B['label']}")
    row("C  Color",      f"{C['count']} distinct clusters - {C['label']}")
    row("D  Diameter",   f"{D['diameter_mm']:.1f} mm - {D['label']}")
    pdf.ln(3)

    # ── Measurements ──
    section("Clinical Measurements")
    if meas:
        row("Area",        f"{meas['area_mm2']} mm2  ({meas['area_px']} px)")
        row("Perimeter",   f"{meas['perimeter_mm']} mm")
        row("Coverage",    f"{meas['coverage_pct']} % of image")
        row("Mean Color",  meas['mean_color_hex'].upper())
    pdf.ln(3)

    # ── Risk Score ──
    section("Risk Assessment")
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(*_hex_to_rgb(risk["color"]))
    pdf.cell(0, 10, safe(f"  {risk['level']}   Score: {risk['score']} / 10.0"), ln=True)
    pdf.cell(0,  8, safe(f"  {risk['label']}"), ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "I", 9)
    pdf.ln(2)
    pdf.multi_cell(0, 6, safe(
        "DISCLAIMER: This report is generated by an AI screening tool and does NOT "
        "constitute a medical diagnosis. Always consult a qualified dermatologist "
        "for clinical evaluation."
    ))

    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()


def _hex_to_rgb(hex_color: str):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
