"""
dermascan_app.py
----------------
DermaScan AI — Clinical Skin Lesion Analysis
A feature-rich Streamlit app built on top of the ISIC 2018 U-Net pipeline.

Run:
    streamlit run dermascan_app.py
"""

import os
import sys
import io
import warnings
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import torch
from PIL import Image
import streamlit as st
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from model import UNet
import app_analysis as ana

# ── Inline constants (no config.py needed in Space) ───────────────────────────
MODEL_REPO = "pavanpraneeth/isic-unet"
MODEL_FILE = "best_model.pth"
IMAGE_CHANNELS = 3
MASK_CHANNELS  = 1

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps")  if torch.backends.mps.is_available()
    else torch.device("cpu")
)

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DermaScan AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Premium CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark background */
.stApp { background: #09090f; color: #e8e4ff; }
[data-testid="stSidebar"] { background: #111120 !important; border-right: 1px solid rgba(120,100,255,.18); }
[data-testid="stSidebar"] .stMarkdown { color: #e8e4ff; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #141430;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: 1px solid rgba(120,100,255,.2);
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #7a76a8 !important;
    border-radius: 8px !important;
    font-weight: 600;
    font-size: 0.88rem;
    padding: 8px 16px !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,#7c5cfc,#c550ff) !important;
    color: #fff !important;
}

/* Metrics */
[data-testid="metric-container"] {
    background: #141430;
    border: 1px solid rgba(120,100,255,.18);
    border-radius: 12px;
    padding: 16px !important;
}
[data-testid="stMetricValue"] { color: #e8e4ff !important; font-weight: 800; }
[data-testid="stMetricLabel"] { color: #7a76a8 !important; font-size: 0.78rem !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #7c5cfc, #c550ff) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 12px 32px !important;
    transition: all 0.2s !important;
    width: 100%;
}
.stButton > button:hover { opacity: 0.88; transform: translateY(-1px); box-shadow: 0 8px 24px rgba(124,92,252,.35); }

/* File uploader */
[data-testid="stFileUploader"] {
    background: #141430;
    border: 2px dashed rgba(124,92,252,.4);
    border-radius: 16px;
    padding: 20px;
}

/* Expander */
.streamlit-expanderHeader {
    background: #141430 !important;
    border-radius: 10px !important;
    color: #a89aff !important;
    font-weight: 600;
}

/* Info / warning / success boxes */
.stAlert { border-radius: 10px !important; }

/* Divider */
hr { border-color: rgba(120,100,255,.15) !important; }

/* Code block override */
code { background: #0d0d1e !important; color: #c3e88d !important; }

/* Custom card */
.ds-card {
    background: #141430;
    border: 1px solid rgba(120,100,255,.18);
    border-radius: 16px;
    padding: 20px 22px;
    margin-bottom: 14px;
}
.ds-card h4 { color: #a89aff; margin: 0 0 10px 0; font-size: 0.9rem; letter-spacing: 0.06em; text-transform: uppercase; }

/* Risk badge */
.risk-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 24px;
    font-weight: 800;
    font-size: 0.85rem;
    letter-spacing: 0.05em;
}

/* Color swatch */
.swatch {
    display: inline-block;
    width: 28px;
    height: 28px;
    border-radius: 6px;
    border: 1px solid rgba(255,255,255,.15);
    margin-right: 6px;
    vertical-align: middle;
}

/* Progress bar override */
.stProgress > div > div > div { background: linear-gradient(90deg,#7c5cfc,#c550ff) !important; border-radius: 4px; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: rgba(124,92,252,.3); border-radius: 4px; }

/* Sidebar section */
.sidebar-section {
    background: rgba(124,92,252,.08);
    border: 1px solid rgba(124,92,252,.2);
    border-radius: 12px;
    padding: 14px;
    margin-bottom: 14px;
}

/* Image caption override */
[data-testid="caption"] { color: #7a76a8 !important; font-size: 0.8rem !important; text-align: center; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: small HTML cards
# ─────────────────────────────────────────────────────────────────────────────
def html_card(title: str, content: str) -> str:
    return f"""<div class="ds-card"><h4>{title}</h4>{content}</div>"""

def pill(text: str, color: str) -> str:
    return f'<span style="background:{color}22;color:{color};padding:3px 10px;border-radius:12px;font-size:.8rem;font-weight:700;border:1px solid {color}55;">{text}</span>'

def risk_pill(level: str, color: str) -> str:
    return f'<span class="risk-badge" style="background:{color}22;color:{color};border:1px solid {color}55;">{level}</span>'


# ─────────────────────────────────────────────────────────────────────────────
# Model Loading (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading U-Net model…")
def load_model():
    from huggingface_hub import hf_hub_download
    model  = UNet(in_channels=IMAGE_CHANNELS, out_channels=MASK_CHANNELS)

    try:
        ckpt_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
        state     = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(state["model_state_dict"])
        best_dice = state.get("best_val_dice", 0.0)
        status    = f"Loaded  |  Val Dice {best_dice:.4f}"
    except Exception as e:
        status = f"Warning: could not load checkpoint ({e})"

    model.eval().to(DEVICE)
    return model, DEVICE, status


model, DEVICE, model_status = load_model()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Branding
    st.markdown("""
    <div style="text-align:center;padding:12px 0 20px;">
        <div style="font-size:2.4rem;">🔬</div>
        <div style="font-size:1.3rem;font-weight:800;background:linear-gradient(135deg,#7c5cfc,#c550ff);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            DermaScan AI
        </div>
        <div style="color:#7a76a8;font-size:.78rem;margin-top:4px;">
            Clinical Skin Lesion Analysis
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sidebar-section">
        <div style="font-size:.75rem;color:#7a76a8;text-transform:uppercase;
                    letter-spacing:.08em;font-weight:700;margin-bottom:6px;">Model Status</div>
        <div style="font-size:.85rem;color:#a89aff;">{model_status}</div>
        <div style="font-size:.75rem;color:#7a76a8;margin-top:4px;">Device: {DEVICE}</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Demographics ──
    st.markdown("#### 👤 Patient Context *(optional)*")
    st.caption("Risk score is adjusted if details are provided.")

    age_over_50    = st.checkbox("Age > 50", value=False)
    fair_skin      = st.checkbox("Fair skin / Fitzpatrick I–II", value=False)
    family_history = st.checkbox("Family history of melanoma", value=False)
    prev_melanoma  = st.checkbox("Previous melanoma", value=False)
    high_sun       = st.checkbox("High sun exposure history", value=False)

    demographics = {
        "age_over_50":      age_over_50,
        "fair_skin":        fair_skin,
        "family_history":   family_history,
        "prev_melanoma":    prev_melanoma,
        "high_sun_exposure":high_sun,
    }

    st.divider()

    # ── Lesion Tracker ──
    st.markdown("#### 📅 Evolution Tracker")
    st.caption("Upload a previous scan to detect lesion growth.")
    prev_upload = st.file_uploader(
        "Previous scan (optional)", type=["jpg","jpeg","png"],
        key="prev_scan", label_visibility="collapsed"
    )

    st.divider()

    st.markdown("""
    <div style="font-size:.72rem;color:#7a76a8;line-height:1.6;">
    ⚠️ <strong style="color:#f59e0b;">Disclaimer</strong><br>
    DermaScan AI is a computer-aided
    <em>screening tool</em>, not a medical
    device. Always consult a qualified
    dermatologist for clinical evaluation.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:8px 0 28px;">
    <h1 style="font-size:2.2rem;font-weight:900;margin:0;letter-spacing:-.02em;">
        🔬 DermaScan <span style="background:linear-gradient(135deg,#7c5cfc,#c550ff);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">AI</span>
    </h1>
    <p style="color:#7a76a8;margin:6px 0 0;font-size:1rem;">
        Upload a dermoscopy image for automated ABCDE analysis, clinical measurements,
        risk scoring, and explainability heatmaps — powered by a trained U-Net.
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Upload Section
# ─────────────────────────────────────────────────────────────────────────────
col_up, col_pre = st.columns([1, 1], gap="large")

with col_up:
    uploaded = st.file_uploader(
        "Upload dermoscopy image",
        type=["jpg","jpeg","png"],
        label_visibility="collapsed",
        key="main_upload",
    )

    if uploaded:
        pil_img    = Image.open(uploaded).convert("RGB")
        image_rgb  = np.array(pil_img)
        st.image(image_rgb, caption="Uploaded Image", use_container_width=True)

with col_pre:
    if uploaded:
        quality = ana.check_image_quality(image_rgb)
        q_color = "#22c55e" if quality["ok"] else ("#f59e0b" if len(quality["issues"]) == 1 else "#ef4444")
        q_icon  = "✅" if quality["ok"] else ("⚠️" if len(quality["issues"]) == 1 else "❌")

        st.markdown(f"""
        <div class="ds-card">
            <h4>Image Quality Check</h4>
            <div style="font-size:2rem;font-weight:900;color:{q_color};">{quality['score']}<span style="font-size:1rem;color:#7a76a8;font-weight:400;"> / 100</span></div>
            <div style="color:{q_color};font-weight:700;margin-bottom:10px;">{q_icon} {"Good to analyze" if quality['ok'] else "Proceed with caution"}</div>
        """, unsafe_allow_html=True)

        cq1, cq2, cq3 = st.columns(3)
        cq1.metric("Blur Score",  quality["blur"])
        cq2.metric("Brightness",  quality["brightness"])
        cq3.metric("Contrast",    quality["contrast"])

        if quality["issues"]:
            for issue in quality["issues"]:
                st.warning(issue)
        st.markdown("</div>", unsafe_allow_html=True)

if not uploaded:
    st.info("👆  Upload a dermoscopy image to begin analysis.")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Run Analysis
# ─────────────────────────────────────────────────────────────────────────────
run_btn = st.button("🚀  Run DermaScan Analysis", use_container_width=False)

if not run_btn and "results" not in st.session_state:
    st.stop()

if run_btn:
    with st.spinner("Running segmentation and clinical analysis…"):

        # ── Segmentation ──
        mask, prob_map = ana.run_segmentation(model, DEVICE, image_rgb)

        # ── ABCDE ──
        A = ana.abcde_asymmetry(mask)
        B = ana.abcde_border(mask)
        C = ana.abcde_color(image_rgb, mask)
        D = ana.abcde_diameter(mask)

        # ── Risk ──
        risk = ana.compute_risk(A, B, C, D, demographics)

        # ── Measurements ──
        meas = ana.compute_measurements(image_rgb, mask)

        # ── Visuals ──
        overlay   = ana.make_overlay(image_rgb, mask)
        gradcam   = ana.make_gradcam(model, DEVICE, image_rgb)

        # ── Evolution (if prev scan provided) ──
        evo = None
        if prev_upload:
            prev_img  = np.array(Image.open(prev_upload).convert("RGB"))
            prev_mask, _ = ana.run_segmentation(model, DEVICE, prev_img)
            evo = ana.compare_masks(prev_mask, mask)

        # Store in session
        st.session_state["results"] = dict(
            mask=mask, prob_map=prob_map, overlay=overlay,
            gradcam=gradcam, A=A, B=B, C=C, D=D,
            risk=risk, meas=meas, evo=evo, quality=quality,
        )

# Pull from session
res = st.session_state["results"]
mask, prob_map  = res["mask"], res["prob_map"]
overlay, gradcam = res["overlay"], res["gradcam"]
A, B, C, D      = res["A"], res["B"], res["C"], res["D"]
risk, meas, evo = res["risk"], res["meas"], res["evo"]

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# --- TOP SUMMARY ROW ---
# ─────────────────────────────────────────────────────────────────────────────
s1, s2, s3, s4, s5 = st.columns(5)
s1.metric("Risk Level",   risk["level"])
s2.metric("A — Asymmetry", f"{A['score']} / 2",  delta=A["label"])
s3.metric("B — Border",    f"{B['score']:.2f}",   delta=B["label"])
s4.metric("C — Colors",    C["count"],            delta=C["label"])
s5.metric("Diameter Est.", f"{D['diameter_mm']} mm", delta=D["label"])

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# --- TABS ---
# ─────────────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🎯  Overview",
    "🔬  ABCDE Analysis",
    "📐  Measurements",
    "🧠  Explainability",
    "📅  Evolution",
    "📄  Report",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    ov1, ov2, ov3 = st.columns([1.1, 1.05, 0.95])

    with ov1:
        st.markdown("**Segmentation Overlay**")
        st.image(overlay, use_container_width=True)
        st.markdown("**Predicted Mask**")
        mask_vis = (mask.astype(np.uint8) * 255)
        st.image(mask_vis, use_container_width=True, clamp=True)

    with ov2:
        gauge_color = risk["color"]

        # ── Score number displayed cleanly above gauge ──
        st.markdown(f"""
        <div style="text-align:center;padding:10px 0 4px;">
            <div style="font-size:.8rem;color:#7a76a8;letter-spacing:.08em;
                        text-transform:uppercase;font-weight:700;margin-bottom:4px;">
                Risk Score
            </div>
            <div style="font-size:3rem;font-weight:900;color:{gauge_color};line-height:1;">
                {risk['score']}
                <span style="font-size:1.1rem;color:#7a76a8;font-weight:400;">/ 10</span>
            </div>
            <div style="margin-top:6px;">
                <span style="background:{gauge_color}22;color:{gauge_color};
                             padding:4px 16px;border-radius:20px;font-weight:800;
                             font-size:.85rem;border:1px solid {gauge_color}55;">
                    {risk['level']}
                </span>
            </div>
            <div style="color:#7a76a8;font-size:.82rem;margin-top:6px;">{risk['label']}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Gauge (number hidden — shown above instead) ──
        fig = go.Figure(go.Indicator(
            mode="gauge",
            value=risk["score"],
            gauge={
                "axis": {
                    "range": [0, 10],
                    "tickvals": [0, 2.5, 5, 7.5, 10],
                    "ticktext": ["0", "2.5", "5", "7.5", "10"],
                    "tickcolor": "#7a76a8",
                    "tickfont": {"color": "#7a76a8", "size": 11},
                },
                "bar": {"color": gauge_color, "thickness": 0.3},
                "bgcolor": "#1a1a36",
                "borderwidth": 1,
                "bordercolor": "rgba(120,100,255,.3)",
                "steps": [
                    {"range": [0,   2.5], "color": "#0d2b12"},
                    {"range": [2.5, 5.0], "color": "#2b2100"},
                    {"range": [5.0, 10],  "color": "#2b0a0a"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.85,
                    "value": risk["score"],
                },
            },
        ))
        fig.update_layout(
            height=200,
            margin=dict(l=16, r=16, t=8, b=8),
            paper_bgcolor="#09090f",
            font={"family": "Inter"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with ov3:
        # ABCDE bar chart
        criterion_labels = ["A  Asymmetry", "B  Border", "C  Color Var.", "D  Diameter"]
        raw_vals = [
            A["score"] / 2.0,
            B["score"],
            min(C["count"] / 6.0, 1.0),
            float(D["risk"]),
        ]
        bar_colors = [
            "#ef4444" if v > 0.5 else "#f59e0b" if v > 0.2 else "#22c55e"
            for v in raw_vals
        ]
        fig2 = go.Figure(go.Bar(
            x=raw_vals,
            y=criterion_labels,
            orientation="h",
            marker=dict(color=bar_colors, line=dict(color="rgba(0,0,0,0)")),
            text=[f"{v:.0%}" for v in raw_vals],
            textposition="outside",
            textfont=dict(color="#e8e4ff", size=11, family="Inter"),
        ))
        fig2.update_layout(
            title=dict(text="ABCD Criteria", font={"color": "#e8e4ff", "size": 13}),
            xaxis=dict(range=[0, 1.25], showgrid=False, visible=False),
            yaxis=dict(tickfont={"color": "#a89aff", "size": 12}),
            plot_bgcolor="#141430",
            paper_bgcolor="#09090f",
            margin=dict(l=8, r=40, t=36, b=8),
            height=210,
        )
        st.plotly_chart(fig2, use_container_width=True)

        if meas:
            st.markdown(f"""
            <div class="ds-card" style="margin-top:4px;padding:14px 18px;">
                <h4>Quick Stats</h4>
                <table style="width:100%;font-size:.85rem;border-collapse:collapse;">
                    <tr style="border-bottom:1px solid rgba(120,100,255,.1);">
                        <td style="color:#7a76a8;padding:5px 0;">Area</td>
                        <td style="text-align:right;color:#a89aff;font-weight:700;">{meas['area_mm2']} mm²</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(120,100,255,.1);">
                        <td style="color:#7a76a8;padding:5px 0;">Perimeter</td>
                        <td style="text-align:right;color:#a89aff;font-weight:700;">{meas['perimeter_mm']} mm</td>
                    </tr>
                    <tr>
                        <td style="color:#7a76a8;padding:5px 0;">Coverage</td>
                        <td style="text-align:right;color:#a89aff;font-weight:700;">{meas['coverage_pct']}%</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)




# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ABCDE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("### ABCDE Clinical Criteria")
    st.caption("The ABCDE rule is the standard dermatological checklist for melanoma screening. "
               "Each criterion is computed directly from the segmentation mask and the original image.")

    def abcde_card(letter, name, value_str, label, risk_flag, detail):
        icon  = "🔴" if risk_flag else "🟢"
        color = "#ef4444" if risk_flag else "#22c55e"
        st.markdown(f"""
        <div class="ds-card" style="border-left:4px solid {color};">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                <div>
                    <span style="font-size:1.6rem;font-weight:900;color:{color};">{letter}</span>
                    <span style="font-size:1rem;font-weight:700;color:#a89aff;margin-left:8px;">{name}</span>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:1.4rem;font-weight:800;color:#e8e4ff;">{value_str}</div>
                    <div style="font-size:.8rem;color:{color};font-weight:700;">{icon} {label}</div>
                </div>
            </div>
            <div style="color:#7a76a8;font-size:.85rem;margin-top:10px;line-height:1.6;">{detail}</div>
        </div>
        """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        abcde_card(
            "A", "Asymmetry",
            f"{A['score']} / 2",
            A["label"], A["risk"],
            f"Horizontal axis asymmetry: <strong>{A['axis_h']:.3f}</strong>&nbsp;&nbsp;"
            f"Vertical axis asymmetry: <strong>{A['axis_v']:.3f}</strong><br>"
            "Score of 1 = asymmetric on one axis. Score 2 = both axes. "
            "Threshold per axis: 0.18 overlap mismatch ratio."
        )
        abcde_card(
            "C", "Color Variegation",
            f"{C['count']} clusters",
            C["label"], C["risk"],
            f"K-Means clustering (k≤6) applied to lesion pixels. "
            f"<br>Clusters with &gt;3% of lesion pixel population are counted as significant. "
            f"≥3 distinct clusters raises concern."
        )

    with c2:
        abcde_card(
            "B", "Border Irregularity",
            f"{B['score']:.3f}",
            B["label"], B["risk"],
            f"Irregularity = 1 − (4π·Area / Perimeter²). "
            f"Circularity index: <strong>{B['circularity']:.3f}</strong> "
            f"(1.0 = perfect circle). "
            "Score > 0.25: irregular. > 0.50: highly irregular."
        )
        abcde_card(
            "D", "Diameter",
            f"{D['diameter_mm']} mm",
            D["label"], D["risk"],
            f"Estimated from pixel area assuming {ana.PIXELS_PER_MM:.0f} px/mm (standard dermoscope). "
            f"Area: <strong>{D['area_mm2']} mm²</strong> ({D['area_px']} px). "
            "Clinical threshold: > 6 mm → warning."
        )

    # Colour palette for C
    if C["hex_colors"]:
        st.markdown("---")
        st.markdown("**Detected Lesion Color Clusters**")
        swatches = "".join(
            f'<span class="swatch" style="background:{hx};" title="{hx}"></span>'
            f'<code style="font-size:.8rem;margin-right:10px;">{hx}</code>'
            for hx in C["hex_colors"]
        )
        st.markdown(swatches, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MEASUREMENTS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("### Clinical Measurements")
    st.caption(f"All real-world estimates assume {ana.PIXELS_PER_MM:.0f} px/mm dermoscope calibration.")

    if not meas:
        st.warning("No lesion detected in the mask — check segmentation threshold.")
        st.stop()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Area",           f"{meas['area_mm2']} mm²",  delta=f"{meas['area_px']} px")
    m2.metric("Perimeter",      f"{meas['perimeter_mm']} mm")
    m3.metric("Image Coverage", f"{meas['coverage_pct']} %")
    m4.metric("Mean Lesion Color", meas["mean_color_hex"].upper())

    st.markdown("---")
    c_det, c_vis = st.columns([1, 1])

    with c_det:
        st.markdown(f"""
        <div class="ds-card">
            <h4>Full Measurement Table</h4>
            <table style="width:100%;border-collapse:collapse;font-size:.88rem;">
                <tr style="border-bottom:1px solid rgba(120,100,255,.12);">
                    <td style="padding:8px 4px;color:#7a76a8;">Area (pixels)</td>
                    <td style="padding:8px 4px;color:#e8e4ff;font-weight:700;text-align:right;">{meas['area_px']:,}</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(120,100,255,.12);">
                    <td style="padding:8px 4px;color:#7a76a8;">Area (mm²)</td>
                    <td style="padding:8px 4px;color:#e8e4ff;font-weight:700;text-align:right;">{meas['area_mm2']}</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(120,100,255,.12);">
                    <td style="padding:8px 4px;color:#7a76a8;">Perimeter (px)</td>
                    <td style="padding:8px 4px;color:#e8e4ff;font-weight:700;text-align:right;">{meas['perimeter_px']}</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(120,100,255,.12);">
                    <td style="padding:8px 4px;color:#7a76a8;">Perimeter (mm)</td>
                    <td style="padding:8px 4px;color:#e8e4ff;font-weight:700;text-align:right;">{meas['perimeter_mm']}</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(120,100,255,.12);">
                    <td style="padding:8px 4px;color:#7a76a8;">Image Coverage</td>
                    <td style="padding:8px 4px;color:#e8e4ff;font-weight:700;text-align:right;">{meas['coverage_pct']} %</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(120,100,255,.12);">
                    <td style="padding:8px 4px;color:#7a76a8;">Mean Color (RGB)</td>
                    <td style="padding:8px 4px;color:#e8e4ff;font-weight:700;text-align:right;">{tuple(meas['mean_color_rgb'])}</td>
                </tr>
                <tr>
                    <td style="padding:8px 4px;color:#7a76a8;">Color Std Dev</td>
                    <td style="padding:8px 4px;color:#e8e4ff;font-weight:700;text-align:right;">{tuple(meas['std_color_rgb'])}</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with c_vis:
        # Bounding-box visualization
        bbox_img = image_rgb.copy()
        x, y, bw, bh = meas["bbox"]
        cv2.rectangle(bbox_img, (x, y), (x+bw, y+bh), (124, 92, 252), 2)
        # Centroid
        cx_m = x + bw // 2
        cy_m = y + bh // 2
        cv2.drawMarker(bbox_img, (cx_m, cy_m), (0, 229, 163), cv2.MARKER_CROSS, 16, 2)
        st.image(bbox_img, caption="Bounding box + lesion centroid", use_container_width=True)

        # Mean color swatch
        mc = meas["mean_color_hex"]
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;margin-top:12px;">
            <div style="width:50px;height:50px;border-radius:10px;background:{mc};border:1px solid rgba(255,255,255,.2);"></div>
            <div>
                <div style="font-weight:700;color:#e8e4ff;">{mc.upper()}</div>
                <div style="color:#7a76a8;font-size:.8rem;">Mean lesion color</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Probability heatmap
        st.markdown("**Segmentation Confidence Map**")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        fig3, ax = plt.subplots(figsize=(5, 4))
        fig3.patch.set_facecolor("#09090f")
        ax.set_facecolor("#09090f")
        im = ax.imshow(prob_map, cmap="magma", vmin=0, vmax=1)
        cbar = fig3.colorbar(im, ax=ax)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
        ax.set_title("P(lesion) per pixel", color="#a89aff", fontsize=11)
        ax.axis("off")
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("### Model Explainability — Grad-CAM")
    st.caption(
        "Gradient-weighted Class Activation Mapping (Grad-CAM) reveals which image regions "
        "activated the model's last encoder layer most strongly during prediction. "
        "Hot regions (red/yellow) had the greatest influence on the segmentation output."
    )

    e1, e2, e3 = st.columns(3)
    with e1:
        st.image(image_rgb, caption="Original", use_container_width=True)
    with e2:
        st.image(gradcam, caption="Grad-CAM Heatmap", use_container_width=True)
    with e3:
        st.image(overlay, caption="Segmentation", use_container_width=True)

    st.markdown("---")
    st.markdown(f"""
    <div class="ds-card">
        <h4>How to Read This</h4>
        <p style="color:#a89aff;font-size:.88rem;line-height:1.7;">
        🔴 <strong>Red / Yellow</strong> — Regions the encoder focused on most intensely.<br>
        🔵 <strong>Blue / Dark</strong> — Low activation regions, less important to the prediction.<br><br>
        Ideally, the hottest regions should overlap with the visible lesion.
        If activation appears in the background, it may indicate the model is relying
        on surrounding skin texture — a sign the pretrained encoder would be beneficial.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — EVOLUTION
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("### Lesion Evolution Tracker")

    if evo is None:
        st.info(
            "📅 Upload a **previous scan** in the sidebar to compare this lesion "
            "against an earlier timepoint and detect growth."
        )
    else:
        growth_color = "#ef4444" if evo["warning"] else "#22c55e"
        ev1, ev2, ev3, ev4 = st.columns(4)
        ev1.metric("Previous Area", f"{evo['area_old_mm2']} mm²")
        ev2.metric("Current Area",  f"{evo['area_new_mm2']} mm²")
        ev3.metric("Growth",        f"{evo['growth_pct']} %",
                   delta=("⚠ Growth detected" if evo["warning"] else "Stable"))
        ev4.metric("Shape Overlap (IoU)", f"{evo['iou']:.3f}")

        if evo["warning"]:
            st.error("⚠️ **Significant change detected.** Growth > 20% or shape overlap < 0.70. Please consult a dermatologist.")
        else:
            st.success("✅ Lesion appears stable compared to the previous scan.")

        st.markdown("---")
        st.markdown("**Change Map**")
        st.image(evo["change_image"], use_container_width=False, width=400,
                 caption="🟢 Stable  |  🔴 New growth  |  🔵 Regression")
        st.caption("Pixel-level comparison between previous and current segmentation masks.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — REPORT
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("### Clinical Report")

    # Preview
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    st.markdown(f"""
    <div class="ds-card">
        <h4>Report Preview</h4>
        <div style="font-family:'JetBrains Mono',monospace;font-size:.82rem;line-height:2;color:#a89aff;">
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━<br>
        &nbsp;&nbsp;DERMASCAN AI — CLINICAL LESION REPORT<br>
        &nbsp;&nbsp;Generated: {timestamp}<br>
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━<br><br>
        <span style="color:#7a76a8;">IMAGE QUALITY</span><br>
        &nbsp;&nbsp;Score: {res['quality']['score']} / 100 — {"OK" if res['quality']['ok'] else "Issues detected"}<br><br>
        <span style="color:#7a76a8;">ABCDE ANALYSIS</span><br>
        &nbsp;&nbsp;A  Asymmetry:  {A['score']} / 2  — {A['label']}<br>
        &nbsp;&nbsp;B  Border:     {B['score']:.3f}    — {B['label']}<br>
        &nbsp;&nbsp;C  Color:      {C['count']} clusters — {C['label']}<br>
        &nbsp;&nbsp;D  Diameter:   {D['diameter_mm']} mm — {D['label']}<br><br>
        <span style="color:#7a76a8;">CLINICAL MEASUREMENTS</span><br>
        &nbsp;&nbsp;Area:       {meas.get('area_mm2','—')} mm²<br>
        &nbsp;&nbsp;Perimeter:  {meas.get('perimeter_mm','—')} mm<br>
        &nbsp;&nbsp;Coverage:   {meas.get('coverage_pct','—')} %<br><br>
        <span style="color:#7a76a8;">RISK ASSESSMENT</span><br>
        <span style="color:{risk['color']};font-weight:700;">
        &nbsp;&nbsp;Score: {risk['score']} / 10.0 — {risk['level']}<br>
        &nbsp;&nbsp;{risk['label']}<br></span><br>
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━<br>
        <span style="color:#555;font-size:.75rem;">DISCLAIMER: AI screening tool — NOT a medical diagnosis.</span><br>
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        </div>
    </div>
    """, unsafe_allow_html=True)

    # PDF Download
    st.markdown("---")
    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        pdf_bytes = ana.generate_pdf(
            image_rgb, mask,
            res["quality"],
            {"A": A, "B": B, "C": C, "D": D},
            risk, meas,
        )
        if pdf_bytes:
            st.download_button(
                label="⬇️  Download PDF Report",
                data=pdf_bytes,
                file_name=f"dermascan_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.info("Install `fpdf2` for PDF export: `pip install fpdf2`")

    with col_dl2:
        # Plain-text download (always works)
        txt_report = f"""DERMASCAN AI — CLINICAL LESION REPORT
Generated: {timestamp}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IMAGE QUALITY
  Score:      {res['quality']['score']} / 100
  Blur:       {res['quality']['blur']}
  Brightness: {res['quality']['brightness']}
  Contrast:   {res['quality']['contrast']}

ABCDE ANALYSIS
  A  Asymmetry:    {A['score']} / 2  — {A['label']}    (H={A['axis_h']}, V={A['axis_v']})
  B  Border:       {B['score']:.3f}       — {B['label']}
  C  Color:        {C['count']} clusters  — {C['label']}
  D  Diameter:     {D['diameter_mm']} mm — {D['label']}   (Area: {D['area_mm2']} mm²)

CLINICAL MEASUREMENTS
  Area:        {meas.get('area_mm2','—')} mm²  ({meas.get('area_px','—')} px)
  Perimeter:   {meas.get('perimeter_mm','—')} mm
  Coverage:    {meas.get('coverage_pct','—')} % of image
  Mean Color:  {meas.get('mean_color_hex','—').upper()}

RISK ASSESSMENT
  Score:  {risk['score']} / 10.0
  Level:  {risk['level']}
  Label:  {risk['label']}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DISCLAIMER: This report is generated by an AI screening tool and does
NOT constitute a medical diagnosis. Always consult a qualified
dermatologist for clinical evaluation.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        st.download_button(
            label="⬇️  Download Text Report",
            data=txt_report,
            file_name=f"dermascan_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True,
        )
