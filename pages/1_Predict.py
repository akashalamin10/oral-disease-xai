import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from src.model_loader import load_all_models
from src.predictor import predict
from src.utils import overlay_heatmap, MODEL_ORDER

st.set_page_config(page_title="Predict — OralXAI", page_icon="🔬", layout="wide")

@st.cache_resource(show_spinner="Downloading models (first time only)...")
def get_models():
    return load_all_models(use_hf_hub=True, hf_repo_id="akash4529/oral-disease-sage")

st.title("Oral Disease Prediction")
st.markdown("Upload an intraoral image to get a SAGE ensemble prediction with Grad-CAM explanations.")
st.markdown("---")

model_dense, model_b4, model_swin, gate = get_models()

uploaded = st.file_uploader("Upload Intraoral Image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    pil_img = Image.open(uploaded).convert("RGB")

    col_img, col_info = st.columns([1, 2])
    with col_img:
        st.image(pil_img, caption="Uploaded image", use_column_width=True)

    with col_info:
        with st.spinner("Running SAGE ensemble prediction..."):
            result = predict(pil_img, model_dense, model_b4, model_swin, gate)
            st.session_state["last_result"] = result

        conf = result["confidence"]
        pred = result["predicted_display"]

        if conf >= 80:
            st.success(f"Prediction: **{pred}**")
        elif conf >= 60:
            st.warning(f"Prediction: **{pred}**")
        else:
            st.error(f"Prediction: **{pred}** (low confidence)")

        st.metric("SAGE Confidence", f"{conf:.2f}%")
        st.metric("Spatial Consensus Score (SCS)", f"{result['scs']['mean']:.4f}")

        st.markdown("**Gate Weights**")
        gw = result["gate_weights"]
        gcols = st.columns(3)
        for i, name in enumerate(MODEL_ORDER):
            gcols[i].metric(name, f"{gw[name]:.3f}")

    st.markdown("---")
    st.subheader("Per-class Probabilities (SAGE)")

    sage_probs = result["sage_probs"]
    classes    = list(sage_probs.keys())
    probs      = list(sage_probs.values())
    colors     = ["#3b82f6" if c != pred else "#10b981" for c in classes]

    fig, ax = plt.subplots(figsize=(10, 3))
    bars = ax.barh(classes, [p * 100 for p in probs], color=colors, height=0.5)
    ax.set_xlabel("Probability (%)")
    ax.set_xlim(0, 105)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("Grad-CAM Heatmaps")
    st.caption("Red = high attention | Blue = low attention")

    base   = result["base_img"]
    cams   = result["cams"]
    fig2, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(base)
    axes[0].set_title("Original", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    for i, name in enumerate(MODEL_ORDER):
        overlay = overlay_heatmap(base, cams[name], alpha=0.45)
        axes[i+1].imshow(overlay)
        axes[i+1].set_title(f"{name}\nweight={gw[name]:.3f}", fontsize=10)
        axes[i+1].axis("off")

    plt.suptitle(
        f"SAGE: {pred} ({conf:.1f}%) | SCS={result['scs']['mean']:.3f}",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.markdown("---")
    with st.expander("Individual Model Predictions"):
        import pandas as pd
        rows = []
        for cls in classes:
            rows.append({
                "Class": cls,
                "SAGE (%)": f"{sage_probs[cls]*100:.2f}",
                "DenseNet-121 (%)": f"{result['probs_dense'][cls]*100:.2f}",
                "EfficientNet-B4 (%)": f"{result['probs_b4'][cls]*100:.2f}",
                "Swin-Tiny (%)": f"{result['probs_swin'][cls]*100:.2f}",
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

    with st.expander("SCS Details"):
        scs = result["scs"]
        st.markdown(f"""
        | Pair | Score |
        |------|-------|
        | DenseNet-121 vs EfficientNet-B4 | `{scs['dense_b4']:.4f}` |
        | DenseNet-121 vs Swin-Tiny | `{scs['dense_swin']:.4f}` |
        | EfficientNet-B4 vs Swin-Tiny | `{scs['b4_swin']:.4f}` |
        | **Mean SCS** | **`{scs['mean']:.4f}`** |
        """)
        st.caption("SCS = 0.5 × (IoU + normalized cosine similarity) per pair. Higher = more consistent explanations.")

    st.info("Go to **Report** in the sidebar to download a PDF report of this prediction.")

else:
    st.info("Please upload an intraoral image (JPG or PNG) to begin prediction.")
    st.markdown("""
    **Supported conditions:**
    - Calculus · Caries · Gingivitis
    - Hypodontia · Tooth Discoloration · Ulcers
    """)
