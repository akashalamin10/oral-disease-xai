import streamlit as st

st.set_page_config(page_title="About — OralXAI", page_icon="ℹ️", layout="wide")

st.title("About OralXAI")
st.markdown("---")

st.markdown("""
### Project Overview
**OralXAI** is an explainable deep learning system for automatic oral disease classification from intraoral images.
It uses a novel **SAGE (Spatial-Aware Gated Ensemble)** architecture combining three state-of-the-art models
with a learnable gating network, achieving **98.95% accuracy** on a 6-class oral disease dataset.
""")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Model Architecture")
    st.markdown("""
    | Model | Type | Input | Test Acc | Macro F1 |
    |-------|------|-------|----------|----------|
    | DenseNet-121 | CNN | 224×224 | 98.95% | 0.9809 |
    | EfficientNet-B4 | CNN | 380×380 | 98.88% | 0.9858 |
    | Swin-Tiny | Vision Transformer | 224×224 | — | — |
    | **SAGE Ensemble** | Gated | — | — | — |

    **SAGE Gate:** A 2-layer MLP (21 inputs → 32 → 3) that learns per-image weights
    based on individual model probabilities and Spatial Consensus Score (SCS).
    """)

with col2:
    st.markdown("### Dataset")
    st.markdown("""
    | Property | Value |
    |----------|-------|
    | Total images | 11,653 |
    | Classes | 6 |
    | Train | 6,716 |
    | Validation | 1,443 |
    | Test | 1,432 |

    **Classes:** Calculus · Caries · Gingivitis · Hypodontia · Tooth Discoloration · Ulcers
    """)

st.markdown("---")
st.markdown("### XAI Methods")

xcol1, xcol2 = st.columns(2)
with xcol1:
    st.info("""
    **Grad-CAM**

    Gradient-weighted Class Activation Mapping applied to all 3 models independently.
    Highlights image regions most important for each model's decision.
    """)

with xcol2:
    st.info("""
    **Spatial Consensus Score (SCS)** — *Original Contribution*

    Measures agreement between Grad-CAM heatmaps across all 3 models using IoU + cosine similarity.
    Higher SCS = more consistent explanations = more trustworthy prediction.

    `SCS = 0.5 × (IoU + normalized cosine similarity)`
    """)

st.markdown("---")
st.markdown("### Thesis Information")
st.markdown("""
| | |
|---|---|
| **Title** | Explainable Oral Disease Classification Using SAGE Ensemble Deep Learning with Web-Based Clinical Decision Support |
| **Author** | MD. AL AMIN AKASH |
| **Institution** | Daffodil International University |
| **Degree** | MSc in Computer Science and Engineering |
| **Year** | 2026 |
| **Target Journal** | Computers in Biology and Medicine (Q1, Elsevier) |
""")

st.markdown("---")
st.caption("OralXAI — For research purposes only. Not a substitute for professional dental diagnosis.")
