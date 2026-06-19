import streamlit as st

st.set_page_config(
    page_title="OralXAI — Oral Disease Classification",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0; }
    .sub-title  { font-size: 1.1rem; color: #555; margin-top: 0.2rem; }
    .badge      { display:inline-block; padding:3px 10px; border-radius:20px;
                  font-size:0.78rem; font-weight:600; margin:2px; }
    .badge-blue { background:#dbeafe; color:#1e40af; }
    .badge-green{ background:#dcfce7; color:#166534; }
    .badge-purple{background:#ede9fe; color:#5b21b6; }
    .card       { background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px;
                  padding:1.2rem 1.5rem; margin-bottom:1rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">OralXAI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Explainable AI System for Oral Disease Classification</p>', unsafe_allow_html=True)
st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="card">
    <h4>SAGE Ensemble</h4>
    <p>Spatial-Aware Gated Ensemble of DenseNet-121, EfficientNet-B4, and Swin-Tiny for robust prediction.</p>
    <span class="badge badge-blue">DenseNet-121</span>
    <span class="badge badge-blue">EfficientNet-B4</span>
    <span class="badge badge-purple">Swin-Tiny</span>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="card">
    <h4>XAI Methods</h4>
    <p>Grad-CAM heatmaps for all 3 models with Spatial Consensus Score (SCS) — original novelty metric.</p>
    <span class="badge badge-green">Grad-CAM</span>
    <span class="badge badge-green">SCS Score</span>
    <span class="badge badge-green">Gate Weights</span>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="card">
    <h4>PDF Report</h4>
    <p>Generate a downloadable clinical-style report with prediction, heatmaps, and XAI analysis.</p>
    <span class="badge badge-purple">Auto Report</span>
    <span class="badge badge-purple">Download PDF</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("### 6 Oral Disease Classes")
classes = {
    "Calculus": "Tartar buildup on teeth surface",
    "Caries": "Tooth decay / dental cavities",
    "Gingivitis": "Gum inflammation and disease",
    "Hypodontia": "Missing / absent teeth condition",
    "Tooth Discoloration": "Staining or color change of teeth",
    "Ulcers": "Oral sores and mucosal ulcers"
}
cols = st.columns(3)
for i, (cls, desc) in enumerate(classes.items()):
    with cols[i % 3]:
        st.info(f"**{cls}**\n\n{desc}")

st.markdown("---")
st.markdown("### How to Use")
st.markdown("""
1. Click **Predict** in the sidebar
2. Upload an intraoral image (JPG/PNG)
3. View SAGE ensemble prediction + Grad-CAM heatmaps
4. Click **Report** to download a PDF report
""")

st.markdown("---")
st.caption("OralXAI | MSc Thesis — MD. AL AMIN AKASH | Daffodil International University | 2026")
