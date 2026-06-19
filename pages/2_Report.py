import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from datetime import datetime

from src.report import generate_report

st.set_page_config(page_title="Report — OralXAI", page_icon="📄", layout="wide")

st.title("Download PDF Report")
st.markdown("Generate and download a clinical-style PDF report of your last prediction.")
st.markdown("---")

if "last_result" not in st.session_state:
    st.warning("No prediction found. Please go to **Predict** first and upload an image.")
    st.stop()

result = st.session_state["last_result"]
pred   = result["predicted_display"]
conf   = result["confidence"]
scs    = result["scs"]["mean"]

st.success(f"Last prediction: **{pred}** ({conf:.2f}% confidence | SCS={scs:.4f})")

col1, col2, col3 = st.columns(3)
col1.metric("Predicted Class", pred)
col2.metric("Confidence", f"{conf:.2f}%")
col3.metric("SCS Score", f"{scs:.4f}")

st.markdown("---")
st.markdown("### Report Contents")
st.markdown("""
The PDF report includes:
- Prediction summary (class, confidence, SCS)
- Gate weights for all 3 models
- Per-class probability table (SAGE + individual models)
- Original image + 3 Grad-CAM heatmaps
- System and dataset information
- Disclaimer
""")

if st.button("Generate PDF Report", type="primary"):
    with st.spinner("Generating report..."):
        pdf_bytes = generate_report(result)

    fname = f"OralXAI_Report_{pred.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    st.download_button(
        label="Download PDF Report",
        data=pdf_bytes,
        file_name=fname,
        mime="application/pdf"
    )
    st.success(f"Report ready: {fname}")
