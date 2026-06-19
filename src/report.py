import io
import os
import numpy as np
from datetime import datetime
from fpdf import FPDF
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

from src.utils import overlay_heatmap, IMG_SIZE

class OralXAIReport(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(26, 26, 46)
        self.cell(0, 10, "OralXAI - Clinical Prediction Report", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 6, f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}", align="C", new_x="LMARGIN", new_y="NEXT")
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"MD. AL AMIN AKASH | Daffodil International University | Page {self.page_no()}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(26, 26, 46)
        self.set_fill_color(235, 240, 255)
        self.cell(0, 8, f"  {title}", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def key_value(self, key, value, color=(0, 0, 0)):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(80, 80, 80)
        self.cell(60, 7, key)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*color)
        self.cell(0, 7, str(value), new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)

def _pil_to_bytes(pil_img, size=(80, 80)):
    pil_img = pil_img.resize(size)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return buf

def _cam_to_pil(base_img, cam):
    overlay = overlay_heatmap(base_img, cam, alpha=0.5)
    arr = (overlay * 255).astype(np.uint8)
    return Image.fromarray(arr)

def generate_report(result: dict) -> bytes:
    pdf = OralXAIReport()
    pdf.add_page()

    pdf.section_title("1. Prediction Summary")
    conf = result["confidence"]
    conf_color = (0, 150, 0) if conf >= 80 else (200, 100, 0) if conf >= 60 else (200, 0, 0)
    pdf.key_value("Predicted Class:", result["predicted_display"], color=conf_color)
    pdf.key_value("SAGE Confidence:", f"{conf:.2f}%", color=conf_color)
    pdf.key_value("SCS (mean):", f"{result['scs']['mean']:.4f}")
    pdf.key_value("SCS DenseNet-B4:", f"{result['scs']['dense_b4']:.4f}")
    pdf.key_value("SCS DenseNet-Swin:", f"{result['scs']['dense_swin']:.4f}")
    pdf.key_value("SCS B4-Swin:", f"{result['scs']['b4_swin']:.4f}")
    pdf.ln(4)

    pdf.section_title("2. Gate Weights (SAGE)")
    gw = result["gate_weights"]
    pdf.key_value("DenseNet-121 weight:", f"{gw['DenseNet-121']:.4f}")
    pdf.key_value("EfficientNet-B4 weight:", f"{gw['EfficientNet-B4']:.4f}")
    pdf.key_value("Swin-Tiny weight:", f"{gw['Swin-Tiny']:.4f}")
    pdf.ln(4)

    pdf.section_title("3. Per-class Probabilities (SAGE Ensemble)")
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(220, 230, 255)
    pdf.cell(70, 7, "Class", border=1, fill=True)
    pdf.cell(40, 7, "SAGE %", border=1, fill=True)
    pdf.cell(40, 7, "DenseNet %", border=1, fill=True)
    pdf.cell(40, 7, "EfficientNet %", border=1, fill=True, new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 9)
    for cls in result["sage_probs"]:
        is_pred = cls == result["predicted_display"]
        if is_pred:
            pdf.set_fill_color(220, 255, 220)
        else:
            pdf.set_fill_color(255, 255, 255)
        pdf.cell(70, 6, cls, border=1, fill=True)
        pdf.cell(40, 6, f"{result['sage_probs'][cls]*100:.2f}%", border=1, fill=True)
        pdf.cell(40, 6, f"{result['probs_dense'][cls]*100:.2f}%", border=1, fill=True)
        pdf.cell(40, 6, f"{result['probs_b4'][cls]*100:.2f}%", border=1, fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    pdf.section_title("4. Input Image & Grad-CAM Heatmaps")

    orig_buf = _pil_to_bytes(result["pil_image"], size=(100, 100))
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(0, 6, "Original Image:", new_x="LMARGIN", new_y="NEXT")
    pdf.image(orig_buf, x=10, w=60)
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(0, 6, "Grad-CAM Heatmaps:", new_x="LMARGIN", new_y="NEXT")

    x_positions = [10, 75, 140]
    base = result["base_img"]
    cams = result["cams"]
    model_names = ["DenseNet-121", "EfficientNet-B4", "Swin-Tiny"]
    y_start = pdf.get_y()

    for i, name in enumerate(model_names):
        cam_pil = _cam_to_pil(base, cams[name])
        buf = _pil_to_bytes(cam_pil, size=(90, 90))
        pdf.image(buf, x=x_positions[i], y=y_start, w=58)

    pdf.set_y(y_start + 62)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(80, 80, 80)
    for i, name in enumerate(model_names):
        w_val = result["gate_weights"][name]
        pdf.set_x(x_positions[i])
        pdf.cell(58, 5, f"{name} (w={w_val:.3f})", align="C")
    pdf.ln(8)
    pdf.set_text_color(0, 0, 0)

    pdf.section_title("5. System Information")
    pdf.key_value("Models used:", "DenseNet-121, EfficientNet-B4, Swin-Tiny")
    pdf.key_value("Ensemble method:", "SAGE (Spatial-Aware Gated Ensemble)")
    pdf.key_value("XAI method:", "Grad-CAM + Spatial Consensus Score (SCS)")
    pdf.key_value("Dataset:", "Oral Disease Dataset - 6 classes, 11,653 images")
    pdf.key_value("Test Accuracy:", "98.95% (DenseNet), 98.88% (EfficientNet)")
    pdf.key_value("Macro AUC:", "0.9997")
    pdf.ln(4)

    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 5,
        "DISCLAIMER: This report is generated by an AI system for research purposes only. "
        "It should NOT be used as a substitute for professional dental or medical diagnosis. "
        "Always consult a qualified dentist or oral health professional.")

    buf_out = io.BytesIO()
    pdf.output(buf_out)
    buf_out.seek(0)
    return buf_out.read()
