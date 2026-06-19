# OralXAI — Oral Disease Classification System

**MSc Thesis Project | MD. AL AMIN AKASH | Daffodil International University | 2026**

An explainable AI system for oral disease classification using SAGE ensemble
(DenseNet-121 + EfficientNet-B4 + Swin-Tiny) with Grad-CAM XAI and PDF report generation.

---

## Step-by-Step Deployment Guide (A to Z)

### STEP 1 — Prepare Your Models

Copy your 4 trained `.pth` files into the `models/` folder:

```
models/
├── densenet121_best.pth       (27 MB)
├── efficientnetb4_best.pth    (67 MB)
├── swintiny_best.pth          (105 MB)
└── sage_gate.pth              (small)
```

> Make sure the filenames match EXACTLY as above.

---

### STEP 2 — Test Locally First

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the app:
```bash
streamlit run app.py
```

Open browser at `http://localhost:8501`
Upload a test image and verify prediction works.

---

### STEP 3 — Push to GitHub

1. Create a new GitHub repository (public):
   - Go to github.com → New repository
   - Name: `oral-disease-xai`
   - Set to **Public**
   - Click Create

2. Push your project:
```bash
git init
git add .
git commit -m "Initial commit — OralXAI system"
git remote add origin https://github.com/YOUR_USERNAME/oral-disease-xai.git
git push -u origin main
```

> NOTE: GitHub blocks files > 100MB. Your `swintiny_best.pth` is 105MB.
> You need Git LFS for this file. See STEP 3B below.

#### STEP 3B — Git LFS for Large Model Files

```bash
git lfs install
git lfs track "*.pth"
git add .gitattributes
git add models/
git commit -m "Add model files via LFS"
git push
```

If Git LFS gives issues, use Hugging Face Hub instead (STEP 3C).

#### STEP 3C — Alternative: Host Models on Hugging Face Hub

1. Create account at huggingface.co
2. Create new model repo: `your-username/oral-disease-sage`
3. Upload all 4 `.pth` files there
4. In `src/model_loader.py`, replace the load lines with:
```python
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id="your-username/oral-disease-sage", filename="densenet121_best.pth")
model_dense.load_state_dict(torch.load(path, map_location=device))
```
5. Add `huggingface_hub` to `requirements.txt`

---

### STEP 4 — Deploy on Streamlit Cloud (Free)

1. Go to **share.streamlit.io**
2. Click **"New app"**
3. Connect your GitHub account
4. Select your repository: `oral-disease-xai`
5. Set **Main file path**: `app.py`
6. Click **Deploy**

Wait 3–5 minutes for first deployment (installs packages).

Your app will be live at:
`https://your-username-oral-disease-xai-app-XXXXX.streamlit.app`

---

### STEP 5 — Share Your App

Your live URL works on any device, any browser, worldwide — for free.

---

## Project Structure

```
oral-disease-xai/
├── app.py                      # Homepage
├── pages/
│   ├── 1_Predict.py            # Upload + predict + GradCAM
│   ├── 2_Report.py             # PDF report download
│   └── 3_About.py              # Thesis info
├── src/
│   ├── model_loader.py         # Loads all models
│   ├── predictor.py            # SAGE prediction pipeline
│   ├── gradcam.py              # GradCAM implementation
│   ├── scs.py                  # Spatial Consensus Score
│   ├── report.py               # PDF generator
│   └── utils.py                # Preprocessing + helpers
├── models/                     # Your .pth files go here
├── requirements.txt
├── packages.txt
└── .streamlit/config.toml
```

---

## Important Notes

- **RAM**: Streamlit Cloud gives 1GB free RAM. All 3 models use ~300–400MB total (CPU, float32).
- **Sleep**: Free apps sleep after 7 days of no visits. They wake up automatically when visited.
- **Upload limit**: Max 10MB image upload (set in config.toml).
- **Disclaimer**: This system is for research only, not medical diagnosis.

---

## Citation

If you use this system in research, please cite:

> AL AMIN AKASH, MD. (2026). OralXAI: Explainable Oral Disease Classification
> Using SAGE Ensemble Deep Learning. MSc Thesis, Daffodil International University.
