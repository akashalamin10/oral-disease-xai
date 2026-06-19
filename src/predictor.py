import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from PIL import Image

from src.utils import (
    preprocess_224, preprocess_380, resize_cam_to,
    CLASS_NAMES, CLASS_DISPLAY, MODEL_ORDER, IMG_SIZE
)
from src.gradcam import (
    GradCAM, get_target_layer_dense,
    get_target_layer_b4, get_target_layer_swin,
    swin_reshape_transform
)
from src.scs import compute_scs

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(pil_image, model_dense, model_b4, model_swin, gate):
    img_pil = pil_image.convert("RGB")

    img_224 = preprocess_224(img_pil).unsqueeze(0).to(DEVICE)
    img_380 = preprocess_380(img_pil).unsqueeze(0).to(DEVICE)

    model_dense.eval(); model_b4.eval(); model_swin.eval()

    with torch.no_grad():
        probs_d = torch.softmax(model_dense(img_224), dim=1).cpu().numpy()[0]
        probs_b = torch.softmax(model_b4(img_380),    dim=1).cpu().numpy()[0]
        probs_s = torch.softmax(model_swin(img_224),  dim=1).cpu().numpy()[0]

    mean_p     = (torch.tensor(probs_d) + torch.tensor(probs_b) + torch.tensor(probs_s)) / 3.0
    tgt_class  = torch.tensor([mean_p.argmax().item()], dtype=torch.long).to(DEVICE)

    cd = GradCAM(model_dense, get_target_layer_dense(model_dense))
    cb = GradCAM(model_b4,    get_target_layer_b4(model_b4))
    cs = GradCAM(model_swin,  get_target_layer_swin(model_swin),
                 reshape_transform=swin_reshape_transform)

    img_224.requires_grad_(False)
    img_380.requires_grad_(False)

    cam_d  = resize_cam_to(cd(img_224, tgt_class)[0], IMG_SIZE)
    cam_b  = resize_cam_to(cb(img_380, tgt_class)[0], IMG_SIZE)
    cam_s  = resize_cam_to(cs(img_224, tgt_class)[0], IMG_SIZE)

    cd.remove(); cb.remove(); cs.remove()

    scs = compute_scs(cam_d, cam_b, cam_s)

    gate_feats = np.concatenate([
        probs_d.reshape(1, -1),
        probs_b.reshape(1, -1),
        probs_s.reshape(1, -1),
        np.array([[scs["dense_b4"], scs["dense_swin"], scs["b4_swin"]]]),
        np.array([[scs["mean"]]])
    ], axis=1).astype(np.float32)

    gate.eval()
    with torch.no_grad():
        feat_tensor  = torch.tensor(gate_feats, device=DEVICE)
        gate_weights = gate(feat_tensor).cpu().numpy()[0]
        sage_probs   = (gate_weights[0] * probs_d +
                        gate_weights[1] * probs_b +
                        gate_weights[2] * probs_s)

    pred_idx   = int(sage_probs.argmax())
    base_img   = np.array(img_pil.resize((IMG_SIZE, IMG_SIZE))) / 255.0

    return {
        "predicted_class":  CLASS_NAMES[pred_idx],
        "predicted_display": CLASS_DISPLAY[pred_idx],
        "confidence":        float(sage_probs.max() * 100),
        "sage_probs":        dict(zip(CLASS_DISPLAY, sage_probs.tolist())),
        "probs_dense":       dict(zip(CLASS_DISPLAY, probs_d.tolist())),
        "probs_b4":          dict(zip(CLASS_DISPLAY, probs_b.tolist())),
        "probs_swin":        dict(zip(CLASS_DISPLAY, probs_s.tolist())),
        "gate_weights":      dict(zip(MODEL_ORDER, gate_weights.tolist())),
        "scs":               scs,
        "cams":              {"DenseNet-121": cam_d, "EfficientNet-B4": cam_b, "Swin-Tiny": cam_s},
        "base_img":          base_img,
        "pil_image":         img_pil
    }
