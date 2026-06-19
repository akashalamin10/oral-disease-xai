import numpy as np

def cam_iou(cam_a, cam_b, threshold=0.5):
    a = (cam_a >= threshold).astype(np.float32)
    b = (cam_b >= threshold).astype(np.float32)
    inter = (a * b).sum()
    union = np.clip(a + b, 0, 1).sum()
    return float(inter / union) if union > 0 else 0.0

def cam_cosine(cam_a, cam_b):
    a = cam_a.flatten()
    b = cam_b.flatten()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0

def compute_scs(cam_d, cam_b4, cam_s):
    db = 0.5 * (cam_iou(cam_d, cam_b4) + (cam_cosine(cam_d, cam_b4) + 1) / 2)
    ds = 0.5 * (cam_iou(cam_d, cam_s)  + (cam_cosine(cam_d, cam_s)  + 1) / 2)
    bs = 0.5 * (cam_iou(cam_b4, cam_s) + (cam_cosine(cam_b4, cam_s) + 1) / 2)
    scs_mean = (db + ds + bs) / 3.0
    return {
        "dense_b4": round(db, 4),
        "dense_swin": round(ds, 4),
        "b4_swin": round(bs, 4),
        "mean": round(scs_mean, 4)
    }
