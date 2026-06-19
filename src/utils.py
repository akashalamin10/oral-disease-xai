import numpy as np
import cv2
from torchvision import transforms

CLASS_NAMES = ["calculus", "caries", "gingivitis", "hypodontia", "toothDiscoloration", "ulcers"]
CLASS_DISPLAY = ["Calculus", "Caries", "Gingivitis", "Hypodontia", "Tooth Discoloration", "Ulcers"]
MODEL_ORDER = ["DenseNet-121", "EfficientNet-B4", "Swin-Tiny"]

IMG_SIZE     = 224
IMG_SIZE_B4  = 380
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

preprocess_224 = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

preprocess_380 = transforms.Compose([
    transforms.Resize((IMG_SIZE_B4, IMG_SIZE_B4)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

def resize_cam_to(cam, size):
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    cam = cv2.resize(cam, (size, size))
    return cam.astype(np.float32)

def overlay_heatmap(base_img_np, cam, alpha=0.45):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap  = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay  = (base_img_np * (1 - alpha) + heatmap / 255.0 * alpha)
    overlay  = np.clip(overlay, 0, 1)
    return overlay
