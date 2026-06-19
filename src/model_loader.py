import torch
import torch.nn as nn
import timm

NUM_CLASSES = 6

class TimmBackboneHead(nn.Module):
    """Matches the exact architecture used in training:
    self.backbone = timm.create_model(name, pretrained=True, num_classes=0, global_pool='avg')
    self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, num_classes))
    """
    def __init__(self, timm_name, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(timm_name, pretrained=False, num_classes=0, global_pool="avg")
        in_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.head(self.backbone(x))


def build_densenet121(num_classes=NUM_CLASSES):
    return TimmBackboneHead("densenet121", num_classes=num_classes)

def build_efficientnetb4(num_classes=NUM_CLASSES):
    return TimmBackboneHead("efficientnet_b4", num_classes=num_classes)

def build_swintiny(num_classes=NUM_CLASSES):
    return TimmBackboneHead("swin_tiny_patch4_window7_224", num_classes=num_classes)

class SAGEGate(nn.Module):
    def __init__(self, in_features=22):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 3),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        return self.net(x)

def load_all_models(model_dir="models", device="cpu", use_hf_hub=True, hf_repo_id="akash4529/oral-disease-sage"):
    if use_hf_hub:
        from huggingface_hub import hf_hub_download
        dense_path = hf_hub_download(repo_id=hf_repo_id, filename="densenet121_best.pth")
        b4_path    = hf_hub_download(repo_id=hf_repo_id, filename="efficientnetb4_best.pth")
        swin_path  = hf_hub_download(repo_id=hf_repo_id, filename="swintiny_best.pth")
        gate_path  = hf_hub_download(repo_id=hf_repo_id, filename="sage_gate.pth")
    else:
        dense_path = f"{model_dir}/densenet121_best.pth"
        b4_path    = f"{model_dir}/efficientnetb4_best.pth"
        swin_path  = f"{model_dir}/swintiny_best.pth"
        gate_path  = f"{model_dir}/sage_gate.pth"

    model_dense = build_densenet121()
    model_dense.load_state_dict(torch.load(dense_path, map_location=device))
    model_dense.to(device).eval()

    model_b4 = build_efficientnetb4()
    model_b4.load_state_dict(torch.load(b4_path, map_location=device))
    model_b4.to(device).eval()

    model_swin = build_swintiny()
    model_swin.load_state_dict(torch.load(swin_path, map_location=device))
    model_swin.to(device).eval()

    gate = SAGEGate(in_features=22)
    gate.load_state_dict(torch.load(gate_path, map_location=device))
    gate.to(device).eval()

    return model_dense, model_b4, model_swin, gate
