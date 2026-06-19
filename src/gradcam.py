import torch
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer, reshape_transform=None):
        self.model = model
        self.target_layer = target_layer
        self.reshape_transform = reshape_transform
        self.gradients = None
        self.activations = None
        self._hooks = []
        self._register()

    def _register(self):
        def fwd_hook(_, __, output):
            self.activations = output.detach()
        def bwd_hook(_, __, grad_output):
            self.gradients = grad_output[0].detach()
        self._hooks.append(self.target_layer.register_forward_hook(fwd_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(bwd_hook))

    def __call__(self, x, target_class):
        self.model.zero_grad()
        logits = self.model(x)
        loss = logits[0, target_class[0]]
        loss.backward()

        grads = self.gradients
        acts  = self.activations

        if self.reshape_transform:
            grads = self.reshape_transform(grads)
            acts  = self.reshape_transform(acts)

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1).squeeze(0)
        cam = torch.relu(cam).cpu().numpy()
        return cam[np.newaxis]

    def remove(self):
        for h in self._hooks:
            h.remove()

def get_target_layer_dense(model):
    return model.backbone.features.denseblock4.denselayer16.conv2

def get_target_layer_b4(model):
    return model.backbone.blocks[-1][-1].conv_pwl

def get_target_layer_swin(model):
    return model.backbone.layers[-1].blocks[-1].norm2

def swin_reshape_transform(tensor):
    B, N, C = tensor.shape
    H = W = int(N ** 0.5)
    return tensor.reshape(B, H, W, C).permute(0, 3, 1, 2)
