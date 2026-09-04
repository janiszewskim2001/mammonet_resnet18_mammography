import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """Grad-CAM dla dowolnej warstwy konwolucyjnej.

    Dla MammoNet warstwą docelową jest pool3 (mapa 16x16), dla ResNet-18 layer4
    (mapa 7x7). Różnica rozdzielczości przekłada się bezpośrednio na ostrość mapy
    i musi być uwzględniona przy porównywaniu obu modeli.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove(self):
        for h in self.handles:
            h.remove()

    def __call__(self, x, class_idx=None):
        assert x.dim() == 4 and x.size(0) == 1

        self.model.eval()
        self.model.zero_grad()

        out = self.model(x)
        probs = F.softmax(out, dim=1)
        if class_idx is None:
            class_idx = int(out.argmax(1).item())
        out[0, class_idx].backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("hooki nie przechwyciły aktywacji ani gradientów")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = cam.squeeze().cpu().numpy()

        peak = cam.max()
        cam = cam / peak if peak > 1e-8 else np.zeros_like(cam)

        return cam, float(probs[0, class_idx])


def upsample(cam, size):
    """Interpolacja dwuliniowa do rozmiaru obrazu.

    scipy.ndimage.zoom z order=3 przy skalowaniu z mapy 7x7 tworzy sztucznie gładkie,
    okrągłe artefakty niezwiązane z treścią obrazu.
    """
    t = torch.from_numpy(cam).float().unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t.squeeze().numpy()


def target_layer(model, arch):
    return model.pool3 if arch == "mammonet" else model.layer4
