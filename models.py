import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


class MammoNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # osobne instancje, żeby wyjście trzeciego bloku dało się jednoznacznie
        # przechwycić hookiem przy wyznaczaniu map Grad-CAM
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.attention = SpatialAttention()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.attention(x)
        return self.fc(self.gap(x).flatten(1))


def build_resnet18(num_classes=2, pretrained=True):
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_attention_gradient(model, device="cpu"):
    """Sprawdza, czy gradient dociera do warstwy uwagi.

    Warstwa utworzona wewnątrz forward() zamiast w konstruktorze nie trafia do
    model.parameters() i nigdy nie jest trenowana, a sieć uczy się bez żadnego
    komunikatu o błędzie.
    """
    model.eval()
    dummy = torch.randn(2, 1, 128, 128, device=device)
    model(dummy).sum().backward()
    norm = model.attention.conv.weight.grad.norm().item()
    model.zero_grad()
    return norm
