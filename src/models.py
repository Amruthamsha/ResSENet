"""
Architecture definitions for the ResSENet FER2013 benchmark.

Core contribution: SE embedded *inside* the residual branch (Eq. 1: X_hat =
X + SE(F(X))), contrasted against a post-hoc SE placement (SE(X + F(X))) as
an iso-parametric control condition, plus standard baselines (BaselineCNN,
ResNet-18/50, VGG-13, EfficientNet-B0).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .config import DEFAULT_CONFIG


class SEModule(nn.Module):
    """Squeeze-and-Excitation block (Eq. 2: s = sigma(W2 . delta(W1 . GAP(X))))."""

    def __init__(self, channels: int, r: int = DEFAULT_CONFIG.se_ratio):
        super().__init__()
        mid = max(channels // r, 4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, mid, bias=False)
        self.fc2 = nn.Linear(mid, channels, bias=False)

    def forward(self, x):
        b, c, _, _ = x.shape
        s = self.gap(x).view(b, c)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s)).view(b, c, 1, 1)
        return x * s  # channel-wise scaling


class ResSEBlock(nn.Module):
    """SE embedded *inside* the residual branch (Eq. 1: X_hat = X + SE(F(X)))."""

    def __init__(self, in_ch, out_ch, stride=1, use_residual=True, use_se=True):
        super().__init__()
        self.use_residual = use_residual
        self.use_se = use_se

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        if use_se:
            self.se = SEModule(out_ch)

        self.shortcut = nn.Sequential()
        if use_residual and (stride != 1 or in_ch != out_ch):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.use_se:
            out = self.se(out)  # SE acts on residual branch F(x)
        if self.use_residual:
            out = out + self.shortcut(x)  # X_hat = X + SE(F(X))
        return F.relu(out)


class ResSENet(nn.Module):
    """
    4-stage network. Spatial: 48->24->12->6, Channels: 64->128->256->512.
    ~4.2M params. Supports ablation via use_residual / use_se flags.
    """

    def __init__(self, num_classes=7, use_residual=True, use_se=True, dropout=DEFAULT_CONFIG.dropout):
        super().__init__()
        self.use_residual = use_residual
        self.use_se = use_se

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )

        self.stage1 = self._make_stage(64, 64, stride=1, n=2)
        self.stage2 = self._make_stage(64, 128, stride=2, n=2)
        self.stage3 = self._make_stage(128, 256, stride=2, n=2)
        self.stage4 = self._make_stage(256, 512, stride=2, n=2)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512, 128)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def _make_stage(self, in_ch, out_ch, stride, n=2):
        layers = [ResSEBlock(in_ch, out_ch, stride=stride, use_residual=self.use_residual, use_se=self.use_se)]
        for _ in range(n - 1):
            layers.append(ResSEBlock(out_ch, out_ch, stride=1, use_residual=self.use_residual, use_se=self.use_se))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x).flatten(1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)

    def get_last_conv_layer(self):
        """Returns last conv layer for GradCAM."""
        return self.stage4[-1].conv2


# ── SE-ResNet Post-Hoc (reviewer control condition) ──────────────────────
# Post-hoc: X_hat = SE(X + F(X)) — SE acts AFTER residual addition, on the
# combined/mixed representation, unlike ResSEBlock above where SE acts only
# on F(X) before the identity addition. Iso-parametric with ResSENet, so it
# isolates the effect of *where* SE is placed.


class SEPostHocBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEModule(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)  # X + F(X) — identity added FIRST
        out = self.se(out)  # SE(X + F(X)) — post-hoc, on the mixed tensor
        return F.relu(out)


class SEResNetPostHoc(nn.Module):
    """Iso-parametric control: identical backbone/param count to ResSENet,
    only difference is SE placement (post-hoc vs. pre-addition)."""

    def __init__(self, num_classes=7, dropout=DEFAULT_CONFIG.dropout):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.stage1 = self._make_stage(64, 64, stride=1, n=2)
        self.stage2 = self._make_stage(64, 128, stride=2, n=2)
        self.stage3 = self._make_stage(128, 256, stride=2, n=2)
        self.stage4 = self._make_stage(256, 512, stride=2, n=2)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512, 128)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def _make_stage(self, in_ch, out_ch, stride, n=2):
        layers = [SEPostHocBlock(in_ch, out_ch, stride=stride)]
        for _ in range(n - 1):
            layers.append(SEPostHocBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x).flatten(1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)

    def get_last_conv_layer(self):
        return self.stage4[-1].conv2


# ── Baselines ──────────────────────────────────────────────────────────


class BaselineCNN(nn.Module):
    """Shallow 3-block CNN baseline, ~1M params."""

    def __init__(self, num_classes=7, dropout=DEFAULT_CONFIG.dropout):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def get_resnet18(pretrained=False, num_classes=7):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    m = models.resnet18(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def get_vggnet13(pretrained=False, num_classes=7):
    weights = models.VGG13_BN_Weights.DEFAULT if pretrained else None
    m = models.vgg13_bn(weights=weights)
    m.classifier[6] = nn.Linear(4096, num_classes)
    return m


def get_resnet50(pretrained=True, num_classes=7):
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    m = models.resnet50(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def get_efficientnet_b0(pretrained=True, num_classes=7):
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    m = models.efficientnet_b0(weights=weights)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    return m


def get_ressenet_pretrained(num_classes=7):
    """ResSENet+ variant: initialise stem from ResNet-18 ImageNet weights, rest random."""
    m = ResSENet(num_classes=num_classes, use_residual=True, use_se=True)
    rn18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    with torch.no_grad():
        w = rn18.conv1.weight  # shape [64,3,7,7]
        w_resized = F.interpolate(w.permute(0, 1, 2, 3).float(), size=(3, 3), mode="bilinear", align_corners=False)
        m.stem[0].weight.copy_(w_resized)
    return m
