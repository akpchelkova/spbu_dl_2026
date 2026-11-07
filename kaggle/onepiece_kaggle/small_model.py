"""СОБСТВЕННАЯ компактная CNN на depthwise separable convolutions.

Модель создаётся с нуля, без timm и без предобученных весов.
Архитектура: Conv stem -> inverted residual blocks -> pooling -> head.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvNormAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, groups=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2,
                      groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )


class SqueezeExcite(nn.Module):
    def __init__(self, ch, reduction=8):
        super().__init__()
        hidden = max(8, ch // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.network = nn.Sequential(
            nn.Conv2d(ch, hidden, 1), nn.SiLU(inplace=True),
            nn.Conv2d(hidden, ch, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.network(self.pool(x))


class InvertedBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, expand=3):
        super().__init__()
        hidden = in_ch * expand
        self.body = nn.Sequential(
            ConvNormAct(in_ch, hidden, kernel=1),
            ConvNormAct(hidden, hidden, kernel=3, stride=stride, groups=hidden),
            SqueezeExcite(hidden),
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = (in_ch == out_ch and stride == 1)

    def forward(self, x):
        output = self.body(x)
        return x + output if self.skip else output


class SmallAnimeCNN(nn.Module):
    def __init__(self, num_classes=18):
        super().__init__()
        self.features = nn.Sequential(
            ConvNormAct(3, 24, stride=2),
            InvertedBlock(24, 32, stride=2, expand=2),
            InvertedBlock(32, 32, stride=1, expand=2),
            InvertedBlock(32, 48, stride=2, expand=3),
            InvertedBlock(48, 48, stride=1, expand=3),
            InvertedBlock(48, 80, stride=2, expand=3),
            InvertedBlock(80, 80, stride=1, expand=3),
            InvertedBlock(80, 112, stride=2, expand=3),
            InvertedBlock(112, 112, stride=1, expand=3),
            InvertedBlock(112, 160, stride=1, expand=3),
            ConvNormAct(160, 256, kernel=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(nn.Dropout(p=0.25), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))

    def get_classifier(self):
        return self.classifier


if __name__ == '__main__':
    model = SmallAnimeCNN()
    params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {params:,}; fp32 weights ~ {params * 4 / (1024**2):.2f} MiB')
    with torch.no_grad():
        print('Output shape:', tuple(model(torch.randn(2, 3, 160, 160)).shape))
