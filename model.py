"""
model.py  —  Self-contained UNet definition for HF Spaces.
Copied from the training project (model.py).
"""

import torch
import torch.nn as nn
from typing import List


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: List[int] = [64, 128, 256, 512],
    ) -> None:
        super().__init__()

        self.encoders = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        ch = in_channels
        for feat in features:
            self.encoders.append(DoubleConv(ch, feat))
            ch = feat

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        ch = features[-1] * 2

        self.up_convs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for feat in reversed(features):
            self.up_convs.append(nn.ConvTranspose2d(ch, feat, kernel_size=2, stride=2))
            self.decoders.append(DoubleConv(feat * 2, feat))
            ch = feat

        self.output_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections: List[torch.Tensor] = []

        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]
        for up_conv, decoder, skip in zip(self.up_convs, self.decoders, skip_connections):
            x = up_conv(x)
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = decoder(x)

        return self.sigmoid(self.output_conv(x))
