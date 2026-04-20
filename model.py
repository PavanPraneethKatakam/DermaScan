"""
model.py
--------
Two segmentation architectures for binary skin lesion segmentation:

  1. UNet      – Classic U-Net with encoder–decoder and skip connections.
  2. BaselineCNN – Identical encoder/bottleneck/decoder but WITHOUT skip
                   connections, used to isolate the contribution of skip
                   connections to segmentation performance.

Channel progression (encoder → bottleneck → decoder):
    3 → 64 → 128 → 256 → 512 → 1024 → 512 → 256 → 128 → 64 → 1
"""

import torch
import torch.nn as nn
from typing import List


# ---------------------------------------------------------------------------
# Shared building block
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """
    Two consecutive Conv2d → BatchNorm2d → ReLU blocks.

    This is the fundamental repeated unit in both the U-Net encoder/decoder
    and the Baseline CNN.

    Args:
        in_channels  (int): Number of input feature maps.
        out_channels (int): Number of output feature maps.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Build the sequential double-conv block."""
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
        """Apply double-conv block to input tensor."""
        return self.block(x)


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """
    U-Net for binary segmentation.

    Architecture:
        Encoder : 4 DoubleConv blocks (64→128→256→512), each followed by
                  MaxPool2d(2, 2) for spatial downsampling.
        Bottleneck : DoubleConv block (512→1024).
        Decoder : 4 blocks, each using ConvTranspose2d for upsampling, then
                  concatenating the corresponding encoder skip connection,
                  then a DoubleConv block.
        Output  : 1×1 Conv2d followed by Sigmoid activation.

    Args:
        in_channels  (int): Number of input image channels (default 3 for RGB).
        out_channels (int): Number of output segmentation channels (default 1).
        features     (List[int]): Channel sizes for encoder blocks.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: List[int] = [64, 128, 256, 512],
    ) -> None:
        """Initialise U-Net encoder, bottleneck, decoder, and output head with skip connections."""
        super().__init__()

        # ---- Encoder ----
        self.encoders = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        ch = in_channels
        for feat in features:
            self.encoders.append(DoubleConv(ch, feat))
            ch = feat

        # ---- Bottleneck ----
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        ch = features[-1] * 2  # 1024

        # ---- Decoder ----
        self.up_convs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for feat in reversed(features):
            # Upsample: 2× spatial, halve channels
            self.up_convs.append(
                nn.ConvTranspose2d(ch, feat, kernel_size=2, stride=2)
            )
            # After skip-connection concat: feat (upsampled) + feat (skip) → feat
            self.decoders.append(DoubleConv(feat * 2, feat))
            ch = feat

        # ---- Output head ----
        self.output_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through U-Net.

        Args:
            x (Tensor): Input image batch, shape (N, 3, H, W).

        Returns:
            Tensor: Sigmoid-activated segmentation map, shape (N, 1, H, W).
        """
        skip_connections: List[torch.Tensor] = []

        # Encoder path
        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path (skip connections in reverse order)
        skip_connections = skip_connections[::-1]
        for up_conv, decoder, skip in zip(
            self.up_convs, self.decoders, skip_connections
        ):
            x = up_conv(x)

            # Handle potential size mismatch due to odd spatial dimensions
            if x.shape != skip.shape:
                x = nn.functional.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=False
                )

            x = torch.cat([skip, x], dim=1)
            x = decoder(x)

        return self.sigmoid(self.output_conv(x))


# ---------------------------------------------------------------------------
# Baseline CNN (no skip connections)
# ---------------------------------------------------------------------------

class BaselineCNN(nn.Module):
    """
    Baseline segmentation CNN — identical to U-Net in structure but WITHOUT
    skip connections between encoder and decoder.

    This model serves as an ablation baseline to quantify the benefit of skip
    connections.  The decoder receives only the upsampled feature maps from
    the previous decoder stage (or the bottleneck), with no concatenated
    encoder features.

    Args:
        in_channels  (int): Number of input image channels (default 3).
        out_channels (int): Number of output segmentation channels (default 1).
        features     (List[int]): Channel sizes for encoder blocks.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: List[int] = [64, 128, 256, 512],
    ) -> None:
        """Initialise Baseline CNN encoder, bottleneck, decoder, and output head (no skip connections)."""
        super().__init__()

        # ---- Encoder ----
        self.encoders = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        ch = in_channels
        for feat in features:
            self.encoders.append(DoubleConv(ch, feat))
            ch = feat

        # ---- Bottleneck ----
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        ch = features[-1] * 2  # 1024

        # ---- Decoder (NO skip connections → input channels halved) ----
        self.up_convs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for feat in reversed(features):
            self.up_convs.append(
                nn.ConvTranspose2d(ch, feat, kernel_size=2, stride=2)
            )
            # No concatenation → input to DoubleConv is just `feat`
            self.decoders.append(DoubleConv(feat, feat))
            ch = feat

        # ---- Output head ----
        self.output_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Baseline CNN (no skip connections).

        Args:
            x (Tensor): Input image batch, shape (N, 3, H, W).

        Returns:
            Tensor: Sigmoid-activated segmentation map, shape (N, 1, H, W).
        """
        # Encoder path — no skip connections saved
        for encoder in self.encoders:
            x = encoder(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path — only upsampled features, no concat
        for up_conv, decoder in zip(self.up_convs, self.decoders):
            x = up_conv(x)
            x = decoder(x)

        return self.sigmoid(self.output_conv(x))
