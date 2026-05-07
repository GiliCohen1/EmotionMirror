"""
model/model.py

Two model architectures:

1. CustomCNN — built from scratch. 4 conv blocks with batch norm and dropout.
   Fast to train, great for understanding what's happening. ~2.1M params.

2. EfficientNetEmotion — fine-tunes EfficientNet-B0 pretrained on ImageNet.
   Better accuracy (~70% vs ~62%), production model. ~4.0M params.

Design choice: both share the same interface (forward returns logits)
so the training loop works identically for both.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class ConvBlock(nn.Module):
    """
    A single conv block: Conv → BN → ReLU → MaxPool → Dropout.

    We use batch norm before activation (pre-activation style) —
    this stabilizes training especially with small datasets.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CustomCNN(nn.Module):
    """
    Custom CNN for 48×48 grayscale (repeated to 3ch) face images.

    Architecture:
      Input: (B, 3, 48, 48)
      Block 1: 3 → 32 ch,  pool → (B, 32, 24, 24)
      Block 2: 32 → 64 ch, pool → (B, 64, 12, 12)
      Block 3: 64 → 128 ch, pool → (B, 128, 6, 6)
      Block 4: 128 → 256 ch, pool → (B, 256, 3, 3)
      Flatten → FC(2304 → 512) → Dropout → FC(512 → 7)

    Why this architecture:
      - 4 blocks with doubling channels is a well-proven pattern for small images
      - Batch norm in every block makes training stable
      - Two FC layers give capacity without overfitting
      - Spatial dropout (Dropout2d) in conv blocks > standard dropout for conv features
    """

    def __init__(self, num_classes: int = 7, dropout: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3, 32, pool=True, dropout=0.1),
            ConvBlock(32, 64, pool=True, dropout=0.1),
            ConvBlock(64, 128, pool=True, dropout=0.2),
            ConvBlock(128, 256, pool=True, dropout=0.2),
        )

        # After 4 max-pools on 48×48 input: 48 / 2^4 = 3
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Kaiming init for conv layers, zeros for bias."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_feature_layer(self) -> nn.Module:
        """Returns the last conv block — used by GradCAM as the target layer."""
        return self.features[-1].block[-3]  # last conv before the final relu


class EfficientNetEmotion(nn.Module):
    """
    EfficientNet-B0 fine-tuned for emotion classification.

    Strategy:
      - Load ImageNet pretrained weights
      - Replace the classifier head with a new one for 7 classes
      - First N epochs: freeze backbone, only train head (warm-up)
      - Then unfreeze all and train with lower LR for backbone

    Why EfficientNet-B0 specifically:
      - Best accuracy/params trade-off in the EfficientNet family
      - 224×224 input is reasonable for face crops
      - Widely used so easy to find benchmarks to compare against
    """

    def __init__(
        self,
        num_classes: int = 7,
        pretrained: bool = True,
        dropout: float = 0.4,
    ):
        super().__init__()

        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.efficientnet_b0(weights=weights)

        # Keep the backbone (features)
        self.features = base.features
        self.avgpool = base.avgpool

        # Replace the classifier head
        in_features = base.classifier[1].in_features  # 1280
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )

        self._freeze_backbone()

    def _freeze_backbone(self):
        """Freeze backbone for warm-up phase."""
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Call this after warm-up epochs to fine-tune the whole network."""
        for param in self.features.parameters():
            param.requires_grad = True
        print("[Model] Backbone unfrozen — full fine-tuning enabled")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_feature_layer(self) -> nn.Module:
        """Returns last conv block — used by GradCAM."""
        return self.features[-1]


def build_model(cfg: dict) -> nn.Module:
    """
    Factory function: returns the right model from config.

    Usage:
        model = build_model(cfg)
    """
    model_cfg = cfg["model"]
    num_classes = cfg["data"]["num_classes"]
    architecture = model_cfg["architecture"]

    if architecture == "custom_cnn":
        model = CustomCNN(
            num_classes=num_classes,
            dropout=model_cfg["dropout"],
        )
        print(f"[Model] CustomCNN: {sum(p.numel() for p in model.parameters()):,} params")

    elif architecture == "efficientnet_b0":
        model = EfficientNetEmotion(
            num_classes=num_classes,
            pretrained=model_cfg.get("pretrained", True),
            dropout=model_cfg["dropout"],
        )
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Model] EfficientNet-B0: {total:,} total params, {trainable:,} trainable")

    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    return model
