from __future__ import annotations

import timm
from torch import Tensor, cat, nn

from .families import BACKBONES, CREATE_KWARGS

# Target indices follow TARGET_NAMES: mass, calories, fat, carb, protein.
HEAD_GROUPS = {
    "single": ((0, 1, 2, 3, 4),),
    "grouped": ((0,), (1,), (2, 3, 4)),
    "per-target": ((0,), (1,), (2,), (3,), (4,)),
}
HEAD_WIDTH = 256
# Pooling of the token sequence for the plain ViTs (the only backbones with a CLS token); None keeps
# timm's default (CLS for DeiT and DINOv2, global average pooling for every other backbone).
POOLINGS = ("cls", "mean", "cls+mean")


class RegressionHead(nn.Module):
    """Shared FC trunk, then one head per target group; outputs follow TARGET_NAMES."""

    def __init__(
        self,
        features: int,
        heads: str = "single",
        hidden: tuple[int, ...] = (512,),
        head_width: int = HEAD_WIDTH,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = features
        for size in hidden:
            layers += [nn.Linear(width, size), nn.ReLU()]
            width = size
        self.trunk = nn.Sequential(*layers)
        self.groups = HEAD_GROUPS[heads]
        # A head without its own hidden layer is linear, so grouping it changes nothing.
        self.heads = nn.ModuleList(
            nn.Sequential(nn.Linear(width, head_width), nn.ReLU(), nn.Linear(head_width, len(group)))
            if head_width
            else nn.Linear(width, len(group))
            for group in self.groups
        )
        order = [index for group in self.groups for index in group] # flatten the groups into a single list of target indices [0, 1, 2, 3, 4]
        self.order = [order.index(index) for index in range(len(order))] # save index of each target in the flattened list, so we can reorder the outputs to match the original target order (just in case the groups are not in order)

    def forward(self, features: Tensor) -> Tensor:
        features = self.trunk(features)
        outputs = cat([head(features) for head in self.heads], dim=1)
        return outputs[:, self.order]


class NutritionNet(nn.Module):
    """timm backbone, then a ``RegressionHead``."""

    def __init__(
        self,
        backbone: str,
        heads: str = "single",
        hidden: tuple[int, ...] = (512,),
        pretrained: bool = True,
        head_width: int = HEAD_WIDTH,
        pooling: str | None = None,
    ) -> None:
        super().__init__()
        # Dropout and stochastic depth are off for every backbone: timm's defaults differ
        # (swin_t has drop path 0.1, the rest 0), and the comparison needs one recipe.
        self.backbone = timm.create_model(
            BACKBONES[backbone],
            pretrained=pretrained,
            num_classes=0,  # without the ImageNet classifier
            drop_rate=0.0,
            drop_path_rate=0.0,
            **CREATE_KWARGS.get(backbone, {}),
        )
        if pooling and not hasattr(self.backbone, "num_prefix_tokens"):
            raise ValueError(f"{backbone} has no CLS token; pooling applies to plain ViTs only")
        self.pooling = pooling
        features = self.backbone.head_hidden_size  # pooled output size, after conv_head
        self.head = RegressionHead(features * (2 if pooling == "cls+mean" else 1), heads, hidden, head_width)

    def features(self, images: Tensor) -> Tensor:
        """The pooled backbone output that the head receives."""
        if not self.pooling:
            return self.backbone(images)
        return pool_tokens(self.backbone.forward_features(images), self.backbone.num_prefix_tokens, self.pooling)

    def forward(self, images: Tensor) -> Tensor:
        return self.head(self.features(images))

    def freeze_backbone(self, unfreeze: int | None) -> None:
        """Train only the last ``unfreeze`` backbone stages and the layers after them.

        ``None`` trains the whole backbone; ``0`` freezes it (feature extraction). The stages
        are ``blocks`` in MobileNets and ViTs (one per transformer block), ``stages`` in
        FastViT and ConvNeXt, and ``layers`` in Swin.
        """
        if unfreeze is None:
            return
        self.backbone.requires_grad_(False)
        if unfreeze == 0:
            return
        container = next(
            name for name in ("blocks", "stages", "layers") if hasattr(self.backbone, name)
        )
        for stage in list(getattr(self.backbone, container))[-unfreeze:]:
            stage.requires_grad_(True)
        names = [name for name, _ in self.backbone.named_children()]
        for name in names[names.index(container) + 1 :]:
            getattr(self.backbone, name).requires_grad_(True)

    def train(self, mode: bool = True) -> "NutritionNet":
        super().train(mode)
        # Every backbone BatchNorm keeps its ImageNet running statistics, so training and
        # evaluation normalise identically; its scale and shift still train when unfrozen.
        # Updating the statistics made v4m's validation unstable. Models with only LayerNorm
        # (ViTs, Swin) have no running statistics, so this changes nothing for them.
        for module in self.backbone.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self


def pool_tokens(tokens: Tensor, prefix: int, pooling: str) -> Tensor:
    """Pool ViT tokens (after the final norm): CLS, the mean of the patch tokens, or both."""
    cls, mean = tokens[:, 0], tokens[:, prefix:].mean(dim=1)  # registers (DINOv2) are left out
    return {"cls": cls, "mean": mean, "cls+mean": cat((cls, mean), dim=1)}[pooling]
