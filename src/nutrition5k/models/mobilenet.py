from __future__ import annotations

import timm
from torch import Tensor, cat, nn

# ImageNet-1k checkpoints from timm, all pretrained at 224 px.
BACKBONES = {
    "v2": "mobilenetv2_100.ra_in1k",
    "v3": "mobilenetv3_large_100.ra_in1k",
    "v4s": "mobilenetv4_conv_small.e2400_r224_in1k",
    "v4m": "mobilenetv4_conv_medium.e500_r224_in1k",
}
# Target indices follow TARGET_NAMES: mass, calories, fat, carb, protein.
HEAD_GROUPS = {
    "single": ((0, 1, 2, 3, 4),),
    "grouped": ((0,), (1,), (2, 3, 4)),
    "per-target": ((0,), (1,), (2,), (3,), (4,)),
}
HEAD_WIDTH = 256


class NutritionNet(nn.Module):
    """MobileNet backbone, shared FC trunk, and one small head per target group."""

    def __init__(
        self,
        backbone: str,
        heads: str = "single",
        hidden: tuple[int, ...] = (512,),
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            BACKBONES[backbone], pretrained=pretrained, num_classes=0
        ) # num_classes = 0 so the classification heads are not included
        layers: list[nn.Module] = []
        width = self.backbone.head_hidden_size  # pooled output size, after conv_head
        for size in hidden:
            layers += [nn.Linear(width, size), nn.ReLU()]
            width = size
        self.trunk = nn.Sequential(*layers)
        self.groups = HEAD_GROUPS[heads]
        self.heads = nn.ModuleList(
            nn.Sequential(
                nn.Linear(width, HEAD_WIDTH), nn.ReLU(), nn.Linear(HEAD_WIDTH, len(group))
            )
            for group in self.groups
        )
        order = [index for group in self.groups for index in group] # flatten the groups into a single list of target indices [0, 1, 2, 3, 4]
        self.order = [order.index(index) for index in range(len(order))] # save index of each target in the flattened list, so we can reorder the outputs to match the original target order (just in case the groups are not in order)

    def forward(self, images: Tensor) -> Tensor:
        features = self.trunk(self.backbone(images))
        outputs = cat([head(features) for head in self.heads], dim=1)
        return outputs[:, self.order]

    def freeze_backbone(self, unfreeze: int | None) -> None:
        """Train only the last ``unfreeze`` backbone stages and the layers after them.

        ``None`` trains the whole backbone; ``0`` freezes it (feature extraction).
        """
        if unfreeze is None:
            return
        self.backbone.requires_grad_(False)
        if unfreeze == 0:
            return
        for stage in list(self.backbone.blocks)[-unfreeze:]:
            stage.requires_grad_(True)
        names = [name for name, _ in self.backbone.named_children()]
        for name in names[names.index("blocks") + 1 :]:
            getattr(self.backbone, name).requires_grad_(True)

    def train(self, mode: bool = True) -> "NutritionNet":
        super().train(mode)
        # Frozen BatchNorm layers keep their ImageNet running statistics.
        for module in self.backbone.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm) and not any(
                parameter.requires_grad for parameter in module.parameters()
            ):
                module.eval()
        return self
