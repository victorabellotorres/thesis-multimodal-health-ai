"""Backbones grouped by design family; outputs and notebooks follow the same grouping.

Plain dictionaries without timm, so the data commands work without the training extra.
"""

from .hierarchical import HIERARCHICAL
from .hybrid_mobile import HYBRID_MOBILES
from .mobilenet import MOBILENETS
from .plain_vit import CREATE_KWARGS, PLAIN_VITS

FAMILIES = {
    "mobilenet": MOBILENETS,
    "hybrid_mobile": HYBRID_MOBILES,
    "plain_vit": PLAIN_VITS,
    "hierarchical": HIERARCHICAL,
}
BACKBONES = {key: name for family in FAMILIES.values() for key, name in family.items()}
FAMILY = {key: family for family, backbones in FAMILIES.items() for key in backbones}
# ViTs and Swin fix their input size when created, so these families always run at 224 px.
FIXED_SIZE = {"plain_vit": 224, "hierarchical": 224}

__all__ = ["BACKBONES", "CREATE_KWARGS", "FAMILIES", "FAMILY", "FIXED_SIZE"]
