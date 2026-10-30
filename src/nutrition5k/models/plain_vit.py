# Plain ViTs: global self-attention from the first block, no hierarchy. Every one is ImageNet-1k except
# DINOv2 (self-supervised on LVD-142M); deit_s is its supervised control at the same ViT-S size.
PLAIN_VITS = {
    "deit_t": "deit_tiny_patch16_224.fb_in1k",  # MobileNetV3 size
    "deit_s": "deit_small_patch16_224.fb_in1k",
    "dinov2_s": "vit_small_patch14_reg4_dinov2.lvd142m",  # ViT-S/14 with 4 register tokens
}
# DINOv2 is released at 518 px; timm interpolates its position embeddings to 224 px.
CREATE_KWARGS = {"dinov2_s": {"img_size": 224}}
