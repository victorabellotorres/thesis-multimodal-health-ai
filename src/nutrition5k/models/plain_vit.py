# Plain ViTs: global self-attention from the first block, no hierarchy. Every one is ImageNet-1k except
# DINOv2 (self-supervised on LVD-142M); deit_s is its supervised control at the same ViT-S size.
PLAIN_VITS = {
    "deit_t": "deit_tiny_patch16_224.fb_in1k",  # MobileNetV3 size
    "deit_s": "deit_small_patch16_224.fb_in1k",
    "dinov2_s": "vit_small_patch14_reg4_dinov2.lvd142m",  # ViT-S/14 with 4 register tokens
    # Large references (~300M, 14x dinov2_s), not finalists: is a much larger backbone worth it? DINOv2-L isolates
    # the size; DINOv3-L (LVD-1689M, RoPE, 4 registers, DINOv3 License) adds the newer pretraining.
    "dinov2_l": "vit_large_patch14_reg4_dinov2.lvd142m",
    "dinov3_l": "vit_large_patch16_dinov3.lvd1689m",
}
# DINOv2 is released at 518 px and DINOv3 at 256 px; timm adapts their position embeddings to 224 px.
CREATE_KWARGS = {name: {"img_size": 224} for name in ("dinov2_s", "dinov2_l", "dinov3_l")}
