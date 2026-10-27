# timm checkpoints, all used at 224 px. Every one is ImageNet-1k except DINOv2 (self-supervised on
# LVD-142M), which isolates the effect of pretraining against DeiT at the same ViT-S size.
TRANSFORMERS = {
    "deit_t": "deit_tiny_patch16_224.fb_in1k",  # plain ViT, MobileNetV3 size
    # FastViT-SA12: convolutional token mixing (RepMixer) in stages 1-3, self-attention in stage 4.
    "fastvit_sa12": "fastvit_sa12.apple_in1k",
    "dinov2_s": "vit_small_patch14_reg4_dinov2.lvd142m",  # ViT-S/14 with 4 register tokens
    "swin_t": "swin_tiny_patch4_window7_224.ms_in1k",  # hierarchical, shifted windows
}
# DINOv2 is released at 518 px; timm interpolates its position embeddings to 224 px.
CREATE_KWARGS = {"dinov2_s": {"img_size": 224}}
# Every transformer runs at this size: the ViTs and Swin fix their input size when created.
FIXED_SIZE = 224
