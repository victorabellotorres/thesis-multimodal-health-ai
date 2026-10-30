# Mobile hybrids: convolutions in the early stages, attention in the last ones, designed for phones.
# Each one has an attention-free twin, so the pairs isolate the effect of attention. ImageNet-1k weights.
HYBRID_MOBILES = {
    # FastViT: RepMixer (reparameterisable depthwise convolutions) in stages 1-3 in both; stage 4 is
    # RepMixer in S12 and self-attention in SA12. Pretrained at 256 px, used at 224 like the rest.
    "fastvit_s12": "fastvit_s12.apple_in1k",
    "fastvit_sa12": "fastvit_sa12.apple_in1k",
    # MobileNetV4-Hybrid-M: V4-Conv-M (v4m, same e500_r224 recipe) with attention blocks in the last stages.
    "v4hm": "mobilenetv4_hybrid_medium.e500_r224_in1k",
}
