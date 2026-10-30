# Hierarchical models: four stages with local token mixing, at the same cost (~28M parameters,
# ~4.5 G MACs, LayerNorm only), so the pair isolates windowed attention against convolution.
HIERARCHICAL = {
    "swin_t": "swin_tiny_patch4_window7_224.ms_in1k",  # shifted-window attention
    "convnext_t": "convnext_tiny.fb_in1k",  # 7x7 depthwise convolutions
}
