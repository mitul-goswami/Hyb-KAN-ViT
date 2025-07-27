model:
  name: "HybKANViT"
  img_size: 224
  patch_size: 16
  in_chans: 3
  num_classes: 1000
  embed_dim: 768
  depth: 12
  num_heads: 12
  mlp_ratio: 4.0
  qkv_bias: True
  drop_rate: 0.0
  attn_drop_rate: 0.0
  drop_path_rate: 0.1
  hybrid_type: 1  # 1: Wav-KAN encoder + Eff-KAN head, 2: Eff-KAN encoder + Wav-KAN head
  wavelet_type: "dog"  # Options: dog, mexican_hat, morlet

efficient_kan:
  grid_size: 5
  spline_order: 3
  scale_base: 1.0
  scale_spline: 1.0
  grid_range: [-1.5, 1.5]
  grid_eps: 0.02
  num_grids: 8

wavelet_kan:
  num_scales: 6
  initial_scale: 1.0
  central_freq: 5.0  # For Morlet wavelet
  decomposition_levels: 4
  pruning_ratio: 0.4
  scale_base: 1.0
  grid_eps: 0.02
