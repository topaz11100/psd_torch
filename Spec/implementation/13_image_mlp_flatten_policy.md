# Image datasets as flattened MLP sequences

## Purpose

All image-format datasets must be usable through two mutually explicit runtime paths:

1. **Frame path** for CNN and frame-based backbones.
2. **Flattened MLP path** for dense SNN/MLP-style families, including vanilla IF/LIF/RF, DH-SNN/D-RF, `my_*`, SpikeGRU, and non-frame analysis tools.

The project therefore treats the image-to-MLP conversion as a stage contract rather than an implicit model-side reshape.

## Covered datasets

The contract applies to the official image-format prepared datasets:

```text
mnist
cifar-10
cifar-100
n-mnist
cifar10-dvs
dvs128-gesture
```

Sequential image derivatives such as `s-mnist`, `ps-mnist`, and `s-cifar10` are already prepared as temporal feature sequences and do not need a separate frame-to-MLP conversion.

## Data-preprocessing contract

`data_prep` records an `image_mlp_flatten_contract` block in each image-format manifest. Static image datasets must provide a stored `model_input_flatten` split payload with sample shape `(T,F)` in addition to the frame-shaped CNN payload. Event-frame datasets may reconstruct the flattened view from their stored `(T,C,H,W)` event-frame tensor, but the manifest must still declare the flattened training view.

Required manifest fields:

```yaml
image_mlp_flatten_contract:
  enabled: true
  scope: all_non_frame_mlp_training_and_checkpoint_signal_analysis
  training_view_name_for_mlp: model_input_flatten  # or sequence_input for reconstructed event-frame views
  rank_contract: per_sample_(T,F)
```

## Training contract

`model_training` resolves the structured `ModelSpec` first, then calls `select_training_view_for_model(...)` and `validate_image_mlp_flatten_contract(...)` before constructing loaders or models.

Frame families:

```text
cnn_lif
cnn_rf
cnn
spikformer
```

receive a rank-4 per-sample tensor `(T,C,H,W)`.

All other families receive a rank-2 per-sample tensor `(T,F)`, where `T == bundle.sequence_length` and `F == bundle.input_dim`. Failure to provide that view is a preprocessing error, not a model-forward fallback.

## Signal-analysis contract

Checkpoint-based signal analysis uses the same structured model fields and the same view resolver as training. Dense/MLP analysis therefore consumes the exact flattened view that training consumed; CNN/Spikformer analysis consumes the frame view. This prevents a checkpoint trained on `(T,F)` from being analyzed against `(T,C,H,W)` by accident.

## Rationale

The RF and D-RF paths are now specified as discrete-time filters: the model learns a per-step pole and its analysis is defined over the sequence index. For image data, the MLP experiment should therefore receive an explicit time-major sequence `(T,F)` rather than silently inventing a reshape inside the neuron or model builder. The frame path remains available only when the selected architecture is explicitly frame-based.
