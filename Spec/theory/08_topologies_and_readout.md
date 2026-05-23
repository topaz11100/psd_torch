# Topologies and Readout

## MLP stack

`mlp_stack` is the main configurable topology. A hidden block is:

```text
Linear feedforward + optional spike-only recurrent weight + cell dynamics
```

For recurrent blocks:

```text
I_t = W_ff x_t + R s_{t-1} + b
```

where `s_{t-1}` is the previous output spike from the same layer.

## Fixed topologies

The current fixed topology family is separate from `CellSpec`. Fixed topologies are not MLP cell replacements. Current smoke-supported fixed topologies include:

- GRU
- SSM/S4 alias
- VGG
- ResNet
- SpikeTransformer

Each fixed topology must expose a trace interface compatible with `LayerTraceRecord` and `SignalMapEmitter`.

## Readout

Current readouts:

- `final_if`
- `final_mem`

Readouts may produce logits without output spikes. Missing output spike series must be treated as unavailable, not fabricated.
