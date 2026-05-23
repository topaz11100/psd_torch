# Trace and Signal Map Semantics

## Raw trace

Raw trace tensors use external layout:

```text
B,T,*
```

where `B` is sample/probe batch, `T` is the full simulation-time axis, and `*` is the layer-specific feature shape.

The time axis is never chunked for analysis. Trace artifacts may chunk only along the sample axis.

## Signal map

Signal analysis consumes a standardized map:

```text
S,R,T
```

where `S` is sample count, `R` is row/neuron/channel/component count, and `T` is time.

Conversions:

- `B,T,F -> S,R,T` as `B,F,T`.
- `B,T,C,H,W -> S,R,T` as `B,C*H*W,T` with flatten-order metadata.

## Official trace series

Current series names are explicit. The analysis path avoids ambiguous membrane-only labels.

- `input_current`
- `membrane_pre`
- `decision`
- `spike`
- `membrane_post`
- `rf_real_pre`
- `rf_imag_pre`
- `rf_real_post`
- `rf_imag_post`
- `output_membrane_pre`
- `output_decision`
- `logits`

## Metadata

Trace and signal-map records preserve:

- run identifier,
- checkpoint epoch and optional checkpoint identifier,
- split, scope, probe family, optional label,
- sample indices and labels where available,
- layer index and layer name,
- signal kind and series,
- scenario and constraint hash when available,
- row-axis semantics and warnings.

## Unavailable series

If a readout does not emit output spikes, the analysis must not fabricate zero spikes. Missing series are skipped or recorded with status/reason metadata in manifests.
