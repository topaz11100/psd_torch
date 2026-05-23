# Architecture Contract

## Current package

The current implementation root is:

```text
src/psd_snn/
```

Major modules:

- `config`: dataclass specs and validation.
- `models`: MLP cells, fixed topologies, factory, checkpoints.
- `analysis`: common orchestration, trace, signal-map, PSD/PCA/2D FFT, dynamics, distance.
- `artifacts`: writers, identity, trace writer, reader, plotting helpers.
- `cli`: current user-facing entrypoints.
- `training`: minimum synthetic training smoke path.

## Current CLI modules

- `psd_snn.cli.train`
- `psd_snn.cli.analyze_signal`
- `psd_snn.cli.analyze_fft2d`
- `psd_snn.cli.analyze_dynamics`
- `psd_snn.cli.plot_artifacts`

## Non-current layers

Historical root scripts and old root modules are not authoritative current entrypoints. Current runnable examples live in `examples/bash`.
