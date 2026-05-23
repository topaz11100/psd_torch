# Config Contract

## Core dataclasses

The current config contract is defined in `src/psd_snn/config/specs.py`.

Key objects:

- `ExperimentConfig`
- `ModelSpec`
- `TopologySpec`
- `CellSpec`
- `ReadoutSpec`
- `ConstraintSpec`
- `ProbeSpec`
- `SignalAnalysisSpec`
- `PSDAnalysisSpec`
- `FFT2DAnalysisSpec`
- `TraceSaveSpec`

## Topology and cell validation

- `mlp_stack` uses `CellSpec`.
- Fixed topologies are selected by topology kind and are not cell replacements.
- `if`, `lif`, and `rf` are cell kinds.
- Scenario constraints apply only to MLP hidden layers in the current phase.

## Spectral validation

- Spectral axis is `exact` or `userbin`.
- User-bin reducer is `mean` or `median`.
- Distance metrics are `centered_l2` and `diff_l2`.
- PCA fixed-reference mode requires reference metadata.
- 2D FFT row-axis binning requires meaningful row-axis semantics.

## Trace validation

`uint8` trace storage is valid only for spike-like series. Non-spike traces use floating dtypes.
