# PSD/SNN Current Specification

This directory is the current specification for the refactored PSD/SNN analysis workspace. It is split into theory documents and implementation contracts.

## Theory documents

| File | Purpose |
|---|---|
| `theory/00_overview.md` | End-to-end conceptual model |
| `theory/01_signal_trace_and_signal_map.md` | Trace and signal-map semantics |
| `theory/02_psd_representatives.md` | PSD-first representatives |
| `theory/03_pca_fixed_reference.md` | PCA representative and fixed-reference basis |
| `theory/04_fft2d.md` | Independent 2D FFT analysis |
| `theory/05_probe_families.md` | Probe-family definitions |
| `theory/06_spiking_cells_if_lif_rf.md` | IF/LIF/RF cell dynamics |
| `theory/07_constraints_clip_structure_clipstructure.md` | Scenario constraints |
| `theory/08_topologies_and_readout.md` | MLP, fixed topologies, and readout |
| `theory/09_dynamics_statistics.md` | Dynamics statistics |
| `theory/10_artifacts_distance_and_manifests.md` | Artifacts, manifests, and distance |

## Implementation documents

| File | Purpose |
|---|---|
| `implementation/00_architecture.md` | Package architecture |
| `implementation/01_config_contract.md` | Config dataclass contract |
| `implementation/02_cli_contract.md` | CLI behavior |
| `implementation/03_model_factory_and_checkpoints.md` | Model factory and checkpoint restore |
| `implementation/04_trace_signal_analysis_pipeline.md` | Trace-to-analysis pipeline |
| `implementation/05_artifact_writer_reader_plotting.md` | Artifact I/O and plotting |
| `implementation/06_examples_contract.md` | Examples and config-template contract |
| `implementation/99_acceptance_audit.md` | Acceptance checks |

## Status vocabulary

- `implemented`: current code path exists and is tested or smoke-tested.
- `intentionally unsupported`: prohibited by current design or deferred by explicit policy.
- `future work`: desirable but outside the current phase.

## Archive boundary

Historical material belongs in `old/`, `Origin/`, `origin/`, or `references/`. Current specs must describe `src/psd_snn` and `examples/` as the active code and usage layers.
