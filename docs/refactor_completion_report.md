# Refactor Completion Report

## Status

The analysis-core refactor is ready after documentation cleanup. The current code path is centered on `src/psd_snn`, with canonical examples under `examples/` and current specifications under `Spec/`.

## Completed areas

- Spiking cell path for IF, LIF, and RF within the MLP topology.
- Spike-only recurrent MLP hidden blocks.
- Scenario constraints: `none`, `clip`, `structure`, and `clipstructure`.
- Trace collection and `S,R,T` signal map conversion.
- PSD representatives: mean, median, row-preserving element PSD, and PCA.
- Fixed-reference PCA basis fit/apply, basis tensor artifact, and `pca_basis_id` consistency.
- 2D FFT as an independent analysis method with `spectral_matrix_2d` artifacts.
- Strict spectral distance compatibility for exact/user-bin axes and PCA basis identity.
- Checkpoint-mode `analyze_signal` and `analyze_fft2d` smoke flows.
- Trace tensor chunk writing and manifest outputs.
- Artifact reader and basic plotting CLI.
- Minimum synthetic training/checkpoint/analyze smoke path.
- Canonical examples and runnable config templates.

## Remaining future work

- Real dataset ingestion and preprocessing integration.
- Full launch packaging for multi-run experiment management.
- Publication-style figure rendering.
- Larger-scale training and scheduler utilities.
- Additional paper-model fidelity work beyond current fixed-topology smoke contracts.

## Archive policy

`old/`, `Origin/`, `origin/`, and `references/` remain preserved as reference material. Current docs should not instruct users to run archive code as the active pipeline.

## Merge readiness

The code has no known merge blocker from the latest audit. The remaining blocker was documentation drift: stale current docs described old root launchers and old CSV categories. This patch replaces those stale docs with current theory/implementation specs and removes obsolete current-layer files.
