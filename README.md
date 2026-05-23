# PSD/SNN Analysis Workspace

This repository contains the refactored PSD/SNN analysis workspace. The current implementation lives under `src/psd_snn` and separates model training, checkpoint analysis, dynamics statistics, artifact I/O, and plotting.

## Current canonical package

```text
src/psd_snn/
  cli/
  config/
  models/
  analysis/
  artifacts/
  training/
```

Historical source bundles remain available in archive/reference directories. They are not current runnable entrypoints.

## Current CLI entrypoints

Use the package CLI modules rather than old root-level launchers.

```bash
PYTHONPATH=src python -m psd_snn.cli.train --help
PYTHONPATH=src python -m psd_snn.cli.analyze_signal --help
PYTHONPATH=src python -m psd_snn.cli.analyze_fft2d --help
PYTHONPATH=src python -m psd_snn.cli.analyze_dynamics --help
PYTHONPATH=src python -m psd_snn.cli.plot_artifacts --help
```

The canonical execution examples are under `examples/`.

```bash
source examples/bash/00_env.sh
examples/bash/12_end_to_end_train_analyze_plot.sh
```

## Implemented analysis core

The current refactor implements:

- MLP topology with IF, LIF, and RF cell choices.
- Spike-only SRNN recurrence for recurrent MLP hidden blocks.
- `none`, `clip`, `structure`, and `clipstructure` topology scenarios.
- `final_if` and `final_mem` readouts.
- Trace collection with raw `B,T,*` layout and SignalMap conversion to `S,R,T`.
- PSD representatives: `mean`, `median`, `element_psd`, and `pca`.
- Fixed-reference PCA basis fit/apply with `pca_basis_id` consistency.
- Independent 2D FFT analysis using `spectral_matrix_2d` artifacts.
- Strict spectral distance compatibility for curves, 1D matrices, and 2D matrices.
- Trace tensor chunk artifacts and summary/manifest CSV outputs.
- Minimal synthetic training/checkpoint pipeline for smoke tests.
- Artifact reader and basic plotting CLI.

## Documentation

- `Spec/README.md` is the current specification index.
- `Spec/theory/` contains current theoretical definitions.
- `Spec/implementation/` maps the theory to implemented code paths.
- `examples/README.md` explains runnable examples and config templates.
- `docs/final_audit_report.md` records the most recent audit result.
- `docs/refactor_completion_report.md` summarizes the present completion boundary.

## Archive/reference directories

The following directories are historical or reference material only:

```text
old/
Origin/
origin/
references/
```

Do not treat archive material as current runnable code unless a current `src/psd_snn` module or `examples/` workflow explicitly wraps it.

## Current completion boundary

The refactor is complete for the analysis-core phase: MLP cell dynamics, checkpoint analysis, fixed-reference PCA, 2D FFT, strict artifact identity, fixed topology smoke support, artifact reader/plotting basics, minimum training CLI, and canonical examples.

Future work is intentionally separate:

- real dataset integration,
- paper-ready figure styling,
- launch packaging,
- large-scale training orchestration,
- optional model-family extensions beyond the current smoke contracts.
