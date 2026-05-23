# Traceability

| Requirement area | Current evidence |
|---|---|
| Current package root | `src/psd_snn/` |
| CLI separation | `src/psd_snn/cli/` |
| MLP topology/cell separation | `src/psd_snn/models/mlp/`, `src/psd_snn/models/cells/` |
| Scenario constraints | `src/psd_snn/models/constraints/`, config validation |
| Trace contract | `analysis/trace`, `analysis/signal_map` |
| PSD representatives | `analysis/signal` |
| Fixed-reference PCA | `analysis/signal/pca_basis_store.py` |
| 2D FFT | `analysis/signal/fft2d.py`, `cli/analyze_fft2d.py` |
| Artifact identity and distance | `artifacts/identity.py`, `artifacts/distance_writer.py` |
| Examples | `examples/` |
| Acceptance tests | `tests/` |

This file is a coverage map, not a substitute for tests.
