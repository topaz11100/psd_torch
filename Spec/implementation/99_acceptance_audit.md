# Acceptance Audit

## Test commands

```bash
PYTHONPATH=src pytest -q tests
find examples/bash -name "*.sh" -print0 | xargs -0 -I{} bash -n {}
```

CLI help smoke:

```bash
PYTHONPATH=src python -m psd_snn.cli.train --help
PYTHONPATH=src python -m psd_snn.cli.analyze_signal --help
PYTHONPATH=src python -m psd_snn.cli.analyze_fft2d --help
PYTHONPATH=src python -m psd_snn.cli.analyze_dynamics --help
PYTHONPATH=src python -m psd_snn.cli.plot_artifacts --help
```

## Required checks

- No removed experiment package is restored.
- Current examples use `psd_snn.cli` entrypoints.
- Trace artifacts preserve time axis and chunk only sample axis.
- PSD and 2D FFT use finalize-only dB conversion.
- Exact and user-bin artifacts are not directly compared.
- PCA distance requires the same non-empty `pca_basis_id`.
- 2D FFT uses `spectral_matrix_2d` artifacts.
- Artifact outputs include run/checkpoint/split/scope/probe metadata where applicable.
- README, examples, and Spec describe current code paths.

## Merge readiness

A merge is ready when tests pass in the available environment, optional backend skips are explained, docs match current code paths, and no current docs direct users to removed launchers.
