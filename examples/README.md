# Examples

This directory contains the canonical user-facing examples for the refactored PSD/SNN pipeline.

## Layout

```text
examples/
  bash/                 # runnable shell workflows
  configs/commented/    # heavily commented YAML templates for reading
  configs/runnable/     # JSON configs intended for parser/smoke use
```

Archive directories such as `old/`, `Origin/`, `origin/`, and `references/` are historical material. They are not the current example layer.

## Quick start

```bash
source examples/bash/00_env.sh
examples/bash/01_train_synthetic_mlp.sh
examples/bash/03_analyze_signal_mean_median.sh
examples/bash/07_analyze_fft2d_exact.sh
examples/bash/11_plot_artifacts.sh
```

The most compact smoke flow is:

```bash
examples/bash/12_end_to_end_train_analyze_plot.sh
```

## Functional flows

- `01_train_synthetic_mlp.sh`: one-run synthetic MLP checkpoint creation.
- `02_train_synthetic_fixed_topology.sh`: supported fixed-topology training smoke when enabled by the CLI.
- `03_analyze_signal_mean_median.sh`: PSD curve representatives.
- `04_analyze_signal_element_psd.sh`: row-preserving 1D spectral matrix analysis.
- `05_analyze_signal_pca_fit_per_checkpoint.sh`: per-checkpoint PCA basis fitting.
- `06_analyze_signal_pca_fixed_reference.sh`: reference checkpoint PCA basis fit and target checkpoint apply.
- `07_analyze_fft2d_exact.sh`: exact 2D FFT spectral matrix.
- `08_analyze_fft2d_userbin_time_frequency.sh`: time-frequency user bins.
- `09_analyze_fft2d_userbin_row_frequency.sh`: row-frequency user bins for ordered row axes.
- `10_analyze_fft2d_userbin_both_frequency_axes.sh`: two-axis binning for ordered row axes.
- `11_plot_artifacts.sh`: artifact reader based plotting.
- `12_end_to_end_train_analyze_plot.sh`: train, analyze, read, and plot smoke chain.

## Current CLI modules

```bash
python -m psd_snn.cli.train
python -m psd_snn.cli.analyze_signal
python -m psd_snn.cli.analyze_fft2d
python -m psd_snn.cli.analyze_dynamics
python -m psd_snn.cli.plot_artifacts
```

## Artifact outputs

Common outputs include:

- `analysis_manifest.csv`
- `trace_manifest.csv`
- `spectral_curve.csv`
- `spectral_matrix_1d.csv`
- `spectral_matrix_2d.csv`
- `spectral_matrix_2d_row_axis.csv`
- `spectral_matrix_2d_column_axis.csv`
- `pca_basis.csv`
- `spectral_distance.csv`
- tensor chunks under an artifact trace directory
- PCA basis tensor artifacts

Trace values are stored as tensor chunks, not as value-dump CSV files.

## Intentionally out of scope for examples

These examples are smoke and usage templates, not a full experiment launcher. Real dataset ingestion, large-scale run scheduling, and publication figure styling belong to later phases.
