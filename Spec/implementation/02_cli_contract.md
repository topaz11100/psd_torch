# CLI Contract

## `train`

Creates a minimum synthetic checkpoint for smoke and compatibility tests. It stores state dictionary and metadata, not a pickled model object.

## `analyze_signal`

Runs PSD representatives from synthetic traces or checkpoint/probe mode. Supports mean, median, element PSD, PCA, fixed-reference PCA, trace saving, and summary/manifest writing.

## `analyze_fft2d`

Runs independent 2D FFT analysis from synthetic traces or checkpoint/probe mode. Produces `spectral_matrix_2d` and row/column axis sidecars.

## `analyze_dynamics`

Reports parameter/state statistics where implemented. Its CLI help must behave like a normal CLI help command.

## `plot_artifacts`

Reads artifact outputs and renders basic plots. Plotting reads artifacts, not training state.

## Config and override policy

Config files provide the main contract. CLI flags may override fields where implemented, but the effective config must still satisfy validation.
