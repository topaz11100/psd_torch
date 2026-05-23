# Artifact Writer, Reader, and Plotting Contract

## Writers

Current writers create summary and manifest CSV outputs plus tensor artifacts.

CSV outputs include:

- `analysis_manifest.csv`
- `trace_manifest.csv`
- `spectral_curve.csv`
- `spectral_matrix_1d.csv`
- `spectral_matrix_2d.csv`
- `spectral_matrix_2d_row_axis.csv`
- `spectral_matrix_2d_column_axis.csv`
- `pca_basis.csv`
- `spectral_distance.csv`

## Trace tensors

Trace chunks are tensor files. Manifest rows describe layout, shape, dtype, compression, sample range, and time length.

## PCA basis tensors

PCA basis metadata is summarized in `pca_basis.csv`. The basis tensor is stored separately with mean, components, explained variance, explained variance ratio, and basis identifier.

## Reader and plotting

ArtifactReader loads summary files, sidecars, PCA basis tensors, and trace chunks. Plotting reads artifacts through the reader and does not depend on raw model execution.

## Disallowed output style

Current writers do not store raw trace values in CSV files.
