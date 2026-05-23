# Artifacts, Distance, and Manifests

## Artifact types

Current summary artifacts include:

- `spectral_curve`
- `spectral_matrix_1d`
- `spectral_matrix_2d`
- `pca_basis`
- `spectral_distance`
- `trace_manifest`
- `analysis_manifest`

Trace tensors and PCA bases are stored as tensor artifacts. CSV files store summary and manifest rows.

## Trace artifact policy

Trace tensor chunks preserve raw `B,T,*` shape. Chunking occurs over the sample axis only. Time length is recorded in the manifest.

## Distance compatibility

Distance is computed only when artifact identities match. Compatibility includes:

- artifact type,
- spectral axis,
- scale,
- centering and window policy,
- bin policy,
- PCA basis identity where applicable,
- row-axis semantics for 2D matrices,
- row shift policy for 2D matrices.

## Metrics

Only `centered_l2` and `diff_l2` are current metrics.

## Manifest role

Manifests record success and failure status, artifact paths, run/checkpoint/probe metadata, and reason strings for unavailable or unsupported paths.
