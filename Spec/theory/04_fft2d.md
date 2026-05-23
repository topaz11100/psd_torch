# Independent 2D FFT Analysis

## Role

2D FFT is an independent signal-analysis method, not a PSD representative. It analyzes a full row-time matrix for each sample.

Input:

```text
S,R,T
```

For each sample, 2D FFT is computed over row and time axes after optional centering and windowing.

## Exact axis

The row axis uses a full row-frequency grid. The time axis uses a one-sided time-frequency grid. Row-frequency shift policy is recorded in metadata.

The exact output matrix has shape:

```text
F_row,F_time
```

## User-bin axes

User-bin aggregation happens after 2D FFT on raw power, not by pooling the original trace.

Supported axis policies:

- `time_frequency`: bin time-frequency only.
- `row_frequency`: bin row-frequency only.
- `both_frequency_axes`: bin both axes.

Bin reduction is `mean` or `median` on raw power. dB conversion occurs after finalization.

## Row-axis semantics

Row-frequency interpretation depends on row ordering.

- `unordered`: exact 2D FFT may be computed with warning metadata, but row-frequency user-bin analysis is not allowed.
- `group_ordered`: allowed for grouped MLP rows.
- `feature_ordered`: allowed when feature order is meaningful.
- `spatial_flattened`: allowed for flattened image feature maps with flatten-order metadata.
- `channel_ordered`: allowed for channel-only order.
- `pca_component`: allowed for PCA components, with conservative interpretation.

## Artifact

The current artifact type is:

```text
spectral_matrix_2d
```

Row and column sidecar files provide axis metadata.
