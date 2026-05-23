# PSD Representatives

## PSD-first principle

The current analysis follows:

```text
representative(PSD(row signal))
```

rather than computing PSD on a pre-averaged representative signal.

Given a signal map `S,R,T`, the time FFT is applied along `T` for each `(sample,row)` pair. Raw power is computed first. Row/sample reductions, user-bin aggregation, and distance calculations operate on power spectra.

## Exact axis

The exact axis keeps the native one-sided FFT frequency grid. It is compatible only with other exact-axis artifacts with the same frequency identity.

## User-bin axis

User-bin analysis groups exact frequency bins into explicit bin edges. Bin representative values are computed on raw power using `mean` or `median`. Empty bins use the configured empty-bin policy. dB conversion happens after finalization only.

## Representatives

### `mean`

Rows are averaged in PSD space, giving a spectral curve per sample and a run-level curve after sample accumulation.

### `median`

Rows are median-reduced in PSD space. This is distinct from user-bin median aggregation.

### `element_psd`

The row axis is preserved. The output is a 1D spectral matrix with shape `R,F` or `R,Bin`.

### `pca`

PCA transforms `S,R,T` into `S,K,T`. Component trajectories then receive PSD analysis. PCA distance requires the same non-empty basis identifier.

## Distance metrics

Only two distance metrics are current:

- `centered_l2`: subtract a global mean for the object being compared, then compute L2.
- `diff_l2`: compare first differences along the frequency axis.

Exact and user-bin artifacts are never directly compared.
