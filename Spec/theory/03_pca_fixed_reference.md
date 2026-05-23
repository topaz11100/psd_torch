# PCA Representative and Fixed Reference Basis

## Input and fit matrix

PCA input is a signal map:

```text
S,R,T
```

For basis fitting, observations are sample-time pairs and features are rows:

```text
(S*T, R)
```

If centering is enabled, the feature mean is subtracted before SVD.

## Basis and projection

A basis has components:

```text
W: R,K
mu: R
```

A sample map `X_s: R,T` is projected as:

```text
Y_s = (X_s^T - mu) @ W
```

The projected trajectory is then stored as `S,K,T` and analyzed with the PSD representative path.

## Sign convention

Component signs use `largest_abs_loading_positive`: for each component, the loading with largest absolute magnitude is forced positive.

## Basis modes

### `fit_per_checkpoint`

Each checkpoint fits its own basis. Distance between independently fitted bases is disabled unless the artifacts share an explicit basis identifier.

### `fixed_reference`

A reference checkpoint/split/probe/scope/layer/series fits one basis. Target checkpoints apply the same basis. The resulting artifacts carry the same `pca_basis_id` and can be compared.

## Basis record

A PCA basis record contains:

- basis identifier,
- reference checkpoint metadata,
- split, scope, and probe family,
- layer and series identity,
- component matrix and feature mean,
- explained variance and explained variance ratio,
- row count and component count,
- basis tensor artifact path.

## Strict comparison rule

PCA spectral artifacts are comparable only when both sides have the same non-empty `pca_basis_id`. Missing or different basis identifiers are incompatible.
