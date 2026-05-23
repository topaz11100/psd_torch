# Probe Families

## Canonical families

The current canonical probe families are:

- `balanced_global`
- `distributed_set`
- `label_set`
- `label_single`

No other family name is current.

## `balanced_global`

Selects a balanced set across labels using deterministic sampling.

## `distributed_set`

Reflects empirical class distribution. Quotas are deterministic and recorded with class counts and seed metadata.

## `label_set`

Selects samples for one or more target labels.

## `label_single`

Selects one deterministic sample per label. It is a top-level family available to all signal-analysis methods.

`label_single` may exclude samples selected by another canonical family. Exclusion metadata is recorded separately from the family name.

## Manifest metadata

Probe manifests and batches preserve:

- split,
- scope,
- selected sample indices,
- labels,
- class counts and quotas where applicable,
- seed,
- exclusion family and exclusion scope where applicable,
- deterministic probe manifest identifier.

Sample indices are split-level indices, not batch-local offsets.
