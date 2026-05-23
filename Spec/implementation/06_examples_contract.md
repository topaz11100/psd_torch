# Examples Contract

## Canonical examples

User-facing workflows live under:

```text
examples/bash/
examples/configs/commented/
examples/configs/runnable/
```

## Bash scripts

Example shell scripts must:

- use `python -m psd_snn.cli.*`,
- set `PYTHONPATH=src`,
- use smoke-friendly defaults,
- avoid external dataset download,
- document arguments and accepted values.

## Config templates

Commented YAML templates are explanatory. Runnable JSON files are parser/smoke inputs.

## Archive boundary

Historical root launchers are not current examples. Archive/reference directories are read-only context.
