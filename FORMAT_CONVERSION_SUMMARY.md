# Format Conversion Summary

This package applies the requested project-wide serialization policy for the PSD codebase.

## Final artifact policy

| Artifact class | Final format |
|---|---|
| User-authored config files | YAML (`.yaml` / `.yml`) |
| Scenario configs | YAML |
| Manifests | YAML |
| Model checkpoint metadata | JSON-compatible payload inside `.pt` checkpoint |
| Training metrics | CSV |
| Dataset PSD/FFT numeric outputs | CSV |
| Model PSD/FFT analysis numeric outputs | CSV |
| DI summary/resolved-config/result tables | CSV |
| Plot image outputs | PNG, with YAML plotting manifest |

## Main implementation notes

- `src/util/config.py` is the shared YAML/CSV serialization helper module.
- `src/util/config_cli.py` now accepts YAML config files only.
- Prepared data bundles write and resolve `manifest.yaml`.
- Analysis, dataset, plotting, DI, smoke-matrix, and checkpoint-accuracy manifests are YAML.
- Numeric and tabular experiment outputs are emitted through CSV helpers.
- `src/model_training.py` keeps checkpoint metadata JSON-compatible and stores an explicit `checkpoint_metadata_json` field inside the `.pt` checkpoint payload.
- All project config files under `config/` were converted from `.json` to `.yaml`.

## Validation performed

- Confirmed actual `.json` file count in the converted tree: 0.
- Confirmed `config/` YAML file count: 1041.
- Confirmed all `config/**/*.yaml` files parse successfully with PyYAML.
- Bash wrapper syntax check passed for `bash/*.sh`.
- Python bytecode compilation passed for `src`, `tests`, `tools`, `reference/SNNs`, and `repair_checkpoint.py`.
- Targeted pytest set passed: 48 passed, 1 skipped.
  - `tests/test_config_cli.py`
  - `tests/test_smoke_matrix_tool_static.py`
  - `tests/test_model_training_ddp_static.py`
  - `tests/test_bash_nohup_logs.py`
  - `tests/test_data_prep_config.py`
  - `tests/test_dataset_matrix_and_image_flatten.py`
  - `tests/test_model_training_smoke.py`
  - `tests/test_psd_analysis_pca_outputs.py`
  - `tests/test_signal_window_config.py`
  - `tests/test_data_prep_multi_dataset.py`
  - `tests/test_entrypoint_help.py`
  - `tests/test_entrypoint_light_imports.py`
  - `tests/test_model_training_constraints_cli.py`
  - `tests/test_model_training_pca_cli.py`
  - `tests/test_psd_analysis_pca_cli.py`
  - `tests/test_psd_analysis_pca.py`
  - `tests/test_element_fft.py`
  - `tests/test_model_training_ddp_smoke.py`

The full pytest suite was attempted but exceeded this environment's execution timeout; no failure was observed before the timeout. The targeted tests above completed successfully.
