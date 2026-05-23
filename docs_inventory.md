# Entrypoint / Dependency Inventory

## Legacy Entrypoints
- `src/model_training.py`
- `src/psd_analysis.py`
- `src/2d_fft_analysis.py`
- `src/element_psd.py`
- `src/dataset_psd.py`
- `src/plotting.py`

## New Refactor Root
- `src/psd_snn/cli/*`
- `src/psd_snn/config/specs.py`
- `src/psd_snn/analysis/*`
- `src/psd_snn/models/readout/factory.py`

## Deleted Legacy Feature
- `src/reinterpretation/` (removed)
