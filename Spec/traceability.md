# 명세-구현 대응표

| 명세 항목 | 구현 경로 |
|---|---|
| JSON 설정 로드/검증 | `src/util/config_cli.py`, `config/README.md` |
| seed 정책 | `src/util/random.py`, `src/util/early_seed.py` |
| data_prep | `src/data_prep.py`, `src/data/preprocessing.py`, `src/data/storage.py`, `src/data/registry.py` |
| dataset PSD | `src/dataset_psd.py`, `src/signal/psd_utils.py`, `src/signal/family_spectral_analysis.py` |
| dataset FFT | `src/dataset_fft.py`, `src/signal/psd_utils.py` |
| model training | `src/model_training.py`, `src/model/training.py`, `src/model/snn_builder.py`, `src/readout/readout.py` |
| model PSD | `src/psd_analysis.py` |
| element PSD | `src/element_psd.py`, `src/analysis_matrix_common.py` |
| 2D FFT | `src/2d_fft_analysis.py`, `src/analysis_matrix_common.py` |
| CSV schema | `src/util/csv_schema.py` |
| plotting | `src/plotting.py` |
| bash 실행 | `bash/*.sh` |
| 계약 테스트 | `tests/test_config_cli.py`, `tests/test_data_prep_config.py`, `tests/test_entrypoint_help.py`, `tests/test_entrypoint_light_imports.py`, `tests/test_seed_and_input_exclusion.py` |
