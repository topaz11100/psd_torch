# 명세-구현 대응표

| 명세 항목 | 구현 경로 |
|---|---|
| YAML 설정 로드/검증 | `src/util/config_cli.py`, `config/*.yaml`, `config/README.md` |
| 중첩 DI 설정 로드 | `src/DI.py`, `config/DI.yaml`, `bash/DI.sh` |
| seed 정책 | `src/util/random.py`, `src/util/early_seed.py` |
| data_prep | `src/data_prep.py`, `src/data/preprocessing.py`, `src/data/storage.py`, `src/data/registry.py` |
| dataset PSD | `src/dataset_psd.py`, `src/signal/psd_utils.py`, `src/signal/family_spectral_analysis.py` |
| dataset FFT | `src/dataset_fft.py`, `src/signal/psd_utils.py` |
| model training | `src/model_training.py`, `src/model/training.py`, `src/model/snn_builder.py`, `src/readout/readout.py` |
| vanilla constraint 적용 범위 | `src/model/snn_builder.py`, `src/model/constraints.py` |
| `my_*` 뉴런 구현 | `src/neurons/my_DH_SNN_neuron.py`, `src/neurons/my_D_RF_neuron.py`, `src/neurons/my_R_DH_SNN_neuron.py` |
| `my_*` branch/filter property | `src/signal/filter_property.py`, `src/psd_analysis.py` |
| model PSD/filter 분석 | `src/psd_analysis.py` |
| element PSD | `src/element_psd.py`, `src/analysis_matrix_common.py` |
| element FFT | `src/element_fft.py`, `src/analysis_matrix_common.py` |
| 2D FFT | `src/2d_fft_analysis.py`, `src/analysis_matrix_common.py` |
| CSV schema | `src/util/csv_schema.py` |
| plotting | `src/plotting.py` |
| bash 실행 | `bash/*.sh` |
| 이론 문서 | `spec/Theory/` |
| 구현 계약 문서 | `spec/Implementation/` |
| 논문 요약 markdown | `spec/Theory/Literature/` |
| PDF 원문 | `paper/` |
