# 구현 구조

현재 실행 계층은 root `src/*.py` entrypoint다.

| stage | 파일 |
|---|---|
| data_prep | `src/data_prep.py` |
| dataset PSD | `src/dataset_psd.py` |
| dataset FFT | `src/dataset_fft.py` |
| training | `src/model_training.py` |
| model PSD | `src/psd_analysis.py` |
| element PSD | `src/element_psd.py` |
| 2D FFT | `src/2d_fft_analysis.py` |
| plotting | `src/plotting.py` |

지원 모듈은 `src/data`, `src/model`, `src/neurons`, `src/readout`, `src/signal`, `src/stat`, `src/util`로 나눈다.
