# PSD/SNN Refactor Workspace

새 구조는 실행 단위를 분리한다.

- train
- analyze_psd / analyze_element_psd / analyze_fft2d / analyze_pca_psd
- analyze_dynamics

설정은 dataclass 기반(`src/psd_snn/config/specs.py`)으로 정의하며, probe/spectral/readout 정책을 명시한다.
