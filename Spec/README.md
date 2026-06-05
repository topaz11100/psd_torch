# PSD Project Specification

이 디렉터리는 프로젝트가 제공하는 실행 계약, 산출물 계약, 분석 이론을 한 곳에 고정한다. 2026년 개정판의 핵심 목표는 다음 세 가지다.

1. **재현 가능한 파이프라인**: 모든 설정은 `seed=0`, `prep_root=/home/yongokhan/workspace/data/prep_data`, `raw_data=/home/yongokhan/workspace/data/raw_data`(`data_prep` 전용), 산출물 루트 `/home/yongokhan/workspace/result/...`를 기본 계약으로 삼는다.
2. **학습-분석 체크포인트 호환성**: `model_training`이 저장하는 `.pt`는 `psd_analysis`, `2d_fft_analysis`, `element_psd`, `element_fft`, accuracy-eval 도구가 동일한 방식으로 읽을 수 있어야 한다. DDP/`torch.compile` wrapper prefix는 로드 시 정규화한다.
3. **신호의 주파수 구조 해석**: 모델 입력, hidden membrane/spike, output trace를 같은 좌표계의 시간 신호로 변환하고, PSD/PCA/2D FFT를 통해 layer 간 정보 전달 구조를 비교한다.

## 문서 구조

- `implementation/`: CLI, YAML config/manifest, checkpoint JSON-compatible metadata, 산출물 CSV, 실행 스크립트가 지켜야 하는 기계적 계약.
  - `implementation/09_compile_sequence_policy.md`: SNN layer sequence-level compile 및 preallocated buffer 계약.
  - `implementation/10_timestamped_output_policy.md`: 반복 실행 산출물을 실행시각 폴더에 자동 저장하는 경로 계약.
  - `implementation/11_first_spike_regularizer_compile_policy.md`: first-spike readout compile-friendly tensor path와 signal/PSD regularizer eager-GPU 분리 계약.
- `theory/`: PSD 대표곡선, PCA 고정 기준계, 2D FFT, probe family, 뉴런 동역학, readout 및 distance metric의 이론적 의미.
- `traceability.md`: 구현 요구와 파일 위치의 대응표.

## 파이프라인 요약

```text
raw_data
  └─ data_prep.py
       → prep_data/<dataset>/manifest.yaml + prepared tensors

prep_data + model_training.py
  └─ result/.../run_<timestamp>/checkpoints/checkpoint_epoch_*.pt
  └─ result/.../run_<timestamp>/metrics/training_metrics.csv

checkpoint + prep_data
  ├─ psd_analysis.py       → layer/scope별 PSD 대표곡선, 거리, 분산, filter 통계
  ├─ 2d_fft_analysis.py    → neuron × time map의 2D spectral matrix
  ├─ element_psd.py        → neuron별 one-dimensional PSD power matrix
  ├─ element_fft.py        → neuron별 complex FFT component matrix
  └─ dataset_fft.py        → model-independent dataset input frequency baseline
```

## 실행 순서 계약

`bash/model_training.sh`, `bash/model_training_ddp.sh`, `bash/psd_analysis.sh`, `bash/fft2d_analysis.sh`, `bash/element_psd.sh`, `bash/element_fft.sh`, `bash/plotting.sh`는 2차원 config 그룹 계약을 따른다. 같은 그룹의 config들은 병렬로 실행되며, 다음 그룹은 이전 그룹의 모든 프로세스가 성공적으로 끝난 뒤 시작한다. 반면 `data_prep`, `dataset_psd`, `dataset_fft`는 하나의 config 내부 `dataset: [...]` 배열을 받아 데이터셋 종류를 직렬 처리한다.
