# PSD Project Specification

이 디렉터리는 프로젝트의 실행 계약, 산출물 계약, 분석 이론을 한 곳에 고정한다. 현재 구조는 `psd`를 단일 research repo로 유지하되, 대용량 데이터·체크포인트·실행 산출물은 repo 밖 artifact로 관리하는 것을 전제로 한다.

## 핵심 원칙

1. **재현 가능한 파이프라인**: raw data → prepared bundle → training checkpoint → PSD/FFT/filter analysis 산출물의 lineage를 config와 checkpoint metadata로 보존한다.
2. **학습-분석 체크포인트 호환성**: `model_training`이 저장하는 `.pt`는 `psd_analysis`, `fft2d_analysis`, `element_psd`, `element_fft`가 동일한 방식으로 읽을 수 있어야 한다. DDP/`torch.compile` wrapper prefix는 로드 시 정규화한다.
3. **신호의 주파수 구조 해석**: 모델 입력, hidden membrane/spike, output trace를 같은 좌표계의 시간 신호로 변환하고, PSD/PCA/2D FFT/filter-property 통계로 layer 간 정보 전달 구조를 비교한다.
4. **vanilla constraint와 proposed branch 구조 분리**: `clip/structure/clipstructure`는 vanilla `if/lif/rf`에만 적용하고, `my_*` 계열은 branch-count와 branch/filter 통계로 분석한다.

## 문서 구조

- `Implementation/`: CLI, YAML config/manifest, checkpoint metadata, 산출물 CSV, 실행 스크립트가 지켜야 하는 기계적 계약.
  - `Implementation/01_config_contract.md`: 중첩 clean config와 parser flattening 계약.
  - `Implementation/02_cli_contract.md`: CLI/YAML 병합과 bash launcher 계약.
  - `Implementation/09_compile_sequence_policy.md`: SNN layer sequence-level compile 및 preallocated buffer 계약.
  - `Implementation/10_timestamped_output_policy.md`: 반복 실행 산출물을 실행시각 폴더에 자동 저장하는 경로 계약.
  - `Implementation/11_first_spike_regularizer_compile_policy.md`: first-spike readout compile-friendly tensor path와 signal/PSD regularizer eager-GPU 분리 계약.
  - `Implementation/12_direct_discrete_rf_policy.md`: vanilla RF direct-discrete pole radius policy와 checkpoint metadata 계약.
- `Theory/`: PSD 대표곡선, PCA 고정 기준계, 2D FFT, probe family, 뉴런 동역학, readout 및 distance metric의 이론적 의미.
  - `Theory/Proposed/`: `multi_base`에서 이식한 `my_*` 뉴런, branch multiplicity, direct discrete pole contract, filter-property 분석의 상세 이론.
  - `Theory/Literature/`: 논문별 이론 요약 markdown. PDF 원문은 루트 `paper/` 아래에 둔다.
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
  ├─ psd_analysis.py       → layer/scope별 PSD 대표곡선, 거리, 분산, filter/branch 통계
  ├─ 2d_fft_analysis.py    → neuron × time map의 2D spectral matrix
  ├─ element_psd.py        → neuron별 one-dimensional PSD power matrix
  ├─ element_fft.py        → neuron별 complex FFT component matrix
  └─ dataset_fft.py        → model-independent dataset input frequency baseline
```

## 실행 순서 계약

`bash/` 아래 launcher는 stage별 단일 실행 wrapper다. 첫 번째 인자로 config 경로를 받고, 생략하면 `config/<stage>.yaml`을 사용한다. 현재 repo에는 병렬 scenario group runner를 두지 않는다. sweep이 필요하면 clean template를 복사하거나 외부 generator가 resolved config를 만든 뒤 각 launcher를 반복 호출한다.

`data_prep`, `dataset_psd`, `dataset_fft`는 하나의 config 내부 `dataset: [...]` 배열을 받아 데이터셋 종류를 직렬 처리할 수 있다. prepared bundle을 쓰는 분석 stage는 checkpoint와 manifest를 기준으로 독립 실행한다.
