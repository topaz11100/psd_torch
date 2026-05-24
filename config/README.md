# JSON 설정 상세 설명

모든 실행 설정은 `config/*.json`에 둔다. JSON은 주석을 지원하지 않으므로 각 인수의 의미와 허용 범위는 이 문서에서 설명한다.

## 공통 규칙

- 설정 파일 확장자는 `.json`만 허용한다.
- 최상위 값은 객체여야 한다.
- 각 파일은 stage key 아래에 실제 설정을 둔다. 예: `{"dataset_psd": {...}}`.
- CLI 인자가 JSON 값보다 우선한다.
- JSON에 알 수 없는 key가 있으면 오류가 난다.
- 경로 placeholder `/ABS/PATH/TO/...`는 실제 절대경로로 바꿔야 한다.
- `true`, `false`, `null`, 숫자, 문자열, 배열은 JSON 표준 형식으로 작성한다.

## 공통 인수

| 인수 | 역할 | 자료형 | 필수 | 허용값/범위 | 예시 | 사용 stage |
|---|---|---:|---:|---|---|---|
| `dataset` | 데이터셋 토큰 | string | 예 | 등록된 dataset token | `mnist`, `shd` | data_prep, dataset_psd, dataset_fft, model_training, 분석 stage |
| `prep_root` | prepared bundle 루트 | string | 예 | 절대경로 권장 | `/data/prepared` | data_prep 제외 대부분 stage |
| `output_root` | 산출물 루트 | string | 예 | 절대경로 권장 | `/runs/exp1/psd` | dataset/model 분석, 학습 metadata |
| `seed` | 난수 시드 | integer | 예/권장 | 0 이상 정수 권장 | `0` | 모든 계산 stage |
| `num_workers` | DataLoader worker 수 | integer | 아니오 | 0 이상 | `0`, `4` | DataLoader 사용 stage |
| `gpu_index` | CUDA 장치 index | integer | 예 | 0 이상 | `0` | dataset/model 분석, 학습 |

## `data_prep.json`

| 인수 | 역할 | 자료형 | 필수 | 허용값/범위 | 예시 |
|---|---|---:|---:|---|---|
| `dataset` | 전처리할 데이터셋 | string | 예 | `src/data/specs.py` 등록 토큰 | `mnist` |
| `raw_data_root` | 원본 데이터 루트 | string | 예 | 절대경로 권장 | `/data/raw` |
| `prep_root` | prepared 출력 루트 | string | 예 | 절대경로 권장 | `/data/prepared` |
| `seed` | 전처리 난수 시드 | integer | 예 | 0 이상 정수 권장 | `0` |
| `force_overwrite` | 기존 prepared bundle 삭제 후 재작성 | boolean/string | 아니오 | `true`, `false`, `"true"`, `"false"` | `false` |
| `download` | torchvision 계열 데이터 자동 다운로드 | boolean/string | 아니오 | bool 또는 bool 문자열 | `false` |
| `max_samples` | split별 최대 샘플 수 | integer/null | 아니오 | `null` 또는 양의 정수 | `null`, `1000` |
| `prep_profile` | 전처리 축/시간 프로필 | string/null | 아니오 | `project_standard`, `need_high_cifar10_dvs_t16`, `drf_shd_t250`, `dh_snn_shd_t1000` | `project_standard` |
| `deap_label_axis` | DEAP 라벨 축 | string | DEAP에서 사용 | `valence`, `arousal` | `valence` |
| `deap_num_classes` | DEAP 라벨 bin 수 | integer | DEAP에서 사용 | `2`, `3` | `3` |
| `shd_dt_ms` | SHD event bin 폭(ms) | number | SHD에서 사용 | 양수 | `1.0` |
| `shd_max_time` | SHD 최대 시간(초) | number | SHD에서 사용 | 양수 | `1.2` |
| `ssc_dt_ms` | SSC event bin 폭(ms) | number | SSC에서 사용 | 양수 | `1.0` |
| `ssc_max_time` | SSC 최대 시간(초) | number | SSC에서 사용 | 양수 | `1.0` |

`prep_profile`이 `project_standard`가 아니면 출력이 `<prep_root>/<prep_profile>/<dataset>` 아래에 생성될 수 있다. 이후 stage의 `prep_root`는 manifest가 실제로 존재하는 루트로 맞춘다.

## `dataset_psd.json`

| 인수 | 역할 | 자료형 | 필수 | 허용값/범위 | 예시 |
|---|---|---:|---:|---|---|
| `dataset` | prepared dataset token | string | 예 | manifest와 일치 | `mnist` |
| `prep_root` | prepared 루트 | string | 예 | `<dataset>/manifest.json` 포함 | `/data/prepared` |
| `output_root` | dataset PSD CSV 출력 루트 | string | 예 | 절대경로 권장 | `/runs/exp1/dataset_psd` |
| `batch_size` | 분석 batch 크기 | integer | 예 | 1 이상 | `128` |
| `gpu_index` | CUDA 장치 index | integer | 예 | 0 이상 | `0` |
| `seed` | probe 선택 시드 | integer | 예 | 0 이상 권장 | `0` |
| `num_workers` | DataLoader worker 수 | integer | 아니오 | 0 이상 | `0` |

이 stage는 input 데이터 자체를 분석하므로 `signal_kind=input`이 정상이다.

## `dataset_fft.json`

인수는 `dataset_psd.json`과 같다. 출력 category는 `dataset_fft`이며 모델 2D FFT category와 섞지 않는다.

## `model_training.json`

| 인수 | 역할 | 자료형 | 필수 | 허용값/범위 | 예시 |
|---|---|---:|---:|---|---|
| `dataset` | 학습 dataset token | string | 예 | prepared manifest와 일치 | `mnist` |
| `prep_root` | prepared 루트 | string | 예 | manifest 포함 | `/data/prepared` |
| `model` | 모델 token | string | 예 | `lif_soft_fixed`, `rf_soft_fixed`, `vgg11_lif_soft_fixed` 등 | `lif_soft_fixed` |
| `hidden_spec` | hidden width 또는 CNN 고정값 | string | 예 | dense: `256,128`, CNN: `-` | `256,128` |
| `readout_mode` | readout 방식 | string | 예 | `temporal_membrane`, `first_spike`, `max_rate`, `spikegru_max_over_time` | `temporal_membrane` |
| `epochs` | 총 epoch 수 | integer | 예 | 1 이상 | `10` |
| `batch_size` | 학습 batch 크기 | integer | 예 | 1 이상 | `128` |
| `lr` | learning rate | number | 예 | 양수 | `0.001` |
| `seed` | 학습 시드 | integer | 예 | 0 이상 권장 | `0` |
| `gpu_index` | CUDA 장치 index | integer | 아니오 | 0 이상 | `0` |
| `num_workers` | DataLoader worker 수 | integer | 아니오 | 0 이상 | `0` |
| `anal_epoch_list` | checkpoint 저장 epoch | integer array | 아니오 | 1~`epochs` | `[10]` |
| `checkpoint_root` | `.pt` checkpoint 출력 디렉터리 | string | 예 | 빈 디렉터리 권장 | `/runs/exp1/checkpoints` |
| `metric_root` | `training_metrics.csv` 출력 디렉터리 | string | 예 | checkpoint_root 외부 | `/runs/exp1/metrics` |
| `output_root` | 실행 metadata용 출력 루트 | string/null | 아니오 | 경로 또는 null | `/runs/exp1/train` |
| `v_th` | threshold 기본값 | number | 아니오 | 양수 권장 | `1.0` |
| `resume_checkpoint` | 이어 학습 checkpoint | string/null | 아니오 | `.pt` 경로 | `null` |
| `regularization_lambda1` | input-hidden PSD 정규화 가중치 | number | 아니오 | 실수 | `0.0` |
| `regularization_lambda2` | adjacent hidden PSD 정규화 가중치 | number | 아니오 | 실수 | `0.0` |
| `regularization_signal` | 정규화 대상 trace | string | 아니오 | `y_mem`, `y_spike` | `y_mem` |
| `regularization_curve_scale` | 정규화 PSD scale | string | 아니오 | `raw`, `db` | `raw` |
| `regularization_centering` | 정규화 centering | string | 아니오 | `raw`, `centered` | `raw` |
| `regularization_reducer` | 정규화 대표화 | string | 아니오 | `mean`, `median` | `mean` |
| `regularization_distance_metric` | 정규화 거리 | string | 아니오 | `centered_l2`, `diff_l2` | `centered_l2` |

## 모델 분석 설정

`psd_analysis.json`, `element_psd.json`, `fft2d_analysis.json`은 공통적으로 다음 인수를 사용한다.

| 인수 | 역할 | 자료형 | 필수 | 허용값/범위 | 예시 |
|---|---|---:|---:|---|---|
| `checkpoint` | `.pt` 파일 또는 `.pt`만 포함하는 디렉터리 | string | 예 | 파일/디렉터리 경로 | `/runs/exp1/checkpoints` |
| `dataset` | checkpoint dataset token | string | 예 | checkpoint metadata와 일치 | `mnist` |
| `prep_root` | prepared 루트 | string | 예 | manifest 포함 | `/data/prepared` |
| `output_root` | 분석 CSV 출력 루트 | string | 예 | 절대경로 권장 | `/runs/exp1/psd_analysis` |
| `anal_batch` | 분석 batch 크기 | integer | 예 | 1 이상 | `128` |
| `gpu_index` | CUDA 장치 index | integer | 예 | 0 이상 | `0` |
| `seed` | probe 선택 시드 | integer/null | 아니오 | null이면 checkpoint seed 사용 | `0` |
| `num_workers` | DataLoader worker 수 | integer | 아니오 | 0 이상 | `0` |
| `low_vram` | CPU staging 사용 여부 | integer/bool | 아니오 | `0` 또는 `1` | `0` |

`psd_analysis` 추가 인수:

| 인수 | 역할 | 자료형 | 필수 | 허용값/범위 | 예시 |
|---|---|---:|---:|---|---|
| `analysis_distance_metric` | curve distance metric | string | 아니오 | `centered_l2`, `diff_l2` | `centered_l2` |
| `enable_pairwise_dependency_appendix` | pairwise appendix 저장 여부 | boolean | 아니오 | true/false | `false` |

모델 분석 stage는 hidden/output 계층만 분석한다.

## `plotting.json`

| 인수 | 역할 | 자료형 | 필수 | 허용값/범위 | 예시 |
|---|---|---:|---:|---|---|
| `input` | CSV 파일 또는 CSV 디렉터리 | string | 예 | 존재하는 경로 | `/runs/exp1/psd_analysis` |
| `output` | PNG 출력 루트 | string/null | 아니오 | 경로 또는 null | `/runs/exp1/plots` |
| `output_root` | `output`의 호환 alias | string/null | 아니오 | `output`과 동시 사용 금지 | `null` |
| `format` | figure 포맷 | string | 아니오 | `png` | `png` |
| `overwrite` | 기존 figure 덮어쓰기 | boolean | 아니오 | true/false | `true` |
| `manifest_name` | plotting manifest 파일명 | string | 아니오 | 경로 구분자 없는 `.csv` 파일명 | `recursive_plot_manifest.csv` |
| `include_filter_count` | filter plot에 count 포함 | boolean | 아니오 | true/false | `false` |
