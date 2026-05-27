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
- `bash/model_training.sh`, `bash/model_training_ddp.sh`는 여러 config path를 인자로 받으면 config별 백그라운드 프로세스를 동시에 띄운다. JSON 내부의 list를 sweep으로 해석하지 않는다.

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
| `model` | 모델 token | string | 예 | 아래 "SRNN/모델 선택법" 참조 | `lif_soft_fixed`, `lif_R_soft_fixed` |
| `hidden_spec` | hidden width 또는 CNN 고정값 | string | 예 | dense/SRNN: `256,128`, CNN: `-` | `256,128` |
| `readout_mode` | readout 방식 | string | 예 | `temporal_membrane`, `final_membrane`, `first_spike`, `max_fire`, `spikegru_max_over_time` | `temporal_membrane` |
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

### model_training 병렬 시나리오 실행

`bash/model_training.sh`와 `bash/model_training_ddp.sh`는 config 파일 하나만 실행하는 단일 wrapper가 아니라 config 목록을 순회해 각 시나리오를 별도 백그라운드 프로세스로 띄우는 launcher다. 각 프로세스는 같은 `RUN_STAMP`와 config stem을 사용해 `logs/<stage>/<stamp>__<config-stem>.log`에 기록한다. `bash/psd_analysis.sh`도 config 목록을 받지만, 분석 job은 nohup을 유지한 채 config별로 직렬 실행한다.

```bash
# 단일 기본 config
bash/model_training.sh
bash/model_training_ddp.sh

# 여러 비-DDP 시나리오 병렬 실행
bash/model_training.sh \
  config/train/mnist_lif.json \
  config/train/mnist_rf.json \
  config/train/mnist_lif_R.json

# 여러 2-GPU DDP 시나리오 병렬 실행
bash/model_training_ddp.sh \
  config/ddp_train/mnist_lif.json \
  config/ddp_train/mnist_rf.json
```

Wrapper 파일 안의 `CONFIG_PATHS=(...)` 배열에 config를 직접 추가해 고정 launcher로 써도 된다. CLI 인자를 넘기면 `CONFIG_PATHS` 기본 배열은 무시되고 인자 목록만 실행된다.

병렬 실행 시 각 config의 `checkpoint_root`, `metric_root`, `output_root`는 서로 다른 디렉터리여야 한다. 같은 출력 루트를 공유하면 checkpoint 빈 디렉터리 검증 또는 metric 저장에서 충돌한다. DDP wrapper는 각 config마다 `torchrun --standalone --nproc_per_node=2`를 별도 실행하므로 config 하나가 GPU 2장을 점유한다.

### SRNN/모델 선택법

`model`은 `src/model/model_registry.py`의 canonical token parser를 통과해야 한다. 하이픈은 내부적으로 언더스코어로 정규화되지만, config에는 언더스코어 표기를 권장한다.

| 목적 | token 형식 | 예시 | `hidden_spec` | 권장 `readout_mode` |
|---|---|---|---|---|
| 일반 dense IF | `if_<soft|hard>_<fixed|train>` | `if_soft_fixed` | `256,128` | `temporal_membrane`, `final_membrane`, `first_spike`, `max_fire` |
| 일반 dense LIF | `lif_<soft|hard>_<fixed|train>` | `lif_soft_fixed` | `256,128` | `temporal_membrane`, `final_membrane`, `first_spike`, `max_fire` |
| 일반 dense RF | `rf_<soft|hard|none>_<fixed|train>` | `rf_soft_fixed`, `rf_none_train` | `256,128` | `temporal_membrane`, `final_membrane`, `first_spike`, `max_fire` |
| SRNN/dense recurrent LIF | `lif_R_<soft|hard>_<fixed|train>` | `lif_R_soft_fixed` | `256,128` | `temporal_membrane`, `final_membrane`, `max_fire` |
| SRNN/dense recurrent RF | `rf_R_<soft|hard|none>_<fixed|train>` | `rf_R_soft_fixed`, `rf_R_none_train` | `256,128` | `temporal_membrane`, `final_membrane`, `max_fire` |
| fixed CNN LIF | `<vgg11|resnet18>_lif_<soft|hard>_<fixed|train>` | `vgg11_lif_soft_fixed` | `-` | `temporal_membrane`, `final_membrane`, `max_fire` |
| fixed CNN RF | `<vgg11|resnet18>_rf_<soft|hard>_<fixed|train>` | `resnet18_rf_soft_fixed` | `-` | `temporal_membrane`, `final_membrane`, `max_fire` |
| 보조 sequence 모델 | `spikegru`, `spikingssm`, `spikformer` | `spikegru` | 모델별 builder 계약 확인 | `spikegru_max_over_time`는 `spikegru` 전용 |

Readout 의미:

- `temporal_membrane`: output layer를 non-spiking/no-reset으로 두고 전체 시간의 `output_membrane` logits를 시간축 평균해 class logits로 사용한다.
- `final_membrane`: output layer를 non-spiking/no-reset으로 두고 마지막 timestep의 `output_membrane[:, -1, :]`를 class logits로 사용한다.
- `max_fire`: output spike를 시간축 합산한 class별 발화횟수(`sum_t spike`)를 logits로 사용한다.
- `max_rate`는 구형 checkpoint/CLI 호환을 위한 deprecated alias이며 새 config 예시에서는 사용하지 않는다.

선택 순서:

1. **비교하려는 neuron family**를 먼저 고른다. LIF/RF 비교는 같은 reset/threshold suffix를 맞춘다.
2. **recurrent 효과**가 필요하면 dense token에 `_R`을 붙인다. CNN token에는 `_R`을 붙이지 않는다.
3. **reset suffix**는 IF/LIF에서는 `soft` 또는 `hard`를 명시한다. RF에서는 `soft`, `hard`, `none`을 사용할 수 있으며 `none`은 reset을 비활성화한다.
4. **threshold suffix**는 `fixed` 또는 `train`을 명시한다. `train`은 threshold parameter 학습 비교가 목적일 때만 쓴다.
5. **입력 view**는 model family에 따라 registry에서 선택된다. CNN family는 frame/image view를 사용하므로 `hidden_spec`을 `-`로 둔다.

`tc_lif`, `ts_lif`, `dh_snn`, `d_rf` 계열도 공식 root pipeline `model` token으로 지원한다.

추가 family token 규칙(v1):

- **TC-LIF**: `tc_lif`, `tc_lif_R`
  - alias: `tc`, `tc_R`, `tclif`, `tclif_R`
- **TS-LIF**: `ts_lif`, `ts_lif_R`
  - alias: `ts`, `ts_R`, `tslif`, `tslif_R`
- **DH-SNN**: `dh_snn_<branch>`, `dh_snn_R_<branch>`
  - alias: `dh`, `dh_R`, `dh_<branch>`, `dh_R_<branch>`
  - branch 생략 시 canonical은 `dh_snn_4`, `dh_snn_R_4`
  - branch는 양의 정수만 허용
- **D-RF(v1)**: `d_rf_<branch>`
  - branch 생략 시 canonical은 `d_rf_4`
  - branch는 양의 정수만 허용
  - `d_rf_R`, `d_rf_R_<branch>`는 지원하지 않는다(DRFLayer true recurrent dynamics 미노출).

주의: 위 TC/TS/DH/D-RF token에는 `soft/hard/fixed/train` suffix를 붙이지 않는다.

### 대표화 방법 선택법

대표화는 많은 row/channel/neuron 신호를 비교 가능한 1-D 곡선 또는 저차원 mode 신호로 압축하는 규칙이다. 현재 `model_training`의 in-loop PSD regularization은 `regularization_reducer`로 scalar 대표화만 선택한다.

| 방법 | 의미 | 장점 | 권장 상황 |
|---|---|---|---|
| `mean` | row별 PSD를 산술평균해 대표 곡선을 만든다. | 가장 안정적이고 smooth하다. | 기본값, 전체 에너지 경향 비교 |
| `median` | row별 PSD의 중앙값으로 대표 곡선을 만든다. | 일부 neuron/channel의 outlier에 덜 민감하다. | sparse spike, 일부 row만 과활성인 경우 |

`mean`과 `median`은 row 순서 또는 row 간 공분산을 보존하지 않는다. row 집단의 주성분 방향, mode별 주파수, mode 간 cross-spectrum을 보려면 PCA 대표화를 사용한다. PCA 대표화는 `regularization_reducer` 값이 아니라 별도 PCA basis와 PCA penalty/analysis 설정으로 관리한다.

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
| `psd_curve_tokens` | 분석할 PSD curve token 목록 | string array | 아니오 | `<exact\|userbin>_<mean\|median>_<raw\|centered>_<raw\|db>` | `["exact_mean_centered_raw"]` |
| `analysis_userbin_edges` | userbin PSD의 normalized frequency 구간. edge list 또는 단일 bin width | number array/number/null | userbin token 사용 시 예 | `[0.0,0.05,...,0.5]` 또는 `0.05`, 범위 `[0,0.5]` | `[0.0,0.1,0.2,0.3,0.4,0.5]` |
| `analysis_userbin_reducer` | userbin 내부 native frequency bin 집계 방식 목록 | string array | 아니오 | `mean`, `median`, `sum` | `["mean", "sum"]` |
| `analysis_distance_metric` | curve distance metric 목록 | string array | 아니오 | `centered_l2`, `diff_l2` | `["centered_l2", "diff_l2"]` |
| `enable_pairwise_dependency_appendix` | pairwise appendix 저장 여부 | boolean | 아니오 | true/false | `false` |

`psd_analysis`는 `psd_curve_tokens`, `analysis_userbin_reducer`, `analysis_distance_metric`의 가능한 조합을 계산한다. `analysis_userbin_reducer`는 userbin token에만 적용되며 exact token은 중복 계산하지 않는다. `analysis_userbin_edges`는 explicit edge list 또는 단일 width 값을 받는다. 단일 width `0.05`는 `[0.0, 0.05, ..., 0.5]`로 해석된다. `analysis_userbin_count`와 `analysis_userbin_width`는 공식 설정에서 사용하지 않는다.

모델 분석 stage는 hidden/output 계층만 분석한다.

### PCA 신호분석 설정법

PCA 분석은 row/channel/neuron 축을 고정 basis로 투영해 mode 신호를 만든 뒤 PSD를 계산하는 확장 분석이다. 현재 root pipeline의 `psd_analysis`에서 아래 옵션으로 직접 제어한다.

| 인수 | 역할 | 자료형 | 필수 | 허용값/범위 | 예시 |
|---|---|---:|---:|---|---|
| `enable_pca_1d` | PCA mode별 1-D PSD 분석 사용 | boolean/string | 아니오 | bool 또는 bool 문자열 | `true` |
| `enable_pca_mimo` | PCA mode 간 cross-spectrum/MIMO 분석 사용 | boolean/string | 아니오 | bool 또는 bool 문자열 | `true` |
| `pca_ref_epoch` | PCA basis를 fit할 기준 checkpoint epoch | integer | PCA 사용 시 예 | `anal_epoch_list`에 포함된 양의 정수 | `1` |
| `pca_min_train_accuracy` | 기준 epoch train accuracy gate | number | 아니오 | `0.0`~`1.0` | `0.0` |
| `pca_dim_per_layer` | layer별 PCA 차원 | integer array | 아니오 | 양의 정수 배열, tail broadcast | `[4]`, `[8, 4, 4]` |

운영 규칙:

1. `pca_ref_epoch`의 checkpoint에서 layer/family별 basis를 한 번 fit한다.
2. 같은 run의 다른 checkpoint에는 동일 basis id를 적용한다.
3. 서로 다른 dataset, split, scope, layer, family, basis id에서 나온 PCA projection끼리는 직접 distance를 계산하지 않는다.
4. `pca_dim_per_layer`는 0-based layer index에 대응한다. 지정값이 layer 수보다 짧으면 마지막 값을 남은 layer에 반복 적용한다.
5. PCA basis metadata에는 dataset, checkpoint, layer, signal family, row semantics, component 수, basis shape를 남긴다.

대표 산출물은 `pca_reference/`, `pca_mode_traces/`, `pca_mimo_traces/`, `pca_cross_traces/`처럼 scalar representative 산출물과 분리한다.
기본값은 `enable_pca_1d=true`, `enable_pca_mimo=true`, `pca_ref_epoch=1`, `pca_min_train_accuracy=0.0`이며, `pca_ref_epoch`가 checkpoint 목록에 없으면 실행을 실패한다.

### PCA 대표신호 기반 규제 설정법

현재 `regularization_lambda1`, `regularization_lambda2`는 input-hidden 및 adjacent hidden 사이의 scalar representative PSD 곡선 거리만 계산한다. PCA 대표신호 규제는 아래 별도 옵션(`lambda_psd_*`)으로 추가되고 기존 의미를 바꾸지 않는다.

| 인수 | 역할 | 자료형 | 허용값/범위 | 예시 |
|---|---|---:|---|---|
| `lambda_psd_rep_1d` | mean/median 대표 PSD penalty 계수 | number | 임의 실수, `0.0`이면 비활성 | `0.1` |
| `lambda_psd_pca` | PCA mode별 1-D PSD와 PCA MIMO/cross-spectrum penalty에 공통 적용되는 단일 계수 | number | 임의 실수, `0.0`이면 비활성 | `0.05` |
| `psd_reg_variant` | penalty에 사용할 신호 변형 | string | `raw`, `centered` | `centered` |
| `psd_reg_output_family` | hidden 출력 family | string | `spike`, `membrane` | `spike` |
| `pca_dim_per_layer` | fixed PCA reference bank 차원 | integer array | 양의 정수 배열 | `[4]` |

PCA 규제를 켜면 학습 시작 전에 no-grad reference batch로 layer별 `x_basis/y_basis`와 centroid를 고정하고, 학습 minibatch에서는 그 basis만 적용한다. basis 자체는 penalty gradient 대상이 아니다.
현재 DDP(`ddp=true`)에서 PCA PSD 규제(`lambda_psd_pca`)는 rank0 기준 reference bank를 broadcast하는 정책을 사용한다.

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

### data_prep dataset 다중 실행 정책

- data_prep에서만 `dataset`에 문자열 또는 문자열 리스트를 허용한다.
- 문자열이면 단일 dataset 전처리를 수행한다.
- 리스트이면 dataset별 직렬 전처리를 수행한다.
- 리스트 실행은 병렬 실행이 아니며 하나의 프로세스에서 순서대로 처리한다.
- 다른 stage에서는 dataset list를 허용하지 않는다.
- 다른 stage에서 list를 넣어도 sweep으로 해석하지 않는다.
- data_prep list 실행에서는 `raw_data_root`, `prep_root`, `seed`, `prep_profile`, `max_samples` 등 동일 옵션이 각 dataset에 동일하게 적용된다.


### model_training DDP 옵션
- `ddp`: 2-GPU DDP 사용 여부(`true/false`).
- `ddp_world_size`: 현재 `2`만 허용한다.
- `batch_size_is_global`: `batch_size`를 global batch로 해석한다. DDP에서는 `true`만 허용한다.
- DDP 모드에서는 `batch_size`를 global batch로 받아 rank별로 `batch_size // 2`를 사용한다.
- DDP 실행은 `bash/model_training_ddp.sh` 또는 `torchrun --standalone --nproc_per_node=2 ...`를 사용한다.

- DDP 스모크 테스트는 CUDA 2-GPU 환경에서만 수행 가능하다.
- DDP에서는 batch_size를 global batch로 해석하며 GPU별 per-rank batch는 batch_size/2이다.
- DDP에서는 batch_size가 반드시 짝수여야 한다.
- DDP 실행 시 checkpoint와 metric CSV 저장은 rank0만 수행한다.


### 제약(constraint) 적용 범위 주의

이번 토큰 확장 패치 기준으로 `tc_lif`, `ts_lif`, `dh_snn`, `d_rf` family에는 `clip/structure/clipstructure` constraint 모드를 적용하지 않는다.
## Constraint 설정 (clip / structure / clipstructure)

`model_training`에서 hidden dense layer 전용 constraint를 지원한다.

- `scenario_mode`: `none`, `clip`, `structure`, `clipstructure` (`clip_structure` alias)
- `alpha_clip_edges` (`lif_alpha_clip_edges` alias): **layer/group/bounds 3D** LIF clip 경계 (`[[[lo, hi], ...], ...]`, 각 bounds는 `[0,1]`)
- `w_clip_edges` (`rf_frequency_clip_edges` alias): **layer/group/bounds 3D** RF clip 경계 (`[[[lo, hi], ...], ...]`, 각 bounds는 `[0,0.5]`)
  - unit: `normalized_frequency_cyc_per_sample_nyquist_0p5`
- `band_edge`: hidden layer별 cumulative boundary list (`null` 또는 `[b1, b2, ...]`)
- `tear` (`constraint_tear` alias): 1-based hidden index, 해당 layer부터 constraint 적용

동작 규칙:
- output layer에는 constraint를 적용하지 않는다.
- `band_edge=null`이면 해당 layer에서 group 수에 맞춰 뉴런을 균등 분할한다.
- `band_edge=[5,10]`이면 그룹은 `[0,5)`, `[5,10)`, `[10,width)`로 나뉜다.
- `clipstructure`에서는 clip bounds는 모든 hidden layer에 적용되고, structure mask만 `tear` 이후 hidden layer부터 적용된다.
- structure mode에서 첫 hidden layer raw input projection에는 feedforward mask를 적용하지 않는다.
- recurrent 모델은 첫 hidden layer에도 recurrent mask를 적용할 수 있다.
- RF damping magnitude는 v1에서 CLI clip 대상이 아니다.

v1 지원 family:
- `lif`, `rf` (dense)

v1 미지원 family:
- `tc_lif`, `ts_lif`, `dh_snn`, `d_rf`, `spikegru`, `spikformer`, `spikingssm`, `cnn_lif/cnn_rf/vgg/resnet`, conv/residual arch
- 미지원 family에서 `scenario_mode != none`이면 `ValueError`를 발생시킨다.

resume 정책:
- checkpoint의 `constraint_metadata`와 현재 실행 constraint mode가 다르면 fail-fast(`ValueError`).

DDP:
- constraint plan은 deterministic 규칙으로 생성되며 각 rank에서 동일 bounds/mask를 만든다.
