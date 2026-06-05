# Config Directory Contract

이 디렉터리는 실행 가능한 YAML 설정의 기준 위치다. 모든 설정은 YAML root에 stage key를 둔다.

```yaml
model_training:
  dataset: mnist
  prep_root: /home/yongokhan/workspace/data/prep_data
  seed: 0
```

## 공통 기본값

| 항목 | 값 |
|---|---|
| `raw_data_root` | `/home/yongokhan/workspace/data/raw_data` (`data_prep` 전용) |
| `prep_root` | `/home/yongokhan/workspace/data/prep_data` |
| 산출물 루트 | `/home/yongokhan/workspace/result/...` |
| `seed` | `0` |

`raw_data_root`는 `data_prep` 전용 인자다. 다른 stage는 이미 준비된 `prep_root` 산출물만 읽으므로 raw-data 경로를 갖지 않는다. public config 키는 `raw_data_root` 하나만 사용한다.

`model_training`은 산출물을 `checkpoint_root`와 `metric_root`로만 제어한다. 이전의 `output_root` 계열 training 인자는 실제 저장 경로에 쓰이지 않으므로 public config에서 제거했다. `output_root`는 PSD/FFT/plotting 등 분석 stage에서만 사용한다.

## Base clean config

`config/base/*.clean.yaml`은 특정 실험 시나리오가 없는 백지 설정이다. 모든 필수 인자를 포함한다.

- `data_prep.clean.yaml`
- `model_training.clean.yaml`
- `model_training_ddp.clean.yaml`
- `psd_analysis.clean.yaml`
- `fft2d_analysis.clean.yaml`
- `element_psd.clean.yaml`
- `element_fft.clean.yaml`
- `dataset_fft.clean.yaml`
- `dataset_psd.clean.yaml`
- `plotting.clean.yaml`


## Model schema

`model_training` public config는 monolithic model token을 쓰지 않고 다음 explicit fields를 사용한다.

```yaml
neuron_type: lif
recurrent: false
reset: soft
v_th: [fixed, 1.0]
filter: train
```

- `neuron_type`: `lif`, `rf`, `tc`, `ts`, `dh_snn_4`, `d_rf_4`, `spikegru`, `spikeformer`, `vgg11_lif`, `resnet18_lif` 등 큰 family.
- `recurrent`: 기존 `_R` suffix에 해당한다.
- `reset`: `soft`, `hard`, `none`. 해당 없는 구조 모델은 `null`을 둔다.
- `v_th`: `["fixed"|"train", initial_value]`.
- `filter`: `"train"`이면 filter parameter를 학습하고, 숫자 문자열이면 해당 값으로 고정한다. 고정값이 clip/constraint 범위 밖이면 layer construction 단계에서 에러를 낸다.

## Scenario 대응표

`config/ddp_train_scenario/<group>/<case>.yaml`으로 학습한 경우, 같은 `<group>/<case>` 이름으로 다음 분석 설정이 존재한다.

| 목적 | 디렉터리 | stage key |
|---|---|---|
| PSD 대표곡선/거리/분산 | `config/psd_analysis_scenario` | `psd_analysis` |
| 2D FFT | `config/fft2d_analysis_scenario` | `fft2d_analysis` |
| 2D FFT 별칭 | `config/2dfft_scenario` | `fft2d_analysis` |
| neuron별 element PSD | `config/element_psd_scenario` | `element_psd` |
| neuron별 element FFT | `config/element_fft_scenario` | `element_fft` |
| dataset input FFT baseline | `config/dataset_fft_scenario` | `dataset_fft` |
| 뉴런 motivation PSD scenario | `config/neuron_motivation_scenario` | `model_training`, `psd_analysis` |

## Dataset 배열 직렬 실행

다음 stage는 config의 `dataset`을 배열로 받을 수 있다.

- `data_prep`
- `dataset_fft`
- `dataset_psd`

배열이면 데이터셋을 직렬 실행한다. `dataset_fft`와 `dataset_psd`는 기본적으로 `output_root/<stage>_<timestamp>/<dataset>/...`로 산출물을 분리한다. `timestamped_output=false`이면 기존처럼 `output_root/<dataset>/...`를 사용한다.

```yaml
dataset_fft:
  dataset: [mnist, shd]
  prep_root: /home/yongokhan/workspace/data/prep_data
  output_root: /home/yongokhan/workspace/result/dataset_fft
  batch_size: 128
  gpu_index: 0
  seed: 0
  num_workers: 0
```


다른 stage에서는 dataset list를 허용하지 않는다. 즉 `model_training`, `psd_analysis`, `fft2d_analysis`, `element_psd`, `element_fft`, `plotting`은 단일 dataset token만 받으며, sweep은 bash의 2차원 config group으로 표현한다.

주요 선택형 인자는 다음과 같다.

- `signal_curve_centering`: `raw`, `centered`
- `signal_curve_space`: `exact`, `userbin`
- `signal_curve_scale`: `raw`, `db`, `area`
- `signal_curve_userbin_edges`: explicit normalized-frequency edge array for userbin PSD curves
- `signal_curve_userbin_reducer`: `mean`, `median`, `sum`
- `psd_reg_output_family`: `spike`, `membrane`
- `analysis_distance_metric`: `centered_l2`, `diff_l2`
- `signal_window`: `hann`, `none`. `hann`은 기존 Hann taper를 적용하고, `none`은 window 없이 periodogram/PSD를 계산한다.
- `analysis_checkpoint_epochs`: 분석/evaluation/metric 기록/체크포인트 저장을 수행할 epoch 배열. YAML에서는 한 줄 배열로 작성한다. 예: `analysis_checkpoint_epochs: [1, 5, 10, 20, 30]`

## Bash 2차원 그룹 실행

다음 bash 스크립트는 `CONFIG_GROUP_*` 2차원 배열 계약을 따른다.

- `bash/model_training.sh`
- `bash/model_training_ddp.sh`
- `bash/psd_analysis.sh`
- `bash/fft2d_analysis.sh`
- `bash/element_psd.sh`
- `bash/element_fft.sh`
- `bash/plotting.sh`

같은 group의 config들은 병렬 실행되고, group 사이에는 직렬 barrier가 있다.

```bash
CONFIG_GROUP_0=("case_a.yaml" "case_b.yaml")
CONFIG_GROUP_1=("case_c.yaml")
CONFIG_GROUPS=(CONFIG_GROUP_0 CONFIG_GROUP_1)
```

학습 launcher에 CLI 인자로 디렉터리를 넘기면 leaf 폴더 단위로 직렬 group을 만들고, 같은 leaf 폴더 안의 config는 병렬 실행한다. d_rf를 제외한 RF family config는 같은 leaf 안에서도 RF-only 병렬 group으로 분리한다. 개별 YAML 파일 인자만 넘긴 경우에는 하나의 병렬 group으로 취급한다.

## `torch.compile` / precision 설정

`model_training`과 `model_training_ddp`의 공개 config key는 단순화했다. compile 적용 방식은 코드 레벨에서 하나로 고정한다.

| key | 의미 | 기본 |
|---|---|---|
| `compile` | `torch.compile`/regional compile 최적화 사용 여부. | `true` |
| `compile_cpu_threads` | compile/Inductor 준비에 사용할 CPU thread 수. 생략 시 DDP는 2, 단일 프로세스는 기존 auto-cap 정책을 쓴다. `PSD_TORCH_COMPILE_CPU_THREADS`로도 지정할 수 있다. | `null` 또는 DDP config의 `2` |
| `ddp_timeout_minutes` | DDP/NCCL collective timeout. 긴 첫 compile/eval compile 중 watchdog timeout을 방지한다. | `120` |
| `amp` | AMP 모드. `off` 또는 `on`만 허용한다. `on`은 내부적으로 항상 `bf16_safe` 정책으로 실행하며 CUDA forward에만 BF16 autocast를 적용하고 readout/loss/PSD/FFT/signal-analysis는 FP32로 보호한다. | `off` |

고정 compile 정책:

- CUDA: regional sequence compile은 `backend=inductor`, `fullgraph=true`, `dynamic=false`로 적용한다.
- CPU: 긴 Inductor codegen 지연을 피하기 위해 `backend=eager`로 고정한다.
- `torch.compiler.set_stance('eager_then_compile')`를 사용할 수 있으면 적용한다.
- IF/LIF/RF/TC-LIF/TS-LIF/DH-SNN/CNN2D/SpikGRU는 timestep 1개가 아니라 layer sequence 전체를 regional compile 단위로 사용한다.
- sequence backend는 `compiled_sequence_prealloc`으로 고정한다. timestep output은 compile 대상 sequence 함수 내부에서 preallocated tensor에 기록한다.
- step-level compile과 프로젝트 모델의 top-level fallback compile은 기본 경로에서 제거한다. compile hook이 sequence region을 설치하지 못하면 eager sequence path로 실행한다.
- D-RF는 원본 `BiRFModel.forward` 경로를 보존하며 regional rewrite/outer model compile 대상으로 삼지 않는다.
- 반복 실험 cold-start 완화를 위해 compile cache를 `.torch_compile_cache/rank*` 아래에 구성한다. DDP bash 실행에서는 `--compile-cache-root`와 `--experiment-name`을 주면 `<cache_root>/<experiment_name>/<config_stem>/rank*` 구조로 분산 저장한다.
- DDP compile CPU thread 기본값은 rank당 2이다. `compile_cpu_threads` config/CLI 값이 있으면 그 값을 우선하고, 그 다음 `PSD_TORCH_COMPILE_CPU_THREADS` 환경변수를 본다. 명시값은 8-thread 자동 cap에 의해 잘리지 않는다.
- DDP/NCCL timeout은 `ddp_timeout_minutes`로 설정하며 기본 120분이다. 긴 compile 구간에서 rank 간 도착 시간이 달라도 기본 10분 watchdog timeout으로 abort되지 않도록 한다.
- compile을 켜면 static shape 성능을 위해 training dataset이 global batch 이상인 경우 training loader에 `drop_last=true` 정책을 사용한다.
- DDP evaluation은 rank0-only가 아니라 모든 rank가 test split의 strided subset을 평가하고, loss/correct/total metric을 `all_reduce`로 합산한다. evaluation DataLoader의 nominal batch size는 train per-rank batch와 동일하게 둔다.
- TF32는 별도 config key 없이 항상 활성화한다. 텐서 dtype은 FP32를 유지하고, CUDA FP32 matmul/conv 내부 precision만 TF32를 허용한다.
- CNN 계열 입력/모델은 channels-last memory format을 적용한다.
- CNN core 전체 compile은 기본 비활성이다. `PSD_ENABLE_CNN_CORE_COMPILE=1`일 때만 no-trace CNN core를 compile한다. layer 내부 recurrent sequence compile과 별도 실험 옵션으로 취급한다.
- 체크포인트에는 실제 적용된 compile/precision/DDP-eval 정책 metadata를 기록한다. config에서 커스텀할 수 있는 compile 관련 인수는 `compile`과 `compile_cpu_threads`이며, DDP runtime timeout은 `ddp_timeout_minutes`로 둔다.

`amp="on"`은 Linear/Conv 등 큰 forward 연산에만 CUDA BF16 autocast를 허용하고, readout/loss/PSD/FFT/signal-analysis 경로는 FP32로 되돌린다.

Sequence output buffer는 preallocated tensor slice-write 방식으로 고정한다. public config key나 환경 변수 선택지는 제공하지 않는다.

`compile_startup_status` 로그의 `sequence_backend`, `sequence_buffer_mode`, `compile_child_region_count`, `compiled_region_count`를 함께 확인한다. graph break / recompile 확인이 필요하면 다음처럼 실행한다.

```bash
TORCH_LOGS="graph_breaks,recompiles" python src/model_training.py --config config/model_training.yaml
```


## Timestamped output policy

결과를 쓰는 entrypoint는 기본적으로 실제 산출물 root 아래에 실행시각 폴더를 추가한다. 반복 실행해도 같은 종류의 결과가 같은 폴더를 덮어쓰지 않도록 하기 위한 정책이다. 일반 분석 stage는 `<stage>_<timestamp>` 하위 폴더를 사용하고, `model_training`의 `checkpoints`/`metrics` leaf는 같은 parent의 `run_<timestamp>/checkpoints`, `run_<timestamp>/metrics`로 묶인다.

| key | 의미 | 기본 |
|---|---|---|
| `timestamped_output` | `true`이면 실제 저장 경로에 실행시각 폴더를 추가한다. `false`이면 지정 root에 직접 저장한다. | `true` |
| `run_timestamp` | 자동 생성 timestamp 대신 사용할 폴더 suffix. 테스트/재현용으로 사용한다. | `null` |

예시:

```text
output_root=/home/yongokhan/workspace/result/base/psd_analysis
→ /home/yongokhan/workspace/result/base/psd_analysis/psd_analysis_20260530_181234_123456

checkpoint_root=/home/yongokhan/workspace/result/model_training/checkpoints
metric_root=/home/yongokhan/workspace/result/model_training/metrics
→ /home/yongokhan/workspace/result/model_training/run_20260530_181234_123456/checkpoints
→ /home/yongokhan/workspace/result/model_training/run_20260530_181234_123456/metrics
```

`data_prep`은 prepared data 생성 단계이므로 이 정책 대상이 아니며, 기존 `force_overwrite` 계약을 사용한다.

## Checkpoint path 규칙

학습 scenario `<group>/<case>`의 기본 산출물은 다음과 같다.

```text
/home/yongokhan/workspace/result/ddp_train_scenario/<group>/<case>/checkpoints
/home/yongokhan/workspace/result/ddp_train_scenario/<group>/<case>/metrics
```

분석 scenario는 해당 checkpoint directory를 `checkpoint`로 참조한다.


## DI stage

`config/DI.yaml`은 prepared data에서 직접 dataset-level frequency discriminative index를 계산한다. `psd_value_transform`은 `raw`, `db`, `area`를 지원한다.
