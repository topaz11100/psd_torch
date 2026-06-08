# YAML Config Contract

모든 entrypoint는 `--config <path.yaml>`을 지원하며, YAML root는 stage 이름을 key로 갖는 객체다. 공식 설정 파일 확장자는 `.yaml`/`.yml`만 사용한다.

```yaml
model_training:
  data:
    dataset: ""
    prep_root: ""
  model:
    neuron_type: ""
    v_th: ["", ""]
  regularization:
    proposed_branch:
      lambda_branch_ortho: ""
      lambda_branch_s: ""
```

## Clean config 정책

현재 `config/` 아래에는 실행 가능한 완성 scenario config를 두지 않는다. 각 stage별 clean template만 둔다.

| 파일 | stage key | 용도 |
|---|---|---|
| `config/data_prep.yaml` | `data_prep` | raw dataset을 prepared bundle로 변환 |
| `config/dataset_psd.yaml` | `dataset_psd` | prepared bundle의 dataset-level PSD 산출 |
| `config/dataset_fft.yaml` | `dataset_fft` | prepared bundle의 dataset-level FFT 산출 |
| `config/model_training.yaml` | `model_training` | 학습. `distributed.gpu_index` 배열 길이에 따라 단일 GPU 또는 자동 DDP 실행 |
| `config/psd_analysis.yaml` | `psd_analysis` | checkpoint 기반 layer/signal PSD 분석 |
| `config/element_psd.yaml` | `element_psd` | element-level PSD 분석 |
| `config/element_fft.yaml` | `element_fft` | element-level FFT 분석 |
| `config/fft2d_analysis.yaml` | `fft2d_analysis` | 2D FFT 분석 |
| `config/plotting.yaml` | `plotting` | CSV 산출물 plotting |
| `config/DI.yaml` | `DI` | discriminative-index 분석 |

각 template는 parser가 받는 모든 인자를 포함한다. 값은 `""`, `[]`, 또는 `v_th: ["", ""]`처럼 공란으로 둔다. 공란 값은 `parse_args_with_config`에서 무시되고, 필수 인자가 채워지지 않았으면 명시적인 missing-required 오류가 난다.

## 중첩 YAML 계약

YAML은 사람이 읽기 쉬운 의미 단위로 중첩한다. config loader는 leaf key를 parser argument 이름으로 flatten한다.

예를 들어 다음 두 설정은 parser 입장에서는 같은 값이다.

```yaml
model_training:
  lambda_branch_s: 0.001
```

```yaml
model_training:
  regularization:
    proposed_branch:
      lambda_branch_s: 0.001
```

권장 중첩 단위는 다음과 같다.

| group | 포함 인자 예 |
|---|---|
| `data` | `dataset`, `prep_root`, `raw_data_root` |
| `model` | `neuron_type`, `recurrent`, `branch`, `hidden_spec`, `readout_mode`, `v_th` |
| `optimization` | `epochs`, `batch_size`, `lr`, `seed`, `num_workers` |
| `regularization.signal_curve` | PSD curve representation 관련 인자 |
| `regularization.psd_regularizer` | PSD regularization loss weight |
| `regularization.proposed_branch` | `my_*` branch regularization/schedule |
| `scenario_constraints` | vanilla-only `clip/structure` 관련 인자 |
| `distributed` | `gpu_index`, global batch, DDP timeout 등 multi-GPU 실행 인자 |
| `output` | checkpoint/metric/result path |
| `runtime` | compile, AMP, low-vram 등 실행 정책 |

## `gpu_index` 배열 계약

`model_training`의 `gpu_index`는 항상 배열로 기록한다.

```yaml
model_training:
  distributed:
    gpu_index: [0]
```

- `gpu_index: [k]`: 단일 프로세스 학습. 프로세스가 `cuda:k`를 직접 사용한다.
- `gpu_index: [a, b, ...]`: `src.model_training`이 `torch.distributed.run`으로 재실행된다. parent 프로세스는 `CUDA_VISIBLE_DEVICES=a,b,...`를 설정하고, child rank는 `LOCAL_RANK=0..N-1`을 사용한다.
- 별도의 `model_training_ddp.yaml`, `model_training_ddp.sh`, `ddp`, `ddp_world_size` clean 설정은 사용하지 않는다. world size는 `len(gpu_index)`와 `WORLD_SIZE`에서 결정된다.

## `v_th` 배열 계약

`v_th`는 항상 배열로 기록한다.

```yaml
model_training:
  model:
    v_th: ["train", "1.0"]
```

단일 threshold만 쓰는 경우에도 배열로 쓰며, hidden/output threshold 분리 또는 threshold 학습 정책 확장을 위해 `[policy, initial_value]` 형태를 기본으로 한다.

## Dataset 배열 계약

`data_prep`, `dataset_fft`, `dataset_psd`의 `dataset`은 문자열 또는 문자열 배열이다. 배열이면 stage 내부에서 직렬 실행한다.

```yaml
dataset_fft:
  data:
    dataset:
      - mnist
      - shd
    prep_root: /home/yongokhan/workspace/data/prep_data
  output:
    output_root: /home/yongokhan/workspace/result/dataset_fft
```

동일한 raw/prepared directory에 대해 동시에 쓰기 작업이 겹치지 않도록 dataset stage는 병렬 bash 그룹 대신 내부 직렬 실행을 기본으로 한다.

## Scenario config 폐기

과거의 `config/ddp_train_scenario`, `config/psd_analysis_scenario`, `config/*_scenario` 디렉터리는 제거했다. 새 실험은 clean template를 복사해 채우거나, 별도 sweep generator가 실행 직전에 resolved config를 생성하는 방식으로 관리한다. resolved config는 Git 추적 대상이 아니라 run artifact로 남긴다.
