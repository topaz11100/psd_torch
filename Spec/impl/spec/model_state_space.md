# SpikingSSM auxiliary diversity 구현 명세

## 1. 범위

이 문서는 `Spec/theory/models/state_space.md` 를 구현으로 연결한다. State-space 실험은 주 분석 모델이 아니라 보조 다양성 실험이다. `paper/state_space` 의 SpikingSSMs 논문과 local 저자 제공 코드에서 가장 단순한 sequential MNIST classification profile 을 가져와 고정하고, dataset 만 adapter 로 바꿔 실행한다.

공식 구현 token 은 아래 하나다.

```text
spikingssm
```

`spikingssm` 은 `n_layers=2`, `d_model=400`, `d_state=64` profile 을 뜻한다. token 이름에 hidden size 와 dataset 이름을 붙이지 않는다.

## 2. official source contract

공식 source 는 아래다.

1. `Origin/state_space_sd4/models/spike/ss4d.py::SpikingSSM`
2. `Origin/state_space_sd4/models/spike/ss4d.py::SS4D`
3. `Origin/state_space_sd4/models/spike/ss4d.py::S4DKernel`
4. `Origin/state_space_sd4/src/models/spike/neuron.py::SDNNeuron`
5. `Origin/state_space_sd4/src/models/spike/neuron.py::BPTTNueron`
6. `Origin/state_space_sd4/src/models/spike/neuron.py::SLTTNueron`
7. `Origin/state_space_sd4/configs/experiment/spikingssm/minst.yaml`
8. `Origin/state_space_sd4/train.py`

`Origin/state_space_sd4/configs/experiment/spikingssm/pminst.yaml`, LRA configs, WikiText-103 config, DVS128 Gesture config 는 official `spikingssm` model source 가 아니다. 이 문서의 diversity run 은 dataset adapter 만 바꾼다.

External S4 dependency 는 local bundle 에 없을 수 있다. 구현은 존재하지 않는 local path 를 명세 근거로 만들지 말고, 실행 시 사용한 dependency path 를 metadata 의 `external_dependency_path` 로 기록한다.

## 3. fixed profile table

| field | value |
| --- | --- |
| `model_profile` | `spikingssm` |
| `paper_experiment` | `sequential_mnist_classification` |
| `origin_code_path` | `Origin/state_space_sd4/models/spike/ss4d.py` |
| `origin_train_entrypoint` | `Origin/state_space_sd4/train.py` |
| `origin_class_name` | `SpikingSSM` |
| `origin_block_name` | `SS4D` |
| `origin_config_path` | `Origin/state_space_sd4/configs/experiment/spikingssm/minst.yaml` |
| `dataset_permute` | false |
| `n_layers` | 2 |
| `d_model` | 400 |
| `d_state` | 64 |
| `bidirectional` | false |
| `prenorm` | false |
| `dropout` | 0.1 |
| `optimizer` | Adam |
| `layer_lr` | 0.001 |
| `optimizer_lr` | 0.01 |
| `batch_size` | 50 |
| `max_epochs` | 100 |
| `seed` | 1111 |
| `structure_variation` | `none` |

Optimizer policy:
- Official PSD project execution uses Adam.
- The SpikingSSM diversity profile does not expose a parameter-decay control.
- Source configs that include decay-like optimizer fields are recorded only as provenance when needed and are not implemented as project-side knobs.

## 4. 구현 요구사항

### IMPL-MODEL-SSM-001

요구사항:
Model registry 는 `spikingssm` 하나를 official state-space diversity profile 로 제공해야 한다.

적용 대상:
- `src/model/model_registry.py`
- `src/model/author_adapter_state_space.py`
- `src/psd_analysis.py`

수용 기준:
- registry 는 `Origin/state_space_sd4/models/spike/ss4d.py::SpikingSSM` 를 기준으로 wrapper 를 만든다.
- `n_layers=2`, `d_model=400`, `d_state=64`, `dropout=0.1` 을 유지한다.
- model token 은 `spikingssm` 으로 저장한다.
- 이전 긴 fixed-profile token 패턴을 official metadata 로 저장하지 않는다.

### IMPL-MODEL-SSM-002

요구사항:
State-space source architecture 를 wrapper 에서 임의 변경하지 않아야 한다.

적용 대상:
- `Origin/state_space_sd4/models/spike/ss4d.py`
- `src/model/author_adapter_state_space.py`

수용 기준:
- `SpikingSSM`, `SS4D`, `S4DKernel` 의 forward 순서를 보존한다.
- `SS4D` 내부 default `neuron="sdn"` source behavior 를 보존한다.
- residual add, prenorm/postnorm, dropout, GLU 순서를 바꾸지 않는다.
- `Loihi-inspired simplified S4D small`, non-spiking S4D, standalone SDN 을 official `spikingssm` profile 로 사용하지 않는다.

### IMPL-MODEL-SSM-003

요구사항:
Dataset 변경은 dataset adapter 와 classifier head output dimension 으로만 처리해야 한다.

적용 대상:
- `src/model/author_adapter_state_space.py`
- `src/data/registry.py`
- `src/data_prep.py`

수용 기준:
- adapter output 은 `[B,L,400]` 이다.
- `L` 은 dataset 에서 정의한 sequence length 또는 adapter 가 만든 sequence length 로 metadata 에 기록한다.
- classifier head output dimension 은 class 수에 맞게 바꿀 수 있다.
- SS4D layer 수, state dimension, model dimension, neuron registry 를 dataset 별로 바꾸지 않는다.
- dataset adapter 는 `adapter_is_model_variation = false` 로 기록한다.

### IMPL-MODEL-SSM-004

요구사항:
State-space observer 는 model input layout 에 맞춰 logical PSD view 를 해석해야 한다.

적용 대상:
- `src/model/author_adapter_state_space.py`
- `src/psd_analysis.py`
- `src/signal/family_spectral_analysis.py`

수용 기준:
- source input `[B,L,D]` 와 internal `[B,D,L]` 를 metadata 로 기록한다.
- `psd_time_axis = L`, `psd_row_axes = D` 또는 hook family 에 맞는 state/channel axes 를 기록한다.
- PSD 분석을 위해 source input 을 별도 canonical physical layout 으로 강제 저장하지 않는다.
- complex state 는 hook 가능한 경우 real row 와 imaginary row 로 분리한다.

### IMPL-MODEL-SSM-005

요구사항:
State-space observer 는 raw activation archive 를 만들지 않고 hook 가능한 family 를 PSD accumulator 로 streaming 해야 한다.

적용 대상:
- `src/model/author_adapter_state_space.py`
- `src/psd_analysis.py`
- `src/signal/family_spectral_analysis.py`

수용 기준:
- `x_probe`, `z_front`, `x_layer`, `i_state`, `z_state`, `y_mem`, `y_spike` 를 hook 가능한 범위에서 수집한다.
- analysis checkpoint epoch 에서 full raw state archive 를 저장하지 않는다.
- `i_state` 와 `z_state` 는 observe-only auxiliary family 로 기록한다.

### IMPL-MODEL-SSM-006

요구사항:
Run metadata 는 이 실험이 보조 diversity run 임을 명시해야 한다.

적용 대상:
- `src/util/metadata.py`
- `src/psd_analysis.py`

수용 기준:
- `analysis_role = auxiliary_diversity` 를 기록한다.
- `optimizer_family = Adam` 을 기록한다.
- `weight_decay` 계열 optimizer field 는 metadata 에 official setting 으로 기록하지 않는다.
- `main_analysis_target = dense_snn` 을 기록한다.
- `model_profile = spikingssm` 을 기록한다.
- `paper_experiment`, `origin_code_path`, `origin_neuron_path`, `origin_config_path`, `origin_class_name`, `external_dependency_path`, `structure_variation = none` 을 기록한다.
- generic `state_space` label 만 저장하지 않는다.
