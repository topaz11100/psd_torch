# Spikformer auxiliary diversity 구현 명세

## 1. 범위

이 문서는 `Spec/theory/models/spikformer.md` 를 구현으로 연결한다. Spikformer 실험은 주 분석 모델이 아니라 보조 다양성 실험이며, CIFAR10-DVS 저자 코드의 `2-256` profile 하나를 고정해서 dataset 만 바꿔 실행한다.

공식 구현 token 은 아래 하나다.

```text
spikformer
```

## 2. official source contract

공식 source 는 아래다.

1. `Origin/spikformer/cifar10dvs/model.py::spikformer`
2. `Origin/spikformer/cifar10dvs/model.py::Spikformer`
3. `Origin/spikformer/cifar10dvs/model.py::SPS`
4. `Origin/spikformer/cifar10dvs/model.py::SSA`
5. `Origin/spikformer/cifar10dvs/model.py::Block`
6. `Origin/spikformer/cifar10dvs/model.py::MLP`
7. `Origin/spikformer/cifar10dvs/train.py`

CIFAR static source `Origin/spikformer/cifar10/` 와 ImageNet source `Origin/spikformer/imagenet/` 은 이 diversity profile 의 official source 가 아니다. DVS128 Gesture 에 전용 local source path 가 없더라도 model source 를 바꾸지 않고 CIFAR10-DVS `2-256` profile 을 사용한다.

## 3. fixed profile table

| field | value |
| --- | --- |
| `model_profile` | `spikformer` |
| `paper_experiment` | `cifar10_dvs_neuromorphic_classification` |
| `origin_code_path` | `Origin/spikformer/cifar10dvs/model.py` |
| `origin_train_entrypoint` | `Origin/spikformer/cifar10dvs/train.py` |
| `origin_factory_name` | `spikformer` |
| `origin_class_name` | `Spikformer` |
| `model_size` | `2-256` |
| `patch_size` | 16 |
| `in_channels` | 2 |
| `img_size` | 128x128 |
| `embed_dims` | 256 |
| `depths` | 2 |
| `num_heads` | 16 |
| `mlp_ratios` | 4 |
| `sr_ratios` | 1 |
| `time_steps` | 16 |
| `optimizer` | Adam |
| `learning_rate` | 0.1, source learning-rate setting |
| `warmup_epochs` | 10 |
| `mixup` | 0.5 |
| `label_smoothing` | 0.1 |
| `batch_size` | 16 |
| `structure_variation` | `none` |

Optimizer policy:
- Official PSD project execution uses Adam.
- Decoupled parameter decay and `weight_decay` style knobs are not part of this implementation profile.
- If the source paper or parser exposes another optimizer family, it is treated as source provenance and not as an official project-side training feature.

## 4. 구현 요구사항

### IMPL-MODEL-SPIK-001

요구사항:
Spikformer registry 는 `spikformer` 하나를 official diversity profile 로 제공해야 한다.

적용 대상:
- `src/model/model_registry.py`
- `src/model/author_adapter_spikformer.py`
- `src/psd_analysis.py`

수용 기준:
- official profile 은 `Origin/spikformer/cifar10dvs/model.py::spikformer` factory 를 기준으로 한다.
- CIFAR static, ImageNet, DVS128-specific source profile 을 dataset 별로 자동 선택하지 않는다.
- `model_size = 2-256` 을 metadata 로 기록한다.
- 이전 긴 fixed-profile token 패턴을 official metadata 로 저장하지 않는다.

### IMPL-MODEL-SPIK-002

요구사항:
Spikformer source architecture 를 wrapper 에서 임의 변경하지 않아야 한다.

적용 대상:
- `Origin/spikformer/cifar10dvs/model.py`
- `src/model/author_adapter_spikformer.py`

수용 기준:
- `patch_size=16`, `embed_dims=256`, `num_heads=16`, `depths=2`, `mlp_ratios=4`, `sr_ratios=1` 을 유지한다.
- SPS 의 convolution-spiking-pooling stage 와 RPE stage 를 유지한다.
- SSA 의 spike-form Q/K/V, no-softmax attention product, projection LIF 구조를 유지한다.
- MLP 의 `Conv1d -> BN -> LIF -> Conv1d -> BN -> LIF` 구조를 유지한다.
- VSA, IF, ReLU, LeakyReLU ablation 을 official Spikformer diversity profile 에 넣지 않는다.

### IMPL-MODEL-SPIK-003

요구사항:
Dataset 변경은 dataset adapter 와 classifier head output dimension 으로만 처리해야 한다.

적용 대상:
- `src/model/author_adapter_spikformer.py`
- `src/data/registry.py`
- `src/data_prep.py`

수용 기준:
- adapter output 은 `[B,T,2,128,128]` 이다.
- `T=16` 을 유지한다.
- classifier head output dimension 은 class 수에 맞게 바꿀 수 있다.
- embedding dimension, encoder depth, attention head 수, patch size, SPS/SSA/MLP 구조를 dataset 별로 바꾸지 않는다.
- static image 또는 non-DVS input 을 사용할 경우 adapter 의 two-channel frame construction 을 manifest 에 기록한다.

### IMPL-MODEL-SPIK-004

요구사항:
Spikformer official PSD observer 는 raw input `x_probe` 에만 image-form row-time 변환을 적용해야 한다.

적용 대상:
- `src/model/author_adapter_spikformer.py`
- `src/psd_analysis.py`
- `src/signal/family_spectral_analysis.py`

수용 기준:
- source input `[B,T,2,128,128]` 와 internal `[T,B,2,128,128]` 를 metadata 로 기록한다.
- raw input PSD view 는 `[B, R = 2 * 128 * 128, T]` 로 해석한다.
- `psd_time_axis = T`, `psd_row_axes = polarity/channel/spatial` 을 metadata 로 기록한다.
- token axis 를 temporal PSD time axis 로 쓰지 않는다.
- SPS embedding, token representation, attention score matrix, $QK^\top$, token-token adjacency map 은 official image-form PSD family 로 저장하지 않는다.

### IMPL-MODEL-SPIK-005

요구사항:
Spikformer observer 는 raw activation archive 를 만들지 않고, raw input PSD summary 를 accumulator 로 streaming 해야 한다.

적용 대상:
- `src/model/author_adapter_spikformer.py`
- `src/psd_analysis.py`
- `src/signal/family_spectral_analysis.py`

수용 기준:
- official capture family 는 `x_probe` 로 제한한다.
- analysis checkpoint epoch 에서 full raw token archive 를 저장하지 않는다.
- `x_embed`, `x_tok`, `x_layer`, `y_mem`, `y_spike` 는 이 official image-form PSD profile 의 필수 산출물이 아니다. 필요하면 appendix object 로 별도 명세한다.
- pair distance table 에 Spikformer internal token family 를 official comparison row 로 넣지 않는다.

### IMPL-MODEL-SPIK-006

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
- `model_profile = spikformer` 를 기록한다.
- `paper_experiment`, `origin_code_path`, `origin_factory_name`, `origin_class_name`, `origin_train_entrypoint`, `paper_setting`, `structure_variation = none` 을 기록한다.
- generic `spiking_transformer` label 만 저장하지 않는다.
- analysis CSV 를 주 산출물로 저장하고, PNG 는 `plotting.py` 단계에서만 생성한다.

### IMPL-MODEL-SPIK-007

요구사항:
Need High Max-Former/MS-QKFormer 비교는 Spikformer official diversity profile 이 아니라 independent reinterpretation entry point 로 실행해야 한다.

적용 대상:
- `Spec/impl/spec/reinterpretation.md`
- `src/reinterpretation/`

수용 기준:
- Need High output root 는 core Spikformer output root 와 분리한다.
- Max-Former/MS-QKFormer 를 `spikformer` metadata 로 기록하지 않는다.

## 2026-05-02 수정 고정

- `capture_hidden=True` 일 때 transformer block output 을 hidden record 로 수집한다.
- block output 은 `signal_kind = feature`, `series = block_output` 으로 기록한다.
- PSD regularization 및 PSD analysis 는 이 block output feature track 을 사용할 수 있다.
