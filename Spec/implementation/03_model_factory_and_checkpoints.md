# Model Factory and Checkpoint Contract

## Checkpoint schema

`model_training.py`가 저장하는 checkpoint는 최소한 다음 정보를 포함해야 한다.

```python
{
  "schema_version": "psd_checkpoint_v1",
  "checkpoint_schema_version": "psd_checkpoint_v1",  # 호환 alias
  "checkpoint_format": "state_dict_payload",
  "state_dict_key_format": "unwrapped_eager",
  "epoch": int,
  "state_dict": Mapping[str, Tensor],
  "model_token": str,
  "model_family": str,
  "readout_mode": str,
  "dataset_token": str,
  "seed": int,
  "model_config": {...},
  "readout_config": {...},
  "training_args": {...},
  "axis_metadata_ref": {...},
  "metric_snapshot": {...}
}
```

`state_dict`는 반드시 wrapper가 제거된 실제 모델의 key만 저장한다. 따라서 key는 `module.` 또는 `_orig_mod.` prefix를 갖지 않아야 한다. 저장 시에도 key별 prefix 정규화를 수행하고, 로더도 구버전 checkpoint를 위해 이 prefix들을 key별로 반복 제거한다.

## Wrapper 제거 순서

DDP와 `torch.compile`이 동시에 사용될 때 wrapper가 중첩될 수 있다.

\[
\text{DDP}(\text{OptimizedModule}(M)) \quad\text{또는}\quad \text{OptimizedModule}(\text{DDP}(M)).
\]

프로젝트는 다음 속성을 반복적으로 따라 실제 모델을 찾는다.

1. `.module` (DDP)
2. `._orig_mod` (`torch.compile`의 OptimizedModule)

이 절차는 `src/util/checkpoints.py::unwrap_model`에 고정된다.

## 로드 호환성

분석 entrypoint는 다음 순서로 모델을 복원한다.

1. `model_token` 또는 `training_args.model`로 model spec을 canonicalize한다.
2. `model_config`에서 `input_dim`, `sequence_length`, `num_classes`, `input_shape`, `hidden_spec`, `v_th`를 읽는다.
3. `readout_config` 또는 `readout_mode`로 readout을 만든다.
4. `state_dict`, `model_state_dict`, `model` key 중 첫 번째 mapping을 읽고 prefix를 정규화한다.
5. DDP/`torch.compile` prefix를 key별로 제거하고, 구버전 VGG first-layer alias 및 pure-Torch LIF/RF runtime buffer 차이는 호환 대상으로 처리한다. 그 외 실제 parameter mismatch는 명시적으로 missing/unexpected key를 보고한다.

이 계약은 `psd_analysis`, `2d_fft_analysis`, `element_psd`, `element_fft`, `checkpoint_accuracy_eval_plot`에 공통 적용된다.

## CNN backbone 고정

`resnet`/`vggnet` 계열 model token은 reference/SNNs에서 **reference 입력 front-end나 데이터셋 가정은 가져오지 않고**, topology와 잔차 결합 의미만 가져온다. 입력 채널 수, 시간 프레임 수, 이미지 해상도는 항상 `prep_data` manifest의 `input_shape`가 결정한다. 따라서 MNIST류 정적 이미지, CIFAR류 정적 이미지, N-MNIST/CIFAR10-DVS/DVS-Gesture128 같은 frame sequence를 같은 builder가 처리한다.

- ResNet: `reference/SNNs/sew_resnet.py`에서 ResNet-18 BasicBlock 개수 `[2,2,2,2]`와 SEW connection function `ADD`만 사용한다. reference의 데이터셋 전용 입력 front-end는 사용하지 않는다. 첫 BasicBlock이 prepared frame channel을 직접 받아 필요하면 shortcut projection으로 channel/stride를 맞춘다.
- VGG: `reference/SNNs/vgg_snn.py`에서 VGG-11 channel/pool schedule `[64, M, 128, M, 256, 256, M, 512, 512, M, 512, 512, M]`만 사용한다. 입력 channel/resolution은 prepared bundle metadata에 따른다. 작은 정적 이미지가 pool stack에서 붕괴하지 않도록 max-pool은 ceil-mode를 사용하고, head는 adaptive average pooling으로 고정한다.
- 구현 경로: 공식 builder는 project-native `FixedCNN2DClassifier`와 `CNN2DResidualBlock`을 사용한다. reference는 VGG-11 channel/pool schedule, SEW-ResNet18 BasicBlock 개수 `[2,2,2,2]`, SEW residual merge(`ADD`)의 의미만 제공한다. 입력 channel·time·resolution은 `prep_data` manifest에서 오며, spiking neuron runtime은 compile 대상인 project pure-Torch CNN2D LIF/RF 계층이다. `SNNBenchCNNAdapter`는 보조 경로로 남기되 기본 경로로 사용하지 않는다.

PSD/FFT 분석은 이러한 topology를 layer name과 hidden trace schema의 기준으로 사용한다. VGG와 SEW-ResNet 입력은 prepared frame을 직접 사용하며, 별도의 Poisson 재인코딩을 수행하지 않는다. checkpoint metadata에는 `reference_backbone_contract`, `cnn_lif_backend`, `sew_resnet_connect_function`, `resnet_input_projection`, `input_policy`가 기록되어 분석 시 같은 구조로 복원된다.

