# fixed CNN-SNN 구현 명세

## 1. 범위

이 문서는 `Spec/theory/models/cnn.md` 를 구현으로 연결한다. CNN 실험은 architecture topology 를 고정하고 spiking neuron family 만 LIF 또는 RF 로 교체한다.

## 2. 구현 요구사항

### IMPL-MODEL-CNN-001

요구사항:
CNN model registry 는 공식 CNN full token 을 `<vgg11|resnet18>_<lif|rf>_<soft|hard>_<fixed|train>` 형식으로 제한해야 한다. architecture-only, profile-only, neuron-only 실행 식별자는 두지 않는다.

적용 대상:
- `src/model/model_registry.py`
- `src/model/snn_builder.py`
- `src/neurons/cnn2d.py`

수용 기준:
- CLI 또는 config 에서 임의 CNN depth, 임의 kernel ablation, LeNet token 을 공식 CNN experiment 로 선택할 수 없게 한다.
- 별도 family/profile alias 는 공식 실행 token 으로 사용하지 않는다.
- 공식 token 은 architecture, neuron, reset, threshold 를 모두 명시한다.
- 새 CNN topology 가 필요하면 이 문서와 theory model document 를 먼저 개정한다.

### IMPL-MODEL-CNN-002

요구사항:
CNN experiment 의 full token 안에서 neuron field 는 `lif` 또는 `rf` 만 허용한다.

적용 대상:
- `src/neurons/LIF_neuron.py`
- `src/neurons/RF_neuron.py`
- `src/model/snn_builder.py`
- `bash/model_training.sh`, `bash/psd_analysis.sh`

수용 기준:
- 같은 dataset/architecture/reset/threshold/seed scenario 안에서 변경되는 구조 축은 neuron family 뿐이다.
- threshold, reset, recurrence, pooling, residual operator 는 neuron family 비교 중에 별도 sweep 하지 않는다.
- threshold 와 reset profile 은 full model token 에 포함되지만, LIF/RF 비교축과 동시에 바꾸지 않는다.

### IMPL-MODEL-CNN-003

요구사항:
ResNet-18 BasicBlock 은 second conv branch와 shortcut projection을 current-domain에서 더한 뒤 post-add spiking activation에 통과시킨다. PSD capture point `residual_add` 는 branch-only 신호가 아니라 완전한 잔차 연결 이후 신호를 기록해야 한다.

적용 대상:
- `src/model/snn_builder.py`
- `src/neurons/cnn2d.py`

수용 기준:
- residual output 은 `post_spike(branch_current + shortcut_current)` 의미론을 따른다.
- `residual_add / layer_input` 은 `branch_current + shortcut_current` 이다.
- `residual_add / membrane` 과 `residual_add / spike` 는 위 residual-add 입력에 대한 post-add spiking activation 결과다.
- `iter_named_layers()` 와 trace record 이름은 모두 `*_residual_add` 를 사용한다.

### IMPL-MODEL-CNN-004

요구사항:
CNN effective input capture 는 raw image diagnostic 과 layer effective input 을 분리하고, image-form signal 은 timestep 을 보존한 row-time view 로 accumulator 에 전달해야 한다.

적용 대상:
- `src/psd_analysis.py`
- `src/signal/family_spectral_analysis.py`

수용 기준:
- raw flatten image PSD 는 `x_probe` 또는 dataset diagnostic 으로 기록된다.
- `[B,T,C,H,W]` signal 은 `[B,R=C*H*W,T]` logical view 로 해석한다.
- static repeat input 의 timestep 을 제거하지 않는다.
- VGG-style backbone 에서는 pooling 이후 stage signal 과 classifier-adjacent `z_front` 를 official image-form capture point 로 둔다.
- ResNet-style backbone 에서는 shortcut 이 더해진 `residual_add` post-activation output, stage endpoint, classifier-adjacent `z_front` 를 official image-form capture point 로 둔다.
- CNN layer 에 실제로 들어가는 representation 은 `z_front` 또는 해당 layer 의 `x_layer` 로 기록된다.
- 두 family 를 같은 pair source 로 합치지 않는다.
- PSD analysis 는 CNN model input physical layout 을 고정 `(R, T)` shape 로 바꾸도록 요구하지 않고, axis metadata 로 logical PSD view 를 해석한다.
- official numeric artifact 는 analysis CSV 이며 PNG 는 `plotting.py` 단계에서만 생성한다.

### IMPL-MODEL-CNN-005

요구사항:
CNN front-end 뒤에 dense SNN tail 을 붙이는 경우 tail 구조와 main analysis role 은 `Spec/impl/spec/model_dense.md` 를 따라야 한다.

적용 대상:
- `src/model/model_registry.py`
- `src/model/snn_builder.py`
- `src/psd_analysis.py`

수용 기준:
- CNN feature extractor 는 `analysis_role = auxiliary_or_frontend` 로 기록할 수 있지만, main target 을 대체하지 않는다.
- CNN-dense SNN 또는 CNN-recurrent dense SNN run 의 tail 은 `hidden_spec` 과 `neuron_token` 으로 구성되며 `Spec/impl/spec/model_dense.md` 의 slot contract 를 따른다.
- CNN effective input 은 `z_front` 또는 dense SNN 첫 layer `x_layer` 로 기록한다.
- output metadata 는 CNN full token 과 dense SNN tail full token 을 모두 기록한다.


### IMPL-MODEL-CNN-006

요구사항:
fixed CNN full token 을 사용할 때 `hidden_spec` 은 dense tail width 로 해석하지 않는다.

적용 대상:
- `src/model/model_registry.py`
- `src/model/snn_builder.py`
- `bash/model_training.sh`

수용 기준:
- CNN full token 과 함께 들어온 `hidden_spec` 값 `-`, empty string, `default` 는 모두 `None` 으로 정규화한다.
- fixed CNN builder 에 dataset default hidden size 를 전달하지 않는다.
- CNN backbone 구조는 full token 의 architecture field 로만 결정한다.
- CNN-dense tail 이 필요한 별도 실험은 fixed CNN full token 이 아니라 tail 명세가 포함된 별도 token 과 별도 문서로 정의한다.

## 2026-05-02 수정 고정

- CNN 계열 모델 입력은 `(B,T,C,H,W)` 만 허용한다. rank-3 flattened input 의 자동 reshape 는 금지한다.
- VGG 계열 trace 에서 `layer_input` 과 `membrane` 은 pooling 이전 신호로 기록한다. `spike` 만 pooling 이후 layer output 으로 기록한다.
- MNIST 는 새 실험 preset 에서 제외한다. MNIST 관련 코드 자체는 기존 산출물 호환을 위해 삭제하지 않는다.
