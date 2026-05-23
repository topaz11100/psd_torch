# 스파이킹 젤리 모델 변경

**User:** Anonymous  
**Created:** 5/22/2026 21:15:16  
**Updated:** 5/22/2026 22:24:14  
**Exported:** 5/23/2026 2:24:07  
**Link:** [https://chatgpt.com/c/6a1048cf-dfec-8321-bfcd-3bfe0ed15bf5](https://chatgpt.com/c/6a1048cf-dfec-8321-bfcd-3bfe0ed15bf5)  

## Prompt:
2026. 5. 22. 오후 9:15:11

저기 나와있는 모델들 "전부" 스파이킹 젤리 기반으로 싹다 바꿔버릴거야 가능한가?

[Attachment: psd(58).zip]

## Response:
2026. 5. 22. 오후 9:15:16

먼저 업로드된 코드의 모델 정의와 학습 구조를 확인해서, 전부 SpikingJelly로 바꾸는 일이 모델 래퍼 교체 수준인지 입력 인코딩·loss·학습 루프까지 바꿔야 하는지 판단하겠습니다.

## Response:
2026. 5. 22. 오후 9:15:36

확인 결과, 이미 일부는 SpikingJelly를 쓰지만 핵심 LIF/RF 계열과 SpikeGRU 등은 순수 PyTorch 또는 래퍼라서 신경동역학을 보존한 포팅이 필요합니다.

## Response:
2026. 5. 22. 오후 9:19:34

> **Model 전환과 상태 점검 정리**
>
> 현재 author_adapter_state_space에서 `_psd_origin_state_space_ss4d` 모듈을 가져오는 문제와 `sys.path` 설정 안 되어 있어 실패할 가능성이 있습니다. 전환에 해당 의존성 문제는 중요하지만, 간단한 변환은 불가능합니다. 모델 목록은 다음과 같습니다:
> 
> - Dense: LIF, RF
> - CNN: VGG, ResNet
> - Auxiliary: SpikingSSM, Spikformer 등
>
> 9m 52s 동안 생각함

## Response:
2026. 5. 22. 오후 9:25:09

가능합니다. 다만 정확히 말하면:

**"전부 SpikingJelly 기반으로 통합" 은 가능** 합니다.  
**"전부 SpikingJelly 내장 뉴런으로 단순 치환" 은 불가능하거나 연구 의미가 깨집니다.**

이 차이가 핵심입니다.

---

# 결론

압축 파일 기준으로 보면 모델들은 크게 아래처럼 나뉩니다.

| 구분 | SpikingJelly 전환 가능성 | 판단 |
|---|---:|---|
| dense LIF / recurrent LIF | 가능 | 비교적 쉬움. 단, 현재 코드의 learnable alpha, threshold, reset 의미를 보존하려면 custom node 필요 |
| dense RF / recurrent RF | 가능 | 내장 뉴런으로는 안 됨. RF dynamics를 SpikingJelly-style custom module로 만들어야 함 |
| VGG11 / ResNet18 LIF-RF CNN | 가능 | 이미 `src/model/snnbench_*` 쪽에서 SpikingJelly를 쓰고 있음. 정리/통합 대상 |
| DH-SNN | 가능 | 단순 치환 불가. dendritic branch dynamics를 custom module로 포팅해야 함 |
| D-RF | 가능 | 단순 치환 불가. BiRF/D-RF 필터 구조를 custom module로 포팅해야 함 |
| TC-LIF / TS-LIF | 가능 | 원본 node wrapper를 SpikingJelly `MemoryModule` 또는 `BaseNode` 스타일로 정리하면 됨 |
| Spikformer | 가능 | 원본이 구버전 `clock_driven` 를 쓰므로 `activation_based` 로 마이그레이션 필요 |
| SpikingSSM | 가능 | SSM 구조 자체는 그대로 두고, spike/state 뉴런 부분을 SpikingJelly-style로 감싸야 함 |
| SpikeGRU | 가능하지만 선행 수정 필요 | 현재 `src/model/snn_builder.py` 안의 `SpikGRUCellBlock.step()` 가 구현 중간에서 끊겨 있음. 이건 먼저 고쳐야 함 |
| first-spike readout | 가능 | 이미 SpikingJelly 버전 원본 폴더가 있음. 다만 readout/loss는 뉴런이 아니라 판독 방식이라 별도 관리가 맞음 |

즉, **전부 가능은 맞습니다.**  
하지만 방법은 "LIFNode로 다 바꾸자" 가 아니라, **SpikingJelly 프레임워크 위에 각 뉴런모델을 custom node로 재구현** 하는 쪽이어야 합니다.

---

# 왜 단순 치환이 안 되나

현재 프로젝트의 핵심은 "모델 구조" 보다 **뉴런 동역학 차이** 입니다.

예를 들어:

- LIF는 1차 누적/누설입니다.
- RF는 공진(resonance) 상태가 있고, 내부적으로 $x$, $y$ 같은 2차 상태를 씁니다.
- D-RF는 filter/branch 구조가 들어갑니다.
- DH-SNN은 dendritic branch와 branch별 time constant가 중요합니다.
- TC-LIF, TS-LIF는 시간상수 또는 dual-timescale 성질이 핵심입니다.
- SpikeGRU는 gate, current, membrane, spike가 얽힌 recurrent cell입니다.
- SpikingSSM은 state-space kernel 자체가 모델 정체성입니다.

이걸 전부 `neuron.LIFNode` 나 `MultiStepLIFNode` 로 바꾸면, 겉으로는 SNN이지만 **원래 비교하려던 뉴런모델 차이가 사라집니다.** 그러면 PSD 분석 결과도 논문/연구 질문과 맞지 않게 됩니다.

따라서 목표는 이렇게 잡아야 합니다.

> "모든 모델을 SpikingJelly의 실행/상태관리/모니터링/서로게이트 체계에 맞춘다.  
> 단, 각 뉴런모델의 수식과 상태 업데이트는 보존한다."

SpikingJelly 자체도 PyTorch 기반 SNN 프레임워크이고, custom neuron 확장을 위한 `BaseNode` 계열을 제공합니다. 문서에서도 core neuron module과 base class를 통해 뉴런을 확장할 수 있는 구조를 설명합니다. ([GitHub](https://github.com/fangwei123456/spikingjelly))

---

# 현재 코드에서 특히 주의할 부분

## 1. 이미 SpikingJelly가 들어간 부분이 있음

`src/model/snnbench_backbones.py` 와 `src/model/snnbench_adapter.py` 는 이미 `spikingjelly.activation_based` 를 사용합니다.

즉, CNN 계열은 "새로 바꾸는 작업" 이라기보다:

- 현재 wrapper 정리
- RF custom node를 더 깔끔하게 SpikingJelly `MemoryModule` 스타일로 정리
- reset 처리 통일
- trace capture 통일

이쪽에 가깝습니다.

SpikingJelly에서는 stateful module을 batch 처리 후 reset해야 하고, `functional.reset_net` 를 사용하는 방식이 문서화되어 있습니다. 이 부분은 현재 프로젝트의 PSD capture와 학습 루프에서도 반드시 일관되게 맞춰야 합니다. ([SpikingJelly](https://spikingjelly.readthedocs.io/zh-cn/latest/tutorials/en/basic_concept.html))

---

## 2. Spikformer는 구버전 API 문제 있음

압축 파일 안의 `Origin/spikformer/.../model.py` 쪽은 `spikingjelly.clock_driven` 를 씁니다. 그런데 SpikingJelly는 구버전의 `clock_driven`, `event_driven` 계열이 `activation_based` 등으로 바뀐 흐름이 있습니다. 공식 README도 구버전 모듈 rename을 명시합니다. ([GitHub](https://github.com/fangwei123456/spikingjelly))

그래서 Spikformer는 단순히 import만 바꾸는 게 아니라:

```python
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from spikingjelly.clock_driven import layer, surrogate, functional
```

이런 부분을 현재식으로 바꾸고, `step_mode`, 입력 shape, reset 방식, trace 기록 방식을 확인해야 합니다.

---

## 3. 버전 고정이 먼저 필요함

압축 파일 안에는 서로 다른 원본 코드들이 섞여 있습니다.

- `Origin/need-high/requirements.txt`: `spikingjelly==0.0.0.0.12`
- `Origin/readout/first_spike/requirements.txt`: `spikingjelly==0.0.0.0.14`

반면 현재 SpikingJelly GitHub README 기준 최신 개발 흐름은 Python과 PyTorch 요구조건이 더 올라가 있습니다. README에는 최신 버전이 `python>=3.11`, `torch>=2.6.0` 를 요구하고 `torch==2.7.1` 에서 테스트되었다고 되어 있습니다. ([GitHub](https://github.com/fangwei123456/spikingjelly))

그래서 먼저 정해야 합니다.

| 선택 | 장점 | 단점 |
|---|---|---|
| 구버전 SpikingJelly 유지 | 원본 코드 호환 쉬움 | `clock_driven` 구식 API, 최신 기능/문서와 불일치 |
| 최신 `activation_based` 로 통일 | 장기 유지보수 좋음 | Spikformer, first-spike, old origin code 마이그레이션 필요 |

내 판단은 **최신 `activation_based` 기준으로 통일하는 쪽** 이 맞습니다.  
이 프로젝트는 단순 재현 코드가 아니라 PSD 분석 플랫폼이므로, 오래된 원본 API에 계속 끌려가는 구조는 나중에 더 큰 문제가 됩니다.

---

# 권장 전환 방식

## 1단계: "공식 실험 경로" 만 먼저 바꾸기

압축 파일의 `Origin/` 전체를 전부 건드리면 일이 너무 커집니다. `Origin/` 은 원본 보관소에 가깝고, 실제 프로젝트 실행 경로는 주로 아래입니다.

```text
src/model/
src/neurons/
src/readout/
src/model_training.py
src/psd_analysis.py
bash/model_training*.sh
bash/psd_analysis*.sh
```

따라서 먼저 바꿀 대상은 이것입니다.

```text
src/neurons/LIF_neuron.py
src/neurons/RF_neuron.py
src/neurons/DH_SNN_neuron.py
src/neurons/D_RF_neuron.py
src/neurons/TC_LIF_neuron.py
src/neurons/TS_LIF_neuron.py
src/neurons/cnn2d.py
src/model/snn_builder.py
src/model/snnbench_backbones.py
src/model/author_adapter_spikformer.py
src/model/author_adapter_state_space.py
```

`Origin/` 은 reference로 남기고, 필요한 수식만 현재 코드 쪽으로 옮기는 게 안전합니다.

---

## 2단계: 공통 interface 만들기

전부 SpikingJelly 기반으로 바꾸려면 각 모델마다 따로 땜질하면 안 됩니다. 공통 interface가 필요합니다.

예를 들면 이런 식입니다.

```python
class PSDSpikingLayer(...):
    def reset(self):
        ...

    def forward(self, x, return_traces=False):
        # return: membrane_seq, spike_seq
        ...

    def get_last_layer_input(self):
        ...

    def filter_stats_vectors(self):
        ...
```

그리고 모든 뉴런 계열이 같은 출력을 내야 합니다.

```text
input_sequence: (B,T,C_in)
membrane_seq:   (B,T,C_out)
spike_seq:      (B,T,C_out)
layer_input:    (B,T,C_out)
```

CNN이면:

```text
input_sequence: (B,T,C,H,W)
membrane_seq:   (B,T,C_out,H_out,W_out)
spike_seq:      (B,T,C_out,H_out,W_out)
layer_input:    (B,T,C_out,H_out,W_out)
```

이걸 지켜야 PSD 분석 코드가 안 깨집니다.

---

## 3단계: 모델별 포팅 난이도

| 모델 | 추천 포팅 방법 |
|---|---|
| LIF | SpikingJelly `LIFNode` 근사 가능. 하지만 현재 코드와 완전 동일하게 하려면 custom LIF node |
| RF | custom `RFNode` 필수 |
| D-RF | custom `DRFNode` 또는 현재 `BiRFModel` wrapper를 SpikingJelly `MemoryModule` 스타일로 감싸기 |
| DH-SNN | branch state를 가진 custom `DHSNNNode` |
| TC-LIF | 기존 원본 node를 SpikingJelly-style로 재작성 |
| TS-LIF | short/long timescale state를 보존한 custom node |
| CNN LIF | 현재 `snnbench_backbones.py` 기반 정리 |
| CNN RF | 현재 `SNNBenchRFNode` 를 SpikingJelly memory module로 정리 |
| Spikformer | `clock_driven` 에서 `activation_based` 로 전환 |
| SpikingSSM | SSM block은 유지, spike neuron/state 부분만 SpikingJelly-style |
| SpikeGRU | 먼저 `step()` 완성. 이후 custom recurrent spiking cell로 포팅 |

---

# 가장 큰 위험 3개

## 1. PSD 결과가 바뀔 수 있음

SpikingJelly로 바꿀 때 수식이 조금만 달라도 PSD가 달라집니다.

예를 들어 현재 LIF는 대략 이런 형태입니다.

$$
v_t=\alpha v_{t-1}+I_t
$$

이걸 SpikingJelly 기본 `LIFNode` 의 `tau`, `decay_input`, `v_reset` 설정으로 바꿨을 때 정확히 같은 업데이트가 되는지 검증해야 합니다. 안 맞으면 PSD 곡선이 달라지고, "프레임워크 전환 효과" 와 "뉴런모델 효과" 가 섞입니다.

그래서 각 포팅 후에는 반드시 아래를 비교해야 합니다.

```text
old_layer(input) vs new_layer(input)

1. membrane_seq shape 동일
2. spike_seq shape 동일
3. layer_input shape 동일
4. 같은 seed에서 numerical difference 허용범위 내
5. readout score 동일 또는 거의 동일
6. PSD curve 차이가 충분히 작음
```

---

## 2. SpikeGRU는 지금 그대로는 변환 이전에 구현 보완이 필요함

`src/model/snn_builder.py` 의 `SpikGRUCellBlock.step()` 는 현재 `z_t`, `drive_t` 까지만 계산하고 실제 return이 없습니다. 따라서 `spikegru` 를 "전부 SpikingJelly 기반" 으로 바꾸려면, 먼저 이 cell 수식을 완성해야 합니다.

이건 단순 프레임워크 문제가 아니라 **현재 구현 완성도 문제** 입니다.

---

## 3. reset과 trace capture가 핵심임

SpikingJelly에서는 stateful SNN을 다룰 때 reset을 빼먹으면 학습/추론 결과가 잘못되거나 backward 오류가 날 수 있습니다. 공식 문서도 batch 처리 후 `functional.reset_net` 를 호출하는 예를 강조합니다. ([SpikingJelly](https://spikingjelly.readthedocs.io/zh-cn/latest/tutorials/en/basic_concept.html))

PSD 분석에서는 더 중요합니다.  
왜냐하면 이전 batch의 membrane state가 다음 batch에 남으면 PSD가 오염됩니다.

또 SpikingJelly에는 output이나 membrane attribute를 기록하는 monitor 기능이 있지만, 현재 프로젝트는 `layer_input`, `membrane`, `spike`, `i_current`, `z_gate` 같은 family별 기록이 필요합니다. 따라서 기본 monitor만 쓰기보다는 프로젝트 전용 trace interface를 유지하는 게 맞습니다. SpikingJelly 문서도 output monitor와 attribute monitor를 제공한다고 설명합니다. ([SpikingJelly](https://spikingjelly.readthedocs.io/zh-cn/latest/tutorials/en/monitor.html))

---

# 내가 추천하는 최종 구조

```text
src/
  sj_layers/
    base.py
    lif.py
    rf.py
    dh_snn.py
    d_rf.py
    tc_lif.py
    ts_lif.py
    spikegru.py
    cnn_nodes.py

  model/
    snn_builder.py
    model_registry.py
    author_adapter_spikformer.py
    author_adapter_state_space.py

  readout/
    readout.py
```

그리고 기존 `src/neurons/` 는 점진적으로 deprecated 처리합니다.

새 구조의 원칙은 이겁니다.

```text
1. 모든 spiking layer는 reset() 을 가진다.
2. 모든 spiking layer는 return_traces=True 를 지원한다.
3. 모든 layer는 membrane_seq, spike_seq, layer_input_seq 를 같은 convention으로 반환한다.
4. 모델 registry token은 그대로 둔다.
5. SpikingJelly built-in neuron은 LIF처럼 의미가 맞는 경우에만 쓴다.
6. RF/D-RF/DH/TC/TS/SpikeGRU/SSM은 custom SpikingJelly-style module로 만든다.
```

---

# 최종 판단

**가능합니다.**  
그리고 연구 코드 장기 유지보수 관점에서는 하는 게 맞습니다.

다만 "싹 다 SpikingJelly 내장 뉴런으로 갈아끼우기" 는 하면 안 됩니다.  
그렇게 하면 RF, D-RF, DH-SNN, TC-LIF, TS-LIF, SpikeGRU, SpikingSSM의 고유 동역학이 사라져서 PSD 분석 연구의 의미가 깨집니다.

해야 할 방향은 이것입니다.

> **SpikingJelly 기반 통합 + custom neuron dynamics 보존 + PSD trace interface 고정**

추가로, 내가 현재 실행 환경에서 확인했을 때 `spikingjelly` 패키지는 설치되어 있지 않았습니다. 그래서 여기서는 import-level 실행 검증은 못 했고, 압축 파일의 코드 구조와 공식 SpikingJelly 문서 기준으로 판단했습니다.

---

**Sources:**

- [GitHub - fangwei123456/spikingjelly: SpikingJelly is an open-source deep learning framework for Spiking Neural Network (SNN) based on PyTorch. · GitHub](https://github.com/fangwei123456/spikingjelly)
- [Basic Conception — spikingjelly  文档](https://spikingjelly.readthedocs.io/zh-cn/latest/tutorials/en/basic_concept.html)

## Prompt:
2026. 5. 22. 오후 10:12:36

SpikingJelly 프레임워크 위에 각 뉴런모델을 custom node로 재구현 <- 당연히 내 의도는 이거야

## Response:
2026. 5. 22. 오후 10:12:40

방향은 확정됐으니, 이제 각 뉴런 수식은 보존하면서 SpikingJelly의 상태관리, step mode, reset, monitor 체계로 감싸는 설계 단위로 좁히겠습니다.

## Response:
2026. 5. 22. 오후 10:12:57

먼저 weight layer와 neuron node를 분리하고, `src/neurons/*`, `src/model/snn_builder.py`, CNN 어댑터, Spikformer/SSM adapter를 중심 실행 경로로 잡겠습니다.

## Response:
2026. 5. 22. 오후 10:17:07

> **Implementation of custom memory helper for RF**
>
> I can implement a helper function to handle memory as a scalar reset value before forward pass, like:
> 
> ```python
> if not isinstance(self.v, torch.Tensor):
>     self.v = torch.zeros_like(current)
> elif self.v.shape != current.shape:
>     # Handle reshaping
> ```
> 
> This matches SpikingJelly’s memory pattern, using scalar initialization. I'll include the RF update formulas as well:
> 
> \[
> x_t=\rho(\cos \phi x_{t-1}-\sin \phi y_{t-1})+\beta_x I_t
> \]
> \[
> y_t=\rho(\sin \phi x_{t-1}+\cos \phi y_{t-1})+\beta_y I_t
> \]
> 
> I’ll ensure I use correct math formatting as per the rules, and will clarify the spiking node's output, which should deliver both membrane signal and spike.
>
> 7m 44s 동안 생각함

## Response:
2026. 5. 22. 오후 10:20:24

맞다. 그 의도라면 **가능하고, 방향도 맞습니다.**  
이 경우 목표는 다음 한 문장으로 정리됩니다.

> 기존 뉴런모델의 수식은 보존하고, 실행 구조만 SpikingJelly의 `activation_based` 상태관리 체계로 통일한다.

즉, **LIFNode로 전부 갈아끼우는 게 아니라** , RF, D-RF, DH-SNN, TC-LIF, TS-LIF, SpikeGRU, SpikingSSM 각각을 **SpikingJelly-style custom node/block** 로 다시 쓰는 겁니다.

---

# 1. 핵심은 "custom node" 만이 아니라 "custom node + wrapper layer" 입니다

현재 네 코드의 뉴런 레이어들은 대부분 이런 구조입니다.

```text
input_sequence
→ input_weight / conv / recurrent_weight
→ neuron dynamics
→ spike_seq
→ membrane_seq
→ layer_input trace
```

SpikingJelly식으로 제대로 바꾸려면 이걸 둘로 나누는 게 좋습니다.

```text
synapse wrapper
→ Linear / Conv2d / recurrent input 계산

custom neuron node
→ membrane / spike / reset / internal state 계산
```

예를 들면:

```text
SJLinearSpikingLayer
  ├─ nn.Linear 또는 spikingjelly.activation_based.layer.Linear
  └─ CustomLIFNode / CustomRFNode / CustomTCNode / ...
```

CNN이면:

```text
SJConv2DSpikingLayer
  ├─ Conv2d / BatchNorm2d
  └─ CustomLIFNode / CustomRFNode
```

이 구조가 좋은 이유는, 나중에 `Linear`, `Conv2d`, `BatchNorm`, `ResidualBlock`, `TransformerBlock` 은 그대로 두고 뉴런만 갈아끼울 수 있기 때문입니다.

---

# 2. BaseNode만 쓰면 안 됩니다

여기서 중요한 판단이 하나 있습니다.

**모든 모델을 SpikingJelly의 `BaseNode` 로만 만들려고 하면 안 됩니다.**  
SpikingJelly의 `BaseNode` 는 기본적으로 "전압을 충전하고, threshold로 spike를 내고, reset하는" 미분가능 spiking neuron의 base class입니다. 문서상 `BaseNode` 는 `neuronal_charge`, `neuronal_fire`, `neuronal_reset`, `single_step_forward`, `multi_step_forward` 구조를 갖고, `step_mode='s'` 또는 `step_mode='m'` 를 지원합니다. 또한 `v_reset=None` 일 때 soft reset, `v_reset` 이 값일 때 hard reset 의미를 갖습니다. ([SpikingJelly](https://spikingjelly.readthedocs.io/zh-cn/latest/APIs/spikingjelly.activation_based.neuron.core.html))

그런데 네 모델들 중 일부는 단순 threshold neuron이 아닙니다.

| 모델 | 추천 base |
|---|---|
| LIF | `BaseNode` 가능 |
| RF | `BaseNode` 가능하지만 내부 state가 $x$, $y$ 이므로 override 필요 |
| TC-LIF | `BaseNode` 가능 |
| TS-LIF | `BaseNode` 가능 |
| DH-SNN | `BaseNode` 가능하지만 branch state가 많아서 `MemoryModule` 이 더 깔끔할 수 있음 |
| D-RF / BiRF | `MemoryModule` 또는 `nn.Module` 권장 |
| SpikeGRU | `MemoryModule` / custom recurrent block 권장 |
| SpikingSSM | `MemoryModule` / custom SSM block 권장 |
| Spikformer | 내부 neuron은 `BaseNode`, 전체 block은 transformer wrapper |

즉, 정확한 원칙은 이겁니다.

> thresholded membrane neuron은 `BaseNode` 로 만든다.  
> sequence convolution, SSM, GRU gate, dendritic multi-branch block은 `MemoryModule` 또는 일반 `nn.Module` 로 만든다.

SpikingJelly 문서에서도 `BaseNode` 자체가 `MemoryModule` 을 기반으로 하며, stateful module은 reset 관리가 필요합니다. ([SpikingJelly](https://spikingjelly.readthedocs.io/zh-cn/latest/APIs/spikingjelly.activation_based.neuron.core.html))

---

# 3. 네 코드에서 바꿔야 할 실제 대상

현재 압축 파일 기준으로 핵심 대상은 여기입니다.

```text
src/neurons/
  LIF_neuron.py
  RF_neuron.py
  DH_SNN_neuron.py
  D_RF_neuron.py
  TC_LIF_neuron.py
  TS_LIF_neuron.py
  cnn2d.py

src/model/
  snn_builder.py
  snnbench_backbones.py
  snnbench_adapter.py
  author_adapter_spikformer.py
  author_adapter_state_space.py
```

특히 현재 상태는 이렇게 보면 됩니다.

| 파일 | 현재 상태 | 바꿀 방향 |
|---|---|---|
| `LIF_neuron.py` | 자체 PyTorch 구현 | SpikingJelly `BaseNode` 기반 custom LIF |
| `RF_neuron.py` | 자체 PyTorch exact-ZOH RF | SpikingJelly `BaseNode` 또는 `MemoryModule` 기반 custom RF |
| `DH_SNN_neuron.py` | origin wrapper | wrapper 제거, 수식 직접 custom node화 |
| `D_RF_neuron.py` | origin `BiRFModel` wrapper | wrapper 제거, BiRF kernel/FFT/spike 경로 직접 module화 |
| `TC_LIF_neuron.py` | origin `TCLIFNode` wrapper | origin node 의존 제거, 수식 직접 구현 |
| `TS_LIF_neuron.py` | origin `TSLIFNode` wrapper | origin node 의존 제거, 수식 직접 구현 |
| `cnn2d.py` | 자체 Conv+LIF/RF | Conv wrapper + custom node로 분리 |
| `snn_builder.py` | 모든 family 빌드 담당 | 새 custom layer registry로 연결 |
| `snnbench_*` | 이미 SpikingJelly 일부 사용 | 새 custom node와 trace interface로 정리 |

---

# 4. 가장 먼저 정해야 할 convention

현재 프로젝트는 입력이 대체로 이렇게 생겼습니다.

```text
(B,T,C)
(B,T,C,H,W)
```

그런데 SpikingJelly multi-step 예제와 문서 convention은 보통 다음입니다.

```text
(T,N,C)
(T,N,C,H,W)
```

문서 예시도 `x_seq` 를 `[T,N,C]` 형태로 둡니다. ([SpikingJelly](https://spikingjelly.readthedocs.io/zh-cn/latest/tutorials/en/basic_concept.html))

그래서 둘 중 하나를 선택해야 합니다.

## 선택 A: 프로젝트 convention 유지

```text
외부 interface: (B,T,*)
내부 node 계산: for t in range(T)
```

장점:

- 기존 `psd_analysis.py`
- 기존 `LayerRecord`
- 기존 `readout`
- 기존 dataset output

거의 안 건드립니다.

단점:

- SpikingJelly의 `step_mode='m'` 를 완전히 활용하기 어렵습니다.

## 선택 B: SpikingJelly convention으로 통일

```text
외부 interface까지 (T,B,*)
```

장점:

- SpikingJelly 문서/모듈과 잘 맞습니다.
- `functional.set_step_mode(net,'m')` 를 쓰기 쉬움.

단점:

- 현재 프로젝트 전반의 shape 처리를 많이 고쳐야 합니다.

내 판단은 **A가 맞습니다.**  
즉, 프로젝트 외부 convention은 계속 `(B,T,*)` 로 두고, custom wrapper 내부에서만 필요할 때 transpose 하는 게 안전합니다.

---

# 5. 추천 디렉토리 구조

기존 `src/neurons/` 를 바로 갈아엎기보다 새 디렉토리를 만들고 연결하는 게 좋습니다.

```text
src/
  sj_custom/
    __init__.py
    surrogate.py
    base.py
    lif.py
    rf.py
    dh_snn.py
    d_rf.py
    tc_lif.py
    ts_lif.py
    spikegru.py
    ssm.py
    layers.py
    trace.py

  model/
    snn_builder.py
    model_registry.py
```

역할은 이렇게 나눕니다.

```text
surrogate.py
  - project surrogate spike 정의

base.py
  - trace 가능한 MemoryModule/BaseNode 공통 class

lif.py
  - CustomLIFNode

rf.py
  - CustomRFNode

tc_lif.py
  - CustomTCLIFNode

ts_lif.py
  - CustomTSLIFNode

dh_snn.py
  - CustomDHSNNNode

d_rf.py
  - BiRFKernel, CustomDRFBlock

spikegru.py
  - CustomSpikeGRUCell

layers.py
  - LinearSpikingLayer
  - RecurrentLinearSpikingLayer
  - Conv2DSpikingLayer
```

---

# 6. 공통 trace interface는 반드시 고정해야 합니다

PSD 분석 때문에 trace interface가 제일 중요합니다.  
모든 뉴런이 최소한 아래를 똑같이 내야 합니다.

```python
membrane_seq
spike_seq
layer_input_seq
```

권장 contract는 이것입니다.

```text
Dense:
  input_sequence:   (B,T,C_in)
  layer_input_seq:  (B,T,C_out)
  membrane_seq:     (B,T,C_out)
  spike_seq:        (B,T,C_out)

CNN:
  input_sequence:   (B,T,C_in,H,W)
  layer_input_seq:  (B,T,C_out,H_out,W_out)
  membrane_seq:     (B,T,C_out,H_out,W_out)
  spike_seq:        (B,T,C_out,H_out,W_out)
```

그리고 모델별 추가 trace는 선택적으로 붙입니다.

| 모델 | 추가 trace |
|---|---|
| RF | `x_state`, `y_state`, `center_frequency`, `damping` |
| DH-SNN | `branch_input`, `branch_state`, `soma_membrane` |
| D-RF | `kernel`, `filtered_signal`, `raw_membrane` |
| TC-LIF | `v1`, `v2` |
| TS-LIF | `v_s`, `v_l`, `s_s`, `s_l` |
| SpikeGRU | `i_current`, `z_gate`, `hidden`, `spike` |
| SpikingSSM | `ssm_state`, `filtered_signal`, `spike` |

SpikingJelly에는 output/attribute monitor가 있지만, 네 프로젝트는 `layer_input`, `membrane`, `spike`, `i_current`, `z_gate` 같은 family별 trace가 필요하므로 기본 monitor만으로는 부족합니다. 문서상 monitor는 특정 layer의 output이나 attribute 기록용이고, 기록은 `.records` 에 쌓이는 구조입니다. ([SpikingJelly](https://spikingjelly.readthedocs.io/zh-cn/latest/tutorials/en/monitor.html))

따라서 monitor는 보조 도구로 쓰고, **PSD용 trace는 네 custom layer가 직접 반환** 하는 게 맞습니다.

---

# 7. 공통 base skeleton

이런 식의 base를 두면 됩니다.

```python
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from spikingjelly.activation_based import base

class PSDTraceMemoryModule(base.MemoryModule):
    def __init__(self) -> None:
        super().__init__()
        self._trace_enabled = False
        self._trace: dict[str, list[torch.Tensor]] = {}

    def enable_trace(self, enabled: bool = True) -> None:
        self._trace_enabled = bool(enabled)
        self.clear_trace()

    def clear_trace(self) -> None:
        self._trace = {}

    def record_trace(self, name: str, value: torch.Tensor) -> None:
        if not self._trace_enabled:
            return
        self._trace.setdefault(name, []).append(value)

    def stacked_trace(self) -> dict[str, torch.Tensor]:
        return {
            key: torch.stack(values, dim=1).contiguous()
            for key, values in self._trace.items()
        }

    def filter_stats_vectors(self) -> dict[str, torch.Tensor]:
        return {}
```

이 base는 SpikingJelly의 state reset 체계에 들어갑니다. SpikingJelly 문서도 stateful module을 batch마다 reset해야 한다고 설명하고, `functional.reset_net(net)` 를 제공합니다. ([SpikingJelly](https://spikingjelly.readthedocs.io/zh-cn/latest/tutorials/en/basic_concept.html))

---

# 8. LIF custom node 방향

현재 네 LIF는 대략 이겁니다.

$$
v_t=\alpha v_{t-1}+I_t
$$

$$
s_t=H(v_t-\theta)
$$

soft reset이면:

$$
v_t \leftarrow v_t-\theta s_t
$$

hard reset이면:

$$
v_t \leftarrow v_t(1-s_t)
$$

이건 `BaseNode` 로 만들기 쉽습니다. 다만 SpikingJelly 내장 `LIFNode` 로 대체하면 현재 코드의 `alpha_bounds`, `trainable_threshold`, `emit_spike`, `reset_enabled`, PSD trace 의미가 어긋날 수 있습니다.

따라서 LIF도 직접 custom으로 만드는 게 맞습니다.

---

# 9. RF custom node 방향

현재 RF는 exact-ZOH resonate-and-fire입니다. 내부 상태가 $x$, $y$ 두 개입니다.

현재 코드의 핵심은 이 구조입니다.

$$
x_t=\rho(\cos\phi x_{t-1}-\sin\phi y_{t-1})+\beta_x I_t
$$

$$
y_t=\rho(\sin\phi x_{t-1}+\cos\phi y_{t-1})+\beta_y I_t
$$

$$
s_t=H(x_t-\theta)
$$

여기서 PSD 관점의 membrane은 보통 $x_t-\theta$ 입니다.

이건 `BaseNode` 로도 가능하지만, `v` 하나만 가진다고 생각하면 어색합니다.  
실제로는 아래처럼 두 state를 등록하는 쪽이 더 정확합니다.

```python
self.register_memory("x", 0.0)
self.register_memory("y", 0.0)
```

그리고 `self.v = self.x` 처럼 threshold 기준 전압만 노출하면 됩니다.

RF는 네 연구에서 제일 중요하므로, **SpikingJelly 내장 node를 쓰면 안 됩니다.**  
반드시 현재 exact-ZOH 식을 그대로 옮겨야 합니다.

---

# 10. TC-LIF custom node 방향

현재 origin TC-LIF의 핵심은 두 compartment입니다.

$$
v_{1,t}=v_{1,t-1}-\sigma(d_1)v_{2,t-1}+I_t
$$

$$
v_{2,t}=v_{2,t-1}+\sigma(d_2)v_{1,t}
$$

$$
s_t=H(v_{2,t}-\theta)
$$

soft reset에서는 대략:

$$
v_{1,t}\leftarrow v_{1,t}-\gamma s_t
$$

$$
v_{2,t}\leftarrow v_{2,t}-\theta s_t
$$

이건 `BaseNode` 로 잘 옮길 수 있습니다.

주의점은 하나입니다.  
현재 wrapper는 origin node의 `decay_factor` 를 그대로 쓰고, `surrogate_function` 만 프로젝트 surrogate로 바꿉니다. 포팅 후에도 `decay_factor`, `gamma`, `v_threshold`, `hard_reset` 의미를 그대로 보존해야 합니다.

---

# 11. TS-LIF custom node 방향

현재 TS-LIF는 short/long segment 비슷한 두 전압 state를 씁니다.

핵심 형태는 이렇습니다.

$$
v_{s,t}=a_0v_{s,t-1}+a_1I_t-yv_{l,t-1}
$$

$$
v_{l,t}=a_2v_{l,t-1}+a_3I_t-kv_{s,t}
$$

그리고 spike가 두 개 나옵니다.

$$
s_s=H(v_s-\theta)
$$

$$
s_l=H(v_l-\theta)
$$

최종 출력은 origin 코드 기준으로:

$$
s=\alpha_s s_s+\alpha_l s_l
$$

이 점이 중요합니다.  
TS-LIF의 output은 반드시 binary spike가 아닐 수 있습니다. `alpha_s`, `alpha_l` 로 섞인 weighted spike입니다. 그러므로 이 모델은 단순한 "0/1 spike node" 라고 보면 안 됩니다.

SpikingJelly `BaseNode` 로 만들 수는 있지만, `neuronal_fire()` 를 그대로 쓰기보다는 `sl_neuronal_fire()` 를 따로 두고 custom `single_step_forward()` 를 쓰는 게 맞습니다.

---

# 12. DH-SNN custom node 방향

DH-SNN은 현재 origin wrapper라서 가장 먼저 구조를 이해하고 다시 써야 합니다.

현재 wrapper가 쓰는 핵심은:

```text
dense input
→ branch별 dendritic input d_input
→ branch sum
→ soma membrane
→ spike
```

수식으로 보면 대략:

$$
d_{t}=\beta d_{t-1}+(1-\beta)I_t
$$

$$
l_t=\sum_b d_{t,b}
$$

$$
v_t=\alpha v_{t-1}+(1-\alpha)R_ml_t
$$

$$
s_t=H(v_t-\theta)
$$

여기서 branch dimension이 중요합니다.

따라서 DH-SNN은 이렇게 설계하는 게 좋습니다.

```text
CustomDHSNNNode
  state:
    d_input: (B,C_out,B_branch)
    mem:     (B,C_out)
    spike:   (B,C_out)

  parameter:
    tau_n:   (C_out,B_branch) 또는 origin과 동일 shape
    tau_m:   (C_out)
    v_th:    (C_out)
```

이건 `BaseNode` 로도 가능하지만, branch trace가 중요해서 `MemoryModule` 에 직접 구현하는 쪽이 더 깔끔합니다.

---

# 13. D-RF custom block 방향

D-RF는 일반 time-step recurrent neuron이 아니라, 사실상 **SSM kernel convolution + spike** 구조입니다.

현재 origin `BiRFModel` 은 다음을 합니다.

```text
u: (B,H,L)
kernel k 생성
FFT convolution
y = irfft(rfft(u) * rfft(k))
y = y + D * u
s = spike(y.real - 1)
```

이건 `BaseNode` 보다 `MemoryModule` 또는 일반 `nn.Module` 로 구현하는 게 맞습니다.

권장 구조:

```text
CustomBiRFKernel
  - log_dt
  - C
  - log_A_real
  - A_imag

CustomDRFBlock
  - input_adapter
  - CustomBiRFKernel
  - D
  - surrogate spike
  - return membrane_seq, spike_seq
```

D-RF는 time-step loop로 억지 변환하지 않는 게 좋습니다.  
FFT kernel convolution이라는 모델 정체성을 유지해야 합니다.

---

# 14. SpikeGRU는 먼저 수식을 완성해야 합니다

현재 `src/model/snn_builder.py` 의 `SpikGRUCellBlock.step()` 는 구현이 중간에서 끊긴 상태입니다.  
`z_t`, `drive_t` 까지만 계산하고 실제 membrane/current/spike update와 return이 없습니다.

따라서 SpikeGRU 포팅 순서는 이렇습니다.

```text
1. 원 논문 또는 origin 구현 기준으로 step 수식 확정
2. 현재 PyTorch 버전에서 먼저 완성
3. regression test 작성
4. SpikingJelly MemoryModule로 이동
```

SpikeGRU는 `BaseNode` 로 만들지 않는 게 좋습니다. 이유는 output이 단순 thresholded membrane이 아니라 gate가 개입된 recurrent state이기 때문입니다.

---

# 15. Spikformer는 API 마이그레이션 문제

Spikformer 쪽은 뉴런 수식보다 API 문제가 큽니다.  
SpikingJelly는 `0.0.0.0.14` 부터 `clock_driven`, `event_driven` 같은 구 모듈명이 rename되었다고 문서에 명시되어 있습니다. ([SpikingJelly](https://spikingjelly.readthedocs.io/))

따라서 Spikformer는 다음 순서가 맞습니다.

```text
1. origin import 제거 또는 최소화
2. clock_driven import를 activation_based로 변경
3. MultiStepLIFNode 사용부 shape 확인
4. attention block trace hook 재작성
5. functional.reset_net 호출 위치 통일
```

Spikformer는 "custom neuron" 보다 "구버전 SpikingJelly 코드의 현대화" 에 가깝습니다.

---

# 16. reset 규칙은 무조건 엄격해야 합니다

이 부분은 대충 하면 PSD가 오염됩니다.

SpikingJelly 문서도 stateful module은 batch 처리 후 reset해야 하며, reset하지 않으면 inference 출력이 틀리거나 training backward error가 날 수 있다고 설명합니다. ([SpikingJelly](https://spikingjelly.readthedocs.io/zh-cn/latest/tutorials/en/basic_concept.html))

네 프로젝트에서는 더 엄격해야 합니다.

```python
from spikingjelly.activation_based import functional

for x, y in loader:
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    functional.reset_net(model)
```

PSD 분석에서도 마찬가지입니다.

```python
for x, y in psd_loader:
    functional.reset_net(model)
    result = model(x, capture_hidden=True)
    functional.reset_net(model)
```

**forward 전 reset, forward 후 reset 둘 다 넣는 게 안전합니다.**

---

# 17. 포팅 우선순위

내가 추천하는 순서는 이겁니다.

| 순서 | 대상 | 이유 |
|---:|---|---|
| 1 | LIF | 기준 모델. test baseline 만들기 쉬움 |
| 2 | RF | 네 연구 핵심. exact-ZOH 보존 검증 필요 |
| 3 | CNN LIF/RF | dense LIF/RF node 재사용 가능 |
| 4 | TC-LIF | origin wrapper 제거 난이도 중간 |
| 5 | TS-LIF | weighted spike 때문에 조금 더 주의 |
| 6 | DH-SNN | branch state 때문에 trace 설계 필요 |
| 7 | D-RF | FFT/SSM kernel이라 별도 block으로 설계 |
| 8 | SpikingSSM | D-RF와 비슷하게 SSM block화 |
| 9 | Spikformer | API migration + trace hook 문제 |
| 10 | SpikeGRU | 현재 step 미완성이라 마지막 |

---

# 18. equivalence test가 필수입니다

이 작업은 "돌아간다" 만 보면 안 됩니다.  
각 모델마다 기존 구현과 새 구현이 같은지 확인해야 합니다.

최소 test는 이렇게 해야 합니다.

```text
입력:
  same seed
  same random input
  same parameter copy

검사:
  old membrane_seq == new membrane_seq
  old spike_seq == new spike_seq
  old layer_input == new layer_input
  shape 동일
  dtype 동일
  device 동일
  reset 후 state 초기화 동일
```

허용 오차는 모델마다 다릅니다.

| 모델 | 허용 오차 |
|---|---|
| LIF | 거의 exact |
| RF | 거의 exact. complex 연산 때문에 작은 오차 가능 |
| TC-LIF | 거의 exact |
| TS-LIF | 거의 exact |
| DH-SNN | origin의 dtype/device 처리 때문에 약간 차이 가능 |
| D-RF | FFT 때문에 작은 numerical 차이 가능 |
| Spikformer | API migration 후 완전 동일 어려울 수 있음 |
| SpikingSSM | kernel 구현 방식에 따라 오차 가능 |
| SpikeGRU | 수식 재확정 전에는 비교 불가 |

PSD 분석에서는 특히 다음도 비교해야 합니다.

```text
old PSD curve
new PSD curve
peak frequency
band energy
low/high frequency ratio
layer-wise PSD ranking
```

---

# 19. 최종 구조는 이렇게 가면 됩니다

```text
src/sj_custom/
  surrogate.py
    ProjectFastSigmoid
    project_spike

  base.py
    PSDTraceMemoryModule
    PSDBaseNode

  lif.py
    CustomLIFNode

  rf.py
    CustomRFNode

  tc_lif.py
    CustomTCLIFNode

  ts_lif.py
    CustomTSLIFNode

  dh_snn.py
    CustomDHSNNNode

  d_rf.py
    BiRFKernel
    CustomDRFBlock

  spikegru.py
    CustomSpikeGRUCell
    CustomSpikeGRUBlock

  ssm.py
    CustomSpikingSSMBlock

  layers.py
    LinearSpikingLayer
    RecurrentLinearSpikingLayer
    Conv2DSpikingLayer
```

그리고 `snn_builder.py` 는 이렇게 단순화합니다.

```text
_build_dense_family_layer()
  lif     → LinearSpikingLayer(CustomLIFNode)
  rf      → LinearSpikingLayer(CustomRFNode)
  tc_lif  → LinearSpikingLayer(CustomTCLIFNode)
  ts_lif  → LinearSpikingLayer(CustomTSLIFNode)
  dh_snn  → LinearSpikingLayer(CustomDHSNNNode)
  d_rf    → CustomDRFBlock

_build_conv2d_family_layer()
  cnn_lif → Conv2DSpikingLayer(CustomLIFNode)
  cnn_rf  → Conv2DSpikingLayer(CustomRFNode)
```

---

# 20. 최종 판단

네 의도라면 **전부 가능** 합니다.

하지만 구현 철학은 이렇게 고정해야 합니다.

```text
1. SpikingJelly 내장 LIFNode로 단순 대체하지 않는다.
2. 각 뉴런모델의 원래 수식을 custom node/block로 보존한다.
3. BaseNode는 threshold neuron에만 쓴다.
4. D-RF, SpikeGRU, SpikingSSM은 MemoryModule/block로 구현한다.
5. 외부 shape는 기존 프로젝트처럼 (B,T,*) 를 유지한다.
6. reset_net과 trace 반환 규칙을 모든 모델에 강제한다.
7. 기존 구현과 새 구현의 membrane/spike/PSD equivalence test를 통과시킨다.
```

한마디로 말하면, 네가 하려는 건 **"SpikingJelly 기반 연구용 뉴런모델 라이브러리화"** 입니다.  
단순 리팩토링이 아니라, PSD 분석을 위한 공통 SNN execution substrate를 만드는 작업입니다.

---

**Sources:**

- [Core Neuron Modules — spikingjelly  文档](https://spikingjelly.readthedocs.io/zh-cn/latest/APIs/spikingjelly.activation_based.neuron.core.html)



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)