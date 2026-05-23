# 스파이킹 젤리 LIF API

**User:** Anonymous  
**Created:** 5/22/2026 19:34:09  
**Updated:** 5/23/2026 2:22:35  
**Exported:** 5/23/2026 2:30:20  
**Link:** [https://chatgpt.com/c/6a10310a-4a98-8324-a126-b3ab35b57474](https://chatgpt.com/c/6a10310a-4a98-8324-a126-b3ab35b57474)  

## Prompt:
2026. 5. 22. 오후 7:34:08

스파이킹 젤리에서 바닐라 lif 레이어 는 어떤 api를 쓰면 돼?

## Response:
2026. 5. 22. 오후 7:34:09

> **SpikingJelly LIFNode 공식 문서 검색**
>
> [SpikingJelly is an open-source deep learning framework ...](https://github.com/fangwei123456/spikingjelly)
> [spikingjelly.activation_based.neuron package](https://spikingjelly.readthedocs.io/zh-cn/latest/APIs/spikingjelly.activation_based.neuron.html)
> [SpikingJelly](https://ai-data-base.com/wp-content/uploads/2023/10/2310.16620v1.pdf)
> [CLIF: Complementary Leaky Integrate-and-Fire Neuron for ...](https://arxiv.org/html/2402.04663v5)
> [SpikingJelly: An open-source machine learning infrastructure ...](https://pmc.ncbi.nlm.nih.gov/articles/PMC10558124/)
>
> **문서 서명 정보 검색**
>
> [spikingjelly.activation_based.neuron package — spikingjelly  文档](https://spikingjelly.readthedocs.io/zh-cn/latest/APIs/spikingjelly.activation_based.neuron.html)
>
> **찾기 위한 인용과 서명 확인**
>
> [Core Neuron Modules — spikingjelly  文档](https://spikingjelly.readthedocs.io/zh-cn/latest/APIs/spikingjelly.activation_based.neuron.core.html)
>
> **설명 및 코드 예시 제공**
>
> LIFNode와 관련된 API를 다룰 때, `activation_based`와 `clock_driven` 구분 및 `LIFNode`와 "layer" 대신 뉴런 활성화 모듈을 삽입해야 합니다. 예시로 Linear 계층과 함께 `v_reset` 하드 리셋을 다룰 수 있습니다. 또한, LIFNode는 공식 문서 174–182라인을 참조하며, 코드 예시와 함께 설명해야 합니다.
>
> 15s 동안 생각함

공식 문서 기준으로 확인해서 답하겠습니다. 핵심은 "레이어" 라기보다 PyTorch `nn.Module` 안에 넣는 **뉴런 노드(Node)** API를 쓰는 구조입니다.

## 결론

스파이킹젤리에서 **바닐라 LIF 뉴런 레이어** 는 보통 아래 API를 쓰면 됩니다.

```python
from spikingjelly.activation_based import neuron

lif = neuron.LIFNode()
```

또는 공식 문서가 권장하는 직접 import 방식은 다음입니다.

```python
from spikingjelly.activation_based.neuron import LIFNode

lif = LIFNode()
```

공식 문서 기준으로 `LIFNode` 는 `spikingjelly.activation_based.neuron` 아래의 **core neuron module** 이고, 내부 하위 경로인 `spikingjelly.activation_based.neuron.lif` 에서 직접 import하는 것은 비권장입니다. ([spikingjelly.readthedocs.io](https://spikingjelly.readthedocs.io/zh-cn/latest/APIs/spikingjelly.activation_based.neuron.html))

---

## 가장 표준적인 사용 예시

```python
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate, functional

net = nn.Sequential(
    nn.Linear(784, 256),
    neuron.LIFNode(
        tau=2.0,
        decay_input=True,
        v_threshold=1.0,
        v_reset=0.0,
        surrogate_function=surrogate.ATan(),
        detach_reset=False,
        step_mode='s',
        backend='torch'
    ),
    nn.Linear(256, 10),
    neuron.LIFNode(
        tau=2.0,
        decay_input=True,
        v_threshold=1.0,
        v_reset=0.0,
        surrogate_function=surrogate.ATan(),
        detach_reset=False,
        step_mode='s',
        backend='torch'
    )
)

x = torch.rand(32, 784)
spike = net(x)

functional.reset_net(net)
```

여기서 **실제로 LIF 역할을 하는 것은** `neuron.LIFNode(...)` 입니다. `nn.Linear`, `nn.Conv2d` 같은 것은 시냅스 가중치층이고, 그 뒤에 `LIFNode` 를 붙여서 막전위 누적, 누설, 발화, 리셋을 수행합니다.

---

## LIFNode 주요 인자

공식 시그니처는 다음 형태입니다.

```python
LIFNode(
    tau=2.0,
    decay_input=True,
    v_threshold=1.0,
    v_reset=0.0,
    surrogate_function=surrogate.Sigmoid(),
    detach_reset=False,
    step_mode='s',
    backend='torch',
    store_v_seq=False
)
```

공식 문서에서 `LIFNode` 는 막전위 시간상수 `tau`, 입력 감쇠 여부 `decay_input`, 임계값 `v_threshold`, 리셋 전압 `v_reset`, surrogate function, reset detach 여부, step mode, backend, voltage sequence 저장 여부를 받는다고 설명합니다. ([spikingjelly.readthedocs.io](https://spikingjelly.readthedocs.io/zh-cn/latest/APIs/spikingjelly.activation_based.neuron.core.html))

| 인자 | 의미 |
|---|---|
| `tau` | 막전위 시간상수. 클수록 천천히 누설됨 |
| `decay_input` | 입력 $X[t]$ 도 누설식 안에 넣을지 여부 |
| `v_threshold` | 발화 임계값 |
| `v_reset=0.0` | hard reset. 발화 후 막전위를 0으로 리셋 |
| `v_reset=None` | soft reset. 발화 후 $v_threshold$ 만큼 빼는 방식 |
| `surrogate_function` | Heaviside 발화 함수의 역전파 근사 |
| `detach_reset` | reset 연산을 계산 그래프에서 detach할지 여부 |
| `step_mode='s'` | 단일 time step 입력 |
| `step_mode='m'` | 여러 time step 입력 |
| `backend='torch'` | 기본 PyTorch backend |
| `store_v_seq=True` | multi-step에서 시간별 막전위 `v_seq` 저장 |

---

## 동역학

스파이킹젤리 `LIFNode` 의 subthreshold 동역학은 `decay_input` 설정에 따라 달라집니다.

`decay_input=True` 일 때:

$$
H[t]=V[t-1]+\frac{1}{\tau}(X[t]-(V[t-1]-V_{reset}))
$$

`decay_input=False` 일 때:

$$
H[t]=V[t-1]-\frac{1}{\tau}(V[t-1]-V_{reset})+X[t]
$$

공식 문서도 위 두 식을 `LIFNode` 의 동역학으로 제시합니다. ([spikingjelly.readthedocs.io](https://spikingjelly.readthedocs.io/zh-cn/latest/APIs/spikingjelly.activation_based.neuron.core.html))

---

## 단일 스텝 vs 멀티 스텝

### 1. 단일 스텝

입력이 매번 `(B, features)` 또는 `(B, C, H, W)` 로 들어옵니다.

```python
lif = neuron.LIFNode(step_mode='s')

for t in range(T):
    y_t = lif(x_seq[t])
```

### 2. 멀티 스텝

입력이 한 번에 `(T, B, ...)` 로 들어옵니다.

```python
lif = neuron.LIFNode(step_mode='m')

y_seq = lif(x_seq)  # x_seq.shape = [T, B, ...]
```

문서상 `step_mode` 는 `'s'` 또는 `'m'` 을 받을 수 있고, multi-step에서는 입력 shape가 `[T, N, *]` 형태입니다. ([spikingjelly.readthedocs.io](https://spikingjelly.readthedocs.io/zh-cn/latest/APIs/spikingjelly.activation_based.neuron.core.html))

---

## 네 연구 코드 기준으로 추천

PSD 분석용으로 레이어별 spike/membrane을 관찰하려면 일단 기본은 이 조합이 가장 무난합니다.

```python
from spikingjelly.activation_based import neuron, surrogate

lif = neuron.LIFNode(
    tau=2.0,
    decay_input=True,
    v_threshold=1.0,
    v_reset=0.0,
    surrogate_function=surrogate.ATan(),
    detach_reset=False,
    step_mode='m',
    backend='torch',
    store_v_seq=True
)
```

이렇게 하면 multi-step 입력을 받고, `store_v_seq=True` 로 시간별 막전위도 저장할 수 있습니다. 단, 메모리를 더 씁니다. 공식 문서도 `store_v_seq=False` 가 메모리 절약에 유리하다고 설명합니다. ([spikingjelly.readthedocs.io](https://spikingjelly.readthedocs.io/zh-cn/latest/APIs/spikingjelly.activation_based.neuron.core.html))

정리하면:

```python
from spikingjelly.activation_based import neuron

lif = neuron.LIFNode(...)
```

이게 스파이킹젤리에서 쓰는 **기본 바닐라 LIF API** 입니다.

---

**Sources:**

- [spikingjelly.activation_based.neuron package — spikingjelly  文档](https://spikingjelly.readthedocs.io/zh-cn/latest/APIs/spikingjelly.activation_based.neuron.html)

## Prompt:
2026. 5. 22. 오후 7:36:04

tau <- 이거 학습 파라미터로 넣으려면?

## Response:
2026. 5. 22. 오후 7:36:05

> **Searching SpikingJelly tau learnable parameter documentation**
>
> [SpikingJelly is an open-source deep learning framework ...](https://github.com/fangwei123456/spikingjelly)
> [Classify DVS128 Gesture — spikingjelly alpha 文档](https://spikingjelly.readthedocs.io/zh_CN/0.0.0.0.6/clock_driven_en/14_classify_dvsg.html)
> [spikingjelly](https://pypi.org/project/spikingjelly/)
> [SpikingJelly](https://ai-data-base.com/wp-content/uploads/2023/10/2310.16620v1.pdf)
> [[2310.16620] SpikingJelly: An open-source machine ...](https://arxiv.org/abs/2310.16620)
> [Incorporating Learnable Membrane Time Constant To ...](https://openaccess.thecvf.com/content/ICCV2021/papers/Fang_Incorporating_Learnable_Membrane_Time_Constant_To_Enhance_Learning_of_Spiking_ICCV_2021_paper.pdf)
> [(PDF) SpikingJelly: An open-source machine learning ...](https://www.researchgate.net/publication/374526125_SpikingJelly_An_open-source_machine_learning_infrastructure_platform_for_spike-based_intelligence)
> [SpikingJelly: An open-source machine learning infrastructure ...](https://pmc.ncbi.nlm.nih.gov/articles/PMC10558124/)
> [SpikingJelly: An open-source machine learning ...](https://hal.science/hal-04337978v1/file/Fang_2023.pdf)
>
> **Searching for official documentation**
>
> [Core Neuron Modules — spikingjelly  文档](https://spikingjelly.readthedocs.io/zh-cn/latest/APIs/spikingjelly.activation_based.neuron.core.html)
> [Incorporating Learnable Membrane Time Constant To Enhance Learning of Spiking Neural Networks](https://openaccess.thecvf.com/content/ICCV2021/papers/Fang_Incorporating_Learnable_Membrane_Time_Constant_To_Enhance_Learning_of_Spiking_ICCV_2021_paper.pdf)
> [GitHub - fangwei123456/spikingjelly: SpikingJelly is an open-source deep learning framework for Spiking Neural Network (SNN) based on PyTorch. · GitHub](https://github.com/fangwei123456/spikingjelly)
>
> **확인해야 할 PLIF 클래스 및 tau 속성**
>
> [Core Neuron Modules — spikingjelly  文档](https://spikingjelly.readthedocs.io/zh-cn/latest/APIs/spikingjelly.activation_based.neuron.core.html)
>
> 21s 동안 생각함

공식 API 이름을 확인해서 답하겠습니다. 스파이킹젤리에는 `tau` 를 직접 `nn.Parameter` 로 바꾸기보다, 이미 그 용도로 만든 **Parametric LIF** 노드가 있습니다.

## 결론

`tau` 를 학습 파라미터로 넣고 싶으면 `LIFNode(tau=...)` 가 아니라 **`ParametricLIFNode`** 를 쓰면 됩니다.

```python
from spikingjelly.activation_based import neuron, surrogate

lif = neuron.ParametricLIFNode(
    init_tau=2.0,
    decay_input=True,
    v_threshold=1.0,
    v_reset=0.0,
    surrogate_function=surrogate.ATan(),
    detach_reset=False,
    step_mode='m',
    backend='torch'
)
```

공식 문서에서 `ParametricLIFNode` 는 `init_tau` 를 받고, 내부적으로

$$
\frac{1}{\tau}=\mathrm{Sigmoid}(w)
$$

로 두며, $w$ 를 학습 가능한 파라미터로 둔다고 설명합니다. 즉 `tau` 자체를 직접 `nn.Parameter` 로 두는 것이 아니라, **누설 계수 $1/\tau$** 를 안정적인 범위 안에서 학습시키는 구조입니다. ([SpikingJelly](https://spikingjelly.readthedocs.io/zh-cn/latest/APIs/spikingjelly.activation_based.neuron.core.html))

---

## 일반 LIF와 PLIF 차이

### 고정 tau

```python
lif = neuron.LIFNode(tau=2.0)
```

이 경우 `tau=2.0` 은 그냥 고정 하이퍼파라미터입니다.

---

### 학습 tau

```python
lif = neuron.ParametricLIFNode(init_tau=2.0)
```

이 경우 `init_tau=2.0` 은 초기값이고, 실제 학습되는 파라미터는 내부의 `w` 입니다.

스파이킹젤리 공식 API에는 다음처럼 정의되어 있습니다.

```python
neuron.ParametricLIFNode(
    init_tau=2.0,
    decay_input=True,
    v_threshold=1.0,
    v_reset=0.0,
    surrogate_function=...,
    detach_reset=False,
    step_mode='s',
    backend='torch',
    store_v_seq=False
)
```

공식 문서상 `ParametricLIFNode` 는 `LIFNode` 와 같은 막전위 동역학을 쓰되, $\frac{1}{\tau}$ 를 `Sigmoid(w)` 로 파라미터화합니다. ([SpikingJelly](https://spikingjelly.readthedocs.io/zh-cn/latest/APIs/spikingjelly.activation_based.neuron.core.html))

---

## optimizer에는 따로 넣을 필요 없음

모델 안에 `ParametricLIFNode` 를 넣고 평범하게 하면 됩니다.

```python
model = nn.Sequential(
    nn.Linear(784, 256),
    neuron.ParametricLIFNode(init_tau=2.0, surrogate_function=surrogate.ATan()),
    nn.Linear(256, 10),
    neuron.ParametricLIFNode(init_tau=2.0, surrogate_function=surrogate.ATan())
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

`model.parameters()` 안에 `ParametricLIFNode` 의 학습 파라미터 `w` 가 자동으로 들어갑니다.

---

## 현재 tau 값을 보고 싶으면

```python
for name, m in model.named_modules():
    if isinstance(m, neuron.ParametricLIFNode):
        tau = 1.0 / m.w.sigmoid()
        print(name, tau.item())
```

왜 이렇게 보냐면 실제 저장된 학습 파라미터가 `tau` 가 아니라 `w` 이기 때문입니다.

$$
\tau=\frac{1}{\mathrm{Sigmoid}(w)}
$$

따라서 `w` 가 커지면 `Sigmoid(w)` 가 커지고, $\tau$ 는 작아집니다. 반대로 `w` 가 작아지면 $\tau$ 는 커집니다.

---

## 주의할 점

`ParametricLIFNode` 는 기본적으로 **한 노드 레이어 안에서 tau를 공유**하는 방식으로 보는 게 맞습니다. PLIF 논문도 같은 레이어 안의 뉴런들은 membrane time constant를 공유하고, 서로 다른 레이어는 다른 time constant를 갖게 한다고 설명합니다. ([CVF Open Access](https://openaccess.thecvf.com/content/ICCV2021/papers/Fang_Incorporating_Learnable_Membrane_Time_Constant_To_Enhance_Learning_of_Spiking_ICCV_2021_paper.pdf))

즉 이런 구조라면:

```python
nn.Linear(784, 256),
neuron.ParametricLIFNode(init_tau=2.0),
```

256개 뉴런 각각이 전부 다른 $\tau$ 를 갖는 것이 아니라, 이 `ParametricLIFNode` 모듈 하나가 하나의 학습 가능한 시간상수 계수를 갖는다고 이해하면 됩니다.

---

## 네 연구 코드 기준 추천

PSD 분석이나 레이어 주파수 특성을 볼 목적이면, 일단 아래처럼 쓰는 게 가장 깔끔합니다.

```python
from spikingjelly.activation_based import neuron, surrogate

lif = neuron.ParametricLIFNode(
    init_tau=2.0,
    decay_input=True,
    v_threshold=1.0,
    v_reset=0.0,
    surrogate_function=surrogate.ATan(),
    detach_reset=False,
    step_mode='m',
    backend='torch',
    store_v_seq=True
)
```

- `init_tau=2.0`: 초기 막전위 시간상수
- `ParametricLIFNode`: tau 학습 가능
- `store_v_seq=True`: PSD 분석용으로 막전위 시계열 저장 가능
- `step_mode='m'`: 입력을 $(T,B,...)$ 형태로 한 번에 처리

정리하면, **tau를 학습시키려면 `LIFNode` 에서 `tau` 를 어떻게 바꾸는 게 아니라 `ParametricLIFNode(init_tau=...)` 로 바꾸는 것**이 스파이킹젤리 표준 방식입니다.

---

**Sources:**

- [Core Neuron Modules — spikingjelly  文档](https://spikingjelly.readthedocs.io/zh-cn/latest/APIs/spikingjelly.activation_based.neuron.core.html)
- [Incorporating Learnable Membrane Time Constant To Enhance Learning of Spiking Neural Networks](https://openaccess.thecvf.com/content/ICCV2021/papers/Fang_Incorporating_Learnable_Membrane_Time_Constant_To_Enhance_Learning_of_Spiking_ICCV_2021_paper.pdf)

## Prompt:
2026. 5. 22. 오후 7:38:13

ParametricLIFNode 는 기본적으로 한 노드 레이어 안에서 tau를 공유 하는게 아닌 모든 뉴런이 독립 tau를 갖게하려면?

## Response:
2026. 5. 22. 오후 7:38:16

> **SpikingJelly 공식 문서 검색**
>
> [SpikingJelly is an open-source deep learning framework ...](https://github.com/fangwei123456/spikingjelly)
> [Clock driven: Neurons — spikingjelly alpha 文档](https://spikingjelly.readthedocs.io/zh_CN/0.0.0.0.6/clock_driven_en/0_neuron.html)
> [spikingjelly](https://pypi.org/project/spikingjelly/)
> [SpikingJelly: An open-source machine learning ...](https://www.science.org/doi/10.1126/sciadv.adi1480)
> [Spyx: A Library for Just-In-Time Compiled Optimization of ...](https://arxiv.org/html/2402.18994v1)
> [SpikingJelly](https://ai-data-base.com/wp-content/uploads/2023/10/2310.16620v1.pdf)
> [SpikingJelly](https://open-neuromorphic.org/neuromorphic-computing/software/snn-frameworks/spikingjelly/)
> [SpikingJelly: An open-source machine learning infrastructure ...](https://pmc.ncbi.nlm.nih.gov/articles/PMC10558124/)
> [spikingjelly - conda-forge](https://anaconda.org/conda-forge/spikingjelly)
> [Incorporating Learnable Membrane Time Constant To ...](https://openaccess.thecvf.com/content/ICCV2021/papers/Fang_Incorporating_Learnable_Membrane_Time_Constant_To_Enhance_Learning_of_Spiking_ICCV_2021_paper.pdf)
>
> **Searching for official OpenDocs core information**
>
> [spikingjelly.activation_based.neuron — spikingjelly  文档](https://spikingjelly.readthedocs.io/zh_CN/latest/_modules/spikingjelly/activation_based/neuron.html)
>
> **열기 ParametricLIFNode 주변의 라인 찾기**
>
> [spikingjelly.activation_based.neuron — spikingjelly  文档](https://spikingjelly.readthedocs.io/zh_CN/latest/_modules/spikingjelly/activation_based/neuron.html)
>
> 1m 1s 동안 생각함

구현 관점에서 답하겠습니다. 핵심은 `ParametricLIFNode` 의 기본 구현은 보통 **스칼라 학습 파라미터 하나**라서, 뉴런별 독립 $\tau$ 를 원하면 커스텀 노드를 만드는 쪽이 정석입니다.

`ParametricLIFNode` 의 소스 기준으로 보면 `self.w = nn.Parameter(torch.as_tensor(init_w))` 라서 기본은 스칼라입니다. 다만 `torch` backend에서는 그 `w` 를 입력 텐서에 broadcast되도록 벡터/텐서 파라미터로 바꾸면 뉴런별 $\tau$ 를 구현할 수 있습니다.

## 결론

`ParametricLIFNode` 자체에는 **"뉴런별 tau" 옵션이 없습니다.**  
공식 소스 기준으로 `ParametricLIFNode` 는 내부에서

$$
\frac{1}{\tau}=\mathrm{Sigmoid}(w)
$$

를 쓰고, `w` 를 `nn.Parameter(torch.as_tensor(init_w))` 로 만듭니다. 즉 기본값은 **스칼라 파라미터 하나**입니다. ([SpikingJelly](https://spikingjelly.readthedocs.io/zh_CN/latest/_modules/spikingjelly/activation_based/neuron.html))

뉴런마다 독립 $\tau$ 를 주려면 `w` 를 **입력 feature shape에 맞는 텐서 파라미터** 로 바꿔야 합니다.

---

# 1. 가장 단순한 방법: `ParametricLIFNode` 를 상속해서 `w` 를 텐서로 교체

```python
import math
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate

class NeuronwiseParametricLIFNode(neuron.ParametricLIFNode):
    def __init__(
        self,
        param_shape,
        init_tau=2.0,
        decay_input=True,
        v_threshold=1.0,
        v_reset=0.0,
        surrogate_function=None,
        detach_reset=False,
        step_mode="s",
        store_v_seq=False,
    ):
        if isinstance(param_shape, int):
            param_shape = (param_shape,)

        if surrogate_function is None:
            surrogate_function = surrogate.Sigmoid()

        if init_tau <= 1.0:
            raise ValueError("ParametricLIFNode requires init_tau > 1.0")

        super().__init__(
            init_tau=init_tau,
            decay_input=decay_input,
            v_threshold=v_threshold,
            v_reset=v_reset,
            surrogate_function=surrogate_function,
            detach_reset=detach_reset,
            step_mode=step_mode,
            backend="torch",
            store_v_seq=store_v_seq,
        )

        init_w = -math.log(init_tau - 1.0)

        self.w = nn.Parameter(
            torch.full(
                size=tuple(param_shape),
                fill_value=init_w,
                dtype=torch.float32,
            )
        )

    @property
    def tau(self):
        return 1.0 / self.w.sigmoid()
```

중요한 점은 `backend="torch"` 로 고정하는 것입니다. 공식 구현에서 `ParametricLIFNode` 의 `torch` multi-step은 Python loop로 `single_step_forward` 를 호출하므로 PyTorch broadcasting이 자연스럽게 됩니다. 반면 `cupy` / `triton` backend는 전용 kernel에 `self.w.sigmoid()` 를 넘기는 구조라서 벡터 `w` 를 공식적으로 보장한다고 보기 어렵습니다. ([SpikingJelly](https://spikingjelly.readthedocs.io/zh_CN/latest/_modules/spikingjelly/activation_based/neuron.html))

---

# 2. MLP에서 모든 뉴런이 독립 tau를 갖는 경우

예를 들어 hidden dimension이 256이면:

```python
lif = NeuronwiseParametricLIFNode(
    param_shape=(256,),
    init_tau=2.0,
    decay_input=True,
    v_threshold=1.0,
    v_reset=0.0,
    surrogate_function=surrogate.ATan(),
    step_mode="m",
    store_v_seq=True,
)
```

입력 shape가 다음과 같다고 하겠습니다.

$$
x_{\mathrm{seq}}\in\mathbb{R}^{T\times B\times 256}
$$

그러면 `w.shape == (256,)` 이고, 각 hidden unit마다 서로 다른 $\tau_i$ 를 갖습니다.

```python
print(lif.w.shape)
print(lif.tau.shape)
```

출력:

```text
torch.Size([256])
torch.Size([256])
```

즉

$$
\tau_1,\tau_2,\dots,\tau_{256}
$$

가 전부 독립적으로 학습됩니다.

---

# 3. CNN에서 "모든 뉴런" 이 독립 tau를 갖는 경우

CNN 출력이 예를 들어

$$
x[t]\in\mathbb{R}^{B\times 64\times 32\times 32}
$$

라면, 진짜로 모든 spatial neuron이 독립 $\tau$ 를 갖게 하려면:

```python
lif = NeuronwiseParametricLIFNode(
    param_shape=(64, 32, 32),
    init_tau=2.0,
    surrogate_function=surrogate.ATan(),
    step_mode="m",
    store_v_seq=True,
)
```

이 경우 $\tau$ 개수는

$$
64\times 32\times 32=65536
$$

개입니다.

즉 channel, height, width 위치마다 전부 다른 $\tau$ 를 가집니다.

---

# 4. CNN에서는 보통 "channel-wise tau" 가 더 현실적임

CNN에서 모든 spatial 위치마다 $\tau$ 를 독립으로 두면 파라미터가 너무 많고, 해석도 지저분해집니다. 보통 더 자연스러운 방식은 **채널별 tau** 입니다.

```python
lif = NeuronwiseParametricLIFNode(
    param_shape=(64, 1, 1),
    init_tau=2.0,
    surrogate_function=surrogate.ATan(),
    step_mode="m",
    store_v_seq=True,
)
```

이 경우 입력 shape가

$$
B\times 64\times 32\times 32
$$

여도 PyTorch broadcasting에 의해 각 채널마다 하나의 $\tau_c$ 가 적용됩니다.

즉

$$
\tau_1,\tau_2,\dots,\tau_{64}
$$

만 학습됩니다.

정리하면:

| 방식 | `param_shape` | 의미 |
|---|---:|---|
| layer-wise | `()` | 레이어 전체가 tau 하나 공유 |
| dense neuron-wise | `(F,)` | MLP feature별 tau |
| conv channel-wise | `(C,1,1)` | CNN 채널별 tau |
| conv true neuron-wise | `(C,H,W)` | CNN 위치까지 포함한 모든 뉴런별 tau |

---

# 5. 기존 `ParametricLIFNode` 객체를 직접 고치는 방법

상속 클래스를 만들기 싫으면 이렇게도 할 수 있습니다.

```python
import math
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate

lif = neuron.ParametricLIFNode(
    init_tau=2.0,
    surrogate_function=surrogate.ATan(),
    step_mode="m",
    backend="torch",
    store_v_seq=True,
)

init_tau = 2.0
init_w = -math.log(init_tau - 1.0)

lif.w = nn.Parameter(torch.full((256,), init_w))
```

단, 이건 **optimizer 만들기 전에** 해야 합니다.

```python
# 먼저 w 교체
lif.w = nn.Parameter(torch.full((256,), init_w))

# 그 다음 optimizer 생성
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

반대로 optimizer를 이미 만든 뒤에 `lif.w` 를 교체하면 optimizer가 새 파라미터를 추적하지 못합니다. 그 경우 optimizer를 다시 만들어야 합니다.

---

# 6. 왜 `w` 를 직접 tau로 두지 않나?

`ParametricLIFNode` 는 $\tau$ 를 직접 학습하지 않고 다음처럼 둡니다.

$$
\frac{1}{\tau}=\mathrm{Sigmoid}(w)
$$

따라서

$$
\tau=\frac{1}{\mathrm{Sigmoid}(w)}
$$

입니다.

이렇게 하면 $\mathrm{Sigmoid}(w)\in(0,1)$ 이므로

$$
\tau>1
$$

이 자동으로 보장됩니다. 공식 구현도 `init_tau > 1.0` 을 assert하고, `init_w = -math.log(init_tau - 1.0)` 로 초기화합니다. ([SpikingJelly](https://spikingjelly.readthedocs.io/zh_CN/latest/_modules/spikingjelly/activation_based/neuron.html))

---

# 7. 연구 관점 추천

네 목적이 **레이어 PSD 특성 분석** 이라면 추천 순서는 이렇습니다.

## 1순위: layer-wise tau

```python
neuron.ParametricLIFNode(init_tau=2.0)
```

가장 해석이 쉽습니다.  
"이 레이어의 시간상수" 라고 말할 수 있습니다.

## 2순위: channel-wise tau

```python
NeuronwiseParametricLIFNode(param_shape=(C, 1, 1))
```

CNN에서는 이 정도가 가장 타협적입니다.  
"채널마다 다른 temporal filtering 특성" 이라고 해석할 수 있습니다.

## 3순위: true neuron-wise tau

```python
NeuronwiseParametricLIFNode(param_shape=(C, H, W))
```

가능은 하지만, 해석이 어려워집니다.  
PSD 분석에서 레이어 대표신호를 만들 때도 각 위치가 서로 다른 필터 특성을 가지므로, 평균 PSD의 의미가 더 흐려질 수 있습니다.

---

## 핵심만 말하면

MLP에서 뉴런별 $\tau$ 를 원하면:

```python
NeuronwiseParametricLIFNode(param_shape=(hidden_dim,))
```

CNN에서 채널별 $\tau$ 를 원하면:

```python
NeuronwiseParametricLIFNode(param_shape=(C, 1, 1))
```

CNN에서 진짜 모든 spatial neuron별 $\tau$ 를 원하면:

```python
NeuronwiseParametricLIFNode(param_shape=(C, H, W))
```

그리고 반드시 `backend="torch"` 로 쓰는 게 안전합니다.

---

**Sources:**

- [spikingjelly.activation_based.neuron — spikingjelly  文档](https://spikingjelly.readthedocs.io/zh_CN/latest/_modules/spikingjelly/activation_based/neuron.html)

## Prompt:
2026. 5. 22. 오후 7:40:35

cupy 벡엔드 쓰려면 메모리 모듈 가지고 구현해야겠네

## Response:
2026. 5. 22. 오후 7:40:37

> **SpikingJelly의 CuPy backend와 메모리 모듈 관련**
>
> 맞아요, CuPy backend와 neuron-wise learnable tau를 사용하고 싶다면, 커스텀 MemoryModule/BaseNode를 구현하고, 그 안에 custom neuronal_charge와 multi_step_forward를 추가해야 할 것 같아요. 최신 정보는 웹 검색을 통해 공식 문서를 참고하는 것이 좋겠어요.
>
> **Searching SpikingJelly memory module CuPy backend docs**
>
> [SpikingJelly is an open-source deep learning framework ...](https://github.com/fangwei123456/spikingjelly)
> [spikingjelly.clock_driven.neuron 源代码](https://spikingjelly.readthedocs.io/zh-cn/0.0.0.0.6/_modules/spikingjelly/clock_driven/neuron.html)
> [spikingjelly](https://pypi.org/project/spikingjelly/)
> [SpikingJelly](https://ai-data-base.com/wp-content/uploads/2023/10/2310.16620v1.pdf)
>
> **SpikingJelly 메모리 모듈 및 CuPy 문서 참고**
>
> 답변에 인용을 추가해야 하니, 메모리 모듈 관련 문서, BaseNode 소스, CuPy로 구현된 신경망 튜토리얼을 찾아봐야 할 것 같아요. 관련 문서를 열어서 보다 정확한 정보를 확인하고, 최신 업데이트를 바탕으로 답변을 보강할 수 있을 거예요.
>
> **브라우징 관련 문서 및 튜토리얼**
>
> [编写CUPY神经元 — spikingjelly  文档](https://spikingjelly.readthedocs.io/zh-cn/latest/tutorials/cn/cupy_neuron.html)
> [GitHub - fangwei123456/spikingjelly: SpikingJelly is an open-source deep learning framework for Spiking Neural Network (SNN) based on PyTorch. · GitHub](https://github.com/fangwei123456/spikingjelly)
>
> **메모리 모듈 문서 열기**
>
> Base.MemoryModule 문서를 열어보는 것이 좋겠네요. 필요한 정보를 찾을 수 있을 것 같아요. 클릭해서 문서를 확인하고, 어떤 내용이 중요한지 빠르게 파악해보겠습니다.
>
> **열기 base.MemoryModule 문서 찾기**
>
> [spikingjelly.activation_based.tensor_cache package — spikingjelly  文档](https://spikingjelly.readthedocs.io/zh-cn/latest/APIs/spikingjelly.activation_based.cuda_kernel.tensor_cache.html)
>
> **MemoryModule 문서 찾아보기**
>
> 클릭이 잘못된 것 같아서, Base 문서에서 MemoryModule을 찾아봐야겠어요. 올바른 라인이 아닌 것 같아서 다시 찾아봐야 할 것 같네요. 문서를 열고 정확히 필요한 부분을 찾아서 확인해야겠어요.
>
> **메모리 모듈 찾기 위한 문서 검색**
>
> [spikingjelly.activation_based.base module — spikingjelly  文档](https://spikingjelly.readthedocs.io/zh-cn/latest/APIs/spikingjelly.activation_based.base.html)
>
> **라인 244 주변 확인**
>
> 244번 라인과 관련된 부분을 더 확인해 봐야 할 것 같아요. 등록과 관련된 부분도 다시 살펴봐야 할 것 같고, 그 주변 코드들도 열어보아서 추가적인 내용을 찾는 것이 필요할 거예요.
>
> **조치 요청 및 등록 관련 문제 분석**
>
> [spikingjelly.activation_based.base module — spikingjelly  文档](https://spikingjelly.readthedocs.io/zh-cn/latest/APIs/spikingjelly.activation_based.base.html)
>
> **ParametricLIFNode와 CuPy 백엔드 확인**
>
> ParametricLIFNode가 CuPy 백엔드를 지원하는지 확인하고, 관련된 공식 소스를 찾아야 할 것 같아요. 문서를 열어서 Parametric 관련 내용을 찾아보고, 해당 백엔드에 대한 정보를 얻어 정리해 보겠습니다.
>
> **검색 중 ParametricLIFNode와 지원되는 백엔드 관련 정보**
>
> [Core Neuron Modules — spikingjelly  文档](https://spikingjelly.readthedocs.io/zh-cn/latest/APIs/spikingjelly.activation_based.neuron.core.html)
>
> **Parametric 공식 및 문서 확인**
>
> Parametric 공식 관련 정보가 필요하니, 590번 라인 주변을 열어서 확인해야 할 것 같아요. 공식적인 출처에서 Parametric에 관한 세부사항을 찾아서 정확한 정보를 얻고, 정리해볼게요.
>
> **Searching for official sources on parametric formula lines**
>
> [Core Neuron Modules — spikingjelly  文档](https://spikingjelly.readthedocs.io/zh-cn/latest/APIs/spikingjelly.activation_based.neuron.core.html)
>
> **모듈 구현 방식과 설계 옵션 설명**
>
> MemoryModule은 상태 관리만 가능하고, CuPy 가속화에는 적합하지 않습니다. CuPy로 진정한 가속화를 원하면 CUDA 커널이나 auto_cuda가 필요합니다. 디자인 옵션은:
> 
> 1. **Torch 연산을 사용하는 MemoryModule**: GPU에서 실행 가능하지만 CuPy 백엔드는 지원하지 않음.
> 2. **CuPy 백엔드 및 벡터 τ**: MemoryModule과 사용자 정의 CuPy 커널(FPTT+BPTT) 또는 SpikingJelly의 커스텀 autograd 함수 사용.
> 3. **기존 ParametricLIFNode 커널 수정**: α 벡터를 사용하여 gradient 계산을 구현.
> 
> 또한, τ 파라미터 기울기와 뉴런별 τ에 대해 논의하며, 해당 커널에서의 업데이트 방식을 설명할 필요가 있습니다.
>
> 29s 동안 생각함

맞습니다. 단, 정확히 말하면 **MemoryModule만으로는 CuPy backend가 되는 것은 아니고**, 구조는 다음처럼 잡아야 합니다.

## 결론

뉴런별 독립 $\tau_i$ 를 쓰면서 `cupy` backend까지 쓰려면:

```text
MemoryModule 기반 커스텀 노드
+ register_memory("v", ...)
+ multi_step_forward(...)
+ custom CuPy CUDA kernel
+ custom autograd Function
```

까지 가야 합니다.

스파이킹젤리 문서상 `MemoryModule` 은 상태를 갖는 모듈의 기본 클래스이고, `register_memory` 로 막전위 같은 상태변수를 등록합니다. 기본 `supported_backends` 는 `("torch",)` 이므로, `cupy` 를 진짜 지원하려면 해당 모듈에서 별도 backend 처리를 구현해야 합니다. ([SpikingJelly](https://spikingjelly.readthedocs.io/zh-cn/latest/APIs/spikingjelly.activation_based.base.html))

---

# 왜 단순 상속으로는 부족한가

이전에 말한 방식:

```python
lif.w = nn.Parameter(torch.full((hidden_dim,), init_w))
```

이건 **torch backend에서는 가능**합니다. PyTorch broadcasting이 알아서 처리하기 때문입니다.

하지만 `cupy` backend는 내부에서 multi-step 동역학을 fused CUDA kernel로 계산합니다. 스파이킹젤리 공식 설명도 `cupy` / `triton` backend는 multi-step neuron을 빠르게 하기 위한 backend라고 설명합니다. ([GitHub](https://github.com/fangwei123456/spikingjelly))

즉 `cupy` backend에서는 다음 문제가 생깁니다.

```text
기존 ParametricLIFNode CuPy kernel:
    w 또는 1/tau를 scalar처럼 처리하는 전제

우리가 원하는 구현:
    뉴런마다 w_i, tau_i가 다름
```

그래서 단순히 `w.shape = (N,)` 으로 바꾸는 것만으로는 안전하지 않습니다. 커널이 `w[i]` 를 읽도록 설계되어 있어야 합니다.

---

# 구현 방향

## 1. 상태 관리는 `MemoryModule`

구조는 이런 식입니다.

```python
from spikingjelly.activation_based import base

class NeuronwisePLIFNode(base.MemoryModule):
    def __init__(self, ...):
        super().__init__()
        self.step_mode = "m"
        self.backend = "cupy"
        self.register_memory("v", 0.0)
```

`register_memory("v", ...)` 는 막전위 상태를 등록하는 역할입니다. 문서에서도 `register_memory` 는 막전위 같은 stateful variable을 저장하고, `reset()` 시 초기값으로 되돌리는 용도라고 설명합니다. ([SpikingJelly](https://spikingjelly.readthedocs.io/zh-cn/latest/APIs/spikingjelly.activation_based.base.html))

---

## 2. `torch` backend는 검증용으로 먼저 구현

먼저 이 식이 맞는지 PyTorch로 확인해야 합니다.

`decay_input=True` 일 때:

$$
H_{t,i}=V_{t-1,i}+\alpha_i(X_{t,i}-(V_{t-1,i}-V_{reset}))
$$

여기서

$$
\alpha_i=\mathrm{Sigmoid}(w_i)=\frac{1}{\tau_i}
$$

입니다. `ParametricLIFNode` 공식 문서도 $\frac{1}{\tau}=\mathrm{Sigmoid}(w)$ 이고 $w$ 가 학습 파라미터라고 설명합니다. ([SpikingJelly](https://spikingjelly.readthedocs.io/zh-cn/latest/APIs/spikingjelly.activation_based.neuron.core.html))

따라서 뉴런별 PLIF는 그냥 scalar $\alpha$ 를 vector $\alpha_i$ 로 바꾼 형태입니다.

---

## 3. `cupy` backend는 flatten된 $[T,N]$ 기준으로 커널 작성

스파이킹젤리의 CuPy neuron 튜토리얼도 multi-step CUDA kernel 입력을 기본적으로 다음처럼 봅니다.

```text
x_seq: [T,N]
v_init: [N]
spike_seq: [T,N]
v_seq: [T,N]
```

다차원 뉴런이면 $N$ 은 flatten된 뉴런 개수입니다. ([SpikingJelly](https://spikingjelly.readthedocs.io/zh-cn/latest/tutorials/cn/cupy_neuron.html))

따라서 CNN 출력이 원래

$$
x_{\mathrm{seq}}\in\mathbb{R}^{T\times B\times C\times H\times W}
$$

라면 kernel 내부에서는 보통

$$
x_{\mathrm{seq}}\rightarrow\mathbb{R}^{T\times N}
$$

로 flatten해서 처리하고, 끝나면 원래 shape로 되돌립니다. 공식 튜토리얼 예제도 `x_seq.flatten(1)` 로 flatten하고 출력 후 `view(x_seq.shape)` 로 되돌립니다. ([SpikingJelly](https://spikingjelly.readthedocs.io/zh-cn/latest/tutorials/cn/cupy_neuron.html))

---

# 핵심 커널에서 바뀌어야 하는 부분

기존 scalar PLIF 느낌은 대충 이렇습니다.

```cpp
h = v_prev + alpha * (x - (v_prev - v_reset));
```

뉴런별 tau를 쓰려면 이렇게 되어야 합니다.

```cpp
int n = index;          // neuron index
float alpha_n = sigmoid(w[n]);

h = v_prev + alpha_n * (x - (v_prev - v_reset));
```

즉 kernel argument가 scalar `w` 가 아니라 pointer여야 합니다.

```cpp
const float* w
```

그리고 각 뉴런 index마다:

```cpp
float alpha = 1.0f / (1.0f + expf(-w[n]));
```

을 읽어야 합니다.

---

# BPTT도 직접 맞춰야 함

forward만 만들면 끝이 아닙니다. 학습 가능한 $\tau_i$, 즉 $w_i$ 를 쓰려면 backward에서 `grad_w` 도 계산해야 합니다.

$$
\alpha_i=\mathrm{Sigmoid}(w_i)
$$

이므로

$$
\frac{\partial \alpha_i}{\partial w_i}
=\alpha_i(1-\alpha_i)
$$

입니다.

`decay_input=True` 에서는

$$
H_{t,i}=V_{t-1,i}+\alpha_i(X_{t,i}-(V_{t-1,i}-V_{reset}))
$$

이므로 직접항만 보면

$$
\frac{\partial H_{t,i}}{\partial w_i}
=(X_{t,i}-(V_{t-1,i}-V_{reset}))\alpha_i(1-\alpha_i)
$$

입니다.

다만 실제 BPTT에서는 $V_{t-1,i}$ 자체도 이전 시간의 $w_i$ 에 의존하므로, 시간 방향으로 누적되는 gradient까지 커널에서 처리해야 합니다. 그래서 **FPTT kernel만 고치면 안 되고 BPTT kernel도 같이 고쳐야 합니다.**

스파이킹젤리 CuPy neuron 튜토리얼도 forward kernel뿐 아니라 backward kernel과 autograd function을 함께 구성하는 방식으로 예제를 보여줍니다. ([SpikingJelly](https://spikingjelly.readthedocs.io/zh-cn/latest/tutorials/cn/cupy_neuron.html))

---

# 가장 현실적인 구현 순서

## 1단계: torch 버전

먼저 `MemoryModule` 로 torch-only 뉴런을 만듭니다.

```text
NeuronwisePLIFNode
- step_mode="m"
- backend="torch"
- w: nn.Parameter shape [N] 또는 broadcast shape
- v: memory
- spike_seq, v_seq 저장 옵션
```

이 단계에서 검증할 것:

```text
1. 출력 spike_seq shape
2. v reset 정상 동작
3. w.grad 생성 여부
4. tau_i 값 범위
5. 기존 ParametricLIFNode scalar tau와 동일 조건에서 결과 일치
```

---

## 2단계: cupy forward kernel

그 다음 `x_seq.flatten(1)`, `v.flatten()`, `w.flatten()` 기준으로 forward kernel을 만듭니다.

필요 입력:

```text
x_seq_flat: [T,N]
v_init_flat: [N]
w_flat: [N]
v_threshold: float
v_reset: float or None
decay_input: bool
detach_reset: bool
```

출력:

```text
spike_seq_flat: [T,N]
v_seq_flat: [T,N]
h_seq_flat: [T,N]
```

스파이킹젤리 튜토리얼에서도 `h_seq` 는 charge 후 fire 전 전압이고 backward에 필요하다고 설명합니다. ([SpikingJelly](https://spikingjelly.readthedocs.io/zh-cn/latest/tutorials/cn/cupy_neuron.html))

---

## 3단계: cupy backward kernel

여기서 `grad_x_seq`, `grad_v_init`, `grad_w` 를 계산해야 합니다.

특히 `grad_w` 는 뉴런별로 합산됩니다.

```text
grad_w[n] = sum over t of contribution(t,n)
```

즉 커널 설계에서 atomic add가 필요할 수도 있고, 아니면 neuron별 thread가 시간축을 순회하면서 자기 `grad_w[n]` 를 누적하게 만들 수도 있습니다. 후자가 구조상 더 깔끔합니다.

---

# 연구 코드 관점 추천

네 상황에서는 바로 CuPy부터 가지 말고 이렇게 가는 게 맞습니다.

## 추천 구조

```text
src/model/neuron/
    neuronwise_plif.py
        NeuronwisePLIFNode
        torch reference forward
        cupy forward/backward wrapper
```

그리고 CLI나 config에서는:

```text
--neuron plif_neuronwise
--plif_tau_scope layer | channel | neuron
--backend torch | cupy
```

정도로 분리하는 게 좋습니다.

---

# 해석 관점에서는 주의해야 함

뉴런별 $\tau_i$ 는 가능하지만, PSD 분석 목적에서는 해석이 훨씬 어려워집니다.

| tau 범위 | 해석 |
|---|---|
| layer-wise tau | "이 레이어의 시간상수" 라고 해석 가능 |
| channel-wise tau | "이 채널의 temporal filtering 특성" 으로 해석 가능 |
| neuron-wise tau | 위치/뉴런마다 필터가 달라져 레이어 대표 PSD 해석이 흐려짐 |

네 연구가 **레이어 출력의 PSD 대표화와 주파수 특성 분석** 이 목적이라면, CNN에서는 `channel-wise tau` 가 가장 균형이 좋습니다.

```text
MLP: feature-wise tau 가능
CNN: channel-wise tau 권장
CNN true neuron-wise tau: 가능하지만 해석성 낮음
```

---

## 한 줄 결론

네 말이 맞습니다. **CuPy backend까지 제대로 쓰려면 `MemoryModule` 기반 커스텀 뉴런을 만들고, multi-step FPTT/BPTT CuPy kernel에서 `w[n]` 또는 `alpha[n]` 를 뉴런별로 읽도록 구현해야 합니다.** 단순히 `ParametricLIFNode.w` 를 벡터로 바꾸는 방식은 torch backend용 임시 구현으로 보는 게 맞습니다.

---

**Sources:**

- [spikingjelly.activation_based.base module — spikingjelly  文档](https://spikingjelly.readthedocs.io/zh-cn/latest/APIs/spikingjelly.activation_based.base.html)
- [GitHub - fangwei123456/spikingjelly: SpikingJelly is an open-source deep learning framework for Spiking Neural Network (SNN) based on PyTorch. · GitHub](https://github.com/fangwei123456/spikingjelly)

## Prompt:
2026. 5. 22. 오후 7:42:21

아 뭐야 제공하는 레이어 api 가 아니면 cupy 못써?

## Response:
2026. 5. 22. 오후 7:42:25

> **Searching the web for SpikingJelly cupy backend documentation**
>
> [SpikingJelly is an open-source deep learning framework ...](https://github.com/fangwei123456/spikingjelly)
> [欢迎来到惊蜇(SpikingJelly)的文档— spikingjelly 文档](https://spikingjelly.readthedocs.io/)
> [SpikingJelly: An open-source machine learning ...](https://www.science.org/doi/10.1126/sciadv.adi1480)
> [SpikingJelly](https://ai-data-base.com/wp-content/uploads/2023/10/2310.16620v1.pdf)
> [[2310.16620] SpikingJelly: An open-source machine ...](https://arxiv.org/abs/2310.16620)
> [spikingjelly](https://pypi.org/project/spikingjelly/)
> [SpikingJelly](https://open-neuromorphic.org/neuromorphic-computing/software/snn-frameworks/spikingjelly/)
> [SpikingJelly: An open-source machine learning infrastructure ...](https://pmc.ncbi.nlm.nih.gov/articles/PMC10558124/)
> [spikingjelly - conda-forge](https://anaconda.org/conda-forge/spikingjelly)
>
> **검색: open source 문서 및 지원되는 백엔드**
>
> [Core Neuron Modules — spikingjelly  文档](https://spikingjelly.readthedocs.io/zh-cn/latest/APIs/spikingjelly.activation_based.neuron.core.html)
> [GitHub - fangwei123456/spikingjelly: SpikingJelly is an open-source deep learning framework for Spiking Neural Network (SNN) based on PyTorch. · GitHub](https://github.com/fangwei123456/spikingjelly)
>
> **백엔드 문서 세부 사항 찾기**
>
> [Core Neuron Modules — spikingjelly  文档](https://spikingjelly.readthedocs.io/zh-cn/latest/APIs/spikingjelly.activation_based.neuron.core.html)
>
> **찾기 CuPy 신경망 구현 튜토리얼**
>
> [Implement CUPY Neuron — spikingjelly  文档](https://spikingjelly.readthedocs.io/zh-cn/latest/tutorials/en/cupy_neuron.html)
>
> **확인 중: 레이어 API와 뉴런 API 차이**
>
> 우리가 다루고 있는 "레이어 API"와 "뉴런 API"에 대한 차이를 설명할 생각이에요. `nn.Module`을 정의할 때 `.cuda()`를 사용하면 CUDA를 사용하지만, cupy는 사용하지 않아요. SpikingJelly에서는 `backend='cupy'`를 설정할 때 `supported_backends` 목록에 'cupy'가 포함되어야만 됩니다. ParametricLIFNode에서는 `supported_backends` 목록을 확인하고 지원되는 백엔드를 사용해야 해요.
>
> **검토와 검증을 위한 SpikingJelly API 분석**
>
> [Core Neuron Modules — spikingjelly  文档](https://spikingjelly.readthedocs.io/zh-cn/latest/APIs/spikingjelly.activation_based.neuron.core.html)
>
> **SpikingJelly에서 Cupy 사용법과 백엔드 지원**
>
> "레이어 API"는 SpikingJelly의 `layer.Conv2d`를 의미할 수 있지만, 꼭 필요하진 않아요. SpikingJelly에서 `backend='cupy'`를 직접 사용하는 건 안 되지만, 커스텀 네이언에서 cupy를 사용할 수 있어요. 예를 들어, `BaseNode`를 서브클래스하여 `multi_step_forward`를 유지하거나, CUDA 커널을 사용하는 방법 등이 있어요. 최신 문서에 따르면 `ParametricLIFNode`는 학습 가능한 w를 포함하고, `self.supported_backends`와 `step_mode`에 따라서 백엔드가 다를 수 있어요.
>
> 29s 동안 생각함

질문을 두 갈래로 나눠서 봐야 합니다. **PyTorch CUDA를 쓰는 것** 과 **SpikingJelly의 `backend="cupy"` 고속 뉴런 커널을 쓰는 것** 은 같은 말이 아닙니다.

아니, **GPU 자체를 못 쓰는 건 아닙니다.**  
구분이 필요합니다.

# 핵심 구분

| 경우 | 가능 여부 | 의미 |
|---|---:|---|
| 커스텀 `nn.Module` 을 `.cuda()` 로 돌림 | 가능 | PyTorch CUDA 사용 |
| 커스텀 뉴런을 torch 연산으로 구현 | 가능 | GPU에서 돌아가지만 Python loop / PyTorch op 기반 |
| `backend="cupy"` 만 붙이면 자동 가속 | 불가능 | 해당 모듈이 CuPy backend를 구현해야 함 |
| SpikingJelly 기본 `LIFNode`, `ParametricLIFNode` 사용 | 가능 | 지원되는 경우 CuPy/Triton backend 사용 가능 |
| 커스텀 뉴런인데 CuPy backend까지 원함 | 가능하지만 직접 구현 필요 | FPTT/BPTT CUDA kernel 또는 SpikingJelly auto_cuda 구현 필요 |

즉 네가 말한 "제공하는 레이어 API가 아니면 CuPy 못써?" 에 대한 답은:

> **SpikingJelly의 `backend="cupy"` 고속 경로는, 그 뉴런 모듈이 CuPy backend를 구현해둔 경우에만 쓸 수 있다.**  
> 하지만 커스텀 뉴런도 CuPy를 못 쓰는 것은 아니고, 직접 CuPy/CUDA backend를 구현해야 한다.

---

# `backend="cupy"` 는 마법 스위치가 아님

SpikingJelly 공식 문서에서 `backend` 는 뉴런 레이어의 backend이고, **step mode에 따라 지원 backend가 다르며 `self.supported_backends` 를 출력해서 확인하라** 고 되어 있습니다. 지원되는 경우에만 `"cupy"` 또는 `"triton"` 이 빠릅니다. ([SpikingJelly](https://spikingjelly.readthedocs.io/zh-cn/latest/APIs/spikingjelly.activation_based.neuron.core.html))

즉 이런 식으로 확인해야 합니다.

```python
from spikingjelly.activation_based import neuron

lif = neuron.ParametricLIFNode(step_mode="m")
print(lif.supported_backends)
```

출력 안에 `"cupy"` 가 있으면:

```python
lif.backend = "cupy"
```

또는 생성 시:

```python
lif = neuron.ParametricLIFNode(
    init_tau=2.0,
    step_mode="m",
    backend="cupy"
)
```

처럼 쓸 수 있습니다.

---

# 기본 `LIFNode`, `ParametricLIFNode` 는 backend 인자가 있음

공식 API에서 `LIFNode` 는 `backend="torch"` 인자를 받고, `step_mode` 에 따라 `cupy` / `triton` backend가 가능할 수 있다고 설명합니다. ([SpikingJelly](https://spikingjelly.readthedocs.io/zh-cn/latest/APIs/spikingjelly.activation_based.neuron.core.html))

`ParametricLIFNode` 도 마찬가지로 `backend` 인자를 받습니다. 그리고 내부 동역학은

$$
\frac{1}{\tau}=\mathrm{Sigmoid}(w)
$$

이고 $w$ 가 학습 가능한 파라미터라고 문서에 명시되어 있습니다. ([SpikingJelly](https://spikingjelly.readthedocs.io/zh-cn/latest/APIs/spikingjelly.activation_based.neuron.core.html))

그러니까 **레이어 단위 tau 공유 PLIF** 는 그냥 이렇게 하면 됩니다.

```python
from spikingjelly.activation_based import neuron, surrogate

plif = neuron.ParametricLIFNode(
    init_tau=2.0,
    decay_input=True,
    v_threshold=1.0,
    v_reset=0.0,
    surrogate_function=surrogate.ATan(),
    detach_reset=False,
    step_mode="m",
    backend="cupy"
)
```

이건 SpikingJelly가 이미 제공하는 PLIF이므로 CuPy backend를 쓸 수 있습니다.

---

# 문제는 "뉴런별 독립 tau" 임

기본 `ParametricLIFNode` 는 PLIF 구조 자체는 제공하지만, 우리가 말한 것처럼 `w` 를 뉴런별 벡터/텐서로 확장하면 이야기가 달라집니다.

torch backend에서는 broadcasting 때문에 대충 이렇게 해도 돌아갈 가능성이 큽니다.

```python
plif.w = torch.nn.Parameter(torch.zeros(hidden_dim))
```

하지만 `backend="cupy"` 는 내부 fused CUDA kernel이 정해져 있습니다. 커널이 원래 scalar `w` 를 전제로 작성되어 있다면, 네가 `w` 를 벡터로 바꿔도 CuPy kernel이 자동으로 `w[i]` 를 읽어주지 않습니다.

즉:

```text
기본 ParametricLIFNode
    w: scalar
    cupy kernel: scalar w 전제

뉴런별 tau PLIF
    w: [N]
    cupy kernel: w[n] 를 읽도록 수정 필요
```

이 차이입니다.

---

# 커스텀 뉴런도 CuPy 가능은 함

SpikingJelly 문서에는 아예 **"Implement CUPY Neuron"** 튜토리얼이 있습니다. 이 튜토리얼은 커스텀 CuPy neuron을 구현하려면 element-wise CUDA kernel, `torch.autograd.Function` 의 custom backward, 그리고 `spikingjelly.activation_based.cuda_kernel.auto_cuda` API를 알아야 한다고 전제합니다. ([SpikingJelly](https://spikingjelly.readthedocs.io/zh-cn/latest/tutorials/en/cupy_neuron.html))

문서에서 CuPy neuron의 forward propagation through time은 기본적으로 다음 형태를 씁니다.

```text
v_init: [N]
x_seq: [T,N]
spike_seq: [T,N]
v_seq: [T,N]
```

여기서 다차원 뉴런이면 $N$ 은 flatten된 뉴런 수입니다. ([SpikingJelly](https://spikingjelly.readthedocs.io/zh-cn/latest/tutorials/en/cupy_neuron.html))

그리고 backward도 따로 필요합니다. 문서가 BPTT 입력으로 `grad_spike_seq`, `grad_v_seq` 를 받고, 출력으로 `grad_x_seq`, `grad_v_init` 를 계산한다고 설명합니다. ([SpikingJelly](https://spikingjelly.readthedocs.io/zh-cn/latest/tutorials/en/cupy_neuron.html))

뉴런별 tau면 여기에 추가로:

```text
grad_w: [N]
```

도 계산해야 합니다.

---

# 네 경우의 정확한 결론

## 1. 그냥 PLIF 쓰기

```python
neuron.ParametricLIFNode(step_mode="m", backend="cupy")
```

가능합니다. 이건 공식 API가 제공하는 범위입니다.

## 2. 뉴런별 tau를 torch backend로 쓰기

가능합니다. `w` 를 벡터/텐서 파라미터로 바꾸고 PyTorch broadcasting을 쓰면 됩니다.

```python
backend="torch"
```

이 경우 GPU에서도 돌아갑니다. 단, SpikingJelly CuPy fused kernel은 아닙니다.

## 3. 뉴런별 tau를 CuPy backend로 쓰기

그냥은 안 됩니다.  
직접 커널이 `w[n]` 를 읽도록 만들어야 합니다.

대략 forward kernel에서 이런 식이어야 합니다.

```cpp
int n = index;
float alpha = 1.0f / (1.0f + expf(-w[n]));
h_seq[t] = v_v_seq[t] + alpha * (x_seq[t] - (v_v_seq[t] - v_reset));
```

그리고 backward에서 `grad_w[n]` 도 시간축으로 누적해야 합니다.

---

# 실용적인 선택

너는 지금 연구용으로 PSD 분석과 뉴런 동역학 해석을 같이 보고 있으니까, 현실적인 선택지는 이겁니다.

| 목표 | 추천 |
|---|---|
| 빠르게 실험 | 기본 `ParametricLIFNode(step_mode="m", backend="cupy")` |
| 뉴런별 tau 아이디어 검증 | torch backend 커스텀 PLIF |
| 논문용 최종 대규모 실험 | 검증 후 CuPy kernel 구현 |
| 해석성 중시 | layer-wise 또는 channel-wise tau |

한 문장으로 정리하면:

**제공 API가 아니면 GPU를 못 쓰는 게 아니라, SpikingJelly의 `backend="cupy"` 고속 뉴런 커널을 자동으로 못 쓰는 겁니다.** 커스텀 뉴런은 torch backend로 먼저 만들고, 성능이 필요하면 CuPy FPTT/BPTT kernel을 별도로 구현해야 합니다.

---

**Sources:**

- [Core Neuron Modules — spikingjelly  文档](https://spikingjelly.readthedocs.io/zh-cn/latest/APIs/spikingjelly.activation_based.neuron.core.html)

## Prompt:
2026. 5. 22. 오후 7:44:04

cupy 벡엔드 안쓰면 그냥 torch 로만 구현한거나 메모리 모듈로 구현한거나 속도차이 없지 않아?

## Response:
2026. 5. 22. 오후 7:44:06

> **답변 요약**
>
> 사용자에게는 SpikingJelly, torch와 MemoryModule의 속도 차이에 대해 설명해야 할 것 같습니다. MemoryModule은 상태 관리에 집중되고 속도에 큰 차이는 없습니다. 성능은 벡터화, torch 루프 사용, 루프 최적화 및 메모리 관리 방식에 따라 달라집니다. SpikingJelly의 MultiStep 모듈은 최적화 가능하지만 기본적인 성능 개선은 제한적입니다.
>
> **MemoryModule와 속도 차이 설명**
>
> LIF 의존성으로 인해 torch 백엔드는 시간이 지남에 따라 루프를 돌며 여러 커널을 실행합니다. cupy 또는 triton은 커널 출시 오버헤드를 줄이며 시간을 통합합니다. `MemoryModule`은 상태 초기화 용도이며 속도 차이는 큰 차이가 없지만, `reset`, `detach`, `set_step_mode` 등의 기능을 제공합니다. 작은 네트워크나 긴 시간 T에서는 이 오버헤드가 더욱 중요할 수 있습니다.
>
> **SpikingJelly MemoryModule 및 Torch 성능 비교**
>
> [SpikingJelly is an open-source deep learning framework ...](https://github.com/fangwei123456/spikingjelly)
> [spikingjelly.activation_based.base 源代码](https://spikingjelly.readthedocs.io/zh-cn/latest/_modules/spikingjelly/activation_based/base.html)
> [Interoperability — CuPy 13.4.0 documentation](https://docs.cupy.dev/en/v13.4.0/user_guide/interoperability.html)
> [SpikingJelly](https://ai-data-base.com/wp-content/uploads/2023/10/2310.16620v1.pdf)
> [spikingjelly](https://pypi.org/project/spikingjelly/)
> [SpikingJelly: An open-source machine learning ...](https://www.science.org/doi/10.1126/sciadv.adi1480)
> [Distributed communication package - torch.distributed](https://docs.pytorch.org/docs/stable/distributed.html)
> [CuPy: NumPy & SciPy for GPU](https://cupy.dev/)
> [SpikingJelly: An open-source machine learning infrastructure ...](https://pmc.ncbi.nlm.nih.gov/articles/PMC10558124/)
>
> **MemoryModule와 nn.Module 비교**
>
> `MemoryModule`은 상태 초기화와 관련이 있고, `nn.Module`과의 비교에서 속도 차이는 수학적 구현에 따라 달라집니다. `cupy` 백엔드를 사용하지 않으면 wrapper 차이는 미미하며, 메서드와 상태 초기화로 인한 약간의 속도 저하가 있을 수 있습니다. `functional.multi_step_forward`에서는 잠재적인 차이가 있을 수 있습니다.
>
> 21s 동안 생각함

맞습니다. **`cupy` backend를 안 쓸 거면 `MemoryModule` 로 구현한 것과 그냥 PyTorch `nn.Module` 로 구현한 것 사이에 본질적인 속도 이득은 거의 없습니다.**

정확히 나누면 이렇습니다.

# 1. `MemoryModule` 은 가속용이 아님

`MemoryModule` 의 역할은 대략 이겁니다.

```text
상태변수 등록
reset 관리
detach 관리
step_mode 관리
SpikingJelly 생태계와 호환
```

예를 들면 막전위 $v$ 를

```python
self.register_memory("v", 0.0)
```

처럼 등록하고, `reset()` 하면 초기 상태로 돌리는 식입니다. 공식 문서도 `MemoryModule` 을 상태를 갖는 모듈 예제로 설명하고, `register_memory` 를 통해 stateful variable을 등록하는 구조를 보여줍니다. ([SpikingJelly](https://spikingjelly.readthedocs.io/zh-cn/latest/_modules/spikingjelly/activation_based/base.html?utm_source=chatgpt.com))

즉 **속도를 빠르게 해주는 클래스가 아니라 상태 관리를 표준화해주는 클래스** 입니다.

---

# 2. 같은 torch 연산이면 속도는 거의 같다

예를 들어 둘 다 내부에서 이렇게 돈다고 하겠습니다.

```python
for t in range(T):
    h = v + alpha * (x_seq[t] - v)
    spike = surrogate_heaviside(h - v_threshold)
    v = h * (1.0 - spike)
```

이걸

```python
class MyLIF(nn.Module):
    ...
```

로 만들든,

```python
class MyLIF(base.MemoryModule):
    ...
```

로 만들든, 실제 계산은 결국 PyTorch tensor 연산입니다.

따라서 병목은 보통:

```text
1. 시간축 Python loop
2. 매 time step마다 발생하는 여러 torch CUDA kernel launch
3. surrogate gradient 연산
4. v_seq, spike_seq 저장 메모리
```

입니다.

`MemoryModule` 을 쓴다고 이 병목이 사라지지 않습니다.

---

# 3. SpikingJelly에서 빠른 건 `MemoryModule` 이 아니라 `cupy` / `triton` backend

SpikingJelly 쪽 설명도 구조가 명확합니다. `torch` backend는 코딩과 디버깅이 쉽고, `cupy` 또는 `triton` backend는 더 빠른 학습 속도를 위한 backend라고 설명합니다. ([GitHub](https://github.com/fangwei123456/spikingjelly?utm_source=chatgpt.com))

과거 문서에서도 multi-step LIF에서 `cupy` backend가 naive PyTorch backend보다 훨씬 빠르다고 설명합니다. ([SpikingJelly](https://spikingjelly.readthedocs.io/zh-cn/0.0.0.0.6/clock_driven_en/11_cext_neuron_with_lbl.html?utm_source=chatgpt.com))

즉 속도 차이는 여기서 납니다.

```text
torch backend:
    time step마다 PyTorch op 여러 개 실행

cupy/triton backend:
    시간축 반복과 뉴런 연산을 CUDA kernel 쪽으로 fusion
```

핵심은 **fusion** 입니다.  
`MemoryModule` 을 쓴다고 fusion이 생기지는 않습니다.

---

# 4. 그럼 `MemoryModule` 을 왜 쓰냐?

속도 때문이 아니라 **SpikingJelly 스타일에 맞추기 위해서** 씁니다.

예를 들어 커스텀 뉴런을 만들 때:

```text
- functional.reset_net(model) 과 호환
- 상태변수 v 관리
- step_mode="s" / "m" 관리
- store_v_seq 같은 기록 구조 만들기 쉬움
- 나중에 cupy backend 추가할 때 구조 잡기 쉬움
```

이런 장점이 있습니다.

하지만 **torch-only 실험에서 속도만 보면 그냥 `nn.Module` 로 잘 짠 코드와 거의 같습니다.**

---

# 5. 오히려 그냥 `nn.Module` 이 약간 더 빠를 수도 있음

아주 작은 네트워크, 작은 batch, 작은 hidden dimension에서는 Python overhead가 상대적으로 커집니다. 이 경우 `MemoryModule`, `BaseNode` 같은 추상화 계층이 약간의 오버헤드를 만들 수 있습니다.

하지만 일반적인 GPU 학습에서는 병목이 대개 tensor 연산과 kernel launch라서, 이 차이는 크지 않습니다.

정리하면:

| 구현 | 속도 | 장점 |
|---|---:|---|
| 순수 `nn.Module` torch 구현 | 비슷하거나 약간 빠를 수 있음 | 단순함 |
| `MemoryModule` torch 구현 | 거의 비슷 | reset/state 관리 편함 |
| SpikingJelly 기본 `LIFNode` torch backend | 비슷 | 검증된 구현 |
| SpikingJelly 기본 `LIFNode` cupy/triton backend | 빠름 | fused backend |
| 커스텀 뉴런 + 직접 cupy kernel | 빠를 수 있음 | 구현 난도 높음 |

---

# 6. 네 경우에는 이렇게 판단하면 됨

## 뉴런별 $\tau_i$ 를 빠르게 검증하고 싶다

그냥 torch backend로 구현하면 됩니다.

```python
class NeuronwisePLIF(nn.Module):
    ...
```

또는

```python
class NeuronwisePLIF(base.MemoryModule):
    ...
```

둘 다 괜찮습니다.

속도보다는 구현 편의성을 보고 고르면 됩니다.

---

## SpikingJelly 전체 구조에 끼워 넣고 싶다

이 경우는 `MemoryModule` 이 낫습니다.

```python
from spikingjelly.activation_based import base

class NeuronwisePLIF(base.MemoryModule):
    ...
```

이렇게 하면 `functional.reset_net(model)` 같은 SpikingJelly 유틸과 잘 맞습니다.

---

## 대규모 실험에서 속도가 중요하다

그때는 torch 구현으로는 한계가 있습니다.

공식 튜토리얼도 커스텀 CuPy 뉴런을 구현하려면 CUDA kernel, `torch.autograd.Function`, SpikingJelly의 `auto_cuda` 구조를 알아야 한다고 설명합니다. ([SpikingJelly](https://spikingjelly.readthedocs.io/zh-cn/latest/tutorials/cn/cupy_neuron.html?utm_source=chatgpt.com))

즉 진짜 속도 최적화는:

```text
MemoryModule 사용 여부
```

가 아니라

```text
torch loop냐
fused cupy/triton kernel이냐
```

에서 갈립니다.

---

# 결론

네 말이 맞습니다.

**`cupy` / `triton` backend를 안 쓸 거면 `MemoryModule` 로 구현하든 그냥 PyTorch `nn.Module` 로 구현하든 속도 차이는 본질적으로 거의 없습니다.**

`MemoryModule` 은 **속도용이 아니라 SpikingJelly식 상태 관리와 reset 호환성용** 입니다.

그래서 네가 뉴런별 $\tau_i$ 를 실험하려는 단계라면:

```text
1. 먼저 torch-only로 단순 구현
2. 성능/해석 결과 확인
3. 쓸 가치가 있으면 cupy/triton kernel 구현 고려
```

이 순서가 맞습니다.

---

**Sources:**

- [spikingjelly.activation_based.base 源代码](https://spikingjelly.readthedocs.io/zh-cn/latest/_modules/spikingjelly/activation_based/base.html?utm_source=chatgpt.com)
- [SpikingJelly is an open-source deep learning framework ...](https://github.com/fangwei123456/spikingjelly?utm_source=chatgpt.com)

## Prompt:
2026. 5. 22. 오후 7:45:27

멀티스텝도 효과가 없나?

## Response:
2026. 5. 22. 오후 7:45:33

> **웹을 통한 처리 속도 향상에 대한 설명**
>
> 사용자가 묻고 있는 "멀티스텝도 효과가 없나?"는 cupy와 torch에 관련된 질문입니다. 상태가 업그레이드와 관련되었을 수 있기 때문에, 이 경우 최신 문서를 참고해야 할 필요가 있습니다. 멀티스텝은 속도나 편리성에 영향을 미칠 수 있으며, 구현 방식에 따라 다를 수 있습니다.
>
> **검색 중 SpikingJelly 멀티스텝 동작 문서**
>
> [SpikingJelly is an open-source deep learning framework ...](https://github.com/fangwei123456/spikingjelly)
> [spikingjelly.activation_based.base 源代码](https://spikingjelly.readthedocs.io/zh-cn/latest/_modules/spikingjelly/activation_based/base.html)
> [spikingjelly](https://pypi.org/project/spikingjelly/)
> [SpikingJelly](https://ai-data-base.com/wp-content/uploads/2023/10/2310.16620v1.pdf)
>
> **문서 및 백엔드 관련 자료 확인**
>
> 결론을 위해, 공식 문서에서 LIF 지원 백엔드와 멀티스텝 토치 루프에 대한 내용을 찾아봐야 할 것 같습니다. 검색 결과를 참고하여 관련 인용을 추가할 수 있고, 이를 통해 더 정확한 정보를 제공할 수 있을 것 같습니다.
>
> **참조 및 문서 검색**
>
> [spikingjelly.activation_based.base — spikingjelly  文档](https://spikingjelly.readthedocs.io/zh-cn/latest/_modules/spikingjelly/activation_based/base.html)
>
> **멀티스텝의 효과 및 최적화**
>
> 멀티스텝을 사용할 때, 순차적 업데이트가 필요한 경우 (예: SpikingJelly의 `step_mode='m'` 적용) 성능 향상이 있을 수 있습니다. 그러나 일반적인 토치 신경망에서는 큰 속도 차이가 없고, 이 방식은 주로 모델에 따라 유용한 경우가 많습니다. 특히 CNN이나 Linear 레벨에서 큰 차이가 납니다.
>
> 26s 동안 생각함

이 질문은 핵심이 있습니다. **multi-step 자체는 효과가 있습니다.** 다만 그 효과가 "뉴런 동역학 계산을 fused CUDA로 빠르게 만든다" 는 뜻인지는 backend에 따라 다릅니다.

맞습니다. **커스텀 뉴런을 `torch` backend로만 구현할 경우, multi-step이 뉴런 계산 자체를 크게 빠르게 만들지는 않습니다.**  
하지만 **전체 네트워크 기준으로는 효과가 있을 수 있습니다.** 핵심은 "어느 부분을 multi-step으로 처리하느냐" 입니다.

# 1. 뉴런 자체만 보면 효과는 작다

`MemoryModule` 의 기본 `multi_step_forward` 는 공식 소스 기준으로 내부에서 그냥 시간축을 돌면서

```python
for t in range(T):
    y = self.single_step_forward(x_seq[t])
```

를 호출합니다. 즉 multi-step 입력 $[T,N,*]$ 을 받기는 하지만, torch backend에서는 시간 방향으로 순차 실행합니다. 문서에도 `multi_step_forward` 는 `single_step_forward` 를 $T$ 번 호출해서 구현된다고 되어 있습니다. ([SpikingJelly](https://spikingjelly.readthedocs.io/zh-cn/latest/_modules/spikingjelly/activation_based/base.html))

그래서 커스텀 뉴런이 이런 구조라면:

```python
class MyNeuron(base.MemoryModule):
    def multi_step_forward(self, x_seq):
        y_seq = []
        for t in range(x_seq.shape[0]):
            y_seq.append(self.single_step_forward(x_seq[t]))
        return torch.stack(y_seq)
```

이건 사실상 네가 직접 밖에서

```python
for t in range(T):
    y_t = neuron(x_seq[t])
```

하는 것과 속도 차이가 거의 없습니다.

왜냐하면 LIF/PLIF/RF 같은 뉴런은 상태가 있습니다.

$$
V_t=f(X_t,V_{t-1})
$$

이므로 $t=10$ 을 계산하려면 $t=9$ 의 $V$ 가 필요합니다. 따라서 torch tensor만으로 시간축 전체를 완전히 병렬화하기 어렵습니다.

---

# 2. 진짜 속도 차이는 `cupy` / `triton` 에서 난다

SpikingJelly 문서도 multi-step neuron의 장점 중 하나로 `cupy` backend 사용을 말합니다. `cupy` backend는 여러 연산을 하나의 CUDA kernel로 fusion하기 때문에 naive PyTorch backend보다 빠르다고 설명합니다. ([SpikingJelly](https://spikingjelly.readthedocs.io/zh-cn/0.0.0.0.6/clock_driven_en/11_cext_neuron_with_lbl.html))

즉 뉴런 계산 속도 관점에서는:

```text
single-step torch loop
≈ multi-step torch loop
< multi-step cupy/triton fused kernel
```

입니다.

따라서 **커스텀 뉴런 + torch-only multi-step** 은 속도보다는 구조 정리용입니다.

---

# 3. 그래도 multi-step이 완전히 무의미한 건 아님

multi-step의 효과는 뉴런보다 **Conv, Linear, BatchNorm 같은 무상태층(stateless layer)** 에서 큽니다.

예를 들어 나쁜 구조는 이겁니다.

```python
for t in range(T):
    x = conv(x_seq[t])
    x = lif(x)
    x = fc(x)
```

이러면 `conv`, `fc` 도 $T$ 번 호출됩니다.

하지만 더 좋은 구조는 이겁니다.

```python
x_seq = conv_multi_step(x_seq)   # [T,B,C,H,W] 전체 처리
s_seq = lif_multi_step(x_seq)    # 뉴런은 시간축 순차 처리
y_seq = fc_multi_step(s_seq)     # [T,B,F] 전체 처리
```

여기서 `conv` 나 `fc` 는 시간축과 배치축을 합쳐서

$$
[T,B,*]\rightarrow[TB,*]
$$

처럼 처리할 수 있습니다. 그러면 Conv/Linear를 $T$ 번 호출하지 않고 **한 번의 큰 배치 연산** 으로 처리할 수 있습니다.

SpikingJelly의 `SeqToANNContainer` 도 시간 차원과 배치 차원을 합쳐 forward하는 컨테이너로 설명되어 있습니다. ([SpikingJelly](https://spikingjelly.readthedocs.io/zh-cn/latest/APIs/spikingjelly.activation_based.layer.html))

---

# 4. 정리하면 이렇게 봐야 함

| 대상 | torch multi-step 효과 |
|---|---:|
| 커스텀 LIF/PLIF/RF 뉴런 자체 | 속도 효과 작음 |
| 상태 있는 recurrent 동역학 | 시간축 병렬화 어려움 |
| Conv2d, Linear, BatchNorm | 효과 있음 |
| 전체 네트워크 구조 정리 | 효과 있음 |
| SpikingJelly reset/state 호환성 | 효과 있음 |
| `cupy` / `triton` backend 사용 | 효과 큼 |

즉 **뉴런 자체만 보면 multi-step torch는 빠른 게 아니라 편한 것** 에 가깝습니다.

하지만 **네트워크 전체에서 Conv/Linear를 time loop 밖으로 빼면 multi-step은 충분히 의미가 있습니다.**

---

# 5. 네가 커스텀 뉴런을 만들 때 추천 구조

뉴런 자체는 이렇게 짜는 게 맞습니다.

```python
class NeuronwisePLIF(base.MemoryModule):
    def __init__(self, ...):
        super().__init__()
        self.step_mode = "m"
        self.backend = "torch"
        self.register_memory("v", 0.0)

    def single_step_forward(self, x):
        # x: [B, ...]
        ...

    def multi_step_forward(self, x_seq):
        # x_seq: [T, B, ...]
        y_seq = []
        for t in range(x_seq.shape[0]):
            y_seq.append(self.single_step_forward(x_seq[t]))
        return torch.stack(y_seq, dim=0)
```

이 구조는 속도 때문이라기보다:

```text
1. SpikingJelly의 step_mode="m" 흐름과 맞음
2. functional.reset_net(model) 과 맞음
3. 나중에 cupy/triton kernel 붙이기 쉬움
4. Conv/Linear multi-step 처리와 조합하기 쉬움
5. PSD 분석에서 [T,B,*] 형태를 유지하기 좋음
```

때문에 쓰는 겁니다.

---

# 6. 네 연구 코드 기준 판단

네가 **뉴런별 tau** 를 검증하려는 단계라면:

```text
1단계: torch multi-step 커스텀 뉴런으로 구현
2단계: Conv/Linear는 step_mode="m" 또는 SeqToANN 방식으로 시간축 batch화
3단계: 병목 측정
4단계: 뉴런이 병목이면 cupy/triton 커널 구현 고려
```

이 순서가 맞습니다.

반대로 처음부터 CuPy 커널까지 짜는 건 과합니다. 뉴런별 $\tau_i$ 가 성능이나 해석에서 실제로 의미 있는지 먼저 확인해야 합니다.

---

## 한 줄 결론

**torch backend만 쓸 경우, multi-step은 뉴런 계산 자체를 크게 빠르게 하지는 않습니다.**  
하지만 **Conv/Linear 같은 무상태층을 $[T,B,*]\rightarrow[TB,*]$ 로 한 번에 처리하게 만들 수 있어서 전체 네트워크 속도에는 효과가 있을 수 있습니다.**

---

**Sources:**

- [spikingjelly.activation_based.base — spikingjelly  文档](https://spikingjelly.readthedocs.io/zh-cn/latest/_modules/spikingjelly/activation_based/base.html)

## Prompt:
2026. 5. 22. 오후 7:49:29

이거 말인데 지금 psd 분석하는 파트 어떻게 되어있는거야? 저장한 모델을 읽어서 입력주고 출력을 받아야하는데 이거에 맞게 미리 구현했어야하나?

[Attachment: psd_54_matrix_analysis_spec_aligned(1).zip]

## Response:
2026. 5. 22. 오후 7:49:31

> **압축파일 내용 목록 출력**
>
>
> **압축 해제 후 파일 추출**
>
>
> **Inspecting psd_analysis.py 코드**
>
>
> **엘리드된 라인 출력 확인**
>
>
> **Inspecting main function for processing and saving**
>
>
> **검색 및 코드 출력**
>
>
> **검사 훈련 체크포인트 메타데이터**
>
>
> **파일 검사 및 출력**
>
>
> **특정 부분만 출력하기**
>
>
> **모델 빌더 이어서 보기**
>
>
> **LIF 및 CNN2D 코드 라인 출력**
>
>
> **데이터 레지스트리 검사 및 형태 정규화**
>
>
> **데이터 파일 일부 줄 출력**
>
>
> **검사 모델 구성 및 PSD 분석**
>
>
> **분석 함수 코드 및 출력 확인**
>
>
> **Bash 스크립트 분석 및 출력**
>
>
> **Inspecting bash psd_analysis script**
>
>
> **분석 프로세스 설명**
>
> 현재 PSD 분석은 체크포인트 기반입니다. 모델을 .pt 파일로 로드하고, 메타데이터에서 모델을 재구성하여 추론을 실행합니다. 모델은 `capture_hidden` 프로토콜이 구현되어야 하며, 새로운 맞춤형 모델은 추적 기록을 노출하도록 구현돼야 합니다.
>
> **모델 아키텍처 점검**
>
> 모델 구조를 복구하려면 `model_token`과 `model_config`가 필요하고, `capture_hidden=True`를 지원해야 합니다. 현재 코드에서는 `bundle.training_view_name`을 사용하고, 입력과 기록을 모델로 전달합니다. 설정된 출력 레이어가 분석에 제대로 사용되지 않을 수 있으므로 점검이 필요해 보입니다.
>
> **Inspecting PSD analysis and model checkpoints**
>
>
> **SpikingJelly와 PSD 분석 연결**
>
> SpikingJelly의 LIF 노드 구현에 대해 이야기했는데, PSD 분석은 현재 프로젝트 래퍼에 의존하고 있습니다. 모델 레이어를 SpikingJelly API로 교체하면 `capture_hidden`을 위한 래퍼나 포워드 훅을 구현해야 할 수도 있습니다. 기존의 `snnbench` 어댑터를 점검해보는 것이 좋을 것 같습니다.
>
> **모델 어댑터 코드 검사**
>
>
> **체크포인트 저장 확인 필요**
>
> 모델 훈련이 계속 진행된 후 체크포인트 저장 부분을 점검해야 할 것 같습니다. 이 부분이 잘 구현되어 있는지 확인해보면 좋겠네요.
>
> **모델 훈련 체크포인트 저장 코드 확인**
>
>
> **모델 저장 및 분석 프로토콜 설명**
>
> 모델을 저장할 때 활성화 출력을 저장하지 않으며, 체크포인트에는 메타데이터와 상태 딕셔너리만 저장됩니다. 분석은 모델을 코드로 재구성하고 실행합니다. 모델 클래스 변경 시, `capture_hidden` 지원 및 프로토콜을 준수해야 하며, 기존 체크포인트도 오래된 모델을 통해 활용할 수 있습니다.
>
> **후크 점검 필요**
>
> 앞으로 진행되는 작업에서 필요한 후크가 있는지 점검해야 할 것 같습니다. 특정 후크가 누락되었거나 구현되지 않은 부분이 있을 수 있으니, 이를 확인하는 것이 중요할 것 같아요.
>
> **검색하여 후크 확인하기**
>
>
> **PSD 분석 흐름 및 구현 안내**
>
> 현재 PSD 분석 스크립트의 진입점은 `bash/psd_analysis.sh`에서 `python -m src.psd_analysis ...`를 실행하는 것입니다. 이후 진행 흐름은 모델과 체크포인트를 불러와 분석하는 단계로 나뉘며, `build_snn_classifier` 등 과정을 통해 모델이 재구성됩니다. 새 SpikingJelly LIF 커스텀도 이에 맞춰야 합니다.
>
> **모델 클래스에서 트레이스 캡처 및 alpha 처리**
>
> 모델 클래스에서 트레이닝/체크포인트 전에는 `state_dict` 키와 `model_token/model_config`가 일치해야 하므로 트레이스 캡처를 구현해야 합니다. 하지만 트레이닝 중 캡처는 저장할 필요가 없습니다. `LIFLayer`는 이미 각 뉴런 및 채널에 대한 alpha 값을 저장하고 있으며, PSD 분석기에서 이를 추출할 수 있습니다.
>
> **training.py에서 capture_hidden 처리**
>
> 정규 훈련 중에는 `capture_hidden=False`가 기본값입니다(정규화 활성화 시 제외). 현재 훈련의 PSD 정규화는 `capture_hidden`을 사용하지만, 저장된 체크포인트에는 히든 데이터가 저장되지 않습니다. PSD 정규화 없이 훈련되어도, 분석에서는 `capture_hidden=True`가 잘 작동합니다. `training.py`에서 상세히 확인해야 합니다.
>
> **Inspecting capture_hidden setting in training.py**
>
>
> **_reset_stateful_model 점검 필요**
>
> `_reset_stateful_model`을 점검해봐야 할 것 같습니다. 이 함수가 현재 코드에서 어떤 역할을 하고 있는지 확인하고, 모델의 상태를 리셋하는 과정이 정상적으로 이루어지는지 살펴보아야 할 것 같아요.
>
> **모델 초기화 상태 점검**
>
>
> **Inspecting _reset_stateful_model 함수**
>
>
> **상태 초기화 필요성 언급**
>
> 분석 코드에서는 `reset_state`를 호출하지 않습니다. 현재 모델은 `forward` 호출 시 각기 다른 상태를 유지하지 않지만, `SNNClassifier`는 `forward` 내부에서 상태를 초기화합니다. SpikingJelly의 `MemoryModule` 같은 상태 가진 모듈은 매 배치마다 혹은 `forward` 내에서 상태 초기화가 필요합니다.
>
> **Inspecting reset_state method in SNNClassifier**
>
>
> **분석에서 모델 입력 처리**
>
> 현재 분석에서 모델 입력에 해당하는 `bundle.training_view_name`을 사용하고 있습니다. 이는 모델에 데이터가 전달되어야 하기 때문이며, 그 후 트레이스가 PSD 맵으로 변환됩니다. 별도의 `psd_view_name`을 사용하면 안 될 것 같지만, 현 코드에서는 문제가 없을 것으로 보입니다.
>
> 3m 3s 동안 생각함

업로드한 zip 기준으로 실제 구현을 확인하겠습니다. 확인할 지점은 `psd_analysis` 실행 흐름, checkpoint 로딩, 모델 재구성, 입력 batch 주입, 레이어 출력 수집 방식입니다.

현재 구현을 보면 "저장된 activation을 읽는 방식" 이 아니라, **checkpoint를 읽고 모델을 재구성한 뒤 probe 입력을 다시 forward해서 activation trace를 새로 뽑는 방식** 입니다. 따라서 저장 시점에 activation을 저장할 필요는 없지만, 모델 forward가 분석용 trace를 반환하도록 미리 맞춰져 있어야 합니다.

## 결론

현재 zip의 `psd_analysis` 는 이미 네가 말한 방식으로 되어 있습니다.

```text
저장된 checkpoint(.pt) 읽기
→ checkpoint metadata로 모델 재구성
→ state_dict 로드
→ prepared probe set에서 입력 batch 생성
→ model(input, capture_hidden=True) 실행
→ input / hidden layer / output trace 수집
→ trace를 (samples, rows, time) PSD map으로 변환
→ PSD curve / dispersion / distance CSV 저장
```

즉 **activation을 checkpoint에 미리 저장하는 구조가 아닙니다.**  
분석 시점에 checkpoint 모델을 다시 forward해서 trace를 새로 뽑습니다.

다만 중요한 전제가 있습니다.

> 현재 `psd_analysis.py` 는 아무 PyTorch 모델이나 자동으로 hook해서 분석하는 범용 분석기가 아닙니다.  
> 모델이 프로젝트가 정한 `capture_hidden` / `ForwardResult` / `LayerRecord` 인터페이스를 구현해야 합니다.

---

# 1. 현재 checkpoint 저장 구조

`src/model_training.py` 에서 checkpoint payload를 이렇게 저장합니다.

관련 위치:

```text
src/model_training.py
- 180-215: _checkpoint_payload()
- 561-579: checkpoint 저장
```

checkpoint 안에는 최소한 이런 게 들어갑니다.

```python
{
    "epoch": ...,
    "model_token": ...,
    "model_config": ...,
    "state_dict": model.state_dict(),
    "readout_config": ...,
    "dataset_token": ...,
    "axis_metadata_ref": ...,
    ...
}
```

즉 모델 객체 전체를 pickle로 저장하는 게 아니라,

```text
모델을 재구성할 metadata
+ state_dict
```

를 저장합니다.

그래서 분석할 때는 **같은 모델 생성 코드가 있어야** 합니다.

---

# 2. PSD analysis에서 모델을 어떻게 읽는가

`src/psd_analysis.py` 의 핵심은 여기입니다.

```text
src/psd_analysis.py
- 56-60: torch.load 로 checkpoint 읽기
- 184-228: checkpoint metadata로 모델 재구성
```

실제 흐름은 이렇습니다.

```python
payload = torch.load(checkpoint_path, map_location="cpu")

model_token = payload["model_token"]
model_config = payload["model_config"]
readout_config = payload["readout_config"]

spec = canonicalize_model_token(model_token)

model = build_snn_classifier(
    model_token=spec,
    input_dim=model_config["input_dim"],
    sequence_length=model_config["sequence_length"],
    num_classes=model_config["num_classes"],
    input_shape=model_config.get("input_shape"),
    arch_spec=...,
    output_layer_overrides=readout.output_layer_overrides(),
    v_th=...,
).to(device)

model.load_state_dict(payload["state_dict"])
model.eval()
```

즉 PSD analysis는 checkpoint 안의 `model_token`, `model_config`, `readout_config` 를 보고 **동일한 모델 구조를 다시 만든 뒤** `state_dict` 를 입힙니다.

---

# 3. 입력은 어떻게 넣는가

관련 위치:

```text
src/psd_analysis.py
- 1170-1182: train/test split 각각에서 probe subset 분석
- 255-271: balanced_global probe subset 구성
- 231-237: 모델 입력 형태로 변환
```

분석 대상은 full train/test 전체가 아니라, 각 split에서 seed 기반으로 뽑은 `balanced_global` probe subset입니다.

```python
for split_name, split_dataset in (("train", bundle.train_dataset), ("test", bundle.test_dataset)):
    analysis_dataset = dataset_for_view(split_dataset, bundle.training_view_name)
    maps_by_key.update(
        _collect_signal_maps(...)
    )
```

입력 변환은 모델 family에 따라 다릅니다.

```python
if model family in {"cnn_lif", "cnn_rf"}:
    tensor = torch.as_tensor(inputs)
else:
    tensor = canonicalize_model_input_batch(...)
```

정리하면:

| 모델 | 분석 입력 형태 |
|---|---|
| dense SNN | $(B,T,C)$ |
| CNN SNN | $(B,T,C,H,W)$ |

여기서 중요한 점은 `psd_view_name` 을 직접 먹이는 게 아니라, **모델이 실제로 받는 training view를 먹인 뒤** 그 trace를 PSD map으로 바꾼다는 것입니다.

---

# 4. hidden output은 어떻게 받는가

핵심 호출은 이겁니다.

```text
src/psd_analysis.py
- 365-383: 일반 경로
- 409-449: low_vram 경로
```

실제 호출:

```python
result = model(model_inputs, capture_hidden=True)
```

그리고 `result` 에서 다음을 꺼냅니다.

```python
result.input_record
result.hidden_records
result.output_record
```

프로젝트 내부 표준은 `src/model/snn_builder.py` 에 있습니다.

```text
src/model/snn_builder.py
- 35-51: LayerRecord, ForwardResult 정의
```

구조는 대략 이겁니다.

```python
@dataclass
class LayerRecord:
    layer_name: str
    membrane: torch.Tensor
    spike: torch.Tensor
    layer_input: torch.Tensor | None = None

@dataclass
class ForwardResult:
    hidden_records: list[LayerRecord]
    output_record: LayerRecord
    input_record: torch.Tensor | None = None
```

따라서 모델 forward는 반드시 이런 식으로 동작해야 합니다.

```python
result = model(x, capture_hidden=True)

result.hidden_records[i].membrane
result.hidden_records[i].spike
result.hidden_records[i].layer_input
result.output_record.membrane
result.output_record.spike
```

이 인터페이스가 없으면 현재 `psd_analysis.py` 는 작동하지 않습니다.

---

# 5. 기존 커스텀 LIF/RF는 이미 이 구조에 맞춰져 있음

예를 들어 dense LIF는:

```text
src/neurons/LIF_neuron.py
- 131-174: forward(..., return_traces=False)
```

여기서 `return_traces=True` 일 때:

```python
return membrane_seq, spike_seq
```

를 반환하고, 내부에서:

```python
self._last_layer_input = ...
```

도 저장합니다.

`SNNClassifier.forward()` 는 이걸 받아서 `LayerRecord` 로 감쌉니다.

관련 위치:

```text
src/model/snn_builder.py
- 457-479: dense SNN forward
```

CNN도 동일합니다.

```text
src/neurons/cnn2d.py
- 131-163: CNN2DLIFLayer.forward()
- 297-353: CNN2DRFLayer.forward()
```

CNN forward는 $(B,T,C,H,W)$ 를 받아서, conv를 시간-배치 flatten 방식으로 처리한 뒤, 시간 루프에서 membrane/spike trace를 쌓습니다.

---

# 6. PSD 계산은 어떤 trace에 대해 하나

관련 위치:

```text
src/psd_analysis.py
- 284-335: _maps_from_record()
```

각 hidden layer에서 기본으로 저장하는 것은 다음입니다.

```text
layer_input
membrane
spike
```

그리고 일부 모델은 추가로:

```text
x_layer
i_current
z_gate
y_mem
y_spike
readout_mem
```

도 저장합니다.

그 뒤 모든 trace는 PSD 계산 전에 공식 map 형태로 바뀝니다.

```text
(samples, rows, time)
```

관련 위치:

```text
src/signal/psd_utils.py
- 162-179: trace_tensor_to_channel_major_maps()
```

예를 들면:

| 원래 trace | PSD map |
|---|---|
| $(B,T,C)$ | $(B,C,T)$ |
| $(B,T,C,H,W)$ | $(B,C \times H \times W,T)$ |

그 다음:

```text
src/psd_analysis.py
- 1194-1203: compute_family_spectral_summary()
```

에서 PSD curve와 dispersion을 계산합니다.

---

# 7. CSV 저장 구조

관련 위치:

```text
src/psd_analysis.py
- 1218-1227: checkpoint별 CSV 저장
- 1231-1246: epoch trend CSV 저장
```

checkpoint별로 저장하는 것:

```text
analysis_curve
analysis_dispersion
pair_distance
filter_snapshot
layer_distance_profile
layer_dispersion_profile
```

여러 checkpoint를 한 번에 넣으면 전체 epoch trend도 저장합니다.

```text
traces/layer_distance_trend
traces/layer_dispersion_trend
traces/filter_trend
```

---

# 8. 그럼 "미리 구현했어야 하나?"

## 정확한 답

**activation 자체를 미리 저장할 필요는 없습니다.**  
하지만 **모델이 analysis protocol을 만족하도록 미리 구현되어 있어야 합니다.**

즉 필요한 것은 이것입니다.

```text
1. checkpoint에 model_token, model_config, readout_config, state_dict가 저장되어야 함
2. build_snn_classifier() 가 같은 모델을 다시 만들 수 있어야 함
3. model.forward(x, capture_hidden=True) 가 ForwardResult를 반환해야 함
4. hidden layer가 membrane/spike/layer_input trace를 반환해야 함
5. layer 이름 순서를 iter_named_layers() 로 제공할 수 있어야 함
```

기존 프로젝트 커스텀 LIF/RF 계열은 이미 이 조건을 만족합니다.

---

# 9. SpikingJelly LIFNode로 바꾸면?

여기가 중요합니다.

그냥 이런 식으로 쓰면:

```python
nn.Sequential(
    nn.Linear(...),
    neuron.LIFNode(...),
    nn.Linear(...),
    neuron.LIFNode(...),
)
```

현재 `psd_analysis.py` 와는 바로 맞지 않습니다.

이유는 `nn.Sequential` 모델은 보통 이런 걸 반환하지 않기 때문입니다.

```python
ForwardResult(
    input_record=...,
    hidden_records=[...],
    output_record=...
)
```

또한 SpikingJelly `LIFNode` 는 stateful이므로 batch마다 reset도 필요합니다.  
현재 `psd_analysis.py` 는 일반 forward 전에 `functional.reset_net(model)` 을 자동으로 호출하지 않습니다. 기존 커스텀 LIF/RF는 forward 내부에서 membrane을 local zero tensor로 새로 만들기 때문에 문제가 없었던 겁니다.

따라서 SpikingJelly로 바꾸려면 최소한 wrapper가 필요합니다.

---

# 10. SpikingJelly를 쓰려면 맞춰야 할 wrapper 구조

대략 이런 계층을 만들어야 합니다.

```python
class SpikingJellyLIFLayer(nn.Module):
    def __init__(self, input_size, output_size, ...):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.lif = neuron.LIFNode(
            tau=2.0,
            step_mode="m",
            store_v_seq=True,
            ...
        )

    def forward(self, input_sequence, *, return_traces=False):
        # input_sequence: (B,T,C)
        current_seq = self.fc(input_sequence)

        # SpikingJelly multi-step은 보통 (T,B,C)를 받으므로 변환 필요
        sj_input = current_seq.transpose(0, 1).contiguous()

        spike_seq_tbc = self.lif(sj_input)
        spike_seq = spike_seq_tbc.transpose(0, 1).contiguous()

        if return_traces:
            membrane_seq = self.lif.v_seq.transpose(0, 1).contiguous()
            self._last_layer_input = current_seq
            return membrane_seq, spike_seq

        self._last_layer_input = None
        return None, spike_seq
```

그리고 모델 wrapper는 기존 `SNNClassifier.forward()` 처럼:

```python
result = ForwardResult(
    input_record=input_sequence,
    hidden_records=[
        LayerRecord(
            layer_name="layer_01",
            membrane=membrane_seq,
            spike=spike_seq,
            layer_input=layer_input_seq,
        )
    ],
    output_record=...
)
```

를 반환해야 합니다.

---

# 11. 특히 reset 문제

SpikingJelly `LIFNode` 는 내부 상태 $v$ 를 가집니다.  
따라서 분석 batch마다 다음 중 하나가 필요합니다.

```python
from spikingjelly.activation_based import functional

functional.reset_net(model)
```

또는 wrapper forward 시작에서:

```python
functional.reset_net(self)
```

를 호출해야 합니다.

그렇지 않으면:

```text
앞 batch의 막전위 상태가 다음 batch로 넘어감
→ PSD trace가 오염됨
→ 분석 결과가 틀어짐
```

기존 프로젝트 커스텀 LIF/RF는 forward마다 state를 새로 만들기 때문에 이 문제가 없었습니다.

---

# 12. 현재 LIFLayer는 이미 뉴런별 alpha 구조임

추가로 중요한 점이 있습니다.

현재 프로젝트의 `src/neurons/LIF_neuron.py` 는 SpikingJelly `LIFNode` 가 아니라 커스텀 LIF입니다. 그런데 이 커스텀 LIF는 이미 `alpha_raw` 를 출력 뉴런 수만큼 가집니다.

관련 위치:

```text
src/neurons/LIF_neuron.py
- 74-77: alpha_raw = nn.Parameter(torch.empty(self.output_size))
- 102-105: effective_alpha()
```

즉 dense LIF에서는:

```text
alpha_raw.shape = (output_size,)
```

입니다.

CNN LIF도:

```text
src/neurons/cnn2d.py
- 89-91: alpha_raw = nn.Parameter(torch.empty(self.output_size))
```

즉 CNN에서는 channel-wise alpha입니다.

따라서 현재 프로젝트 기준으로는 이미:

```text
dense LIF: 출력 뉴런별 alpha
CNN LIF: 출력 채널별 alpha
```

를 학습하는 구조입니다.

이걸 SpikingJelly `ParametricLIFNode` 로 바꾸면 오히려 기본 PLIF는 layer-wise scalar $w$ 이므로, 기존 프로젝트 LIF보다 tau/alpha 자유도가 줄어들 수 있습니다.

---

# 최종 판단

현재 `psd_analysis` 는 네가 말한 방식으로 이미 짜여 있습니다.

```text
checkpoint 읽기
→ 모델 재구성
→ probe input forward
→ hidden/output trace 수집
→ PSD 계산
```

다만 이건 **프로젝트 모델 인터페이스에 맞는 모델** 에 대해서만 됩니다.

따라서 SpikingJelly `LIFNode` / `ParametricLIFNode` 로 바꾸려면 핵심은 `psd_analysis.py` 를 크게 고치는 것보다, 새 레이어/모델을 기존 인터페이스에 맞추는 겁니다.

필수 구현은 이겁니다.

```text
1. build_snn_classifier() 에서 새 SpikingJelly 기반 모델을 재구성 가능하게 만들기
2. checkpoint model_token/model_config/state_dict가 그 모델과 호환되게 저장되게 하기
3. model.forward(x, capture_hidden=True) 가 ForwardResult 반환하게 하기
4. 각 layer가 return_traces=True일 때 membrane_seq, spike_seq 반환하게 하기
5. SpikingJelly state reset을 batch마다 보장하기
6. 필요하면 filter_stats_vectors() 로 tau/alpha도 CSV에 남기기
```

즉 **activation을 저장하도록 미리 학습 코드를 바꿀 필요는 없지만, 분석 가능한 forward 인터페이스는 모델 구현 단계에서 맞춰야 합니다.**

## Prompt:
2026. 5. 22. 오후 7:55:15

1. 근데 이거 모델을 왜 메타데이터로 재구성해야해? .pt 에 모델 구조같은거 다 있지 않아?
그 트레이스 받는 설정 켜야해서 그런가?
2. 이거 신호처리는 gpu 에서 하게 되어있지?

## Response:
2026. 5. 22. 오후 7:55:21

> **파일 내용 확인 및 분석**
>
>
> **파일 목록 확인 및 리스트 출력**
>
>
> **특정 함수 라인 검색**
>
>
> **Inspecting psd_analysis collect and compute summaries**
>
>
> **출력에서 누락된 라인 확인**
>
>
> **Inspecting psd_utils 및 family 파일**
>
>
> **Inspecting family_spectral_analysis.py 코드**
>
>
> **Explaining metadata reconstruction for model checkpointing**
>
> The metadata needs to be reconstructed to ensure the model is correctly built and includes features like `capture_hidden`, which may not be present in a saved full model. Saving only the `state_dict` is safer and more portable, as saving the full module could cause issues with refactoring and class implementation changes.
>
> **Clarifying model saving and signal processing**
>
> In PyTorch, saving only the `state_dict` is preferred because saving the full model can be problematic due to Python class paths being fragile and unsafe for future compatibility. The `state_dict` along with metadata allows for reproducibility and flexibility in analysis functions. Regarding signal processing, I need to check if GPU processing is used in the current code.
>
> 50s 동안 생각함

# 1. 왜 메타데이터로 모델을 재구성하나?

`.pt` 파일에 모델 구조가 **항상 들어있는 게 아닙니다.**  
PyTorch의 `.pt` 는 그냥 `torch.save(...)` 로 저장한 객체 파일일 뿐입니다. 무엇이 들어있는지는 저장 코드가 결정합니다.

현재 프로젝트는 대략 이런 방식입니다.

```python
checkpoint = {
    "model_token": ...,
    "model_config": ...,
    "readout_config": ...,
    "state_dict": model.state_dict(),
    ...
}
```

즉 `.pt` 안에 **완성된 모델 객체 자체** 가 들어있는 게 아니라,

```text
모델을 다시 만들기 위한 설정값
+ 학습된 파라미터 state_dict
```

가 들어있습니다.

그래서 분석할 때는 다음 순서가 필요합니다.

```text
1. checkpoint에서 model_token, model_config, readout_config 읽기
2. build_snn_classifier(...) 로 같은 구조의 모델 생성
3. state_dict 로드
4. model.eval()
5. model(input, capture_hidden=True) 실행
```

## 그럼 모델 전체를 저장하면 안 되나?

가능은 합니다.

```python
torch.save(model, "model.pt")
```

이렇게 하면 모델 객체 전체가 pickle로 저장됩니다.  
하지만 연구 코드에서는 보통 권장하지 않습니다.

이유는 명확합니다.

| 방식 | 장점 | 단점 |
|---|---|---|
| `torch.save(model)` | 바로 로드 가능 | 코드 경로, 클래스 이름, 모듈 구조가 조금만 바뀌어도 깨짐 |
| `state_dict + metadata` | 재현성 좋고 관리 쉬움 | 모델을 다시 build해야 함 |
| `state_dict` 만 저장 | 가장 일반적 | 구조 정보가 없으면 복원 불가 |

현재 방식은 두 번째입니다.

```text
state_dict + metadata
```

이 방식이 실험 관리에는 더 안전합니다.

---

## "트레이스 받는 설정 켜야 해서 그런가?"

**그것도 이유 중 하나지만, 정확히는 더 근본적인 이유가 있습니다.**

PSD 분석은 단순히 최종 logits만 있으면 안 됩니다.  
분석하려면 forward 결과가 다음을 줘야 합니다.

```text
input_record
hidden_records
output_record
```

그리고 각 hidden record 안에:

```text
layer_input
membrane
spike
```

같은 trace가 있어야 합니다.

즉 모델 forward가 이런 식으로 동작해야 합니다.

```python
result = model(x, capture_hidden=True)

result.hidden_records[0].membrane
result.hidden_records[0].spike
result.hidden_records[0].layer_input
```

그래서 `.pt` 에 모델 객체가 들어있냐 아니냐와 별개로, **모델 클래스 자체가 분석용 trace 반환 인터페이스를 구현해야 합니다.**

정리하면:

```text
모델 재구성 이유:
1. 현재 checkpoint가 모델 객체가 아니라 state_dict + metadata 구조라서
2. 같은 구조를 build한 뒤 weight를 입혀야 해서
3. 분석용 forward인 capture_hidden=True를 호출해야 해서
4. readout_config, input_shape, sequence_length, dataset metadata까지 맞춰야 해서
```

따라서 "트레이스 설정 때문" 만은 아닙니다.  
**현재 저장 정책 자체가 모델 객체 저장이 아니라 재구성형 checkpoint 저장이기 때문** 입니다.

---

# 2. 신호처리는 GPU에서 하게 되어 있나?

## 기본 의도는 GPU 분석입니다

현재 `psd_analysis.py` 는 `--gpu_index` 를 받고, CUDA가 없으면 에러를 내도록 되어 있습니다.

```python
if not torch.cuda.is_available():
    raise RuntimeError(...)
```

그리고 모델도 GPU로 올립니다.

```python
model = build_snn_classifier(...).to(device)
```

입력도 GPU로 보냅니다.

```python
tensor.to(device=device, dtype=torch.float32, non_blocking=True)
```

따라서 **모델 forward는 GPU에서 돕니다.**

---

## PSD 계산도 기본 경로에서는 GPU에서 돕니다

현재 non-low-vram 경로에서는 trace를 CPU로 내리지 않고 모읍니다.

```python
input_maps = trace_tensor_to_channel_major_maps(input_trace.detach())
collected[...].append(input_maps.detach())
```

`model_inputs` 가 GPU에 있고, `result.hidden_records` 도 GPU tensor이므로, 여기서 만들어진 `maps` 도 GPU tensor입니다.

그 다음 PSD summary 계산에서:

```python
summary = compute_family_spectral_summary(maps, ...)
```

를 호출합니다.

내부에서는:

```python
torch.fft.rfft(...)
```

를 씁니다.

즉 `maps` 가 CUDA tensor이면:

```text
Hann window 생성
centering
rFFT
power 계산
mean / median
variance / MAD
raw / dB 변환
```

까지 GPU에서 계산됩니다.

마지막에 CSV로 저장해야 하므로 결과 curve만 CPU numpy로 변환됩니다.

```python
value.detach().cpu().numpy()
```

즉 기본 경로는:

```text
GPU forward
→ GPU trace map
→ GPU FFT / PSD
→ GPU reduction
→ 최종 CSV 저장 직전 CPU 변환
```

입니다.

---

# 3. 단, low_vram 모드에서는 다릅니다

`LOW_VRAM=True` 경로에서는 trace를 바로 CPU로 내립니다.

```python
cpu_tensor = tensor.detach().to(device="cpu")
```

그리고 CPU tensor를 모아서 PSD 계산합니다.

따라서 low-vram 모드에서는:

```text
GPU forward
→ trace CPU 이동
→ CPU에서 PSD 계산
```

에 가깝습니다.

즉 속도는 떨어지지만 GPU 메모리는 아낍니다.

---

# 4. 정확한 상태 요약

| 단계 | 기본 모드 | low_vram 모드 |
|---|---|---|
| checkpoint load | CPU | CPU |
| 모델 재구성 | GPU로 이동 | GPU로 이동 |
| probe input | GPU 이동 | GPU 이동 |
| model forward | GPU | GPU |
| hidden/membrane/spike trace | GPU 유지 | CPU로 이동 |
| PSD FFT | GPU | CPU |
| PSD reduction | GPU | CPU |
| CSV 저장 | CPU 변환 후 저장 | CPU 저장 |

---

# 5. 중요한 주의점

현재 구조는 **GPU-first** 이지만, 완전한 의미에서 "항상 GPU" 는 아닙니다.

기본 모드에서는 GPU에서 신호처리합니다.  
하지만 `--low_vram 1` 을 켜면 trace를 CPU로 내려서 처리합니다.

그리고 마지막 CSV 저장은 당연히 CPU / NumPy / 파일 I/O 단계가 필요합니다.

---

# 최종 답

## 1번

`.pt` 에 모델 구조가 자동으로 다 들어있는 게 아닙니다.  
현재 프로젝트는 모델 객체가 아니라 **metadata + state_dict** 를 저장합니다. 그래서 PSD 분석 때 `model_token`, `model_config`, `readout_config` 로 모델을 재구성한 뒤 `state_dict` 를 로드합니다.

그리고 네 말처럼 trace를 받으려면 모델 forward가 `capture_hidden=True` 를 지원해야 하므로, 그냥 일반 `nn.Sequential` 모델을 저장해둔 것만으로는 PSD 분석이 안 됩니다.

## 2번

기본 모드는 **모델 forward와 PSD 신호처리 모두 GPU에서 하도록 되어 있습니다.**  
특히 `torch.fft.rfft` 기반 PSD 계산은 입력 `maps` 가 CUDA tensor이면 GPU에서 실행됩니다.

단, `low_vram` 모드를 쓰면 trace를 CPU로 내려서 PSD를 계산하므로 속도는 느려지고 GPU 메모리는 줄어듭니다.

## Prompt:
2026. 5. 22. 오후 7:59:55

뭔가 더 좋은 방법이 없을까?

## Response:
2026. 5. 22. 오후 8:00:00

> **파일 검사 및 목록화**
>
>
> **파일 검사 수행**
>
>
> **Inspecting specific code chunks**
>
>
> **불러오기 코드의 일부 출력**
>
>
> **PSD 유틸리티 코드 분석**
>
>
> **검색 및 코드 스니펫 추출**
>
>
> **검색 결과 확인 및 출력**
>
>
> **크게 분석 및 함수 정의 검사**
>
>
> **분석 파일 및 모델 정의 점검**
>
>
> **스크립트 코드 분석하기**
>
>
> **고려 중인 모델 분석 개선 아이디어**
>
> 모델 전체를 저장하는 대신 메타데이터와 상태 사전만 저장하는 것이 더 효율적일 것 같습니다. 또한, 모델 클래스와 분석을 분리하는 `TraceAdapter` 또는 `AnalysisModelAdapter`를 도입하는 방법도 고려 중입니다. 트레이스를 저장하는 방법은 여러 가지가 있으니, 이를 어떻게 구현할지에 대한 합리적인 타협점이 필요합니다.
>
> **분석 방법 개선 중**
>
> 정확한 샘플을 위해 평균 및 중앙값을 샘플별로 계산하고, 각 샘플의 row variance 또는 MAD도 해당 샘플별로 처리하고 있습니다. 현재는 배치마다 값들을 CPU에서 누적하는 방식으로 메모리 관리와 속도를 개선하려고 합니다. GPU 자원을 효율적으로 사용할 수 있도록 개선할 필요가 있습니다.
>
> **모델 추적 및 체크포인트 최적화**
>
> 가장 큰 개선점 중 하나는 작은 키별 곡선 처리입니다. `TraceRecorder`를 활용하여 비침투적인 방식으로 모델의 전방향 훅을 설정하고, `LIFNode`에서 스파이크 시퀀스와 막 전위 데이터를 처리할 수 있습니다. 또한, 체크포인트는 모델 클래스와 메타데이터를 저장하여, 모델 전체를 저장하는 것보다 안전한 방식으로 관리할 수 있습니다.
>
> **PSD 분석 및 개선 방향**
>
> 현재 코드 상태에서 필요한 개선점은 `streaming accumulator`, `trace adapter/hook`, `checkpoint schema`입니다. 개선을 위해 `analysis adapter`를 활용하여 추출 성능을 향상시킬 수 있습니다. 모델 구조 저장은 이전 질문에서 논의된 내용이므로, 이번에는 체크포인트 메타데이터 설계를 유지하며 그에 맞춰 최적화하는 방향으로 접근할 수 있습니다.
>
> **트레이스 기록 및 SpikingJelly 관리**
>
> 현재 코드에는 훈련 중 활성화를 미리 저장하지 않고, 선택적인 활성화 캐시를 이용하는 방식이 필요합니다. PSD는 배치 단위로 스트리밍하여 처리하고, SpikingJelly나 커스텀 뉴런에 공통된 트레이스 프로토콜을 위한 어댑터를 만들면 좋습니다. 전체 모델 객체를
>
> **PSD 분석 개선 방향 및 우선순위 설정**
>
> API 변경을 염두에 두고, 정확한 클래스명보다는 "혹은 SpikingJelly의 monitor/hook 계열"로 언급하는 것이 좋겠습니다. 이번 추천方案은 `state_dict + metadata`를 유지하며, 트레이스 관련 처리는 `TraceAdapter`로 분리하고, PSD 배치 스트리밍 누적기로 개선합니다. 우선순위로는 `StreamingFamilySpectralAccumulator` 구현을 가장 먼저 진행해야겠네요.
>
> **캐시 저장 및 설계 최적화**
>
> `HookTraceAdapter` 사용을 포함하여, 체크포인트와 관련된 캐시 저장 방안을 논의할 필요가 있습니다. 트레이스 맵은 `trace_cache/*`에, 요약은 `summary_cache/*`에 저장하고, 각 캐시의 키를 명확히 정의하죠. 또한, 훈련 중에 활성화 상태를 저장하는 것보다는, 모델 상태를 `state dict`로 저장하는 것이 더 효과적인 선택일 것 같습니다.
>
> **최종 구현 방식 및 코드 설명**
>
> 빠른 반복 분석을 위한 최적화 방법으로는 전체 모델을 저장하는 대신, 활성화 캐시나 스펙트럼 캐시를 사용하는 것이 더 좋습니다. 우선 `streaming accumulator`를 구현한 후 체크포인트 메타데이터를 유지하고, SpikingJelly 통합 시 어댑터를 추가합니다. 관련 코드 줄 번호도 적절히 참고해야겠어요.
>
> **검색된 함수 및 이터러블 목록 추출**
>
>
> **검토 및 캐시 저장 전략 분석**
>
>
> **최종 답변 요약 및 구현 계획**
>
> 현재 존재하는 함수가 비슷한 작업을 수행한다는 점을 언급할 수 있습니다. 하지만 방법론에 대한 최적의 해답은 딱 하나가 아닌 '현실적인 조합'입니다. 체크포인트에 모델 객체를 저장하는 것보다는, 모델 팩토리 설정을 사용한 방법을 제안하는 게 더 좋습니다. **
>
> 1m 56s 동안 생각함

가능합니다. 지금 방식은 "틀린 방식" 이 아니라 연구 코드에서 흔한 방식인데, SpikingJelly로 확장하거나 뉴런 모델을 자주 바꿀 계획이면 더 나은 구조로 분리하는 게 맞습니다. 실제 zip 구조를 조금 더 확인한 뒤, 바로 적용 가능한 설계안을 기준으로 답하겠습니다.

확인 결과, 현재 코드의 가장 큰 구조적 약점은 checkpoint 재구성 자체가 아니라 **trace를 전부 모았다가 `torch.cat` 한 뒤 PSD를 계산하는 방식** 입니다. 모델이 커지거나 probe 수가 늘면 여기서 GPU 메모리와 I/O가 터질 가능성이 큽니다.

있습니다. 내가 보기에는 **더 좋은 구조** 는 아래입니다.

```text
checkpoint(.pt)
→ ModelFactory가 모델 재구성
→ TraceAdapter가 모델별 trace 추출
→ PSDAccumulator가 batch마다 PSD 누적
→ 최종 summary만 CSV 저장
```

즉 핵심은 **모델 재구성 방식을 바꾸는 것** 이 아니라, 현재 `psd_analysis` 의 두 부분을 분리하는 겁니다.

1. **trace 추출부** 를 모델 내부 `capture_hidden=True` 에 너무 묶지 않기  
2. **PSD 계산부** 를 trace 전체 저장 후 계산이 아니라 streaming 누적 방식으로 바꾸기  

---

# 1. 지금 구조의 가장 큰 약점

현재 코드는 대략 이렇습니다.

```python
result = model(model_inputs, capture_hidden=True)

for record in result.hidden_records:
    maps = trace_tensor_to_channel_major_maps(record.membrane)
    collected[key].append(maps.detach())

return {
    key: torch.cat(values, dim=0)
    for key, values in collected.items()
}
```

즉 흐름이:

```text
모든 probe batch forward
→ 모든 layer trace를 collected에 저장
→ key별로 torch.cat
→ 그 다음 PSD 계산
```

입니다.

이 방식은 작은 실험에서는 괜찮습니다.  
하지만 CNN, 긴 $T$, 많은 checkpoint, 많은 layer, membrane/spike/layer_input을 다 저장하면 메모리가 크게 늘어납니다.

특히 현재 기본 모드는 GPU tensor를 그대로 모으므로:

```text
GPU forward
→ GPU trace maps 누적
→ torch.cat
→ PSD 계산
```

이라서 GPU memory pressure가 큽니다.

`low_vram` 을 켜면 CPU로 내리기는 하지만, 본질은 여전히 같습니다.

```text
CPU에 trace maps를 전부 모음
→ 나중에 PSD 계산
```

즉 `low_vram` 은 "GPU 메모리 절약" 이지, "좋은 streaming 구조" 는 아닙니다.

---

# 2. 제일 먼저 바꿀 부분: Streaming PSD Accumulator

현재 분석은 `psd_exact` 만 쓰고, `include_spectrogram=False` 입니다.  
그러면 전체 trace를 모을 필요가 거의 없습니다.

더 좋은 방식은 batch가 들어올 때마다 바로 PSD를 계산해서 **충분통계량만 누적** 하는 겁니다.

## 현재 방식

```text
batch 1 trace 저장
batch 2 trace 저장
batch 3 trace 저장
...
전체 trace cat
PSD 계산
summary 생성
```

## 개선 방식

```text
batch 1 trace → PSD 계산 → curve sum 누적
batch 2 trace → PSD 계산 → curve sum 누적
batch 3 trace → PSD 계산 → curve sum 누적
...
누적값 / sample 수
summary 생성
```

이렇게 하면 메모리는 거의 다음 수준으로 고정됩니다.

```text
현재 batch trace
+ layer별 PSD curve 누적값
```

전체 trace를 보관하지 않아도 됩니다.

---

# 3. 이 방식이 수학적으로 맞나?

현재 `compute_family_spectral_summary()` 가 하는 핵심은 다음입니다.

```text
1. trace maps: (samples, rows, time)
2. raw / centered variant 생성
3. exact periodogram 계산
4. row 방향 mean 또는 median으로 대표 curve 생성
5. sample 방향 평균
6. variance / MAD curve 생성
7. 필요하면 raw / dB scale 저장
```

여기서 sample 방향 평균은 streaming으로 정확히 계산 가능합니다.

예를 들어 batch $b$ 에서 대표 PSD curve 합을 $S_b(f)$, batch sample 수를 $n_b$ 라고 하면 전체 평균은:

$$
\bar{P}(f)=\frac{\sum_b S_b(f)}{\sum_b n_b}
$$

입니다.

따라서 batch마다 PSD를 계산해서 sum만 누적해도 최종 결과는 같습니다.

단, 중요한 주의가 있습니다.

> dB 변환은 batch마다 먼저 하면 안 됩니다.  
> raw power를 먼저 평균낸 뒤 dB로 바꿔야 현재 코드 의미와 맞습니다.

즉 streaming accumulator는 내부적으로 raw power curve를 누적하고, 마지막 `finalize()` 단계에서 dB를 만들어야 합니다.

---

# 4. 추천 구조

현재:

```text
_collect_signal_maps()
→ maps_by_key
→ compute_family_spectral_summary(maps)
→ distance 계산
→ CSV 저장
```

개선:

```text
_collect_signal_summaries()
→ summaries_by_key
→ distance 계산
→ CSV 저장
```

구조적으로는 이렇게 됩니다.

```python
accumulators = defaultdict(FamilyPSDAccumulator)

for inputs, target in loader:
    model_inputs = prepare(inputs)
    trace_result = trace_adapter.forward_trace(model_inputs)

    for key, trace_tensor in trace_result.items():
        maps = trace_tensor_to_channel_major_maps(trace_tensor)
        accumulators[key].update(maps)

summaries = {
    key: acc.finalize()
    for key, acc in accumulators.items()
}
```

그러면 기존의 이후 부분은 거의 그대로 쓸 수 있습니다.

```python
layer_distance_rows = _layer_distance_rows_for_checkpoint(
    summaries=summaries,
    ...
)

pair_rows = _pair_rows_for_checkpoint(
    summaries=summaries,
    ...
)
```

왜냐하면 pair distance나 drift distance는 결국 raw trace가 아니라 **summary curve** 만 필요하기 때문입니다.

---

# 5. 2순위 개선: TraceAdapter 도입

현재는 모델이 반드시 이런 인터페이스를 가져야 합니다.

```python
result = model(x, capture_hidden=True)

result.hidden_records
result.output_record
result.input_record
```

이건 프로젝트 내부 커스텀 모델에는 좋습니다.  
하지만 SpikingJelly `LIFNode`, `ParametricLIFNode`, future RF, exact ZOH RF 등을 섞기 시작하면 모델 forward에 분석 코드를 계속 집어넣게 됩니다.

더 좋은 방식은 `TraceAdapter` 를 두는 겁니다.

```python
class TraceAdapter:
    def reset(self):
        ...

    def forward_trace(self, model, x):
        ...
        return list_of_trace_records
```

그리고 모델 종류별로 adapter를 나눕니다.

```text
ProjectForwardResultAdapter
    기존 모델용
    model(x, capture_hidden=True)를 그대로 사용

SpikingJellyHookAdapter
    SpikingJelly 모델용
    forward hook / pre hook / v_seq를 이용해서 trace 수집

ManualRFTraceAdapter
    exact ZOH RF용
    state 변수 y, z, spike, membrane 등을 명시적으로 수집
```

이러면 `psd_analysis.py` 는 모델 세부 구현을 몰라도 됩니다.

---

# 6. SpikingJelly를 붙일 때 특히 좋은 구조

SpikingJelly 모델은 내부 state가 있습니다.  
따라서 adapter가 batch마다 반드시 reset을 보장해야 합니다.

```python
from spikingjelly.activation_based import functional

class SpikingJellyTraceAdapter:
    def forward_trace(self, model, x):
        functional.reset_net(model)

        # 필요하면 store_v_seq=True 설정
        # forward hook으로 spike output 수집
        # module.v_seq로 membrane 수집

        y = model(x)

        records = ...
        return records
```

이렇게 해야 합니다.

아니면 앞 batch의 막전위가 다음 batch로 넘어가서 PSD가 오염됩니다.

---

# 7. checkpoint는 지금 방식 유지가 낫다

"더 좋은 방법" 이라고 해서 `.pt` 에 모델 객체 전체를 저장하는 방향은 비추천입니다.

```python
torch.save(model, "model.pt")
```

이 방식은 당장은 편하지만 연구 코드에서는 위험합니다.

```text
1. 클래스 경로가 바뀌면 로드 실패
2. 모델 코드가 조금 바뀌면 호환성 깨짐
3. pickle 기반이라 장기 재현성이 낮음
4. 분석용 adapter 교체가 어려움
```

지금처럼:

```text
state_dict
+ model_token
+ model_config
+ readout_config
+ dataset metadata
```

를 저장하는 방식이 더 좋습니다.

다만 개선할 수는 있습니다.

## checkpoint metadata 개선안

현재보다 더 명시적으로:

```text
model_factory_token
model_config
neuron_config
readout_config
trace_adapter_token
checkpoint_schema_version
code_version 또는 git_commit
dataset_manifest_hash
```

를 넣는 게 좋습니다.

특히 SpikingJelly 모델을 도입하면 `trace_adapter_token` 이 중요해집니다.

예:

```json
{
  "model_factory_token": "snnbench_vgg11_lif",
  "trace_adapter_token": "spikingjelly_v_seq_hook",
  "neuron_config": {
    "node": "ParametricLIFNode",
    "backend": "cupy",
    "step_mode": "m",
    "store_v_seq": true,
    "tau_scope": "layer"
  }
}
```

이렇게 저장해두면 분석 코드가 어떤 adapter를 써야 하는지 자동으로 알 수 있습니다.

---

# 8. activation을 학습 중에 미리 저장하는 건 좋은 방법인가?

대부분의 경우 **아닙니다.**

학습 중 checkpoint마다 activation trace를 저장하면:

```text
1. 저장 용량이 너무 커짐
2. 학습이 느려짐
3. probe set seed를 나중에 바꾸기 어려움
4. 분석할 signal 종류를 나중에 바꾸기 어려움
5. layer_input / membrane / spike / RF state 등 추가할 때 재학습 필요
```

그래서 activation을 checkpoint 안에 넣는 방식은 비추천입니다.

다만 **캐시는 괜찮습니다.**

---

# 9. 좋은 절충안: Analysis cache

분석을 여러 번 반복할 거면 다음 캐시가 좋습니다.

## 1단계 cache: trace cache

```text
checkpoint hash
+ dataset manifest hash
+ probe seed
+ trace adapter version
+ signal list
```

를 key로 해서 trace map을 저장합니다.

장점:

```text
PSD 설정을 바꿔도 재사용 가능
```

단점:

```text
저장 용량 큼
```

## 2단계 cache: summary cache

```text
checkpoint hash
+ dataset manifest hash
+ probe seed
+ PSD algorithm version
+ signal list
+ centering / reducer / scale policy
```

를 key로 해서 PSD summary만 저장합니다.

장점:

```text
저장 용량 작음
반복 plotting / distance 재계산 빠름
```

단점:

```text
PSD 정의를 바꾸면 재사용 불가
```

네 프로젝트에는 **summary cache가 더 적합** 합니다.  
이미 분석 목적이 curve, dispersion, distance CSV이기 때문입니다.

---

# 10. 내가 추천하는 최종 설계

## 유지할 것

```text
checkpoint = metadata + state_dict
```

이건 유지하는 게 맞습니다.

## 바꿀 것 1: 분석 runner 분리

```text
CheckpointLoader
    checkpoint 읽기
    metadata 검증
    model 재구성

TraceAdapter
    model별 trace 수집
    reset 보장
    layout 정규화

FamilyPSDAccumulator
    batch별 PSD 누적
    final summary 생성

CSVWriter
    curve / dispersion / distance 저장
```

## 바꿀 것 2: `_collect_signal_maps()` 제거 또는 축소

현재 병목 함수는 `_collect_signal_maps()` 입니다.  
이 함수가 raw maps를 다 모으기 때문입니다.

새 함수는 이런 식이어야 합니다.

```python
def _collect_signal_summaries(...):
    accumulators = defaultdict(FamilyPSDAccumulator)

    for batch in loader:
        result = trace_adapter.forward_trace(model, batch)

        for key, tensor in result:
            maps = trace_tensor_to_channel_major_maps(tensor)
            accumulators[key].update(maps)

    return {
        key: acc.finalize()
        for key, acc in accumulators.items()
    }
```

그리고 main에서는:

```python
summaries_by_key.update(
    _collect_signal_summaries(...)
)
```

로 받으면 됩니다.

---

# 11. 구현 우선순위

## 1순위

**Streaming PSD accumulator 구현**

이게 효과가 가장 큽니다.

```text
메모리 감소
GPU OOM 감소
checkpoint 여러 개 분석 안정화
low_vram 의존 감소
```

## 2순위

**TraceAdapter 도입**

이건 SpikingJelly, RF, future neuron model 확장성을 위해 필요합니다.

```text
기존 커스텀 모델
SpikingJelly LIF/PLIF
exact ZOH RF
SNNBench adapter
```

를 같은 `psd_analysis.py` 로 처리할 수 있게 됩니다.

## 3순위

**summary cache 추가**

같은 checkpoint를 여러 분석 조건으로 반복할 때 유용합니다.

---

# 최종 판단

더 좋은 방법은 있습니다.  
내 기준으로는 이게 최선입니다.

```text
1. 모델 저장은 지금처럼 metadata + state_dict 유지
2. 모델 객체 전체 저장은 하지 않음
3. activation을 checkpoint에 미리 저장하지 않음
4. 대신 TraceAdapter로 trace 추출 방식을 모델 밖으로 분리
5. PSD는 전체 trace를 모으지 말고 batch streaming accumulator로 계산
6. 반복 분석용으로 summary cache 추가
```

가장 먼저 고칠 부분은 **Streaming PSD accumulator** 입니다.  
이건 현재 결과 정의를 거의 유지하면서 메모리 구조만 개선할 수 있어서, 위험 대비 효과가 제일 큽니다.

## Prompt:
2026. 5. 22. 오후 8:31:17

아 약간 저 레이어 막전위 시계열 맵, 스파이크 출력 맵을 내놓는? 출력하는? 그런 걸 모듈화하면 좋을거같은데

## Response:
2026. 5. 22. 오후 8:31:19

> **파일 분석 및 구조 설계 검토**
>
> 사용자가 이전에 업로드한 파일에 대한 분석이 필요할 것 같아요. 해당 파일을 검사하여 설계를 더 구체적이고 정확하게 제공할 수 있을지 확인해야 할 것 같아요. 파일을 열고 그 내용을 분석한 후, 필요한 작업을 진행할 예정입니다.
>
> **파일 압축 풀고 구조 설계 제안**
>
> 업로드된 파일을 압축 해제하고 그 내용을 확인해야 할 것 같아요. 설계를 제안할 때, "TraceEmitter/SignalMapExporter" 모듈을 활용해 구현을 설명할 수 있을 것 같아요. 사용자에게 구체적인 아키텍처와 코드 구조를 제공하려면, 현재 코드 구조를 점검하고 패치 포인트를 제안해야 할 필요가 있습니다.
>
> **압축 파일 내용 확인**
>
> 업로드된 zip 파일의 내용을 확인하려면, 먼저 파일을 압축 해제하고 그 안에 포함된 파일 이름들을 들여다봐야 할 것 같아요. 이를 위해 컨테이너를 사용해서 파일을 검사하고, 필요한 정보를 추출할 계획입니다.
>
> **검사 및 출력 zip 파일 이름**
>
>
> **필요한 파일 검사**
>
> 업로드된 파일을 확인하려면, 관련 파일들을 확인해야 할 것 같아요. 아마도 `/mnt/data/psd_inspect` 디렉터리에서 압축을 풀어 해당 파일들을 검사해야 할 것입니다. 이 절차를 통해 필요한 정보를 확실히 파악할 수 있도록 할 예정입니다.
>
> **압축 해제 및 파일 목록 확인**
>
>
> **psd_analysis 함수 점검**
>
> `psd_analysis`와 관련된 중요한 함수들을 점검해야 할 것 같아요. 이를 통해 필요한 기능이 잘 작동하는지 확인하고, 사용자에게 정확한 정보와 도움을 제공할 수 있도록 점검할 예정입니다.
>
> **검사 psd_analysis 관련 함수**
>
>
> **sed 출력 확인**
>
> `sed` 명령어 실행 후 출력이 잘리거나 이상해졌어요. 필요한 출력 부분을 다시 점검하고, 관련된 조각들을 확인하여 정확한 결과를 얻을 수 있도록 해야 할 것 같아요. 이를 통해 문제가 무엇인지 파악할 수 있을 것입니다.
>
> **추출된 psd 분석 파일 일부**
>
>
> **출력된 관련 코드 조각 추출**
>
>
> **SNNClassifier 시작 부분 점검**
>
> SNNClassifier의 시작 부분을 좀 더 상세히 점검할 필요가 있을 것 같아요. 이렇게 하면 초기 설정이나 구성 요소들이 정확히 작동하는지 확인하고, 이후 단계에서 문제 없이 진행될 수 있도록 할 수 있을 것입니다.
>
> **검사 SNNClassifier 클래스 코드**
>
>
> **제안된 모듈 구성 및 책임**
>
> 사용자에게 현재 코드 포인트를 명확하게 설명하면서 구체적인 모듈 설계와 기능을 정의해야 할 것 같습니다. 예를 들어, `TraceEmitter`와 `SignalMapEmitter` 같은 모듈 이름들을 제안할 수 있습니다. 각 모듈의 책임을 설명하고, 예시적인 의사 코드나 개요를 포함할 수 있습니다. 
> 
> `TraceEmitter`와 `SignalMapEmitter`는 각각 Raw Trace 수집과 Tensor 데이터 변환과 같은 중요한 역할을 할 수 있겠네요.
>
> 49s 동안 생각함

맞습니다. 그 방향이 더 깔끔합니다. 지금은 "모델 forward", "trace 추출", "PSD용 map 변환", "분석 누적" 이 한 흐름에 섞여 있어서, 네가 말한 **막전위 시계열 맵 / 스파이크 출력 맵을 내놓는 모듈** 로 빼는 게 구조적으로 맞습니다.

맞습니다. 그게 더 좋은 설계입니다.

정확히는 **"모델이 레이어별 막전위/스파이크 시계열을 내놓는 부분"** 을 하나의 독립 모듈로 빼는 게 좋습니다. 지금 구조에서는 그 역할이 `psd_analysis.py` 안의 `_collect_signal_maps()` 와 `_maps_from_record()` 에 섞여 있습니다.

---

# 핵심 설계 방향

현재 흐름은 대략 이렇습니다.

```text
psd_analysis.py
    checkpoint 로드
    모델 재구성
    probe batch 생성
    model(input, capture_hidden=True)
    result.hidden_records에서 membrane/spike 추출
    trace_tensor_to_channel_major_maps(...)
    PSD 계산
    CSV 저장
```

여기서 아래 부분을 독립 모듈로 빼자는 겁니다.

```text
model(input)
→ layer별 membrane trace
→ layer별 spike trace
→ layer별 layer_input trace
→ PSD 분석용 map으로 정규화
```

이 모듈의 이름은 예를 들면 다음이 좋습니다.

```text
src/analysis/trace_emitter.py
src/analysis/signal_map_emitter.py
src/analysis/layer_trace.py
```

내 기준으로는 **`signal_map_emitter.py`** 가 가장 직관적입니다.

---

# 1. "trace" 와 "map" 을 구분해야 함

여기서 용어를 분리하는 게 중요합니다.

## trace

모델에서 직접 나온 원래 시계열입니다.

예를 들어 dense SNN이면:

$$
V\in\mathbb{R}^{B\times T\times F}
$$

CNN SNN이면:

$$
V\in\mathbb{R}^{B\times T\times C\times H\times W}
$$

이게 **trace** 입니다.

---

## map

PSD 분석을 위해 축을 정리한 형태입니다.

현재 프로젝트의 PSD map 표준은:

```text
(samples, rows, time)
```

입니다.

즉 dense trace는:

$$
B\times T\times F
\rightarrow
B\times F\times T
$$

CNN trace는:

$$
B\times T\times C\times H\times W
\rightarrow
B\times (C H W)\times T
$$

이게 **signal map** 입니다.

따라서 네가 말한 "레이어 막전위 시계열 맵, 스파이크 출력 맵" 은 코드상으로는 이렇게 부르는 게 정확합니다.

```text
LayerTrace        : 원래 trace tensor
LayerSignalMap    : PSD-ready map tensor
```

---

# 2. 추천 모듈 구조

```text
src/analysis/
    __init__.py
    layer_trace.py
    signal_map_emitter.py
    trace_adapter.py
    psd_accumulator.py
```

각 역할은 이렇게 나누면 됩니다.

| 파일 | 역할 |
|---|---|
| `layer_trace.py` | trace/map record용 dataclass 정의 |
| `trace_adapter.py` | 모델별 trace 추출 방식 정의 |
| `signal_map_emitter.py` | trace를 PSD-ready map으로 변환해서 내보냄 |
| `psd_accumulator.py` | map을 받아 PSD summary를 streaming 누적 |

처음부터 전부 만들 필요는 없습니다.  
가장 먼저 만들 것은 `layer_trace.py` 와 `signal_map_emitter.py` 입니다.

---

# 3. dataclass 설계

예를 들어 이런 구조가 좋습니다.

```python
# src/analysis/layer_trace.py

from dataclasses import dataclass
from typing import Optional
import torch

@dataclass(frozen=True)
class LayerTraceRecord:
    layer_name: str
    layer_index: int
    signal_kind: str      # input | hidden | output
    series: str           # x_probe | layer_input | membrane | spike | y_mem | y_spike ...
    scope: str            # train_balanced_global | test_balanced_global 등
    probe_family: str
    label: Optional[int]
    tensor: torch.Tensor  # raw trace: (B,T,...) 기준

@dataclass(frozen=True)
class LayerSignalMapRecord:
    layer_name: str
    layer_index: int
    signal_kind: str
    series: str
    scope: str
    probe_family: str
    label: Optional[int]
    maps: torch.Tensor    # PSD-ready map: (samples, rows, time)
```

핵심은 `tensor` 와 `maps` 를 분리하는 것입니다.

---

# 4. 현재 모델용 TraceAdapter

현재 프로젝트 모델은 이미 이 인터페이스를 씁니다.

```python
result = model(model_inputs, capture_hidden=True)
```

그리고 `result` 안에:

```text
input_record
hidden_records
output_record
```

가 있습니다.

따라서 현재 모델용 adapter는 간단합니다.

```python
# src/analysis/trace_adapter.py

from typing import Iterable
import torch

from src.analysis.layer_trace import LayerTraceRecord

class ForwardResultTraceAdapter:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.layer_index_by_name = {}

        if hasattr(model, "iter_named_layers"):
            for idx, (name, _layer) in enumerate(model.iter_named_layers(), start=1):
                self.layer_index_by_name[str(name)] = idx

        self.layer_index_by_name.setdefault("input", 0)
        self.layer_index_by_name.setdefault("output", 999)

    def emit_traces(
        self,
        model_inputs: torch.Tensor,
        *,
        scope: str,
        probe_family: str,
        label: int | None,
    ) -> Iterable[LayerTraceRecord]:
        result = self.model(model_inputs, capture_hidden=True)

        input_trace = result.input_record if result.input_record is not None else model_inputs

        yield LayerTraceRecord(
            layer_name="input",
            layer_index=0,
            signal_kind="input",
            series="x_probe",
            scope=scope,
            probe_family=probe_family,
            label=label,
            tensor=input_trace,
        )

        for record in result.hidden_records:
            layer_name = str(record.layer_name)
            layer_index = int(self.layer_index_by_name.get(layer_name, len(self.layer_index_by_name) + 1))

            if getattr(record, "layer_input", None) is not None:
                yield LayerTraceRecord(
                    layer_name=layer_name,
                    layer_index=layer_index,
                    signal_kind="hidden",
                    series="layer_input",
                    scope=scope,
                    probe_family=probe_family,
                    label=label,
                    tensor=record.layer_input,
                )

            yield LayerTraceRecord(
                layer_name=layer_name,
                layer_index=layer_index,
                signal_kind="hidden",
                series="membrane",
                scope=scope,
                probe_family=probe_family,
                label=label,
                tensor=record.membrane,
            )

            yield LayerTraceRecord(
                layer_name=layer_name,
                layer_index=layer_index,
                signal_kind="hidden",
                series="spike",
                scope=scope,
                probe_family=probe_family,
                label=label,
                tensor=record.spike,
            )

            for extra_name in ("x_layer", "i_current", "z_gate", "y_mem", "y_spike"):
                tensor = getattr(record, extra_name, None)
                if isinstance(tensor, torch.Tensor):
                    yield LayerTraceRecord(
                        layer_name=layer_name,
                        layer_index=layer_index,
                        signal_kind="hidden",
                        series=extra_name,
                        scope=scope,
                        probe_family=probe_family,
                        label=label,
                        tensor=tensor,
                    )

        output_record = result.output_record
        output_index = int(self.layer_index_by_name.get("output", 999))

        for series in ("layer_input", "membrane", "spike", "readout_mem"):
            tensor = getattr(output_record, series, None)
            if isinstance(tensor, torch.Tensor):
                yield LayerTraceRecord(
                    layer_name="output",
                    layer_index=output_index,
                    signal_kind="output",
                    series=series,
                    scope=scope,
                    probe_family=probe_family,
                    label=label,
                    tensor=tensor,
                )
```

이 adapter는 현재 프로젝트 모델에 그대로 맞습니다.

---

# 5. SignalMapEmitter

그 다음 trace를 map으로 바꾸는 모듈입니다.

```python
# src/analysis/signal_map_emitter.py

from typing import Iterable
import torch

from src.analysis.layer_trace import LayerTraceRecord, LayerSignalMapRecord
from src.signal.psd_utils import trace_tensor_to_channel_major_maps

class SignalMapEmitter:
    def __init__(self, *, low_vram: bool = False):
        self.low_vram = bool(low_vram)

    def trace_to_maps(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.low_vram:
            tensor = tensor.detach().to(device="cpu", dtype=torch.float32)
        else:
            tensor = tensor.detach()
            if tensor.dtype != torch.float32:
                tensor = tensor.to(dtype=torch.float32)

        return trace_tensor_to_channel_major_maps(tensor).contiguous()

    def emit_maps(
        self,
        trace_records: Iterable[LayerTraceRecord],
    ) -> Iterable[LayerSignalMapRecord]:
        for record in trace_records:
            yield LayerSignalMapRecord(
                layer_name=record.layer_name,
                layer_index=record.layer_index,
                signal_kind=record.signal_kind,
                series=record.series,
                scope=record.scope,
                probe_family=record.probe_family,
                label=record.label,
                maps=self.trace_to_maps(record.tensor),
            )
```

이러면 `psd_analysis.py` 에서는 더 이상 `_maps_from_record()` 같은 함수가 필요 없습니다.

---

# 6. psd_analysis.py 쪽 사용 방식

현재는 `_collect_signal_maps()` 안에서 모든 걸 합니다.

개선 후에는 이런 식이 됩니다.

```python
trace_adapter = ForwardResultTraceAdapter(model)
map_emitter = SignalMapEmitter(low_vram=LOW_VRAM)

collected = defaultdict(list)

with torch.inference_mode():
    for family_id, family, label, subset in _probe_subsets(...):
        scope = f"{split_name}_{family_id}"

        for inputs, _target in loader:
            model_inputs = _prepared_input_for_model(model, inputs, device=device)

            trace_records = trace_adapter.emit_traces(
                model_inputs,
                scope=scope,
                probe_family=family,
                label=label,
            )

            for map_record in map_emitter.emit_maps(trace_records):
                key = (
                    map_record.layer_name,
                    map_record.layer_index,
                    map_record.signal_kind,
                    map_record.series,
                    map_record.scope,
                    map_record.probe_family,
                    map_record.label,
                )
                collected[key].append(map_record.maps)
```

이러면 최소한 기존 구조를 크게 망가뜨리지 않고도 모듈화할 수 있습니다.

---

# 7. 더 좋은 최종형: `collected` 도 없애기

위 구조는 모듈화는 되지만 여전히 `collected` 에 map을 전부 모읍니다.  
최종적으로는 이렇게 가는 게 더 좋습니다.

```python
trace_adapter = ForwardResultTraceAdapter(model)
map_emitter = SignalMapEmitter(low_vram=LOW_VRAM)
accumulators = defaultdict(PSDAccumulator)

for inputs, _target in loader:
    model_inputs = _prepared_input_for_model(model, inputs, device=device)

    trace_records = trace_adapter.emit_traces(
        model_inputs,
        scope=scope,
        probe_family=family,
        label=label,
    )

    for map_record in map_emitter.emit_maps(trace_records):
        key = map_record.key()
        accumulators[key].update(map_record.maps)

summaries = {
    key: accumulator.finalize()
    for key, accumulator in accumulators.items()
}
```

이게 최종적으로 가장 좋은 구조입니다.

```text
trace 추출
→ map 변환
→ PSD 누적
```

이 세 단계가 완전히 분리됩니다.

---

# 8. SpikingJelly로 바꿀 때도 이 구조가 유리함

지금 프로젝트 모델은 `capture_hidden=True` 로 trace를 줍니다.  
하지만 SpikingJelly 모델은 보통 그렇게 안 줍니다.

그 경우에는 adapter만 바꾸면 됩니다.

```text
ForwardResultTraceAdapter
    현재 프로젝트 모델용

SpikingJellyTraceAdapter
    SpikingJelly LIFNode / ParametricLIFNode용

SNNBenchTraceAdapter
    SNNBench VGG / ResNet용

RFTraceAdapter
    exact ZOH RF 전용
```

PSD 분석 코드는 그대로 둡니다.

즉:

```text
모델별 차이
→ TraceAdapter 안에 가둠

PSD 분석 공통 처리
→ SignalMapEmitter / PSDAccumulator가 담당
```

이렇게 됩니다.

---

# 9. SpikingJelly adapter는 이런 식

SpikingJelly에서는 대략 다음이 필요합니다.

```python
from spikingjelly.activation_based import functional, neuron

class SpikingJellyTraceAdapter:
    def __init__(self, model):
        self.model = model
        self.records = []
        self.handles = []

    def install_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, neuron.BaseNode):
                module.store_v_seq = True
                handle = module.register_forward_hook(self._make_hook(name))
                self.handles.append(handle)

    def _make_hook(self, name):
        def hook(module, inputs, output):
            # output: spike_seq
            # module.v_seq: membrane sequence
            # SpikingJelly는 보통 (T,B,...) 이므로 (B,T,...) 로 바꿔야 함
            spike = output.transpose(0, 1).contiguous()
            membrane = module.v_seq.transpose(0, 1).contiguous()

            self.records.append((name, membrane, spike))
        return hook

    def emit_traces(self, model_inputs, *, scope, probe_family, label):
        functional.reset_net(self.model)
        self.records.clear()

        _ = self.model(model_inputs)

        for idx, (name, membrane, spike) in enumerate(self.records, start=1):
            yield LayerTraceRecord(
                layer_name=name,
                layer_index=idx,
                signal_kind="hidden",
                series="membrane",
                scope=scope,
                probe_family=probe_family,
                label=label,
                tensor=membrane,
            )

            yield LayerTraceRecord(
                layer_name=name,
                layer_index=idx,
                signal_kind="hidden",
                series="spike",
                scope=scope,
                probe_family=probe_family,
                label=label,
                tensor=spike,
            )
```

핵심은 `psd_analysis.py` 를 SpikingJelly 전용으로 더럽히지 않는 겁니다.  
SpikingJelly 특수 처리는 adapter 안에만 둡니다.

---

# 10. 네가 말한 "내놓는 모듈" 의 정확한 역할

내가 권장하는 모듈은 저장까지 담당하지 않는 게 좋습니다.

나쁜 구조:

```text
TraceModule
    모델 forward
    trace 추출
    map 변환
    PSD 계산
    CSV 저장
```

이렇게 하면 또 비대해집니다.

좋은 구조:

```text
TraceAdapter
    raw trace를 내놓음

SignalMapEmitter
    raw trace를 PSD-ready map으로 바꿔 내놓음

PSDAccumulator
    map을 받아 PSD summary를 누적

CSVWriter
    summary를 파일로 저장
```

즉 "내놓는 모듈" 은 **generator처럼 record를 흘려주는 역할** 이 좋습니다.

```text
emit_traces()
emit_maps()
```

처럼 설계하는 게 맞습니다.

---

# 11. 지금 코드에서 바로 줄일 수 있는 부분

현재 `psd_analysis.py` 에서 분리 대상은 정확히 이 부분입니다.

```text
_maps_from_record()
_trace_to_cpu_maps()
_collect_signal_maps()
```

이 셋을 새 모듈로 빼면 됩니다.

추천 이동:

```text
_trace_to_cpu_maps()
_maps_from_record()
    → src/analysis/signal_map_emitter.py

model(model_inputs, capture_hidden=True) 호출부
result.hidden_records 순회부
    → src/analysis/trace_adapter.py
```

그러면 `psd_analysis.py` 는 다음만 담당합니다.

```text
1. CLI
2. checkpoint/model/dataset 준비
3. trace/map emitter 호출
4. PSD summary 계산
5. distance/CSV 저장
```

이게 훨씬 맞습니다.

---

# 최종 결론

네 생각이 맞습니다.

지금 프로젝트에는 **"레이어별 시계열 신호를 표준 형태로 뽑아주는 모듈"** 이 필요합니다.  
나는 이걸 아래 두 단계로 나누는 걸 추천합니다.

```text
TraceAdapter
    모델에서 layer_input / membrane / spike trace를 추출

SignalMapEmitter
    trace를 (samples, rows, time) PSD-ready map으로 변환
```

이렇게 해두면 나중에 SpikingJelly, SNNBench VGG/ResNet, exact ZOH RF, 뉴런별 tau PLIF를 넣어도 `psd_analysis.py` 본체를 크게 건드리지 않아도 됩니다.  
가장 먼저 할 수정은 `_maps_from_record()` 와 `_collect_signal_maps()` 를 이 구조로 분리하는 것입니다.

## Prompt:
2026. 5. 22. 오후 8:36:42

근데 이거 할려면 모델 구축할때 넣어야하잖아 그래서 말인데 모델을 고정으로 만들지 말고(resnet 같이 고정이면 어쩔 수 없지만, 모델 빌더 같은걸 이용해서 필요한걸 적층식으로 쌓게 하는건 어떨까? 근데 코드보면 이미 그러고 있을거같은데?

## Response:
2026. 5. 22. 오후 8:36:44

> **파일 검사 계획 세우기**
>
> 사용자가 코드에 대해 물어본 것 같아요. 업로드된 zip 파일에 접근해야 할 것 같아서 그 내용을 파악하려면 파일을 풀어헤치고 코드를 확인해야 할 것 같습니다. 아직 파일 검색이 아닌 직접 업로드된 파일을 사용하기 때문에, 도구를 쓰는 대신 파이썬을 사용해 파일을 검사할 계획이에요.
>
> **Inspecting uploaded zip file contents**
>
>
> **트리 검사 계획**
>
> 트리를 점검해야 할 것 같아요. 어떤 자료 구조나 코드의 트리 구조가 필요한지 생각해야 합니다. 그렇다면 트리에서 확인할 항목을 더 명확히 정의하고, 구체적으로 어떤 검사를 해야 할지 접근법을 정리해야 할 것 같아요.
>
> **파일 트리 검사**
>
>
> **snn_builder, model_registry, arch_spec, psd_analysis 점검하기**
>
> snn_builder, model_registry, arch_spec, psd_analysis를 점검해야 할 것 같습니다. 각 구성 요소가 관련된 역할이 무엇인지, 상호작용하는 방식이 어떻게 되는지 확인하겠습니다. 어떤 부분에 문제가 있을지 검토하고, 필요한 수정이나 분석을 진행해야 할 것 같아요.
>
> **검색 및 분석 로직 확인**
>
>
> **코드 스니펫 점검하기**
>
> 코드 스니펫을 점검해야 할 것 같아요. 내가 확인해야 하는 구체적인 부분이 무엇인지 파악하고, 코드가 잘 작동하는지 점검해야 합니다. 오류가 있으면 수정도 해야 하고, 그 외에 최적화가 필요한 부분도 있을지 고민해봐야겠어요.
>
> **코드 스니펫 검토**
>
>
> **코드 스니펫 검사하기**
>
>
> **빌드완성도 높이기**
>
> 더 완성도 높은 빌드를 해야겠어요. 현재 빌드가 아직 부족한 부분이 있을 수 있어서, 이를 채우고 더욱 밀도 있는 구조로 만들어 보도록 할게요. 세부사항까지 신경 쓰며 모든 요소를 충분히 다루는 것이 중요할 것 같아요.
>
> **파일 코드 일부 출력**
>
>
> **arch_spec 점검 및 문제 해결**
>
> 출력이 잘린 것 같아서, 이 정도면 충분할지 모르겠어요. arch_spec을 다시 점검하고 나머지 부분을 확인해야 할 것 같아요. 그 후에 문제를 해결할 수 있도록 필요한 수정 사항을 반영해야 할 것 같습니다.
>
> **파일 내용 일부 출력 및 검토**
>
>
> **훈련 체크포인트 메타데이터 점검**
>
> 훈련 체크포인트 메타데이터를 점검해야 할 것 같아요. 이 부분이 중요한 역할을 할 수 있어서, 필요한 정보를 세밀하게 확인하고 문제가 있는지 다시 살펴봐야 해요. 체크포인트 상태나 관련 데이터를 꼼꼼히 점검하겠습니다.
>
> **검사 훈련 체크포인트 메타데이터**
>
>
> **psd_analysis _maps_from_record 점검**
>
> psd_analysis와 관련된 _maps_from_record 함수, 특히 330번째 줄 이전의 코드 라인들을 점검해야 할 것 같아요. 이 부분이 문제 해결에 영향을 줄 수 있으니까, 해당 코드의 로직이나 구현이 정확한지 한 번 살펴봐야 할 필요가 있어요.
>
> **Inspecting psd_analysis.py lines 280-335**
>
>
> **모델 구축 답변 준비**
>
> 모델 빌더는 이미 층을 축적하므로, 부분적으로 맞습니다. 코드 파일 라인 번호를 언급하는 건 적합하지 않을 것 같지만, 경로:라인을 통해 언급은 가능할 것 같습니다. 또한, 빌드 중 추적을 삽입하는 아이디어를 다뤄야겠네요. 예를 들어, `build_snn_classifier`와 `arch_spec`로 구성된 시스템 내에서 구현된 층들을 모듈화할 수 있어요.
>
> 1m 15s 동안 생각함

이건 실제 코드 구조를 봐야 정확히 답할 수 있습니다. 지금 질문의 핵심은 "모델을 쌓는 빌더 구조가 이미 있는지" 와 "trace 추출 모듈을 빌더 단계에 끼울 수 있는지" 입니다.

코드 확인 결과, 네 예상이 맞습니다. **이미 적층식 모델 빌더는 존재합니다.** 다만 "레이어를 쌓는 빌더" 와 "레이어 trace/map을 표준으로 내놓는 빌더" 는 아직 분리되어 있지 않습니다.

네 생각이 맞습니다. 그리고 코드 확인해보니 **이미 어느 정도는 그렇게 되어 있습니다.**  
다만 현재 구조는 "적층식 모델 빌더" 는 있는데, **"trace/map 출력까지 표준화된 적층식 빌더" 는 아닙니다.**

정확히 나누면 이렇습니다.

---

# 1. 현재 코드도 이미 적층식 빌더다

현재 핵심 빌더는 이 함수입니다.

```text
src/model/snn_builder.py
build_snn_classifier()
```

여기서 dense SNN은 이미 이런 식으로 만들어집니다.

```text
arch_spec 또는 hidden_sizes 해석
→ layer_specs 생성
→ for hidden_index, layer_spec in resolved_layer_specs
→ build_layer_from_spec(...)
→ hidden_layers.append(layer)
→ SNNClassifier(hidden_layers=ModuleList(...), output_layer=...)
```

관련 위치:

```text
src/model/snn_builder.py:981-1115
```

특히 이 부분이 적층식 빌더입니다.

```python
for hidden_index, layer_spec in enumerate(resolved_layer_specs, start=1):
    layer = build_layer_from_spec(...)
    hidden_layers.append(layer)
    layer_meta.append(...)
```

즉 네가 말한 "모델 빌더 같은 걸 이용해서 필요한 걸 적층식으로 쌓는 구조" 는 **이미 있음** 입니다.

---

# 2. 현재 layer spec 구조도 이미 있다

`src/model/arch_spec.py` 에 이미 이런 layer spec이 있습니다.

```python
@dataclass(frozen=True)
class DenseLayerSpec:
    width: int

@dataclass(frozen=True)
class ConvLayerSpec:
    out_channels: int
    kernel_size: int
    stride: int
    padding: int
    ...

@dataclass(frozen=True)
class ResidualBlockSpec:
    out_channels: int
    kernel_size: int
    stride: int
    padding: int
    ...
```

관련 위치:

```text
src/model/arch_spec.py:11-49
```

즉 이미 구조는:

```text
LayerSpec
→ build_layer_from_spec()
→ 실제 layer 생성
```

입니다.

그래서 "완전히 새로 모델 빌더를 만들자" 보다는, **현재 빌더를 trace-aware builder로 정리하는 방향** 이 맞습니다.

---

# 3. 현재 부족한 점

현재 문제는 "모델 적층" 이 아니라 **trace 출력 인터페이스가 모델 forward 안에 강하게 묶여 있다는 점** 입니다.

현재 dense 모델 forward는 이렇게 되어 있습니다.

```text
src/model/snn_builder.py:457-479
```

요약하면:

```python
for meta, layer in zip(self.layer_meta[:-1], self.hidden_layers):
    membrane, spike = layer(current, return_traces=capture_hidden)
    current = spike

    if capture_hidden:
        layer_input = getattr(layer, "_last_layer_input", None)
        hidden_records.append(
            LayerRecord(
                layer_name=meta.name,
                membrane=membrane,
                spike=spike,
                layer_input=layer_input,
            )
        )
```

즉 현재 설계는 다음을 전제합니다.

```text
각 layer가 forward(..., return_traces=True)를 지원해야 함
각 layer가 _last_layer_input을 내부에 저장해야 함
모델 forward가 LayerRecord를 직접 조립해야 함
```

이건 작동은 합니다.  
하지만 SpikingJelly, SNNBench, RF, future neuron model을 계속 섞으면 점점 지저분해집니다.

---

# 4. 그래서 네 말대로 "레이어 출력 모듈화" 가 맞다

가장 좋은 구조는 이겁니다.

```text
LayerSpec
→ LayerFactory
→ TraceableLayer
→ TemporalStackModel
→ TraceAdapter
→ SignalMapEmitter
→ PSDAccumulator
```

각 역할은 이렇게 나눕니다.

| 구성 | 역할 |
|---|---|
| `LayerSpec` | 어떤 레이어를 쌓을지 정의 |
| `LayerFactory` | spec을 실제 PyTorch module로 변환 |
| `TraceableLayer` | membrane/spike/layer_input trace를 표준 형태로 노출 |
| `TemporalStackModel` | TraceableLayer들을 순서대로 실행 |
| `TraceAdapter` | 모델 출력 trace를 공통 record로 변환 |
| `SignalMapEmitter` | trace를 PSD-ready map으로 변환 |
| `PSDAccumulator` | map을 받아 PSD summary 누적 |

핵심은 **trace를 각 레이어의 표준 출력 계약으로 만들자** 는 것입니다.

---

# 5. 내가 추천하는 새 표준 인터페이스

레이어는 최소한 이 인터페이스를 만족해야 합니다.

```python
@dataclass
class LayerTrace:
    layer_name: str
    layer_index: int
    layer_input: torch.Tensor | None
    membrane: torch.Tensor | None
    spike: torch.Tensor | None
    extras: dict[str, torch.Tensor]
```

그리고 각 레이어 wrapper는 이렇게 동작합니다.

```python
class TraceableTemporalLayer(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        *,
        capture_trace: bool = False,
    ) -> tuple[torch.Tensor, LayerTrace | None]:
        ...
```

출력은 항상:

```text
next_x, trace
```

입니다.

예를 들어 LIF layer면:

```text
next_x = spike
trace.layer_input = current
trace.membrane = membrane_seq
trace.spike = spike_seq
```

RF layer면:

```text
next_x = spike
trace.layer_input = current
trace.membrane = membrane_seq
trace.spike = spike_seq
trace.extras["i_current"] = ...
trace.extras["y_mem"] = ...
```

이런 식입니다.

---

# 6. 그러면 모델 forward가 훨씬 단순해짐

현재는 모델 forward가 레이어 내부 구현을 너무 많이 압니다.

개선 후에는 모델 forward가 이렇게 됩니다.

```python
class TemporalStackClassifier(nn.Module):
    def forward(self, x, *, capture_hidden: bool = False):
        traces = []
        current = x

        for layer_index, block in enumerate(self.blocks, start=1):
            current, trace = block(
                current,
                capture_trace=capture_hidden,
            )

            if trace is not None:
                traces.append(trace)

        output, output_trace = self.output_block(
            current,
            capture_trace=True,
        )

        return ForwardResult(
            input_record=x,
            hidden_records=traces,
            output_record=output_trace,
        )
```

이렇게 되면 `TemporalStackClassifier` 는 LIF인지 RF인지 SpikingJelly인지 몰라도 됩니다.  
그냥 block이 `next_x, trace` 를 주면 됩니다.

---

# 7. SpikingJelly도 이 구조에 넣을 수 있음

SpikingJelly `LIFNode` 를 그대로 모델에 넣으면 현재 PSD 분석과 안 맞습니다.  
하지만 wrapper로 감싸면 됩니다.

```python
class SpikingJellyLIFTraceableLayer(nn.Module):
    def __init__(self, linear, lif_node, layer_name, layer_index):
        super().__init__()
        self.linear = linear
        self.lif_node = lif_node
        self.layer_name = layer_name
        self.layer_index = layer_index

    def forward(self, x, *, capture_trace=False):
        # x: (B,T,F)
        current = self.linear(x)

        # SpikingJelly multi-step은 보통 (T,B,F)
        sj_input = current.transpose(0, 1).contiguous()

        spike_tbf = self.lif_node(sj_input)
        spike = spike_tbf.transpose(0, 1).contiguous()

        if not capture_trace:
            return spike, None

        membrane = self.lif_node.v_seq.transpose(0, 1).contiguous()

        trace = LayerTrace(
            layer_name=self.layer_name,
            layer_index=self.layer_index,
            layer_input=current,
            membrane=membrane,
            spike=spike,
            extras={},
        )

        return spike, trace
```

이렇게 하면 SpikingJelly를 써도 모델 전체 인터페이스는 그대로 유지됩니다.

---

# 8. ResNet 같은 고정 구조는 어떻게 하나?

네 말대로 ResNet은 완전 자유 적층식으로 다루기 어렵습니다.  
하지만 그래도 **block 단위 traceable wrapper** 로 만들 수 있습니다.

예를 들면:

```text
TraceableBasicBlock
    conv1 + lif1
    conv2 + lif2
    residual add
    residual activation
    trace:
        block_conv1_membrane
        block_conv1_spike
        block_residual_add_membrane
        block_residual_add_spike
```

현재 코드도 이미 비슷한 흔적이 있습니다.

```text
src/model/snn_builder.py
CNN2DResidualBlock.trace_records(...)
```

그리고 `FixedCNN2DClassifier.forward()` 에서 residual block이면 `trace_records(meta.name)` 를 호출합니다.

관련 위치:

```text
src/model/snn_builder.py:379-384
```

즉 ResNet도 완전히 불가능한 게 아니라, **block 내부 trace 목록을 표준 record로 emit하게 만들면 됩니다.**

---

# 9. 현재 코드에서 가장 이상적인 리팩토링 방향

## 하지 말아야 할 것

```text
psd_analysis.py가 모델 내부 구조를 직접 다 알게 만들기
SpikingJelly hook 코드를 psd_analysis.py에 직접 넣기
모델마다 if model_type == ... 분기 늘리기
checkpoint에 activation 저장하기
```

이 방향은 나중에 반드시 복잡해집니다.

---

## 해야 할 것

현재 구조를 이렇게 바꾸는 게 좋습니다.

```text
src/model/
    arch_spec.py              # 유지
    model_registry.py         # 유지
    layer_factory.py          # build_layer_from_spec 이동/확장
    trace_types.py            # LayerTrace, ForwardTrace 정의
    traceable_layers.py       # 공통 wrapper / interface
    temporal_stack.py         # TemporalStackClassifier

src/analysis/
    trace_adapter.py          # 모델별 trace adapter
    signal_map_emitter.py     # trace -> (samples, rows, time)
    psd_accumulator.py        # streaming PSD 누적
```

현재 `snn_builder.py` 가 너무 많은 일을 합니다.

```text
모델 토큰 해석
레이어 생성
CNN backbone 생성
forward 정의
trace 조립
metadata 생성
```

그래서 최소한 아래는 빼는 게 좋습니다.

```text
LayerRecord / ForwardResult
→ src/model/trace_types.py

build_layer_from_spec()
→ src/model/layer_factory.py

trace -> map 변환
→ src/analysis/signal_map_emitter.py

_collect_signal_maps()
→ src/analysis/trace_collection.py 또는 trace_adapter.py
```

---

# 10. 지금 코드 기준으로 가장 작은 수정안

대규모 리팩토링 없이 먼저 할 수 있는 최소 수정은 이겁니다.

## 1단계: `LayerRecord`, `ForwardResult` 확장

현재:

```python
@dataclass
class LayerRecord:
    layer_name: str
    membrane: torch.Tensor
    spike: torch.Tensor
    layer_input: torch.Tensor | None = None
```

개선:

```python
@dataclass
class LayerRecord:
    layer_name: str
    membrane: torch.Tensor | None
    spike: torch.Tensor | None
    layer_input: torch.Tensor | None = None
    signal_kind: str = "hidden"
    series: str | None = None
    extras: dict[str, torch.Tensor] | None = None
```

그러면 RF, DH-SNN, SpikingJelly, SNNBench가 특수 신호를 억지 attr로 붙이지 않아도 됩니다.

현재 `psd_analysis.py` 는 `x_layer`, `i_current`, `z_gate`, `y_mem`, `y_spike` 를 `getattr` 로 찾고 있습니다.

```text
src/psd_analysis.py:297-300
```

이걸 `extras` 기반으로 바꾸면 훨씬 깔끔합니다.

---

## 2단계: 레이어가 `trace()` 를 반환하게 만들기

현재는 layer가 `_last_layer_input` 을 숨겨둡니다.

```python
layer_input = getattr(layer, "_last_layer_input", None)
```

이건 약간 지저분합니다.

더 좋은 건 layer forward가 명시적으로 반환하는 겁니다.

```python
next_x, trace = layer.forward_trace(current)
```

또는 기존 호환을 위해:

```python
membrane, spike, trace = layer(
    current,
    return_traces=True,
    return_record=True,
)
```

---

## 3단계: `SNNClassifier.forward()` 는 trace 조립만 담당

현재 모델 forward는 어느 정도 이미 그렇게 하고 있습니다.  
다만 `_last_layer_input` 에 의존하는 부분을 제거하면 됩니다.

---

# 11. "모델을 고정으로 만들지 말자" 에 대한 답

dense 계열은 이미 고정 모델이 아닙니다.  
`hidden_spec` / `arch_spec` 로 쌓습니다.

CNN 계열은 두 갈래입니다.

| 계열 | 현재 구조 |
|---|---|
| dense LIF/RF | 이미 적층식 |
| fixed CNN VGG/ResNet | 고정 backbone spec 존재 |
| SNNBench VGG/ResNet | adapter에서 별도 wrapping |
| SpikeGRU | 별도 고정 classifier |
| Spikformer / SpikingSSM | author adapter |

그래서 "전부 하나의 자유 빌더로 통합" 은 비추천입니다.  
대신 이렇게 하는 게 맞습니다.

```text
공통화할 것:
    trace 출력 계약
    signal map 변환
    PSD 분석 파이프라인

모델별로 남길 것:
    실제 backbone 구조
    ResNet block 내부 동작
    SpikingJelly node reset/state 처리
    SNNBench adapter
```

즉 **모델 구조까지 억지로 하나로 통합하지 말고, trace interface를 통합** 해야 합니다.

이게 더 안전합니다.

---

# 12. 최종 판단

네 말은 절반은 이미 구현되어 있고, 절반은 아직 부족합니다.

## 이미 되어 있는 것

```text
LayerSpec 기반 적층식 모델 빌더
build_layer_from_spec()
SNNClassifier(hidden_layers=ModuleList)
Fixed CNN backbone spec
LayerRecord / ForwardResult 기반 trace 반환
```

## 부족한 것

```text
trace 출력이 표준 layer interface로 분리되어 있지 않음
_psd_analysis.py가 trace/map 추출을 직접 담당함
_last_layer_input 같은 암묵적 상태에 의존함
SpikingJelly layer를 넣으려면 wrapper가 필요함
특수 신호가 extras 구조가 아니라 getattr 기반임
```

따라서 가장 좋은 방향은:

```text
기존 build_snn_classifier() 유지
+ build_layer_from_spec() 를 traceable layer factory로 확장
+ 모든 layer가 LayerTrace를 명시적으로 반환하게 함
+ psd_analysis.py는 TraceAdapter / SignalMapEmitter만 호출하게 함
```

입니다.

한 줄로 말하면:

**모델 빌더는 이미 있다. 새로 만들기보다, 현재 빌더가 쌓는 각 layer를 "traceable block" 으로 표준화하는 게 맞다.**

## Prompt:
2026. 5. 22. 오후 8:45:12

vgg, resnet, s4, gru, spike-transformer 는 전부 모두 모델 형태를 논문 보고 고정시킬거고
변하는건 오직 mlp 형태에서의 모델이야
대규모 리팩터링할거야 어떤 방안이 있을까?

## Response:
2026. 5. 22. 오후 8:45:18

현재 코드 구조를 기준으로 보겠습니다. 먼저 zip의 빌더, 뉴런, PSD 분석 의존관계를 확인한 뒤, 고정 backbone과 가변 MLP를 분리한 리팩터링안을 제시하겠습니다.

## Response:
2026. 5. 22. 오후 8:45:22

확인해보니 빌더 자체는 이미 있으므로, 핵심은 `snn_builder.py`를 쪼개는 것입니다. MLP는 전용 가변 빌더로 두고, 고정 논문 모델은 factory와 trace 계약만 맞추는 쪽이 안전합니다.

## Prompt:
2026. 5. 22. 오후 8:49:52

아 mlp 형태라는건 뉴런모델에 제한이 있는게 아니라 mlp형태로 쌓는걸 얘기하는거야 나중에 dh-snn, tc-lif 같은거 추가해서 lif 로 쌓을거거든

## Response:
2026. 5. 22. 오후 8:50:18

맞습니다. 여기서 핵심은 고정 백본과 달리 MLP만 dense topology와 neuron/cell family를 분리해, 뉴런 동역학을 갈아끼우는 빌더로 재설계하는 것입니다.

## Response:
2026. 5. 22. 오후 8:52:17

> **정확한 해석 수정하고 신경망 차원과 아키텍처 구분하기**
>
> MLP 형태는 아키텍처로, 뉴런 모델 축이 별개로 다뤄져야 해요. 이를 수정하면서, MLP 블록 내에서 다양한 신경망 셀을 사용할 수 있다는 점도 언급할 필요가 있어요. 이를 위해 "architecture=mlp_stack, cell=lif|rf|tc_lif|dh_snn|d_rf|spikingjelly_lif|spikingjelly_plif" 형태로 표현할 수 있을 거예요.
>
> 9m 27s 동안 생각함

## Response:
2026. 5. 22. 오후 8:54:45

맞습니다. 그러면 설계를 이렇게 고쳐야 합니다.

## 핵심 정정

여기서 **MLP 모델** 이라는 말은:

```text
LIF만 쓰는 MLP
```

가 아니라,

```text
입력 → dense temporal block → dense temporal block → ... → output
```

처럼 **위상(topology)이 MLP형 적층 구조** 라는 뜻입니다.

그 안의 temporal cell 또는 neuron update rule은 바뀔 수 있어야 합니다.

```text
MLP topology
    + LIF cell
    + RF cell
    + TC-LIF cell
    + TS-LIF cell
    + DH-SNN cell
    + D-RF cell
    + PLIF cell
    + neuronwise PLIF cell
```

즉 최종적으로는 이런 조합이 가능해야 합니다.

```text
mlp_stack + lif
mlp_stack + rf
mlp_stack + tc_lif
mlp_stack + dh_snn
mlp_stack + d_rf
mlp_stack + spikingjelly_lif
mlp_stack + neuronwise_plif
```

따라서 이전에 말한 "MLP-SNN" 을 **특정 뉴런 모델군** 으로 보면 안 되고, **dense 적층 topology** 로 봐야 합니다.

---

# 1. 가장 중요한 분리

리팩터링의 중심은 이겁니다.

```text
architecture / topology
    모델의 형태

cell / neuron / temporal operator
    시간 업데이트 방식

readout
    출력 해석 방식

trace
    분석용 신호 추출 방식
```

현재 코드는 이 네 가지가 꽤 섞여 있습니다.  
특히 `model_token` 이 많은 것을 동시에 담고 있습니다.

예를 들어 현재 토큰은 대략 이런 의미를 가집니다.

```text
lif_soft_fixed
    topology = dense MLP
    neuron = LIF
    reset = soft
    threshold = fixed

rf_R_soft_train
    topology = dense MLP
    neuron = RF
    recurrent = True
    reset = soft
    threshold = trainable

vgg11_lif_soft_fixed
    topology = VGG11
    neuron = LIF
    reset = soft
    threshold = fixed
```

이걸 대규모 리팩터링에서는 분리해야 합니다.

---

# 2. 새 분류 체계

## 2.1 TopologySpec

모델의 **형태** 만 표현합니다.

```python
@dataclass(frozen=True)
class TopologySpec:
    kind: str
```

구체적으로는:

```python
@dataclass(frozen=True)
class MLPStackSpec:
    kind: str = "mlp_stack"
    hidden_widths: tuple[int, ...] = ()
```

```python
@dataclass(frozen=True)
class FixedPaperSpec:
    kind: str
    variant: str = "paper_default"
```

예시는 이렇습니다.

```yaml
topology:
  kind: mlp_stack
  hidden_widths: [256, 256, 128]
```

```yaml
topology:
  kind: resnet18
  variant: paper_default
```

```yaml
topology:
  kind: spikegru
  variant: paper_2x128
```

---

## 2.2 CellSpec

여기에 LIF, RF, TC-LIF, DH-SNN 같은 것을 둡니다.

```python
@dataclass(frozen=True)
class CellSpec:
    kind: str
    recurrent: bool = False
    reset_mode: str | None = None
    trainable_threshold: bool = False
    branch: int | None = None
    tau_scope: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
```

예시는 이렇습니다.

```yaml
cell:
  kind: lif
  recurrent: false
  reset_mode: soft
  trainable_threshold: false
```

```yaml
cell:
  kind: tc_lif
  recurrent: false
  reset_mode: soft
  trainable_threshold: false
```

```yaml
cell:
  kind: dh_snn
  recurrent: true
  branch: 4
  reset_mode: soft
  trainable_threshold: false
```

여기서 `cell` 이라는 이름을 쓰는 게 좋습니다.  
왜냐하면 DH-SNN, D-RF, TC-LIF는 단순한 "점 뉴런 하나" 라기보다 **시간 동역학을 가진 layer cell/operator** 에 가깝기 때문입니다.

---

## 2.3 BuildSpec

최종 모델 빌드 설정은 이렇게 됩니다.

```python
@dataclass(frozen=True)
class BuildSpec:
    topology: MLPStackSpec | FixedPaperSpec
    cell: CellSpec | None
    output_cell: CellSpec | None
    readout: ReadoutSpec
    trace: TraceSpec
```

MLP-LIF는:

```yaml
model:
  topology:
    kind: mlp_stack
    hidden_widths: [256, 256, 128]

  cell:
    kind: lif
    recurrent: false
    reset_mode: soft
    trainable_threshold: false

  output_cell:
    kind: lif
    reset_mode: soft
    trainable_threshold: false

  readout:
    mode: max_over_time
```

MLP-TC-LIF는:

```yaml
model:
  topology:
    kind: mlp_stack
    hidden_widths: [256, 256, 128]

  cell:
    kind: tc_lif
    recurrent: false

  output_cell:
    kind: tc_lif

  readout:
    mode: max_over_time
```

MLP-DH-SNN은:

```yaml
model:
  topology:
    kind: mlp_stack
    hidden_widths: [256, 256]

  cell:
    kind: dh_snn
    recurrent: true
    branch: 4

  output_cell:
    kind: lif

  readout:
    mode: max_over_time
```

여기서 중요한 점은 **MLP topology는 그대로이고 cell만 바뀐다** 는 겁니다.

---

# 3. 현재 코드와의 차이

현재 구조는 대략 이렇습니다.

```text
ModelSpec.family == "lif"
    → DenseLayerSpec(width)
    → LIFLayer(input_size, output_size)

ModelSpec.family == "rf"
    → DenseLayerSpec(width)
    → RFLayer(input_size, output_size)

ModelSpec.family == "tc_lif"
    → DenseLayerSpec(width)
    → TCLIFLayer(input_size, output_size)

ModelSpec.family == "dh_snn"
    → DenseLayerSpec(width)
    → DHSNNLayer(input_size, output_size)
```

즉 지금도 **사실상 MLP topology + neuron family** 구조입니다.  
다만 코드상으로는 `ModelSpec.family` 가 topology와 cell을 동시에 표현합니다.

리팩터링 후에는 이렇게 바뀌어야 합니다.

```text
TopologySpec(kind="mlp_stack", hidden_widths=[...])
CellSpec(kind="lif")
    → MLPStackFactory
    → DenseTemporalBlock + LIFCell

TopologySpec(kind="mlp_stack", hidden_widths=[...])
CellSpec(kind="tc_lif")
    → MLPStackFactory
    → DenseTemporalBlock + TCLIFCell

TopologySpec(kind="mlp_stack", hidden_widths=[...])
CellSpec(kind="dh_snn", branch=4)
    → MLPStackFactory
    → DenseTemporalBlock + DHSNNCell
```

즉 `MLPStackFactory` 는 cell 종류를 몰라도 되고, `CellFactory` 에게 맡기면 됩니다.

---

# 4. 핵심 클래스 구조

## 4.1 TemporalCell

cell은 입력 current sequence를 받아 membrane/spike를 냅니다.

```python
class TemporalCell(nn.Module):
    output_size: int

    def forward_cell(
        self,
        input_current: torch.Tensor,
        *,
        trace_level: str,
    ) -> CellOutput:
        ...
```

`input_current` 의 표준 shape는:

```text
(B,T,F)
```

출력은:

```python
@dataclass
class CellOutput:
    membrane: torch.Tensor
    spike: torch.Tensor
    extras: dict[str, torch.Tensor] = field(default_factory=dict)
```

LIF면:

```text
extras = {}
```

TC-LIF면 예를 들어:

```text
extras = {
    "tau": ...,
    "filter_state": ...,
}
```

DH-SNN이면:

```text
extras = {
    "branch_current": ...,
    "dendrite_state": ...,
}
```

이런 식으로 갑니다.

---

## 4.2 DenseTemporalBlock

MLP의 한 층은 이렇게 정의합니다.

```python
class DenseTemporalBlock(nn.Module):
    def __init__(
        self,
        *,
        name: str,
        index: int,
        input_size: int,
        output_size: int,
        cell: TemporalCell,
    ) -> None:
        super().__init__()
        self.name = name
        self.index = index
        self.synapse = nn.Linear(input_size, output_size)
        self.cell = cell

    def forward_block(
        self,
        x: torch.Tensor,
        *,
        trace_level: str,
    ) -> tuple[torch.Tensor, LayerRecord | None]:
        layer_input = self.synapse(x)
        cell_output = self.cell.forward_cell(
            layer_input,
            trace_level=trace_level,
        )

        record = None
        if trace_level in {"hidden", "full"}:
            record = LayerRecord(
                layer_name=self.name,
                layer_index=self.index,
                membrane=cell_output.membrane,
                spike=cell_output.spike,
                layer_input=layer_input if trace_level == "full" else None,
                extras=cell_output.extras,
            )

        return cell_output.spike, record
```

이 구조에서는 MLP topology가 cell을 신경 쓰지 않습니다.

```text
DenseTemporalBlock
    Linear
    + 어떤 TemporalCell이든 가능
```

---

## 4.3 MLPStackClassifier

```python
class MLPStackClassifier(nn.Module):
    def __init__(
        self,
        *,
        blocks: list[DenseTemporalBlock],
        output_block: DenseTemporalBlock,
        metadata: dict[str, Any],
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.output_block = output_block
        self.metadata = dict(metadata)

    def forward_records(
        self,
        x: torch.Tensor,
        *,
        trace_level: str = "output",
    ) -> ForwardResult:
        current = x
        hidden_records: list[LayerRecord] = []

        for block in self.blocks:
            current, record = block.forward_block(
                current,
                trace_level=trace_level,
            )
            if record is not None:
                hidden_records.append(record)

        output_spike, output_record = self.output_block.forward_block(
            current,
            trace_level="full",
        )

        if output_record is None:
            raise RuntimeError("Output block must produce output_record.")

        return ForwardResult(
            input_record=x if trace_level == "full" else None,
            hidden_records=hidden_records,
            output_record=output_record,
        )

    def forward(self, x: torch.Tensor, *, capture_hidden: bool = False) -> ForwardResult:
        return self.forward_records(
            x,
            trace_level="full" if capture_hidden else "output",
        )
```

이렇게 하면 기존 학습 코드와도 당장은 호환됩니다.

---

# 5. CellFactory

핵심은 `CellFactory` 입니다.

```python
class CellFactory:
    def build(
        self,
        cell_spec: CellSpec,
        *,
        output_size: int,
        v_threshold: float,
    ) -> TemporalCell:
        kind = cell_spec.kind

        if kind == "lif":
            return LIFCell(
                output_size=output_size,
                recurrent=cell_spec.recurrent,
                reset_mode=cell_spec.reset_mode,
                trainable_threshold=cell_spec.trainable_threshold,
                v_threshold=v_threshold,
            )

        if kind == "tc_lif":
            return TCLIFCell(
                output_size=output_size,
                recurrent=cell_spec.recurrent,
                v_threshold=v_threshold,
                **cell_spec.params,
            )

        if kind == "dh_snn":
            return DHSNNCell(
                output_size=output_size,
                recurrent=cell_spec.recurrent,
                branch=int(cell_spec.branch or 4),
                v_threshold=v_threshold,
                **cell_spec.params,
            )

        raise ValueError(f"Unsupported cell kind: {kind!r}")
```

나중에 새 뉴런 모델을 추가할 때는:

```text
src/model/cells/new_cell.py 추가
CellFactory에 등록
```

만 하면 됩니다.

`psd_analysis.py`, `training.py`, `MLPStackFactory` 는 거의 안 바뀌어야 합니다.

---

# 6. 다만 현재 neuron layer는 이미 Linear를 포함하고 있음

현재 코드의 `LIFLayer`, `RFLayer`, `TCLIFLayer`, `DHSNNLayer` 는 대부분 이런 형태입니다.

```text
input_size, output_size를 받아서
내부에서 Linear 또는 weight를 가지고
시간 루프까지 처리
```

즉 이미 `DenseTemporalBlock + TemporalCell` 로 쪼개져 있지 않습니다.

그래서 리팩터링 경로는 두 가지입니다.

---

## 방안 A: 기존 layer를 유지하고 block adapter로 감싸기

가장 안전한 1차 리팩터링입니다.

```python
class ExistingDenseLayerBlock(nn.Module):
    def __init__(
        self,
        *,
        name: str,
        index: int,
        layer: nn.Module,
    ) -> None:
        super().__init__()
        self.name = name
        self.index = index
        self.layer = layer

    def forward_block(
        self,
        x: torch.Tensor,
        *,
        trace_level: str,
    ) -> tuple[torch.Tensor, LayerRecord | None]:
        need_trace = trace_level in {"hidden", "full"}

        membrane, spike = self.layer(
            x,
            return_traces=need_trace,
        )

        record = None
        if need_trace:
            layer_input = getattr(self.layer, "_last_layer_input", None)
            extras = {}
            for name in ("i_current", "z_gate", "y_mem", "y_spike"):
                value = getattr(self.layer, name, None)
                if isinstance(value, torch.Tensor):
                    extras[name] = value

            record = LayerRecord(
                layer_name=self.name,
                layer_index=self.index,
                membrane=membrane,
                spike=spike,
                layer_input=layer_input if trace_level == "full" else None,
                extras=extras,
            )

        return spike, record
```

이 방식의 장점:

```text
현재 LIFLayer, RFLayer, TCLIFLayer, DHSNNLayer를 거의 그대로 사용 가능
위험 낮음
기존 checkpoint 호환이 쉬움
```

단점:

```text
synapse와 cell이 아직 완전히 분리되지는 않음
내부 _last_layer_input 의존이 남음
```

---

## 방안 B: 진짜로 Synapse + Cell로 분해

최종적으로 더 좋은 구조입니다.

```text
DenseTemporalBlock
    synapse = Linear
    cell = LIFCell / RFCell / TCLIFCell / DHSNNCell
```

장점:

```text
구조가 매우 깔끔함
neuronwise tau, channel-wise tau, SpikingJelly cell 적용이 쉬움
trace 수집이 명시적임
cell 비교 실험이 공정해짐
```

단점:

```text
현재 neuron layer들을 많이 고쳐야 함
기존 checkpoint state_dict 호환이 깨질 수 있음
```

내 추천은 이겁니다.

```text
1차 리팩터링: 방안 A
2차 리팩터링: 새 뉴런부터 방안 B 구조로 작성
3차 리팩터링: 기존 LIF/RF/TC-LIF/DH-SNN도 점진적으로 Cell 구조로 이관
```

대규모 리팩터링이라도 한 번에 B로 가면 디버깅 지옥이 될 가능성이 큽니다.

---

# 7. 최종 모델 분류는 이렇게 가야 함

## 고정 paper model

```text
vgg11
resnet18
s4
spikegru
spikformer
```

이쪽은 topology가 고정입니다.

```text
FixedModelFactory
```

로 만듭니다.

예:

```yaml
model:
  topology:
    kind: spikegru
    variant: paper_2x128
  cell: null
```

```yaml
model:
  topology:
    kind: spikformer
    variant: paper_default
  cell: null
```

만약 VGG/ResNet에서 LIF/RF를 바꾸고 싶다면:

```yaml
model:
  topology:
    kind: vgg11
    variant: snnbench
  cell:
    kind: lif
    reset_mode: soft
    trainable_threshold: false
```

즉 topology는 고정, cell만 선택 가능하게 둘 수 있습니다.

---

## 가변 MLP-stack model

```text
mlp_stack
```

이쪽은 topology도 바뀝니다.

```yaml
model:
  topology:
    kind: mlp_stack
    hidden_widths: [512, 256, 128]

  cell:
    kind: dh_snn
    branch: 4
    recurrent: true
```

즉 `mlp_stack` 은 "LIF 전용" 이 아니라 **dense temporal block을 몇 개 쌓을지 정하는 구조** 입니다.

---

# 8. 리팩터링 후 registry 구조

```python
class ModelRegistry:
    def build(
        self,
        build_spec: BuildSpec,
        *,
        input_dim: int,
        sequence_length: int,
        input_shape: tuple[int, ...] | None,
        num_classes: int,
        v_threshold: float,
    ) -> nn.Module:
        topology_kind = build_spec.topology.kind

        if topology_kind == "mlp_stack":
            return self.mlp_stack_factory.build(
                build_spec=build_spec,
                input_dim=input_dim,
                sequence_length=sequence_length,
                num_classes=num_classes,
                v_threshold=v_threshold,
            )

        if topology_kind in self.fixed_factories:
            return self.fixed_factories[topology_kind].build(
                build_spec=build_spec,
                input_dim=input_dim,
                sequence_length=sequence_length,
                input_shape=input_shape,
                num_classes=num_classes,
                v_threshold=v_threshold,
            )

        raise ValueError(f"Unsupported topology: {topology_kind!r}")
```

핵심은:

```text
topology_kind == "mlp_stack"
    → MLPStackFactory

topology_kind == "vgg11" / "resnet18" / "s4" / "spikegru" / "spikformer"
    → FixedModelFactory
```

입니다.

---

# 9. 기존 token은 legacy alias로 바꾸는 게 좋음

기존:

```text
lif_soft_fixed
rf_R_soft_train
vgg11_lif_soft_fixed
resnet18_rf_soft_fixed
spikegru
spikformer
spikingssm
```

이걸 완전히 버리면 기존 실험과 checkpoint가 깨집니다.  
따라서 `legacy_token_to_build_spec()` 를 둡니다.

예:

```python
def legacy_token_to_build_spec(
    model_token: str,
    hidden_sizes: list[int] | None,
) -> BuildSpec:
    if model_token == "lif_soft_fixed":
        return BuildSpec(
            topology=MLPStackSpec(
                hidden_widths=tuple(hidden_sizes or ()),
            ),
            cell=CellSpec(
                kind="lif",
                recurrent=False,
                reset_mode="soft",
                trainable_threshold=False,
            ),
            output_cell=CellSpec(
                kind="lif",
                recurrent=False,
                reset_mode="soft",
                trainable_threshold=False,
            ),
            readout=ReadoutSpec(...),
            trace=TraceSpec(...),
        )

    if model_token == "spikegru":
        return BuildSpec(
            topology=FixedPaperSpec(
                kind="spikegru",
                variant="paper_2x128",
            ),
            cell=None,
            output_cell=None,
            readout=ReadoutSpec(...),
            trace=TraceSpec(...),
        )
```

이렇게 하면 기존 CLI/체크포인트도 살릴 수 있습니다.

---

# 10. Trace 설계도 이 구조에 맞춰야 함

MLPStack은 cell이 바뀌어도 trace를 공통으로 내야 합니다.

공통 trace:

```text
layer_input
membrane
spike
```

cell별 추가 trace:

```text
LIF
    없음 또는 alpha, threshold

RF
    filter state
    branch state

TC-LIF
    time constant 관련 state
    temporal current

DH-SNN
    dendritic branch current
    dendritic membrane
    soma membrane

GRU
    z_gate
    i_current
    y_mem
    y_spike

S4
    state
    output sequence
    maybe kernel/filter response

Spikformer
    block spike
    attention-related trace
```

따라서 `LayerRecord` 는 반드시 `extras` 를 가져야 합니다.

```python
@dataclass
class LayerRecord:
    layer_name: str
    layer_index: int
    membrane: torch.Tensor | None = None
    spike: torch.Tensor | None = None
    layer_input: torch.Tensor | None = None
    extras: dict[str, torch.Tensor] = field(default_factory=dict)
```

PSD 분석은 이렇게만 보면 됩니다.

```python
for record in result.hidden_records:
    emit(record.layer_input, series="layer_input")
    emit(record.membrane, series="membrane")
    emit(record.spike, series="spike")

    for series, tensor in record.extras.items():
        emit(tensor, series=series)
```

그러면 더 이상:

```python
getattr(record, "z_gate")
getattr(record, "i_current")
getattr(record, "y_mem")
```

이런 지저분한 코드가 필요 없습니다.

---

# 11. 새 디렉토리 구조 추천

지금 상황에 맞춘 더 정확한 구조는 이겁니다.

```text
src/model/
    build_spec.py
    registry.py
    outputs.py
    checkpoint_schema.py

    topology/
        __init__.py
        mlp_stack.py
        fixed_base.py

    cells/
        __init__.py
        factory.py
        lif.py
        rf.py
        tc_lif.py
        ts_lif.py
        dh_snn.py
        d_rf.py
        plif.py
        spikingjelly.py

    fixed/
        __init__.py
        vgg.py
        resnet.py
        spikegru.py
        s4.py
        spikformer.py

    legacy/
        __init__.py
        token_parser.py
        checkpoint_v1.py
```

분석 쪽은:

```text
src/analysis/
    trace_emitter.py
    signal_map_emitter.py
    psd_accumulator.py
    psd_pipeline.py
```

현재 `snn_builder.py` 는 최종적으로:

```text
src/model/legacy/snn_builder_compat.py
```

같은 역할만 남기거나 삭제하는 게 좋습니다.

---

# 12. 중요한 설계 원칙

## 원칙 1. MLP topology는 neuron kind를 몰라야 함

나쁜 구조:

```python
if cell.kind == "lif":
    build_lif_mlp(...)
elif cell.kind == "dh_snn":
    build_dh_snn_mlp(...)
```

좋은 구조:

```python
for width in hidden_widths:
    cell = cell_factory.build(cell_spec, output_size=width)
    block = DenseTemporalBlock(
        input_size=prev_width,
        output_size=width,
        cell=cell,
    )
```

---

## 원칙 2. cell은 PSD 분석을 몰라야 함

cell은 trace tensor만 내면 됩니다.  
PSD map 변환은 analysis 모듈이 합니다.

```text
cell
    membrane, spike, extras 반환

analysis
    trace → (samples, rows, time) map
    map → PSD
```

---

## 원칙 3. fixed paper model은 억지로 MLPStack에 넣지 말 것

VGG, ResNet, S4, SpikeGRU, Spikformer는 topology가 고정입니다.  
이걸 MLP builder처럼 일반화하려고 하면 오히려 논문 재현성이 떨어집니다.

```text
MLPStackFactory
    자유 실험용

FixedModelFactory
    논문 구조 재현용
```

이렇게 분리하는 게 맞습니다.

---

## 원칙 4. checkpoint에는 BuildSpec을 저장할 것

기존 `model_token` 중심 저장은 너무 뭉뚱그립니다.

새 checkpoint는 이렇게 가야 합니다.

```python
checkpoint = {
    "schema_version": 2,
    "build_spec": build_spec.to_dict(),
    "state_dict": model.state_dict(),
    "dataset_metadata": ...,
    "training_metadata": ...,
}
```

이렇게 해야 분석 시점에:

```text
build_spec 읽기
→ registry로 모델 재구성
→ state_dict load
→ trace_level="full" forward
→ PSD 분석
```

이 깔끔하게 됩니다.

---

# 13. 리팩터링 우선순위

## 1단계: `LayerRecord`, `ForwardResult` 분리

먼저 `snn_builder.py` 에서 빼세요.

```text
src/model/outputs.py
```

그리고 `extras` 를 추가합니다.

---

## 2단계: `BuildSpec`, `CellSpec`, `MLPStackSpec` 추가

기존 model token을 당장 버리지 말고, 변환기를 둡니다.

```text
legacy token
→ BuildSpec
```

---

## 3단계: MLPStackFactory 추가

처음에는 기존 `LIFLayer`, `RFLayer`, `TCLIFLayer`, `DHSNNLayer` 를 그대로 감싸도 됩니다.

```text
ExistingDenseLayerBlock
```

으로 시작하세요.

---

## 4단계: CellFactory 추가

기존 layer 생성 로직을 `CellFactory` 또는 `DenseLayerFactory` 로 옮깁니다.

초기에는:

```text
cell.kind == lif
    → 기존 LIFLayer(input_size, output_size)

cell.kind == tc_lif
    → 기존 TCLIFLayer(input_size, output_size)

cell.kind == dh_snn
    → 기존 DHSNNLayer(input_size, output_size)
```

이렇게 해도 됩니다.

---

## 5단계: fixed model factory 분리

`SpikeGRUClassifier`, SNNBench VGG/ResNet, Spikformer adapter, S4 adapter를 `fixed/` 아래로 옮깁니다.

---

## 6단계: trace/map emitter 분리

`psd_analysis.py` 에서 모델별 record 처리와 map 변환을 빼세요.

```text
TraceEmitter
SignalMapEmitter
PSDAccumulator
```

---

# 최종 정리

네가 말한 게 정확합니다.

**MLP형 모델** 은 "LIF 전용 MLP" 가 아닙니다.  
정확한 정의는:

```text
dense temporal block을 여러 층 쌓는 topology
```

입니다.

그 안에 들어가는 cell은 자유롭게 바뀌어야 합니다.

```text
mlp_stack + lif
mlp_stack + rf
mlp_stack + tc_lif
mlp_stack + dh_snn
mlp_stack + d_rf
mlp_stack + plif
mlp_stack + spikingjelly_lif
```

따라서 리팩터링의 핵심은:

```text
TopologySpec = mlp_stack / vgg / resnet / s4 / gru / spikformer
CellSpec = lif / rf / tc_lif / dh_snn / d_rf / ...
```

로 분리하는 겁니다.

한 줄로 말하면:

**MLP는 모델 family가 아니라 topology이고, LIF/RF/TC-LIF/DH-SNN은 그 topology 안에 꽂히는 temporal cell입니다. 이 둘을 분리해서 MLPStackFactory + CellFactory 구조로 가는 게 맞습니다.**

## Response:
2026. 5. 22. 오후 8:54:45

## 결론

네 조건이면 **전부를 하나의 범용 모델 빌더로 통합하려고 하면 안 됩니다.**  
가장 좋은 설계는 이겁니다.

```text
고정 논문 모델군
    VGG
    ResNet
    S4 / state-space
    SpikeGRU
    Spike-Transformer / Spikformer
        → FixedModelFactory로 고정 생성

가변 실험 모델군
    MLP-SNN
        → MLPStackBuilder로 적층 생성

공통화할 것
    checkpoint schema
    forward output contract
    trace emission contract
    PSD-ready signal map contract
    readout interface
```

즉 **모델 구조를 전부 공통화하지 말고, 모델이 내놓는 결과와 trace 인터페이스를 공통화** 해야 합니다.

---

# 1. 현재 코드 상태 판단

현재 코드도 이미 어느 정도 적층식 빌더입니다.

확인한 구조상 이미 이런 게 있습니다.

```text
src/model/arch_spec.py
    DenseLayerSpec
    ConvLayerSpec
    ResidualBlockSpec

src/model/snn_builder.py
    build_snn_classifier()
    build_layer_from_spec()
    SNNClassifier
    FixedCNN2DClassifier
    SpikGRUClassifier
```

즉 지금도 MLP 계열은 대략:

```text
hidden_spec
→ DenseLayerSpec 리스트
→ build_layer_from_spec()
→ hidden_layers.append(...)
→ SNNClassifier(...)
```

흐름입니다.

하지만 문제는 `snn_builder.py` 가 너무 많은 일을 합니다.

```text
모델 토큰 해석
레이어 생성
MLP 빌드
CNN 빌드
ResNet block 처리
SpikeGRU 구현
author adapter 호출
ForwardResult 정의
LayerRecord 정의
trace 조립
metadata 생성
```

이걸 계속 유지하면 나중에 SpikingJelly LIF, neuronwise PLIF, S4, Spikformer, 논문 고정 모델, PSD trace 분석이 전부 한 파일에 엉킵니다.

대규모 리팩터링을 한다면 이걸 분리해야 합니다.

---

# 2. 핵심 방향

## 하지 말아야 할 방향

```text
모든 모델을 하나의 LayerSpec 빌더로 억지 통합
```

이건 비추천입니다.

VGG, ResNet, S4, GRU, Spike-Transformer는 논문 구조를 보고 고정시킬 거라면, 굳이 MLP처럼 자유 적층식으로 만들 필요가 없습니다. 오히려 논문 재현성이 떨어집니다.

---

## 해야 할 방향

```text
1. MLP만 자유 적층식 builder로 둔다.
2. 논문 모델은 fixed factory로 둔다.
3. 모든 모델이 같은 ForwardResult / TraceRecord 계약을 만족하게 한다.
4. PSD 분석은 모델 종류를 몰라도 trace record만 받아 처리하게 한다.
```

이게 제일 안정적입니다.

---

# 3. 추천 전체 구조

리팩터링 후 디렉토리는 이런 식이 좋습니다.

```text
src/
    model/
        __init__.py

        specs.py
        registry.py
        checkpoint_schema.py

        outputs.py
        trace_types.py

        mlp/
            __init__.py
            specs.py
            builder.py
            blocks.py
            neuron_factory.py

        fixed/
            __init__.py
            vgg.py
            resnet.py
            spikegru.py
            s4.py
            spikformer.py

        adapters/
            __init__.py
            snnbench.py
            author_spikformer.py
            author_state_space.py

    analysis/
        __init__.py
        trace_emitter.py
        signal_map_emitter.py
        psd_accumulator.py
        psd_pipeline.py
```

현재 `src/model/snn_builder.py` 는 최종적으로 없애거나, 옛 CLI 호환용 wrapper만 남기는 게 좋습니다.

---

# 4. 가장 중요한 분리: architecture와 neuron 분리

현재는 `model_token` 하나에 너무 많은 정보가 들어갑니다.

예를 들어:

```text
lif_soft_fixed
rf_R_soft_train
vgg11_lif_soft_fixed
resnet18_rf_soft_fixed
```

이 방식은 처음에는 편하지만, 모델이 많아지면 parser가 계속 복잡해집니다.

대규모 리팩터링에서는 아래처럼 분리하는 게 낫습니다.

```python
@dataclass(frozen=True)
class BuildSpec:
    architecture: ArchitectureSpec
    neuron: NeuronSpec | None
    readout: ReadoutSpec
    trace: TraceSpec
```

예시는 이렇습니다.

```python
@dataclass(frozen=True)
class ArchitectureSpec:
    family: str          # "mlp", "vgg11", "resnet18", "spikegru", "s4", "spikformer"
    variant: str | None  # "paper_default", "snnbench", "author", ...

@dataclass(frozen=True)
class NeuronSpec:
    kind: str            # "lif", "rf", "plif", "neuronwise_plif", ...
    reset_mode: str | None
    trainable_threshold: bool
    recurrent: bool = False
    branch: int | None = None
    tau_scope: str | None = None
```

그러면 MLP는:

```yaml
model:
  architecture:
    family: mlp
    hidden: [256, 256, 128]
  neuron:
    kind: lif
    reset_mode: soft
    trainable_threshold: false
    recurrent: false
```

VGG는:

```yaml
model:
  architecture:
    family: vgg11
    variant: snnbench
  neuron:
    kind: lif
    reset_mode: soft
    trainable_threshold: false
```

SpikeGRU는:

```yaml
model:
  architecture:
    family: spikegru
    variant: paper_2x128
  neuron: null
```

Spikformer는:

```yaml
model:
  architecture:
    family: spikformer
    variant: paper_default
  neuron: null
```

이렇게 됩니다.

논문 모델 자체에 뉴런 구조가 내장되어 있으면 `neuron=None` 으로 두면 됩니다.

---

# 5. MLP만 진짜 builder로 만든다

MLP 쪽은 네가 원하는 대로 적층식으로 가면 됩니다.

## MLP spec

```python
@dataclass(frozen=True)
class MLPStackSpec:
    input_dim: int
    sequence_length: int
    hidden_widths: tuple[int, ...]
    num_classes: int
    output_policy: str = "spiking_output"
```

더 확장하고 싶으면 layer별 spec으로 둡니다.

```python
@dataclass(frozen=True)
class MLPBlockSpec:
    name: str
    out_features: int
    neuron: NeuronSpec
    recurrent: bool = False
```

그러면 나중에 이런 것도 가능합니다.

```yaml
architecture:
  family: mlp
  blocks:
    - name: layer_01
      out_features: 256
      neuron: lif
    - name: layer_02
      out_features: 256
      neuron: neuronwise_plif
    - name: layer_03
      out_features: 128
      neuron: rf
```

초기에는 width만 받는 간단한 구조로 시작해도 됩니다.

```text
--arch mlp
--hidden 256,256,128
--neuron lif
```

---

# 6. 고정 모델은 FixedModelFactory로 둔다

VGG, ResNet, S4, GRU, Spike-Transformer는 논문 구조를 고정한다면, 이쪽은 자유 builder가 아니라 factory가 맞습니다.

```python
class FixedModelFactory(Protocol):
    model_id: str

    def build(
        self,
        *,
        input_dim: int,
        sequence_length: int,
        input_shape: tuple[int, ...] | None,
        num_classes: int,
        build_spec: BuildSpec,
    ) -> nn.Module:
        ...
```

예:

```python
class SpikeGRUFactory:
    model_id = "spikegru.paper_2x128"

    def build(...):
        return SpikeGRUClassifier(
            input_dim=input_dim,
            sequence_length=sequence_length,
            num_classes=num_classes,
            hidden_size=128,
        )
```

```python
class SNNBenchVGG11Factory:
    model_id = "vgg11.snnbench"

    def build(...):
        return build_snnbench_vgg11(...)
```

```python
class SpikformerFactory:
    model_id = "spikformer.author_default"

    def build(...):
        return build_spikformer_author_classifier(...)
```

핵심은 **논문 모델은 registry에서 선택만 하고, 내부 구조는 고정** 하는 겁니다.

---

# 7. 모델 forward 결과 계약을 통일해야 함

지금은 `ForwardResult`, `LayerRecord` 가 `snn_builder.py` 안에 있습니다.  
이건 별도 파일로 빼야 합니다.

```text
src/model/outputs.py
src/model/trace_types.py
```

추천 구조는 이렇습니다.

```python
from dataclasses import dataclass, field
import torch

@dataclass
class TraceRecord:
    layer_name: str
    layer_index: int
    signal_kind: str      # "input", "hidden", "output"
    series: str           # "x_probe", "layer_input", "membrane", "spike", ...
    tensor: torch.Tensor
    layout: str = "B,T,*"

@dataclass
class LayerRecord:
    layer_name: str
    layer_index: int
    membrane: torch.Tensor | None = None
    spike: torch.Tensor | None = None
    layer_input: torch.Tensor | None = None
    extras: dict[str, torch.Tensor] = field(default_factory=dict)

@dataclass
class ForwardResult:
    output_record: LayerRecord
    hidden_records: list[LayerRecord] = field(default_factory=list)
    input_record: torch.Tensor | None = None
    traces: list[TraceRecord] = field(default_factory=list)
```

중요한 변화는 `extras` 입니다.

지금 코드는 GRU 쪽에서:

```python
setattr(layer_record, "i_current", ...)
setattr(layer_record, "z_gate", ...)
setattr(layer_record, "y_mem", ...)
setattr(layer_record, "y_spike", ...)
```

이런 식으로 붙입니다.

이건 나중에 S4, Spikformer까지 가면 지저분해집니다.  
앞으로는 이렇게 해야 합니다.

```python
LayerRecord(
    layer_name="layer_01",
    layer_index=1,
    membrane=layer1_mem,
    spike=layer1_spike,
    layer_input=input_sequence,
    extras={
        "i_current": layer1_current,
        "z_gate": layer1_gate,
        "y_mem": layer1_mem,
        "y_spike": layer1_spike,
    },
)
```

그러면 PSD 분석은 더 이상 `getattr(record, "z_gate")` 같은 식으로 모델별 신호 이름을 뒤지지 않아도 됩니다.

---

# 8. trace level을 binary가 아니라 단계화해야 함

현재는:

```python
model(x, capture_hidden=True)
```

입니다.

대규모 리팩터링에서는 이걸 바꾸는 게 좋습니다.

```python
class TraceLevel:
    NONE = "none"
    OUTPUT = "output"
    HIDDEN = "hidden"
    FULL = "full"
```

의미는 이렇습니다.

| trace level | 용도 | 저장 신호 |
|---|---|---|
| `NONE` | 빠른 forward | output만 최소 계산 |
| `OUTPUT` | 일반 학습/평가 | output membrane/spike |
| `HIDDEN` | PSD regularization | hidden membrane/spike |
| `FULL` | PSD analysis | input, layer_input, membrane, spike, extras |

현재 학습 regularization은 주로 `y_mem` 또는 `y_spike` 만 필요합니다.  
PSD 분석은 `layer_input`, `membrane`, `spike`, `extras` 까지 필요합니다.

즉 `capture_hidden=True` 하나로 모든 것을 켜면 메모리를 낭비합니다.

새 forward는 이런 식이 좋습니다.

```python
result = model.forward_records(x, trace_level="full")
```

기존 호환을 위해 당분간은:

```python
def forward(self, x, *, capture_hidden=False):
    trace_level = "full" if capture_hidden else "output"
    return self.forward_records(x, trace_level=trace_level)
```

를 유지하면 됩니다.

---

# 9. MLP block은 traceable block으로 만든다

MLP는 네가 직접 제어하는 모델이므로, 모델 구축 시점에 traceable block을 넣는 게 맞습니다.

```python
class TraceableMLPBlock(nn.Module):
    def __init__(
        self,
        *,
        name: str,
        index: int,
        input_size: int,
        output_size: int,
        neuron: nn.Module,
    ):
        super().__init__()
        self.name = name
        self.index = index
        self.synapse = nn.Linear(input_size, output_size)
        self.neuron = neuron

    def forward_records(self, x, *, trace_level: str):
        layer_input = self.synapse(x)
        membrane, spike = self.neuron(layer_input, return_traces=trace_level != "none")

        record = None
        if trace_level in {"hidden", "full"}:
            record = LayerRecord(
                layer_name=self.name,
                layer_index=self.index,
                membrane=membrane,
                spike=spike,
                layer_input=layer_input if trace_level == "full" else None,
            )

        return spike, record
```

이렇게 하면 MLP stack은 깔끔해집니다.

```python
class MLPStackClassifier(nn.Module):
    def forward_records(self, x, *, trace_level: str = "output"):
        current = x
        hidden_records = []

        for block in self.blocks:
            current, record = block.forward_records(current, trace_level=trace_level)
            if record is not None:
                hidden_records.append(record)

        output_spike, output_record = self.output_block.forward_records(
            current,
            trace_level="full",
        )

        return ForwardResult(
            input_record=x if trace_level == "full" else None,
            hidden_records=hidden_records,
            output_record=output_record,
        )
```

이게 가변 MLP에는 가장 좋은 구조입니다.

---

# 10. 고정 모델은 trace adapter로 감싼다

VGG, ResNet, S4, GRU, Spike-Transformer는 내부 구조를 고정할 거면, 모든 레이어를 `TraceableMLPBlock` 처럼 바꾸려고 하지 마세요.

고정 모델은 wrapper 또는 adapter가 더 낫습니다.

```python
class TraceAdapter(Protocol):
    def emit_records(
        self,
        model: nn.Module,
        x: torch.Tensor,
        *,
        trace_level: str,
    ) -> ForwardResult:
        ...
```

예:

```python
class SpikeGRUTraceAdapter:
    def emit_records(self, model, x, *, trace_level: str):
        return model.forward_records(x, trace_level=trace_level)
```

Spikformer 같은 author model은 hook 기반 adapter가 낫습니다.

```python
class SpikformerTraceAdapter:
    def emit_records(self, model, x, *, trace_level: str):
        # forward hook 또는 wrapper 내부 저장값으로 attention/spike/state trace 수집
        ...
```

ResNet은 block 내부 주요 지점만 record하면 됩니다.

```text
resnet_block_01_conv1_membrane
resnet_block_01_conv1_spike
resnet_block_01_residual_add_membrane
resnet_block_01_residual_add_spike
```

즉 고정 모델은 **모델 구조는 고정, trace 추출 방식만 adapter로 표준화** 하는 게 맞습니다.

---

# 11. PSD 분석은 모델을 몰라야 함

리팩터링 후 `psd_analysis.py` 는 모델 종류를 몰라야 합니다.

지금은 대략 이런 흐름입니다.

```text
model(x, capture_hidden=True)
→ result.hidden_records 직접 순회
→ record.membrane, record.spike, getattr(record, "i_current") 직접 확인
→ trace_tensor_to_channel_major_maps()
→ PSD 계산
```

개선 후에는 이래야 합니다.

```text
trace_emitter.emit(...)
→ TraceRecord stream
→ SignalMapEmitter
→ PSDAccumulator
→ CSVWriter
```

즉:

```python
records = trace_emitter.emit(
    model=model,
    inputs=model_inputs,
    scope=scope,
    trace_level="full",
)

for record in records:
    map_record = signal_map_emitter.to_map(record)
    psd_accumulator.update(map_record)
```

이렇게 되면 PSD 분석은 MLP인지 ResNet인지 S4인지 모릅니다.

---

# 12. SignalMapEmitter 표준화

네가 말한 "레이어 막전위 시계열 맵, 스파이크 출력 맵을 내놓는 모듈" 은 이 역할입니다.

```python
@dataclass
class SignalMapRecord:
    layer_name: str
    layer_index: int
    signal_kind: str
    series: str
    scope: str
    maps: torch.Tensor  # (samples, rows, time)
```

변환 규칙은 고정합니다.

```text
Dense trace:
    (B,T,F)
    → (B,F,T)

CNN trace:
    (B,T,C,H,W)
    → (B,C*H*W,T)

Transformer trace:
    (B,T,N,D)
    → (B,N*D,T)
    또는 분석 정의에 따라 token별/feature별 분리

S4 trace:
    (B,T,D)
    → (B,D,T)
```

이걸 모델마다 따로 하지 말고 `SignalMapEmitter` 하나에서 처리합니다.

---

# 13. checkpoint schema도 바꿔야 함

현재처럼 metadata + state_dict 방식은 유지하는 게 좋습니다.  
하지만 metadata schema는 더 명확해야 합니다.

추천 checkpoint 구조:

```python
checkpoint = {
    "schema_version": 2,

    "build_spec": {
        "architecture": {
            "family": "mlp",
            "hidden": [256, 256, 128],
        },
        "neuron": {
            "kind": "lif",
            "reset_mode": "soft",
            "trainable_threshold": False,
            "recurrent": False,
        },
        "readout": {
            "mode": "max_over_time",
        },
        "trace": {
            "schema_version": 1,
            "default_layout": "B,T,*",
        },
    },

    "state_dict": model.state_dict(),

    "dataset": {...},
    "training": {...},
    "code": {
        "git_commit": "...",
    },
}
```

중요한 건 `model_token` 하나에 의존하지 않는 겁니다.

단, 기존 checkpoint를 살리려면 v1 loader가 필요합니다.

```python
def load_checkpoint_build_spec(payload):
    if payload.get("schema_version", 1) == 1:
        return convert_legacy_payload_to_build_spec(payload)
    return BuildSpec.from_payload(payload["build_spec"])
```

---

# 14. registry 구조

모델 빌드는 registry 하나로 통합합니다.

```python
class ModelFactoryRegistry:
    def __init__(self):
        self._factories = {}

    def register(self, architecture_family: str, factory):
        self._factories[architecture_family] = factory

    def build(self, build_spec: BuildSpec, **runtime_kwargs):
        family = build_spec.architecture.family
        factory = self._factories[family]
        return factory.build(build_spec=build_spec, **runtime_kwargs)
```

등록 예:

```python
registry.register("mlp", MLPStackFactory())
registry.register("vgg11", VGG11Factory())
registry.register("resnet18", ResNet18Factory())
registry.register("spikegru", SpikeGRUFactory())
registry.register("s4", S4Factory())
registry.register("spikformer", SpikformerFactory())
```

그러면 학습과 분석은 똑같이 씁니다.

```python
model = model_registry.build(
    build_spec,
    input_dim=input_dim,
    sequence_length=sequence_length,
    input_shape=input_shape,
    num_classes=num_classes,
)
```

---

# 15. 리팩터링 단계

## 1단계: 타입 분리

먼저 이것부터 빼세요.

```text
src/model/snn_builder.py
    LayerRecord
    ForwardResult
    LayerMeta
```

이걸 새 파일로 이동합니다.

```text
src/model/outputs.py
```

그리고 `extras` 를 추가합니다.

```python
@dataclass
class LayerRecord:
    layer_name: str
    layer_index: int
    membrane: torch.Tensor | None = None
    spike: torch.Tensor | None = None
    layer_input: torch.Tensor | None = None
    extras: dict[str, torch.Tensor] = field(default_factory=dict)
```

이 단계에서는 기존 코드가 깨지지 않게 `layer_index` 기본값을 허용해도 됩니다.

---

## 2단계: trace emitter 분리

`psd_analysis.py` 안의 아래 역할을 밖으로 뺍니다.

```text
_maps_from_record()
_trace_to_cpu_maps()
_collect_signal_maps()
```

새 파일:

```text
src/analysis/trace_emitter.py
src/analysis/signal_map_emitter.py
```

이 단계의 목표는 이것입니다.

```text
psd_analysis.py가 record.membrane, record.spike, getattr(record, "z_gate") 를 직접 보지 않게 만들기
```

---

## 3단계: MLP builder 분리

현재 `snn_builder.py` 의 MLP 관련 부분을 옮깁니다.

```text
src/model/mlp/specs.py
src/model/mlp/neuron_factory.py
src/model/mlp/blocks.py
src/model/mlp/builder.py
```

여기서만 `hidden_widths` 를 받아 적층식으로 만듭니다.

---

## 4단계: fixed model factory 분리

아래 모델들은 전부 별도 factory로 뺍니다.

```text
src/model/fixed/vgg.py
src/model/fixed/resnet.py
src/model/fixed/spikegru.py
src/model/fixed/s4.py
src/model/fixed/spikformer.py
```

기존 `snnbench_adapter.py`, `author_adapter_spikformer.py`, `author_adapter_state_space.py` 는 이쪽으로 이동하거나 factory에서 호출하게 합니다.

---

## 5단계: checkpoint schema v2 도입

기존:

```text
model_token
model_config
readout_config
state_dict
```

새 구조:

```text
schema_version
build_spec
state_dict
dataset_metadata
training_metadata
```

단, 기존 checkpoint 로드를 버리면 안 됩니다.  
`legacy_model_token → BuildSpec` 변환기를 둬야 합니다.

---

## 6단계: PSD streaming accumulator 도입

이건 구조 안정화 후에 해도 됩니다.

현재는 trace map을 전부 모은 뒤 PSD를 계산합니다.  
최종적으로는 batch마다 PSD를 계산해서 누적하는 방식이 좋습니다.

```text
batch trace
→ signal map
→ PSD curve 누적
→ 최종 summary
```

이렇게 하면 GPU 메모리 부담이 크게 줄어듭니다.

---

# 16. 새 구조에서 학습 루프는 어떻게 바뀌나

현재 학습은:

```python
result = model(device_inputs, capture_hidden=capture_hidden)
analysis = readout.analyze_output_record(
    result.output_record.membrane,
    result.output_record.spike,
)
```

입니다.

새 구조에서도 당장은 유지할 수 있습니다.

```python
result = model.forward_records(
    device_inputs,
    trace_level="hidden" if regularization_enabled else "output",
)
analysis = readout.analyze_output_record(
    result.output_record.membrane,
    result.output_record.spike,
)
```

기존 호환 wrapper:

```python
def forward(self, x, *, capture_hidden=False):
    return self.forward_records(
        x,
        trace_level="full" if capture_hidden else "output",
    )
```

이렇게 두면 기존 코드와 새 코드를 동시에 운용할 수 있습니다.

---

# 17. 내가 추천하는 최종 정책

## 모델 구조 정책

| 모델군 | 구조 |
|---|---|
| MLP-SNN | 자유 적층식 builder |
| VGG | 고정 factory |
| ResNet | 고정 factory |
| S4 / state-space | 고정 factory |
| SpikeGRU | 고정 factory |
| Spike-Transformer / Spikformer | 고정 factory |

## 공통 interface

| 항목 | 공통화 여부 |
|---|---|
| input layout | 공통화 |
| output membrane/spike | 공통화 |
| hidden trace record | 공통화 |
| signal map 변환 | 공통화 |
| PSD 계산 | 공통화 |
| checkpoint schema | 공통화 |
| 내부 모델 구조 | 공통화하지 않음 |

이게 가장 안전합니다.

---

# 18. 성공 기준

리팩터링이 잘 된 상태는 다음 조건을 만족해야 합니다.

```text
1. MLP hidden 구조를 바꿔도 fixed model 코드가 안 바뀐다.
2. VGG/ResNet/S4/GRU/Spikformer를 추가해도 psd_analysis.py가 안 바뀐다.
3. 새 뉴런 모델을 MLP에 추가할 때는 neuron_factory.py만 주로 바뀐다.
4. 새 PSD 대상 신호를 추가할 때는 LayerRecord.extras 또는 TraceRecord만 쓰면 된다.
5. psd_analysis.py 안에 model family별 if 분기가 거의 없다.
6. checkpoint에는 모델 객체가 아니라 build_spec + state_dict가 저장된다.
7. 기존 checkpoint는 legacy converter로 계속 읽힌다.
```

---

# 최종 권장안

대규모 리팩터링을 한다면 이렇게 가세요.

```text
1. 고정 논문 모델은 FixedModelFactory로 분리
2. MLP만 MLPStackBuilder로 적층식 구성
3. model_token 중심 설계를 BuildSpec 중심 설계로 변경
4. LayerRecord / ForwardResult를 snn_builder.py 밖으로 이동
5. LayerRecord에 extras 추가
6. trace 추출과 signal map 변환을 analysis 모듈로 분리
7. psd_analysis.py는 TraceRecord stream만 받아 처리
8. checkpoint schema v2 도입
9. 기존 checkpoint용 legacy 변환기 유지
```

한 줄로 정리하면:

**모델 구조는 MLP만 자유롭게 만들고, 나머지는 논문 고정 factory로 둔다. 대신 모든 모델이 같은 trace/output 계약을 지키게 만들어서 학습, PSD 분석, checkpoint 로딩이 모델 종류와 독립적으로 돌아가게 만드는 게 맞습니다.**

## Prompt:
2026. 5. 22. 오후 8:58:29

그러니까 너가 생각하는건
어떤 모델에 대한 메타데이터를 정교하게 표준화해서 그걸 빌더에도 사용하고 나중에 모델 저장/로드에도 사용하자는거지? 토폴로지/시냅스/뉴런모델 모두 각각 분리해서 저장해 모듈성을 높이고 관찰 막전위/출력스파이크 맵도 구조 정해놓고

## Response:
2026. 5. 22. 오후 8:58:33

네, 방향은 맞습니다. 다만 실험 재현과 분석에 필요한 단위까지만 표준화하는 설계로 현재 코드 구조에 맞춰 정리하겠습니다.

## Response:
2026. 5. 22. 오후 8:59:46

> **Conceptual confirmation and file consideration**
>
> 사용자는 개념적인 확인을 원하지만, 업로드된 파일을 참조해야 할 수도 있어요. 그들이 코드를 점검해 달라고 하지 않았지만, 업로드된 프로젝트에 대해 얘기할 가능성이 높아요. 코드를 볼 필요가 있다면, 이미 업로드된 파일을 사용할 수 있습니다. 그래서, “정확히는 코드 보면...”이라고 언급할 수 있겠네요!
>
> 3m 20s 동안 생각함

## Response:
2026. 5. 22. 오후 9:01:53

맞습니다. 내가 제안하는 핵심은 정확히 그겁니다.

다만 표현을 조금 더 엄밀하게 하면:

> **모델을 설명하는 표준 명세(BuildSpec)를 만들고, 이 명세를 학습 시 모델 빌드, checkpoint 저장, checkpoint 로드, PSD 분석용 재구성에 모두 공통으로 사용하자** 는 것입니다.

그리고 별도로,

> **모델이 내놓는 관찰 신호(trace)와 PSD 분석용 신호 맵(signal map)의 구조도 표준화하자** 는 것입니다.

즉 큰 축은 두 개입니다.

```text
1. BuildSpec
   모델을 어떻게 만들 것인가

2. TraceSpec / SignalMapSpec
   모델에서 무엇을 관찰하고, 어떤 형태로 분석할 것인가
```

---

# 1. 모델 쪽: BuildSpec을 단일 기준으로 둔다

현재처럼 `model_token="lif_soft_fixed"` 하나로 많은 의미를 압축하면 나중에 터집니다.

대신 아래처럼 분리합니다.

```text
BuildSpec
    topology
    synapse
    temporal cell / neuron model
    readout
    trace policy
```

여기서 중요한 점은 **MLP에서는 topology, synapse, neuron을 분리** 하고, **VGG/ResNet/S4/GRU/Spike-Transformer 같은 고정 모델은 factory token 중심으로 저장** 하는 겁니다.

---

## MLP 계열

MLP 계열은 진짜로 적층식 빌더를 씁니다.

```text
topology = mlp_stack
blocks = [
    Linear + LIF,
    Linear + LIF,
    Linear + LIF,
    Linear + output_cell
]
```

이때 LIF 자리에 나중에 TC-LIF, DH-SNN, PLIF, RF, D-RF 등을 꽂을 수 있어야 합니다.

즉 MLP는 "LIF 모델" 이 아니라:

```text
dense temporal block을 여러 층 쌓는 topology
```

입니다.

---

## 고정 논문 모델 계열

반대로 아래 모델들은 논문 구조를 보고 고정시킨다고 했으니, 굳이 완전한 블록 단위 spec으로 풀 필요가 없습니다.

```text
VGG
ResNet
S4
SpikeGRU
Spike-Transformer / Spikformer
```

이쪽은 예를 들어:

```yaml
architecture:
  family: resnet
  variant: paper_resnet18
```

정도만 저장하고, 실제 구조는 `ResNetFactory` 가 책임지는 게 낫습니다.

즉:

```text
MLP
    상세 BuildSpec으로 자유 적층

고정 논문 모델
    architecture family + variant로 재구성
```

이렇게 나누는 게 맞습니다.

---

# 2. BuildSpec 예시

MLP-LIF라면 이런 식입니다.

```yaml
schema_version: 2

model:
  topology:
    kind: mlp_stack
    hidden_widths: [256, 256, 128]

  synapse:
    kind: linear
    bias: true

  cell:
    kind: lif
    reset_mode: soft
    trainable_threshold: false
    recurrent: false
    alpha_scope: neuron

  output_cell:
    kind: lif
    reset_mode: soft
    trainable_threshold: false
    recurrent: false

  readout:
    kind: temporal_max
    source: output_membrane

  trace:
    default_layout: B,T,*
    capture:
      - input
      - layer_input
      - membrane
      - spike
```

TC-LIF로 바꾸고 싶으면 topology는 그대로 두고 cell만 바꿉니다.

```yaml
model:
  topology:
    kind: mlp_stack
    hidden_widths: [256, 256, 128]

  synapse:
    kind: linear
    bias: true

  cell:
    kind: tc_lif
    reset_mode: soft
    trainable_threshold: false
    recurrent: false
    tau_scope: neuron
    params:
      init_tau: 2.0

  output_cell:
    kind: lif
    reset_mode: soft
    trainable_threshold: false

  readout:
    kind: temporal_max
    source: output_membrane
```

DH-SNN이면:

```yaml
model:
  topology:
    kind: mlp_stack
    hidden_widths: [256, 256]

  synapse:
    kind: linear
    bias: true

  cell:
    kind: dh_snn
    recurrent: true
    branch_count: 4
    reset_mode: soft
    trainable_threshold: false

  output_cell:
    kind: lif
    reset_mode: soft
```

즉 같은 MLP topology에서 cell만 갈아끼우는 구조입니다.

---

# 3. checkpoint에는 BuildSpec + state_dict를 저장한다

checkpoint는 이렇게 가야 합니다.

```python
checkpoint = {
    "schema_version": 2,
    "build_spec": build_spec.to_dict(),
    "state_dict": model.state_dict(),
    "dataset_spec": dataset_spec.to_dict(),
    "training_spec": training_spec.to_dict(),
    "metrics": metrics,
}
```

중요한 점은 이것입니다.

```text
BuildSpec
    모델을 다시 만들기 위한 구조 정보

state_dict
    학습된 파라미터

dataset_spec
    어떤 입력 구조와 데이터 기준으로 학습했는지

training_spec
    학습 조건

trace_spec
    분석 시 어떤 신호를 뽑을 수 있는지
```

즉 checkpoint에 모델 객체 전체를 저장하지 않습니다.

```python
torch.save(model, path)
```

이건 피하는 게 좋습니다.  
대신:

```text
BuildSpec으로 모델 재구성
→ state_dict 로드
```

가 맞습니다.

이 방식의 장점은 큽니다.

```text
1. 모델 코드가 정리되어도 재구성 가능
2. 분석 시 같은 BuildSpec으로 모델 복원 가능
3. checkpoint가 어떤 실험인지 명확함
4. 새 cell을 추가해도 registry만 확장하면 됨
5. 모델 저장/로드와 PSD 분석이 같은 기준 정보를 사용함
```

---

# 4. 모델 빌더는 BuildSpec만 보고 만든다

최종적으로 빌더는 이렇게 동작해야 합니다.

```text
BuildSpec
→ ModelRegistry
→ topology factory 선택
→ synapse factory 선택
→ cell factory 선택
→ model 생성
```

예시 흐름:

```text
topology.kind == mlp_stack
    → MLPStackFactory

cell.kind == lif
    → LIFCellFactory

synapse.kind == linear
    → LinearSynapseFactory
```

고정 모델이면:

```text
topology.kind == resnet
    → ResNetFactory

topology.variant == paper_resnet18
    → 논문 구조 고정 생성
```

즉 빌더는 `model_token` 문자열을 파싱하는 게 아니라, 정규화된 spec을 읽습니다.

---

# 5. trace 쪽도 표준화한다

여기가 PSD 분석과 직접 연결됩니다.

내가 생각하는 구조는 **모델이 바로 PSD map을 내놓는 것** 이 아니라, 두 단계를 분리하는 겁니다.

```text
1. 모델
   raw trace를 내놓음

2. SignalMapEmitter
   raw trace를 PSD-ready map으로 변환
```

왜냐하면 모델이 바로 PSD map까지 알면 모델 코드가 분석 코드에 오염됩니다.

---

## 5.1 모델이 내놓는 raw trace

모델은 이런 record를 내놓습니다.

```python
@dataclass
class LayerRecord:
    layer_name: str
    layer_index: int
    signal_kind: str  # input, hidden, output

    layer_input: torch.Tensor | None = None
    membrane: torch.Tensor | None = None
    spike: torch.Tensor | None = None

    extras: dict[str, torch.Tensor] = field(default_factory=dict)
```

기본 신호는 이 셋입니다.

```text
layer_input
membrane
spike
```

특수 뉴런 모델은 `extras` 에 더 넣습니다.

예를 들면 DH-SNN:

```python
extras = {
    "dendrite_current": ...,
    "dendrite_membrane": ...,
    "soma_membrane": ...,
}
```

GRU 계열:

```python
extras = {
    "z_gate": ...,
    "i_current": ...,
    "y_mem": ...,
    "y_spike": ...,
}
```

S4 계열:

```python
extras = {
    "state": ...,
    "filtered_output": ...,
}
```

이렇게 하면 PSD 분석 코드는 모델별로 `getattr(record, "z_gate")` 같은 식으로 더럽게 뒤질 필요가 없습니다.

---

## 5.2 raw trace의 표준 layout

모델에서 나오는 raw trace는 기본적으로 다음 layout으로 통일합니다.

```text
B,T,*
```

예를 들어:

```text
MLP trace
    B,T,F

CNN trace
    B,T,C,H,W

Transformer trace
    B,T,N,D

S4 trace
    B,T,D
```

여기서 $B$ 는 batch, $T$ 는 time입니다.

이렇게 해야 `SignalMapEmitter` 가 일관되게 처리할 수 있습니다.

---

# 6. PSD-ready signal map도 표준화한다

PSD 분석용 map은 무조건 이 형태로 갑니다.

```text
samples, rows, time
```

즉 tensor shape는:

```text
S,R,T
```

입니다.

변환 규칙은 다음처럼 고정합니다.

| raw trace | signal map |
|---|---|
| `B,T,F` | `B,F,T` |
| `B,T,C,H,W` | `B,C*H*W,T` |
| `B,T,N,D` | `B,N*D,T` |
| `B,T,D` | `B,D,T` |

즉 모델은 raw trace만 내고, 분석 모듈이 다음 변환을 합니다.

```text
raw trace
→ channel-major signal map
→ PSD
```

---

# 7. SignalMapRecord 구조

분석 쪽 record는 이렇게 둡니다.

```python
@dataclass
class SignalMapRecord:
    checkpoint_id: str
    split: str
    probe_family: str
    label: int | None

    layer_name: str
    layer_index: int
    signal_kind: str  # input, hidden, output
    series: str       # layer_input, membrane, spike, z_gate, ...

    maps: torch.Tensor  # S,R,T
```

예를 들면:

```text
checkpoint = epoch_010
split = test
probe_family = balanced_global
layer_name = layer_02
series = membrane
maps.shape = B,F,T
```

또는 CNN이면:

```text
layer_name = block_03
series = spike
maps.shape = B,C*H*W,T
```

이 구조가 있으면 PSD 분석은 모델 종류와 상관없이 돌아갑니다.

---

# 8. 전체 파이프라인은 이렇게 된다

최종 구조는 이겁니다.

```text
checkpoint load
    ↓
build_spec 읽기
    ↓
ModelRegistry.build(build_spec)
    ↓
state_dict load
    ↓
probe input forward
    ↓
ForwardResult / LayerRecord 생성
    ↓
TraceEmitter가 TraceRecord stream 생성
    ↓
SignalMapEmitter가 S,R,T map 생성
    ↓
PSDAccumulator가 PSD 누적
    ↓
CSV / plot / distance 저장
```

중요한 건 `psd_analysis.py` 가 모델 구조를 몰라야 한다는 겁니다.

```text
나쁜 구조:
    if model == lif:
        ...
    elif model == gru:
        ...
    elif model == s4:
        ...

좋은 구조:
    for trace_record in trace_emitter.emit(result):
        map_record = signal_map_emitter.to_map(trace_record)
        psd_accumulator.update(map_record)
```

---

# 9. "토폴로지/시냅스/뉴런모델 모두 각각 분리해서 저장" 에 대한 정확한 답

## MLP는 그렇습니다

MLP 계열은 아래를 분리해서 저장하는 게 맞습니다.

```text
topology
    몇 층인지, 각 층 width가 얼마인지

synapse
    Linear인지, bias를 쓰는지, normalization을 붙이는지

cell / neuron
    LIF인지, RF인지, TC-LIF인지, DH-SNN인지

readout
    output membrane을 어떻게 class score로 바꾸는지

trace
    어떤 신호를 분석 대상으로 내보낼 수 있는지
```

---

## 고정 논문 모델은 일부만 분리합니다

VGG, ResNet, S4, SpikeGRU, Spikformer는 논문 구조를 고정한다면, 모든 내부 synapse를 spec에 세세하게 쓰는 건 별로입니다.

이쪽은:

```text
architecture.family
architecture.variant
optional neuron spec
readout spec
trace spec
```

정도만 저장해도 됩니다.

예:

```yaml
model:
  topology:
    kind: resnet
    variant: paper_resnet18

  cell:
    kind: lif
    reset_mode: soft

  readout:
    kind: temporal_mean

  trace:
    capture:
      - membrane
      - spike
```

ResNet 내부의 conv kernel size, block 수, shortcut 구조는 `paper_resnet18` factory가 책임집니다.

---

# 10. 이 구조에서 "관찰 막전위/출력스파이크 맵" 은 어디서 정해지나?

두 군데에서 정합니다.

## 10.1 TraceSpec

어떤 신호를 뽑을지 정합니다.

```yaml
trace:
  layout: B,T,*
  capture:
    input: true
    layer_input: true
    membrane: true
    spike: true
    extras:
      - z_gate
      - i_current
      - dendrite_current
```

이건 모델 forward 또는 trace emitter가 참고합니다.

---

## 10.2 SignalMapSpec

어떤 방식으로 map으로 바꿀지 정합니다.

```yaml
signal_map:
  layout: S,R,T
  flatten_spatial: true
  flatten_token_feature: true
  dtype: float32
```

이건 PSD 분석 쪽이 참고합니다.

즉:

```text
TraceSpec
    모델에서 무엇을 꺼낼지

SignalMapSpec
    꺼낸 신호를 어떤 PSD map 형태로 만들지
```

입니다.

이 둘을 합쳐서 생각하면 네가 말한 "관찰 막전위/출력스파이크 맵 구조" 가 됩니다.

---

# 11. 가장 중요한 설계 원칙

## 원칙 1. BuildSpec은 모델 생성의 단일 기준이어야 함

학습할 때도:

```python
model = build_model(build_spec)
```

분석할 때도:

```python
model = build_model(checkpoint["build_spec"])
model.load_state_dict(checkpoint["state_dict"])
```

같은 경로를 써야 합니다.

---

## 원칙 2. checkpoint에는 activation을 저장하지 않음

막전위와 spike trace를 checkpoint에 저장하면 안 됩니다.

이유는:

```text
저장 용량 증가
probe set 변경 불가
trace 대상 변경 불가
PSD 정의 변경 시 재사용 어려움
```

checkpoint에는 구조와 파라미터만 저장합니다.

```text
build_spec + state_dict
```

trace는 분석할 때 다시 forward해서 뽑습니다.

---

## 원칙 3. 모델은 PSD를 몰라야 함

모델은 다음까지만 책임집니다.

```text
input을 받아 forward
layer별 raw trace 반환
```

PSD map 변환, FFT, distance 계산은 분석 모듈 책임입니다.

---

## 원칙 4. 분석 코드는 모델 종류를 몰라야 함

분석 코드는 이것만 보면 됩니다.

```text
TraceRecord
SignalMapRecord
```

모델이 LIF인지, DH-SNN인지, S4인지, Spikformer인지는 몰라야 합니다.

---

# 12. 최종적으로 내가 말하는 구조

한 줄로 말하면 네 말이 맞습니다.

```text
정교하게 표준화된 model metadata를 만들고,
그것을 빌더, 저장, 로드, 분석에 모두 사용한다.
```

단, 그 metadata는 그냥 잡다한 정보가 아니라 구조화된 spec이어야 합니다.

```text
BuildSpec
    topology
    synapse
    temporal cell / neuron
    readout
    trace policy

Checkpoint
    BuildSpec
    state_dict
    dataset_spec
    training_spec

TraceRecord
    layer별 raw membrane / spike / extra state

SignalMapRecord
    PSD 분석용 S,R,T map
```

최종 목표는 이겁니다.

```text
새 MLP 뉴런 모델 추가
    → cell factory만 추가

새 고정 논문 모델 추가
    → fixed model factory만 추가

새 관찰 신호 추가
    → LayerRecord.extras에 추가

새 PSD 분석 방식 추가
    → analysis 모듈만 수정

checkpoint 저장/로드
    → BuildSpec + state_dict로 통일
```

즉 **모델 생성, 모델 저장, 모델 복원, 레이어 trace 관찰, PSD 분석이 전부 같은 표준 명세 위에서 돌아가게 만드는 것** 입니다.

## Prompt:
2026. 5. 22. 오후 9:14:08

1. 내가 생각하는 구조는 모델이 바로 PSD map을 내놓는 것 이 아니라, 두 단계를 분리하는 겁니다. <- 왜? 어차피 표준 정해주면 모델이 맵을 내놔도 되잖아

## Response:
2026. 5. 22. 오후 9:14:12

가능은 하지만, 핵심은 작동 여부가 아니라 map 변환을 core 모델 책임으로 둘 때 장기적으로 안전한지입니다. Map 변환은 모델 정의보다 분석 view에 가깝습니다.

## Response:
2026. 5. 22. 오후 9:15:30

> **모델과 맵 구분 제안**
>
> 질문에 맞게 답변을 간결하게 하려고 해요. 모델이 맵을 출력할 수는 있지만, 추천하지 않아요. 이유는: (1) PSD 맵은 뷰/레이아웃 변환이라 모델 의미와 다르고, (2) 같은 추적도 여러 방식으로 맵핑될 수 있죠. 솔루션은 맵을 모델 외부에서 처리하는 형태가 더 나을 것 같아요.
>
> 3m 2s 동안 생각함

## Response:
2026. 5. 22. 오후 9:17:14

가능합니다. 네 말이 **원칙적으로 틀린 건 아닙니다.**  
표준을 강하게 정하면 모델이 바로 PSD map을 내놓게 만들 수도 있습니다.

다만 내가 분리하자고 한 이유는, **PSD map은 모델의 자연스러운 출력이 아니라 분석용 표현** 이기 때문입니다.

정확히 구분하면 이렇습니다.

```text
모델 고유 출력
    layer_input
    membrane trace
    spike trace
    extra state trace

분석용 표현
    PSD-ready signal map: (samples, rows, time)
```

즉 막전위 $V$ 와 스파이크 $S$ 는 모델 동역학의 산물입니다.  
반면 PSD map은 그 산물을 **분석하기 좋게 축을 바꾸고 펼친 표현** 입니다.

---

# 1. 모델이 map을 직접 내놓으면 뭐가 문제인가?

## 1.1 map 변환은 분석 정책이다

예를 들어 CNN trace가 있다고 합시다.

$$
V\in\mathbb{R}^{B\times T\times C\times H\times W}
$$

이걸 PSD map으로 바꿀 때 방법이 여러 개입니다.

### 방식 A: 모든 위치를 row로 펼침

```text
B,T,C,H,W
→ B,C*H*W,T
```

이건 현재 네 분석에 가까운 방식입니다.

---

### 방식 B: spatial average 후 channel별 PSD

```text
B,T,C,H,W
→ B,C,T
```

이러면 채널별 temporal filtering 특성을 보기 좋습니다.

---

### 방식 C: 위치별 PSD를 따로 보고 나중에 channel별 통계

```text
B,T,C,H,W
→ B,C,H*W,T
```

이러면 channel과 spatial position을 분리해서 분석할 수 있습니다.

---

### 방식 D: 특정 layer group만 추출

ResNet이라면:

```text
block1.conv1 membrane
block1.conv2 membrane
block1.residual_add membrane
block1.output spike
```

중 무엇을 PSD 대상으로 할지도 분석 정책입니다.

따라서 `PSD map` 은 단순한 "표준 출력" 이 아니라, **어떤 관점으로 신호를 볼 것인가** 에 대한 선택입니다.

모델이 map을 직접 내놓으면 이 선택이 모델 안에 박힙니다.  
나중에 분석 관점을 바꾸면 모델 코드도 바꿔야 합니다.

---

# 2. trace는 보존 정보이고, map은 투영 정보다

raw trace는 원래 구조를 보존합니다.

```text
B,T,F
B,T,C,H,W
B,T,N,D
```

하지만 map은 보통 구조를 펼칩니다.

```text
B,F,T
B,C*H*W,T
B,N*D,T
```

이 변환은 편하지만, 일부 구조 정보를 잃거나 약화시킵니다.

예를 들어:

```text
B,T,C,H,W → B,C*H*W,T
```

로 만들면 row index $r$ 가 원래 어느 channel, 어느 height, 어느 width였는지 metadata 없이는 바로 알기 어렵습니다.

물론 metadata를 붙이면 됩니다.

```text
row_index → (channel, height, width)
```

하지만 그러면 결국 모델이 PSD map뿐 아니라 map metadata까지 책임져야 합니다.  
이건 모델 책임이 아니라 분석 모듈 책임에 가깝습니다.

---

# 3. 같은 trace에서 여러 분석 map을 만들 수 있다

같은 막전위 trace 하나에서 만들 수 있는 분석 표현은 많습니다.

```text
PSD map
spectrogram map
autocorrelation map
raster plot
layer-wise mean membrane curve
channel-wise PSD
spatial-wise PSD
sample-wise PSD
class-wise PSD
```

즉 raw trace는 공통 원천이고, map은 목적별 변환입니다.

만약 모델이 PSD map을 직접 내놓게 하면:

```text
PSD 분석에는 좋음
다른 분석에는 애매함
```

이 됩니다.

반대로 모델이 raw trace를 내놓고, 바깥에서 map을 만들면:

```text
같은 trace로 PSD도 하고
spectrogram도 하고
raster plot도 하고
channel-wise 분석도 가능
```

합니다.

---

# 4. 모델 코드가 분석 코드에 오염된다

모델의 본래 책임은 이것입니다.

```text
input sequence
→ layer dynamics
→ output membrane/spike
```

PSD map 변환의 책임은 이것입니다.

```text
trace tensor
→ S,R,T map
→ FFT
→ PSD summary
```

모델이 map을 직접 내놓으면 모델 내부에 이런 코드가 들어갑니다.

```python
if tensor.ndim == 3:
    maps = tensor.transpose(1, 2)

elif tensor.ndim == 5:
    maps = tensor.permute(0, 2, 3, 4, 1).flatten(1, 3)

elif tensor.ndim == 4:
    maps = tensor.permute(0, 2, 3, 1).flatten(1, 2)
```

이건 모델 동역학 코드가 아닙니다.  
분석용 shape normalization 코드입니다.

이걸 LIF, RF, TC-LIF, DH-SNN, VGG, ResNet, S4, GRU, Spikformer마다 넣으면 같은 로직이 여러 군데 복제됩니다.

나중에 map 규칙을 바꾸면 모든 모델을 고쳐야 합니다.  
그건 좋지 않습니다.

---

# 5. 학습 forward와 분석 forward가 달라진다

학습 중에는 보통 다음이 필요합니다.

```text
output membrane
output spike
loss 계산
backprop
```

PSD 분석 중에는 다음이 필요합니다.

```text
hidden layer_input
hidden membrane
hidden spike
extras
detach
float32 변환
S,R,T map 변환
FFT
```

이 둘은 요구사항이 다릅니다.

모델이 PSD map까지 직접 내놓으면 학습 forward가 불필요하게 무거워질 수 있습니다.  
특히 trace를 map으로 바꾸는 과정은 `.contiguous()`, `flatten`, `permute`, `detach`, CPU 이동 같은 처리가 섞일 수 있습니다.

이런 처리는 학습 중 gradient 흐름과 메모리 사용량에 영향을 줄 수 있습니다.

그래서 나는 이렇게 나누는 게 낫다고 봅니다.

```text
model.forward_records(...)
    gradient가 필요한 모델 출력과 raw trace

SignalMapEmitter
    분석 시점에서 detach된 trace를 S,R,T map으로 변환
```

---

# 6. checkpoint와 분석 버전을 독립시킬 수 있다

이 부분이 중요합니다.

모델이 PSD map을 직접 내놓으면, map 정의가 모델 구현 일부가 됩니다.

예를 들어 나중에 map 정의를 바꾼다고 합시다.

```text
기존:
    CNN map = B,C*H*W,T

변경:
    CNN map = B,C,T after spatial mean
```

모델이 map을 직접 내놓는 구조라면:

```text
모델 코드 수정
→ checkpoint 분석 호환성 애매
→ 기존 결과와 새 결과 비교 어려움
```

반대로 map 변환기를 분리하면:

```text
같은 checkpoint
같은 raw trace
다른 SignalMapSpec
→ 서로 다른 PSD 분석 가능
```

즉 모델의 BuildSpec과 분석의 SignalMapSpec을 독립적으로 버전 관리할 수 있습니다.

```text
BuildSpec
    모델이 무엇인가

TraceSpec
    무엇을 관찰할 것인가

SignalMapSpec
    관찰 신호를 어떻게 map으로 바꿀 것인가

PSDSpec
    어떤 PSD 계산법을 쓸 것인가
```

이 분리가 장기적으로 훨씬 안전합니다.

---

# 7. 그래도 모델이 map을 내놓게 만들 수 있는 경우

내가 반대하는 건 **core model.forward가 PSD map을 직접 책임지는 것** 입니다.

하지만 아래 구조는 괜찮습니다.

```text
model
    raw trace 반환

AnalysisModelWrapper
    model을 감싸서 signal map 반환
```

즉 이런 방식은 좋습니다.

```python
result = model.forward_records(x, trace_level="full")
map_records = signal_map_emitter.from_forward_result(result)
```

또는 wrapper로:

```python
map_records = analysis_model.emit_signal_maps(x)
```

이때 `analysis_model` 은 진짜 모델이 아니라 분석용 wrapper입니다.

구조로 보면:

```text
Core model
    model dynamics 책임

TraceEmitter
    raw trace record 생성

SignalMapEmitter
    PSD-ready map 생성
```

입니다.

사용자 입장에서는 `emit_signal_maps()` 한 번만 호출해도 되지만, 내부적으로는 분리되어 있습니다.

---

# 8. 내가 추천하는 절충안

네가 말한 직관도 맞습니다.  
"모델이 표준 map을 바로 내놓으면 편하지 않나?" 라는 말은 실용적으로 맞습니다.

그래서 최종 구조는 이렇게 하면 됩니다.

## core forward

```python
result = model.forward_records(
    x,
    trace_level="full",
)
```

반환:

```text
ForwardResult
    input_record
    hidden_records
    output_record
```

각 record에는:

```text
layer_input
membrane
spike
extras
```

가 들어 있습니다.

---

## analysis convenience method

```python
map_records = model.emit_signal_maps(
    x,
    trace_level="full",
    signal_map_spec=signal_map_spec,
)
```

하지만 이 메서드는 모델 내부에서 직접 map을 만드는 게 아니라:

```python
def emit_signal_maps(self, x, trace_level, signal_map_spec):
    result = self.forward_records(x, trace_level=trace_level)
    return SignalMapEmitter(signal_map_spec).from_forward_result(result)
```

처럼 동작합니다.

즉 외부에서 보기에는 모델이 map을 내놓는 것처럼 보이지만, 설계상으로는 분리되어 있습니다.

이게 가장 좋은 절충입니다.

---

# 9. 최종 권장 구조

나는 아래 구조를 추천합니다.

```text
CoreModel
    forward_records()
    raw trace 반환

TraceEmitter
    ForwardResult → TraceRecord stream

SignalMapEmitter
    TraceRecord → SignalMapRecord
    maps.shape = S,R,T

PSDAccumulator
    SignalMapRecord → PSD summary
```

그리고 편의를 위해 wrapper를 둡니다.

```python
class AnalysisModelWrapper:
    def __init__(self, model, signal_map_emitter):
        self.model = model
        self.signal_map_emitter = signal_map_emitter

    def emit_signal_maps(self, x, *, trace_level="full"):
        result = self.model.forward_records(x, trace_level=trace_level)
        return self.signal_map_emitter.from_forward_result(result)
```

이러면 네가 원하는 사용성도 나옵니다.

```python
map_records = analysis_model.emit_signal_maps(x)
```

하지만 core model은 PSD map 정책에 묶이지 않습니다.

---

# 10. 한 줄 결론

**모델이 PSD map을 직접 내놓는 것은 가능하지만, core model의 책임으로 두면 분석 정책이 모델 코드에 박혀서 확장성이 떨어집니다.**  

가장 좋은 방식은:

```text
모델은 raw trace를 표준 형식으로 내놓고,
SignalMapEmitter가 그 trace를 PSD-ready map으로 바꾼다.
```

단, 사용 편의를 위해:

```python
model.emit_signal_maps(...)
```

같은 wrapper 메서드는 만들어도 됩니다.  
중요한 건 **내부 책임 분리** 입니다.



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)