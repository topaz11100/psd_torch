# 브랜치 · 스파이킹 젤리 LIF API

**User:** Anonymous  
**Created:** 5/22/2026 19:39:28  
**Updated:** 5/22/2026 19:40:03  
**Exported:** 5/23/2026 2:30:35  
**Link:** [https://chatgpt.com/c/6a10325a-c0b4-8323-8604-e083ba92314a](https://chatgpt.com/c/6a10325a-c0b4-8323-8604-e083ba92314a)  

## Prompt:
2026. 5. 22. 오후 7:34:08

스파이킹 젤리에서 바닐라 lif 레이어 는 어떤 api를 쓰면 돼?

## Response:
2026. 5. 22. 오후 7:34:09

> **SpikingJelly LIFNode 공식 문서 검색**
>
>
> **문서 서명 정보 검색**
>
>
> **찾기 위한 인용과 서명 확인**
>
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
>
> **Searching for official documentation**
>
>
> **확인해야 할 PLIF 클래스 및 tau 속성**
>
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
2026. 5. 22. 오후 7:39:26

nn.Linear(784, 256),
    neuron.ParametricLIFNode(init_tau=2.0, surrogate_function=surrogate.ATan()),

근데 이런식으로 되면 자연스럽게 bias 항도 추가되는거지?

## Response:
2026. 5. 22. 오후 7:39:28

> **설명: Linear와 ParametricLIFNode에서의 바이어스**
>
> `nn.Linear`는 기본적으로 `bias=True`로 설정되어 있어, affine 변환에 바이어스 벡터 크기 256 값이 포함됩니다. 반면, `ParametricLIFNode` 자체는 바이어스를 가지지 않으며, 학습 가능한 타우 값 `w`만 존재합니다. `v_threshold`는 사용자 정의 시 학습 가능할 수도 있습니다.
>
> **PyTorch NN.Linear 기본 설정과 SpikingJelly 분석**
>
> [Linear — PyTorch 2.12 documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html)
> [SpikingJelly is an open-source deep learning framework ...](https://github.com/fangwei123456/spikingjelly)
> [Classify DVS128 Gesture — spikingjelly alpha 文档](https://spikingjelly.readthedocs.io/zh_CN/0.0.0.0.6/clock_driven_en/14_classify_dvsg.html)
> [nn.Linear in PyTorch: Shapes, Bias, and Examples](https://docs.kanaries.net/topics/Python/nn-linear)
> [Pytorch Tutorial: torch.nn.Linear](https://www.youtube.com/watch?v=AmgBoQjEAZY)
> [What is the class definition of nn.Linear in PyTorch?](https://stackoverflow.com/questions/54916135/what-is-the-class-definition-of-nn-linear-in-pytorch)
> [[DL] torch.nn.Linear 에 대하여 - ds31x - 티스토리](https://ds31x.tistory.com/421)
> [Linear — PyTorch master documentation](https://glaringlee.github.io/generated/torch.nn.Linear.html)
> [How are layer weights and biases initialized by default?](https://discuss.pytorch.org/t/how-are-layer-weights-and-biases-initialized-by-default/13073)
> [Linear in PyTorch](https://dev.to/hyperkai/linear-in-pytorch-4a0)
>
> **Bias 항 포함 여부 및 해석**
>
> 네, `nn.Linear(784, 256)`는 바이어스가 기본적으로 `True`로 설정되어 있어, 이 경우 `y = xW^T + b`로 작동합니다. 이후 PLIF는 바이어스를 적용한 `x`를 받습니다. 바이어스는 각 출력 뉴런에 대해 256 크기의 벡터로, 입력의 DC 성분에 영향을 줄 수 있습니다.
>
> 19s 동안 생각함

핵심은 두 개를 분리해서 봐야 합니다. `bias` 는 `nn.Linear` 쪽 속성이고, `ParametricLIFNode` 가 자동으로 만드는 항은 아닙니다.

네. **추가됩니다.**

단, 정확히 말하면 **`ParametricLIFNode` 가 bias를 추가하는 게 아니라 `nn.Linear(784, 256)` 가 bias를 추가합니다.**

```python
nn.Linear(784, 256)
```

는 기본적으로 아래와 같습니다.

```python
nn.Linear(784, 256, bias=True)
```

PyTorch 공식 문서에서도 `torch.nn.Linear(in_features, out_features, bias=True)` 로 정의되어 있고, 선형층은 $y=xA^T+b$ 를 계산합니다. 즉 `bias=True` 가 기본값입니다. ([PyTorch Docs](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html?utm_source=chatgpt.com))

---

## 따라서 이 블록은 이렇게 동작함

```python
nn.Linear(784, 256),
neuron.ParametricLIFNode(init_tau=2.0, surrogate_function=surrogate.ATan())
```

계산 흐름은 대략 다음입니다.

$$
I[t]=X[t]W^T+b
$$

그 다음 이 $I[t]$ 가 LIF 뉴런 입력으로 들어갑니다.

```text
입력 X[t]
  ↓
Linear: X[t]W^T + b
  ↓
ParametricLIFNode: 막전위 누적, 누설, 발화, 리셋
```

즉 bias는 **시냅스층 출력 전류에 더해지는 상수항** 처럼 작동합니다.

---

## bias shape

```python
nn.Linear(784, 256)
```

이면 파라미터는 다음처럼 생깁니다.

```text
weight: (256, 784)
bias:   (256,)
```

즉 256개 출력 뉴런마다 bias 하나가 있습니다.

확인하면:

```python
fc = nn.Linear(784, 256)

print(fc.weight.shape)  # torch.Size([256, 784])
print(fc.bias.shape)    # torch.Size([256])
```

---

## `ParametricLIFNode` 자체에는 bias가 없음

```python
neuron.ParametricLIFNode(init_tau=2.0)
```

얘가 학습하는 것은 기본적으로 bias가 아니라 **막전위 시간상수 관련 파라미터** 입니다.

공식 문서 기준으로 `ParametricLIFNode` 는 $\frac{1}{\tau}=\operatorname{Sigmoid}(w)$ 형태로 두고, $w$ 를 학습 가능한 파라미터로 둡니다. 

즉 이 구조에서 학습 파라미터는 대략 이렇게 됩니다.

```text
Linear.weight
Linear.bias
ParametricLIFNode.w
```

---

## bias를 없애고 싶으면

명시적으로 이렇게 해야 합니다.

```python
nn.Linear(784, 256, bias=False),
neuron.ParametricLIFNode(init_tau=2.0, surrogate_function=surrogate.ATan())
```

그러면 선형층은

$$
I[t]=X[t]W^T
$$

만 계산하고, 상수항 $b$ 는 없습니다.

---

## PSD 분석 관점에서 중요한 점

네가 레이어 출력의 PSD를 분석한다면 bias는 꽤 중요합니다.

왜냐하면 bias는 시간에 따라 변하지 않는 상수 입력이므로, 매 time step에 같은 방향으로 들어갑니다. 따라서 입력 시계열 관점에서는 **DC 성분 또는 저주파 성분을 강화할 수 있습니다.**

특히 multi-step 입력이 $(T,B,784)$ 형태라면 `Linear` 의 bias는 모든 $t$ 에 대해 broadcast 됩니다.

```text
t=1: X[1]W^T + b
t=2: X[2]W^T + b
t=3: X[3]W^T + b
...
```

따라서 레이어 대표 PSD를 해석할 때:

| 설정 | 의미 |
|---|---|
| `bias=True` | 표준 MLP/FC 구조에 가까움. 하지만 DC/저주파 해석에 bias 영향이 섞임 |
| `bias=False` | 해석이 더 깔끔함. 입력과 weight에 의한 동역학만 보기 쉬움 |

네 목적이 **성능 실험** 이면 `bias=True` 도 자연스럽습니다.  
네 목적이 **레이어 주파수 특성 해석** 이면 `bias=False` 도 강하게 고려할 만합니다.

정리하면:

```python
nn.Linear(784, 256)
```

이렇게 쓰면 **bias는 자동으로 들어갑니다.**

```python
neuron.ParametricLIFNode(...)
```

는 bias를 추가하지 않고, **학습 가능한 tau 계열 파라미터를 추가합니다.**

---

**Sources:**

- [Linear — PyTorch 2.12 documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html?utm_source=chatgpt.com)



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)