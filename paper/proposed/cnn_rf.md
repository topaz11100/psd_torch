# CNN RF 명세서

## 0. 목적

이 문서는 현재 프로젝트가 사용하는 CNN RF layer 의 구현 해석 규칙을 정의한다. 여기서 CNN RF 는 SCNN 맥락에서 convolution kernel 로 만든 입력 전류를 vanilla RF neuron 에 넣는 경로를 뜻한다. 즉 CNN RF 는 별도의 새로운 neuron model 이 아니라, 입력 결합만 convolution 으로 바뀌고 neuron 내부 동역학은 vanilla RF 를 그대로 사용하는 경로다.

문서 분리는 아래와 같다.

- 본 문서: CNN RF layer 의 입력 결합, 학습 파라미터, bias 정책, vanilla RF 와의 관계
- `vanila_rf.md` : RF 동역학, spike 결정 규칙, reset 정책, exact ZOH, 초기화, 저장 해석량
- `paper/proposed/vanila_scenario.md` : clip / structure / structclip 시나리오
- `paper/proposed/filter_analysis.md` : `rho`, `f_cyc_per_sample` 통계와 histogram 저장 규칙
- `paper/proposed/readout.md` : output layer readout 규칙

## 1. 설계 원칙

SCNN 의 CNN RF layer 는 convolution kernel 이 만든 current map 을 각 neuron 에 공급하고, 각 neuron 은 vanilla RF 와 동일한 intrinsic dynamics 를 따른다. 따라서 CNN RF path 의 핵심 차이는 neuron 자체가 아니라 입력 결합 경로에 있다.

CNN RF layer 에서 학습되는 파라미터는 아래와 같다.

1. convolution kernel weight
2. 각 neuron 의 intrinsic RF parameter $b_i$, $\omega_i$

convolution path 에는 bias 를 두지 않으며, bias 는 학습 파라미터에도 포함되지 않는다.

RF 계열의 직접 해석량은 vanilla RF 와 동일하게 아래 두 값이다.

$$
\rho_i = e^{b_i \Delta t}
$$

$$
f_{\mathrm{cyc/sample},i} = \frac{\omega_i \Delta t}{2\pi}
$$

자세한 해석과 저장 규칙은 `vanila_rf.md` 를 따른다.

## 2. 입력 결합

입력 feature map 또는 이전 layer 활성 $u_t$ 가 들어오면, CNN RF layer 로 들어가는 전류는 convolution 으로 만든다.

$$
I_t = K * u_t
$$

여기서 $K$ 는 학습 가능한 convolution kernel 이고, $*$ 는 프로젝트가 채택한 convolution 연산을 뜻한다. 이 경로에는 bias term 이 없다.

이렇게 만들어진 $I_t$ 를 각 neuron element 에 넣은 뒤의 RF state update, spike 결정 변수, reset 정책, exact ZOH 이산화, clip 단위, 초기화, 저장 해석량은 모두 `vanila_rf.md` 와 동일하다.

## 3. vanilla RF 참조 범위

CNN RF 문서에서는 아래 항목을 별도로 다시 정의하지 않는다.

- 연속시간 RF 식
- spike 결정 변수와 출력 spike 정의
- `no_reset` / `soft_reset` 정책
- exact ZOH 구현식과 안정성 해석
- `rho`, `f_{\mathrm{cyc/sample}}` 해석
- RF 동역학 파라미터 초기화와 clip 해석

위 항목은 모두 `vanila_rf.md` 를 그대로 따른다.

## 4. 시나리오와 저장 링크

- CNN RF 실험의 neuron-side 시나리오 해석은 `paper/proposed/vanila_scenario.md` 를 따른다.
- RF filter 통계와 histogram 저장은 `paper/proposed/filter_analysis.md` 를 따른다.
- output layer readout 정의는 `paper/proposed/readout.md` 를 따른다.
- neuron 내부 해석은 `vanila_rf.md` 와 동일하다.
