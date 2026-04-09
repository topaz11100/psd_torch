# Vanila LIF 명세서

## 0. 목적

이 문서는 현재 프로젝트가 사용하는 baseline LIF neuron 의 구현 규칙과 해석 규칙을 정의한다. 본 프로젝트의 vanilla LIF 는 learnable membrane decay coefficient 를 가지는 dense LIF path 이며, output layer 도 같은 base neuron 을 사용한다.

문서 분리는 아래와 같다.

- 본 문서: vanilla LIF 동역학, spike 결정 규칙, 초기화, 저장 해석량
- `paper/proposed/vanila_scenario.md` : clip / structure / structclip 시나리오
- `paper/proposed/filter_analysis.md` : `alpha` 통계와 histogram 저장 규칙
- `paper/proposed/readout.md` : output layer readout 규칙

## 1. 설계 원칙

vanilla LIF layer 는 bias 없는 dense learned coupling 과 learnable decay $\alpha$ 를 갖는 baseline path 다. reset 은 임의의 `v_reset` 을 두지 않고 fixed threshold 기반 subtractive soft reset 으로 고정한다.

이 layer 에서 학습되는 파라미터는 아래와 같다.

1. dense weight
2. 각 neuron 의 membrane decay coefficient $\alpha_i$

bias 는 두지 않으며 학습 파라미터에도 포함되지 않는다.

구현에는 compatibility 용 optional learnable threshold path 가 남아 있을 수 있지만, 공식 PSD 실험 경로는 고정 threshold 설정을 기준으로 문서화한다. output layer 역시 같은 base LIF neuron 이며, output neuron 뒤에 learned NN head 를 두지 않는다.

## 2. LIF 동역학

입력층 또는 이전 레이어의 활성 $u_t$ 가 들어오면 current term 은 dense linear transform 으로 만든다.

$$
I_t = W u_t
$$

즉 vanilla LIF baseline 에서는 bias term 을 두지 않는다.

`recurrent=True` 인 경우에는 이전 시점의 output spike 에 대한 project-side recurrent projection 이 current term 에 더해진다. recurrent adapter 는 intrinsic LIF update 자체를 바꾸지 않는다.

threshold 를 $\theta_i$ 라 할 때, 현재 프로젝트의 baseline LIF update 는

$$
u_{i,t} = \alpha_i u_{i,t-1} + I_{i,t} - \theta_i o_{i,t-1}
$$

$$
m_{i,t} = u_{i,t} - \theta_i
$$

$$
o_{i,t} = O(m_{i,t})
$$

의 의미론으로 해석한다. 여기서 $u_{i,t}$ 는 threshold subtraction 직전의 soma-side 상태, $m_{i,t}$ 는 스파이크 생성함수 $O$ 에 실제로 들어가 spike 여부를 결정하는 막전위 결정 변수, $o_{i,t}$ 는 뉴런의 출력 spike signal 이다.

따라서 membrane signal family 는 단순 저장 state 전체가 아니라 spike 결정에 직접 대응하는 변수 의미론을 가진다. spike signal family 는 뉴런의 출력 spike signal 그 자체다.

reset 은 독립적인 learnable reset level 이 아니라 이전 step output spike 에 비례한 subtractive term 으로만 구현한다.

## 3. 학습 파라미터와 해석량

LIF 계열의 직접 해석량은 `alpha` 다. filter 통계 저장은 raw unconstrained parameter 가 아니라 유효 domain 으로 매핑된 $\alpha_i$ 를 기준으로 한다.

한 LIF dense layer 에서 학습되는 항목은 아래와 같다.

- dense weight $W$
- 각 neuron 의 decay parameter $\alpha_i$

bias 는 사용하지 않으며 학습 파라미터에도 포함되지 않는다.

반대로 reset level choice, surrogate shape hyperparameter, 공식 PSD 경로의 threshold 정책은 고정 실험 설정이다.

## 4. 동역학 학습 파라미터 초기화

vanilla / clip LIF 의 동역학 학습 파라미터 초기화는 fan-based Kaiming / Xavier 가 아니라 유효 물리 파라미터 구간에서의 bounded uniform sampling 을 사용한다.

free LIF 에서는 각 neuron 의 decay coefficient 를

$$
\alpha_i \in (0, 1)
$$

에서 bounded uniform sampling 으로 초기화한다.

`lif_clip`, `lif_structclip` 에서는 같은 bounded-uniform 규칙을 유지하되, 각 neuron 이 배정받은 `alpha_clip_edges` interval 안으로 support 를 좁혀 초기화한다. 경계 끝점이 정확히 0 또는 1 인 경우에도 raw unconstrained parameter 가 무한대로 가지 않도록 구현에서는 작은 trim epsilon 을 둔 열린구간에서 샘플링할 수 있다.

## 5. 시나리오와 저장 링크

- `lif`, `lif_struct`, `lif_clip`, `lif_structclip` 의 그룹 구성과 `tear` 규칙은 `paper/proposed/vanila_scenario.md` 를 따른다.
- LIF filter 통계와 histogram 저장은 `paper/proposed/filter_analysis.md` 를 따른다.
- output layer readout 정의는 `paper/proposed/readout.md` 를 따른다.
- output layer 는 same-base-neuron output layer 이며 추가 NN head 를 두지 않는다.

## 6. 구현 대응

현재 코드 기준 대응은 아래와 같다.

- `src/neurons/LIF_neuron.py` : learnable decay, fixed subtractive soft reset, optional recurrent adapter
- `src/neurons/vanila_lif.py` : LIF dense layer compatibility alias
- `src/util/psd_analysis_driver.py` : RF / LIF PSD variant builder, grouped block 저장, metadata 기록
