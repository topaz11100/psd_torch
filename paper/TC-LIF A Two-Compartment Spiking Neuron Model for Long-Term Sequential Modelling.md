# TC-LIF: 장기 순차 모델링(long-term sequential modelling)을 위한 2-구획(two-compartment) 스파이킹 뉴런 모델

Shimin Zhang1*, Qu Yang2*, Chenxiang Ma1, Jibin Wu1†, Haizhou Li3, 2, Kay Chen Tan1

1 홍콩이공대학교(The Hong Kong Polytechnic University), 중국 홍콩 특별행정구  
2 싱가포르국립대학교(National University of Singapore), 싱가포르  
3 홍콩중문대학교 선전캠퍼스(The Chinese University of Hong Kong, Shenzhen; CUHK-Shenzhen), 중국 광둥성

shi-min.zhang@connect.polyu.hk, quyang@u.nus.edu, chenxiang.ma@connect.polyu.hk, jibin.wu@polyu.edu.hk, haizhou.li@u.nus.edu, kctan@polyu.edu.hk

\* 이 저자들은 동등하게 기여하였다.  
† 교신저자(corresponding author)

저작권 © 2024, Association for the Advancement of Artificial Intelligence (AAAI). 모든 권리 보유.

## 초록(Abstract)

잠재적인 기회와 위험에 연관된 감각 단서(sensory cue)를 식별하는 일은, 유용한 단서들을 긴 지연(long delay)으로 분리하는 무관한 사건들 때문에 자주 복잡해진다. 그 결과, 최첨단 스파이킹 신경망(spiking neural network, SNN)이 서로 멀리 떨어진 단서들 사이의 장기 시간 의존성(long-term temporal dependency)을 확립하는 일은 여전히 도전적인 과제로 남아 있다. 이러한 문제를 해결하기 위해, 우리는 생물학적 영감을 받은 새로운 2-구획 누설 적분-발화(Two-Compartment Leaky Integrate-and-Fire) 스파이킹 뉴런 모델인 TC-LIF를 제안한다. 제안하는 모델은 장기 시간 의존성 학습을 촉진하도록 정교하게 설계된 세포체(somatic) 및 수상돌기(dendritic) 구획을 포함한다. 또한 확장된 시간 구간에 걸쳐 오차 기울기(error gradient)를 전파하는 데 있어 TC-LIF의 유효성을 검증하기 위한 이론적 분석을 제시한다. 다양한 시간 분류(temporal classification) 과제에 대한 실험 결과는 제안된 TC-LIF 모델이 우수한 시간 분류 능력, 빠른 학습 수렴, 그리고 높은 에너지 효율성을 갖는다는 것을 보여준다. 따라서 본 연구는 차세대 뉴로모픽 컴퓨팅(neuromorphic computing) 시스템에서 도전적인 시간 처리 과제를 해결할 수 있는 다양한 기회를 연다. 우리의 코드는 `https://github.com/ZhangShimin1/TC-LIF` 에 공개되어 있다.

## 서론(Introduction)

스파이킹 신경망(spiking neural network, SNN)은 생물학적 개연성(biological plausibility)과 에너지 효율적인 신경 계산(neural computation) 잠재력으로 인해 최근 큰 관심을 받아 왔다 (Maass 1997; Pfeiffer and Pfeil 2018). SNN의 기본 계산 단위인 스파이킹 뉴런(spiking neuron)은 생물학적 뉴런에서 관찰되는 풍부한 신경 동역학(neuronal dynamics)을 모사하여 시공간 패턴(spatio-temporal pattern)의 부호화, 처리, 저장을 가능하게 한다 (Gerstner et al. 2014). 또한 스파이킹 뉴런은 이산적인 스파이크(discrete spike)를 통해 서로 통신하며, 이러한 이벤트 기반(event-driven) 동작은 초저전력 신경 계산을 가능하게 한다 (Davies et al. 2018; Pei et al. 2019). 그 결과 효율적인 학습이 가능해진다 (Ma, Fang, and Wang 2023; Fang et al. 2023; Liu et al. 2022).

실제로 단일-구획(single-compartment) 스파이킹 뉴런 모델은 대규모 뇌 시뮬레이션(brain simulation)과 뉴로모픽 컴퓨팅에서 널리 채택되어 왔으며, 대표적인 예로 누설 적분-발화(Leaky Integrate-and-Fire, LIF) 모델 (Abbott and Kepler 2005), Izhikevich 모델 (Izhikevich 2003), 적응형 지수 적분-발화(Adaptive Exponential Integrate-and-Fire, AdEx) 모델 (Brette and Gerstner 2005)이 있다. 이러한 단일-구획 모델은 생물학적 뉴런을 하나의 전기 회로(single electrical circuit)로 추상화하여, 생물학적 뉴런의 본질적 신경 동역학은 보존하면서도 수상돌기(dendrite)의 복잡한 기하 구조와 수상돌기-세포체 상호작용은 무시한다. 이러한 수준의 추상화는 모델링 부담을 크게 줄여, 대규모 생물학적 신경망의 거동을 연구하거나 뉴로모픽 시스템에서 복잡한 패턴 인식(pattern recognition) 과제를 수행하는 데 더 적합하게 만든다 (Wu et al. 2021a; Chen et al. 2023; Ma et al. 2022).

단일-구획 스파이킹 뉴런 모델은 제한된 시간 맥락(limited temporal context)을 갖는 패턴 인식 과제에서 유망한 성능을 보여주었지만 (Tavanaei et al. 2019; Zhang et al. 2021; Wu et al. 2021b; Yang et al. 2022; Yao et al. 2022), 장기 시간 의존성(long-term temporal dependency)이 필요한 과제를 해결하는 능력은 여전히 제한적이다. 이 문제는 주로 SNN에서 장기 시간적 신용 할당(long-term temporal credit assignment, TCA)을 수행하기 어렵다는 점에서 비롯된다. TCA는 미래의 보상(reward) 또는 페널티(penalty)에 기여한 입력 스파이크를 식별하고, 그에 따라 관련 연결을 강화하거나 약화하는 과정을 의미한다. 그러나 스파이크는 이산적이고 순차적이기 때문에 어떤 시점이나 시퀀스가 예측 오차를 초래했는지 정확히 짚어내기가 어렵다. 시퀀스가 길어질수록 초기 스파이크가 이후 예측에 미치는 영향을 추적하기는 더욱 어려워진다. 따라서 TCA 문제를 해결하는 것은 SNN의 순차 모델링(sequential modelling) 능력을 향상시키는 데 핵심적이다.

SNN의 TCA 문제를 해결하기 위해 크게 두 가지 연구 방향이 추구되어 왔다. 첫 번째 방향은 딥러닝(deep learning)에서 어텐션(attention) 모델이 거둔 최근의 성공에서 영감을 얻는다. 이러한 방법들은 자기-어텐션(self-attention) 메커니즘을 SNN에 통합하여 서로 다른 시점 간 시간 관계를 직접 모델링할 수 있도록 한다 (Qin et al. 2023; Yao et al. 2021). 그러나 자기-어텐션은 계산 비용이 크고 실시간(real-time) 동작이 어렵다. 또한 자기-어텐션은 주류 뉴로모픽 칩과 직접적으로 호환되지 않으므로, 해당 칩이 제공하는 에너지 효율성의 이점을 활용할 수 없다.

다른 연구 방향은 주로 적응형(adaptive) 스파이킹 뉴런 모델에 초점을 맞춘다. 대표적으로 장단기 기억 스파이킹 신경망(Long Short-Term Memory Spiking Neural Network, LSNN)이 있다 (Bellec et al. 2018). LSNN은 LIF 뉴런에 적응형 발화 임계값(adaptive firing threshold) 메커니즘을 도입하여, 각 발화 이후 뉴런의 발화 임계값이 증가하고 이후 천천히 휴지 상태(resting-state)의 임계값으로 감쇠하도록 한다. 이러한 임계값 상승은 정보 저장 수단으로 작동하며, 특히 감쇠율(decay rate)이 느릴 때 장기 TCA를 효과적으로 돕는다 (Bellec et al. 2020). 추가 연구에서는 적응형 발화 임계값을 위해 학습 가능한 시정수(learnable time constant) (Yin, Corradi, and Bohte 2020, 2021) 또는 이중 시정수(dual time constant) (Shaban, Bezugam, and Suri 2021)를 사용함으로써 다중 스케일 시간 정보(multi-scale temporal information)를 보존하고 TCA에 활용할 수 있도록 하였다. 그러나 이러한 연구들은 주로 뉴런 발화 임계값 자체를 강화하는 데 집중해 왔고, 발화 임계값은 단순한 신경 구성 요소이므로 정보 저장 용량에 본질적인 한계가 있다. 따라서 이러한 모델들은 TCA 문제를 해결하는 능력에도 내재적 제한을 가진다.

다중-구획(multi-compartment) 뉴런 모델은 신경과학(neuroscience) 분야에서 오랫동안 광범위하게 연구되어 왔다 (Rall 1964; Pinsky and Rinzel 1994). 이러한 모델은 수상돌기의 복잡한 기하 구조와 수상돌기-세포체 구획 간 상호작용을 충실히 모델링하고자 한다. 그 결과, 다중-구획 모델은 생물학적 뉴런에서 관찰되는 복잡한 신경 동역학을 더 정확히 표현할 수 있으며, 다양한 시간 스케일 사이의 정보 상호작용을 촉진한다 (Stuart and Spruston 2015). 따라서 이들은 장기 TCA 문제를 해결하기 위한 유망한 경로를 제공한다. 더 많은 구획을 도입하면 메모리 용량이 확장되는 장점이 있지만, 모델 복잡성이 증가하여 복잡한 패턴 인식 과제를 실제로 해결하는 데 장애가 될 수 있다.

본 논문에서는 그림 1(a)에 나타낸 바와 같이 일반화된 2-구획 뉴런 모델을 유도한다. 이 뉴런 모델은 더 복잡한 다중-구획 모델의 핵심 특징은 유지하면서도 생물학적 뉴런의 최소 기하 구조(minimal geometry)를 이상적으로 반영한다 (Lin et al. 2017). 이를 바탕으로 장기 TCA 문제 해결에 특화된 2-구획 스파이킹 뉴런 모델인 TC-LIF(Two-Compartment Leaky Integrate-and-Fire)를 추가로 제안한다. 본 연구의 주요 기여는 다음과 같다.

- 우리는 장기 순차 모델링을 촉진하도록 정교하게 설계된, 뇌 영감 기반(brain-inspired)의 2-구획 스파이킹 뉴런 모델 TC-LIF를 제안한다.
- 제안한 TC-LIF 모델이 장기 TCA를 성공적으로 달성하는 데 효과적임을 검증하기 위해 이론적 및 실험적 분석을 제공한다.
- 광범위한 시간 분류 과제에 대한 실험 결과를 통해, TC-LIF가 단일-구획 뉴런 모델보다 더 우수한 순차 모델링 능력, 즉 향상된 분류 정확도, 빠른 학습 수렴, 그리고 높은 에너지 효율성을 제공함을 보인다.

그림 1. (a) 2-구획 Pinsky-Rinzel 피라미드 뉴런의 구조와, (b) LIF 모델 및 제안하는 (c) TC-LIF 모델의 내부 연산을 도시하였다. 제안한 TC-LIF 모델의 (c)에서는 파란색과 빨간색으로 강조된 영역을 통해 각각 수상돌기 구획과 세포체 구획을 구분한다. 이에 반해 (b)의 LIF 모델은 수상돌기 구획을 고려하지 않고 세포체 구획의 동역학만을 포함한다.

## 방법론(Methodology)

이 절에서는 먼저 전통적인 단일-구획 뉴런 모델인 LIF 모델을 소개하고, 장기 의존성을 효과적으로 학습하는 데 내재한 한계를 논의한다. 이어서 잘 알려진 Pinsky-Rinzel 피라미드 뉴런(Pinsky-Rinzel pyramidal neuron)에서 영감을 받은 일반화된 2-구획 스파이킹 뉴런 모델을 제시한다. 이는 장기 TCA 문제 해결에 맞추어 정교하게 설계된 제안 모델 TC-LIF를 전개하기 위한 기반을 제공한다. 또한 TC-LIF 모델이 장기 TCA를 효과적으로 촉진하는 메커니즘을 설명하기 위한 이론적 분석을 제시한다.

### LIF 뉴런은 장기 시간적 신용 할당(long-term TCA)을 수행하는 데 어려움을 겪는다

일반적으로 스파이킹 뉴런은 입력 스파이크로부터 변환된 시냅스 입력(synaptic input)을 막전위(membrane potential)에 적분한다. 누적된 막전위가 발화 임계값(firing threshold)을 넘으면 출력 스파이크(output spike)가 생성되어 다음 뉴런으로 전달된다. LIF 뉴런은 가장 널리 사용되면서도 효과적인 단일-구획 스파이킹 뉴런 모델로, 대규모 뇌 시뮬레이션과 뉴로모픽 컴퓨팅에 널리 사용되어 왔다. LIF 뉴런의 신경 동역학은 다음의 이산시간(discrete-time) 식으로 기술할 수 있다.

$$
U[t] = \beta U[t - 1] - V_{th} S[t - 1] + I[t]
$$
(1)

$$
I[t] = \sum_i \omega_i x_i[t] + b
$$
(2)

$$
S[t] = \Theta(U[t] - V_{th})
$$
(3)

여기서 $U[t]$ 와 $I[t]$ 는 각각 시각 $t$ 에서 뉴런의 막전위와 입력 전류(input current)를 나타낸다. $\beta \equiv \exp(-dt/\tau_m)$ 는 막전위 감쇠 계수(membrane decaying coefficient)이며 범위는 $(0, 1)$ 이다. 여기서 $\tau_m$ 은 막 시정수(membrane time constant), $dt$ 는 시뮬레이션 시간 간격(simulation time step)이다. $x_i$ 는 이전 층에서 입력 뉴런 $i$ 가 출력한 스파이크이고, $\omega_i$ 는 입력 뉴런 $i$ 와 연결된 시냅스 가중치(synaptic weight)를 나타내며, $b$ 는 바이어스(bias) 항이다. 식 (3)에 따라 막전위 $U[t]$ 가 뉴런의 발화 임계값 $V_{th}$ 에 도달하면 출력 스파이크가 생성된다.

최근에는 대리 기울기(surrogate gradient)와 결합된 시간 역전파(backpropagation-through-time, BPTT) 알고리즘이 SNN에서 신용 할당(credit assignment)을 수행하는 효과적인 접근으로 제안되었다 (Wu et al. 2018; Neftci, Mostafa, and Zenke 2019). 이 접근은 제한된 시간 맥락을 포함하는 과제에서는 효과적이지만, 장기 시간 의존성이 필요한 과제에서는 한계를 드러낸다. 그 주된 원인은 역전파 과정에서 오차 기울기가 점차 줄어드는 기울기 소실(vanishing gradient) 문제이다. 이를 더 구체적으로 설명하기 위해, 다음 목적 함수(objective function)를 갖는 SNN의 학습을 생각해 보자.

$$
L(\hat{S}, S) = \frac{1}{N} \sum_{n=1}^{N} L(\hat{S}_n, S_n)
$$
(4)

여기서 $N$ 은 학습 샘플 수, $L$ 은 손실 함수(loss function), $S_n$ 은 네트워크 출력, $\hat{S}_n$ 은 학습 목표(training target)이다. BPTT 알고리즘에 따르면 가중치 $\omega$ 에 대한 기울기는 다음과 같이 계산된다.

$$
\frac{\partial L}{\partial \omega}
= \sum_{t}^{T}
\frac{\partial L}{\partial S[T]}
\frac{\partial S[T]}{\partial U[T]}
\frac{\partial U[T]}{\partial U[t]}
\frac{\partial U[t]}{\partial \omega}
$$
(5)

막 감쇠율이 $\beta \in (0, 1)$ 인 LIF 뉴런의 경우,

$$
\frac{\partial U[T]}{\partial U[t]}
=
\prod_{i=t+1}^{T}
\frac{\partial U[i]}{\partial U[i-1]}
=
\beta^{(T-t)}
$$
(6)

이다. 시간 스텝 $T$ 가 증가할수록 시점 $t$ 가 이후 시점에 미치는 영향은 점점 작아진다. 이는 막전위 감쇠가 초기 정보의 지수 감쇠(exponential decay)를 유발하기 때문이다. 특히 $t$ 가 $T$ 보다 훨씬 작을 때, 식 (6)의 값은 0에 가까워지며 기울기 소실 문제가 발생한다. 결과적으로 LIF와 같은 기존 단일-구획 뉴런 모델은 훨씬 이른 시점으로 기울기를 효과적으로 전파하는 데 어려움을 겪는다. 이는 장기 의존성 학습에 중대한 제약이 되며, 더 강한 장기 TCA 능력을 가진 2-구획 뉴런 모델 개발의 동기가 된다.

### 일반화된 2-구획 스파이킹 뉴런(A Generalized Two-Compartment Spiking Neuron)

P-R 피라미드 뉴런(P-R pyramidal neuron)은 해마(hippocampus)의 CA3 영역에 위치하며, 동물의 기억 저장과 인출에 중요한 역할을 한다 (Pinsky and Rinzel 1994). 연구자들은 이 뉴런 모델을 세포체 구획과 수상돌기 구획 간 상호작용을 시뮬레이션할 수 있는 2-구획 모델로 단순화하였으며, 이는 그림 1(a)에 나타나 있다. 우리는 P-R 모델의 구조를 바탕으로 다음과 같이 정의되는 일반화된 2-구획 스파이킹 뉴런 모델을 개발한다. 이 식의 자세한 유도는 보충 자료에 제시한다.

$$
U^{D}[t] = \alpha_1 U^{D}[t - 1] + \beta_1 U^{S}[t - 1] + I[t]
$$
(7)

$$
U^{S}[t] = \alpha_2 U^{S}[t - 1] + \beta_2 U^{D}[t] - V_{th} S[t - 1]
$$
(8)

$$
S[t] = \Theta(U^{S}[t] - V_{th})
$$
(9)

여기서 $U^{D}$ 와 $U^{S}$ 는 각각 수상돌기 구획과 세포체 구획의 막전위를 의미한다. $\alpha_1$ 과 $\alpha_2$ 는 두 구획의 막전위 감쇠 계수이다. 중요한 점은 이 두 구획의 막전위가 서로 독립적으로 갱신되지 않는다는 것이다. 대신 식 (7)과 식 (8)의 두 번째 항을 통해 서로 결합(coupled)되며, 그 결합 효과는 $\beta_1$ 과 $\beta_2$ 계수에 의해 제어된다. 이 두 구획의 상호작용은 뉴런 동역학을 풍부하게 만들고, 적절히 설계될 경우 기울기 소실 문제를 해결할 수 있다.

### TC-LIF 스파이킹 뉴런 모델(TC-LIF Spiking Neuron Model)

앞서 유도한 일반화된 2-구획 스파이킹 뉴런 모델을 바탕으로, 장기 순차 모델링을 촉진하도록 정교하게 설계된 TC-LIF 뉴런 모델을 제안한다. 일반화된 2-구획 뉴런 모델과 비교했을 때, 우리는 두 구획 모두에서 막 감쇠 계수 $\alpha_1$ 과 $\alpha_2$ 를 제거한다. 이 수정은 의도하지 않은 정보 손실을 일으킬 수 있는 급격한 메모리 감쇠를 피하기 위한 것이다. 또한 지속적인 입력 누적(persistent input accumulation)으로 인해 발생하는 과도한 발화(excess firing)를 방지하기 위해, $\beta_1$ 과 $\beta_2$ 가 서로 반대 부호를 갖도록 설정한다. 제안하는 TC-LIF 모델의 동역학은 다음과 같다.

$$
U^{D}[t] = U^{D}[t - 1] + \beta_1 U^{S}[t - 1] + I[t] - \gamma S[t - 1]
$$
(10)

$$
U^{S}[t] = U^{S}[t - 1] + \beta_2 U^{D}[t] - V_{th} S[t - 1]
$$
(11)

$$
S[t] = \Theta(U^{S}[t] - V_{th})
$$
(12)

여기서 계수 $\beta_1 \equiv -\sigma(c_1)$ 및 $\beta_2 \equiv \sigma(c_2)$ 는 두 구획 간 상호작용을 결정한다. 시그모이드(sigmoid) 함수 $\sigma(\cdot)$ 는 두 계수가 각각 $(-1, 0)$ 과 $(0, 1)$ 범위에 있도록 보장하며, 매개변수 $c_1$ 과 $c_2$ 는 학습 과정에서 자동으로 조정될 수 있다. 이 설계 선택의 효과는 곧 자세히 분석한다. 두 구획의 막전위는 모두 세포체 발화 후 재설정(reset)된다. 특히 수상돌기 구획의 재설정은 스케일링 계수(scaling factor) $\gamma$ 에 의해 제어되는 역전파 스파이크(backpropagating spike)에 의해 유발된다. TC-LIF 모델의 내부 연산은 그림 1(c)에 도시되어 있으며, 이는 그림 1(b)의 LIF 모델보다 더 풍부한 내부 동역학을 보여준다.

위 식에 따르면 $U^{S}$ 는 발화 후 재설정되는 수상돌기 입력의 단기 기억(short-term memory)을 담당한다. 반면 $U^{D}$ 는 외부 입력 정보를 유지하는 장기 기억(long-term memory)으로서, 세포체에서 역전파되는 스파이크에 의해 부분적으로만 재설정된다. 이러한 방식으로 다중 스케일 시간 정보가 TC-LIF에 효과적으로 보존된다. 장기 TCA를 촉진하는 데 있어 TC-LIF의 우수성을 더 명확히 보이기 위해, TC-LIF 모델이 기울기 소실 문제를 어떻게 크게 완화하는지 보여주는 이론적 분석을 아래에 제시한다.

앞서 논의했듯이, 기울기 소실 문제의 주된 원인은 $\partial U[T] / \partial U[t]$ 의 재귀적 계산에 있다. 그러나 이 문제는 제안하는 TC-LIF 모델에서 효과적으로 완화될 수 있으며, 이 모델의 편미분(partial derivative) $\partial U[T] / \partial U[t]$ 는 다음과 같이 계산된다.

$$
\frac{\partial \mathbf{U}[T]}{\partial \mathbf{U}[t]}
=
\prod_{j=t+1}^{T}
\frac{\partial \mathbf{U}[j]}{\partial \mathbf{U}[j - 1]},
\quad
\mathbf{U}[j] = [U^{D}[j], U^{S}[j]]^{T}
$$
(13)

여기서

$$
\frac{\partial \mathbf{U}[j]}{\partial \mathbf{U}[j - 1]}
=
\begin{bmatrix}
\frac{\partial U^{D}[j]}{\partial U^{D}[j-1]} & \frac{\partial U^{D}[j]}{\partial U^{S}[j-1]} \\
\frac{\partial U^{S}[j]}{\partial U^{D}[j-1]} & \frac{\partial U^{S}[j]}{\partial U^{S}[j-1]}
\end{bmatrix}
=
\begin{bmatrix}
\beta_1 \beta_2 + 1 & \beta_1 \\
\beta_1 \beta_2^{2} + 2 \beta_2 & \beta_1 \beta_2 + 1
\end{bmatrix}
$$
(14)

이다. TC-LIF에서 기울기 소실 문제의 심각도를 정량화하기 위해, 우리는 아래 식 (15)와 같이 열 무한 노름(column infinite norm)을 추가로 계산한다.

$$
\left\|
\frac{\partial \mathbf{U}[j]}{\partial \mathbf{U}[j - 1]}
\right\|_{\infty}
=
\max \left(
\beta_1 \beta_2^{2} + \beta_1 \beta_2 + 2 \beta_2 + 1,\;
\beta_1 \beta_2 + \beta_1 + 1
\right)
$$
(15)

$$
=
\beta_1 \beta_2^{2} + \beta_1 \beta_2 + 2 \beta_2 + 1
$$

무한 노름은 긴 시간 구간에 걸친 막전위의 최대 변화율(maximum changing rate)을 의미한다. 제약 최적화(constrained optimization) 방법을 사용하여 $\left\| \partial \mathbf{U}[j] / \partial \mathbf{U}[j - 1] \right\|_{\infty}$ 의 하한(lower bound)을 구하면, 이 값이 1보다 큼을 알 수 있다. 이는 TC-LIF 모델이 기울기 소실 문제를 효과적으로 해결할 수 있음을 시사한다. 그러나 무한 노름의 값이 항상 1을 초과하기 때문에, 필연적으로 기울기 폭주(exploding gradient) 문제에 직면하게 된다. 다행히 실험 결과는 두 번째 사분면(second quadrant)에서 선택한 대부분의 $\{\beta_1, \beta_2\}$ 에 대해 이 값이 1보다 약간 큰 수준에 머무르며, 그 결과 안정적인 학습이 가능함을 보여준다. 자세한 분석은 보충 자료를 참조하라.

또한 TC-LIF 모델은 다음과 같이 단일-구획 형태(single-compartment form)로 재구성할 수 있다는 점에 주목할 필요가 있다.

$$
U^{S}[t]
=
(1 + \beta_1 \beta_2) U^{S}[t - 1]
+ \beta_2 U^{D}[t - 1]
+ \beta_2 I[t]
- (\beta_2 \gamma + V_{th}) S[t - 1]
$$
(16)

본질적으로 위 식은 감쇠하는 입력(decaying input)을 갖는 LIF 뉴런을 닮아 있다. 따라서 제안한 모델을 TC-LIF라고 부르는 것이 적절하다. TC-LIF 모델에서도 메모리 감쇠 문제 자체는 여전히 존재하지만, $U^{D}$ 의 존재가 메모리 손실을 효과적으로 보상하여 기울기 소실 문제를 해결할 수 있다.

## 실험(Experiments)

이 절에서는 먼저 일반화된 2-구획 뉴런의 매개변수 공간(parameter space)을 탐색하여 TC-LIF 설계의 타당성을 검증한다. 이어서 순차형 MNIST(sequential MNIST, S-MNIST), 순열된 순차형 MNIST(permuted sequential MNIST, PS-MNIST), Google Speech Commands(GSC), Spiking Heidelberg Digits(SHD), Spiking Google Speech Commands(SSC)를 포함한 다양한 시간 분류 벤치마크(temporal classification benchmark)에서 TC-LIF 모델을 평가한다. 또한 탁월한 시간 분류 능력, 효과적인 장기 TCA, 빠른 학습 수렴, 높은 에너지 효율성 측면에서 TC-LIF 모델의 우수성을 보이기 위한 포괄적 연구를 수행한다. TC-LIF 모델의 학습에는 대리 기울기와 함께 BPTT를 사용하였다 (Neftci, Mostafa, and Zenke 2019). 실험 설정과 학습 세부사항은 보충 자료에 더 자세히 제시한다.

### 일반화된 2-구획 뉴런의 매개변수 공간 탐색(Parameter Space Exploration for Generalized Two-Compartment Neurons)

P-R 모델을 바탕으로 우리는 $\{\alpha_1, \alpha_2, \beta_1, \beta_2\}$ 로 지배되는 신경 동역학을 갖는 일반화된 2-구획 뉴런 모델을 제안하였다. 앞서 논의했듯이, 막전위에 저장된 메모리의 급격한 감쇠를 완화하기 위해 TC-LIF 모델에서는 $\alpha_1$ 과 $\alpha_2$ 를 모두 1로 설정하였다. 그러나 $\beta_1$ 과 $\beta_2$ 의 초기화(initialization)는 2-구획 뉴런 모델의 학습 수렴에 상당한 영향을 미친다. TC-LIF에서 제안한 매개변수 설정의 효과를 확인하기 위해, 우리는 $\beta_1$ 과 $\beta_2$ 를 네 개의 서로 다른 사분면에 걸쳐 초기화하는 그리드 탐색(grid search)을 수행하고, S-MNIST와 PS-MNIST 데이터셋에서 그 성능을 평가하였다.

그림 2에서 진한 파란색 선은 인접한 시간 스텝 사이의 막전위 편미분 $\partial \mathbf{U}[j] / \partial \mathbf{U}[j - 1]$ 이 1이 되는 위치를 나타낸다. 그 결과, 전체 $\beta$ 공간은 두 영역으로 나뉘며, 세 번째 사분면(third quadrant)은 편미분이 1보다 작은 영역을 나타낸다. 반대로 나머지 세 개의 사분면은 편미분이 1보다 큰 영역을 나타낸다. 각 사분면 안에서는 $\beta_1$ 과 $\beta_2$ 값을 균등 간격으로 설정한 아홉 개의 모델을 평가하였고, 각 점 옆의 숫자는 해당 데이터셋에서의 테스트 정확도(test accuracy)를 의미한다.

결과적으로 두 번째 사분면에서 초기화된 모델을 제외하면, 다른 영역의 모델들은 수렴에 어려움을 겪었다. 특히 $\beta$ 를 세 번째 사분면에서 초기화하면 편미분이 1보다 작아지는 경우에 해당하며, 이는 기울기 소실 문제를 초래한다. 반대로 첫 번째 사분면에서 $\beta$ 를 초기화한 모델은 심각한 기울기 폭주 문제를 겪는다. 기울기 소실과 폭주 모두 네트워크 수렴을 방해한다. 네 번째 사분면에서 $\beta$ 를 초기화하면 이러한 문제를 완화할 수는 있지만, 세포체 구획에 지속적인 음의 입력(식 (11) 참조)을 유발하므로 두 과제 모두에서 좋지 않은 시간 분류 성능을 보인다. 따라서 TC-LIF 모델에서는 $\beta_1 \in (-1, 0)$, $\beta_2 \in (0, 1)$, 즉 두 번째 사분면의 값을 사용하여 초기화하였고, 이후의 모든 실험에서도 이를 일관되게 사용하였다.

그림 2. $\beta_1$ 및 $\beta_2$ 초기화가 S-MNIST와 PS-MNIST 데이터셋의 테스트 정확도에 미치는 영향을 나타낸다. S-MNIST와 PS-MNIST 모두에서, 첫 번째 및 세 번째 사분면에서 초기화된 모델은 각각 심각한 기울기 폭주와 기울기 소실 문제를 겪는다. 그 결과 의미 있는 정보를 전혀 학습하지 못하며 정확도 11.35%에 머무른다. 초록색 점은 학습 정확도 100%까지 수렴할 수 있는 모델을 의미한다.

그림 3. S-MNIST 데이터셋에서 시간에 따른 기울기 변화(gradient evolution)를 나타낸다. 기울기는 3층 피드포워드 SNN(64-256-256)을 사용하여 무작위로 선택한 256개 샘플 배치(batch)에서 계산하였다.

### 우수한 시간 분류 능력(Superior Temporal Classification Capability)

표 1은 다섯 가지 널리 사용되는 시간 분류 데이터셋에서 제안한 TC-LIF 모델의 결과와 기존 연구 결과를 함께 제시한다. 전반적으로 TC-LIF 모델은 비슷한 수의 매개변수를 갖는 최신 단일-구획 뉴런 모델을 일관되게 능가한다.

표 1. S-MNIST, PS-MNIST, GSC, SHD, SSC 데이터셋에서의 모델 성능 비교. 여기서 'FF' 와 'Rec' 는 각각 피드포워드(feedforward) 및 순환(recurrent) 네트워크를 의미한다. * 는 공개된 코드를 사용해 우리가 재현한 결과를 뜻한다.

#### S-MNIST

| 방법(Method) | SNN | 네트워크 | 매개변수(K) | 정확도(%) |
| --- | --- | --- | --- | --- |
| GLIF* (Yao et al. 2022) | Y | FF | 47.1/87.5 | 94.80/95.27 |
| PLIF* (Fang et al. 2021) | Y | FF | 44.8/85.1 | 83.71/87.92 |
| LIF* | Y | FF | 44.8/85.1 | 62.42/72.06 |
| LTMD* (Wang, Cheng, and Lim 2022) | Y | FF | - /85.1 | - /68.56 |
| TC-LIF (ours) | Y | FF | 44.8/85.1 | 96.46/97.35 |
| LSTM (Arjovsky, Shah, and Bengio 2016) | N | Rec | 66.5/ - | 98.20/ - |
| SRNN+ReLU (Yin, Corradi, and Bohte 2020) | N | Rec | 129.6/ - | 98.99/ - |
| LSNN (Bellec et al. 2018) | Y | Rec | 68.2/ - | 93.70/ - |
| AHP (Rao et al. 2022) | Y | Rec | 68.4/ - | 96.00/ - |
| GLIF* (Yao et al. 2022) | Y | Rec | 114.6/157.5 | 95.63/96.64 |
| SRNN+ALIF (Yin, Corradi, and Bohte 2020, 2021) | Y | Rec | 129.6/156.3 | 97.82/98.70 |
| PLIF* (Fang et al. 2021) | Y | Rec | 112.2/155.1 | 90.93/91.79 |
| LIF* | Y | Rec | 112.2/155.1 | 74.91/89.28 |
| LTMD* (Wang, Cheng, and Lim 2022) | Y | Rec | - /155.1 | - /84.62 |
| TC-LIF (ours) | Y | Rec | 63.6/155.1 | 98.79/99.20 |

#### PS-MNIST

| 방법(Method) | SNN | 네트워크 | 매개변수(K) | 정확도(%) |
| --- | --- | --- | --- | --- |
| LIF* | Y | FF | 44.8/85.1 | 11.30/10.00 |
| TC-LIF (ours) | Y | FF | 44.8/85.1 | 80.89/83.98 |
| LSTM (Arjovsky, Shah, and Bengio 2016) | N | Rec | 66.5/ - | 88.00/ - |
| SRNN+ReLU (Yin, Corradi, and Bohte 2020) | N | Rec | 129.6/ - | 93.47/ - |
| GLIF* (Yao et al. 2022) | Y | Rec | 114.6/157.5 | 90.34/90.47 |
| SRNN+ALIF (Yin, Corradi, and Bohte 2020, 2021) | Y | Rec | 129.6/156.3 | 91.00/94.30 |
| LIF* | Y | Rec | 112.2/155.1 | 71.77/80.39 |
| LTMD* (Wang, Cheng, and Lim 2022) | Y | Rec | - /155.1 | - /54.93 |
| TC-LIF (ours) | Y | Rec | 63.6/155.1 | 92.69/95.36 |

#### GSC

| 방법(Method) | SNN | 네트워크 | 매개변수(K) | 정확도(%) |
| --- | --- | --- | --- | --- |
| Rate-based SNN (Yılmaz et al. 2020) | Y | FF | 117 | 75.20 |
| TC-LIF (ours) | Y | FF | 106.2 | 91.35 |
| SRNN+ALIF (Yin, Corradi, and Bohte 2021) | Y | Rec | 221.7 | 92.10 |
| SNN (Salaj et al. 2021) | Y | Rec | 4304.9 | 89.04 |
| SNN with SFA (Salaj et al. 2021) | Y | Rec | 4307 | 91.21 |
| TC-LIF (ours) | Y | Rec | 196.5 | 94.84 |

#### SHD

| 방법(Method) | SNN | 네트워크 | 매개변수(K) | 정확도(%) |
| --- | --- | --- | --- | --- |
| Feed-forward SNN (Cramer et al. 2020) | Y | FF | 108.8 | 48.60 |
| TC-LIF (ours) | Y | FF | 108.8 | 83.08 |
| SRNN (Cramer et al. 2020) | Y | Rec | 108.8 | 71.4 |
| Heterogeneous SRNN (Perez-Nieves et al. 2021) | Y | Rec | 108.8 | 82.70 |
| Attention (Yao et al. 2021) | Y | Rec | 133.8 | 81.45 |
| SRNN + ALIF (Yin, Corradi, and Bohte 2020) | Y | Rec | 142.4 | 84.40 |
| SRNN (Zenke and Vogels 2021) | Y | Rec | 249 | 82.00 |
| SRNN + data augm. (Cramer et al. 2020) | Y | Rec | 1787.9 | 83.20 |
| TC-LIF (ours) | Y | Rec | 141.8 | 88.91 |

#### SSC

| 방법(Method) | SNN | 네트워크 | 매개변수(K) | 정확도(%) |
| --- | --- | --- | --- | --- |
| Feed-forward SNN (Cramer et al. 2020) | Y | FF | 110.8 | 38.50 |
| TC-LIF (ours) | Y | FF | 110.8 | 63.46 |
| SRNN (Cramer et al. 2020) | Y | Rec | 110.8 | 50.90 |
| Heterogeneous SRNN (Perez-Nieves et al. 2021) | Y | Rec | 110.8 | 57.3 |
| TC-LIF (ours) | Y | Rec | 110.8 | 61.09 |

S-MNIST 데이터셋에서 각 샘플은 길이 784의 시퀀스를 가지므로, 모델은 장기 의존성을 학습해야 한다. 예상한 대로 LIF 모델은 이 데이터셋에서 가장 낮은 성능을 보였는데, 이는 앞서 논의한 기울기 소실 문제로 설명될 수 있다. 주목할 점은 최근 제안된 적응형 스파이킹 뉴런 모델인 LSNN (Bellec et al. 2018)과 adaptive LIF(ALIF) (Yin, Corradi, and Bohte 2020, 2021)가 LSTM 모델 (Arjovsky, Shah, and Bengio 2016)과 비슷하거나 더 나은 정확도를 달성했다는 점이다. 제안한 TC-LIF 모델은 이러한 단일-구획 뉴런 모델보다 일관되게 우수한 성능을 보였으며, 이는 다중 스케일 시간 정보를 보존하고 장기 의존성을 처리하는 데 효과적임을 시사한다. 특히 순환 구조(recurrent architecture)에서 99.20% 정확도를 달성하였으며, 우리가 아는 한 이는 해당 데이터셋에서 보고된 SNN 모델 중 최고 성능이다. 보다 어려운 PS-MNIST 데이터셋에서도 같은 결론을 내릴 수 있다.

이미지 데이터셋뿐 아니라, 우리는 더 풍부한 시간 동역학을 갖는 음성 데이터셋(speech dataset)에 대해서도 추가 실험을 수행하였다. 비-스파이킹(non-spiking) GSC 데이터셋에서 TC-LIF 모델은 피드포워드 및 순환 네트워크 각각에 대해 91.35%와 94.84% 정확도를 달성하여, 최신 모델들을 큰 폭으로 능가하였다. SHD와 SSC 데이터셋은 SNN 벤치마킹을 위해 특별히 설계된 뉴로모픽 데이터셋이다. 이 데이터셋들에서도 제안하는 TC-LIF는 기존에 보고된 모든 방법을 크게 능가하는 향상을 보였다.

### 효과적인 장기 시간적 신용 할당(Effective Long-term Temporal Credit Assignment)

TC-LIF 뉴런에서 장기 시간 관계(long-term temporal relationship)가 어떻게 형성되는지 더 잘 이해하기 위해, 우리는 S-MNIST 데이터셋에서 계산된 오차 기울기(error gradient)를 시각화하였다. 시각적 명료성을 높이기 위해, 시각 $t$ 에서 뉴런 $n$ 의 기울기 값 $G_t^n$ 를 $G_t^n / \sum_{i=0}^{N}\sum_{j=0}^{T} G_j^i$ 로 정규화하였다.

그림 3에 나타난 바와 같이, TC-LIF 뉴런은 LIF 및 ALIF 모델에 비해 더 이른 시점으로 더 많은 기울기를 효과적으로 전달할 수 있다. 이는 첫 번째 및 두 번째 층(뉴런 인덱스 0–319)에서 더욱 분명하다. 이러한 결과는 TC-LIF가 장기 TCA를 수행하는 데 탁월한 능력을 가짐을 시사한다.

### 빠른 학습 수렴(Rapid Training Convergence)

장기 TCA를 수행하는 탁월한 능력 덕분에, 제안하는 TC-LIF 모델은 더 안정적인 학습과 더 빠른 네트워크 수렴을 보장한다. 이를 보여주기 위해, 우리는 동일한 학습 조건에서 TC-LIF 모델의 학습 곡선(learning curve)을 LIF, GLIF, PLIF 모델과 비교하였다. 그림 4에서 실선은 평균 정확도(mean accuracy)를 나타내며, 음영 영역은 서로 다른 랜덤 시드(random seed)를 사용한 네 번의 실행(run)에서 얻은 정확도 표준편차(standard deviation)를 나타낸다. 특히 TC-LIF 모델은 두 네트워크 구조 모두에서 약 25 에폭(epoch) 내에 빠르게 수렴하는 반면, LIF 모델은 피드포워드와 순환 네트워크에서 각각 약 100 에폭과 75 에폭이 필요하다. 또한 TC-LIF 모델은 특히 학습 초기 단계에서 다른 모델보다 더 높은 안정성을 보인다.

또한 TC-LIF 모델이 왜 더 안정적인 학습과 더 빠른 수렴을 달성할 수 있는지를 조사하기 위해, 발견된 국소 최소점(local minima) 주변에서 LIF와 TC-LIF의 손실 지형(loss landscape)을 추가로 비교하였다. 그림 5에서 볼 수 있듯이, TC-LIF 모델은 국소 최소점 주변에서 현저히 더 매끄러운 손실 지형을 보인다. 이는 TC-LIF 모델이 향상된 학습 동역학과 수렴 특성을 제공함을 시사한다. 특히 더 매끄러운 손실 지형은 국소 최소점에 갇힐 가능성을 줄여, 더 안정적인 최적화와 더 빠른 수렴으로 이어질 수 있다. 또한 이러한 손실 지형은 과적합(overfitting) 및 과소적합(underfitting) 문제에 덜 취약하므로, 더 강한 일반화(generalization) 능력을 시사한다.

그림 4. (a) 피드포워드와 (b) 순환 네트워크 구조에서 TC-LIF와 다른 단일-구획 스파이킹 뉴런의 학습 곡선을 비교하였다. 평균과 표준편차는 네 번의 실행 결과를 바탕으로 보고하였다.

그림 5. (a, c) LIF 및 (b, d) TC-LIF 뉴런 모델의 손실 지형을 3차원 표면(3D surface)과 2차원 등고선(2D contour) 플롯으로 비교하였다.

### 높은 에너지 효율성(High Energy Efficiency)

지금까지는 제안한 TC-LIF 모델이 모델 복잡도와 계산 효율성 사이에서 적절한 절충(trade-off)을 달성하는지 분명하지 않았다. 이를 확인하기 위해 우리는 LIF, TC-LIF, LSTM 모델의 에너지 효율성을 분석하고 비교하였다. 구체적으로는 입력 데이터 처리와 네트워크 갱신 중에 소비되는 누산(accumulated, AC) 및 곱셈-누산(multiply-and-accumulate, MAC) 연산 수를 계산하였다. 인공신경망(artificial neural network, ANN)에서는 모든 계산이 MAC 연산으로 수행되는 반면, SNN에서는 시냅스 갱신(synaptic update)에 주로 AC 연산이 사용된다. 또한 스파이킹 뉴런의 막전위 갱신에는 여러 MAC 연산이 필요하다는 점에 주목할 필요가 있다. 더 자세한 계산은 보충 자료에 제시한다.

표 2. LIF, LSTM, TC-LIF 모델의 이론적 및 경험적 에너지 비용 비교. 여기서 $m$ 과 $n$ 은 입력 및 출력 뉴런 수이다. $F_{rin}$ 과 $F_{rout}$ 는 입력 및 출력 뉴런의 발화율(firing rate)이다. $E_{AC}$ 와 $E_{MAC}$ 는 각각 AC 및 MAC 연산의 에너지 비용이다.

| 뉴런 모델 | 이론적 에너지 비용 | 경험적 에너지 비용(nJ) |
| --- | --- | --- |
| LSTM | $4(mn + nn)E_{MAC} + 17nE_{MAC}$ | 2,834.7 |
| LIF | $mnF_{rin}E_{AC} + (nn + n)F_{rout}E_{AC} + nE_{MAC}$ | 23.8 |
| TC-LIF | $mnF_{rin}E_{AC} + (nn + 2n)F_{rout}E_{AC} + 2nE_{MAC}$ | 28.2 |

표 2의 이론적 결과에서 보듯이, 두 스파이킹 뉴런(LIF와 TC-LIF)의 에너지 비용은 계산 복잡도가 낮기 때문에 LSTM 모델보다 훨씬 작다. LIF 모델과 비교할 때, 제안하는 TC-LIF 모델은 수상돌기 구획의 추가 연산 때문에 $nF_{rout}E_{AC} + nE_{MAC}$ 만큼의 연산이 더 필요하다. 경험적 에너지 비용(empirical energy cost)을 계산하기 위해, 우리는 S-MNIST 데이터셋의 테스트 샘플에서 무작위로 선택한 한 배치에 대해 추론(inference)을 수행하고, 이들 SNN의 층별 평균 발화율(layer-wise firing rate)을 계산하였다. LIF와 TC-LIF 모델의 층별 발화율은 각각 $[0.219, 0.145, 0.004]$ 와 $[0.294, 0.146, 0.030]$ 으로 비교적 비슷하다. 총 에너지 비용을 계산하기 위해, 우리는 45nm CMOS 공정에 대한 추정값인 $E_{AC} = 0.9\ \mathrm{pJ}$ 와 $E_{MAC} = 4.6\ \mathrm{pJ}$ 를 사용하였다 (Horowitz 2014). 제안하는 TC-LIF 모델은 내부 구조가 더 복잡함에도 불구하고 LIF 모델과 비슷한 에너지 비용을 보인다. 특히 TC-LIF 모델은 더 우수한 시간 분류 성능을 보이면서도 LSTM 모델 대비 100배 이상의 에너지 절감을 달성하였다.

## 결론(Conclusion)

본 논문에서는 생물학적 뉴런의 다중-구획 구조에서 영감을 받아, 스파이킹 뉴런의 장기 순차 모델링 능력을 향상시키기 위한 새로운 TC-LIF 뉴런을 제안하였다. 제안한 TC-LIF 모델의 수상돌기 구획과 세포체 구획은 상호 시너지적으로 작용하여 뉴런 동역학을 풍부하게 만들고, 적절하게 구성될 경우 TCA 문제를 효과적으로 해결한다. 다양한 시간 분류 과제에 대한 이론적 분석과 실험 결과는 우수한 시간 분류 능력, 효과적인 장기 TCA, 빠른 학습 수렴, 높은 에너지 효율성을 포함하여 제안한 TC-LIF 모델의 우수성을 보여준다. 따라서 본 연구는 순차 모델링 과제를 해결하기 위한 보다 효과적이고 효율적인 스파이킹 뉴런 개발에 기여한다. 본 연구에서는 2-구획 뉴런 모델에 초점을 맞추었으며, 이를 더 많은 구획으로 일반화하는 방법은 향후 탐구할 흥미로운 과제로 남아 있다.

## 감사의 말(Acknowledgments)

본 연구는 부분적으로 홍콩 특별행정구(Hong Kong SAR) 연구보조금위원회(Research Grants Council)의 지원(Grant No. PolyU11211521, PolyU15218622, PolyU25216423), 홍콩이공대학교(The Hong Kong Polytechnic University)의 지원(Project IDs: P0039734, P0035379, P0043563, P0046094), 중국 국가자연과학기금(National Natural Science Foundation of China)의 지원(Grant No. U21A20512, 62306259, 62271432), A*STAR, SOITEC, NXP, 싱가포르국립대학교(National University of Singapore)의 FD-fAbrICS: Joint Lab for FD-SOI Always-on Intelligent Connected Systems 지원(Award I2001E0053), 과학기술연구청(Agency for Science, Technology and Research, A*STAR)의 AME Programmatic Funding Scheme 지원(Project No. A18A2b0046), 그리고 A*STAR의 RIE 2020 Advanced Manufacturing and Engineering Human(AME) Programmatic Grant 지원(Grant No. A1687b0033)을 받았다.

## 보충 자료(Supplementary Materials)

### Pinsky-Rinzel 뉴런 모델(Pinsky-Rinzel Neuron Model)

이 절에서는 2-구획 Pinsky-Rinzel(P-R) 피라미드 뉴런 모델로부터 유도된 일반화된 2-구획 스파이킹 뉴런 모델을 제시한다. 2-구획 P-R 뉴런 모델은 CA3 피라미드 세포 내의 복잡한 버스팅(bursting)을 뒷받침하는 정교한 생물물리학적 메커니즘을 설명하면서도 경량 계산(lightweight computation)이 가능하도록 설계되었다. 이 뉴런의 동역학은 다음 식과 같이 연속시간(continuous time) 형태로 표현될 수 있다.

$$
C_m \frac{dV_s}{dt} = -I_{Na} - I_{K} - I_{Leak} + \frac{I_{link}}{P} + I_s
$$
(17)

$$
C_m \frac{dV_d}{dt} = -I_{NaP} - I_{KS} - I_{Leak} - \frac{I_{link}}{1 - P} + I_d
$$
(18)

여기서 $V_s$ 와 $V_d$ 는 각각 세포체 구획과 수상돌기 구획의 막전위이다. $I_{Na}$ 와 $I_{K}$ 는 세포체 구획의 관련 전류를 나타내며, 수상돌기 구획은 느린 칼륨 전류(slow potassium current) $I_{KS}$ 와 지속성 나트륨 전류(persistent sodium current) $I_{NaP}$ 를 포함한다. 세포체와 수상돌기에 입력되는 전류는 각각 $I_s$ 와 $I_d$ 로 표기한다. 특히 본 논문에서는 $I_s = 0$ 으로 가정하고, 입력 전류는 오직 수상돌기 구획에만 주입된다. 또한 막 정전용량(membrane capacitance)과 세포 면적 비율(proportion of cell area)은 각각 $C_m$ 과 $P$ 로 나타낸다.

표 3은 위 식에 등장하는 이온 전류(ionic current)의 상세 계산을 요약한다. 구체적으로 $E_{Na}$, $E_{K}$, $E_{L}$ 은 평형 전위(equilibrium potential)를 의미하며, $g_{Na}$, $g_{K}$, $g_{L}$, $g_c$, $g_{NaP}$, $g_{KS}$ 는 전도도(conductance)를 의미한다.

표 3. 2-구획 P-R 뉴런 모델에서의 이온 전류 계산 요약.

| 이온 전류(Ionic Current) | 계산식(Calculation) |
| --- | --- |
| $I_{Na}$ | $g_{Na} m^{3} h \cdot (V_s - E_{Na})$ |
| $I_{K}$ | $g_{K} n^{4} \cdot (V_s - E_{K})$ |
| $I_{Leak}$ | $g_{L} \cdot (V - E_{L})$ |
| $I_{link}$ | $g_c \cdot (V_d - V_s)$ |
| $I_{NaP}$ | $g_{NaP} l^{3} h \cdot (V_d - E_{Na})$ |
| $I_{KS}$ | $g_{KS} q \cdot (V_d - E_{K})$ |

식 (17)과 식 (18)은 오일러 방법(Euler method)을 사용하여 다음과 같은 이산시간 식으로 얻어진다.

$$
V_s[t + 1]
=
V_s[t]
+
\frac{dt}{C_m}
\left(
-I_{Na}[t]
- I_{K}[t]
- I_{Leak}[t]
+
\frac{I_{link}[t]}{P}
\right)
$$
(19)

$$
V_d[t + 1]
=
V_d[t]
+
\frac{dt}{C_m}
\left(
-I_{NaP}[t]
- I_{KS}[t]
- I_{Leak}[t]
+
\frac{I_{link}[t]}{1 - P}
+ I_d[t]
\right)
$$
(20)

$I_{link}$ 항은 세포체 구획과 수상돌기 구획 사이의 상호작용을 의미한다. 또한 표 3의 이온 전류 식을 식 (19)와 식 (20)에 대입하고, 세포체 출력 막전위에 대한 스파이킹 연산을 통합하면, 식 (7)–(9)에 제시한 일반화된 2-구획 스파이킹 뉴런 모델의 전체 동역학을 얻을 수 있다.

### 에너지 효율성 분석(Energy Efficiency Analysis)

우리는 뉴런 갱신 함수(neuronal update function)를 바탕으로 LSTM, LIF, TC-LIF 순환 네트워크의 이론적 에너지 비용을 분석한다. 표 4는 각 모델의 이론적 에너지 비용 계산을 자세히 제시한다.

표 4. LIF, TC-LIF, LSTM의 에너지 비용 계산.

| 뉴런 모델 | 동역학(Dynamics) | 단계 비용(Step Cost) | 총 비용(Total Cost) |
| --- | --- | --- | --- |
| LIF | $I_t = W_{m,n} X_m + W_{n,n} S^{n}_{t-1}$ | $(mnF_{rin} + nnF_{rout})E_{AC}$ | $mnF_{rin}E_{AC} + (nn + n)F_{rout}E_{AC}$ |
| LIF | $U_t = \beta U_{t-1} + I_t - V_{th} S^{n}_{t-1}$ | $nF_{rout}E_{AC} + nE_{MAC}$ | $+ nE_{MAC}$ |
| TC-LIF | $I_t = W_{m,n} X_m + W_{n,n} S^{n}_{t-1}$ | $(mnF_{rin} + nnF_{rout})E_{AC}$ | $mnF_{rin}E_{AC}$ |
| TC-LIF | $U^{D}_{t} = U^{D}_{t-1} + I_t + \beta_1 U^{S}_{t-1} - \gamma S^{n}_{t-1}$ | $nF_{rout}E_{AC} + nE_{MAC}$ | $+ (nn + 2n)F_{rout}E_{AC}$ |
| TC-LIF | $U^{S}_{t} = U^{S}_{t-1} + \beta_2 U^{D}_{t} - V_{th} S^{n}_{t-1}$ | $nF_{rout}E_{AC} + nE_{MAC}$ | $+ 2nE_{MAC}$ |
| LSTM | $f_t = \sigma_g(W_f x_t + U_f h_{t-1} + b_f)$ | $n(m + n + 2)E_{MAC}$ | $4(mn + nn)E_{MAC}$ |
| LSTM | $i_t = \sigma_g(W_i x_t + U_i h_{t-1} + b_i)$ | $n(m + n + 2)E_{MAC}$ | $17nE_{MAC}$ |
| LSTM | $o_t = \sigma_g(W_o x_t + U_o h_{t-1} + b_o)$ | $n(m + n + 2)E_{MAC}$ |  |
| LSTM | $\hat{c}_t = \sigma_c(W_c x_t + U_c h_{t-1} + b_c)$ | $n(m + n + 4)E_{MAC}$ |  |
| LSTM | $c_t = f_t \odot c_{t-1} + i_t \odot \hat{c}_t$ | $2nE_{MAC}$ |  |
| LSTM | $h_t = o_t \odot \sigma_h(c_t)$ | $5nE_{MAC}$ |  |

### 실험 세부사항(Experimental Details)

#### 데이터셋(Datasets)

이 절에서는 본 연구에 사용한 데이터셋을 소개한다. 이 데이터셋들은 광범위한 과제를 포괄하므로, 서로 다른 유형의 입력 데이터를 처리하는 모델의 능력을 평가할 수 있게 해 준다.

S-MNIST: Sequential-MNIST(S-MNIST) 데이터셋은 원래의 MNIST 데이터셋에서 파생되었으며, 28 × 28 해상도의 손글씨 숫자(grayscale handwritten digit) 이미지 60,000개(학습용)와 10,000개(테스트용)로 구성된다. S-MNIST에서는 각 이미지를 784개 시간 스텝을 갖는 벡터(vector)로 변환하며, 각 픽셀(pixel)이 특정 시점의 하나의 입력값을 나타낸다. 이 데이터셋은 순차적 이미지 분류(sequential image classification) 과제를 해결하는 모델 성능을 평가하는 데 사용된다.

PS-MNIST: Permuted Sequential MNIST(PS-MNIST) 데이터셋은 Sequential MNIST의 변형으로, 각 이미지의 픽셀을 고정된 무작위 순열(random permutation)에 따라 뒤섞는다. 이 데이터셋은 입력 시퀀스가 더 이상 원래 이미지의 공간적 순서(spatial order)를 따르지 않기 때문에 S-MNIST보다 더 어려운 과제이다. 따라서 이 데이터셋을 학습할 때 모델은 픽셀 사이의 복잡하고 비국소적이며 장기적인 의존성을 포착해야 한다.

GSC: Google Speech Commands(GSC)는 두 가지 버전이 있으며, 본 연구에서는 2번째 버전을 사용한다. GSC version 2는 'yes', 'no', 'up', 'down', 'left', 'right' 등 35가지 음성 명령(spoken command)에 대한 길이 1초(one-second-long)의 오디오 클립(audio clip) 105,829개로 구성된다. 이 오디오 클립들은 서로 다른 화자(speaker)가 다양한 환경에서 녹음한 것이므로, 모델 성능을 평가하기 위한 다양한 데이터 분포를 제공한다.

SHD: Spiking Heidelberg Digits 데이터셋은 스파이크 기반 시퀀스 분류(spike-based sequence classification) 벤치마크로, 영어와 독일어로 발화된 숫자 0부터 9까지(총 20개 클래스)를 포함한다. 이 데이터셋은 12명의 서로 다른 화자 녹음으로 구성되며, 그중 두 명은 테스트 세트에만 등장한다. 각 원시 파형(original waveform)은 700개 입력 채널(input channel)에 걸친 스파이크 열(spike train)로 변환되었다. 학습 세트는 8,332개 예제, 테스트 세트는 2,088개 예제로 구성되며, 별도의 검증 세트(validation set)는 없다. SHD는 스파이킹 형식으로 표현된 음성 데이터를 처리하고 분류하는 제안 모델의 성능을 평가할 수 있게 한다.

SSC: Spiking Speech Command 데이터셋은 또 다른 스파이크 기반 시퀀스 분류 벤치마크로, Google Speech Commands version 2 데이터셋으로부터 파생되었으며 많은 화자가 발화한 35개 클래스를 포함한다. 원래 파형은 700개 입력 채널에 걸친 스파이크 열로 변환되었다. 이 데이터셋은 학습, 검증, 테스트 분할로 나뉘며, 각각 75,466개, 9,981개, 20,382개의 예제를 포함한다. SSC는 스파이킹 데이터로 표현된 음성 명령을 처리하고 인식하는 제안 모델의 성능을 평가할 수 있게 한다.

#### 대리 기울기(surrogate gradient)를 이용한 학습(Training with Surrogate Gradient)

SNN 학습은 식 (3), (9), (12)의 $\Theta(x)$ 로 표시되는 스파이크 함수(spike function)의 비미분가능성(non-differentiability) 때문에 어려움을 가진다. 이러한 특성은 특히 역전파와 같은 널리 사용되는 기울기 기반 최적화 방법의 적용을 방해한다. 대리 기울기 접근법은 스파이크 함수의 기울기를 근사하는 대리 기울기(proxy gradient)를 도입함으로써 이 문제를 해결한다. 즉, $\Theta'(x) \approx \theta'(x)$ 로 표현된다. 실제 기울기는 대부분 0이지만, 대리 기울기는 관심 영역에서 0이 아닌 값을 근사한다. 그 결과 대리 기울기가 네트워크 가중치 갱신에 필요한 피드백을 제공하여 역전파 적용이 가능해진다.

본 연구에서는 SNN의 기울기 기반 학습을 가능하게 하기 위해 삼각 함수(triangle function)를 $\theta'(x)$ 로 사용하였다.

$$
\frac{\partial S[t]}{\partial U[t]}
=
\theta'(U[t] - V_{th})
=
\max(1 - |U[t] - V_{th}|, 0)
$$
(21)

여기서 $U[t]$ 는 단일-구획 스파이킹 뉴런의 경우 식 (3)에 나타난 막전위를 의미하며, 2-구획 스파이킹 뉴런의 경우 식 (9)와 식 (12)에 나타난 세포체 막전위를 의미한다.

#### 네트워크 구조(Network Architecture)

우리는 피드포워드 연결 구성과 순환 연결 구성을 모두 사용하여 실험을 수행하였다. 기존 연구와 공정하게 비교하기 위해, 비슷한 수의 매개변수를 갖는 네트워크 구조를 사용하였다. 이 구조들과 해당 매개변수 수는 표 6에 요약되어 있다.

#### TC-LIF 모델 하이퍼파라미터(TC-LIF Model Hyper-parameters)

표 5에는 TC-LIF 뉴런 모델에 대한 구체적인 하이퍼파라미터 설정(hyper-parameter setting)을 정리하였다. 여기에는 수상돌기 재설정 스칼라(dendritic reset scalar) $\gamma$, 스파이크 임계값(spike threshold) $V_{th}$, 그리고 $\beta_1$, $\beta_2$ 의 초기값이 포함된다.

표 5. TC-LIF를 위한 네트워크 하이퍼파라미터.

| 데이터셋 | 네트워크 | $\gamma$ | $\beta_1, \beta_2$ | $V_{th}$ |
| --- | --- | --- | --- | --- |
| S-MNIST | feedforward | 0.5 | (-0.5, 0.5) | 1.0 |
| S-MNIST | recurrent | 0.5 | (-0.8, 0.4) | 1.0 |
| PS-MNIST | feedforward | 0.7 | (-0.5, 0.5) | 1.5 |
| PS-MNIST | recurrent | 0.5 | (-0.2, 0.8) | 1.8 |
| GSC | feedforward | 0.6 | (-0.5, 0.5) | 1.2 |
| GSC | recurrent | 0.7 | (-0.8, 0.8) | 1.25 |
| SHD | feedforward | 0.5 | (-0.5, 0.5) | 1.5 |
| SHD | recurrent | 0.5 | (-0.5, 0.5) | 1.5 |
| SSC | feedforward | 0.5 | (-0.5, 0.5) | 1.5 |
| SSC | recurrent | 0.5 | (-0.5, 0.5) | 1.5 |

표 6. 네트워크 구조와 매개변수 요약.

| 데이터셋 | 네트워크 | 구조(Architecture) | 매개변수(K) |
| --- | --- | --- | --- |
| S-MNIST | feedforward | 40-256-128-10 / 64-256-256-10 | 44.8 / 85.1 |
| S-MNIST | recurrent | 40-200-64-10 / 64-256-256-10 | 63.6 / 155.1 |
| PS-MNIST | feedforward | 40-256-128-10 / 64-256-256-10 | 44.8 / 85.1 |
| PS-MNIST | recurrent | 40-200-64-10 / 64-256-256-10 | 63.6 / 155.1 |
| GSC | feedforward | 40-300-300-12 | 106.2 |
| GSC | recurrent | 40-300-300-12 | 196.5 |
| SHD | feedforward | 700-128-128-20 | 108.8 |
| SHD | recurrent | 700-128-128-20 | 141.8 |
| SSC | feedforward | 700-128-128-35 | 110.8 |
| SSC | recurrent | 700-128-35 | 110.8 |

#### 학습 설정(Training Configuration)

우리는 S-MNIST와 PS-MNIST 데이터셋을 Adam 최적화기(optimizer)를 사용하여 200 에폭 동안 학습하였다. 피드포워드 및 순환 네트워크 모두에서 초기 학습률(initial learning rate)은 0.0005로 설정했으며, 에폭 60과 80에서 학습률을 10배씩 감소시켰다. GSC, SHD, SSC 데이터셋의 경우에는 Adam 최적화기를 사용하여 100 에폭 동안 학습하였다. GSC 데이터셋의 초기 학습률은 피드포워드 및 순환 네트워크 모두에 대해 0.001로 설정하였으며, 에폭 60, 90, 120에서 10배씩 감소시켰다. SHD 데이터셋에서는 피드포워드 및 순환 네트워크의 초기 학습률을 각각 0.0005와 0.005로 설정하고, 10 에폭마다 이전 값의 0.8배로 감소시켰다. SSC 데이터셋의 경우 피드포워드 및 순환 네트워크 모두에 대해 초기 학습률을 0.0001로 설정하고, 10 에폭마다 이전 값의 0.8배로 감소시켰다. 우리는 S-MNIST, PS-MNIST, GSC 과제를 24GB 메모리를 갖는 Nvidia Geforce GTX 3090Ti GPU에서 학습하였고, SHD와 SSC 과제는 12GB 메모리를 갖는 Nvidia Geforce GTX 1080Ti GPU에서 학습하였다.

표 7. S-MNIST에서 학습 전후의 $\beta_1$, $\beta_2$ 및 이에 대응하는 무한 노름 값.

| 학습 전 $\beta_1, \beta_2$ | 학습 전 norm | 학습 후 $\beta_1, \beta_2$ | 학습 후 norm | 테스트 정확도 |
| --- | --- | --- | --- | --- |
| (-0.2, 0.2) | 1.352 | (-0.184, 0.146) | 1.262 | 98.40 |
| (-0.2, 0.4) | 1.688 | (-0.188, 0.307) | 1.539 | 99.07 |
| (-0.2, 0.6) | 2.008 | (-0.203, 0.563) | 1.948 | 99.15 |
| (-0.2, 0.8) | 2.312 | (-0.202, 0.835) | 2.360 | 89.36 |
| (-0.4, 0.2) | 1.304 | (-0.379, 0.159) | 1.248 | 99.01 |
| (-0.4, 0.4) | 1.576 | (-0.370, 0.318) | 1.480 | 99.15 |
| (-0.4, 0.6) | 1.816 | (-0.383, 0.532) | 1.751 | 99.06 |
| (-0.4, 0.8) | 2.024 | (-0.380, 0.700) | 1.949 | 99.04 |
| (-0.6, 0.2) | 1.256 | (-0.621, 0.172) | 1.219 | 99.08 |
| (-0.6, 0.4) | 1.464 | (-0.621, 0.342) | 1.399 | 99.01 |
| (-0.6, 0.6) | 1.624 | (-0.634, 0.594) | 1.587 | 98.96 |
| (-0.6, 0.8) | 1.736 | (-0.612, 0.761) | 1.704 | 98.82 |
| (-0.8, 0.2) | 1.208 | (-0.812, 0.187) | 1.194 | 98.97 |
| (-0.8, 0.4) | 1.352 | (-0.815, 0.360) | 1.321 | 99.20 |
| (-0.8, 0.6) | 1.432 | (-0.812, 0.580) | 1.416 | 98.64 |
| (-0.8, 0.8) | 1.448 | (-0.821, 0.801) | 1.418 | 98.99 |

#### 기울기 폭주 문제 분석(Gradient Exploding Problem Analysis)

우리는 사전에 정의한 영역에서 초기화된 TC-LIF와 관련하여 기울기 폭주 문제의 심각도를 분석하였다. 구체적으로, 두 번째 사분면에서 서로 다른 $(\beta_1, \beta_2)$ 로 초기화한 TC-LIF 모델을 사용해 S-MNIST 데이터셋에서 순환 SNN을 학습시켰다. 우리의 분석은 학습 전후의 $\beta$ 값을 기록하고, 마지막 은닉층(last hidden layer)에서 인접 시간 스텝 사이 편미분의 무한 노름(infinite norm)을 계산하는 과정을 포함한다.

결과적으로, 수렴 후 무한 노름 값이 2.36인 $(-0.2, 0.8)$ 초기화 모델을 제외하면, 두 번째 사분면에서 초기화된 나머지 모델들은 테스트 세트에서 우수한 성능을 보였다. 연속적인 시간 스텝 사이 편미분의 무한 노름이 1을 초과하면 장기 BPTT 과정에서 기울기 폭주가 발생할 가능성을 시사하지만, 우리의 결과는 무한 노름이 1보다 약간 큰 값일 때는 수렴이 크게 저해되지 않음을 보여준다. 고무적으로도, 두 번째 사분면에서 초기화된 대부분의 $\beta_1$ 과 $\beta_2$ 에 대해 대응되는 무한 노름 값은 이 조건을 만족한다. 따라서 TC-LIF 모델을 두 번째 사분면에서 초기화하면, 기울기 폭주 문제에 대한 우려를 완화하면서 안정적인 수렴을 기대할 수 있다.
