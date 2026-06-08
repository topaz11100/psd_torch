# SpikingSSMs(스파이킹 상태공간 모델): 희소하고 병렬적인 스파이킹 상태공간 모델로 긴 시퀀스 학습

Shuaijie Shen\*,1,2, Chao Wang\*,1,2, Renzhuo Huang1,2, Yan Zhong2,3, Qinghai Guo2, Zhichao Lu4, Jianguo Zhang†,1,5, Luziwei Leng†,2

## 소속
1. Southern University of Science and Technology(선전) — Department of Computer Science and Engineering(컴퓨터과학공학과)  
2. Huawei Technologies Co., Ltd.(선전) — ACSLab  
3. Peking University(베이징) — School of Mathematical Sciences(수리과학대학)  
4. City University of Hong Kong(홍콩) — Department of Computer Science(컴퓨터과학과)  
5. Pengcheng Laboratory(선전)

## 이메일
{shensj2024, wangc2023, huangrz2023}@mail.sustech.edu.cn, zhongyan@stu.pku.edu.cn, guoqinghai@huawei.com, zhichao.lu@cityu.edu.hk, zhangjg@sustech.edu.cn, lengluziwei@huawei.com

\* 동등 기여 저자  
† 교신저자  

---

## 초록

저에너지 소비 네트워크로 알려진 spiking neural networks(SNNs, 스파이킹 신경망)은 지난 수십 년간 큰 주목을 받아왔다. SNNs(스파이킹 신경망)은 시각(vision) 과제에서 artificial neural networks(ANNs, 인공 신경망)과 점점 더 경쟁력 있는 성능을 보이고 있지만, 고유한 시간적(temporal) 동역학(dynamics)에도 불구하고 긴 시퀀스(long sequence) 과제에서는 거의 사용되지 않는다. 본 연구에서는 state space models(SSMs, 상태공간 모델)의 시퀀스 학습 능력을 활용하여, 긴 시퀀스 학습을 위한 spiking state space models(SpikingSSMs, 스파이킹 상태공간 모델)을 개발한다. 수상돌기(dendritic) 뉴런 구조에서 영감을 받아, 원래의 SSM 블록(block)과 뉴런 동역학을 계층적으로 통합하면서 희소한 시냅스 연산(sparse synaptic computation)을 구현한다. 또한 이벤트 구동(event-driven) 뉴런 동역학과 병렬 계산(parallel computing) 사이의 충돌을 해결하기 위해, 리셋 이후(after-reset) 막전위(membrane potential)를 정확히 예측하고 학습 가능한 임계값(learnable thresholds)과도 호환되는 경량 surrogate dynamic network(SDN, 대리 동역학 네트워크)를 제안한다. 이는 기존의 반복적(iterative) 방법 대비 학습 속도를 수십~수백 배 수준으로 가속한다. Long Range Arena(LRA) 벤치마크 과제에서 SpikingSSM은 최신 SSM들과 경쟁력 있는 성능을 유지하면서도 평균 90%의 네트워크 희소성(network sparsity)을 달성한다. 언어 모델링(language modeling)에서는, 모델 크기를 1/3로 줄인 상태에서도 WikiText-103 데이터셋에서 기존 spiking large language models(spikingLLMs, 스파이킹 대규모 언어 모델)을 크게 상회하여, 저연산 비용 LLM의 백본(backbone) 아키텍처로서 잠재력을 보여준다.

**코드(Code)** — https://github.com/shenshuaijie/SDN  
**확장 버전(Extended version)** — https://arxiv.org/abs/2408.14909  

---

## 서론(Introduction)

최근 몇 년간 다양한 도메인에서 실제(real-world) 시계열(time-series) 데이터셋이 급증했으며, 이러한 데이터는 종종 수만(tens of thousands) 개의 타임스텝(time step)에 걸친 추론(reasoning)을 요구한다(Tay et al. 2021a). 이에 따라 텍스트(text), 비전(vision), 오디오(audio), 비디오(video) 등 다양한 모달리티(modality)에서 사람 수준의 성능을 목표로, 순차 데이터의 장거리 의존성(long-range dependencies, LRDs)을 모델링하는 다양한 시퀀스 모델이 등장했다(Gu, Goel, and Re 2022).

이들 가운데 Transformer(트랜스포머)(Vaswani et al. 2017)는 비전과 음성 분야에서의 뛰어난 발전으로 인해 큰 관심을 받았다. 그러나 길이 $L$인 입력 시퀀스에 대해 Transformer의 self-attention(자기-어텐션) 모듈은 학습 및 추론에서 $O(L^2)$의 높은 계산 복잡도를 요구한다. self-attention은 Transformer의 핵심 컨텍스트화(contextualizing) 구성요소 중 하나이기 때문이다. Reformer(Kitaev, Kaiser, and Levskaya 2020), BigBird(Zaheer et al. 2020), linear attention 계열(Katharopoulos et al. 2020), Performer(Choromanski et al. 2021) 등 일부 변형 모델이 계산 및 메모리 요구량을 낮추기 위해 제안되었지만, 장거리 추론 성능은 여전히 상당히 미흡하다(Gu, Goel, and Re 2022).

recurrent neural networks(RNNs, 순환 신경망)(Schuster and Paliwal 1997; Sherstinsky 2020)은 가변 길이 입력 시퀀스 학습을 위해 일찍부터 사용되어 왔으며, 시퀀스 길이에 대해 $O(1)$의 연산만 요구한다. 하지만 제한된 은닉 상태 공간(hidden state space)과 기울기 소실(gradient vanishing) 문제로 인해 긴 시퀀스 학습에 한계가 있었다. 이를 해결하기 위해 RWKV(Peng et al. 2023)나 state space models(SSMs, 상태공간 모델)(Gu, Goel, and Re 2022; Gu and Dao 2023)처럼, 장거리 의존성을 처리하면서도 학습 병렬화(parallelizability)와 추론 효율성을 동시에 확보할 수 있도록 은닉 상태 설계를 개선한 방법들이 제안되었다.

RNN은 인지·신경과학적 계산 원리(cognitive and neurological computational principles)(Lipton, Berkowitz, and Elkan 2015)에서 일부 영감을 받았는데, 이러한 원리는 Spiking Neural Networks(SNNs, 스파이킹 신경망)(Maass 1997)라는 또 다른 생물학적 기반(biologically-grounded) 아키텍처의 토대이기도 하다. SNNs(스파이킹 신경망)은 저에너지(low-energy) 컴퓨팅 잠재력으로 인해 지난 수십 년간 많은 주목을 받아왔다. 최근에는 합성곱(convolution) 또는 Transformer 아키텍처 하에서 비전 과제에서 ANN(인공 신경망)에 필적하는 효율을 보인다는 결과도 보고되었다(Che et al. 2022; Zhou et al. 2022; Yao et al. 2024; Che et al. 2024). 그러나 고유한 시간적 동역학이 있음에도 불구하고, SNNs는 긴 시퀀스 과제에서는 거의 활용되지 않는다. 또한 합성곱/Transformer 기반 SNN은 스파이크 기반 표현(spike-based representation)을 개선하기 위해 일정한 시뮬레이션 시간 창(simulation time window)을 필요로 하는 경우가 많아, 대응하는 ANN 대비 추론 지연(inference delay)이 발생한다. 이러한 단점은 RNN 아키텍처 기반 SNN에서는 내재적 시간 축을 동적 계산(dynamic computing)에 활용할 수 있으므로 완화될 수 있다.

본 연구에서는 spiking neuron(스파이킹 뉴런)과 SSM(상태공간 모델)의 통합을 탐구하여, 긴 시퀀스 학습을 위한 SpikingSSMs(스파이킹 상태공간 모델)을 제안한다. 제안 모델은 SSM의 효율적인 병렬 학습(parallel training) 및 장거리 시퀀스 모델링 능력과, SNN의 스파이크 기반 희소 연산(sparse computation) 및 저에너지 특성을 결합한다. 최근 binary SSM(이진 SSM)(Stan and Rhodes 2024)이나 stochastic spiking SSM(확률적 스파이킹 SSM)(Bal and Sengupta 2024)도 제안되었으나, 생물학적 스파이킹 뉴런을 특징짓는 복잡한 동역학을 충분히 탐구하지 않았거나 간과하여 해석가능성(interpretability)이 불완전하고 성능 저하가 발생하는 문제가 있었다. 이를 위해 본 연구는 deterministic reset(결정론적 리셋) 메커니즘을 갖는 Leaky Integrate-and-Fire(LIF, 누설 적분-발화) 뉴런(Gerstner et al. 2014)을 채택한다. 또한 비동기(asynchronous) 이벤트 구동 동역학과 병렬 계산의 충돌을 해소하기 위해, 학습을 가속하면서도 추론(inference) 시에는 추가 파라미터 없이 제거 가능한 SDN(대리 동역학 네트워크)을 제안한다. 등가성(equivalence) 분석을 통해 SDN이 파라메트릭(parametric) LIF 뉴런 모델을 근사(approximate)할 수 있음을 보이고, 병렬 계산 SNN을 위한 범용 모듈(general purpose module)로서의 잠재력도 논의한다.

본 연구의 주요 기여는 다음과 같다.

- 긴 시퀀스 과제를 위해 SpikingSSMs(스파이킹 상태공간 모델)을 제안한다. 이는 SSM의 병렬 계산과 장거리 시퀀스 모델링 강점, 그리고 SNN의 희소 연산 장점을 결합한다.
- 병렬 계산 관점에서 이벤트 구동 뉴런 동역학이 야기하는 어려움을 해결하기 위해, LIF 뉴런의 동역학을 잘 설계된 모델로 근사하는 SDN(대리 동역학 네트워크)을 제안한다. SDN은 추가 연산이 거의 없이 SpikingSSM의 학습을 크게 가속한다.
- 서로 다른 임계값(threshold)에 대해 SDN의 등가성을 보이고, 모델 아키텍처에 학습 가능한 임계값(learnable threshold)을 통합하여 성능을 추가로 향상한다.
- sequential MNIST(sMNIST) 및 permuted sequential MNIST(psMNIST) 분류, Long Range Arena(LRA) 벤치마크, 그리고 WikiText-103 언어 모델링 등 다양한 과제에서 평가한다. 제안 모델은 높은 희소성을 유지하면서도 최신 SSM에 경쟁력 있는 성능을 보이며, 대규모 언어 모델링에서도 스케일 확장성(scalability)을 입증한다.

---

## 관련 연구(Related Work)

### 긴 시퀀스 모델링(Long Sequence Modeling)

시퀀스 모델링의 핵심 문제는 컨텍스트(context)를 어떤 상태(state)로 압축(compress)하는 것이다. 이 문제에 의해, 시퀀스 모델은 효율(efficiency)과 효과(effectiveness) 사이의 트레이드오프(trade-off)를 탐구한다. 예를 들어 attention(어텐션) 메커니즘(Vaswani et al. 2017; Dao et al. 2022; Dao 2023)은 컨텍스트를 전혀 압축하지 않고, 자동회귀(auto-regressive) 추론 동안 전체 컨텍스트(KV cache)를 저장한다. 이는 효과적이지만, 선형 시간(linear-time) 추론과 이차 시간(quadratic-time) 학습을 유발하므로 비효율적이다(Sun et al. 2023; Yang et al. 2023). 반면 recurrent 모델은 컨텍스트를 유한 상태(finite state)로 압축하여 상수 시간(constant-time) 추론과 선형 시간 학습을 제공하지만, 상태가 컨텍스트를 얼마나 잘 압축하는지와 고정된 표현 공간(fixed representation space)에 의해 효과가 제한된다(Peng et al. 2023; Qin et al. 2024).

SSM(상태공간 모델)은 시퀀스 모델링을 위한 유망한 프레임워크로 부상했다. HiPPO(Gu et al. 2020)는 직교 다항식(orthogonal polynomials)을 이용해 긴 입력을 동적·다항식 기반 표현으로 압축하여 이 분야를 혁신했다. S4(Gu, Goel, and Re 2022)는 저랭크 보정(low-rank correction)을 도입해 안정적인 대각화(diagonalization)를 가능하게 하고 Cauchy kernel 연산을 단순화했다. 이후 parallel scan(Smith, Warrington, and Linderman 2023), Fast Fourier Transform(FFT, 빠른 푸리에 변환)(Fu et al. 2023; Duhamel and Vetterli 1990), gating mechanism(게이팅 메커니즘)(Mehta et al. 2023) 등의 기법으로 효율을 개선한 후속 연구가 이어졌다. 최근의 Mamba(Gu and Dao 2023)는 상태 표현의 선택성(selectivity)을 강화하여 컨텍스트 정보를 훼손하지 않으면서 효율과 효과를 균형 있게 달성하는 데 초점을 맞추었고, 하드웨어 최적화 알고리즘과 결합해 언어 모델링처럼 최대 백만 길이(million-length) 시퀀스까지 강한 성능을 보였다.

### 시퀀스 모델링을 위한 SNNs(SNNs for Sequence Modeling)

surrogate gradient(SG, 대리 기울기) 기반 학습 방법이 발전함에 따라, 기존 RNN 아키텍처를 채택한 SNN은 시퀀스 분류 과제에서 높은 정확도를 달성했다(Bellec et al. 2018; Yin, Corradi, and Bohte 2021, 2023). 그러나 아키텍처와 직렬 처리(serial processing) 제약 때문에, 순수 RNN 기반 SNN은 긴 시퀀스 학습에 드물게 적용된다. 따라서 SNN의 효율적인 병렬 계산을 가능하게 하는 것이 중요하다. PSN(Fang et al. 2023)은 스파이킹 뉴런의 reset(리셋)을 제거함으로써 병렬화를 달성했지만, 발화율(firing rate) 증가와 희소성 부족이라는 비용이 따른다. PSU(Li et al. 2024b)는 확률적 리셋(probabilistic reset) 메커니즘을 도입해 적분-발화-리셋(integration-spiking-resetting) 과정을 분리한 parallel spiking unit을 제안하여 희소성을 개선했지만, 학습 파라미터 수가 시퀀스 길이에 대해 이차(quadratic)로 증가하여 스케일 확장에 장애가 된다. Legendre Memory Units(LMU)(Voelker, Kajic, and Eliasmith 2019)를 활용한 SpikingLMUFormer(Liu et al. 2024)는 LMU에 합성곱 층과 스파이킹 활성화를 추가해 긴 시퀀스 모델링에서 트랜스포머를 능가했다. SSM의 최근 진전은 스파이킹 버전 개발도 자극했다. Du, Liu, and Chua는 S4 층 위에 LIF 뉴런을 단순히 쌓은 SpikeS4를 제안해 음성(speech) 과제에 적용했다. Binary S4D(Stan and Rhodes 2024)는 은닉 상태 합에 스파이킹 활성화 함수를 직접 적용해 병렬 학습을 유지했지만, 뉴런 동역학과 희소성을 무시했다. Bal and Sengupta(2024)는 S6 기반 SNN을 제안해 확률적 스파이킹 뉴런을 SSM에 구현하여 희소성을 개선했으나, 확률적 노이즈가 기울기에 섞이면서 원 모델 대비 정확도가 크게 하락했다. 본 연구에서는 결정론적 리셋 동역학을 갖는 스파이킹 뉴런을 채택하고, 비동기 이벤트 구동 특성과 병렬 계산의 충돌을 해결하는 해법을 제시한다.

### 언어 모델링을 위한 SNNs(SNNs for Language Modeling)

저에너지 대규모 언어 모델을 구축하려는 동기에서, 최근 SNN과 언어 모델을 결합하려는 연구가 진행되었다. SpikeGPT(Zhu et al. 2023)는 RWKV(Peng et al. 2023) 블록 출력에 스파이크 활성화를 적용해 대규모 언어 모델링에 적용했다. SpikeBERT(Lv et al. 2024)는 Spikformer(Zhou et al. 2022)를 기반으로 하고 원본 BERT(Devlin et al. 2018)로부터 지식 증류(knowledge distillation)를 수행했다. 본 연구에서는 SSM 아키텍처 기반의 대규모 SNN을 개발하여 언어 모델링에 적용한다.

---

## 방법(Method)

### 사전 지식(Preliminaries)

#### LIF 뉴런(LIF Neuron)

LIF 뉴런은 생물학적 뉴런 모델을 단순화한 형태(Gerstner et al. 2014)로, “누설-적분-발화-리셋(leaky-integrate-fire-reset)” 과정을 포착하며, 계산 가능성(tractability)과 시간 동역학을 균형 있게 제공하므로 머신러닝용 SNN에서 널리 사용된다. 시간 스텝(time step)을 $t$로 두면, LIF 뉴런은 다음과 같이 정의된다.

$$
u'_t = \tau u_{t-1} + I_t \quad (1)
$$

$$
s_t = H(u'_t - v_{th}) \quad (2)
$$

$$
\text{Soft reset:}\; u_t = u'_t - s_t v_{th} \quad (3)
$$

$$
\text{Hard reset:}\; u_t = u'_t(1 - s_t) + s_t u_r \quad (4)
$$

여기서 입력 전류(input current) $I_t$는 뉴런의 누설 막전위(leaky membrane potential) $u$에 선형적으로 적분된다. 이후 막전위 $u'_t$가 임계값(threshold) $v_{th}$를 넘으면 spike(스파이크) $s_t$가 발화되며, $H(\cdot)$는 Heaviside function(헤비사이드 함수)이다. 마지막으로 막전위는 soft reset(소프트 리셋, 식 (3)) 또는 hard reset(하드 리셋, 식 (4))에 따라 리셋된다. 하드/소프트 리셋은 서로 다른 뉴런 메모리 전략을 반영한다. 하드 리셋은 스파이크 이후 과거를 잊고(reset) 리셋 전위(reset potential) $u_r$로 되돌아가며(본 연구에서는 $u_r = 0$), 소프트 리셋은 스파이크 후에도 과거의 막전위 이력을 일정 부분 보존하되 리셋 양만큼 감산한다. 생물학적 뉴런을 더 현실적으로 모사하기 위해, 스파이킹 네트워크에서는 하드 리셋이 가장 흔히 사용된다.

#### SNN의 SG 학습(Surrogate Gradient Training of SNN)

스파이크는 동일한 사건으로 취급되므로 스파이킹 활성화 함수 $H$는 Heaviside 함수로 정의되며, $x=0$에서 미분 불가능하고 그 외 구간에서는 도함수가 0이다. 따라서 surrogate gradient(SG, 대리 기울기) 방법(Esser et al. 2016; Bellec et al. 2018)이 제안되었다. SG 함수는 스파이킹 활성화 함수의 불연속적인 기울기를 근사하는 완화(relaxed) 함수로서, 일반적으로 전 구간에서 미분 가능하고 임계값 근처에서 0이 아닌 도함수 값을 갖는다. 예로 직사각형(rectangular)(Zheng et al. 2021), 삼각형(triangular)(Bellec et al. 2018) 형태 등이 있다.

#### 상태공간 모델(SSM; State Space Model)

SSM(상태공간 모델)은 여러 과학 분야에서 널리 사용되며, 1차원 신호 $x$를 $N$차원 잠재 신호(latent signal) $h$로 사상하고, 이를 다시 1차원 출력 신호 $y$로 투영(project)한다. 이산(discrete) 입력 시퀀스 $x_{1:L}$에 대해, 어떤 이산화(discretization) 규칙(Gu et al. 2024)을 적용하면 SSM은 다음과 같이 정의된다.

$$
h_t = \bar{A} h_{t-1} + \bar{B} x_t \quad (5)
$$

$$
y_t = C h_t \quad (6)
$$

여기서 아래첨자 $t$는 시간 스텝을 의미한다. 파라미터는 상태 행렬(state matrix) $\bar{A} \in \mathbb{R}^{N\times N}$, 그리고 다른 행렬 $\bar{B} \in \mathbb{R}^{N\times 1}$, $C \in \mathbb{R}^{1\times N}$이다. 이론적으로 $\bar{A}$는 효율적 계산을 위해 대각화될 수 있다(Gupta, Gu, and Berant 2022). 한 층(layer)에서 입력은 보통 다차원(multidimensional)이므로, SSM 층은 여러 독립 SSM(서로 다른 파라미터를 갖는)을 병렬로 사용해 다중 특징(feature)을 처리한다.

병렬 계산 관점에서 SSM은, 초기 조건 $y_0 = 0$ 하에, 컨볼루션(convolution) 커널과 입력 신호의 컨볼루션으로 표현할 수 있다.

$$
y_t = \sum_{k=1}^{t} C\, \bar{A}^{\,t-k} \bar{B}\, x_k \quad (7)
$$

실제로는 Fast Fourier Transform(FFT, 빠른 푸리에 변환)을 이용해 이 계산을 $O(L\log L)$ 시간 복잡도로 더 가속할 수 있다(Gupta, Gu, and Berant 2022).

---

### Spiking S4 블록(Spiking S4 Block)

대각(diagonal) 형태의 SSM(Gupta, Gu, and Berant 2022)이 성능을 유지하면서도 모델을 단순화할 수 있음이 알려져 있다. 따라서 본 연구는 제안 방법을 검증하기 위한 백본(backbone)으로 최신 S4D(Gu et al. 2024) 모델을 선택한다. 상태공간 블록의 출력 $y$는 이제 LIF 뉴런으로 활성화(activate)된다. 즉, $y_t = C h_t$를 뉴런의 입력 전류(input current)로 취급한다.

$$
u'_t = \tau u_{t-1} + y_t \quad (8)
$$

$$
s_t = H(u'_t - v_{th}) \quad (9)
$$

스파이킹 출력 $s_t$는 다음 spiking S4 블록의 FC(fully-connected) 층으로 입력되며, 가중치 행렬과의 덧셈(addition) 연산을 통해 저에너지·희소 시냅스 계산을 실현한다. 임계값(threshold) $v_{th}$는 뉴런의 발화율(spiking rate)을 크게 제어하므로, 선행 연구(Rathi and Roy 2021)에 영감을 받아 본 연구에서는 $v_{th}$를 학습 가능한 파라미터로 두어 성능을 최적화한다. 서로 다른 S4 블록과 spiking S4 블록의 비교는 그림 1에 제시한다.

흥미롭게도 신경생물학적 관점에서, spiking S4 블록 구조는 다중 시간 스케일(multi-time scale)을 갖는 수상돌기(dendritic) 뉴런(London and Hausser 2005; Zheng et al. 2024)과 유사하다. 여기서 $h$는 수상돌기(dendrite), $y$는 수상돌기 입력을 집계해 받는 세포체(soma)에 대응하며, 둘 다 자기-순환(self-recurrent) 시간 동역학을 갖는다.

**그림 1(캡션 번역)**  
**그림 1:** SpikingSSM의 아키텍처. (a) 한 층에서의 SpikingSSM 순전파(forward) 계산 그래프. 연산 $r$은 reset(리셋) 메커니즘을 의미한다. 학습 가능한 파라미터 $\theta$는 임계값(threshold)처럼 스파이킹 함수 $f$에 영향을 주는 파라미터들을 의미한다. (b) 서로 다른 SSM들의 비교. 원래 SSM은 부동소수점(float point) 값을 출력한다. SpikingSSM은 원래 SSM의 비선형 함수(non-linear function)를 LIF 뉴런으로 대체하여 더 상위 계층에서 뉴런 동역학을 추가한다. SAF는 spiking activation function(스파이킹 활성화 함수)을 의미한다. 왼쪽 패널은 서로 다른 변수들의 계산 단계와 대응하는 차원(dimension)을 나타내며, $D$, $N$, $L$은 각각 모델 차원(model dimension), SSM의 은닉 차원(hidden dimension), 시퀀스 길이(sequence length)를 의미한다.

---

### SDN(대리 동역학 네트워크; Surrogate Dynamic Network)

출력 $y$는 병렬로 계산될 수 있다. 스파이킹 뉴런에 대한 입력 시퀀스 $y_{1:T}$가 주어졌을 때, 하드 리셋(hard reset) 하에서 $t\in[1,T]$인 막전위 $u_t$는 다음과 같이 쓸 수 있다.

$$
u_t
= \sum_{i=1}^{t}
\left[
\left(\prod_{j=i}^{t-1} (1 - s_j)\right)
\tau^{t-i}\, y_i
\right]
\quad (10)
$$

이 식에서 보듯, 막전위는 과거 스파이킹 이력(spiking history)에 의해 결정되며, 이는 병렬로 계산될 수 없다. 따라서 SNN은 보통 반복적(iterative) 계산 형태를 채택한다. 특히 이벤트 구동 리셋 메커니즘은 스파이킹 활성화의 비선형성 때문에 병렬 계산을 방해하며, 이는 긴 시퀀스 과제에서 현대 하드웨어에서의 효율적 학습을 어렵게 만든다.

한편 신경망(neural networks)은 입력-출력 사상(mapping)을 학습하도록 설계되어 있으며, 현대 하드웨어에서 병렬화할 수 있다. 고정 파라미터를 갖는 스파이킹 뉴런은 동일 입력에 대해 동일 출력을 만들어야 하므로, 뉴런을 “입력을 스파이크 시퀀스로 사상하는 블랙박스(black-box)”로 볼 수 있다. 이는 신경망이 잘하는 문제 설정이다. 따라서 본 연구는 사전 학습된(pre-trained) 신경망을 이용해 스파이크열(spike train)을 병렬로 예측하는 SDN(대리 동역학 네트워크)을 제안한다.

구체적으로, 입력을 출력 스파이크열로 매핑하는 뉴런 동역학을 학습하는 네트워크 $f$를 학습한다. 예를 들어 모든 타임스텝 입력에 기반해 스파이크열을 예측하는 신경망은 다음과 같이 표현될 수 있다.

$$
s_{1:T} = f(I_{1:T}) \quad (11)
$$

여기서 $I_{1:T}$는 시간 1부터 $T$까지의 입력 전류이며, $s_{1:T}$는 네트워크 $f$가 예측한 스파이크열이다.

효율성을 위해 SDN은 매우 작아야 하며, 순전파(inference)가 낮은 계산 비용으로 수행되어야 한다. 실험에서 보이듯 1-D convolution(1차원 합성곱)을 사용하는 3층 네트워크만으로도 뉴런 동역학을 충분히 학습해 스파이크를 정확히 예측할 수 있다(그림 2). 더 빠른 학습과 계산 그래프 단순화를 위해, 과제 네트워크(task network)를 학습할 때 사전 학습된 SDN을 역전파(backpropagation) 없이 추론 모드로 두고, SDN이 예측한 스파이크열과 입력을 이용해 식 (10)으로 막전위를 계산한다. 마지막으로 막전위가 임계값을 넘는지 여부에 따라 스파이크를 결정하여 spiking S4 블록의 출력으로 사용한다.

테스트(test) 시에는 SDN을 유지하여 시퀀스 길이에 대해 선형 시간 복잡도(linear-time complexity)로 병렬 추론을 수행할 수도 있고, SDN을 제거한 뒤 원래의 반복적 리셋 메커니즘으로 돌아가 $O(1)$의 실시간 반복 추론을 수행할 수도 있다. 이때 추가 파라미터는 필요하지 않다.

또한 계산 그래프의 복잡도를 줄이기 위해, SDN이 식 (8)의 $\tau u_{t-1}$ 항(누설 이후 막전위)을 직접 예측하도록 학습할 수도 있다. 이 경우 계산 그래프는 spatial learning through time(SLTT, 시간에 따른 공간 학습)(Meng et al. 2023)과 유사한 형태가 되며, 이는 SNN 학습에서 전통적인 backpropagation through time(BPTT, 시간 역전파)보다 효율적임이 보고되었다. 유도(derivation)의 자세한 내용은 보충 자료(supplement material)에 제공된다.

**그림 2(캡션 번역)**  
**그림 2:** 동일 입력에 대해 서로 다른 방법으로 생성된 막전위 샘플 비교. SDN(하단)이 예측한 막전위는 스파이킹 뉴런(중단)이 생성한 정답(ground truth)을 정확히 근사한다. 리셋이 없을 경우(상단) 막전위가 훨씬 더 많은 스파이크를 생성한다. 두 개의 검은 점선은 각각 리셋 전위(reset potential)와 스파이킹 임계값(spiking threshold)을 나타내며, 각각 0과 1로 설정되어 있다. 빨간 점은 스파이크가 생성되는 시점, 즉 막전위가 임계값을 넘는 순간을 의미한다. 스파이킹 뉴런의 경우 막전위가 임계값을 넘는 즉시 0으로 리셋된다.

**그림 3(캡션 번역)**  
**그림 3:** SDN 학습. 테스트 세트에서의 MSE(평균제곱오차) 손실과 스파이킹 정확도(spiking accuracy)를 나타낸다. SDN은 첫 학습 에폭(epoch) 이후 이미 충분히 높은 정확도에 도달한다.

---

### 학습 가능한 임계값(Learnable Threshold)

임계값(threshold)은 스파이크 생성 시점을 결정하며, SNN의 발화율(spiking rate)을 크게 조절한다. 임계값을 학습 과정에서 최적화하면 성능이 향상될 수 있음이 보고되었다(Rathi and Roy 2021; WANG, Cheng, and Lim 2022). 그렇다면 SpikingSSM 학습 중에 SDN이 서로 다른 임계값을 갖는 뉴런 동역학을 근사할 수 있을까? 본 연구는 등가성(equivalence) 분석을 통해 가능함을 보인다. 초기 막전위와 리셋 전위가 0이라고 할 때, 임계값의 중요한 성질은 다음과 같다.

**성질 1.** 입력과 임계값의 비율(ratio)이 뉴런의 동역학 과정을 결정한다.  
즉 입력과 임계값을 동일한 계수로 스케일링하면, 스파이크열은 변하지 않는다. 뉴런의 동역학 과정을 $f$로 나타내면 다음이 성립한다.

$$
s_{1:T} = f(I_{1:T}; v_{th}) = f(\alpha I_{1:T}; \alpha v_{th}) \quad (12)
$$

**성질 2.** 임계값은 입력 분포(distribution)를 스케일링한다.  
서로 다른 임계값을 갖는 뉴런에서, 임계값은 스케일 계수로 작동하므로, $v_{th}=1.0$인 “일반 SDN”을 구성하고 입력을 $I_{1:T}/v_{th}$처럼 스케일링하여 주입하면 된다. 따라서 이러한 성질에 기반해, 입력에 대한 스케일링 계수를 학습함으로써 SDN에 학습 가능한 임계값을 통합할 수 있다.

---

## 실험(Experiments)

본 절에서는 먼저 SDN의 아키텍처 설계, 학습 및 평가를 소개한다. 또한 SDN이 보조하는 SpikingSSM을 기존 반복적 LIF 뉴런 기반 학습 접근법과 비교하여 학습 속도를 벤치마크한다. 다음으로 서로 다른 규모의 세 가지 벤치마크 과제에서 SpikingSSM을 검증한다: (1) sequential MNIST 및 permuted sequential MNIST 분류, (2) LRA 데이터셋에서의 긴 시퀀스 모델링, (3) WikiText-103에서의 언어 모델링. 마지막으로 소거(ablation) 연구와 계산 비용 분석을 수행한다.

### SDN의 학습 및 평가(Training and Evaluation of SDN)

#### 데이터셋(Dataset)

SDN 학습 데이터셋은 입력 전류(input currents)와 그에 대응하는 목표 막전위(target membrane potentials)로 구성된다. 입력 $\in \mathbb{R}^{L}$은 정규분포 $\mathcal{N}(0,1)$에서 샘플링되며, 시퀀스 길이는 $L=1024$이다. 정답 막전위는 하드 리셋을 갖는 반복적 LIF 뉴런으로 생성한다. 학습 샘플 수는 $10^5$, 테스트 샘플 수는 $10^4$이다.

#### SDN 아키텍처(Architecture of SDN)

SDN은 1-D convolution과 1-D batch normalization으로 구성된 4층 CNN이며, 다음과 같이 표기한다.

`C8k1s1p0g1-C8k8s1p8g8-Trunc-BNrelu + C8k1s1p0g1-BN+relu - C1k1s1p0g1`

여기서 `C`, `k`, `s`, `p`, `g`는 각각 출력 채널 수(output channel), 커널 크기(kernel size), 스트라이드(stride), 패딩(padding), 그룹(group)을 의미하며, 뒤따르는 숫자는 각 값이다. `Trunc`는 입력을 잘라(truncate) 일정 길이를 유지함을 의미한다. 두 개의 `+` 표시는 residual connection(잔차 연결)의 시작과 끝을 의미한다.

이 설정에서 SDN의 총 파라미터 수는 200 미만이며, 백본 네트워크에 비해 매우 작다. 아키텍처의 자세한 내용은 보충 자료에 제공된다.

#### 적합(피팅) 성능(Fitting Ability)

생성된 데이터셋에서 SDN을 MSE(평균제곱오차) 손실로 100 에폭 동안 학습한다. 테스트에서는 예측 막전위로부터 생성된 스파이크를 정답 막전위로부터 생성된 스파이크와 비교하여 스파이킹 정확도를 평가한다. 그림 3에서 보듯 손실은 수렴하며 스파이킹 정확도도 점차 높아진다. 또한 이해를 돕기 위해 SDN이 예측한 막전위 샘플을 제시한다. 그림 2에서 보듯 SDN이 예측한 막전위는 정답을 매우 가깝게 근사한다. 리셋이 없을 경우 막전위는 훨씬 더 많은 스파이크를 생성한다. 일부 경우 네트워크가 막전위를 소폭 잘못 리셋하는 경우가 있는데, 이는 막전위가 임계값에 매우 근접할 때(예: 3번째 스텝) 발생할 수 있다. 이러한 차이는 최종 과제 네트워크 성능에 거의 영향을 미치지 않으며, 소거 실험에서 확인한다.

#### 학습 속도 비교(Comparison on Training Speed)

SDN이 보조하는 SpikingSSM의 학습 속도를, 반복적 LIF 뉴런에 기반한 전통적 학습 방법들과 비교한다. 비교 대상은 BPTT(시간 역전파)와, 계산 그래프를 최적화한 최근 방법인 SLTT이다. 입력은 배치 크기 64의 1-D 시퀀스이며 길이는 $L=1\text{K}, 2\text{K}, 4\text{K}, 8\text{K}$로 변화시킨다. 시간 측정은 단일 GPU에서 수행한다.

**표 1(캡션 번역)**  
**표 1:** 서로 다른 방법의 학습 속도 비교. 입력 배치 크기는 64이다. SDN을 사용한 학습은 큰 가속을 달성하며, 시퀀스 길이가 증가할수록 속도 향상 비율이 더 커진다.

| 방법(Method) | $L=1\text{K}$ (ms) | $L=2\text{K}$ (ms) | $L=4\text{K}$ (ms) | $L=8\text{K}$ (ms) |
|---|---:|---:|---:|---:|
| BPTT | 1370 | 2900 | 8040 | 25600 |
| SLTT | 1210 | 2720 | 7740 | 25600 |
| Ours(제안; SDN) | 183 | 196 | 200 | 253 |
| Ratio(가속 비율) | 7.5× | 15.0× | 40.2× | 101.2× |

표 1에서 보듯 SDN을 사용하면 시퀀스 길이가 길어질수록 가속 비율이 커지며, $L=8\text{K}$에서는 두 자릿수 이상(100× 수준)의 가속을 달성한다. 따라서 SDN은 특히 긴 시퀀스에서 SpikingSSM 학습을 극적으로 가속한다.

---

### SpikingSSM을 이용한 긴 시퀀스 과제(Long Sequence Tasks with SpikingSSM)

#### Sequential MNIST

MNIST 데이터셋(Yann and Cortes 1998)은 0~9 손글씨 숫자의 28×28 회색조 이미지 70,000장으로 구성되며, 60,000장 학습과 10,000장 테스트로 나뉜다. sequential MNIST(sMNIST)(Le, Jaitly, and Hinton 2015)는 2차원 이미지를 784 길이의 1차원 시퀀스로 펼쳐(flatten) 만든다. permuted sequential MNIST(psMNIST)(Le, Jaitly, and Hinton 2015)는 픽셀에 고정된 순열(permutation)을 적용해 시퀀스 내 시간 구조를 왜곡한다. 표 2에서 보듯 SpikingSSM은 sMNIST와 psMNIST 모두에서 다른 방법들과 경쟁력 있는 성능을 보인다.

**표 2(캡션 번역)**  
**표 2:** sMNIST 및 psMNIST에서 SpikingSSM과 다른 방법들의 성능 비교.

| 모델(Model) | SNN 여부 | sMNIST | psMNIST |
|---|:---:|---:|---:|
| LMUformer | No | — | 98.55 |
| S4 | No | 99.63 | 98.70 |
| SpikingLMUformer | Yes | — | 97.92 |
| Binary-S4D | Yes | 99.4 | — |
| S6-based SNN | Yes | — | 98.4 |
| **SpikingSSM(제안)** | **Yes** | **99.6** | **98.4** |

#### LRA(Long Range Arena)

LRA 벤치마크(Tay et al. 2021b)는 긴 컨텍스트(long-context) 시나리오에서 시퀀스 모델을 비교하기 위해 제안되었다. LRA는 1K~16K 스텝 범위의 시퀀스를 다루는 6개 과제로 구성되며, 시각 데이터, 수학 표현식, 텍스트 등 다양한 모달리티를 포함한다. 이 과제들은 텍스트 분류, 문서 검색(document retrieval), 이미지 분류, pathfinder, listops 등 긴 컨텍스트 이해 능력을 평가하도록 설계되었다.

표 3은 Transformer 또는 SSM 아키텍처를 갖는 비스파이킹(non-spiking)·스파이킹(spiking) 네트워크들과 SpikingSSM을 비교한다. SpikingSSM은 S4D-Lin(Gu et al. 2024)의 초기화와 유사한 설정을 사용하는 S4D 기반 아키텍처를 채택한다(자세한 아키텍처는 보충 자료 참고). 원 모델과 비슷한 정확도를 유지하면서도, 평균 약 90%의 네트워크 희소성을 달성한다. 또한 기존 SNN 시퀀스 모델 대비 큰 성능 향상을 보인다. 특히 SpikingSSM은 128×128 길이(총 16,384 스텝)의 장거리 의존성 추론이 필요한 매우 어려운 Path-X 과제를 성공적으로 해결한다.

학습 가능한 임계값을 사용하는 SpikingSSM(-VT)은 고정 임계값(-VF) 대비 전반적인 성능과 희소성에서 더 낫다. 추가 분석 결과, 학습 가능한 임계값은 막전위의 이봉 분포(bimodal distribution)를 촉진하여 스파이킹의 양자화 오차(quantization error)를 줄이고 SNN의 정보 전달(information transmission)을 개선하는데, 이는 선행 연구(Guo et al. 2022)와 일치한다(자세한 내용은 보충 자료 참고).

**표 3(캡션 번역)**  
**표 3:** LRA 데이터셋에서 SpikingSSM과 선행 연구들의 성능 비교. \* 원래 S4D-Lin은 Path-X 과제에 실패했기 때문에, 여기서는 가까운 변형인 S4D-Inv의 결과를 제시한다. -VF와 -VT는 각각 고정 임계값(fixed threshold)과 학습 가능한 임계값(trainable threshold)을 의미한다. 또한 S4D 연구와 동일하게 Path-X 정확도가 없는 경우 50% 정확도를 사용하고, 모든 과제에 대해 평균(AVG)을 계산한다. 각 과제의 spiking rate(발화율)도 함께 산출하였다.

| 모델(Model) | SNN 여부 | LISTOPS | TEXT | RETRIEVAL | IMAGE | PATHFINDER | Path-X | AVG |
|---|:---:|---:|---:|---:|---:|---:|---:|---:|
| Transformer | No | 36.37 | 64.27 | 57.46 | 42.44 | 71.40 | — | 53.66 |
| LMUFormer | No | 34.43 | 68.27 | 78.65 | 54.16 | 69.90 | — | 59.24 |
| S4D-Lin | No | 60.52 | 86.97 | 90.96 | 87.93 | 93.96 | 92.80\* | 85.52 |
| Spiking LMUFormer | Yes | 37.30 | 65.80 | 79.76 | 55.65 | 72.68 | — | 60.20 |
| Binary S4D | Yes | 54.80 | 82.50 | 85.03 | 82.00 | 82.60 | 61.20 | 74.69 |
| S6-based SNN | Yes | 55.70 | 77.62 | 88.48 | 80.10 | 83.41 | — | 72.55 |
| SpikingSSM-VF(제안; 정확도) | Yes | 59.93 | 82.35 | 88.20 | 86.81 | 93.68 | 94.80 | 84.30 |
| SpikingSSM-VF(발화율) |  | (0.13) | (0.10) | (0.06) | (0.22) | (0.07) | (0.10) | (0.11) |
| SpikingSSM-VT(제안; 정확도) | Yes | 60.23 | 80.41 | 88.77 | 88.21 | 93.51 | 94.82 | 84.33 |
| SpikingSSM-VT(발화율) |  | (0.14) | (0.06) | (0.06) | (0.15) | (0.08) | (0.10) | (0.10) |

#### WikiText-103

WikiText-103 데이터셋은 Good 또는 Featured로 평가된 위키피디아 문서들로부터 수집된, 1억 개 이상의 토큰을 포함하는 대규모 텍스트 컬렉션이다. 본 연구는 일반적으로 사용되는 perplexity(PPL, 퍼플렉서티)를 지표로 사용한다. 이 데이터셋은 전체 문서 단위로 구성되어 있어 장기 의존성을 포착해야 하는 모델에 특히 적합하며, 단어 수준(word-level) 언어 모델링의 중요한 벤치마크이다.

실험에서는 S4 모델 대비 더 파라미터 효율적인 설정을 사용했다(자세한 내용은 보충 자료 참고). 파라미터 수가 크게 적음에도, SpikingSSM은 사전학습된 SpikeGPT를 능가하며, ANN 기반 네트워크와의 성능 격차도 상당히 줄인다.

**표 4(캡션 번역)**  
**표 4:** WikiText-103 데이터셋에서 SpikingSSM과 선행 연구들의 성능 비교.

| 모델(Model) | SNN 여부 | PPL | 파라미터(Parameters) |
|---|:---:|---:|---:|
| Transformer | No | 20.51 | 231M |
| S4 | No | 20.95 | 249M |
| SpikeGPT | Yes | 39.75 | 213M |
| **SpikingSSM(제안)** | **Yes** | **33.94** | **75M** |

---

### 소거 연구(Ablation Study)

SDN의 역할을 검증하기 위해, 학습 동안 SpikingSSM에서 LIF 뉴런을 SDN으로 대체했을 때 성능 저하가 발생하는지 소거 실험을 수행한다. 또한 SDN은 사전 학습된 네트워크로서 LIF 뉴런 동역학을 모사하도록 학습되었는데, 이러한 “LIF로 작동하도록 하는 바이어스(bias)”가 실제로 성능에 도움이 되는지도 확인한다.

동일한 아키텍처와 하이퍼파라미터(hyperparameters)를 사용하되 스파이킹 활성화만 다른 세 가지 모델을 구성한다.
- **LIF**: 반복적 LIF 뉴런 사용  
- **SDN-S**: SpikingSSM을 end-to-end로 학습하면서 SDN을 scratch(처음부터)로 학습  
- **SDN**: 사전 학습된 고정 SDN 사용  

세 모델을 sCIFAR10 데이터셋(LRA의 IMAGE 서브셋)에서 학습한다. 결과는 표 5에 제시한다. SDN 모델은 반복적 LIF 뉴런과 비슷한 정확도를 유지하면서도 학습을 크게 가속한다. 반면 SDN-S는 SDN만큼의 성능을 달성하지 못하는데, 이는 SDN을 LIF 뉴런처럼 제한하는 바이어스가 유익함을 보여준다.

**표 5(캡션 번역)**  
**표 5:** sCIFAR10 데이터셋에서의 성능 비교.

| 모델(Model) | 정확도(Accuracy, %) | 발화율(Spiking Rate, %) | 속도(Speed, ms) |
|---|---:|---:|---:|
| LIF | 85.45 | 12.08 | 1480 |
| SDN | 85.57 | 11.92 | 230 |
| SDN-S | 81.52 | 18.30 | 285 |

---

### 계산 비용(Computation Cost)

스파이킹 네트워크가 저에너지로 간주되는 이유는 스파이킹 뉴런의 활성값(activation)이 이진(binary)이며, 일부 뉴로모픽 칩(neuromorphic chip)에서는 이진 활성값과 부동소수점 가중치(weight) 사이의 곱셈을 덧셈만으로 구현할 수 있기 때문이다(예: Speck; Richter et al. 2023). 따라서 SNN의 주요 연산인 시냅스 누산(synaptic Accumulation, AC)은 ANN의 주요 연산인 곱-누산(Multiply-and-Accumulate, MAC)보다 에너지 비용이 낮다. 하드웨어 구현과 뉴런 동역학의 세부는 무시하더라도, 이론적 에너지 소비 분석은 SNN의 효율을 추정할 수 있다. 선행 연구(Richter et al. 2023; Li et al. 2024a)를 따라, MAC의 에너지 비용을 $E_{MAC}=4.6\,\text{pJ}$, AC의 에너지 비용을 $E_{AC}=0.9\,\text{pJ}$로 가정한다(Horowitz 2014).

발화율(spiking rate)은 한 뉴런에서 전체 시간 스텝 대비 스파이크 개수의 비율로 정의한다. 네트워크 전체의 평균 발화율(mean spiking rate)은 모든 뉴런의 발화율 평균이며, 본 연구에서는 이를 spiking rate로 표기한다.

그림 4는 각 층(layer)의 발화율을 보여준다. 또한 파라미터와 계산은 주로 feature-mix 층에서 발생하므로, 해당 층에서의 MAC, AC, 에너지 비용을 산출한다. WikiText-103에서 샘플 길이 $L=8192$인 경우, 모델은 16개 층을 가지며, 스파이크를 $d=1024$에서 $d=2048$로 투영(project)하는 선형 층을 포함한다. 만약 모든 투영이 부동소수점 곱셈으로 수행된다면 275.2G MAC이 필요하며 약 1.265J의 에너지가 든다. 그러나 본 모델에서는 이 층 입력이 이진 값이며 평균 발화율이 30% 미만이다. 그림 4의 발화율을 반영하면 72.66G AC만 필요하며 에너지 비용은 약 65.40mJ이다.

**그림 4(캡션 번역)**  
**그림 4:** sCIFAR10 및 WikiText-103 데이터셋에서 SpikingSSM의 모든 층에 걸친 발화율(spiking rate).

---

## 결론(Conclusion)

본 연구는 LIF 뉴런 동역학을 SSM과 계층적으로 통합하여, 긴 시퀀스 학습에서 경쟁력 있는 성능과 SNN의 효율적인 희소 계산을 동시에 달성하는 SpikingSSM을 제안했다. 반복적 LIF 뉴런을 사용하는 SNN의 효율적 학습을 위해, 병렬 계산으로 LIF 뉴런 동역학을 근사하는 SDN을 제안했으며, 이는 SpikingSSM 학습을 극적으로 가속한다. SDN은 과제 스파이킹 네트워크 학습 시 추론 모드로 사용되어 추가 연산이 거의 없고, 다양한 파라메트릭 LIF 뉴런 모델도 근사할 수 있는 범용 모듈로서의 잠재력을 보였다. LRA와 WikiText-103을 포함한 다양한 벤치마크에서, SpikingSSM은 기존 방법 대비 경쟁력 있는 성능과 높은 희소성 및 저에너지 요구량의 장점을 입증했다. 본 연구는 특히 긴 시퀀스 데이터를 효율적으로 처리해야 하는 분야에서 SNN의 활용 범위를 확장하는 데 기여한다.

---

## 감사의 글(Acknowledgments)

본 연구는 중국 국가 중점 연구개발 프로그램(National Key Research and Development Program of China, 2021YFF1200800), 중국 국가자연과학기금(National Natural Science Foundation of China; Grant No. 62276121, 12326604), 그리고 과학기술 혁신 2030-중대 프로젝트(뇌과학 및 뇌유사 지능 기술; Grant 2022ZD0208700)의 지원을 받았다.

---

## References (번역하지 않음)

Bal, M.; and Sengupta, A. 2024. Rethinking Spiking Neural Networks as State Space Models. arXiv:2406.02923.  
Bellec, G.; Salaj, D.; Subramoney, A.; Legenstein, R.; and Maass, W. 2018. Long short-term memory and learning-to-learn in networks of spiking neurons. In The Thirty-second Conference on Neural Information Processing Systems. Curran Associates Inc.  
Che, K.; Leng, L.; Zhang, K.; Zhang, J.; Meng, Q.; Cheng, J.; Guo, Q.; and Liao, J. 2022. Differentiable hierarchical and surrogate gradient search for spiking neural networks. In The Thirty-sixth Conference on Neural Information Processing Systems, 24975–24990. Curran Associates Inc.  
Che, K.; Zhou, Z.; Yuan, L.; Zhang, J.; Tian, Y.; and Leng, L. 2024. Spatial-Temporal Search for Spiking Neural Networks. arXiv preprint arXiv:2410.18580.  
Choromanski, K. M.; Likhosherstov, V.; Dohan, D.; Song, X.; Gane, A.; Sarlos, T.; Hawkins, P.; Davis, J. Q.; Mohiuddin, A.; Kaiser, L.; Belanger, D. B.; Colwell, L. J.; and Weller, A. 2021. Rethinking Attention with Performers. In International Conference on Learning Representations.  
Dao, T. 2023. Flashattention-2: Faster attention with better parallelism and work partitioning. arXiv preprint arXiv:2307.08691.  
Dao, T.; Fu, D.; Ermon, S.; Rudra, A.; and Re, C. 2022. Flashattention: Fast and memory-efficient exact attention with io-awareness. In The Thirty-sixth Conference on Neural Information Processing Systems, 16344–16359. Curran Associates Inc.  
Devlin, J.; Chang, M.-W.; Lee, K.; and Toutanova, K. 2018. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv preprint arXiv:1810.04805.  
Du, Y.; Liu, X.; and Chua, Y. 2024. Spiking structured state space model for monaural speech enhancement. In 2024 IEEE International Conference on Acoustics, Speech and Signal Processing, 766–770. IEEE.  
Duhamel, P.; and Vetterli, M. 1990. Fast Fourier transforms: a tutorial review and a state of the art. Signal processing, (4): 259–299.  
Esser, S. K.; Merolla, P. A.; Arthur, J. V.; Cassidy, A. S.; Appuswamy, R.; Andreopoulos, A.; Berg, D. J.; McKinstry, J. L.; Melano, T.; Barch, D. R.; di Nolfo, C.; Datta, P.; Amir, A.; Taba, B.; Flickner, M. D.; and Modha, D. S. 2016. Convolutional networks for fast, energy-efficient neuromorphic computing. Proceedings of the National Academy of Sciences, 113: 11441–11446.  
Fang, W.; Yu, Z.; Zhou, Z.; Chen, D.; Chen, Y.; Ma, Z.; Masquelier, T.; and Tian, Y. 2023. Parallel Spiking Neurons with High Efficiency and Ability to Learn Long-term Dependencies. In Oh, A.; Naumann, T.; Globerson, A.; Saenko, K.; Hardt, M.; and Levine, S., eds., Advances in Neural Information Processing Systems, volume 36, 53674–53687. Curran Associates, Inc.  
Fu, D. Y.; Dao, T.; Saab, K. K.; Thomas, A. W.; Rudra, A.; and Re, C. 2023. Hungry hungry hippos: Towards language modeling with state space models. In The eleventh International Conference on Learning Representations.  
Gerstner, W.; Kistler, W. M.; Naud, R.; and Paninski, L. 2014. Neuronal dynamics: From single neurons to networks and models of cognition. Cambridge University Press.  
Gu, A.; and Dao, T. 2023. Mamba: Linear-Time Sequence Modeling with Selective State Spaces. arXiv preprint arXiv:2312.00752.  
Gu, A.; Dao, T.; Ermon, S.; Rudra, A.; and Re, C. 2020. Hippo: Recurrent memory with optimal polynomial projections. In The Thirty-fourth Conference on Neural Information Processing Systems, 1474–1487. Vancouver,Canada: Curran Associates Inc.  
Gu, A.; Goel, K.; and Re, C. 2022. Efficiently Modeling Long Sequences with Structured State Spaces. In International Conference on Learning Representations.  
Gu, A.; Gupta, A.; Goel, K.; and Re, C. 2024. On the parameterization and initialization of diagonal state space models. In Proceedings of the 36th International Conference on Neural Information Processing Systems, NIPS ’22. Red Hook, NY, USA: Curran Associates Inc. ISBN 9781713871088.  
Guo, Y.; Tong, X.; Chen, Y.; Zhang, L.; Liu, X.; Ma, Z.; and Huang, X. 2022. Recdis-snn: Rectifying membrane potential distribution for directly training spiking neural networks. In 2022 IEEE/CVF conference on computer vision and pattern recognition, 326–335. IEEE.  
Gupta, A.; Gu, A.; and Berant, J. 2022. Diagonal state spaces are as effective as structured state spaces. In The Thirty-sixth Conference on Neural Information Processing Systems, 22982–22994. Curran Associates Inc.  
Horowitz, M. 2014. 1.1 Computing’s energy problem (and what we can do about it). In 2014 IEEE International Solid-State Circuits Conference Digest of Technical Papers (ISSCC), 10–14. IEEE.  
Katharopoulos, A.; Vyas, A.; Pappas, N.; and Fleuret, F. 2020. Transformers are RNNs: fast autoregressive transformers with linear attention. In Proceedings of the 37th International Conference on Machine Learning, ICML’20. JMLR.org.  
Kitaev, N.; Kaiser, L.; and Levskaya, A. 2020. Reformer: The Efficient Transformer. In International Conference on Learning Representations.  
Le, Q. V.; Jaitly, N.; and Hinton, G. E. 2015. A Simple Way to Initialize Recurrent Networks of Rectified Linear Units. arXiv:1504.00941.  
Li, B.; Leng, L.; Shen, S.; Zhang, K.; Zhang, J.; Liao, J.; and Cheng, R. 2024a. Efficient Deep Spiking Multilayer Perceptrons With Multiplication-Free Inference. IEEE Transactions on Neural Networks and Learning Systems, 1–13.  
Li, Y.; Sun, Y.; He, X.; Dong, Y.; Zhao, D.; and Zeng, Y. 2024b. Parallel Spiking Unit for Efficient Training of Spiking Neural Networks. In 2024 International Joint Conference on Neural Networks, 1–8.  
Lipton, Z. C.; Berkowitz, J.; and Elkan, C. 2015. A Critical Review of Recurrent Neural Networks for Sequence Learning. arXiv:1506.00019.  
Liu, Z.; Datta, G.; Li, A.; and Beerel, P. A. 2024. LMUFormer: Low Complexity Yet Powerful Spiking Model With Legendre Memory Units. arXiv preprint arXiv:2402.04882.  
London, M.; and Hausser, M. 2005. Dendritic computation. Annu. Rev. Neurosci., 28: 503–532.  
Lv, C.; Li, T.; Xu, J.; Gu, C.; Ling, Z.; Zhang, C.; Zheng, X.; and Huang, X. 2024. SpikeBERT: A Language Spikformer Learned from BERT with Knowledge Distillation. arXiv:2308.15122.  
Maass, W. 1997. Networks of spiking neurons: The third generation of neural network models. Neural Networks, 10(9): 1659–1671.  
Mehta, H.; Gupta, A.; Cutkosky, A.; and Neyshabur, B. 2023. Long range language modeling via gated state spaces. In The eleventh International Conference on Learning Representations.  
Meng, Q.; Xiao, M.; Yan, S.; Wang, Y.; Lin, Z.; and Luo, Z.-Q. 2023. Towards memory-and time-efficient backpropagation for training spiking neural networks. In 2023 IEEE/CVF International Conference on Computer Vision, 6166–6176. Paris, France.  
Peng, B.; Alcaide, E.; Anthony, Q.; Albalak, A.; Arcadinho, S.; Biderman, S.; Cao, H.; Cheng, X.; Chung, M.; Derczynski, L.; Du, X.; Grella, M.; Gv, K.; He, X.; Hou, H.; Kazienko, P.; Kocon, J.; Kong, J.; Koptyra, B.; Lau, H.; Lin, J.; Mantri, K. S. I.; Mom, F.; Saito, A.; Song, G.; Tang, X.; Wind, J.; Wozniak, S.; Zhang, Z.; Zhou, Q.; Zhu, J.; and Zhu, R.-J. 2023. RWKV: Reinventing RNNs for the Transformer Era. In Bouamor, H.; Pino, J.; and Bali, K., eds., Findings of the Association for Computational Linguistics: EMNLP 2023, 14048–14077. Singapore: Association for Computational Linguistics.  
Qin, Z.; Li, D.; Sun, W.; Sun, W.; Shen, X.; Han, X.; Wei, Y.; Lv, B.; Luo, X.; Qiao, Y.; and Zhong, Y. 2024. TransNormerLLM: A Faster and Better Large Language Model with Improved TransNormer. arXiv:2307.14995.  
Rathi, N.; and Roy, K. 2021. Diet-snn: A low-latency spiking neural network with direct input encoding and leakage and threshold optimization. IEEE Transactions on Neural Networks and Learning Systems, 34: 3174–3182.  
Richter, O.; Xing, Y.; Marchi, M. D.; Nielsen, C.; Katsimpris, M.; Cattaneo, R.; Ren, Y.; Liu, L.-Y. D.; Sheik, S.; Demirci, T.; NingQiaoSynSense, A.; Swizerland; SynSense; China, P. R.; Circuits, B.-I.; Lab, S.; for Advanced Materials, Z. I.; of Groningen, U.; Netherlands; Systems, G. C.; Center, M. S.; and Netherlands. 2023. Spike-based dynamic computing with asynchronous sensing-computing neuromorphic chip. Nature Communications, 15.  
Schuster, M.; and Paliwal, K. K. 1997. Bidirectional recurrent neural networks. IEEE transactions on Signal Processing, 45: 2673–2681.  
Sherstinsky, A. 2020. Fundamentals of Recurrent Neural Network (RNN) and Long Short-Term Memory (LSTM) network. Physica D: Nonlinear Phenomena, 404: 132306.  
Smith, J. T.; Warrington, A.; and Linderman, S. W. 2023. Simplified state space layers for sequence modeling. In The eleventh International Conference on Learning Representations.  
Stan, M.-I.; and Rhodes, O. 2024. Learning long sequences in spiking neural networks. Scientific Reports, 14: 21957.  
Sun, Y.; Dong, L.; Huang, S.; Ma, S.; Xia, Y.; Xue, J.; Wang, J.; and Wei, F. 2023. Retentive network: A successor to transformer for large language models. arXiv:2307.08621.  
Tay, Y.; Dehghani, M.; Abnar, S.; Shen, Y.; Bahri, D.; Pham, P.; Rao, J.; Yang, L.; Ruder, S.; and Metzler, D. 2021a. Long Range Arena : A Benchmark for Efficient Transformers. In International Conference on Learning Representations.  
Tay, Y.; Dehghani, M.; Abnar, S.; Shen, Y.; Bahri, D.; Pham, P.; Rao, J.; Yang, L.; Ruder, S.; and Metzler, D. 2021b. Long Range Arena : A Benchmark for Efficient Transformers. In The Ninth International Conference on Learning Representations. Vienna,Austria.  
Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A. N.; Kaiser, L.; and Polosukhin, I. 2017. Attention is all you need. In Proceedings of the 31st International Conference on Neural Information Processing Systems, NIPS’17, 6000–6010. Red Hook, NY, USA: Curran Associates Inc. ISBN 9781510860964.  
Voelker, A.; Kajic, I.; and Eliasmith, C. 2019. Legendre memory units: Continuous-time representation in recurrent neural networks. In The Thirty-third Conference on Neural Information Processing Systems. Curran Associates Inc.  
WANG, S.; Cheng, T. H.; and Lim, M.-H. 2022. LTMD: Learning Improvement of Spiking Neural Networks with Learnable Thresholding Neurons and Moderate Dropout. In Koyejo, S.; Mohamed, S.; Agarwal, A.; Belgrave, D.; Cho, K.; and Oh, A., eds., Advances in Neural Information Processing Systems, volume 35, 28350–28362. Curran Associates, Inc.  
Yang, S.; Wang, B.; Shen, Y.; Panda, R.; and Kim, Y. 2023. Gated linear attention transformers with hardware-efficient training. arXiv:2312.06635.  
Yann, L.; and Cortes, C. 1998. The MNIST database of handwritten digits. http://yann.lecun.com/exdb/mnist/.  
Yao, M.; Hu, J.; Hu, T.; Xu, Y.; Zhou, Z.; Tian, Y.; XU, B.; and Li, G. 2024. Spike-driven Transformer V2: Meta Spiking Neural Network Architecture Inspiring the Design of Next-generation Neuromorphic Chips. In The Twelfth International Conference on Learning Representations. Vienna Austria.  
Yin, B.; Corradi, F.; and Bohte, S. M. 2021. Accurate and efficient time-domain classification with adaptive spiking recurrent neural networks. Nature Machine Intelligence, 3: 905–913.  
Yin, B.; Corradi, F.; and Bohte, S. M. 2023. Accurate online training of dynamical spiking neural networks through forward propagation through time. Nature Machine Intelligence, 5: 518–527.  
Zaheer, M.; Guruganesh, G.; Dubey, A.; Ainslie, J.; Alberti, C.; Ontanon, S.; Pham, P.; Ravula, A.; Wang, Q.; Yang, L.; and Ahmed, A. 2020. Big bird: transformers for longer sequences. In Proceedings of the 34th International Conference on Neural Information Processing Systems, NIPS ’20. Red Hook, NY, USA: Curran Associates Inc. ISBN 9781713829546.  
Zheng, H.; Wu, Y.; Deng, L.; Hu, Y.; and Li, G. 2021. Going Deeper With Directly-Trained Larger Spiking Neural Networks. In The Thirty-Fifth AAAI Conference on Artificial Intelligence, 11062–11070. online: AAAI Press.  
Zheng, H.; Zheng, Z.; Hu, R.; Xiao, B.; Wu, Y.; Yu, F.; Liu, X.; Li, G.; and Deng, L. 2024. Temporal dendritic heterogeneity incorporated with spiking neural networks for learning multi-timescale dynamics. Nature Communications, 15: 277.  
Zhou, Z.; Zhu, Y.; He, C.; Wang, Y.; Yan, S.; Tian, Y.; and Yuan, L. 2022. Spikformer: When spiking neural network meets transformer. arXiv:2209.15425.  
Zhu, R.-J.; Zhao, Q.; Li, G.; and Eshraghian, J. K. 2023. Spikegpt: Generative pre-trained language model with spiking neural networks. arXiv:2302.13939.  
