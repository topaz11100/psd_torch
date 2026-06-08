# 스파이킹 신경망은 고주파 정보가 필요하다(Spiking Neural Networks Need High-Frequency Information)

Yuetong Fang¹, Deming Zhou¹, Ziqing Wang⁴, Hongwei Ren¹, Zecui Zeng³, Lusong Li³, Shibo Zhou²∗, Renjing Xu¹∗

¹ 홍콩과학기술대학교(광저우)(The Hong Kong University of Science and Technology (Guangzhou))  
² Brain Mind Innovation INC  
³ JD Explore Academy  
⁴ 노스웨스턴 대학교(Northwestern University)

- yfang870@connect.hkust-gz.edu.cn  
- bob@brain-mind.com.cn  
- renjingxu@hkust-gz.edu.cn

∗ 교신저자(Corresponding author)

제39회 신경정보처리시스템 학회(Conference on Neural Information Processing Systems, NeurIPS 2025)

## 초록(Abstract)

스파이킹 신경망(Spiking Neural Networks, SNNs)은 이진(0/1) 스파이크(spike)를 통해 정보를 전달함으로써 뇌 영감(brain-inspired) 기반이면서 에너지 효율적인 계산을 약속한다. 그러나 성능은 여전히 인공 신경망(artificial neural networks, ANNs)에 비해 뒤처지며, 이는 종종 희소(sparse)하고 이진(binary)적인 활성(activation)로 인해 발생하는 정보 손실(information loss) 때문이라고 가정되어 왔다. 본 연구는 이러한 오랜 가정을 반박하고, 지금까지 간과되어 온 주파수 편향(frequency bias)을 드러낸다. 즉, 스파이킹 뉴런(spiking neuron)은 본질적으로 고주파(high-frequency) 성분을 억제하고 저주파(low-frequency) 정보를 우선적으로 전파한다. 우리는 이러한 주파수 영역 불균형(frequency-domain imbalance)이 SNN에서 특징 표현(feature representation)의 열화(degradation)를 유발하는 근본 원인이라고 주장한다.

실험적으로, 스파이킹 트랜스포머(Spiking Transformer)에서 토큰 믹싱(token mixing)에 평균 풀링(Avg-Pooling, 저역통과(low-pass))을 채택하면 CIFAR-100에서 성능이 76.73%로 낮아지지만, 이를 최대 풀링(Max-Pool, 고역통과(high-pass))으로 대체하면 top-1 정확도가 79.12%까지 상승한다. 이에 따라 우리는 두 가지 주파수 강화 연산자(frequency-enhancing operator)를 통해 고주파 신호를 복원하는 맥스-포머(Max-Former)를 제안한다: (1) 패치 임베딩(patch embedding) 단계의 추가 최대 풀링(Max-Pool), (2) 자기 어텐션(self-attention) 대신의 깊이별 합성곱(depth-wise convolution, DWC). 특히 Max-Former는 6,399만 개(63.99M) 파라미터만으로 ImageNet에서 82.39% top-1 정확도를 달성하여 Spikformer(74.81%, 66.34M) 대비 +7.58%p 향상된다. 또한 트랜스포머를 넘어 통찰을 확장하여, Max-ResNet-18은 합성곱 기반(convolution-based) 벤치마크에서 최첨단(state-of-the-art) 성능을 달성한다(CIFAR-10: 97.17%, CIFAR-100: 83.06%). 우리는 이 간단하지만 효과적인 해결책이 향후 연구가 스파이킹 신경망의 고유한 특성을 탐구하도록 영감을 주기를 바란다. 코드는 다음에서 제공된다: https://github.com/bic-L/MaxFormer.

그림 1: 스파이킹 트랜스포머(Spiking Transformer) 구조. 토큰 믹싱(token mixing)으로 (a) 평균 풀링(Avg-Pool)과 (b) 최대 풀링(Max-Pool)을 비교하며, (c) 스파이킹 MLP(Spiking MLP, S-MLP) 블록의 상세 구현을 제시한다. 주류(비-스파이킹) 비전 트랜스포머(Vision Transformer) 연구에서는 전역적인 저주파 패턴을 포착하는 Avg-Pool이 Max-Pool(고역통과(high-pass))보다 더 흔한 토큰 믹싱 전략이다[1, 2]. 놀랍게도 스파이킹 트랜스포머에서는 Avg-Pool을 Max-Pool로 대체하면 CIFAR-100에서 +2.39%p 성능 향상을 얻는다.

그림 2: ReLU(정류 선형 단위(rectified linear unit, ReLU))와 스파이킹 뉴런(spiking neuron, S-Neuron) 비교. (a) 입력 이미지; (b) 입력→활성→가중(weighting) 처리 흐름으로 산출된 출력 특징의 푸리에 스펙트럼(Fourier spectrum) 분석(고주파 영역은 표시: 빨간 점선 박스, 최대 진폭의 0.55배를 초과하는 영역) 및 (c) 해당 상대 로그 진폭(relative log amplitude); (d) [9]를 따른 동일한 구조 설정에서, 256 타임스텝(timestep)을 사용하는 변환된 스파이킹 트랜스포머(Spiking Transformer)의 Grad-CAM(Grad-CAM) 비교. 스파이킹 뉴런은 고주파 성분을 빠르게 소산(dissipation)시켜, 결과적으로 특징 표현이 열화된다.

## 1 소개(Introduction)

스파이킹 신경망(Spiking Neural Networks, SNNs)은 기존 인공 신경망(artificial neural networks, ANNs)의 에너지 효율적 대안으로 부상하고 있다[3, 4]. 이러한 효율성은, 스파이킹 뉴런(spiking neuron)이 시공간 동역학(spatiotemporal dynamics)을 활용해 인간 뇌의 생물학적 계산을 모사하는 데서 비롯된다[5]. ANNs에서는 동일한 층(layer)의 모든 뉴런이 실수(real-valued) 기반의 조밀한(dense) 텐서(tensor) 처리가 완료될 때까지 기다려야만 다음 층으로 정보가 흐를 수 있다. 반면 SNNs는 정보를 비동기(asynchronous)적으로 전달하며, 스파이킹 뉴런은 스파이크(“1”)를 수신/발신할 때만 에너지를 소비하고 그 외에는 비활성(inactive) 상태로 남는다[6, 7]. 이 이진 활성 패턴은, ANNs에서 핵심적인 에너지 집약적 곱셈-누산(multiply-and-accumulate, MAC) 연산을 훨씬 단순한 스파이크 기반 누산(accumulation)으로 대체할 수 있게 한다. 이러한 에너지 효율 이점을 바탕으로, 강력한 트랜스포머(Transformer) 구조를 스파이크 기반 계산과 결합한 스파이킹 트랜스포머(Spiking Transformer)와 같은 현대 SNN 변형들이 점차 주목을 받고 있다[8, 9].

에너지 효율성에도 불구하고, 스파이크 기반 계산의 이산적(discrete) 성격은 기회와 도전을 동시에 제공한다. SNN의 주요 장애물은 여전히 ANN 대비 성능 격차(performance gap)이다. 이 격차는 흔히 “표현 오류(representation error)”[10–12]로 설명되는데, 이는 이진 스파이크열(spike train)이 연속적 활성(continuous activation) 대비 특징 표현의 정밀도를 본질적으로 제한한다고 주장한다. 그러나 이는, 표준 딥러닝 문헌에서 저비트(low-bit) 심지어 이진(binary) 네트워크도 유사한 정확도를 달성할 수 있다는 널리 알려진 합의와 상충되는 것으로 보인다[13, 14]. 또한 SNN은 시간적 시퀀스(temporal sequence)에서 동작한다는 점에 주목해야 한다. 스파이킹 뉴런은 각 개별 타임스텝(time step)에서는 엄격히 이진 신호만 전달하지만, $n$개의 시뮬레이션 타임스텝에 걸친 스파이크열은 최소 $\log(n)$-비트(bit) 정밀도로 활성 값을 부호화(encode)할 수 있다[15, 16]. 이러한 상충되는 관찰은 SNN 성능 한계를 이해하는 데 있어 아직 탐구되지 않은 차원이 있음을 시사한다.

주파수 영역(frequency domain)에서 생각하는 것은 자연스럽다. 스파이킹 뉴런은 이산적이고 펄스 형태(pulse-like)의 활성을 생성하므로, 표준 네트워크에서 흔히 사용하는 연속적 활성 함수(continuous activation function)(예: ReLU[17])와는 근본적으로 다른 주파수 응답(frequency response)을 갖는다. 선행 연구들은 스파이킹 뉴런이 신호에 국소적(local) 세부(고주파)를 풍부하게 만들 수 있다고 제안해 왔다[18, 19]. 그러나 그림 2(b–c)에서 우리는 놀라운 현상을 관찰했다. 활성 함수 자체의 성질만을 보는 것이 아니라 입력→활성→가중이라는 종단간(end-to-end) 정보 흐름을 분석하면, 스파이킹 뉴런은 ReLU보다 저주파 정보를 더 두드러지게 전파하는 경향을 보인다.

SNN에서 관찰되는 특징 열화(feature degradation)는, 네트워크가 국소적이고 미세한(fine-grained) 정보를 효과적으로 포착하지 못하게 만드는 고주파 성분의 빠른 소산에서 기인할 수 있다(그림 2(d)). 이 발견을 뒷받침하기 위해, 우리는 비모수(non-parametric) 풀링 연산자, 즉 최대 풀링(Max-Pool)과 평균 풀링(Avg-Pool)을 스파이킹 트랜스포머의 토큰 믹서(token mixer)로 사용하는 간단한 실험을 수행한다(그림 1). 주파수 관점에서, Max-Pool은 국소적 고주파 디테일(예: 국소 에지(edge)/텍스처(texture))을 포착하는 데 강점이 있고, Avg-Pool은 전역적 저주파 패턴을 선호한다. 흥미롭게도 스파이킹 트랜스포머는 토큰 믹싱에서 ANN과 반대의 선호를 보인다. 표준 트랜스포머는 일반적으로 Avg-Pool을 토큰 믹싱에 사용하지만[1, 2], 스파이킹 트랜스포머에서 Avg-Pool을 Max-Pool로 바꾸면 CIFAR-100에서 +2.39%p 향상이 발생하며, 잘 튜닝된 Spikformer[8] 기준선을 0.97%p 상회한다.

요약하면, 본 연구는 “고주파 정보가 SNN에 필수적”이라는 관점을 뒷받침하는 이론적·실증적 증거를 제공한다.

- 네트워크 수준(network level)에서 스파이킹 뉴런이 본질적으로 저역통과(low-pass) 필터로 동작함을 최초로 이론적으로 증명하여, 고주파 특징을 억제하는 경향을 드러낸다.
- 스파이킹 트랜스포머에서 고주파 정보를 복원하는 맥스-포머(Max-Former)를 제안한다. 이는 두 가지 경량(lightweight) 모듈, 즉 패치 임베딩에서의 추가 Max-Pool과 초기 단계 자기 어텐션을 대체하는 깊이별 합성곱(DWC)을 사용한다.
- 고주파 정보 복원은 에너지 비용을 절감하면서도 성능을 크게 향상시킨다. ImageNet에서 Max-Former는 Spikformer 대비 +7.58%p 높은 82.39% top-1 정확도를 달성하며, 에너지 소비는 30% 수준이고 파라미터 수도 더 적다(63.99M vs. 66.34M).
- 이 통찰을 트랜스포머를 넘어 확장하면, Max-ResNet-18은 합성곱 기반 벤치마크에서 최첨단 성능을 달성한다(CIFAR-10: 97.17%, CIFAR-100: 83.06%).

우리는 이 직관적이면서도 강력한 해결책이, 표준 딥러닝에서 확립된 관행을 단순히 이식하는 것을 넘어 스파이킹 신경망의 고유한 특성을 탐구하도록 향후 연구를 촉진할 것이라 믿는다.

## 2 예비 지식 및 관련 연구(Preliminary and Related Works)

### 2.1 스파이킹 뉴런 모델(Spiking Neuron Models)

SNN은 생물학적 영감(biologically-inspired) 기반의 뉴런 모델(neuron model)을 통해 비선형 활성(non-linear activation)을 구현하며, 스파이크 구동(spike-driven) 처리를 수행한다. 누설 적분-발화(Leaky Integrate-and-Fire, LIF) 모델은 이러한 동작의 널리 사용되는 추상화이며, 생물학적 타당성(biological plausibility)과 계산 효율(computational efficiency) 사이에서 효과적인 균형을 제공한다[20]. 시뮬레이션 타임스텝 $n$에서의 이산화(discretized) LIF 모델은 다음과 같이 표현된다.

$$
U[n] = f(V[n-1], I[n]) \qquad (1)
$$

$$
S[n] = H\big(U[n] - V_{th}\big) \qquad (2)
$$

$$
V[n] =
\begin{cases}
U[n] - V_{th}, & S[n] = 1, \\
U[n], & S[n] = 0.
\end{cases}
\qquad (3)
$$

여기서 $\beta$는 감쇠 계수(decay factor), $V_{th}$는 발화 임계값(firing threshold), $H(\cdot)$는 스파이크 생성(spike generation)을 결정하는 헤비사이드 단계 함수(Heaviside step function)이다. 즉, $S[n] = H\big(U[n]-V_{th}\big) = 1$은 $U[n] \ge V_{th}$일 때이며, 그렇지 않으면 비활성($S[n]=0$) 상태로 남는다. LIF 뉴런의 충전(charging) 과정은 $f(\cdot)$로 결정되며 다음과 같다.

$$
f\big(V[n-1], I[n]\big) = \beta V[n-1] + (1-\beta)I[n] \qquad (4)
$$

각 타임스텝 $n$에서 현재 막 전위(membrane potential) $U[n]$는 입력 데이터 또는 Conv/MLP와 같은 중간 연산에 해당하는 시간영역 신호(time-domain signal) $I[n]$을 적분(integrate)하여 갱신된다. 만약 $U[n]$이 임계값 $V_{th}$를 초과하면 뉴런은 스파이크($S[n]=1$)를 발화(fire)한다. $V[n]$은 감쇠 계수 $\beta$와 출력 스파이크 활동(spike activity)이 주어졌을 때 시간에 따른 막 전위를 기록한다. 뉴런이 발화하지 않으면 $V[n]=U[n]$이다.

특히, 타임스텝 사이의 막 전위 감쇠 과정을 제거하면 LIF 모델은 적분-발화(integrate-and-fire, IF) 뉴런으로 단순화된다. 이 경우 충전 과정은 다음과 같다.

$$
f\big(V[n-1], I[n]\big) = V[n-1] + I[n] \qquad (5)
$$

### 2.2 스파이킹 신경망(Spiking Neural Networks)

생물학적 뉴런을 모방하여, SNN은 시간 동역학(temporal dynamics)과 이산 스파이크 기반 통신(discrete spike-based communication)을 통합함으로써 기존 ANNs를 확장한다[5]. 이 스파이크 구동 메커니즘을 활용해, 뉴로모픽 칩(neuromorphic chip)은 이벤트 구동(event-driven) 스파이크 라우팅(routing)과 누산을 통해 계산을 구현하며, 이는 에너지 집약적인 행렬–벡터 곱(matrix–vector multiplication)을 대체한다[7, 6]. 그 결과 높은 병렬성(parallelism), 확장성(scalability), 탁월한 전력 효율(power efficiency)을 달성할 수 있으며, 전력 소모는 보통 수십~수백 밀리와트(mW) 범위에 있다[21].

최근 스파이킹 트랜스포머(Spiking Transformer)와 같은 현대 SNN의 발전은 매력적인 성능과 낮은 에너지 소비를 동시에 보여주고 있다[8, 9, 22]. Spikformer[8]는 스파이크 기반 자기 어텐션(spike-based self-attention) 메커니즘인 스파이킹 자기 어텐션(Spiking Self Attention, SSA)을 개척했다. SSA는 희소한 스파이크 형태(spike-form)의 Query/Key/Value 벡터를 활용해 에너지 집약적인 softmax 연산을 제거한다. 그 성공 이후, 많은 연구가 고급 ANN 트랜스포머 구조를 스파이킹 트랜스포머에 적용하거나[23, 24], 표현 오류를 줄이기 위한 복잡한 스파이크 코딩(spike coding) 메커니즘(예: 다중 임계값(multi-threshold)[25], 다중 스파이크 뉴런(multi-spike neuron)[26, 27])을 고안하는 방향으로 발전해 왔다.

본 연구는 대신 더 근본적인 질문을 다룬다. 즉, ANN 대비 SNN의 성능을 진정으로 제한하는 것은 무엇인가? 우리의 조사 결과, 답은 주파수 특성(frequency properties)에 있었다. 구체적으로, 스파이킹 뉴런은 저역통과(low-pass) 필터로 기능하여 네트워크 내부에서 고주파 디테일의 전파를 방해한다.

## 3 방법(Methods)

본 절에서는 먼저 스파이킹 뉴런의 주파수 특성을 이론적으로 분석한다. 스파이킹 뉴런의 원시 출력 스파이크열(raw output spike train)은 임펄스 형태(spike waveform) 때문에 스펙트럼상 전대역(all-pass)처럼 보일 수 있다. 그러나 그로 인해 나타나는 고주파 성분은 표면적(superficial)이며, 네트워크를 통해 전파될 수 없다. 실제로 스파이킹 뉴런은 네트워크 수준에서 저역통과 필터로 동작한다. 이는 선행 연구들에서 간과되어 온 근본 문제이다. 이 통찰을 바탕으로, 우리는 맥스-포머(Max-Former)를 통해 SNN에서 고주파 정보의 중요성을 검증한다. 맥스-포머는 고역통과(high-pass) 연산자(최대 풀링(Max-Pool)과 깊이별 합성곱(DWC))를 전략적으로 사용하여 고주파 디테일을 복원하고 특징 열화를 방지한다.

그림 3: ReLU와 스파이킹 뉴런의 시간-주파수(time-frequency) 분석. (a) 시간영역 신호: 입력 $x(t)=\frac{1}{3}(\sin(2\pi\cdot100t)+\sin(2\pi\cdot200t)+\sin(2\pi\cdot300t))$ (파랑), ReLU 처리 결과 $r(t)$ (빨강), $\beta=0.25$인 LIF 뉴런의 스파이킹 출력 $s(t)$ (초록). (b) $x(t)$, $r(t)$, $s(t)$에 대한 푸리에 분석(Fourier analysis). (c) 선형 변환(CONV/MLP)된 활성에 대한 푸리에 분석: ReLU는 입력 신호의 주파수 대역폭(bandwidth)을 확장하는 반면, 스파이킹 뉴런은 고주파 감쇠(attenuation)를 보인다.

### 3.1 스파이킹 뉴런은 저역통과 필터이다(Spiking Neurons are Low-pass Filters)

우리는 그림 3에 제시된 입력 $x(t)=\frac{1}{3}(\sin(2\pi\cdot100t)+\sin(2\pi\cdot200t)+\sin(2\pi\cdot300t))$을 사용해 직관적인 시간-주파수 분석을 수행한다. 결과는 세 가지 핵심 관찰을 보여준다. (1) 시간영역에서 ReLU 출력 $r(t)$는 $x(t)>0$을 완벽히 추종하지만, 스파이킹 뉴런은 선택적으로 100Hz에 반응한다(그림 3(a)). (2) 그럼에도 스파이킹 출력의 스펙트럼 응답 $|S(f)|$는 여전히 거의 전대역(all-pass)처럼 보이는데, 이는 시간영역에서 관찰된 저주파 거동과 모순된다. 이러한 허위(spurious) 고주파 성분은 실제 신호로부터 나온 것이 아니라 임펄스 형태의 스파이크 파형 자체에서 비롯된다(그림 3(a–b)). (3) 파형 유발(waveform-induced) 고주파 성분은 층을 넘어 전파될 수 없으며, 결과적으로 네트워크 수준에서 저역통과 거동을 만든다. 입력→활성→선형 변환의 전체 과정을 고려하면, ReLU는 $x(t)$의 주파수 대역폭을 확장하는 반면[28], 스파이킹 뉴런은 강한 고주파 감쇠를 보인다(그림 3(c)).

이제 스파이킹 뉴런의 충전 과정을 분석하여 주파수 선택적(frequency-selective) 특성을 이론적으로 살펴본다. 식 (3)과 (4)로부터 다음을 얻는다.

$$
V[n] = \beta V[n-1] + (1-\beta)I[n] \qquad (6)
$$

$Z\{V[n-1]\}=z^{-1}V(z)$를 사용하여 Z-변환(Z-transform)을 적용하면,

$$
V(z) = \beta z^{-1} V(z) + (1-\beta)I(z) \qquad (7)
$$

이를 정리하면, 입력 전류(input current)에서 막 전위로의 전달 함수(transfer function)는 다음과 같다.

$$
H(z) = \frac{V(z)}{I(z)} = \frac{1-\beta}{1-\beta z^{-1}},\quad 0\le\beta<1 \qquad (8)
$$

식 (8)은 단일 폴(pole) $z=\beta$를 갖는 1차 무한 임펄스 응답(infinite-impulse-response, IIR) 저역통과 필터의 정확한 형태이다. 따라서 $\beta$ (즉, 폴)가 1에 가까워질수록 LIF 뉴런은 더 강한 저주파 선택성(low-frequency selectivity)을 보인다.

또한 감쇠 계수 $\beta$는 막 시간상수(membrane time constant) $\tau$와 $\beta = 1 - \frac{1}{\tau}$로 연관되며, $\tau\in[1, +\infty)$에서 $\tau$가 작을수록 $\beta$도 작아진다. 직관적으로, 더 짧은 시간상수는 막 전위가 더 좁은 시간 창(window)에서 반응하도록 하여, 뉴런이 더 높은 주파수 입력에 민감하게 만든다.

개별 뉴런에서 네트워크 수준 정보 전달로 확장하면, 평균 막 전위(average membrane potential)는 동작 중 스파이크 발화 가능성과 양(+)의 일관된 상관을 갖는다. 우리는 본질적으로 비선형(non-linear)인 스파이크 생성 과정을 발화 임계값 $V_{th}$ 근방에서 선형(linear)으로 근사한다. 발화율(firing rate)을 $f_r(V)$로 두고, 국소 이득(local gain) $k$를 정의하여 Z-영역 스파이크열 $S(z)$를 다음과 같이 근사한다.

$$
k = \left.\frac{\partial f_r}{\partial V}\right|_{V=V_{th}} \qquad (9)
$$

$$
S(z) \approx k\,V(z) \qquad (10)
$$

출력 스파이크열이 인과적 시냅스 커널(causal synaptic kernel) $w[n]$으로 가중될 때, Z-변환된 출력 전류(output current)는 $y[n]=w[n]*s[n]$이며 $Y(z)=W(z)S(z)$로 쓸 수 있다. 식 (8), (10)을 결합하면 입력-출력 전달 함수는 다음과 같다.

$$
H'(z) = \frac{Y(z)}{I(z)} = S(z)W(z)H(z) = k\,W(z)\,\frac{1-\beta}{1-\beta z^{-1}} \qquad (11)
$$

$H(z)$의 1차 IIR 저역통과 특성은, 시냅스 커널 $W(z)$ 또는 스파이크 코딩 과정 $S(z)$가 이득/위상 응답을 바꾸더라도 시스템 $Y(z)$가 본질적으로 저주파 성분을 선호하도록 만든다. 또한 저역통과 항 $\left(\frac{1-\beta}{1-\beta z^{-1}}\right)^L$은 과정 $H'(z)$가 $L$번(층 수) 연쇄(cascade)될 때 시스템의 주파수 선택성을 더 증폭시킨다. 전체 식은 다음과 같다.

$$
H'_L(z) = \frac{Y_L(z)}{I(z)} = \prod_{i=1}^{L}\big(S_i(z)W_i(z)H(z)\big)
= \left(\prod_{i=1}^{L} k_i W_i(z)\right)\left(\frac{1-\beta}{1-\beta z^{-1}}\right)^L \qquad (12)
$$

누설이 없는(non-leaky) IF 뉴런(식 (5))의 특수한 경우에는 $H(z)$가 다음과 같다.

$$
H(z) = \frac{1}{1-z^{-1}} \qquad (13)
$$

이는 $z=1$에 폴을 갖는 이상적(ideal) 이산시간(discrete-time) 저역통과 필터에 해당하며, 앞선 분석과 일관된 결론을 준다.

### 3.2 맥스-포머(Max-Former)

고주파 정보가 SNN에 정말로 중요하며, 이를 복원하면 성능이 향상되는지는 여전히 자명하지 않다. 따라서 우리는 맥스-포머(Max-Former)를 통해 스파이킹 뉴런의 저역통과 필터링 특성을 체계적으로 조사한다. 주파수 효과를 모델 복잡도(model complexity)와 분리하기 위해, (1) 초기 단계에서는 자기 어텐션을 고주파 보존에 유리한 DWC로 대체하고, (2) 패치 임베딩에 Max-Pool을 추가하여 스파이킹 뉴런의 저역통과 선호를 보상한다. 또한 자기 어텐션의 이차(quadratic) 계산 복잡도에 비해, DWC와 Max-Pool은 시퀀스 길이에 대해 선형(linear) 복잡도만 필요하며 파라미터 효율(parameter-efficient)도 높다. 본 연구 전반에서 우리는 LIF 뉴런 모델을 일관되게 사용한다.

#### 3.2.1 전체 구조(Overall Architecture)

그림 4(a)는 맥스-포머(Max-Former)의 전체 프레임워크(framework)를 보여준다. 구조는 3단계(stage)로 구성되며, 각 단계의 토큰 수는 각각 $\frac{H}{4}\times\frac{W}{4}$, $\frac{H}{8}\times\frac{W}{8}$, $\frac{H}{16}\times\frac{W}{16}$이다. 여기서 $H$와 $W$는 입력 이미지의 높이(height)와 너비(width)를 의미한다. 핵심적으로 Max-Former는 시간에 따라 이산 스파이크(discrete spike)를 통해 정보를 처리한다. 이 스파이크 구동(spike-driven) 계산 패러다임은 두 종류의 입력을 지원한다.

1) 이벤트 스트림(Event Streams)  
비동기 이벤트(asynchronous event) $e=[x,y,t,p]$는 공간 좌표 $(x,y)$, 타임스탬프(timestamp) $t$, 극성(polarity) $p$를 포함하며, 시간적 비닝(temporal binning)을 통해 이벤트 프레임(event frame)으로 변환된다. 원 해상도 $d_{to}$와 목표 해상도 $d_t=\alpha d_{to}$가 주어지면, 이벤트는 연속된 $\alpha$개의 빈(bin)에 걸쳐 집계(aggregate)된다.

$$
I_t = \sum_{k=\alpha t}^{\alpha(t+1)-1} S_k \in \mathbb{R}^{2\times h\times w} \qquad (14)
$$

여기서 $S_k$는 원시 이벤트(raw event) 데이터를 의미한다. 이 과정은 원시 이벤트를 잡음 제거(denoise)하고, 목표 시간 해상도에서 프레임 시퀀스(frame sequence)로 변환한다.

2) 정적 이미지(Static Images)  
일반적인 이미지는 다음으로 스파이크 시퀀스로 변환된다: (i) 정적 프레임(static frame)을 $T$번 반복, (ii) 스파이킹 뉴런으로 픽셀 강도(pixel intensity)를 스파이크로 부호화. 결과 입력은 다음과 같이 표현된다.

$ I = \mathrm{Spiking\_Embed}(\{I_t\}_{t=1}^{T}) $

이는 모든 타임스텝에 동일한 정보를 포함한다.

그림 4: (a) 맥스-포머(Max-Former) 개요. 초기 단계에서 자기 어텐션(self-attention) 대신 경량 깊이별 합성곱(lightweight DWC)을 사용하여 고주파 신호를 복원한다. [29]의 계층적(hierarchical) 설계를 따라 Max-Former는 3단계 구조를 채택한다. $D_i$는 $i$번째 단계(stage-$i$)의 특징 차원(feature dimension)이다. (b) 맥스-포머의 패치 임베딩(patch embedding) 단계에서, 고주파 성분을 강화하기 위해 세 가지 구성(Embed-Orig, Embed-Max, Embed-Max+)을 제안한다.

#### 3.2.2 패치 임베딩(Patch Embedding)

입력을 토큰화(tokenize)된 표현으로 변환하기 위해, 입력 $\{S\}\in\mathbb{R}^{T\times C\times H\times W}$가 주어졌을 때 패치 임베딩 과정은 다음과 같다.

$$
Y = G_1(\{S\}) + G_2(\{S\}),\quad Y\in\mathbb{R}^{T\times C'\times H'\times W'} \qquad (15)
$$

여기서 $C'=2C$, 패치 크기(patch size) $P=4$에 대해 $H'=\lfloor H/P\rfloor$, $W'=\lfloor W/P\rfloor$이다. 스파이킹 뉴런의 고유한 주파수 선호를 보완하기 위해, 그림 4(b)와 같이 세 가지 패치 임베딩 구성을 제시한다.

$$
\text{Embed-Orig}:\ (G_1,G_2)=(\text{Embed},\ \text{Embed}) \qquad (16)
$$

$$
\text{Embed-Max}:\ (G_1,G_2)=(\text{Max-Embed},\ \text{Embed}) \qquad (17)
$$

$$
\text{Embed-Max+}:\ (G_1,G_2)=(\text{Max-Embed},\ \text{Max-Embed}) \qquad (18)
$$

여기서 $\text{Embed}\equiv\{\text{LIF}-\text{CONV}-\text{BN}\}$, $\text{Max-Embed}\equiv\{\text{LIF}-\text{CONV}-\text{BN}-\text{MaxPool}\}$이다.

#### 3.2.3 토큰 믹싱(Token Mixing)

트랜스포머(Transformer)에서 일반적으로 하위 층(lower layer)은 더 많은 고주파 디테일을 필요로 하고, 상위 층(higher layer)은 더 전역적인 정보에서 이득을 얻는다[30, 31]. 생물학적 시각(biological vision)과 유사하게, 고주파 디테일은 초기 단계에서 저수준(low-level) 특징을 학습하고 이후 국소→전역(local-to-global) 표현을 점진적으로 구축하는 데 기여한다. 이에 따라 우리는 초기 단계 자기 어텐션을 DWC로 대체하여, 국소 특징 학습에 필수적인 고주파를 보존한다. 입력 임베딩 $Y\in\mathbb{R}^{T\times C\times H\times W}$가 주어졌을 때 스파이킹 DWC는 다음과 같이 정의된다.

$$
Z_c(Y)[i] = \mathrm{LIF}\left(\sum_{j\in\Omega(i)} w_{c,j}\cdot Y_c[j]\right) \qquad (19)
$$

여기서 $\Omega(i)$는 위치 $i$의 국소 이웃(local neighborhood), $w_{c,j}$는 채널 $c$에 대한 학습 가능한 합성곱 가중치(convolution weight)이며, $X_c, Z_c\in\mathbb{R}^{T\times H\times W}$는 채널 $c$의 입력/출력 슬라이스(slice)이다. 마지막 단계에서는 스파이킹 자기 어텐션(Spiking Self Attention, SSA)[8]을 사용해 토큰 믹싱을 수행한다. SSA 계산은 다음과 같다.

$$
Z = \mathrm{LIF}(\mathrm{BN}(Y W)),\quad Z\in\{Q,K,V\} \qquad (20)
$$

$$
\mathrm{SSA}(Q,K,V) = \mathrm{LIF}(QK^{\mathsf{T}}V\cdot s) \qquad (21)
$$

여기서 $Q,K,V\in\mathbb{R}^{T\times N\times H\times W}$는 학습 가능한 선형 층(linear layer)으로부터 생성된 스파이크 형태 텐서이며, $s$는 스케일링 계수(scaling factor)이다. SSA는 부동소수점 곱셈을 제거하여, 스파이크 구동 호환성(spike-driven compatibility)을 보장한다.

### 3.3 멤브레인 쇼트컷(Membrane Shortcut)

그림 5: SNN에서의 쇼트컷(shortcut) 연결. (왼쪽) 스파이크와 막 전위를 결합하는 바닐라 쇼트컷(Vanilla Shortcut). (가운데) 뉴런 충전 전에 스파이크 신호를 더하는 프리-스파이크 쇼트컷(Pre-Spike Shortcut). (오른쪽) 막 전위를 직접 연결하는 멤브레인 쇼트컷(Membrane Shortcut)으로, 동일한 전위 매핑(potential mapping)을 보장하면서 네트워크 전반에서 스파이크 구동 계산 패러다임을 엄격히 유지한다.

잔차 학습(residual learning)과 쇼트컷(shortcut)은 정보 흐름을 보존하고 그래디언트 소실(vanishing gradient)을 완화하는 항등(Identity) 경로를 제공함으로써 매우 깊은 네트워크(deep network)의 학습을 가능하게 한다[32]. SNN에서는 모든 연산이 스파이크 구동 계산 패러다임과 호환되도록 유지하는 것이 중요하다. 그림 5에서 보이듯, 기존 SNN 쇼트컷 구현은 (1) 바닐라 쇼트컷(Vanilla Shortcut)[33], (2) 프리-스파이크 쇼트컷(Pre-Spike Shortcut)[34], (3) 멤브레인 쇼트컷(Membrane Shortcut)[35]의 세 범주로 분류된다.

바닐라 쇼트컷[33]은 스파이크(이진)와 막 전위(연속)를 직접 연결하므로 분포 불일치(distribution mismatch)를 유발하여 항등 매핑(identity mapping) 원리를 본질적으로 위반한다. 프리-스파이크 쇼트컷[34]은 뉴런 충전 전에 스파이크 신호를 더해 합(sum)이 0~2 범위를 갖게 만들며, 이는 이진 스파이크 표현과 SNN의 스파이크 구동 흐름을 교란한다. 본 연구에서는 두 가지 장점을 가진 멤브레인 쇼트컷[35]을 채택한다. 멤브레인 쇼트컷은 막 전위를 직접 연결하여 항등 매핑을 보존하면서도, 출력은 이진 스파이크로 유지되어 스파이크 구동 계산과 본질적으로 호환된다. 바닐라/프리-스파이크 방식과 달리, 이 접근은 잔차 학습 원리와의 수학적 일관성(mathematical consistency)과 스파이크 구동 연산과의 매끄러운 호환성을 동시에 보장한다. 그 영향(성능 및 에너지 비용)에 대한 상세 분석은 부록 D에 제시한다.

## 4 실험(Experiment)

그림 6에서 보이듯, 맥스-포머(Max-Former)는 주파수 강화 연산자(frequency-enhancing operator)인 Max-Pool과 DWC의 장점을 결합하여 고주파 정보를 복원한다. 스파이킹 트랜스포머에서 고주파 정보의 중요성을 실증적으로 평가하기 위해, 우리는 정적(static) 데이터셋(CIFAR-10[36], CIFAR-100[37], ImageNet[38])과 뉴로모픽(neuromorphic) 데이터셋(CIFAR10-DVS[39], DVS128 Gesture[40])에서 포괄적인 실험을 수행하며, 아키텍처 구성은 표 1에 정리한다. 또한 합성곱 구조에서 고주파 복원의 효과를 추가로 조사하기 위해 Max-ResNet을 설계한다(구현은 부록 A.3). 실험 설정과 에너지 분석 방법은 부록 A와 B에 제시한다.

그림 6: 스파이킹 뉴런(Spiking Neuron), 스파이킹 최대 풀링(Spiking Max-Pool), 스파이킹 깊이별 합성곱(Spiking Depth-Wise Convolution), 스파이킹 자기 어텐션(Spiking Self-attention)의 푸리에 스펙트럼(Fourier spectrum).

표 1: 분류(classification) 과제별 맥스-포머(Max-Former) 아키텍처 구성. 표기 DWC-$N$은 커널 크기 $N\times N$의 깊이별 합성곱(depth-wise convolution)을 의미한다. 블록(block) 설정: CIFAR-10/100은 3단계(1/1/2 블록), ImageNet은 3단계(1/3/7 블록), 뉴로모픽은 2단계(1/1 블록).

| 데이터셋(Dataset) | Stage 1 패치 임베딩(Patch Embed) | Stage 1 토큰 믹싱(Token Mix) | Stage 2 패치 임베딩(Patch Embed) | Stage 2 토큰 믹싱(Token Mix) | Stage 3 패치 임베딩(Patch Embed) | Stage 3 토큰 믹싱(Token Mix) |
|---|---|---|---|---|---|---|
| CIFAR10[36]/100[37] | Embed-Orig | Identity | Embed-Max | DWC-3 | Embed-Max | SSA |
| ImageNet[38] | Embed-Orig | DWD-7 | Embed-Max | DWC-5 | Embed-Max | SSA |
| Neuromorphic[39, 40] | Embed-Max+ | DWC-3 | Embed-Max | SSA | — | — |

### 4.1 CIFAR 및 뉴로모픽 데이터셋 결과(Results on CIFAR and Neuromorphic Datasets)

표 2에서 보이듯, 맥스-포머(Max-Former)는 정적 데이터셋(CIFAR10/CIFAR100)과 뉴로모픽 데이터셋(DVS128/CIFAR10-DVS) 모두에서 성능 향상을 제공한다. 특히 CIFAR10/100 분류에서 1단계(stage 1)는 토큰 믹싱에 항등 매핑(identity mapping)만 사용함에도(표 1) 매력적인 결과를 달성한다. 맥스-포머는 $T=4$에서 657만(6.57M) 파라미터만으로 CIFAR10에서 97.04% 정확도를 달성하여, Spikformer(95.51%, 9.32M), S-Transformer(95.60%, 10.28M), QKFormer(96.18%, 6.74M)를 상회한다. CIFAR100에서도 82.65%를 달성하여 Spikformer(78.21%), S-Transformer(78.40%), QKFormer(81.57%)보다 크게 높다.

맥스-포머와 QKFormer는 유사한 계층적(hierarchical) 구조를 공유하지만, QKFormer는 원래 프리-스파이크 쇼트컷(pre-spike shortcut)[24]을 사용한다. 공정 비교를 위해, 우리는 동일한 학습 구성으로 멤브레인 쇼트컷(Membrane Shortcut)을 사용하는 QKFormer 변형을 추가 구현했으며(표에서 MS-QKFormer), 맥스-포머는 CIFAR10에서 0.2%p(97.04% vs. 96.84%), CIFAR100에서 1.08%p(82.65% vs. 81.57%) 더 높고, 파라미터 수도 약간 더 적다(6.57M vs. 6.74M). 뉴로모픽 데이터셋에서도 이 우위는 유지된다. DVS128에서는 멤브레인 쇼트컷을 사용한 MS-QKFormer와 동일하게 98.6%를 달성한다. CIFAR10-DVS에서는 84.2%로 MS-QKFormer(82.3%)보다 1.9%p 높고, S-Transformer(80.0%), SWformer(83.9%) 등 다른 스파이크 구동 모델을 상회한다.

표 2: CIFAR10[36], CIFAR100[37], DVS128[40], CIFAR10-DVS[39] 성능 비교. Param.: 파라미터 수(백만, M), Acc.: top-1 정확도(%), $T$: 시뮬레이션 타임스텝(simulation timestep). *는 동일한 설정으로 스크래치(scratch)부터 학습한 모델.

| 방법(Method) | CIFAR10 Param.(M) | CIFAR10 $T$ | CIFAR10 Acc.(%) | CIFAR100 Param.(M) | CIFAR100 $T$ | CIFAR100 Acc.(%) | DVS128 Param.(M) | DVS128 $T$ | DVS128 Acc.(%) | CIFAR10-DVS Param.(M) | CIFAR10-DVS $T$ | CIFAR10-DVS Acc.(%) | 멤브레인 쇼트컷(Membrane Shortcut) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| ResNet-19 (ANN)[24] | 12.63 | 1 | 94.97 | 12.63 | 1 | 75.35 | — | — | — | — | — | — | — |
| Max-Former (ANN) | 6.57 | 1 | 96.82 | 6.60 | 1 | 82.41 | — | — | — | — | — | — | — |
| Spikformer[8] | 9.32 | 4 | 95.51 | 9.32 | 4 | 78.21 | 2.57 | 16 | 98.3 | 2.57 | 16 | 80.9 | ✗ |
| S-Transformer[41] | 10.28 | 4 | 95.60 | 10.28 | 4 | 78.40 | 2.57 | 16 | 99.3 | 2.57 | 16 | 80.0 | ✓ |
| SWformer[19] | 7.51 | 4 | 96.10 | 7.51 | 4 | 79.30 | - | - | - | 2.05 | 16 | 83.9 | ✓ |
| QKFormer[24] | 6.74 | 4 | 96.18 | 6.74 | 4 | 81.15 | 1.50 | 16 | 98.6 | 1.50 | 16 | 84.0 | ✗ |
| MS-QKFormer* | 6.74 | 4 | 96.84 | 6.74 | 4 | 81.57 | 1.50 | 16 | 98.6 | 1.50 | 16 | 82.3 | ✓ |
| Max-Former* | 6.57 | 4 | 97.04 | 6.60 | 4 | 82.65 | 1.45 | 16 | 98.6 | 1.45 | 16 | 84.2 | ✓ |

표 3: ImageNet 성능 비교. 표기: A2S는 ANN→SNN 변환(ANN-to-SNN conversion), Model-$L$-$D$는 $L$개 블록(block)과 $D$개 채널(channel)을 갖는 모델. 입력 해상도는 224×224. *는 동일한 학습 구성.

| 방법(Methods) | 유형(Type) | 아키텍처(Architecture) | Param.(M) | 전력/에너지(Power, mJ) | 타임스텝(Time Step) | Top-1 Acc.(%) | 멤브레인 쇼트컷(Membrane Shortcut) |
|---|---|---|---:|---:|---:|---:|:---:|
| ViT[42] | ANN | ViT-L/16 | 304.3 | 80.96 | 1 | 79.70 | — |
| DeiT[43] | ANN | DeiT-B | 86.6 | 80.50 | 1 | 81.80 | — |
| PVT[44] | ANN | PVT-Large | 61.4 | 45.08 | 1 | 81.70 | — |
| MST[9] | A2S | Swin Transformer-T | 28.50 | — | 512 | 78.51 | ✗ |
| Spikformer[8] | SNN | Spikformer-8-384 | 16.81 | 7.73 | 4 | 70.24 | ✗ |
| Spikformer-8-768 | SNN | — | 66.34 | 21.48 | 4 | 74.81 | ✗ |
| S-Transformer[41] | SNN | S-Transformer-8-384 | 16.81 | 3.90 | 4 | 72.28 | ✓ |
| S-Transformer-8-768 | SNN | — | 66.34 | 6.10 | 4 | 76.30 | ✓ |
| Meta-Spikformer[45] | SNN | — | 31.3 | 7.80 | 1 | 75.4 | ✓ |
| Meta-Spikformer[45] | SNN | — | 31.3 | 32.80 | 4 | 77.2 | ✓ |
| SWformer[19] | SNN | SWformer-8-512 | 27.6 | 5.08 | 4 | 75.43 | ✓ |
| QKFormer[24] | SNN | HST-10-384 | 16.47 | 15.13 | 4 | 78.80 | ✗ |
| QKFormer[24] | SNN | HST-10-768 | 64.96 | 8.52 | 1 | 81.69 | ✗ |
| MS-QKFormer* | SNN | HST-10-384 | 16.47 | 5.52 | 4 | 76.48 | ✓ |
| MS-QKFormer* | SNN | HST-10-768 | 64.96 | 6.79 | 1 | 77.78 | ✓ |
| Max-Former* | SNN | Max-10-384 | 16.23 | 4.89 | 4 | 77.82 | ✓ |
| Max-Former* | SNN | Max-10-512 | 28.65 | 2.50 | 1 | 75.47 | ✓ |
| Max-Former* | SNN | Max-10-512 | 28.65 | 7.49 | 4 | 79.86 | ✓ |
| Max-Former* | SNN | Max-10-768 | 63.99 | 5.27 | 1 | 78.60 | ✓ |
| Max-Former* | SNN | Max-10-768 | 63.99 | 14.87 | 4 | 82.39 | ✓ |

### 4.2 ImageNet 분류 결과(Results on ImageNet Classification)

표 3은 ImageNet 분류에서 맥스-포머(Max-Former)의 성능을 보여주며, 복잡한 시각 과제에 대한 효과성을 입증한다. Max-Former-10-768($T=4$)은 초기 단계 토큰 믹싱에서 경량(lightweight) 연산만 사용함에도, Spikformer 대비 +7.58%p 높은 82.39% 정확도를 달성하고 에너지는 30% 낮다(14.87mJ vs. 21.48mJ). 또한 512 타임스텝을 요구하는 ANN→SNN 변환 기반 MST 모델(78.51%)보다도 높다. 학습/추론 속도와 메모리 사용은 부록 C에서 분석한다.

우리의 분석은 멤브레인 쇼트컷(Membrane Shortcut)을 사용하는 모델에 초점을 둔다. 이는 프리-스파이크 쇼트컷에서 발생하는 3값(ternary) 스파이크 전송({0,1,2})을 제거하여 에너지 비효율을 낮추면서도, 추가 하드웨어 오버헤드 없이 표준 뉴로모픽 하드웨어와의 완전한 호환성을 유지한다(부록 D 참고). 공정 비교를 위해 우리는 QKFormer의 막 전위(memebrane potential) 변형인 MS-QKFormer를 구현했다. HST-10-384 설정에서 MS-QKFormer는 원본 QKFormer 대비 64% 낮은 에너지(5.52mJ vs. 15.13mJ)를 보인다.

Max-Former-10-384(16.23M, $T=4$)는 77.82%로 MS-QKFormer-10-384(16.47M, 76.48%), S-Transformer-8-768(66.34M, 76.3%), Meta-Spikformer(31.3M, 77.2%)보다 높다. 동일 조건의 에너지 효율을 보면, Max-Former-10-384는 4.89mJ를 소모하여 MS-QKFormer(5.52mJ), S-Transformer(6.10mJ), Meta-Spikformer(32.8mJ)보다 낮다. 전통적인 ANN 모델과 비교해도, Max-Former는 경쟁력 있는 정확도를 유지하면서 구체적인 에너지 효율 이점을 보여준다. 예를 들어 계층적 ANN의 대표격인 PVT-Large와 비교하면, Max-Former-10-768($T=4$)은 유사한 정확도(82.34% vs. 81.70%)를 달성하면서도 에너지는 67% 낮다(14.87mJ vs. 45.08mJ). 이러한 결과는 스파이킹 트랜스포머에서 고주파 정보가 중요함을 확인한다. 즉, 에너지 집약적인 자기 어텐션을 초기 단계에서 경량 DWC로 대체하는 것이 오히려 더 나은 성능을 만든다.

표 4: CIFAR100과 CIFAR10-DVS에서 패치 임베딩/토큰 믹싱 전략에 대한 소거(어블레이션) 결과.

| CIFAR100 패치 임베딩(Patch Embed) | CIFAR100 토큰 믹싱(Token Mix) | CIFAR100 Acc.(%) | CIFAR10-DVS 패치 임베딩(Patch Embed) | CIFAR10-DVS 토큰 믹싱(Token Mix) | CIFAR10-DVS Acc.(%) |
|---|---|---:|---|---|---:|
| Orig/Max/Max | Identity/DWC-3/SSA | 82.65 | Max+/Max | DWC-3/SSA | 84.2 |
| Orig/Orig/Orig | Identity/DWC-3/SSA | 81.63 | Orig/Orig | DWC-3/SSA | 79.2 |
| Orig/Max/Orig | Identity/DWC-3/SSA | 81.88 | Orig/Max | DWC-3/SSA | 81.5 |
| Orig/Max/Max | Identity/Identity/SSA | 81.28 | Max+/Max | DWC-1/SSA | 81.2 |
| Orig/Max/Max | Identity/DWC-5/SSA | 82.02 | Max+/Max | DWC-5/SSA | 82.7 |
| Orig/Max/Max | DWC-7/DWC-5/SSA | 82.42 | Max+/Max | DWC-7/SSA | 82.1 |
| Orig/Max/Max | SSA/SSA/SSA | 82.23 | Max+/Max | SSA/SSA | 83.9 |
| Orig/Orig/Orig | SSA/SSA/SSA | 81.43 | Orig/Orig | SSA/SSA | 79.8 |

### 4.3 소거(어블레이션) 연구(Ablation Study)

우리는 심층 소거 연구(in-depth ablation)를 통해 스파이킹 트랜스포머에서 고주파 정보가 갖는 핵심적 역할을 직접적으로 입증한다.

1) 패치 임베딩 전략 소거  
적절한 패치 임베딩 전략은 스파이킹 트랜스포머의 성능 한계를 끌어올리는 데 도움이 된다. CIFAR100에서 맥스-포머의 패치 임베딩을 제안된 Embed-Orig/Embed-Max/Embed-Max에서 기본 Embed-Orig/Embed-Orig/Embed-Orig로 바꾸면 정확도가 82.65%에서 81.63%로 하락한다. 뉴로모픽 데이터셋은 고주파 성분에 더 강한 의존성을 보인다. CIFAR10-DVS에서 1단계의 Embed-Max+를 Embed-Orig로 바꾸면 84.2%에서 81.5%로 크게 떨어진다. 즉, 패치 임베딩을 통한 고주파 정보 주입은 효과적이다. 순수 SSA 토큰 믹싱만 사용하는 경우에도, 패치 임베딩 전략 최적화는 CIFAR100에서 +0.8%p, CIFAR10-DVS에서 +4.1%p 향상을 제공하며, 스파이킹 구조에서 그 중요성을 강조한다.

2) 토큰 믹싱 전략 소거  
맥스-포머는 토큰 믹싱 전략을 통해 고주파 정보를 추가로 복원한다. SSA가 더 높은 에너지/파라미터 비용을 갖더라도, 맥스-포머는 초기 단계 자기 어텐션을 경량 DWC로 대체하는 것만으로 더 나은 성능을 달성한다. CIFAR100에서 이 대체는 +0.42%p 성능 향상(전 단계 SSA: 82.23% vs. 맥스-포머: 82.65%)으로 이어진다. CIFAR10-DVS에서 하이브리드 토큰 믹싱(DWC-3 + SSA)은 84.2%로, 전체 SSA 변형(83.9%)보다 0.3%p 높다.

고주파 보존을 위한 커널 크기(kernel size) 선택 역시 중요하다. 더 큰 커널(DWC-5/7)은 과도한 평활화(smoothing)로 인해 성능을 저하시킨다(CIFAR100: -0.63%p, CIFAR10-DVS: -2.1%p). 반대로 DWC-1처럼 필터링이 부족하면 성능이 떨어진다(CIFAR10-DVS: 81.2% vs. 84.2%). 이는 스파이킹 트랜스포머에서 고주파/저주파 성분의 균형이 필요함을 보여준다.

종합하면, 맥스-포머의 성능 향상은 파라미터 수와 무관하게 효과적인 고주파 정보 전파에서 비롯되며, 다음으로 뒷받침된다. (1) Max-Pool 기반 패치 임베딩(Embed-Max/Embed-Max+)은 유사한 파라미터 예산에서도 원래 버전보다 일관되게 더 낫다. (2) 더 큰 DWC 커널(DWC-5/DWC-7)은 파라미터를 늘리지만 정확도는 낮춘다(CIFAR100: -0.63%p, CIFAR10-DVS: -2.1%p; DWC-3 기반 맥스-포머 대비). 시각화는 부록 D, 한계 논의는 부록 E에서 다룬다.

### 4.4 합성곱 구조 전반에서의 일반성(Generality across Convolutional Architectures)

우리는 Max-ResNet을 제안하여 맥스-포머의 효과를 합성곱 구조(convolutional architecture)로 확장한다. Max-ResNet의 핵심 수정은 MS-ResNet[46] 대비 단 두 번의 추가 max-pooling 연산을 포함하는 것이다. Max-ResNet의 상세 구현은 부록 A.3에 제공되며, 학습 설정은 부록 A에 제시한다.

고주파 정보는 SNN에 필수적이다. 표 5에서 보이듯, Max-ResNet은 동일한 모델 크기에서도 MS-ResNet 기준선 대비 현저한 성능 향상을 달성한다. 구체적으로 블록 구성(block configuration) [2,2,2,2]에서 Max-ResNet은 CIFAR-10 정확도를 +2.41%p(94.4%→96.81%), CIFAR-100 정확도를 +6.48%p 향상시킨다. [3,3,2] 구성에서도 CIFAR-10과 CIFAR-100에서 각각 +2.25%p, +6.65%p 향상이 관찰된다.

요약하면, Max-ResNet-18은 중간 수준의 모델 크기와 매우 직관적인 고주파 복원 전략만으로 합성곱 기반 기준선들에서 최첨단 성능을 설정한다. 따라서 아키텍처와 무관하게, 고주파 정보를 보존하는 것은 SNN에서 효과적인 특징 표현의 기본 조건이다.

표 5: CIFAR-10과 CIFAR-100에서 다양한 ResNet 아키텍처 비교.

| 아키텍처(Architecture) | 학습 방법(Training Method) | 블록 구성(Block Config.) | 파라미터(Params, M) | 타임스텝(Time Step) | CIFAR-10 Acc.(%) | CIFAR-100 Acc.(%) |
|---|---|---|---:|---:|---:|---:|
| KDSNN-ResNet-18[47] | 지식 증류(Knowledge Distillation) | [2,2,2,2] | 11.22 | 4 | 95.72 | 78.46 |
| MS-ResNet-18[46] | 직접 학습(Direct Training) | [2,2,2,2] | 11.22 | 4 | 94.40 | 75.06 |
| MS-ResNet-18[46] | 직접 학습(Direct Training) | [3,3,2] | 12.50 | 4 | 94.92 | 76.41 |
| MS-ResNet-34[46] | — | [2,2,2,2] | 21.33 | 4 | 94.69 | 75.34 |
| Max-ResNet-18 | 직접 학습(Direct Training) | [2,2,2,2] | 11.22 | 4 | 96.81 (+2.41) | 81.54 (+6.48) |
| Max-ResNet-18 | 직접 학습(Direct Training) | [3,3,2] | 12.50 | 4 | 97.17 (+2.25) | 83.06 (+6.65) |

## 5 결론(Conclusion)

본 연구는 이진 활성 제약(binary activation constraint)이 SNN의 성능 격차를 초래하는 주된 원인이라는 통념에 도전한다. 이론 분석과 실증 검증을 통해, 우리는 스파이킹 뉴런이 네트워크 수준에서 본질적으로 저역통과 필터로 기능하며, 이로 인해 고주파 성분이 빠르게 감쇠되어 특징 표현을 결정적으로 열화시킨다는 점을 최초로 보여준다.

우리는 고주파 정보가 효과적인 스파이킹 계산에 중요함을 입증한다. Max-Former(63.99M 파라미터)는 ImageNet에서 82.39% top-1 정확도를 달성하여 Spikformer(74.81%, 66.34M)보다 +7.58%p 높고, 유사한 모델 크기에서 에너지 소비는 30% 감소한다. 또한 Max-ResNet-18은 합성곱 기반 기준선들 사이에서 최첨단 성능(CIFAR-10: 97.17%, CIFAR-100: 83.06%)을 달성한다. 주목할 점은 이러한 모든 향상이 매우 단순한 수정만으로 얻어지며, 전체 모델 크기조차 약간 감소한다는 것이다. 우리는 이 간단하지만 효과적인 해결책이 ANN 연구에서 확립된 관행을 넘어, 스파이킹 신경망의 고유한 특성을 탐구하도록 향후 연구를 촉진할 것이라 믿는다.

## 6 감사의 글(Acknowledgements)

본 연구는 Major Science and Technology Innovation 2030 “Brain Science and Brain-like Research” 핵심 과제(No.2021ZD0201405), Guangzhou-HKUST(GZ) Joint Funding Program(Grant No. 2023A03J0682), 중국 국가자연과학기금(National Natural Science Foundation of China, Grant No. 62405255), GuangDong Basic and Applied Basic Research Foundation(No. 2023A1515110679)의 지원을 받았으며, Brain Mind Innovation, Inc와의 공동 프로젝트로부터 부분적으로 지원을 받았다.

## 부록 A 스파이킹 트랜스포머에서의 고주파 정보 분석(Analysis of High-Frequency Information in Spiking Transformers)

제안 방법을 검증하기 위해, 우리는 3개의 정적 데이터셋과 2개의 뉴로모픽 데이터셋에서 실험을 수행한다. 본 절에서는 먼저 논문 본문에서 제시한 결과를 얻기 위한 상세 실험 설정을 제공한다. 이어서 스파이킹 트랜스포머에서 고주파 정보의 중요성에 대한 추가 분석을 수행한다. 전체 파라미터 구성은 공개 코드 저장소(https://github.com/bic-L/MaxFormer)를 참고한다.

표 6: 데이터셋별 이미지 분류 하이퍼파라미터(hyperparameter).

| 하이퍼파라미터(Hyper parameters) | ImageNet | CIFAR-10 | CIFAR-100 | Neuromorphic |
|---|---|---|---|---|
| 모델 크기(Model Size) | 10–384 / 10–512 / 10–768 | 4–384 | 4–384 | 2–256 |
| 에폭(Epochs) | 200 | 400 | 400 | 106 |
| 해상도(Resolution) | 224 × 224 | 32 × 32 | 32 × 32 | 128 × 128 |
| 배치 크기(Batch Size) | 512 (8 × 64) | 128 | 64 | 16 |
| 옵티마이저(Optimizer) | AdamW | AdamW | AdamW | AdamW |
| 학습률(Learning rate) | $1.2\times10^{-3}$ ($T=1$) / $1.35\times10^{-3}$ ($T=4$) | $1.50\times10^{-3}$ | $1.50\times10^{-3}$ | $6.00\times10^{-3}$ |
| 학습률 감쇠(Learning rate decay) | Cosine | Cosine | Cosine | Cosine |
| 워밍업 에폭(Warmup epochs) | 5 | 20 | 20 | 10 |
| 가중치 감쇠(Weight decay) | 0.05 | 0.06 | 0.06 | 0.06 |
| RandAugment | 9 / 0.5 | 9-n1 / 0.4 | 9-n1 / 0.4 | — |
| Mixup | 0.25 / 0.4 / 0.8 | 0.5 | 0.75 | 0.5 |
| CutMix | 1 | 0.5 | 0.5 | — |
| Mixup 확률(Mixup prob) | 0.5 | 1 | 1 | 0.5 |
| Erasing 확률(Erasing prob) | 0.0 | 0.25 | 0.25 | — |
| 라벨 스무딩(Label smoothing) | 0.1 | 0.1 | 0.1 | 0.1 |

### A.1 실험 세부사항(Experimental Details)

데이터셋(Datasets)  
우리는 정적 데이터셋(CIFAR-10[36], CIFAR-100[37], ImageNet[38])과 뉴로모픽 데이터셋(CIFAR10-DVS[39], DVS128 Gesture[40])에서 포괄적인 실험을 통해 맥스-포머(Max-Former)를 평가한다. 학습 및 추론 파이프라인(training/inference pipeline)은 SpikingJelly[48]로 구현한다.

정적 데이터셋(Static Datasets)  
정적 이미지 분류를 위해 세 가지 표준 벤치마크에서 평가한다. ImageNet-1k[38]는 컴퓨터 비전에서 가장 널리 사용되는 데이터셋 중 하나로, 128만 개의 학습 이미지, 5만 개의 검증 이미지, 10만 개의 테스트 이미지로 구성되며 1K 클래스(class)를 포함한다. CIFAR-10[36]과 CIFAR-100[37]은 모두 32×32 해상도의 5만 개 학습 이미지와 1만 개 테스트 이미지를 포함한다. 차이는 CIFAR-10이 10개 범주(category) 분류인 반면 CIFAR-100은 100개 범주 분류라는 점이다.

뉴로모픽 데이터셋(Neuromorphic Datasets)  
이벤트 기반(event-based) 시각 과제를 위해 두 가지 표준 벤치마크에서 평가한다. CIFAR10-DVS[39]는 Dynamic Vision Sensor(DVS)로 움직이는 이미지 샘플을 촬영하여 만든 CIFAR-10의 이벤트 기반 버전이며, 10개 클래스에 걸친 1만 개 이벤트 기반 이미지(128×128 픽셀)를 포함한다(학습 9,000 / 테스트 1,000). DVS128 Gesture[40]는 서로 다른 3가지 조명 조건에서 29명이 수행한 11종의 손 제스처(hand gesture) 1,342개 이벤트 기반 기록을 포함하며, 각 기록은 평균 약 6초 길이이다.

하이퍼파라미터(Hyper Parameters)  
학습 스킴(training scheme)은 주로 [24]와 [41]을 따른다. 구체적으로 MixUp[49], CutMix[50], RandAugment[51]를 데이터 증강(data augmentation)에 사용한다. 모델은 AdamW 옵티마이저[52]로 학습하며, ImageNet-1K 분류에서는 가중치 감쇠를 0.05로, 다른 데이터셋에서는 0.06으로 설정한다. 라벨 스무딩(Label Smoothing)[53]은 0.1로 설정한다. 상세 하이퍼파라미터는 표 6에 제시한다. ImageNet 실험에서는 대부분 모델을 학습하기 위해 8개의 NVIDIA A30 GPU를 사용했다. 다만 MaxFormer-10-512($T=4$)와 MaxFormer-10-768($T=4$) 모델은 8개의 NVIDIA H20 GPU를 사용했다. 더 작은 데이터셋(CIFAR10, CIFAR100, DVS128 Gesture, CIFAR10-DVS) 학습에는 단일 A30 GPU를 사용했다.

### A.2 고주파 정보가 성능에 미치는 영향(Impact of High-Frequency Information on Model Performance)

표 7: CIFAR-100에서의 패치 임베딩(patch embedding) 및 토큰 믹싱(token mixing) 구성. Acc.: top-1 정확도(%). DWC-$N$: 커널 크기 $N\times N$의 스파이킹 깊이별 합성곱(spiking depth-wise convolution). SSA: 스파이킹 자기 어텐션(Spiking Self-Attention).

| 설정 | 패치 임베딩(Patch Embed) | 토큰 믹싱(Token Mix) | Acc.(%) |
|---:|---|---|---:|
| (1) | Embed-Orig/Embed-Orig/Embed-Orig | Avg-Pool/Avg-Pool/Avg-Pool | 76.73 |
| (1) | Embed-Orig/Embed-Orig/Embed-Orig | Max-Pool/Max-Pool/Max-Pool | 79.12 |
| (2) | Embed-Orig/Embed-Orig/Embed-Max | Identity/Identity/Identity | 80.11 |
| (2) | Embed-Orig/Embed-Max/Embed-Max | Identity/Identity/Identity | 80.46 |
| (3) | Embed-Orig/Embed-Max/Embed-Max | Avg-Pool/Avg-Pool/Avg-Pool | 77.61 |
| (3) | Embed-Orig/Embed-Max/Embed-Max | Max-Pool/Max-Pool/Max-Pool | 79.78 |
| (3) | Embed-Orig/Embed-Max/Embed-Max | Identity/Max-Pool/Identity | 79.99 |
| (3) | Embed-Orig/Embed-Max/Embed-Max | Identity/Identity/Max-Pool | 80.12 |
| (4) | Embed-Orig/Embed-Max/Embed-Max | SSA/SSA/SSA | 82.13 |
| (4) | Embed-Orig/Embed-Max/Embed-Max | DWC-3/DWC-5/SSA | 82.36 |
| (4) | Embed-Orig/Embed-Max/Embed-Max | DWC-5/DWC-3/SSA | 82.46 |
| (4) | Embed-Orig/Embed-Max/Embed-Max | DWC-5/DWC-7/SSA | 82.45 |
| (4) | Embed-Orig/Embed-Max/Embed-Max | DWC-7/DWC-5/SSA | 82.42 |
| (4) | Embed-Orig/Embed-Max/Embed-Max | DWC-3/DWC-3/SSA | 82.59 |
| (4) | Embed-Orig/Embed-Max/Embed-Max | Identity/DWC-3/SSA | 82.65 |
| (5) | Embed-Orig/Embed-Max/Embed-Max | SSA+DWC-5/SSA+DWC-5/SSA+DWC-5 | 82.09 |
| (5) | Embed-Orig/Embed-Orig/Embed-Orig | SSA+DWC-3/SSA+DWC-3/SSA+DWC-3 | 82.56 |
| (5) | Embed-Orig/Embed-Orig/Embed-Orig | SSA+DWC-5/SSA+DWC-5/SSA+DWC-5 | 82.73 |
| (6) | Embed-Orig/Embed-Max/Embed-Max | SSA/SSA/SSA | 82.13 |
| (6) | Embed-1Max/Embed-Max/Embed-Max | SSA/SSA/SSA | 79.98 |
| (6) | Embed-Max/Embed-Max/Embed-Max | SSA/SSA/SSA | 78.78 |
| (7) | Embed-Orig/Embed-1Max/Embed-1Max | SSA/SSA/SSA | 82.02 |
| (7) | Embed-1Max/Embed-1Max/Embed-1Max | SSA/SSA/SSA | 81.86 |
| (8) | Embed-Orig/Embed-Max/Embed-Max | SDSA/SDSA/SDSA | 81.77 |
| (8) | Embed-Orig/Embed-1Max/Embed-1Max | SDSA/SDSA/SDSA | 81.52 |

표 7에서는 고주파 정보의 영향에 대한 추가 분석을 제공한다. 우리는 표 6의 설정을 사용해 CIFAR-100[37]에서 이 실험을 수행했으며, 다음 측면에 따라 소거 결과를 논의한다.

(a) 패치 임베딩/토큰 믹싱에서의 추가 Max-Pool  
스파이킹 뉴런의 고유한 저역통과 필터 특성 때문에, 고주파 정보는 스파이킹 트랜스포머의 성능에 결정적인 역할을 한다. 표 7(1)의 결과는 max-pooling 연산을 통해 고주파를 전략적으로 보존하면 모델 정확도가 크게 향상됨을 보여준다. 모든 단계에서 평균 풀링을 최대 풀링으로 대체하면 76.73%에서 79.12%로 2.39%p 개선된다. 표 7(2)에서는 마지막 패치 임베딩 블록(80.11%)에서 중간 블록까지 Embed-Max를 확장하면(80.46%) 성능이 점진적으로 증가한다.

그러나 과도한 고주파 정보는 오히려 성능을 해칠 수 있다. 예를 들어 표 7(3)에서 토큰 믹싱을 전 단계 Avg-Pool에서 전 단계 Max-Pool로 바꾸면 77.61%에서 79.78%로 향상되지만, 더 표적화된(targeted) 설정이 더 낫다. 중간 단계에서만 max-pooling을 사용하면 79.99%로 증가하고, 마지막 단계에만 제한하면 80.12%로 더 높다.

이는 스파이킹 뉴런이 저역통과 필터처럼 작동하여, 정보가 네트워크 깊은 곳으로 이동할수록 고주파 성분을 자연스럽게 줄이기 때문이다. 따라서 네트워크의 특정 지점에서 고주파 성분을 전략적으로 다시 추가하는 것이 스파이킹 트랜스포머의 성능 한계를 끌어올리는 데 중요하다.

일반적으로 계층적 스파이킹 트랜스포머에서는 표 7(6–8)과 같이 패치 임베딩에서 Embed-Max가 성능을 향상시킬 수 있다. 그러나 비계층적(non-hierarchical) 구조에서는, 고주파 강화가 원리적으로는 유익할 수 있으나, 패치 임베딩이 전체 과정에서 한 번만 수행되므로 깊은 층에서 개선이 희석되어 실제 영향이 제한적이다. 더 유망한 해결책은 [19]에서 논의하듯 토큰 믹싱을 최적화하는 것이다.

(b) 스파이킹 트랜스포머는 고주파 정보로부터 이득을 얻는다  
생물학적 시각에서 고주파 디테일은 초기 처리 단계가 기초 특징(elementary feature)을 학습하도록 돕고, 이후 국소에서 전역으로 점진적으로 표현을 구축한다. 마찬가지로 표준(비-스파이킹) 트랜스포머에서도 하위 층은 더 많은 고주파 디테일이 필요하고, 상위 층은 전역 정보에서 더 잘 작동한다. 스파이킹 트랜스포머도 같은 설계 철학을 따르지만, 중요한 차이가 있다. 즉, 스파이킹 트랜스포머는 스파이킹 뉴런에 의해 손실되는 고주파 정보를 복원하기 위해 추가적인 주파수 강화 연산(예: max-pooling, 깊이별 합성곱)이 필요하다.

표 7(4)에서 보이듯, 적절한 토큰 믹싱 전략은 스파이킹 트랜스포머의 고주파 정보를 효과적으로 복원하여 성능을 크게 향상시킬 수 있다. SSA를 토큰 믹서로 사용하는 대신 DWC로 대체하면, 성능은 82.13%에서 82.65%로 개선된다. 중요한 점은 이 개선이 파라미터/계산 부담 증가에서 비롯되지 않는다는 것이다. 예를 들어 Identity/DWC-3/SSA 조합은 계산 비용이 더 낮음에도 DWC-3/DWC-5/SSA보다 0.29%p 더 잘 작동한다. 또한 표 7(4–5)의 추가 실험은 전 단계 SSA 네트워크에서도 이러한 관찰이 유지됨을 확인한다. 즉, 고주파 성분 복원은 CIFAR100에서 82.13%에서 82.73%로(+0.6%p) 성능을 유의미하게 최적화한다. 적절한 고주파 강화 전략은 스파이킹 트랜스포머의 잠재력을 온전히 발휘하는 데 필수적이다.

### A.3 Max-ResNet 구현(Max-ResNet Implementation)

그림 7에서 보이듯, Max-ResNet은 MS-ResNet[46]에 매우 작은 구조 변경만 도입한다. 첫 번째 층은 그대로 유지하면서, 그 외 모든 층을 Max-ResNet 층으로 교체한다. 코드는 https://github.com/bic-L/MaxFormer 에서 제공된다.

그림 7: Max-ResNet 개요. 각 블록(block)과 각 층(layer)마다 단일 Max-Pool 연산을 추가한다.

## 부록 B 에너지 분석(Energy Analysis)

스파이킹 트랜스포머의 이론적 에너지 소비(theoretical energy consumption)를 추정하기 위해, 우리는 선행 연구[8, 54, 41, 19]의 방법론을 따른다. 배치 정규화(batch normalization, BN) 층과 합성곱 뒤의 선형 스케일링 변환은 배포(deployment) 시 편향(bias) 항을 추가한 형태로 합성곱 층 자체에 병합될 수 있다. 따라서 일반 관행[8, 54, 41, 19]에서는 이론적 에너지 계산에서 BN의 에너지 소비를 보통 제외한다. 공정 비교를 위해 본 연구도 동일 전략을 채택한다.

스파이킹 트랜스포머에서 에너지 소비는 시냅스 연산(synaptic operations, SOPs)에 정비례하며, 다음으로 계산할 수 있다.

$$
\mathrm{SOPs}(l) = f_r \times T \times \mathrm{FLOPs}(l) \qquad (22)
$$

여기서 $l$은 스파이킹 트랜스포머의 특정 블록 또는 층을 의미하고, $f_r$은 해당 블록/층 입력 스파이크열의 발화율이며, $T$는 스파이킹 뉴런의 시뮬레이션 타임스텝이다.

[7]에서 설명한 45nm 뉴로모픽 칩에서 MAC과 누산(AC) 연산이 구현되며, 각 MAC은 $E_{\mathrm{MAC}}=4.6\,\mathrm{pJ}$, 각 AC는 $E_{\mathrm{AC}}=0.9\,\mathrm{pJ}$의 에너지를 사용한다고 가정하면, 모든 층에서 사용된 MAC/AC 에너지를 합산하여 스파이킹 트랜스포머의 총 에너지를 다음과 같이 추정할 수 있다.

$$
E_{\mathrm{SNN}} = E_{\mathrm{MAC}}\times \mathrm{FLOP}^{1}_{\mathrm{CONV}} + E_{\mathrm{AC}}\times\left(\sum_{n=2}^{N} \mathrm{SOP}^{n}_{\mathrm{SNN\ Conv}} + \sum_{j=1}^{M} \mathrm{SOP}^{j}_{\mathrm{SNN\ FC}}\right) \qquad (23)
$$

여기서 $\mathrm{FLOP}^{1}_{\mathrm{CONV}}$는 첫 번째 층의 부동소수점 연산량으로, 정적 이미지 분류에서 비-스파이크 입력을 스파이크 형태로 변환한다. 이 층은 부동소수점 계산을 수행하므로 $E_{\mathrm{MAC}}$로 에너지를 추정한다. 이후 모든 층은 스파이크 데이터를 처리하므로 $E_{\mathrm{AC}}$로 에너지를 추정한다. 주류 비-스파이킹 트랜스포머의 에너지 소비는 다음으로 추정한다.

$$
E_{\mathrm{ANN}} = E_{\mathrm{MAC}}\times \mathrm{FLOPs} \qquad (24)
$$

## 부록 C 학습/추론 시간 및 메모리 사용량 비교(Comparison on Train/Inference Time and Memory Consumption)

표 8: QKFormer-10-768과 Max-Former-10-768의 학습/추론 시간 및 메모리 사용량 비교. 모든 측정은 시뮬레이션 타임스텝 $T=1$, 배치 크기 $B=32$에서 수행했다. MS-QKFormer는 멤브레인 쇼트컷을 적용한 QKFormer 변형을 의미한다.

| 모델 | 학습 시간(Train Time, s) | 학습 메모리(Train Memory, MB) | 추론 시간(Infer. Time, s) | 추론 메모리(Infer. Memory, MB) |
|---|---:|---:|---:|---:|
| QKFormer (64.96M, $T=1$, $B=32$) | 0.214 | 18227 | 0.053 | 5000 |
| MS-QKFormer* (64.96M, $T=1$, $B=32$) | 0.208 | 17496 | 0.048 | 4822 |
| Max-Former* (63.99M, $T=1$, $B=32$) | 0.179 | 15431 | 0.044 | 4354 |

Max-Former는 더 빠른 학습/추론 속도와 더 적은 메모리 사용을 제공한다. 우리는 ImageNet에서 224×224 입력 해상도를 사용하여 QKFormer[24], 멤브레인 쇼트컷 변형(MS-QKFormer), 그리고 Max-Former를 비교했다. 모든 테스트는 Intel Xeon Gold 6348 CPU(2.60GHz)와 Nvidia A30 GPU가 탑재된 CentOS 7.9 서버에서 수행했다. 표 8에서 보이듯, 동일한 계층적 구조와 쇼트컷 구성을 갖는 MS-QKFormer와 비교할 때 Max-Former는 학습 시간을 14% 줄였고, 추론 시간과 메모리 사용을 10% 줄였다. 또한 원래 QKFormer에 사용된 프리-스파이크 쇼트컷 전략은 처리 시간과 메모리 요구량을 증가시키는 것으로 나타났다.

## 부록 D 스파이킹 신경망에서의 잔차 연결(Residual Connections in Spiking Neural Networks)

(a) 성능과 에너지의 트레이드오프(Performance and Energy Tradeoffs)  
스파이크 기반 계산의 비동기적 특성은 잔차 연결(residual connection) 구현을 어렵게 만든다. 그 결과, SNN 연구 커뮤니티는 알고리즘 및 하드웨어 구현 측면에서 표준화된 잔차 학습 접근에 대해 아직 합의를 이루지 못했다. 본 연구의 핵심 초점은 잔차 학습이 아니지만, 최근 몇 년간 등장한 가장 대표적인 두 방법인 프리-스파이크 쇼트컷(pre-spike shortcut)[55]과 멤브레인 쇼트컷(membrane shortcut)[54]을 상세 비교하고자 한다.

3.3절에서 설명했듯, 프리-스파이크 쇼트컷은 스파이킹 출력 사이에 잔차 연결을 구현하는 반면, 멤브레인 쇼트컷은 막 전위를 직접 연결한다. 알고리즘 설계 관점에서, 멤브레인 쇼트컷은 특히 작은 데이터셋에서 더 나은 성능을 돕는다고 보고된 바 있다[24, 41]. 그러나 우리의 실험은 이 장점이 모든 시나리오에서 보편적이지 않음을 보여준다. 표 9에서 보이듯, 스파이킹 트랜스포머의 패치 임베딩 단계는 에너지 소비의 대부분을 차지한다. 따라서 QKFormer[24]와 같은 계층적 구조의 다중 패치 임베딩은 더 적은 파라미터로 효율적 특징 학습을 가능하게 하지만, 더 높은 에너지 사용을 수반한다. 이는 쇼트컷 방식 선택이 전체 에너지 효율에 특히 큰 영향을 주게 만든다. 224×224 해상도의 ImageNet 이미지를 처리할 때, 프리-스파이크 QKFormer는 멤브레인 쇼트컷 변형 대비 3배의 에너지를 소비하며, 이는 필요한 SOP가 훨씬 많기 때문이다. 다만 이러한 계산 오버헤드는 성능 이득으로 이어지기도 한다(QKFormer vs. MS-QKFormer 비교 시 +2.32%p).

하드웨어 관점에서, 두 쇼트컷 모두 뉴로모픽 칩에서 기술적으로 구현 가능하지만 상당한 도전이 있다[56, 57]. 특히 프리-스파이크 쇼트컷은 3값 스파이크 전송(0, 1, 2)이 발생할 수 있으므로, 칩이 다중 스파이크 연산(multi-spike operation)을 지원해야 하며, 이는 에너지 증가 또는 하드웨어 복잡도 증가로 이어진다[58]. Yao 등[41]은 주소 지정(addressing) 함수를 통해 막 전위를 다음 층의 해당 뉴런으로 전달하여 병합하는 방식으로 멤브레인 쇼트컷을 구현하는 방안을 제안했다. 멤브레인 쇼트컷은 스파이크 구동 계산 패러다임을 엄격히 준수하며, 추가 하드웨어 오버헤드 없이 표준 뉴로모픽 하드웨어에서 지원될 가능성이 있지만, 막 전위를 전송하는 것은 상당한 통신 오버헤드(communication overhead)를 초래하여 실제 구현이 쉽지 않다. 하드웨어 배포에 관한 많은 논의는 현재로서는 쇼트컷을 피하는 것을 선호 접근으로 주장하기도 한다[59, 60]. 그러나 현대 딥러닝에서 잔차 학습이 갖는 중요성을 고려하면, 쇼트컷을 완전히 회피하는 것은 장기적으로 지속 가능하지 않다.

본 연구에서는 공정 비교를 위해 멤브레인 쇼트컷을 사용하는 MS-QKFormer와의 비교를 주로 수행한다. 우리는 SNN 커뮤니티가 쇼트컷 방식이 에너지 효율과 성능 모두에 미치는 큰 영향을 고려하여, 가까운 미래에 표준화된 쇼트컷 구현에 대한 합의를 도출하기를 기대한다.

표 9: QKFormer, 멤브레인 쇼트컷을 적용한 QKFormer(MS-QKFormer), 제안하는 Max-Former의 처리 단계별 에너지 소비(mJ) 비교. *는 동일한 학습 구성.

| 모델 | Stage 1 Patch Embed | Stage 1 Token Mix | Stage 1 MLP | Stage 2 Patch Embed | Stage 2 Token Mix | Stage 2 MLP | Stage 3 Patch Embed | Stage 3 Token Mix | Stage 3 MLP | Classifier | Total Energy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| QKFormer (16.47M, $T=4$, 78.80%) | 1.26 | 0.16 | 0.41 | 2.68 | 0.35 | 0.77 | 2.97 | 2.71 | 3.81 | 0.006 | 15.13 |
| MS-QKFormer* (16.47M, $T=4$, 76.48%) | 1.19 | 0.052 | 0.15 | 0.96 | 0.11 | 0.25 | 0.88 | 0.88 | 1.06 | 0.007 | 5.52 |
| Max-Former* (16.23M, $T=4$, 77.82%) | 0.41 | 0.02 | 0.17 | 0.89 | 0.01 | 0.36 | 0.91 | 0.96 | 1.16 | 0.001 | 4.89 |

## 부록 E 시각화(Visualization)

우리는 크기가 유사한 멤브레인 쇼트컷을 사용하는 4개 스파이킹 트랜스포머의 Grad-CAM 시각화[61]를 제시한다. Spike-Driven Transformer[41] 및 SWFormer[19]와 비교하면, QKFormer[24]와 Max-Former 모두에서 사용되는 계층적 구조가 목표 객체에 더 정확히 초점을 맞추도록 한다. 또한 MS-QKFormer와 비교하면, Max-Former는 더 집중된 활성화 패턴을 보여준다. 예를 들어 북극곰 이미지에서 Max-Former는 배경을 완전히 건너뛰고, 곰의 주요 특징(윤곽이나 털이 아니라 머리)에 정확히 초점을 맞춘다.

## 부록 F 영향 및 한계(Impact and Limitation)

본 연구는 스파이킹 트랜스포머에서 이미 존재하는 많은 구조적 선택(architectural choice)에 대한 이론적 기반을 제공한다. 구체적으로 [41]에서는 MetaFormer의 알려진 관행을 스파이킹 트랜스포머에 그대로 적용하면 좋은 결과가 나오지 않음을 발견했다. 토큰 믹서(token mixer)로 SDSA를 평균 풀링(avg pooling) 연산자로 대체하면 성능이 61.0%에서 41.2%로 크게 저하된다. 유사한 현상은 이전 연구에서도 논의된 바 있다. 예를 들어 Spikformer v2[62]는 원래 Spikformer[8]에서 max-pooling 연산자를 제거하면 성능이 크게 떨어지고, 패치 임베딩 단계에 합성곱 층(고역통과 필터처럼 작동)을 추가하면 성능이 크게 향상됨을 발견했다. 본 연구는 이러한 구조 설계의 근본 원리를 드러낸다. 즉, 스파이킹 트랜스포머는 고유한 저역통과 활성으로 인해 발생하는 특징 열화를 완화하기 위해 고주파 성분을 강화해야 한다.

우리는 여전히 탐구할 공간이 많다는 점을 인지하고 있으며, Max-Former가 향후 연구의 좋은 출발점이 되기를 바란다. 예를 들어 [31]과 유사하게, Max-Former는 주파수 성분을 수동(manual)으로 균형 조절해야 하므로, 서로 다른 과제(task)에 적응할 때 상당한 전문 지식(expertise)이 요구된다. 푸리에 기반(Fourier-based)[63] 또는 웨이블릿 기반(Wavelet-based)[19]과 같은 직접적인 주파수 학습(frequency learning) 접근을 통합하는 것은 더 직관적인 해결책을 제공할 수 있다.

그림 8: 크기가 유사한 4개 스파이킹 트랜스포머의 Grad-CAM 시각화[61] 비교. Spike-Driven Transformer-8-512[41](29.68M), SWFormer-8-512[19](27.6M), 멤브레인 쇼트컷을 적용한 QKFormer-10-512(MS-QKFormer)(29.08M), Max-Former-10-512(28.65M).

그러나 주된 도전은 효율적인 스파이크 기반 주파수 표현(spike-based frequency representation)을 개발하는 데 있다. 전반적으로 우리는 본 연구가, 표준 비-스파이킹 신경망의 확립된 관행을 무리하게 적응시키는 데 과도한 노력을 들이기보다, 스파이킹 뉴런의 고유한 특성을 탐구함으로써 뉴로모픽 컴퓨팅(neuromorphic computing)을 진전시키는 더 많은 향후 연구를 촉발할 것이라 믿는다.

## NeurIPS 논문 체크리스트(NeurIPS Paper Checklist)

1. 주장(Claims)  
질문: 초록과 소개에서 제시된 주요 주장들이 논문의 기여와 범위를 정확히 반영하는가?  
답변: [Yes]  
정당화: 초록과 소개는 논문의 기여와 범위를 명확히 반영한다.

2. 한계(Limitations)  
질문: 저자들이 수행한 작업의 한계를 논문에서 논의하는가?  
답변: [Yes]  
정당화: 부록 E에서 한계를 논의한다.

3. 이론 가정 및 증명(Theory assumptions and proofs)  
질문: 각 이론 결과에 대해, 모든 가정과 완전한(그리고 올바른) 증명을 제공하는가?  
답변: [Yes]  
정당화: 모든 정리(theorem), 수식(formula), 증명이 본문에 명확히 제시되어 있다.

4. 실험 결과 재현성(Experimental result reproducibility)  
질문: 코드/데이터 제공 여부와 무관하게, 주요 주장/결론에 영향을 미치는 수준에서 실험을 재현하는 데 필요한 정보를 충분히 공개하는가?  
답변: [Yes]  
정당화: 부록 A, B에서 실험 세부사항을 제공한다.

5. 데이터와 코드의 공개(Open access to data and code)  
질문: 보충 자료(supplemental material)에 설명된 바와 같이, 주요 실험 결과를 충실히 재현할 수 있을 만큼의 충분한 지침과 함께 데이터/코드를 공개하는가?  
답변: [Yes]  
정당화: 코드는 https://github.com/bic-L/MaxFormer 에서 제공된다.

6. 실험 설정/세부사항(Experimental setting/details)  
질문: 결과를 이해하는 데 필요한 모든 학습/테스트 세부사항(예: 데이터 분할, 하이퍼파라미터, 선택 방법, 옵티마이저 종류 등)을 명시하는가?  
답변: [Yes]  
정당화: 모든 학습/테스트 세부사항이 부록 A에 공개되어 있다.

7. 실험 통계적 유의성(Experiment statistical significance)  
질문: 오차막대(error bar)를 적절하고 올바르게 정의하여 보고하거나, 혹은 통계적 유의성과 관련된 다른 적절한 정보를 제공하는가?  
답변: [No]  
정당화: 오차막대는 계산 비용이 지나치게 크기 때문에 보고하지 않았다.

8. 실험 계산 자원(Experiments compute resources)  
질문: 각 실험에 대해, 재현에 필요한 컴퓨팅 자원(컴퓨트 워커 종류, 메모리, 실행 시간 등)에 대한 충분한 정보를 제공하는가?  
답변: [Yes]  
정당화: 부록 A를 참고.

9. 윤리 강령(Code of ethics)  
질문: 연구가 NeurIPS 윤리 강령(https://neurips.cc/public/EthicsGuidelines)을 모든 면에서 준수하는가?  
답변: [Yes]  
정당화: 본 연구는 모든 면에서 NeurIPS 윤리 강령을 준수한다.

10. 광범위한 영향(Broader impacts)  
질문: 연구의 잠재적 긍정적 사회 영향과 부정적 사회 영향을 모두 논의하는가?  
답변: [NA]  
정당화: 본 연구는 기초 연구(foundational research)로서 특정 응용에 직접 연결되어 있지 않다.

11. 안전장치(Safeguards)  
질문: 오남용 위험이 높은 데이터/모델(예: 사전학습 언어모델, 이미지 생성기, 스크레이핑 데이터셋 등)을 책임 있게 공개하기 위한 안전장치를 설명하는가?  
답변: [NA]  
정당화: 본 논문은 그러한 위험을 제기하지 않는다.

12. 기존 자산의 라이선스(Licenses for existing assets)  
질문: 논문에서 사용된 자산(코드, 데이터, 모델 등)의 제작자/원 소유자를 적절히 인용하며, 라이선스 및 사용 조건을 명시하고 존중하는가?  
답변: [Yes]  
정당화: 논문은 제작자를 적절히 인용하고, 기존 자산의 라이선스 및 사용 조건을 언급하며 존중한다.

13. 신규 자산(New assets)  
질문: 새롭게 도입한 자산이 잘 문서화되어 있으며, 자산과 함께 문서가 제공되는가?  
답변: [NA]  
정당화: 본 논문은 신규 자산을 공개하지 않는다.

14. 크라우드소싱 및 인간 대상 연구(Crowdsourcing and research with human subjects)  
질문: 크라우드소싱 실험 또는 인간 대상 연구의 경우, 참가자 지침 전문과(해당 시) 스크린샷, 보상(있는 경우) 등 세부사항을 포함하는가?  
답변: [NA]  
정당화: 본 논문은 크라우드소싱 또는 인간 대상 연구를 포함하지 않는다.

15. IRB 승인 등(Institutional review board approvals or equivalent for research with human subjects)  
질문: 참가자가 감수한 잠재적 위험, 위험의 고지 여부, 그리고 국가/기관 요구사항에 따른 IRB(또는 동등한 승인/심사) 획득 여부를 설명하는가?  
답변: [NA]  
정당화: 본 논문은 크라우드소싱 또는 인간 대상 연구를 포함하지 않는다.

16. LLM 사용 선언(Declaration of LLM usage)  
질문: LLM 사용이 핵심 방법론의 중요한, 독창적, 또는 비표준 구성 요소인 경우 이를 설명하는가? (단, LLM이 글쓰기/편집/서식 목적에만 사용되어 방법론·엄밀성·독창성에 영향을 주지 않는다면 선언이 필요 없다.)  
답변: [NA]  
정당화: 본 연구는 LLM을 중요한/독창적/비표준 구성 요소로 사용하지 않는다.
