
# 효과적이고 효율적인 장기 시퀀스 모델링(long sequence modeling)을 위한 수상돌기 공진-발화 뉴런(Dendritic Resonate-and-Fire Neuron)

Dehao Zhang, Malu Zhang, Shuai Wang, Jingya Wang, Wenjie Wei, Zeyu Ma, Guoqing Wang, Yang Yang, Haizhou Li

1 중국 전자과기대학(University of Electronic Science and Technology of China)  
2 선전 루프 에어리어 연구소(Shenzhen Loop Area Institute)  
3 홍콩중문대학교 선전(The Chinese University of Hong Kong, Shenzhen)

교신저자(corresponding author): maluzhang@uestc.edu.cn

제39회 신경정보처리시스템 학회(39th Conference on Neural Information Processing Systems, NeurIPS 2025)

## 초록(Abstract)

시퀀스 길이(sequence length)의 폭발적 증가는 효과적이면서도 효율적인 장기 시퀀스 모델링(long sequence modeling)에 대한 요구를 더욱 증대시켰다. 내재적 진동 막 동역학(intrinsic oscillatory membrane dynamics)의 이점을 지닌 공진-발화(Resonate-and-Fire, RF) 뉴런은 입력 신호(input signal)로부터 주파수 성분(frequency component)을 효율적으로 추출하고 이를 시공간 스파이크 열(spatiotemporal spike train)로 부호화할 수 있어 장기 시퀀스 모델링에 적합하다. 그러나 RF 뉴런은 복잡한 시간 과제(complex temporal task)에서 유효 메모리 용량(effective memory capacity)이 제한적이며, 에너지 효율(energy efficiency)과 학습 속도(training speed) 사이의 절충(trade-off)을 보인다. 본 논문은 생물학적 뉴런의 수상돌기 구조(dendritic structure)에 착안하여 다중 수상돌기(multi-dendritic)와 세포체(soma) 구조를 명시적으로 포함하는 수상돌기 공진-발화(Dendritic Resonate-and-Fire, D-RF) 모델을 제안한다. 각 수상돌기 가지(dendritic branch)는 RF 뉴런의 내재적 진동 동역학을 활용하여 특정 주파수 대역(frequency band)을 부호화하고, 그 결과 집합적으로 포괄적인 주파수 표현(frequency representation)을 달성한다. 또한 세포체 구조에는 적응형 임계값(adaptive threshold) 메커니즘을 도입하였다. 이 메커니즘은 과거 스파이킹 활동(historical spiking activity)에 따라 발화 임계값(firing threshold)을 조정함으로써, 장기 시퀀스 과제에서 학습 효율을 유지하면서 중복 스파이크(redundant spike)를 줄인다. 광범위한 실험은 제안 방법이 학습 중 계산 효율(computational efficiency)을 저해하지 않으면서도 희소한 스파이크(sparse spike)를 충분히 보장하는 동시에 경쟁력 있는 정확도(accuracy)를 유지함을 보여준다. 이러한 결과는 에지 플랫폼(edge platform)에서 장기 시퀀스 모델링을 위한 효과적이고 효율적인 해법으로서의 잠재력을 뒷받침한다.

## 1 서론(Introduction)

장기 시퀀스 모델링(long sequence modeling)은 복잡한 시간 패턴(temporal pattern)과 동적 특성(dynamic characteristic)을 효율적으로 포착하며, 음성 인식(speech recognition) [35, 64, 65] 및 뇌전도(electroencephalogram, EEG) 모니터링 [5, 48, 67] 같은 에지 컴퓨팅(edge computing) 시나리오에서 뛰어난 응용 잠재력을 보여준다. 그러나 주류 시퀀스 모델링 방법은 여전히 순환 신경망(Recurrent Neural Networks, RNNs) [49, 52], 트랜스포머(Transformer) [29, 62], 상태공간 모델(state-space model, SSM) [18, 20]에 주로 의존한다. 이러한 접근법은 문맥 정보(contextual information)를 유한 상태(finite state)로 효과적으로 압축하지만, 여전히 광범위한 부동소수점 행렬 곱셈(floating-point matrix multiplication)을 포함하므로 높은 계산 복잡도(computational complexity), 추론 지연(inference latency), 에너지 소비(energy consumption)를 초래한다 [51]. 따라서 높은 성능(performance), 에너지 효율, 빠른 추론(fast inference)을 동시에 달성하는 장기 시퀀스 모델을 설계하는 일은 여전히 핵심적이고 지속적인 연구 과제이다.

뇌의 신경 회로(neural circuit)의 구조와 기능에서 영감을 받은 스파이킹 신경망(Spiking Neural Networks, SNNs)은 생물학적으로 타당하며 계산 효율적인 모델로 부상했다 [37, 59]. 인공 신경망(Artificial Neural Networks, ANNs)과 달리, SNN은 이벤트 구동(event-driven) 계산 능력 [6, 77, 78]과 동적 시간 정보(dynamic temporal information)를 처리할 잠재력 [72, 70, 80]을 지닌다. 이러한 특성은 이진 스파이크(binary spike)를 통해 정보가 전달되도록 하며, 막 전위(membrane potential)를 통해 과거 문맥(historical context)이 유지되게 한다. 여러 연구는 누설 적분-발화(Leaky Integrate-and-Fire, LIF) 뉴런을 기반으로 장기 시퀀스 모델링에 SNN을 적용해 왔다. Zhang 등 [79]은 장기 시간 의존성(long-term temporal dependency)을 더 잘 포착하기 위해 2-구획(two-compartment) LIF 뉴런을 도입했고, Fang 등 [14]은 모든 시간 스텝(timestep)에 걸친 정보 활용을 극대화하기 위해 리셋(reset) 메커니즘을 제거하였다. 그러나 이들 방법은 직렬적인 충전-발화-리셋(charge-fire-reset) 동역학 때문에 장기 시퀀스에서 복잡한 시간 스케일 변동(timescale variation)과 장거리 의존성(long-range dependency)을 포착하지 못해 [66, 72, 76], 복잡한 과제에서 성능이 제한된다.

LIF 뉴런의 효율적 대안으로서 공진-발화(Resonate-and-Fire, RF) 뉴런 [27]은 복소값 상태 변수(complex-valued state variable)를 사용하여 장기 시퀀스 모델링에서 과거 정보를 더욱 효과적으로 보존한다. Higuchi 등 [21]은 RF 뉴런의 리셋 동역학(reset dynamics)에 불응기(refractory) 메커니즘을 통합하여 주파수 선택성(frequency selectivity)을 유지하면서 장기 시퀀스 처리 용량을 크게 향상시키고 스파이크 패턴 희소성(spike pattern sparsity)을 증진하였다. Huang 등 [24]은 리셋 메커니즘을 제거하고 학습 가능한 시정수(learnable time constant)가 RF 뉴런의 내재적 리셋 거동(intrinsic reset behavior)을 포착할 수 있다고 제안하였다. 이 접근은 계산 복잡도를 $O(L^2)$ 에서 $O(L \log L)$ 로 줄여 계산 효율을 크게 향상시킨다. 그럼에도 불구하고 기존 방법은 여전히 두 가지 주요 도전에 직면한다. 첫째, RF 뉴런의 제한된 대역폭(bandwidth)은 복잡한 시간 신호(complex temporal signal)로부터 다양한 주파수 대역 조합을 추출하는 능력을 제한하여, 뉴런이 단순화된 공진기(simplified resonator)처럼 동작하게 만든다. 둘째, RF 뉴런은 에너지 효율과 학습 속도 사이의 절충에 직면한다. 리셋 메커니즘을 제거하면 효율적 학습은 가능하지만 과도한 스파이크 활동(excessive spike activity)이 발생하고, 반대로 리셋을 포함하면 스파이킹 활동은 억제되지만 학습 오버헤드(training overhead)가 증가한다.

생물학적 뉴런의 수상돌기 구조(dendritic structure) [44, 61, 38, 26]에서 영감을 받아, 우리는 효과적이고 효율적인 장기 시퀀스 모델링을 위한 새로운 수상돌기 공진-발화(Dendritic Resonate-and-Fire, D-RF) 뉴런을 제안한다. 이 뉴런은 다중 수상돌기(multi-dendrite)와 세포체(soma) 구조로 구성된다. 첫째, 각 수상돌기 가지는 RF 뉴런의 진동 특성(oscillation characteristic)을 통해 입력 신호의 특정 주파수 응답(frequency response)을 포착하여 여러 시간 스케일(timescale)에 걸친 포괄적 스펙트럼 분해(comprehensive spectral decomposition)를 달성한다. 둘째, 세포체 구조에는 과거 스파이킹 패턴(historical spiking pattern)에 따라 임계값을 동적으로 조정하는 적응형 임계값 메커니즘을 통합하여, 학습 효율을 유지하면서 희소한 스파이크를 달성한다. 장기 시퀀스 과제에 대한 광범위한 실험은 제안 뉴런 모델의 높은 성능과 에너지 효율을 확인한다. 주요 기여는 다음과 같다.

- 우리는 장기 시퀀스 모델링에서 기존 RF 뉴런의 한계를 상세히 분석하고, 제한된 메모리 용량과 에너지 효율 대 학습 속도 사이의 본질적 절충을 강조한다. 첫째, 제한된 대역폭 응답(bandwidth response) 때문에 RF 뉴런은 단순화된 공진기처럼 동작한다. 둘째, 리셋 메커니즘의 존재 여부는 희소 스파이킹(sparse spiking)과 학습 효율(training efficiency) 사이의 충돌을 유발한다.
- 우리는 두 구성 요소로 이루어진 D-RF 뉴런을 제안한다. 첫째, 수상돌기 가지는 RF 동역학을 활용하여 특화된 주파수 선택성(specialized frequency selectivity)을 달성하고, 집합적으로 여러 시간 스케일에 걸친 전체 스펙트럼 커버리지(full spectral coverage)를 가능하게 한다. 둘째, 세포체의 적응형 임계값 메커니즘은 과거 스파이킹 활동에 기반하여 임계값을 동적으로 조정함으로써, 학습 효과를 유지하면서 계산 비용(computational cost)과 에너지 효율 사이의 균형을 맞춘다.
- 광범위한 실험은 본 방법이 다양한 장기 시퀀스 과제에서 경쟁력 있는 성능을 달성함을 보여준다. 또한 학습 효율을 유지하면서 더 희소한 스파이킹 활동을 생성한다. 이러한 결과는 장기 시퀀스 모델링에서 우리 모델이 효과성과 계산 효율성 모두에서 이점을 지님을 보여준다.

## 2 관련 연구(Related Work)

### 2.1 장기 시퀀스 모델링(long sequence modeling)을 위한 고급 스파이킹 뉴런(Advanced Spiking Neurons)

스파이킹 뉴런(spiking neuron)의 동적 특성(dynamic characteristic) 때문에, 이들은 장기 시퀀스 모델링을 처리할 능력을 지닌다고 여겨진다. 그러나 LIF 모델 [17, 28]과 그 변형들 [4, 13, 72]은 시간 과제(temporal task)에서 제한된 메모리 용량(memory capacity)을 보이며, 이는 효과적인 장기 시퀀스 모델링의 핵심 요소로 간주된다. 이러한 한계를 극복하기 위해 여러 연구 [7, 79]는 더 복잡한 신경 동역학(neural dynamics)에서 영감을 얻었다. RF 뉴런 [27]은 고유한 주파수 대역 선호(intrinsic frequency band preference) 때문에 상당한 주목을 받았다. Orchard 등 [41]은 RF 뉴런을 활용해 원시 신호(raw signal)를 희소한 스파이크 열로 부호화함으로써 출력 대역폭(output bandwidth)을 크게 줄였다. 또한 Higuchi 등 [21]은 적응형 감쇠 계수(adaptive decay factor) 메커니즘 [50, 16]과 불응기 메커니즘 [54]을 도입하여, 장거리 시퀀스 모델링(long-range sequence modeling)에서 RF 뉴런의 에너지 효율과 성능 사이의 균형을 향상시켰다. 아울러 RF 기반 모델은 이미지 분류(image classification) [22], 광류 추적(optical flow tracking) [15], 오디오 처리(audio processing) [53, 75]를 포함한 시퀀스 모델링 과제에서 경쟁력 있는 성능을 입증했다. 더 나아가 RF 뉴런은 Loihi [9, 41]와 같은 뉴로모픽 하드웨어(neuromorphic hardware)에서도 효율적으로 구현될 수 있다.

### 2.2 스파이킹 신경망(Spiking Neural Networks)의 학습 전략(Training Strategies)

심층 SNN(deep SNN)의 주류 학습 방법은 ANN-to-SNN 변환(ANN-to-SNN conversion) [12, 47, 63]과 직접 학습(direct training) [68, 69]으로 구분할 수 있다. ANN-to-SNN 변환 방법은 스파이크 발화율(spike firing rate)과 ANN 활성화 함수(activation function) 간의 유사성을 이용하지만, 높은 정확도에 도달하기 위해 많은 시간 스텝이 필요하다. 반면 직접 학습은 제한된 수의 시간 스텝 내에서 동일한 구조를 가진 ANN과 비교 가능한 성능을 SNN이 달성하게 한다. 구체적으로, 직접 학습은 역전파(backpropagation)를 가능하게 하기 위해 대리 그래디언트 함수(surrogate gradient function) [11, 40]를 도입하며, 이를 통해 스파이크 발화 함수(spike firing function)의 비미분 가능성(non-differentiability)을 해결한다. 그러나 직접 학습을 장기 시퀀스 과제에 적용하는 것은 더 큰 도전을 수반한다 [51]. 이러한 과제는 종종 수천 개의 시간 스텝을 요구하기 때문이다. 따라서 일부 연구 [23, 58, 76]는 SNN을 위한 더 효율적인 학습 전략을 탐구하는 경향이 있다. Yin 등 [73]은 특히 뉴런의 리셋 거동(reset behavior)을 중심으로 SNN 학습의 내재적 도전을 추가로 분석했다. 이에 Fang 등 [14]은 스파이킹 뉴런의 동적 과정을 학습 가능한 행렬(learnable matrix)로 바꾸어 리셋 메커니즘을 피했다. 유사하게 Shen 등 [51]은 리셋 과정을 모사하기 위해 SDN 블록(block)을 도입했다. 이러한 방법은 비동기 추론(asynchronous inference) 능력을 유지하면서 SNN의 학습 비용을 크게 줄인다.

## 3 예비 지식(Preliminary)

### 3.1 공진-발화(Resonate-and-Fire, RF) 뉴런

포유류 신경계(mammalian nervous system)의 막 전위에서 관찰되는 감쇠되면서도 지속적인 역치 이하 진동(damped and sustained subthreshold oscillation) [2, 34, 42, 46]에서 영감을 받아 RF 뉴런이 제안되었다 [27]. 입력 신호 $I(t)$ 가 주어졌을 때, 시간 스텝 $t$ 에서 RF 뉴런의 동역학은 다음과 같이 기술될 수 있다.

$$
\frac{d}{dt} z(t) = (b + i\omega) z(t) + I(t)
\tag{1}
$$

$z = u + iv \in \mathbb{C}$ 는 RF 뉴런의 복소 상태(complex state)를 나타낸다. 여기서 $u$ 는 전압 개폐(voltage-gated) 및 시냅스 전류(synaptic current) 동역학을 포착하는 전류 유사 변수(current-like variable)이고, 허수 성분 $v$ 는 전압 유사 변수(voltage-like variable)에 해당한다. $\omega > 0$ 는 뉴런의 각주파수(angular frequency)로, 초당 진동하는 라디안 수를 나타낸다. 감쇠 계수(damping factor) $b < 0$ 는 진동의 지수 감쇠(exponential decay)를 조절한다. 이는 오일러 방법(Euler method) [3]을 사용하여 다음과 같이 이산화(discretization)할 수 있다.

$$
z[t] = \exp\{\delta(b + i\omega)\} \cdot z[t - 1] + \delta I[t]
\tag{2}
$$

$\delta$ 는 이산 시간 스텝(discrete timestep)이다. $z[t]$ 의 실수부(real part)가 임계값(threshold)을 초과하면 뉴런은 스파이크를 발화하고, 그렇지 않으면 침묵 상태를 유지한다. 또한 RF 뉴런은 특정 주파수 대역에 대한 선호를 보인다. 그림 1(a)에서는 서로 다른 주파수의 스파이크 입력에 대한 진동 거동(oscillatory behavior)을 제시한다. 막 전위와 위상 상태(phase state)가 모두 빠르게 누적됨을 관찰할 수 있다.

### 3.2 스파이킹 신경망(Spiking Neural Networks)에서의 직접 학습(Direct Training)

BPTT(backpropagation through time) [68] 및 대리 그래디언트(surrogate gradient) 방법 [40] 덕분에 대규모 SNN의 학습이 가능해졌다. 구체적으로, 시간 스텝 $T$ 에서 가중치 $w^l$ 의 그래디언트는 다음과 같이 표현된다.

$$
\nabla_{w^l} L = \sum_{l=1}^{T} \left( \frac{\partial L}{\partial u^l(t)} \right) S^{l-1}(t)
\tag{3}
$$

여기서 $u^l(t)$ 와 $S^l(t)$ 는 각각 시간 $t$ 에서 $l$ 번째 층(layer)의 막 전위(membrane potential)와 스파이크 방출(spike emission)을 나타낸다. $L$ 은 손실 함수(loss function)이다. 이는 다음과 같이 계산된다.

$$
\frac{\partial L}{\partial u^l(t)}
=
\frac{\partial L}{\partial S^l(t)}
\frac{\partial S^l(t)}{\partial u^l(t)}
+
\frac{\partial L}{\partial u^l(t + 1)}
\frac{\partial u^l(t + 1)}{\partial u^l(t)}
\tag{4}
$$

$$
\frac{\partial L}{\partial S^l(t)}
=
\frac{\partial L}{\partial u^{l+1}(t)}
\frac{\partial u^{l+1}(t)}{\partial S^l(t)}
+
\frac{\partial L}{\partial u^l(t + 1)}
\frac{\partial u^l(t + 1)}{\partial S^l(t)}
\tag{5}
$$

비미분 가능 항(non-differentiable term) $\frac{\partial S^l(t)}{\partial u^l(t)}$ 은 대리 함수(surrogate function)로 대체될 수 있다. $\frac{\partial u^l(t+1)}{\partial S^l(t)}$ 와 $\frac{\partial u^l(t+1)}{\partial u^l(t)}$ 는 계산되어야 하는 시간 그래디언트(temporal gradient)이다. 식 (4)와 식 (5)에서 보듯이, 각 시간 스텝의 그래디언트는 현재 상태뿐 아니라 미래 상태에도 재귀적으로 의존한다. 이러한 층과 시간 스텝 전반의 재귀 의존성은 $O(L^2)$ 의 계산 복잡도를 초래한다.

## 4 방법(Method)

### 4.1 문제 분석(Problem Analysis)

복잡한 과제(complex task)에서의 제한된 성능(Limited Performance): 감쇠 커널(decay kernel)과 복소값 상태(complex-valued state)의 설계 덕분에 RF 뉴런은 뚜렷한 주파수 선택성(frequency selectivity)을 보인다. 그러나 이러한 특성은 다양한 입력 패턴(diverse input pattern)을 구별하는 모델의 능력을 제한하기도 한다. 그림 1(b)와 같이, 서로 다른 주파수 성분을 가진 입력 신호들은 유사한 스파이킹 응답을 유도할 수 있다. 이는 뉴런의 고유 주파수(intrinsic frequency)와 맞지 않는 성분이 억제되어, 네트워크가 다양한 시간 특징(diverse temporal feature)을 포착하고 구분하는 능력이 저하되기 때문이다. 이 한계를 더 검증하기 위해 우리는 단일 RF 뉴런의 주파수 응답을 시각화하였다. 그 결과, 뉴런은 주로 좁은 대역폭 내에서 응답하고 고유 주파수에서 최대값을 보이며, 복잡한 주파수 조합(complex frequency composition)을 포착하기 어렵다는 사실이 드러났다. 이 관찰은 복잡한 시간 특징을 모델링할 때 RF 뉴런이 지니는 고유한 표현 한계(inherent representational limitation)를 보여준다.

그림 1. 문제 분석(Problem Analysis): (a) 서로 다른 스파이크 열(Response of Different Spike Trains): 주파수가 일치하는 입력은 막 전위의 빠른 누적을 유도하는 반면, 주파수가 일치하지 않는 입력은 더 약한 응답을 보인다. (b) 시계열에 대한 제한된 능력(Limited Ability to Time Series): 단일 RF 뉴런은 좁은 대역 선택성(narrow band selectivity) 때문에 서로 다른 주파수 변동 입력에 효과적으로 응답하기 어렵다. (c) 에너지 효율(Energy Efficiency) 대 학습 속도(Training Speed): 시간-가변(time-variant) 방법은 희소 스파이킹을 가능하게 하지만 $O(L^2)$ 의 높은 학습 비용을 가진다. 시간-불변(time-invariant) 방법은 $O(L \log L)$ 복잡도로 더 빠른 학습을 허용하지만, 지속적 발화(continuous fire)를 유발하는 경우가 많다.

에너지 효율과 학습 속도 간 균형의 도전(Challenges in Balancing Energy Efficiency and Training Speed): 장기 시퀀스 과제에서 RF 뉴런은 상당한 학습 비용과 잠재적으로 과도한 에너지 소비를 초래한다. 기존 연구는 두 가지 주요 전략에 초점을 둔다. 첫째, 그림 1(c).1과 같이 Higuchi 등 [21]은 각 스파이크 이후 감쇠 계수(decay coefficient)를 즉시 높이는 적응형 감쇠 계수(adaptive decay factor)를 제안하여 막 전위 진동을 억제하고 희소 스파이크를 촉진한다. 그러나 이 접근은 역전파 중 시간 전개(temporal unfolding)를 필요로 하므로 $O(L^2)$ 복잡도를 가진다. 둘째, Huang 등 [24]은 리셋 메커니즘에 학습 가능한 감쇠 계수(learnable decay factor)를 도입하고, 시간 동역학을 진동 커널(oscillatory kernel)과 입력 신호 사이의 병렬 합성곱(parallel convolutional process)으로 재구성하기 위해 푸리에 변환(Fourier transform) 재정식을 사용하여 계산 복잡도를 $O(L \log L)$ 로 낮춘다. 그러나 감쇠 계수가 시간-불변으로 유지되기 때문에 모델은 스파이크 이전 진폭(pre-spike amplitude)을 보존한다. 그림 1(c).2와 같이, 그 결과 지속적인 버스트 발화(sustained burst firing)가 발생하여 SNN의 저전력(low-power) 장점을 약화시킨다.

### 4.2 수상돌기 공진-발화(Dendritic Resonate-and-Fire) 뉴런

서로 다른 주파수 대역에 걸친 특징을 더 잘 포착하기 위해, 우리는 D-RF 뉴런 모델을 제안한다. 기본 RF 모델(vanilla RF model) [27]과 달리 D-RF 모델은 세포체(soma)와 여러 수상돌기 가지로 구성된다. 각 수상돌기 가지는 입력 신호 $I[t]$ 내의 특정 주파수 선호도(frequency preference)에 대응하는 상태 응답(state response)을 추출한다. $I[t]$ 가 어떤 가지의 선호도와 정렬된 주파수 성분을 포함하면, 해당 가지의 막 전위가 빠르게 누적된다. 세포체는 모든 수상돌기 가지로부터 입력 전류(input current)를 통합하고, 막 전위가 미리 정의된 임계값을 초과하면 스파이크를 생성한다. 구체적으로, $i$ 번째 수상돌기 가지의 막 전위 동역학은 다음과 같이 정의된다.

$$
\frac{d z_i(t)}{dt} = \left\{-\frac{1}{\tau_i} + i\omega_i\right\} \cdot z_i(t) + \gamma_i I(t)
\tag{6}
$$

여기서 $\tau_i$ 와 $\omega_i$ 는 각각 $i$ 번째 수상돌기 가지와 연관된 감쇠 계수(decay factor)와 막 전위 진동 계수(oscillation coefficient)를 나타낸다. $I(t)$ 는 시간 $t$ 에서의 시냅스 전 입력(presynaptic input)을 의미하고, $\gamma_i$ 는 $i$ 번째 수상돌기 가지의 막 정전용량(membrane capacitance)을 나타낸다. 이러한 모델링 틀은 서로 다른 수상돌기 가지가 특정 주파수 성분에 선택적으로 반응할 수 있게 한다. 효율적인 추론(inference)을 가능하게 하기 위해 영차 유지(Zero-Order Hold, ZOH) 방법 [10]을 사용하여 이산화한다. 모든 수상돌기의 막 전위 동역학은 다음과 같다.

$$
Z[t] =
\exp\left\{
\begin{bmatrix}
-\frac{1}{\tau_1} + i\omega_1 & 0 & \cdots & 0 \\
0 & -\frac{1}{\tau_2} + i\omega_2 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & -\frac{1}{\tau_n} + i\omega_n
\end{bmatrix}
\cdot \delta
\right\}
Z[t - 1] + \Gamma I[t]
\tag{7}
$$

여기서 $\delta$ 는 이산 시간 스텝이고, $Z = [z_1, z_2, \cdots, z_n]^T$ 는 개별 수상돌기 가지의 상태를 나타내며, $\Gamma = [\gamma_1, \gamma_2, \cdots, \gamma_n]^T$ 는 각 가지의 시정수(time constant)를 나타낸다. 수상돌기 가지 간 주파수 특성을 더욱 강화하기 위해 각 가지에는 개별 중요도 가중치(importance weight)가 부여된다. 세포체의 동역학은 다음과 같이 정의된다.

$$
H[t] = C\Re\{Z(t)\}, \qquad S[t] = \Theta(H[t] - V_{th}[t])
\tag{8}
$$

$C \in \mathbb{R}^{n \times 1}$ 는 각 수상돌기 가지에 부여된 중요도 가중치이다. $\Theta(\cdot)$ 는 헤비사이드 함수(Heaviside function)를 나타낸다. 세포체의 시냅스 전 막 전위(presynaptic membrane potential) $H[t]$ 가 임계값 $V_{th}$ 를 초과하면 스파이크 $S[t]$ 가 생성된다.

우리는 주파수 대역 응답(frequency band response)을 조사함으로써 수상돌기 설계의 효과를 추가로 분석한다. 감쇠 커널이 $b + i\omega$ 인 시간-불변 RF 뉴런은 커널 $h(n)$ 을 갖는 시간-불변 합성곱 과정(time-invariant convolutional process)으로 모델링할 수 있으며, $h(n) = \exp\{\delta(b + i\omega)\}^n$ 으로 정의된다. 상세한 증명은 부록 A에 제시한다. 그 주파수 응답은 다음과 같다.

$$
\left\|H(\exp\{i\Omega\})\right\|
=
\left|
\sum_{n=0}^{\infty} h(n)\exp\{-i\Omega n\}
\right|
=
\left|
\frac{\delta}{1 - \exp\{\delta b + i(\delta\omega - \Omega)\}}
\right|
\tag{9}
$$

따라서 단일 RF 뉴런은 $\Omega \approx \omega$ 에서 공진 피크(resonance peak)를 가지며, 감쇠 인자 $\delta b$ 가 결정하는 좁은 주파수 대역을 갖는 1차 대역통과 필터(first-order band-pass filter)로 볼 수 있다. 이는 그림 1(b)와 같다. 반면, 우리 D-RF 모델의 주파수 응답은 다음과 같이 정의된다.

$$
\left\|H_{D\text{-}RF}(\exp\{i\Omega\})\right\|
=
\sum_{i=1}^{n} C_i \cdot \left\|H(\exp\{i\Omega\})\right\|,
\qquad
B_{eff} \approx \sum_{i=1}^{n} \beta_i \left|\frac{\tau_i}{\delta}\right|
\tag{10}
$$

$B_{eff}$ 는 D-RF 뉴런의 총 주파수 응답(total frequency response)을 나타내고, $\beta_i \in [0, 1)$ 는 전체 주파수 커버리지(overall frequency coverage)에 대한 $i$ 번째 가지의 독립 기여(independent contribution)를 정량화한다. 결과적으로 단일 가지 대응 모델(single-branch counterpart)과 비교할 때 D-RF 뉴런은 훨씬 더 넓은 주파수 민감도(frequency sensitivity)를 제공한다.

### 4.3 가속되고 효율적인 학습을 위한 적응형 임계값(Adaptive Threshold for Accelerated and Efficient Learning)

학습 속도와 에너지 효율의 균형을 맞추기 위해, 우리는 이전 시간 스텝의 스파이킹 활동에 기반하여 임계값을 동적으로 조정하는 적응형 임계값 전략(adaptive thresholding strategy)을 제안한다. 구체적으로 시간 스텝 $t$ 에서의 임계값은 다음과 같이 정의된다.

$$
V_{th}[t]
=
\sum_{k=1}^{n} \alpha_k \Theta(\Re\{Z[t-k-1, \ldots, t-1]\} - V_{pre}) + V_{pre}
\tag{11}
$$

여기서 $V_{pre}$ 는 원래 임계값(origin threshold)으로 1로 설정되며, $\alpha_k \in (0, 1)$ 는 이전 스파이크(preceding spike)의 중요도를 나타낸다. 그림 2(b)와 같이 $\alpha_k$ 는 적응형 임계값 계산 과정에서 공유되는 파라미터(shared parameter)이다.

그림 2. D-RF 모델의 구조(The Structure of D-RF model): (a) 수상돌기 공진-발화 뉴런(Dendrite Resonate-and-Fire Neuron): D-RF 모델은 여러 수상돌기 가지로 구성되며, 각 가지는 주파수 특이적 동역학(frequency-specific dynamics)을 부호화한다. 세포체는 적응형 임계값 메커니즘을 통해 모든 가지의 막 전위를 통합하여 희소 스파이크를 가능하게 한다. (b) 적응형 임계값(Adaptive Threshold): 시간 $t$ 에서의 임계값은 과거 스파이킹 활동에 기반해 동적으로 결정된다. (c) 병렬 계산(Parallel Computation): 각 가지의 서로 다른 합성곱 커널(convolution kernel)과 적응형 임계값을 위한 인과적 합성곱(causal convolution)은 시간에 걸친 입력의 병렬 처리를 가능하게 한다.

이 과정은 커널 크기 $n$ 을 갖는 1차원 인과적 합성곱(one-dimensional causal convolution)으로 해석할 수 있으며, 이때 커널은 $A = [\alpha_1, \ldots, \alpha_n]$ 으로 정의된다. 스파이킹 과정은 다음과 같이 다시 쓸 수 있다.

$$
S[t]
=
\Theta\left\{
\underbrace{C^l \Re\{Z\}}_{\text{수상돌기 입력}}
-
\left(
\underbrace{\operatorname{Conv1d}(\Theta(C^l \Re\{Z\} - V_{pre}) + V_{pre})}_{\text{적응형 임계값}}
\right)[t]
\right\}
\tag{12}
$$

$\operatorname{Conv1d}(\cdot)$ 는 시간 영역(temporal domain)을 따라 수행되는 인과적 합성곱을 나타내며, 학습의 병렬화 가능성(parallelizable nature)을 유지하면서 더 희소한 스파이크 활동을 가능하게 한다. 우리는 순전파(forward propagation)와 역전파(backward propagation) 과정을 모두 분석함으로써 이 전략의 효과를 보인다.

순전파에서 D-RF 뉴런의 계산 복잡도는 다중 수상돌기 입력(multi-dendritic input)과 적응형 임계값 메커니즘에 의해 결정된다. 시간 전개(temporal unfolding) 없이도 D-RF 뉴런은 병렬화된 형태로 구성될 수 있다. 수상돌기 입력 성분의 경우 각 수상돌기 가지는 독립적으로 작동하고 시간적으로 분리되어 있다. 따라서 시간 $t$ 에서 모든 가지의 막 전위는 다음과 같이 정의된다.

$$
Z^l[t] = \sum_{k=0}^{t} \Gamma^l \exp\{k \cdot \delta D\} \cdot I^l[t-k]
\tag{13}
$$

$D$ 는 개별 수상돌기 가지의 진동 공진기(oscillatory resonator)를 특징짓는 대각 행렬(diagonal matrix)로,
$\operatorname{Diag}\{-1/\tau_1 + i\omega_1,\ -1/\tau_2 + i\omega_2,\ \cdots,\ -1/\tau_n + i\omega_n\}$ 로 정의된다. $D$ 가 시간-불변이므로 식 (13)은 입력 신호 $I$ 와 커널 $K$ 사이의 합성곱으로 다시 쓸 수 있다.

$$
Z[t] = (K * I)[t] = \mathcal{F}^{-1}\{\mathcal{F}\{K\} \cdot \mathcal{F}\{I\}\}[t],
\qquad
K = [\delta D^1, \delta D^2, \ldots, \delta D^n]
\tag{14}
$$

$\mathcal{F}(\cdot)$ 와 $\mathcal{F}^{-1}(\cdot)$ 는 각각 정방향 및 역 푸리에 변환(forward and inverse Fourier transform) 연산을 나타낸다. 따라서 충전 과정(charging process)은 추가적인 학습 오버헤드 없이 $O(L \log L)$ 의 계산 복잡도를 보장한다. 적응형 임계값 메커니즘은 합성곱 연산을 통해 구현되므로 GPU 가속(GPU acceleration)에 적합하다. 또한 $\alpha_k > 0$ 이므로, 이전 시간 스텝의 스파이크 수가 많을수록 시간 $t$ 에서의 임계값이 증가하여 더 희소한 스파이크를 유도한다.

역전파 단계에서는 적응형 임계값이 그래디언트 간 시간 의존성(temporal dependency)을 제거하여 고도로 병렬화된 학습을 가능하게 한다. $I[t] = w^l S^{l-1}[t]$ 라고 두면, 가중치 $w$ 에 대한 손실의 그래디언트는 다음과 같이 표현된다.

$$
\nabla_{w^l} L
=
\underbrace{
\sum_{t=0}^{T}
\frac{\partial L}{\partial S^l[t]}
\frac{\partial S^l[t]}{\partial Z^l[t]}
\frac{\partial Z^l[t]}{\partial w^l}
}_{\text{순차 학습(Sequential Training)}}
=
\underbrace{
\left\langle
\frac{\partial L}{\partial S^l[t]}
\frac{\partial S^l[t]}{\partial Z^l[t]},
(K * S^{l-1})[t]
\right\rangle
}_{\text{병렬 학습(Parallel Training)}}
\tag{15}
$$

$\langle \cdot, \cdot \rangle$ 는 내적(inner product)이다. 도함수 $\frac{\partial S^l[t]}{\partial Z^l[t]}$ 는 다음과 같이 정의할 수 있다.

$$
\frac{\partial S[t]}{\partial Z[t]}
=
C^l G\big(C^l \Re\{Z[t]\} - V_{th}[t]\big)
\frac{\partial \Re\{Z[t]\}}{\partial Z[t]}
\tag{16}
$$

$G(\cdot)$ 는 대리 그래디언트 함수이며, 본 연구에서는 이중 가우시안(double Gaussian) 함수 [72]로 구현한다. 식 (15)와 식 (16)에서 보듯이, 역전파 중 그래디언트는 현재 시간 스텝에만 의존한다. 따라서 적응형 임계값의 도입은 추가적인 학습 복잡도를 유발하지 않으면서도 희소 스파이크 활동을 장려하고 낮은 계산 비용을 유지한다. 상세한 증명은 부록 B에 제시한다.



## 5 실험(Experiment)

### 5.1 최신 기법(SOTA methods)과의 비교(Compare with the SOTA methods)

제안 방법의 효과를 검증하기 위해, 우리는 여러 시계열(time-series) 데이터셋에서 실험을 수행했다. 모든 실험은 최소 다섯 번 반복하였다. 먼저 250 시간 스텝의 Spiking Heidelberg Digits(SHD) [8], 784 시간 스텝의 Sequential MNIST와 Permuted Sequential MNIST(S/PS-MNIST) [31], 그리고 1024 시간 스텝의 더 어려운 Sequential CIFAR10(S-CIFAR10) [7]을 포함한 대표 데이터셋에서 D-RF와 다른 최신(state-of-the-art, SOTA) 모델들의 성능을 비교했다. 표 1에서 보듯이, D-RF는 더 적거나 비슷한 수의 파라미터(parameter)를 사용하면서도 SOTA 성능을 달성한다.

표 1. 다양한 모델의 성능 비교(Performance Comparison of Various Models)

#### S/PS-MNIST (784 time steps)

| 방법(Method) | 모델 크기(Model Size) | 유형(Type) | 병렬화(Parallel) | 수상돌기(Dendritic) | 정확도(Acc.) |
|---|---:|---|:---:|:---:|---:|
| LIF [79] | 85.1K | FF | ✗ | ✗ | 72.06 / 10.00 |
| ALIF [72] | 156.3K | Rec | ✗ | ✗ | 98.70 / 94.30 |
| BRF [21] | 68.9K | Rec | ✗ | ✗ | 99.10 / 95.20 |
| PSN [14] | 2.5M | FF | ✗ | ✓ | 97.90 / 97.80 |
| TC-LIF [79] | 155.1K | Rec | ✗ | ✓ | 99.20 / 95.36 |
| DH-LIF [80] | 0.8M | Rec | ✗ | ✓ | 98.9 / 94.52 |
| PMSN [7] | 156.4K | FF | ✓ | ✓ | 99.50 / 97.80 |
| Ours | 155.1K | FF | ✓ | ✓ | 99.50 / 98.20 |

#### SHD (250 time steps)

| 방법(Method) | 모델 크기(Model Size) | 유형(Type) | 병렬화(Parallel) | 수상돌기(Dendritic) | 정확도(Acc.) |
|---|---:|---|:---:|:---:|---:|
| LIF [74] | 249.0K | Rec | ✗ | ✗ | 84.00 |
| ALIF [72] | 141.3K | Rec | ✗ | ✗ | 84.40 |
| BRF [21] | 108.8K | Rec | ✗ | ✗ | 92.50 |
| PSN [14] | 232.5K | FF | ✓ | ✗ | 89.75 |
| TC-LIF [79] | 141.8K | Rec | ✗ | ✓ | 88.91 |
| DH-LIF [80] | 0.5M | Rec | ✗ | ✓ | 91.34 |
| PMSN [7] | 199.3K | FF | ✓ | ✓ | 95.10 |
| Ours | 155.1K | FF | ✓ | ✓ | 96.20 |

#### S-CIFAR10 (1024 time steps)

| 방법(Method) | 모델 크기(Model Size) | 유형(Type) | 병렬화(Parallel) | 수상돌기(Dendritic) | 정확도(Acc.) |
|---|---:|---|:---:|:---:|---:|
| LIF [74] | 0.18M | FF | ✗ | ✗ | 45.07 |
| PSN [14] | 6.47M | FF | ✓ | ✓ | 55.24 |
| SPSN [14] | 0.18M | FF | ✓ | ✓ | 70.23 |
| PMSN [7] | 0.21M | FF | ✓ | ✓ | 82.14 |
| Ours | 0.21M | FF | ✓ | ✓ | 84.30 |

추가 실험 결과는 RF 기반 모델이 모든 데이터셋에서 비슷한 크기의 LIF 모델보다 일관되게 더 우수함을 보여준다. 이는 RF 뉴런이 장기 시퀀스 과제에서 가진 시간 모델링 능력(temporal modeling capability)을 다시 한 번 확인해 준다. 또한 우리 모델은 다른 수상돌기 기반 모델보다도 더 우수한 인식 성능(recognition performance)을 보인다. DH-LIF 모델 [80]과 비교하면, 우리의 접근은 병렬 계산을 가능하게 하여 수상돌기 뉴런과 연관된 느린 학습 문제를 완화한다. PMSN 모델 [7]과 비교하면, 우리의 임계값 리셋 전략(threshold resetting strategy)은 빈번한 스파이크 생성을 방지하고 시간 의존성(temporal dependency)을 더 효과적으로 포착한다. 보다 어려운 Sequential CIFAR10 데이터셋에서 우리 방법은 84.20%의 인식 정확도를 달성하여 2.16%의 향상을 보인다.

또한 우리는 더 까다로운 LRA 벤치마크(Long Range Arena benchmark) [60]에서도 제안 방법을 평가했다. 표 2에서 보듯이, 우리 모델은 다른 신경 모델(neural model)에 비해 현저히 높은 인식 정확도(recognition accuracy)를 달성한다. 특히 ListOps 과제에서 60.02%의 정확도를 보였다. 더 긴 시간 스텝을 가진 과제에서도 강한 성능을 보이며, Text 과제(4096 time steps)에서는 86.52%, Retrieval 과제(4000 time steps)에서는 90.02%의 정확도를 기록했다. Image 과제에서는 D-RF가 85.32% 정확도를 달성했지만, 이는 SpikingSSM [51]보다 낮다. 이 격차는 SpikingSSM에서 LayerNorm을 사용하여 시간 변동(temporal variance)을 줄이기 때문에 발생한다. 더 나아가 S4 [19]와 같은 ANN 기반 접근과의 성능 격차는 3%를 넘지 않는다. 구체적으로 ListOps 과제에서 우리 모델의 정확도는 0.42% 더 높다. 이러한 결과는 우리 방법의 강한 시간 모델링 능력을 보여준다.

표 2. LRA 벤치마크에서의 모델 정확도 비교(Comparison of Model Accuracy on LRA Benchmark)

| 모델(Model) | SNN | ListOps (2,048) | Text (4,096) | Retrieval (4,000) | Image (1,024) | Pathfinder (1,024) | 평균(Avg.) |
|---|:---:|---:|---:|---:|---:|---:|---:|
| Random | - | 10.00 | 50.00 | 50.00 | 10.00 | 50.00 | 34.00 |
| Transformer [62] | ✗ | 36.37 | 64.27 | 57.46 | 42.44 | 71.40 | 54.39 |
| S4 (Bidirectional) [19] | ✗ | 59.60 | 86.82 | 90.90 | 88.65 | 94.20 | 84.03 |
| Binary S4D [57] | ✓ | 54.80 | 82.50 | 85.03 | 82.00 | 79.79 | 77.39 |
| → + GSU & GeLU | ✓ | 59.60 | 86.50 | 90.22 | 85.00 | 91.30 | 82.52 |
| SpikingSSMs [51] | ✓ | 60.23 | 80.41 | 88.77 | 88.21 | 93.51 | 82.23 |
| Spiking LMU [33] | ✓ | 37.30 | 65.80 | 79.76 | 55.65 | 72.68 | 62.23 |
| ELM Neuron [56] | ✓ | 44.55 | 75.40 | 84.93 | 49.62 | 71.15 | 69.25 |
| SD-TCM [24] | ✓ | 59.20 | 86.33 | 89.88 | 84.77 | 91.76 | 82.39 |
| Ours | ✓ | 60.02 | 86.52 | 90.02 | 85.32 | 92.36 | 82.88 |

### 5.2 가속된 학습과 더 희소한 스파이크(Sparser Spike with Accelerated Training)

D-RF 방법의 희소성(sparsity)을 평가하기 위해, 우리는 LRA 데이터셋 [60]에서 유사한 접근들과 스파이크 발화율(spike firing rate) 및 이론적 에너지 소비(theoretical energy consumption) [45]를 비교했다. 표 3에서 보듯이, 우리 모델은 뚜렷한 이점을 보인다. 구체적으로 ListOps 과제 [39]에서 9.8%의 스파이킹 비율과 62.48mJ의 에너지 소비를 달성한다. 또한 Image 과제 [30]에서 스파이크 발화율은 SD-TCM 방법에 비해 49.7% 감소한다. 이러한 결과는 적응형 임계값 메커니즘이 에너지 효율을 향상시키는 데 효과적임을 보여준다.

표 3. LRA 벤치마크 전반의 지표 비교(Comparison of Metrics across the LRA Benchmark)

#### 스파이킹 비율(Spiking Rate, %)

| 방법(Method) | ListOps | Text | Retrieval | Image | Pathfinder | 평균(Avg.) |
|---|---:|---:|---:|---:|---:|---:|
| SpikingSSM [51] | 13.2 | 10.1 | 6.9 | 22.1 | 7.4 | 11.9 |
| SD-TCM [24]† | 11.2 | 7.9 | 5.7 | 15.7 | 5.8 | 9.3 |
| Ours | 9.8 | 6.3 | 3.3 | 7.9 | 3.2 | 6.1 |

#### 에너지 비용(Energy Cost, mJ)

| 방법(Method) | ListOps | Text | Retrieval | Image | Pathfinder | 평균(Avg.) |
|---|---:|---:|---:|---:|---:|---:|
| SpikingSSM [51] | 84.2 | 355.2 | 237.0 | 708.9 | 65.1 | 290.1 |
| SD-TCM [24]† | 71.4 | 277.8 | 195.7 | 503.6 | 51.0 | 220.6 |
| Ours | 62.5 | 221.5 | 113.3 | 253.4 | 28.1 | 135.8 |

† 원 논문의 코드가 공개되어 있지 않아, 이 결과는 우리가 직접 재현한 값이다.

그림 3. 서로 다른 방법 간 스파이킹 거동 비교(Comparison of Spiking Behavior Across Different Methods): (a) SD-TCM 방법은 막 전위와 스파이크 출력에 반영되듯 지속적인 스파이킹 활동을 보인다. (b) D-RF 방법은 더 희소한 스파이크 생성을 보여주며, 이는 더 효율적인 스파이킹 거동을 시사한다.

또한 서로 다른 리셋 방법(reset method)의 스파이킹 거동을 시각화하여 우리 방법의 희소성을 추가로 확인했다. SD-TCM 모델 [24]은 정적 임계값(static threshold)과 함께 학습 가능한 감쇠 상수(learnable decay constant)가 리셋 과정을 효과적으로 근사한다고 가정한다. 반면 우리 방법은 이전 시간 스텝에 기반하여 임계값을 동적으로 조정한다. 그림 3에서 보듯이, 제안 방법은 스파이크 희소성을 크게 증가시키며, 이는 에너지 효율 측면에서 더 큰 잠재력을 의미한다.

우리 방법은 높은 학습 효율(training efficiency)도 보장한다. 우리는 서로 다른 시퀀스 길이(sequence length)에 대해 에폭(epoch)당 학습 비용을 비교했다. 그림 4(a)와 같이 배치 크기(batch size)를 128로 두었을 때의 평균 에폭 시간을 시각화했다. 그 결과, 제안 방법은 시퀀스 길이가 증가할수록 더 높은 실행 효율(execution efficiency)을 달성한다. 시퀀스 길이 32768에서 우리 방법은 BPTT [68] 대비 581배의 속도 향상을 달성한다. 또한 SDN 방법 [51]과 비교해도 경쟁력 있는 성능을 보이며 4.2배의 가속을 달성한다. 우리는 Text 과제(4096)에 대해서도 이 전략을 추가 검증했다. 그림 4(b)에서 보듯이, 막 전위 누적(membrane potential accumulation)과 스파이크 생성(spike generation) 과정이 고도로 병렬화되어 있기 때문에 D-RF는 GPU에서 더 빠른 시뮬레이션을 달성하며 1.1배의 속도 향상을 보인다. 더욱이 적응형 임계값 메커니즘은 직렬적인 누적-감쇠-발화(serial accumulation-decay-firing) 과정을 병렬화 가능한 형태로 변환할 수 있게 하여 최대 147배의 학습 가속을 가능하게 한다. 이러한 결과는 D-RF가 높은 학습 비용 문제를 효과적으로 해결함을 확인해 준다.

그림 4. D-RF의 학습 속도와 주파수 분석(Train Speed and Frequency Analysis of D-RF): (a) 학습 실행 시간 비교(Comparison of Training Runtime). (b) 순차 학습 대 병렬 학습 비용(Sequential vs. Parallel Training Cost): 제안 방법은 순전파와 역전파를 모두 가속한다. (c) D-RF의 주파수 응답(Frequency Responses of D-RF): D-RF는 더 넓은 주파수 응답을 보인다.

### 5.3 소거 실험(Ablation Experiment)

수상돌기 구조(dendritic structure)와 적응형 임계값 메커니즘의 영향을 평가하기 위해, 우리는 S-CIFAR10과 ListOps 데이터셋에서 소거 실험(ablation experiment)을 수행했다. 우리는 서로 다른 수상돌기 개수($n = 1, 4, 8, 16$)에 따른 성능을 비교했다.

표 4. 소거 실험(Ablation Experiment)

| 데이터셋(Dataset) | 방법(Method) | n=1 | n=4 | n=8 | n=16 |
|---|---|---:|---:|---:|---:|
| S-CIFAR10 | adaptive | 80.3 | 84.3 | 84.6 | 85.1 |
| S-CIFAR10 | w/o | 79.2 | 83.9 | 84.1 | 84.9 |
| ListOps | adaptive | 55.2 | 59.1 | 60.2 | 60.3 |
| ListOps | w/o | 54.2 | 58.9 | 59.2 | 59.6 |

표 4에서 보듯이, 모델 성능은 수상돌기 수가 증가할수록 향상된다. 복잡도(complexity)와 성능 사이의 절충을 고려하여, 우리는 S-CIFAR10 데이터셋에는 $n = 4$, LRA 데이터셋에는 $n = 8$ 을 사용했다. 그림 3(c)에서 각 수상돌기 가지의 주파수 응답을 시각화하였다. 결과는 제안된 D-RF 뉴런이 거의 전체 주파수 스펙트럼을 포착함을 보여주며, 수상돌기 구조의 효과를 다시 확인해 준다. 또한 적응형 임계값 메커니즘의 효과도 평가했다. 실험 결과는 적응형 임계값이 주로 중복 특징 정보(redundant feature information)가 결과에 미치는 부정적 영향을 효과적으로 억제함으로써 모델 성능을 향상시킨다는 점을 보여준다.

## 6 결론(Conclusion)

생물학적 뉴런의 수상돌기 구조에서 영감을 받아, 본 연구는 시계열 신호(time-series signal)에 대한 SNN의 성능을 더욱 향상시키기 위해 D-RF 모델을 제안한다. 이 모델은 다중 수상돌기와 세포체 구조로 이루어진다. 다중 수상돌기 구조는 서로 다른 감쇠 계수(decay factor)를 갖는 가지들로 구성되어, 뉴런이 입력 신호로부터 다중 주파수 정보(multi-frequency information)를 효과적으로 추출할 수 있게 한다. 적응형 임계값(adaptive threshold)을 갖는 세포체는 병렬화 가능한 계산(parallelizable computation)을 가능하게 하면서 희소한 스파이킹을 보장한다. 광범위한 실험은 우리 모델이 효율적 학습을 위해 희소한 스파이킹 활동을 유지하면서도 경쟁력 있는 결과를 달성함을 보여준다. 이러한 결과는 D-RF 방법이 에지 컴퓨팅 플랫폼(edge-computing platform)에서 효과적이고 효율적인 장기 시퀀스 모델링을 가능하게 할 강한 잠재력을 지님을 보여준다.

## 7 감사의 말(Acknowledgments)

본 연구는 부분적으로 중국 국가자연과학재단(National Natural Science Foundation of China, No. 62220106008 및 62271432), 부분적으로 쓰촨성 박사후 혁신 인재 지원 프로젝트(Sichuan Province Innovative Talent Funding Project for Postdoctoral Fellows, BX202405), 부분적으로 선전 과학기술 프로그램(Shenzhen Science and Technology Program, Shenzhen Key Laboratory, Grant No. ZDSYS20230626091302006), 부분적으로 광둥성 혁신·창업 팀 유치 프로그램(Program for Guangdong Introducing Innovative and Entrepreneurial Teams, Grant No. 2023ZT10X044), 그리고 부분적으로 뇌인지 및 뇌모사 지능기술 국가중점실험실(State Key Laboratory of Brain Cognition and Brain-inspired Intelligence Technology, Grant No. SKLBI-K2025010)의 지원을 받았다.



## 부록 A. 공진-발화 뉴런(Resonate-and-Fire Neuron)의 세부 사항(Detail)

### A.1 공진-발화 뉴런의 동역학(Dynamics of Resonate-and-Fire Neuron)

RF 뉴런 과정은 1차 선형 진동 시스템(first-order linear oscillatory system)으로 모델링할 수 있다.

$$
\frac{d}{dt} z(t) = (b + i\omega) z(t) + I(t)
\tag{1}
$$

여기서 $z(t) = u(t) + iv(t) \in \mathbb{C}$ 는 뉴런의 복소 상태를 나타낸다. $b < 0$ 는 감쇠 계수이고, $\omega > 0$ 는 각주파수이며, $I(t)$ 는 뉴런에 대한 외부 입력 신호(external input signal)이다.

### A.2 지수 오일러 방법(Exponential Euler Method)을 통한 이산화(Discretization)

우리는 연속 미분방정식(continuous differential equation)을 다음과 같이 지수 오일러 방법으로 이산화한다.

$$
z(t + \delta) = \exp\{(b + i\omega)\delta\} z(t) + \int_{t}^{t+\delta} \exp\{(b + i\omega)(t + \delta - s)\} I(s)\, ds
\tag{2}
$$

여기서 $\delta$ 는 이산 시간 스텝이다. 입력 $I(s)$ 가 작은 구간 $[t, t+\delta]$ 에서 거의 일정하다고 가정하면, 이 이산화는 다음과 같이 유도될 수 있다.

$$
\begin{aligned}
z(t + \delta)
&= \exp\{(b + i\omega)\delta\}z(t) + \int_{t}^{t+\delta} \exp\{(b + i\omega)(t + \delta - s)\}I(s)\, ds \\
&\approx \exp\{(b + i\omega)\delta\}z(t) + I(t)\int_{t}^{t+\delta} \exp\{(b + i\omega)(t + \delta - s)\}\, ds \\
&= \exp\{(b + i\omega)\delta\}z(t) + I(t)\left[ -\frac{1}{b + i\omega}\exp\{(b + i\omega)(t + \delta - s)\}\right]_{s=t}^{s=t+\delta} \\
&= \exp\{(b + i\omega)\delta\}z(t) + I(t)\left(\frac{-1}{b + i\omega}e^0 - \frac{-1}{b + i\omega}\exp\{(b + i\omega)\delta\}\right) \\
&= \exp\{(b + i\omega)\delta\}z(t) + I(t)\frac{1 - \exp\{(b + i\omega)\delta\}}{b + i\omega}.
\end{aligned}
\tag{3}
$$

$\delta \to 0$ 일 때, 두 번째 항은 테일러 전개(Taylor expansion)를 사용하여 다음과 같이 표현될 수 있다.

$$
I(t)\frac{1 - \exp\{(b + i\omega)\delta\}}{b + i\omega}
\approx
\delta I(t) + O(\delta^2)
\tag{4}
$$

따라서 식 (1)의 이산형은 다음과 같이 표현된다.

$$
z[t] = \exp\{(b + i\omega)\delta\}z[t - 1] + \delta I[t]
\tag{5}
$$

$z[t]$ 는 이산 시간 스텝 $t$ 에서의 복소 상태이며, $\delta$ 는 시간 간격(time interval), $I[t]$ 는 시간 스텝 $t$ 에서의 외부 입력이다.

### A.3 주파수 대역 선호 특성(Frequency Band Preference Characteristics)

주기 입력(periodic input)에 대한 RF 뉴런의 응답을 살펴보기 위해, 우리는 $I(t)$ 와 동일한 주파수를 갖는 식 (5)의 특수해(particular solution)를 찾는다.

$$
I[t] = I_0 \exp\{i\Omega t\delta\},
\qquad
z[t] = H \exp\{i\Omega n\delta\}
\tag{6}
$$

여기서 $H$ 는 응답의 진폭(amplitude)과 위상(phase)을 모두 결정하는 복소 상수(complex constant)이다. 이를 다음과 같이 다시 쓸 수 있다.

$$
H \exp\{i\Omega n\delta\}
=
\exp\{\delta(b + i\omega)\} H \exp\{i\Omega (n-1)\delta\}
+
\delta A \exp\{i\Omega n\delta\}
\tag{7}
$$

따라서 $H$ 는 다음과 같이 정의된다.

$$
H
=
\frac{\delta A}{\exp\{i\Omega\delta\} - \exp\{(b + i\omega)\delta\}}
=
\frac{\delta A}{\exp\{i\Omega\delta\}\left(1 - \exp\{(b + i\omega)\delta\}\exp\{-i\Omega\delta\}\right)}
\tag{8}
$$

전달 함수(transfer function)의 크기(magnitude)는 다음과 같다.

$$
\left\|H(\exp\{i\Omega\})\right\|
=
\left|
\frac{\delta}{1 - \exp\{\delta b + i(\delta\omega - \Omega)\}}
\right|
\tag{9}
$$

따라서 단일 RF 뉴런은 $\Omega \approx \omega$ 에서 공진 피크를 가지며, 감쇠 인자 $\delta b$ 가 결정하는 좁은 주파수 대역을 갖는 1차 대역통과 필터로 볼 수 있다.

RF 뉴런의 위상도(phase diagram)에 대해서는, 입력과 응답 사이의 위상 이동(phase shift)이 공진 특성(resonance property)에 대한 추가 정보를 제공한다.

$$
\phi(\Omega) = \arg(H\{i\Omega\}) = -\tan^{-1}\left(\frac{\Omega - \omega}{b}\right)
\tag{10}
$$

공진 시점인 $\Omega = \omega$ 에서 위상 이동은 $-90^\circ$ 이다. 따라서 공진이 발생하면 RF 뉴런의 위상이 빠르게 누적된다.

### A.4 공진-발화 뉴런에서의 리셋 메커니즘(Reset Mechanism in Resonate-and-Fire Neuron)

#### A.4.1 전통적 소프트 리셋(soft reset)과 하드 리셋(hard reset)

RF 뉴런에 대한 전통적 소프트 리셋과 하드 리셋 메커니즘은 다음과 같이 정의된다.

$$
\Im z[t] =
\begin{cases}
0, & \text{if } \Im\{z[t]\} > V_{th},\ \text{hard reset} \\
\Im\{z[t]\} - V_{th}, & \text{if } \Im\{z[t]\} > V_{th},\ \text{soft reset}
\end{cases}
\tag{11}
$$

식 (11)에서 보듯이, RF 뉴런 상태의 허수부(imaginary part)가 임계값을 넘으면 스파이크가 생성된다. 하드 리셋의 경우 RF 뉴런의 허수부는 0으로 리셋된다. 소프트 리셋의 경우 RF 뉴런의 허수부에서 임계값이 차감된다. 두 경우 모두 RF 뉴런의 원래 진동 동역학(oscillatory dynamics)을 교란한다.

리셋 조건하에서는 시스템이 비선형(nonlinear)이 되므로, Z-변환(Z-transform)을 직접 적용할 수 없다. 그러나 우리는 리셋 메커니즘을 비선형 섭동 항(nonlinear perturbation term)으로 모델링할 수 있다.

$$
z_R[t] = \exp\{(b + i\omega)\} \cdot z_R[t - 1] + I[t] + d[t]
\tag{12}
$$

여기서 $d[t]$ 는 리셋 연산(reset operation)에 의해 도입되는 섭동을 나타낸다.

$$
d[t] = \sum_{k} \big(z_R[t_k] - z[t_k]\big)\cdot \delta[t - t_k]
\tag{13}
$$

여기서 $\delta[t - t_k]$ 는 시간 $t_k$ 에서의 단위 임펄스(unit impulse)이다. Z-변환을 적용하면 다음을 얻는다.

$$
Z\{z_R[t]\} = \frac{Z\{I[t]\} + Z\{d[t]\}}{1 - \exp\{(b + i\omega)\}z^{-1}}
\tag{14}
$$

주기적 리셋 패턴(periodic reset pattern)의 경우, 주파수 응답은 일련의 고조파 성분(harmonic component)을 포함한다.

$$
H_R(\omega')
=
H(\omega')
+
\sum_{n=-\infty}^{\infty} c_n \cdot \delta(\omega' - \omega - n\omega_r)
\tag{15}
$$

여기서 $H(\omega') = \frac{1}{1 - e^{(b+i\omega)}e^{-i\omega'}}$ 는 원래 시스템의 주파수 응답이고, $c_n$ 은 푸리에 계수(Fourier coefficient), $\omega_r$ 는 리셋 주파수(reset frequency)이다. 이는 리셋 메커니즘이 주파수 영역(frequency domain)에 측대역 성분(sideband component)을 도입하여 RF 뉴런의 주파수 선택성을 약화시킴을 보여준다. 따라서 전통적 리셋 메커니즘은 어느 정도 RF 뉴런의 대역 선택성(band-selectivity) 특성을 불가피하게 훼손한다.

#### A.4.2 리셋 메커니즘이 없는 경우(No Reset Mechanism)

리셋 메커니즘이 없으면 RF 뉴런의 대역 선택성은 효과적으로 보존될 수 있지만, 빈번한 스파이크 방출(frequent spike emission)을 초래한다.

공진 주파수(resonant frequency)에서의 정현파 입력(sinusoidal input) $I[t] = I_0 \exp\{i\omega t\}$ 를 고려하자. 차분방정식(difference equation)의 해는 재귀적 전개(recursive expansion)를 통해 다음과 같이 얻어진다.

$$
\begin{aligned}
z[t]
&=
\exp\{(b + i\omega)t\}\cdot z[0]
+
I_0 \sum_{k=0}^{t-1}
\exp\{(b + i\omega)(t-k-1)\}\cdot \exp\{i\omega k\} \\
&=
\exp\{(b + i\omega)t\}\cdot z[0]
+
I_0 \cdot \exp\{i\omega t\}\cdot \frac{1 - \exp\{bt\}}{1 - \exp\{b\}}.
\end{aligned}
\tag{16}
$$

$b < 0$ 이고 $t \gg 0$ 일 때, 첫 번째 항은 사라지므로 다음을 얻는다.

$$
z[t] \approx \frac{I_0 \exp\{i\omega t\}}{1 - \exp\{b\}},
\qquad
\Im(z[t]) \approx \frac{I_0 \sin(\omega t)}{1 - \exp\{b\}}
\tag{17}
$$

따라서 장기 시퀀스 과제에서 RF 뉴런은 빈번한 스파이크 방출을 보일 가능성이 높다. 각 스파이크의 지속 시간(duration)은 다음과 같다.

$$
\Delta t_{spike}
=
\frac{1}{\omega}
\left(
\pi - 2\sin^{-1}\left(\frac{V_{th}(1 - \exp\{b\})}{I_0}\right)
\right)
\tag{18}
$$

여기서 $V_{th}$ 는 임계값이다. 최대 스파이크 지속 시간은 진동 주기의 절반에 해당한다. 그러나 리셋 메커니즘이 없는 경우, 시간-불변 파라미터 설계(time-invariant parameter design)는 RF 뉴런을 고정 커널 $K$ 와 입력 $I[t]$ 사이의 합성곱으로 해석할 수 있게 한다. 이 구조는 학습 중 명확한 이점을 제공하며, $O(L)$ 의 계산 복잡도만을 요구한다.

#### A.4.3 RF 뉴런을 위한 적응형 감쇠 계수(Adaptive Decay Factor)

RF 뉴런의 주파수 대역 선택성과 발화율(firing rate) 감소 사이의 균형을 맞추기 위해, Higuchi 등 [21]은 RF 뉴런에 불응기 메커니즘을 도입하여 과도한 스파이크 방출을 효과적으로 억제했다. 이 메커니즘은 두 가지 주요 구성요소로 이루어진다.

$$
V_{th}(t) = \theta_c + q(t),
\qquad
b(t) = b_c - q(t)
\tag{19}
$$

여기서 $\theta_c$ 는 상수 임계값(constant threshold)이고, $b_c$ 는 감쇠 계수이다. $q(t)$ 는 시간에 따라 지수적으로 감쇠하는 불응기 기간(refractory period)이다.

$$
q(t) = \gamma q(t - 1) + s[t - 1]
\tag{20}
$$

여기서 $\gamma = 0.9$ 는 기본 기간 상수(default period constant)이다. 우리는 시간-가변(time-varying) 설계인 $b(t)$ 가 주파수 응답과 결합될 때 RF 뉴런의 주파수 대역 선택성을 훼손하지 않음을 추가로 분석한다. 식 (6)부터 식 (9)까지와 같이, 시간 $t$ 에서 BRF 뉴런의 주파수 응답은 다음과 같이 정의된다.

$$
H(t,\omega') = \frac{1}{1 - \exp\{(b(t) + i\omega - i\omega')\}}
\tag{21}
$$

이는 시간 $t$ 에서 입력 주파수 $\omega'$ 에 대한 시스템 응답 강도(response strength)를 나타낸다. 이 식이 $\omega' = \omega$ 에서 최대값을 가진다는 사실은 도함수를 계산함으로써 확인할 수 있다. 식 (21)의 극점(extreme point) 조건을 구하기 위해 도함수를 취하면 다음과 같다.

$$
\frac{\partial |H(t,\omega')|}{\partial \omega'}
=
k^2 \cdot
\frac{\partial}{\partial \omega'}
\left[
(1 - \exp\{b(t)\}\cos(\omega - \omega'))^2 + (\exp\{b(t)\}\sin(\omega - \omega'))^2
\right]^{-1/2}
\tag{22}
$$

여기서 $k = |H(t,\omega')|$ 이다. 뒤의 항은 다음과 같이 더 단순화할 수 있다.

$$
\begin{aligned}
\frac{\partial |H(t,\omega')|}{\partial \omega'}
&=
-\frac{1}{2}k^3 \cdot
\frac{\partial}{\partial \omega'}
\left[
(1 - \exp\{b(t)\}\cos(\omega - \omega'))^2 + (\exp\{b(t)\}\sin(\omega - \omega'))^2
\right] \\
&=
-\frac{1}{2}k^3 \cdot
\frac{\partial}{\partial \omega'}
\left[
1 - 2\exp\{b(t)\}\cos(\omega - \omega') + \exp\{2b(t)\}
\right] \\
&=
-\frac{1}{2}k^3 \cdot
\left[
2\exp\{b(t)\}\sin(\omega - \omega')
\right].
\end{aligned}
\tag{23}
$$

따라서 어떤 $b$ 에 대해서도 $\omega' = \omega$ 에서 $|H(t,\omega')|$ 의 1차 도함수는 0이며, 이는 정상점(stationary point)이다. 따라서 2차 도함수를 구하면 다음과 같다.

$$
\left.
\frac{\partial^2 |H(t,\omega')|}{\partial \omega'^2}
\right|_{\omega'=\omega}
=
-|H(t,\omega)|^3 \cdot e^{b(t)} < 0
\tag{24}
$$

따라서 $\omega$ 는 최대점(maximum point)이다. $\omega' = \omega$ 일 때 대역 응답은 최대가 된다. 또한 스파이크가 방출되면 $b(t)$ 값은 감소하여 더 빠른 감쇠율(faster decay rate)을 유도한다.

RF 뉴런의 경우 리셋 메커니즘이 없으면 빈번한 스파이크 방출이 발생하고, 단순히 임계값을 빼는 방식은 주파수 선택성 특성을 손상시킨다. Higuchi 등 [21]은 RF 뉴런에서 에너지 효율과 주파수 선택성을 모두 유지하지만, 이들의 시간-가변 인자는 모델 학습을 크게 방해한다. 따라서 RF 뉴런의 잠재력을 충분히 활용하기 위해서는 효과적이며 원리적인 리셋 메커니즘을 설계하는 것이 중요하다.



## 부록 B. D-RF 모델에서 역전파(backpropagation)의 계산 복잡도(Computational Complexity)

### B.1 역전파를 위한 그래디언트 계산(Gradient Calculation for Backpropagation)

역전파 과정을 보다 명확히 하기 위해, 우리는 손실 함수(loss function)를 가중치 $w^l$ 에 대해 미분한 그래디언트로부터 시작한다.

$$
\nabla_{w^l} L
=
\sum_{t=0}^{T}
\frac{\partial L}{\partial S^l[t]}
\cdot
\frac{\partial S^l[t]}{\partial Z^l[t]}
\cdot
\frac{\partial Z^l[t]}{\partial w^l}
\tag{25}
$$

$S^l[t]$ 는 비선형 활성화 함수(nonlinear activation function)를 통해 막 전위 $Z^l[t]$ 로부터 유도되므로, 스파이크 신호(spike signal)를 막 전위에 대해 미분한 값을 계산해야 한다. 본 연구에서는 이를 이중 가우시안 함수(double Gaussian function) [72]로 구현한다. 따라서 이 도함수는 다음과 같이 주어진다.

$$
\frac{\partial S^l[t]}{\partial Z^l[t]}
=
C^l G\big(C^l \Re\{Z[t]\} - V_{th}[t]\big)
\frac{\partial \Re\{Z[t]\}}{\partial Z[t]}
\tag{26}
$$

여기서 $G(\cdot)$ 는 대리 그래디언트 함수이고, $C^l$ 는 현재 층의 계수(coefficient), $V_{th}[t]$ 는 적응형 임계값이다. 막 전위 $Z^l[t]$ 는 이전 층의 스파이크 신호 $S^{l-1}[t]$ 의 가중합(weighted sum)이므로, 가중치 $w^l$ 에 대한 막 전위의 도함수는 다음과 같다.

$$
\frac{\partial Z^l[t]}{\partial w^l}
=
\sum_{k=0}^{t}
\Gamma^l \exp\{k \cdot \delta D\} \cdot S_{l-1}[t-k]
\tag{27}
$$

여기서 $S_{l-1}[t-k]$ 는 이전 층의 스파이크 신호이며, $\Gamma_l$ 는 상수, $\delta D$ 는 수상돌기 가지와 관련된 파라미터이다. 위 식들을 그래디언트 식에 대입하면 다음을 얻는다.

$$
\nabla_{w^l} L
=
\sum_{t=0}^{T}
\frac{\partial L}{\partial S^l[t]}
\cdot
C^l G\big(C^l \Re\{Z[t]\} - V_{th}[t]\big)
\cdot
\left(
\sum_{k=0}^{t}
\Gamma_l \exp\{k \cdot \delta D\} \cdot S_{l-1}[t-k]
\right)
\tag{28}
$$

보다 빠른 학습을 위해, 우리는 그래디언트 계산을 합성곱을 이용한 병렬 계산 형식으로 변환한다. 따라서 최종 식은 다음과 같다.

$$
\nabla_{w^l} L
=
\left\langle
\frac{\partial L}{\partial S^l[t]}
\cdot
\frac{\partial S^l[t]}{\partial Z^l[t]},
(K * S^{l-1})[t]
\right\rangle
\tag{29}
$$

여기서 $\langle \cdot,\cdot \rangle$ 는 내적이며, $(K * S^{l-1})[t]$ 는 합성곱 연산을 나타낸다. 최종 그래디언트 식은 다음과 같이 요약된다.

$$
\nabla_{w^l} L
=
\sum_{t=0}^{T}
\frac{\partial L}{\partial S^l[t]}
\cdot
\frac{\partial S^l[t]}{\partial Z^l[t]}
\cdot
\frac{\partial Z^l[t]}{\partial w^l}
=
\left\langle
\frac{\partial L}{\partial S^l[t]}
\cdot
\frac{\partial S^l[t]}{\partial Z^l[t]},
(K * S^{l-1})[t]
\right\rangle
\tag{30}
$$

이 과정은 그래디언트 계산이 오차 항(error term)에서 최종 가중치 업데이트 규칙(weight update rule)까지 어떻게 진행되는지를 보여주며, 효율성을 위해 순차 학습과 병렬 학습 전략을 모두 포함한다.

### B.2 계산 복잡도 분석(Analysis of Computational Complexity)

식 (30)에서 보듯이, 제안 방법은 그래디언트 계산을 합성곱 연산으로 변환함으로써 고속 푸리에 변환(Fast Fourier Transform, FFT)의 효율을 활용한다. 그 결과 합성곱 계산 시간은 $O(L \log L)$ 이 된다. 길이 $L$ 의 FFT는 $O(L \log L)$ 에 계산될 수 있고, 합성곱은 주파수 영역에서의 곱셈과 동등하므로, 합성곱의 복잡도는 순차 학습에서의 $O(L^2)$ 에서 병렬 학습에서의 $O(L \log L)$ 로 감소한다. 따라서 합성곱 기반 병렬 학습 접근을 사용하면 역전파 과정 전체의 계산 복잡도는 $O(L \log L)$ 로 줄어들며, 훨씬 더 효율적이 된다.

이 분석은 병렬 계산과 FFT 기반 합성곱을 활용함으로써 학습 비용을 크게 줄일 수 있음을 보여준다. 즉, 복잡도가 $O(L^2)$ 에서 $O(L \log L)$ 로 감소하며, 이는 장기 시퀀스 모델링 과제에서 핵심적이다.



## 부록 C. 실험 세부 사항(Experiment Detail)

### C.1 데이터셋 설명(Dataset Description)

우리는 (P)S-MNIST와 LRA [60]의 각 과제에 대해 더 많은 맥락과 세부 정보를 제공한다. 아래 설명은 주로 [55]를 참고하였다.

- Sequential MNIST(S-MNIST) [31]: 각 $28 \times 28$ 그레이스케일 MNIST 이미지를 길이 784의 1차원 스칼라 시퀀스로 재구성한다. 모델은 이 시간 입력(temporal input)만을 기반으로 10개의 숫자 레이블(0-9) 중 하나를 예측해야 한다.
- Permuted Sequential MNIST(PS-MNIST) [31]: S-MNIST와 동일한 평탄화(flattening) 절차를 따라 길이 784 시퀀스를 만든 뒤, 이를 고정된 순열(fixed permutation)로 재배열하여 0-9 숫자를 분류한다.
- Spiking Heidelberg Digits(SHD) [8]: Heidelberg Digits 데이터셋의 손글씨 숫자(0-9)를 나타내는 10,000개의 스파이크 부호화 패턴(spike-encoded pattern)으로 구성된다. 각 샘플은 분류를 위해 이진 또는 연속 스파이크 열(binary or continuous spike train)로 제공된다.
- Sequential CIFAR-10(S-CIFAR10) [7, 14]: 각 $32 \times 32$ 컬러 이미지를 1,024개의 RGB 삼중항(RGB triplet) 시퀀스로 변환한다. 과제는 이 순차 표현(sequential representation)에 기반하여 이미지를 10개 범주 중 하나로 분류하는 것이다.
- ListOps [39]: 시퀀스 기반 모델(sequence-based model)을 평가하기 위한 데이터셋이다. 최솟값(min), 최댓값(max) 같은 수학 연산과 0-9 범위의 정수 피연산자(integer operand)를 전위 표기(prefix notation)와 괄호로 표현한다. 과제는 수학식의 결과를 계산하는 것이다. 예를 들어 $[\max\ 2\ 6\ [\min\ 9\ 7]\ 0] \rightarrow 7$ 이다. 문자는 원-핫 벡터(one-hot vector)로 부호화되며, 시퀀스는 최대 길이 2,000까지 패딩된다. 식의 결과를 나타내는 10개의 클래스가 존재한다.
- Text [36]: iMDB 영화 리뷰 데이터셋을 기반으로 한 감성 분류(sentiment classification) 데이터셋이다. 과제는 주어진 영화 리뷰가 긍정인지 부정인지 분류하는 것이다. 문자는 정수 토큰(integer token)으로 부호화되며, 시퀀스는 최대 길이 4,096까지 패딩된다. 긍정과 부정을 나타내는 2개의 클래스가 있다. 데이터셋은 25,000개의 학습 샘플과 25,000개의 테스트 샘플로 구성된다.
- Retrieval [43]: ACL Anthology Network 코퍼스(corpus)를 기반으로 한 데이터셋이다. 과제는 두 텍스트 인용(text citation)이 동일한지 여부를 분류하는 것이다. 각 인용은 정수 토큰 시퀀스로 부호화된다. 참고 문헌(reference)은 별도로 압축된 후 최종 분류층(final classification layer)에 전달된다. 이 데이터셋은 텍스트 관계(textual relation)를 표현하고 검색(retrieve)하는 모델의 능력을 평가한다.
- Image [30]: CIFAR-10 데이터셋을 기반으로 한 이미지 분류(image classification) 데이터셋이다. $32 \times 32$ 그레이스케일 이미지를 길이 1,024의 시퀀스로 평탄화한다. 과제는 각 이미지를 10개 범주 중 하나로 분류하는 것이다. 데이터셋은 45,000개의 학습 샘플, 5,000개의 검증 샘플, 10,000개의 테스트 샘플을 포함한다.
- Pathfinder [32]: 경로 찾기(path finding) 과제를 위한 데이터셋이다. $32 \times 32$ 그레이스케일 이미지로 구성되며, 각 이미지는 작은 원으로 표시된 시작점(start point)과 끝점(end point)을 포함한다. 과제는 시작점과 끝점을 연결하는 점선(dashed line) 또는 경로(path)가 존재하는지 여부를 분류하는 것이다. 시퀀스는 길이 1,024로 패딩된다. 데이터셋은 160,000개의 학습 샘플, 20,000개의 검증 샘플, 20,000개의 테스트 샘플을 포함한다. 데이터는 $[-1, 1]$ 범위로 정규화(normalization)되었다.

### C.2 모델 구조(Model Architecture)와 하이퍼파라미터 설정(Hyperparameter Setting)

본 과제에서 우리는 막 전위 사이에 적층된 잔차 블록(residual block, RB)으로 구성된 네트워크 구조를 사용한다. 각 RB 블록은 잔차 연결(residual connection)과 `D-RF Model – 1×1 convolution – Spiking Neuron model – 1×1 convolution`의 순서로 이루어진다. 뉴런 동역학에 수반될 수 있는 부동소수점 곱셈을 제외하면, 다른 모든 연산은 스파이크 구동(spike-driven)으로 수행되며, 이는 자원이 제한된(resource-constrained) 에지 디바이스(edge device)에 더 유리하다. 모든 실험은 NVIDIA GeForce RTX 4090(24G 메모리)와 Intel(R) Xeon(R) Platinum 8370C CPU@2.80GHz를 장착한 Ubuntu 서버에서 수행되었다. 소프트웨어 환경은 PyTorch 2.1.0과 CUDA 11.8이다.

또한 우리는 LRA 데이터셋에서 우리 모델과 기준선 방법(baseline method)의 파라미터 수를 비교하여, 제안 접근의 효율성을 추가로 강조했다. 뉴런 특이적 파라미터(neuron-specific parameter)에 대한 학습률(learning rate)은 0.001로 설정하였고, 네트워크 전체에는 0.005의 전역 학습률(global learning rate)을 적용하였다. 자세한 하이퍼파라미터 설정은 표 5에 정리하였다.

표 5. LRA 과제를 위한 하이퍼파라미터(Hyperparameters for LRA Task)

| 과제(Task) | 깊이(Depth) | 정규화(Norm) | 채널(Channels) | 사전 정규화(Pre-norm) | 드롭아웃(Dropout) | B | 에폭(Epochs) | 가중치 감쇠(Weight Decay) |
|---|---:|---|---:|:---:|---:|---:|---:|---:|
| ListOps | 8 | BatchNorm | 128 | False | 0 | 50 | 40 | 0.05 |
| Text | 6 | BatchNorm | 256 | True | 0 | 16 | 32 | 0.05 |
| Retrieval | 6 | BatchNorm | 256 | True | 0 | 32 | 20 | 0.05 |
| Image | 6 | BatchNorm | 512 | False | 0.1 | 50 | 200 | 0.05 |
| Pathfinder | 6 | BatchNorm | 256 | True | 0.05 | 64 | 200 | 0.03 |

표 6은 다양한 LRA 과제에 대한 모델 크기(model size)를 보여주며, 우리 접근이 다른 기준선과 비교해도 파라미터 효율(parameter efficiency)을 유지함을 보여준다. 표 2의 성능 지표와 표 3의 에너지 소비 결과를 함께 보면, 우리 모델은 경쟁력 있는 정확도를 달성할 뿐 아니라 훨씬 더 희소한 스파이킹 활동을 유도함을 알 수 있다. 이러한 결과는 에지 플랫폼에서 장기 시퀀스 모델링을 위한 효과적이고 효율적인 해법으로서의 잠재력을 강조한다.

표 6. 과제 전반에서 경쟁 네트워크 간 모델 크기 비교(Comparison of Model Size between Competing Networks across Tasks)

| 지표(Metric) | 방법(Method) | S-CIFAR10 | ListOps | Text | Retrieval | Image | Pathfinder |
|---|---|---:|---:|---:|---:|---:|---:|
| Parm. | S4 [19] | 308K | 815K | 843K | 3.6M | 3.6M | 1.3M |
| Parm. | SpikingSSM [51] | 308K | 815K | 843K | 3.6M | 3.6M | 1.3M |
| Parm. | SD-TCM [24]† | - | 272K | 830K | 1.1M | 4.1M | 1.3M |
| Parm. | Ours | 216K | 297K | 841K | 1.1M | 3.2M | 1.3M |

† 원 논문의 코드가 공개되어 있지 않아, 이 결과는 우리가 직접 재현한 값이다.

추가로, 우리는 소거 연구(ablation study)에서 서로 다른 수상돌기 가지 수에 대응하는 모델 파라미터 구성을 보고한다. 수상돌기 수를 늘리면 모델은 커지고 일반적으로 인식 성능도 향상되지만, 그 이득은 점차 포화(saturate)되는 경향을 보인다. 특히 이미지 분류 과제에서 4개의 수상돌기를 사용했을 때 모델은 86.2%의 정확도를 달성했고, 이를 8개로 늘려도 86.7%로 소폭 증가하는 데 그쳤다. 파라미터 오버헤드(parameter overhead)에 비해 성능 향상이 제한적이므로, 우리는 S-CIFAR10 과제의 기본 설정(default configuration)으로 $n = 4$ 를 선택한다.

표 7. 소거 실험(Ablation Experiment)

| 데이터셋(Dataset) | 지표(Metric) | 길이(Length) | n=1 | n=2 | n=4 | n=6 | n=8 | n=16 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| S-CIFAR10 | Accuracy (%) | 1024 | 80.3 | 82.1 | 84.3 | 84.4 | 84.6 | 85.1 |
| S-CIFAR10 | Parameter (M) | 1024 | 209K | 213K | 216K | 219K | 222K | 234K |



## 부록 D. 추가 논의(Further Discussion)와 한계(Limitation)

추가 논의(Further Discussion): Transformer 구조와 유사하게, D-RF 뉴런 역시 적층 가능한 계산 구성요소(stackable computational component)로 기능할 수 있다. 이들은 LRA 벤치마크와 S-CIFAR10 데이터셋 같은 다양한 순차 벤치마크(sequential benchmark)에서 강한 성능을 보였다. 그러나 이들 데이터셋의 시퀀스 길이는 현재 대규모 언어 모델(large language model, LLM)에서 사용되는 길이에 비하면 여전히 상대적으로 제한적이다. 최근 ANN 기반 LLM은 16K에서 1M 토큰(token)에 이르는 입력 시퀀스를 처리할 수 있지만 [1, 25, 71], 이러한 능력은 일반적으로 복잡한 구조 설계와 대규모 파라미터에 의존하므로 상당한 계산 오버헤드(computational overhead)를 수반한다. 본 논문은 신경 동역학(neural dynamics)의 관점에서 접근하여 메모리 용량(memory capacity)과 시간 정보 처리(temporal information processing)를 향상시키는 것을 목표로 한다.

한계(Limitation): 본 연구에는 두 가지 한계가 있다. 첫째, DRF 뉴런은 효율적 계산을 통해 장기 시퀀스 과제에서 모델 성능을 향상시키지만, 추가적인 MAC 연산과 복소값 연산(complex-valued operation)에 대한 의존성 때문에 뉴로모픽 칩(neuromorphic chip)에서의 배포가 제한될 수 있다. 둘째, 본 연구는 주로 분류 과제(classification task)에 초점을 맞추고 있으며, 텍스트 생성(text generation)이나 장기 시퀀스 예측(long sequence prediction) 같은 회귀 과제(regression task)는 다루지 않는다. 따라서 향후 연구에서는 텍스트 생성 과제로 확장하고, 뉴로모픽 칩에서의 구현 전략(implementation strategy)을 추가로 탐구할 것이다.
