# SPIKFORMER: 스파이킹 신경망(Spiking Neural Network, SNN)과 트랜스포머(Transformer)의 만남

(원문: *SPIKFORMER: WHEN SPIKING NEURAL NETWORK MEETS TRANSFORMER*)  
arXiv:2209.15425v2 [cs.NE] 22 Nov 2022

**Zhaokun Zhou**$^{1,2}$, **Yuesheng Zhu**$^{1,2,*}$, **Chao He**$^{4}$, **Yaowei Wang**$^{2}$, **Shuicheng Yan**$^{3}$, **Yonghong Tian**$^{1,2}$, **Li Yuan**$^{1,2,*}$

- $^1$ Peking University  
- $^2$ Peng Cheng Laboratory  
- $^3$ Sea AI Lab  
- $^4$ Shenzhen EEGSmart Technology Co., Ltd.  

연락처: {yuanli-ece}@pku.edu.cn  
\* 교신저자(corresponding author)

---

## 초록(Abstract)

우리는 생물학적으로 타당한(biologically plausible) 두 구조인 스파이킹 신경망(Spiking Neural Network, SNN)과 자기-어텐션(self-attention) 메커니즘을 고려한다. 전자는 에너지 효율적이면서 이벤트 기반(event-driven) 딥러닝 패러다임을 제공하고, 후자는 특징 의존성(feature dependencies)을 포착하는 능력을 통해 트랜스포머(Transformer)가 우수한 성능을 내도록 한다. 직관적으로 두 구조의 결합을 탐색하는 것은 매우 유망하다. 본 논문에서는 자기-어텐션의 표현 능력과 SNN의 생물학적 특성을 함께 활용하기 위해, 새로운 스파이킹 자기-어텐션(Spiking Self Attention, SSA)과 이를 기반으로 한 강력한 프레임워크인 스파이킹 트랜스포머(Spiking Transformer, Spikformer)를 제안한다. Spikformer의 SSA 메커니즘은 소프트맥스(softmax) 없이 스파이크 형태(spike-form)의 쿼리(Query), 키(Key), 밸류(Value)를 사용하여 희소(sparse) 시각 특징을 모델링한다. SSA의 계산은 희소하고 곱셈(multiplication)을 회피하므로 효율적이며, 계산 에너지 소비(computational energy consumption)가 낮다. SSA를 적용한 Spikformer는 뉴로모픽(neuromorphic) 및 정적(static) 데이터셋 모두에서 이미지 분류 성능이 기존 SNN 계열(state-of-the-art SNN-like) 프레임워크를 능가함을 보인다. 예를 들어 Spikformer(파라미터 66.3M)은 SEW-ResNet-152(60.2M, 69.26%)와 유사한 모델 크기임에도, 4 타임스텝(time steps)으로 ImageNet에서 74.81% top-1 정확도를 달성하며, 이는 직접 학습(directly trained)된 SNN 모델 중 최신(state-of-the-art) 성능이다. 코드는 Spikformer에서 제공될 예정이다.

---

## 1 서론(Introduction)

스파이킹 신경망(Spiking Neural Network, SNN)은 제3세대 신경망(Maass, 1997)으로서, 낮은 전력 소비(low power consumption), 이벤트 기반(event-driven) 특성, 그리고 생물학적 그럴듯함(biological plausibility) 때문에 매우 유망하다(Roy et al., 2019). 인공신경망(Artificial Neural Networks, ANNs)의 발전과 함께, SNN은 ResNet 계열 SNN(ResNet-like SNNs)(Hu et al., 2021a; Fang et al., 2021a; Zheng et al., 2021; Hu et al., 2021b), 스파이킹 순환 신경망(Spiking Recurrent Neural Networks)(Lotfi Rezaabad & Vishwanath, 2020), 스파이킹 그래프 신경망(Spiking Graph Neural Networks)(Zhu et al., 2022) 등 ANN의 진보된 아키텍처를 차용하여 성능을 끌어올릴 수 있었다. 한편 트랜스포머(Transformer)는 원래 자연어 처리(natural language processing)를 위해 설계되었으나(Vaswani et al., 2017), 이미지 분류(image classification)(Dosovitskiy et al., 2020; Yuan et al., 2021a), 객체 검출(object detection)(Carion et al., 2020; Zhu et al., 2020; Liu et al., 2021), 의미 분할(semantic segmentation)(Wang et al., 2021; Yuan et al., 2021b) 및 저수준 이미지 처리(low-level image processing)(Chen et al., 2021) 등 컴퓨터 비전 전반으로 확산되었다. 트랜스포머의 핵심 구성요소인 자기-어텐션(self-attention)은 관심 있는 정보에 선택적으로 집중하며, 인간 생물학적 시스템의 중요한 특징이기도 하다(Whittington et al., 2022; Caucheteux & King, 2022). 직관적으로, 두 메커니즘의 생물학적 특성을 고려할 때, 더 진보된 딥러닝을 위해 SNN에 자기-어텐션을 적용하는 것을 탐구하는 것은 흥미롭다.

하지만 자기-어텐션 메커니즘을 SNN으로 옮기는 것은 간단하지 않다(non-trivial). 표준 자기-어텐션(vanilla self-attention, VSA)(Vaswani et al., 2017)에는 쿼리(Query), 키(Key), 밸류(Value)라는 세 구성요소가 있다. 그림 1(a)에서 보이듯이, VSA의 표준 추론(inference)은 먼저 부동소수점(float-point) 형태의 쿼리와 키의 내적(dot product)으로 행렬을 얻은 뒤, 지수(exponential) 계산과 나눗셈(division) 연산을 포함하는 소프트맥스(softmax)로 이를 정규화하여 어텐션 맵(attention map)을 만들고, 이 어텐션 맵으로 밸류(Value)를 가중합한다. 이러한 VSA의 계산 단계는 SNN의 계산 특성(예: 곱셈 회피, 이벤트 기반 계산)과 잘 부합하지 않는다. 더구나 VSA의 막대한 계산 오버헤드(computational overhead)는 SNN에 이를 직접 적용하는 것을 거의 불가능하게 만든다. 따라서 SNN에서 트랜스포머를 개발하려면, 곱셈을 회피할 수 있고 계산 효율이 높은 새로운 자기-어텐션 변형(variant)을 설계해야 한다.

우리는 그림 1(b)에示한 스파이킹 자기-어텐션(Spiking Self Attention, SSA)을 제시한다. SSA는 처음으로 SNN에 자기-어텐션 메커니즘을 도입하며, 스파이크 시퀀스(spike sequences)를 이용해 상호 의존성(interdependence)을 모델링한다. SSA에서는 쿼리(Query), 키(Key), 밸류(Value)가 모두 스파이크 형태로, 0과 1만 포함한다. SNN에서 자기-어텐션 적용을 가로막는 핵심 장애물은 소프트맥스(softmax)이다.  
1) 그림 1에서 보이듯이, 스파이크 형태의 쿼리와 키로 계산한 어텐션 맵은 본질적으로 음수가 없는(non-negative) 성질을 가지며, 이는 관련 없는 특징을 자연스럽게 무시한다. 따라서 VSA에서 소프트맥스가 수행하는 가장 중요한 역할(어텐션 행렬을 비음수로 만드는 것)(Qin et al., 2022)을 SSA에서는 굳이 필요로 하지 않는다.  
2) SSA의 입력과 밸류(Value)는 0/1 스파이크로 구성되어, ANN의 VSA에서 사용하는 부동소수점 입력/밸류에 비해 세밀한(fine-grained) 정보가 적다. 그러므로 이러한 스파이크 시퀀스를 모델링하는 데 부동소수점 쿼리/키 및 소프트맥스 함수는 불필요(redundant)하다. 표 1은 스파이크 시퀀스 처리 측면에서 SSA가 VSA와 경쟁력 있음을 보여준다. 이러한 통찰을 바탕으로, 우리는 SSA의 어텐션 맵에서 소프트맥스 정규화를 제거한다. 이전의 일부 트랜스포머 변형들도 소프트맥스를 버리거나 선형 함수로 대체한 바 있다. 예를 들어 Performer(Choromanski et al., 2020)는 양의 랜덤 특징(positive random feature)으로 소프트맥스를 근사하고, CosFormer(Qin et al., 2022)는 소프트맥스를 ReLU와 코사인 함수로 대체한다.

이와 같은 SSA 설계를 통해, 스파이크 형태의 쿼리/키/밸류 계산은 곱셈을 피하고 논리 AND(logical AND) 연산과 덧셈(addition)으로 수행될 수 있다. 또한 SSA는 매우 효율적이다. 부록 D.1에 보이듯 쿼리/키/밸류가 희소(sparse)한 스파이크 형태이며, 계산도 단순하므로 SSA의 연산 수(number of operations)는 작다. 그 결과 SSA의 에너지 소비(energy consumption)는 매우 낮다. 더 나아가 소프트맥스를 제거한 SSA는 분해 가능(decomposable)하므로, 시퀀스 길이(sequence length)가 한 헤드(head)의 특징 차원보다 클 때 그림 1(b) ① ②와 같이 계산 복잡도를 더 줄일 수 있다.

SNN의 계산 특성에 잘 맞도록 설계된 SSA를 바탕으로, 우리는 스파이킹 트랜스포머(Spiking Transformer, Spikformer)를 개발한다. Spikformer의 개요는 그림 2에示한다. Spikformer는 정적 데이터셋과 뉴로모픽 데이터셋 모두에서 학습 성능을 향상시킨다. 우리가 아는 한, 이는 SNN에서 자기-어텐션 메커니즘과 직접 학습(directly trained) 트랜스포머를 탐구한 최초의 시도이다. 본 연구의 공헌(contribution)은 다음과 같다.

- SNN의 특성에 맞는 새로운 스파이크 형태 자기-어텐션, 스파이킹 자기-어텐션(Spiking Self Attention, SSA)을 설계하였다. 희소한 스파이크 형태의 쿼리/키/밸류를 소프트맥스 없이 사용함으로써, SSA 계산은 곱셈을 회피하며 효율적이다.
- 제안한 SSA를 기반으로 스파이킹 트랜스포머(Spikformer)를 개발하였다. 우리가 아는 한, 이는 SNN에서 자기-어텐션과 트랜스포머를 구현한 최초의 시도이다.
- 광범위한 실험을 통해 제안 아키텍처가 정적 및 뉴로모픽 데이터셋 모두에서 최신(state-of-the-art) SNN을 능가함을 보였다. 특히 4 타임스텝만으로 ImageNet에서 74%를 넘는 정확도를 달성한 것은 직접 학습된 SNN 모델에서 최초이다.

---

## 그림 1

**그림 1: 표준 자기-어텐션(vanilla self-attention, VSA)과 제안하는 스파이킹 자기-어텐션(Spiking Self Attention, SSA)의 비교 도식.** 빨간색 스파이크(spike)는 해당 위치의 값이 1임을 의미한다. 파란색 점선 박스는 행렬 내적(matrix dot product) 연산의 예시를 제공한다. 설명의 편의를 위해 SSA의 헤드(head) 중 하나만을 사용했으며, $N$은 입력 패치(patch) 수, $d$는 한 헤드의 특징 차원(feature dimension)이다. FLOPs는 부동소수점 연산(floating point operations), SOPs는 이론적 시냅스 연산(theoretical synaptic operations)을 의미한다. 한 타임스텝에서 쿼리(Query), 키(Key), 밸류(Value) 간 계산을 수행하는 데 필요한 이론적 에너지 소비량(theoretical energy consumption)은 (Kundu et al., 2021b; Hu et al., 2021a)을 따라 ImageNet 테스트셋에서 8-encoder-block, 512-embedding-dimension Spikformer를 기준으로 산출하였다. SOP와 에너지 소비량 계산의 자세한 내용은 부록 C.2에 포함되어 있다.  
(a) VSA에서 $Q_F$, $K_F$, $V_F$는 부동소수점 형태이다. $Q_F$와 $K_F$의 내적(dot-product) 후 소프트맥스 함수가 어텐션 맵의 음수 값을 양수로 정규화한다.  
(b) SSA에서는 어텐션 맵의 모든 값이 비음수이며, 스파이크 형태의 $Q, K, V$를 사용한 계산은 희소하다(예: VSA의 $77 \times 10^6$ 대비 $5.5 \times 10^6$). 따라서 SSA는 VSA(354.2µJ)보다 에너지 소비가 적다. 또한 SSA는 분해 가능(decomposable)하며($Q, K, V$의 계산 순서를 바꿀 수 있음) 계산 복잡도를 더 줄일 수 있다.

---

## 2 관련 연구(Related Work)

### 비전 트랜스포머(Vision Transformers)

이미지 분류 과제에서 표준 비전 트랜스포머(vision transformer, ViT)는 패치 분할(patch splitting) 모듈, 트랜스포머 인코더 층(layer)들, 그리고 선형 분류 헤드(linear classification head)로 구성된다. 트랜스포머 인코더 층은 자기-어텐션 층과 다층 퍼셉트론(multi-layer perceptron, MLP) 블록으로 이뤄진다. 자기-어텐션은 ViT의 성공을 가능하게 한 핵심 구성요소이다. 쿼리와 키의 내적(dot-product)과 소프트맥스(softmax) 함수를 통해 이미지 패치 특징값(feature value)에 가중치를 부여함으로써, 자기-어텐션은 전역 의존성(global dependence)과 관심 표현(interest representation)을 포착할 수 있다(Katharopoulos et al., 2020; Qin et al., 2022). ViT 구조 개선을 위한 연구도 진행되어 왔다. 예를 들어 패치 분할에 컨볼루션 층(convolution layers)을 사용하면 수렴(convergence)을 가속하고, ViT의 데이터 요구량(data-hungry) 문제를 완화할 수 있음이 알려졌다(Xiao et al., 2021b; Hassani et al., 2021). 또한 자기-어텐션의 계산 복잡도를 줄이거나 시각적 의존성(visual dependencies) 모델링 능력을 향상시키려는 방법도 제안되었다(Song, 2021; Yang et al., 2021; Rao et al., 2021; Choromanski et al., 2020). 본 논문은 SNN에서 자기-어텐션의 효과를 탐색하고, 이미지 분류를 위한 강력한 스파이킹 트랜스포머 모델을 개발하는 데 초점을 둔다.

### 스파이킹 신경망(Spiking Neural Networks)

연속적인 실수값(continuous decimal values)으로 정보를 전달하는 전통적 딥러닝 모델과 달리, SNN은 이산적인 스파이크 시퀀스(discrete spike sequences)를 사용해 정보를 계산하고 전달한다. 스파이킹 뉴런(spiking neurons)은 연속값을 받아 스파이크 시퀀스로 변환한다. 대표적으로 누설 적분-발화(Leaky Integrate-and-Fire, LIF) 뉴런(Wu et al., 2018), PLIF(Fang et al., 2021b) 등이 있다. 심층(deep) SNN을 얻는 방법은 크게 두 가지: ANN-to-SNN 변환(conversion)과 직접 학습(direct training)이다.

ANN-to-SNN 변환(Cao et al., 2015; Hunsberger & Eliasmith, 2015; Rueckauer et al., 2017; Bu et al., 2021; Meng et al., 2022; Wang et al., 2022)에서는, 고성능으로 사전학습된(pre-trained) ANN을 ReLU 활성화 층을 스파이킹 뉴런으로 바꾸는 방식으로 SNN으로 변환한다. 변환된 SNN은 ReLU 활성화를 정확히 근사하기 위해 큰 타임스텝(time steps)이 필요하며, 이로 인해 지연(latency)이 커진다(Han et al., 2020). 직접 학습 분야에서는 SNN을 시뮬레이션 타임스텝에 따라 펼친(unfolded) 뒤, 시간에 따른 역전파(backpropagation through time) 방식으로 학습한다(Lee et al., 2016; Shrestha & Orchard, 2018). 스파이킹 뉴런의 이벤트 트리거(event-triggered) 메커니즘은 미분 불가능(non-differentiable)하기 때문에, 역전파를 위해 대리 그래디언트(surrogate gradient)를 사용한다(Lee et al., 2020; Neftci et al., 2019). Xiao et al. (2021a)은 평형 상태(equilibrium state)에서의 암묵적 미분(implicit differentiation)을 이용해 SNN을 학습한다. 여러 ANN 모델들이 SNN으로 포팅(ported)되어 왔지만, SNN에서의 자기-어텐션 연구는 아직 공백에 가깝다. Yao et al. (2021)은 불필요한 타임스텝을 줄이기 위해 시간적 어텐션(temporal attention)을 제안했다. Zhang et al. (2022a;b)은 제목에 “Spiking Transformer”가 포함되어 있지만, 스파이크 데이터를 처리하기 위해 ANN-Transformer를 사용한다. Mueller et al. (2021)은 ANN-to-SNN 변환 기반 트랜스포머를 제공하지만, 여전히 표준 자기-어텐션(vanilla self-attention)을 사용하므로 SNN 특성과 맞지 않는다. 본 논문에서는 SNN에서 자기-어텐션과 트랜스포머를 구현할 수 있는지의 가능성을 탐색한다.

SNN의 기본 단위인 스파이킹 뉴런(spike neuron)은 입력 전류(resultant current)를 받아 막전위(membrane potential)를 누적하며, 막전위가 임계값(threshold)을 넘는지 비교하여 스파이크(spike)를 생성할지 결정한다. 우리는 모든 실험에서 LIF 스파이킹 뉴런을 사용한다. LIF의 동적 모델(dynamic model)은 다음과 같다.

$$
H[t] = V[t-1] + \frac{1}{\tau}\left(X[t] - (V[t-1] - V_{\text{reset}})\right), \qquad (1)
$$

$$
S[t] = \Theta(H[t] - V_{\text{th}}), \qquad (2)
$$

$$
V[t] = H[t](1 - S[t]) + V_{\text{reset}} S[t], \qquad (3)
$$

여기서 $\tau$는 막 시간 상수(membrane time constant)이고, $X[t]$는 타임스텝 $t$에서의 입력 전류(input current)이다. 막전위 $H[t]$가 발화 임계값(firing threshold) $V_{\text{th}}$를 초과하면, 스파이킹 뉴런은 스파이크 $S[t]$를 발생시킨다. $\Theta(v)$는 헤비사이드 스텝 함수(Heaviside step function)로, $v \ge 0$이면 1, 그렇지 않으면 0이다. $V[t]$는 트리거 이벤트(trigger event) 이후의 막전위이며, 스파이크가 생성되지 않으면 $H[t]$와 같고, 스파이크가 생성되면 리셋 전위(reset potential) $V_{\text{reset}}$로 된다.

---

## 3 방법(Method)

우리는 스파이킹 트랜스포머(Spikformer)를 제안한다. Spikformer는 SNN에 자기-어텐션 메커니즘과 트랜스포머를 결합하여 학습 능력을 향상시킨다. 이하에서는 Spikformer의 전체 구조와 각 구성요소를 설명한다.

---

## 그림 2

**그림 2: 스파이킹 트랜스포머(Spiking Transformer, Spikformer)의 개요.** Spikformer는 스파이킹 패치 분할(Spiking Patch Splitting, SPS) 모듈, Spikformer 인코더(encoder), 그리고 선형 분류 헤드(linear classification head)로 구성된다. 우리는 경험적으로 레이어 정규화(layer normalization, LN)가 SNN에 적용되지 않음을 발견하여, 대신 배치 정규화(batch normalization, BN)를 사용한다.

### 3.1 전체 아키텍처(Overall Architecture)

Spikformer의 개요는 그림 2에示한다. $2$차원 이미지 시퀀스 $I \in \mathbb{R}^{T \times C \times H \times W}$[^1]가 주어졌을 때, 스파이킹 패치 분할(Spiking Patch Splitting, SPS) 모듈은 이를 $D$차원 스파이크 형태 특징 벡터(feature vector)로 선형 투영(linearly project)하고, $N$개의 평탄화된(flattened) 스파이크 형태 패치(patches) 시퀀스 $x$로 분할한다. 부동소수점 형태의 위치 임베딩(position embedding)은 SNN에서 사용할 수 없다. 우리는 조건부 위치 임베딩 생성기(conditional position embedding generator)(Chu et al., 2021)를 사용하여 스파이크 형태의 상대 위치 임베딩(relative position embedding, RPE)을 생성하고, 이를 패치 시퀀스 $x$에 더하여 $X_0$를 얻는다. 조건부 위치 임베딩 생성기는 $3$ 크기의 커널을 가진 $2$D 컨볼루션 층(Conv2d), 배치 정규화(BN), 그리고 스파이킹 뉴런 층(spike neuron layer, $SN$)으로 구성된다. 그 다음 $X_0$를 $L$블록(block)의 Spikformer 인코더로 통과시킨다. 표준 ViT 인코더 블록과 유사하게, Spikformer 인코더 블록은 스파이킹 자기-어텐션(SSA)과 MLP 블록으로 구성되며, SSA와 MLP 블록 모두에 잔차 연결(residual connection)을 적용한다. Spikformer 인코더 블록의 핵심 구성요소인 SSA는 소프트맥스 없이 스파이크 형태의 쿼리 $Q$, 키 $K$, 밸류 $V$를 사용해 이미지의 지역-전역 정보(local-global information)를 효율적으로 모델링한다(자세한 분석은 3.3절). 인코더 출력 특징에 전역 평균 풀링(global average pooling, GAP)을 적용하여 $D$차원 특징을 얻고, 이를 완전연결 분류 헤드(fully connected classification head, CH)에 입력하여 예측 $Y$를 출력한다. Spikformer는 다음과 같이 쓸 수 있다.

$$
x = \mathrm{SPS}(I), \quad I \in \mathbb{R}^{T \times C \times H \times W}, \quad x \in \mathbb{R}^{T \times N \times D}, \qquad (4)
$$

$$
\mathrm{RPE} = SN\big(\mathrm{BN}(\mathrm{Conv2d}(x))\big), \quad \mathrm{RPE} \in \mathbb{R}^{T \times N \times D}, \qquad (5)
$$

$$
X_0 = x + \mathrm{RPE}, \quad X_0 \in \mathbb{R}^{T \times N \times D}, \qquad (6)
$$

$$
X'_l = \mathrm{SSA}(X_{l-1}) + X_{l-1}, \quad X'_l \in \mathbb{R}^{T \times N \times D}, \; l=1\ldots L, \qquad (7)
$$

$$
X_l = \mathrm{MLP}(X'_l) + X'_l, \quad X_l \in \mathbb{R}^{T \times N \times D}, \; l=1\ldots L, \qquad (8)
$$

$$
Y = \mathrm{CH}(\mathrm{GAP}(X_L)). \qquad (9)
$$

[^1]: 뉴로모픽 데이터셋에서는 데이터 형태가 $I \in \mathbb{R}^{T \times C \times H \times W}$이며, $T, C, H, W$는 각각 타임스텝(time step), 채널(channel), 높이(height), 너비(width)를 의미한다. 정적 데이터셋의 $2$D 이미지 $I_s \in \mathbb{R}^{C \times H \times W}$는 $T$번 반복하여 이미지 시퀀스를 만든다.

### 3.2 스파이킹 패치 분할(Spiking Patch Splitting)

그림 2에서 보이듯이, 스파이킹 패치 분할(Spiking Patch Splitting, SPS) 모듈은 이미지를 $D$차원 스파이크 형태 특징으로 선형 투영하고, 고정된 크기의 패치로 분할하는 것을 목표로 한다. SPS는 여러 블록을 포함할 수 있다. Vision Transformer(Xiao et al., 2021b; Hassani et al., 2021)의 컨볼루션 스템(convolutional stem)과 유사하게, 우리는 각 SPS 블록에 컨볼루션 층을 적용하여 Spikformer에 귀납적 편향(inductive bias)을 도입한다. 구체적으로, 이미지 시퀀스 $I \in \mathbb{R}^{T \times C \times H \times W}$가 주어졌을 때:

$$
x = \mathrm{MP}\big(SN(\mathrm{BN}(\mathrm{Conv2d}(I)))\big), \qquad (10)
$$

여기서 $\mathrm{Conv2d}$와 $\mathrm{MP}$는 각각 $2$D 컨볼루션 층(스트라이드 1, 커널 크기 $3 \times 3$)과 맥스 풀링(max-pooling)을 의미한다. SPS 블록의 개수는 1개 이상일 수 있다. 여러 SPS 블록을 사용할 경우, 컨볼루션 층의 출력 채널 수(output channels)를 점진적으로 증가시키며, 최종적으로 패치 임베딩(embedding) 차원과 일치하도록 한다. 예를 들어 출력 임베딩 차원이 $D$이고 SPS 모듈이 4블록이라면, 네 개 컨볼루션 층의 출력 채널 수는 각각 $D/8, D/4, D/2, D$가 된다. 또한 각 SPS 블록 뒤에는 고정된 크기의 $2$D 맥스 풀링 층을 적용하여 특징 크기(feature size)를 다운샘플링(down-sample)한다. SPS를 거친 후, $I$는 이미지 패치 시퀀스 $x \in \mathbb{R}^{T \times N \times D}$로 분할된다.

### 3.3 스파이킹 자기-어텐션 메커니즘(Spiking Self Attention Mechanism)

Spikformer 인코더는 아키텍처의 핵심 구성요소로, 스파이킹 자기-어텐션(SSA)과 MLP 블록을 포함한다. 본 절에서는 SSA에 집중하며, 먼저 표준 자기-어텐션(VSA)을 간단히 복습한다. 입력 특징 시퀀스 $X \in \mathbb{R}^{T \times N \times D}$에 대해, ViT의 VSA는 부동소수점 형태의 쿼리 $Q_F$, 키 $K_F$, 밸류 $V_F$라는 세 구성요소를 가지며, 이는 학습 가능한 선형 행렬 $W_Q, W_K, W_V \in \mathbb{R}^{D \times D}$와 $X$로부터 다음과 같이 계산된다(여기서 $F$는 부동소수점 형태임을 의미한다).

$$
Q_F = X W_Q,\quad K_F = X W_K,\quad V_F = X W_V. \qquad (11)
$$

VSA의 출력은 다음과 같이 계산된다.

$$
\mathrm{VSA}(Q_F, K_F, V_F) = \mathrm{Softmax}\left(\frac{Q_F K_F^\top}{\sqrt{d}}\right) V_F, \qquad (12)
$$

여기서 $d = D/H$는 한 헤드(head)의 특징 차원이며, $H$는 헤드 수(head number)이다. 부동소수점 형태의 밸류 $V_F$를 스파이크 형태의 $V$로 바꾸면, VSA를 SNN에 직접 적용하는 형태를 다음과 같이 표현할 수 있다.

$$
\mathrm{VSA}(Q_F, K_F, V) = \mathrm{Softmax}\left(\frac{Q_F K_F^\top}{\sqrt{d}}\right) V. \qquad (13)
$$

그러나 VSA 계산은 다음 두 이유로 SNN에 적합하지 않다.  
1) $Q_F$, $K_F$의 부동소수점 행렬 곱(multiplication)과 지수/나눗셈을 포함하는 소프트맥스는 SNN의 계산 규칙과 맞지 않는다.  
2) VSA는 시퀀스 길이 $N$에 대해 공간/시간 복잡도가 $O(N^2)$로 증가하는데, 이는 SNN이 요구하는 효율적 계산 요구사항을 충족하지 못한다.

우리는 그림 1(b)와 그림 2 하단에示한 스파이킹 자기-어텐션(Spiking Self-Attention, SSA)을 제안한다. SSA에서는 먼저 학습 가능한 행렬을 통해 쿼리 $Q$, 키 $K$, 밸류 $V$를 계산한 뒤, 서로 다른 스파이킹 뉴런 층을 통해 스파이킹 시퀀스로 변환한다.

$$
Q = SN_Q(\mathrm{BN}(XW_Q)),\quad K = SN_K(\mathrm{BN}(XW_K)),\quad V = SN_V(\mathrm{BN}(XW_V)), \qquad (14)
$$

여기서 $Q, K, V \in \mathbb{R}^{T \times N \times D}$이다. 우리는 어텐션 행렬(attention matrix) 계산 과정이 순수한 스파이크 형태의 쿼리와 키(0과 1만 포함)를 사용해야 한다고 본다. VSA(Vaswani et al., 2017)에 영감을 받아, 행렬 곱 결과의 값이 지나치게 커지는 것을 제어하기 위한 스케일링 계수(scaling factor) $s$를 추가한다. $s$는 SSA의 성질을 바꾸지 않는다. 그림 2에서 보이듯, 스파이크 친화적인(spike-friendly) SSA는 다음과 같이 정의한다.

$$
\mathrm{SSA}_0(Q, K, V) = SN\left(Q K^\top V \cdot s\right), \qquad (15)
$$

$$
\mathrm{SSA}(Q, K, V) = SN\left(\mathrm{BN}(\mathrm{Linear}(\mathrm{SSA}_0(Q, K, V)))\right). \qquad (16)
$$

여기서 소개한 단일 헤드(single-head) SSA는 부록 A에 자세히 설명된 방식으로 다중 헤드(multi-head) SSA로 손쉽게 확장할 수 있다. SSA는 각 타임스텝마다 독립적으로 수행되며, 자세한 내용은 부록 B에 제시한다. 식 (15)에서처럼, SSA는 식 (12)의 소프트맥스 정규화를 제거하고, $Q, K, V$를 직접 곱하여 계산한다. 그림 1(b)에 직관적 계산 예시를示하였다.

우리 SSA에서는 소프트맥스가 불필요할 뿐 아니라 SNN에서 자기-어텐션 구현을 오히려 방해할 수 있다. 형식적으로, 식 (14)에서 $SN_Q$와 $SN_K$가 출력하는 스파이크 시퀀스 $Q, K$는 자연스럽게 비음수(0 또는 1)이므로, 어텐션 맵은 비음수이다. SSA는 관련 있는 특징만 집계(aggregate)하고 관련 없는 정보는 무시한다. 따라서 어텐션 맵의 비음수성을 보장하기 위해 소프트맥스가 필요하지 않다. 또한 ANN의 부동소수점 입력 $X_F$ 및 밸류 $V_F$에 비해, SNN의 자기-어텐션에서 입력 $X$와 밸류 $V$는 스파이크 형태로 정보량이 제한적이다. 이 경우 부동소수점 형태의 $Q_F$, $K_F$ 및 소프트맥스 기반 VSA는 이러한 스파이크 형태 $X, V$를 모델링하는 데 불필요하며, SSA가 $X, V$로부터 VSA보다 더 많은 정보를 얻을 수 없다. 즉 SSA가 VSA보다 SNN에 더 적합하다.

표 1에示한 것처럼, 우리는 SSA의 타당성을 검증하기 위해 제안 SSA와 네 가지 다른 어텐션 맵 계산 방식(attention map variants)을 비교하는 실험을 수행하였다.

---

## 표 1

**표 1: SSA의 타당성(rationality) 분석.** SSA를 다른 어텐션 변형으로 대체하고, Spikformer의 나머지 네트워크 구조는 그대로 유지하였다. CIFAR10-DVS(Li et al., 2017), CIFAR10/100(Krizhevsky, 2009)에서의 정확도(Acc)를 나타낸다. OPs(M)은 연산 수(number of operations)이며, AI, ALeakyReLU, AReLU, Asoftmax에서는 OPs가 FLOPs이고 SOPs는 무시된다. ASSA에서는 OPs가 SOPs이다. P(µJ)는 $Q, K, V$ 간 계산 한 번을 수행할 때의 이론적 에너지 소비량(theoretical energy consumption)이다.

| 방법(Method) | CIFAR10-DVS Acc/OPs(M)/P(µJ) | CIFAR10 Acc/OPs(M)/P(µJ) | CIFAR100 Acc/OPs(M)/P(µJ) |
|---|---:|---:|---:|
| AI | 79.40 / 16.8 / 77 | 93.96 / 6.3 / 29 | 76.94 / 6.3 / 29 |
| ALeakyReLU | 79.80 / 16.8 / 77 | 93.85 / 6.3 / 29 | 76.73 / 6.3 / 29 |
| AReLU | 79.40 / 16.8 / 77 | 94.34 / 6.3 / 29 | 77.00 / 6.3 / 29 |
| Asoftmax | 80.00 / 19.1 / 88 | 94.97 / 6.6 / 30 | 77.92 / 6.6 / 30 |
| ASSA | 80.90 / 0.66 / 0.594 | 95.19 / 1.1 / 0.990 | 77.86 / 1.3 / 1.170 |

- AI: 부동소수점 $Q$와 $K$를 직접 곱해(multiply) 어텐션 맵을 얻는 방식으로, 양/음의 상관(positive and negative correlation)을 모두 보존한다.  
- AReLU: $\mathrm{ReLU}(Q)$와 $\mathrm{ReLU}(K)$의 곱으로 어텐션 맵을 얻는다. AReLU는 $Q, K$의 양수 값을 유지하고 음수는 0으로 만든다.  
- ALeakyReLU: LeakyReLU를 사용하여 일부 음수 값도 유지한다.  
- Asoftmax: VSA를 따라 소프트맥스로 어텐션 맵을 생성한다.  
- 위 네 방법은 동일한 Spikformer 프레임워크를 사용하며, 스파이크 형태의 $V$에 가중치를 부여한다.

표 1에서 ASSA가 AI와 ALeakyReLU보다 우수한 성능을 보인다는 사실은 스파이킹 뉴런($SN$)의 우수성을 보여준다. ASSA가 AReLU보다 좋은 이유는, ASSA가 자기-어텐션에서 더 나은 비선형성(non-linearity)을 제공하기 때문일 수 있다. Asoftmax와 비교해도 ASSA는 경쟁력이 있으며, CIFAR10-DVS 및 CIFAR10에서는 Asoftmax를 오히려 상회한다. 이는 SSA가 정보량이 제한된 스파이크 시퀀스($X$와 $V$)에 대해 VSA보다 더 적합하기 때문으로 볼 수 있다. 또한 ASSA가 $Q, K, V$ 간 계산을 완료하기 위해 요구하는 연산 수 및 이론적 에너지 소비량은 다른 방법들보다 훨씬 낮다.

SSA는 스파이크 시퀀스 모델링을 위해 특별히 설계되었다. $Q, K, V$가 모두 스파이크 형태이므로, 행렬 내적(matrix dot-product) 계산은 논리 AND와 합산(summation) 연산으로 퇴화(degrade)한다. 예를 들어 쿼리의 한 행(row) $q$와 키의 한 열(column) $k$에 대해:

$$
\sum_{i=1}^{d} q_i k_i = \sum_{q_i = 1} k_i.
$$

또한 표 1에서 보이듯, 그림 4와 같이 희소한 스파이크 형태의 $Q, K, V$와 단순화된 계산 덕분에 SSA는 낮은 계산 부담(computation burden)과 에너지 소비를 가진다. 더 나아가 $Q, K, V$의 계산 순서는 바꿀 수 있다: 먼저 $QK^\top$를 계산한 뒤 $V$와 곱하거나, 먼저 $K^\top V$를 계산한 뒤 $Q$와 곱할 수 있다. 시퀀스 길이 $N$이 한 헤드의 차원 $d$보다 클 때, 후자의 계산 순서는 전자의 $O(N^2 d)$보다 작은 $O(N d^2)$ 계산 복잡도를 갖는다. SSA는 전체 계산 과정에서 생물학적 그럴듯함(biological plausibility)과 계산 효율(computationally efficient) 특성을 유지한다.

---

## 4 실험(Experiments)

우리는 정적 데이터셋(static datasets)인 CIFAR, ImageNet(Deng et al., 2009)과 뉴로모픽 데이터셋(neuromorphic datasets)인 CIFAR10-DVS, DVS128 Gesture(Amir et al., 2017)에서 Spikformer의 성능을 평가하였다. 모든 실험 모델은 Pytorch(Paszke et al., 2019), SpikingJelly[^2], Pytorch image models 라이브러리(Timm)[^3]를 기반으로 구현하였다. Spikformer는 스크래치(from scratch)부터 학습하며, 4.1절과 4.2절에서 기존 SNN 모델들과 비교한다. 4.3절에서는 SSA 모듈과 Spikformer의 효과를 보이기 위한 소거(ablation) 실험을 수행한다.

[^2]: https://github.com/fangwei123456/spikingjelly  
[^3]: https://github.com/rwightman/pytorch-image-models

---

## 표 2

**표 2: ImageNet에서의 평가(Evaluation).** Param은 파라미터 수(number of parameters)이다. Power는 ImageNet 테스트셋의 이미지 한 장을 예측할 때의 평균 이론적 에너지 소비량(theoretical energy consumption)이며, 계산 상세는 식 (22)에示한다. Spikformer-L-D는 $L$개의 Spikformer 인코더 블록과 $D$차원 임베딩(embedding dimension)을 갖는 Spikformer 모델을 의미한다. 학습 손실(train loss), 테스트 손실(test loss), 테스트 정확도(test accuracy) 곡선은 부록 D.2에示한다. OPs는 SNN에서는 SOPs, ANN-ViT에서는 FLOPs를 의미한다.

| Methods | Architecture | Param (M) | OPs (G) | Power (mJ) | Time Step | Acc |
|---|---|---:|---:|---:|---:|---:|
| Hybrid training (Rathi et al., 2020) | ResNet-34 | 21.79 | - | - | 250 | 61.48 |
| TET (Deng et al., 2021) | Spiking-ResNet-34 | 21.79 | - | - | 6 | 64.79 |
| SEW-ResNet-34 | 21.79 | - | - | 4 | 68.00 |
| Spiking ResNet (Hu et al., 2021a) | ResNet-34 | 21.79 | 65.28 | 59.295 | 350 | 71.61 |
|  | ResNet-50 | 25.56 | 78.29 | 70.934 | 350 | 72.75 |
| STBP-tdBN (Zheng et al., 2021) | Spiking-ResNet-34 | 21.79 | 6.50 | 6.393 | 6 | 63.72 |
| SEW ResNet (Fang et al., 2021a) | SEW-ResNet-34 | 21.79 | 3.88 | 4.035 | 4 | 67.04 |
|  | SEW-ResNet-50 | 25.56 | 4.83 | 4.890 | 4 | 67.78 |
|  | SEW-ResNet-101 | 44.55 | 9.30 | 8.913 | 4 | 68.76 |
|  | SEW-ResNet-152 | 60.19 | 13.72 | 12.891 | 4 | 69.26 |
| Transformer | Transformer-8-512 | 29.68 | 8.33 | 38.340 | 1 | 80.80 |
| Spikformer | Spikformer-8-384 | 16.81 | 6.82 | 7.734 | 4 | 70.24 |
|  | Spikformer-6-512 | 23.37 | 8.69 | 9.417 | 4 | 72.46 |
|  | Spikformer-8-512 | 29.68 | 11.09 | 11.577 | 4 | 73.38 |
|  | Spikformer-10-512 | 36.01 | 13.67 | 13.899 | 4 | 73.68 |
|  | Spikformer-8-768 | 66.34 | 22.09 | 21.477 | 4 | 74.81 |

### 4.1 정적 데이터셋 분류(Static Datasets Classification)

**ImageNet**은 학습(training)용 약 130만 장의 1,000 클래스 이미지와, 검증(validation)용 50,000 이미지를 포함한다. ImageNet에서 모델 입력 크기는 기본값 224×224로 설정하였다. 최적화기(optimizer)는 AdamW를 사용하며, 배치 크기(batch size)는 128 또는 256으로 설정하고 310 에폭(epoch) 동안 코사인 감쇠(cosine-decay) 학습률(learning rate)을 적용한다. 초기 학습률(initial value)은 0.0005이다. ImageNet과 CIFAR에서 스케일링 계수(scaling factor)는 0.125로 설정했다. 4블록 SPS는 이미지를 196개의 16×16 패치로 분할한다. (Yuan et al., 2021a)을 따라 랜덤 증강(random augmentation), 믹스업(mixup), 컷믹스(cutmix) 등의 표준 데이터 증강을 학습에 적용하였다.

우리는 ImageNet에서 다양한 임베딩 차원과 트랜스포머 블록 수를 갖는 모델들을 실험했으며, 그 결과는 표 2에 정리되어 있다. 또한 시냅스 연산(synaptic operations, SOPs)(Merolla et al., 2014)과 이론적 에너지 소비량도 함께 비교했다. 결과에서 보이듯, Spikformer는 현재 최고 성능의 SNN 모델들에 비해 ImageNet에서 정확도를 크게 향상시킨다. 특히 가장 작은 Spikformer-8-384(16.81M 파라미터)는 ImageNet에서 스크래치 학습으로 70.24% top-1 정확도를 달성하며, 이는 60.19M 파라미터의 최고 직접 학습 모델 SEW-ResNet-152(69.26%)를 능가한다. 더 나아가 Spikformer-8-384의 SOPs 및 이론적 에너지 소비량(6.82G, 7.734mJ)은 SEW-ResNet-152(13.72G, 12.891mJ)보다 낮다. 29.68M 파라미터의 Spikformer-8-512는 73.38%로 이미 최신 성능을 달성하며, 350 타임스텝을 사용하는 변환 모델(Hu et al., 2021a)(72.75%)보다도 높다. Spikformer 블록 수가 증가할수록 ImageNet 정확도도 증가한다: Spikformer-10-512는 73.68%를 얻는다. 임베딩 차원을 늘려도 유사한 경향이 나타나며, Spikformer-8-768은 74.81%로 성능을 더 끌어올리고 SEW-ResNet-152보다 5.55%p 높다. ANN-ViT-8-512는 Spikformer-8-512보다 7.42%p 높지만, 이론적 에너지 소비량은 Spikformer-8-512의 3.31배이다. 그림 3에서는 Spikformer-8-512의 마지막 인코더 블록에서 4번째 타임스텝의 SSA 어텐션 맵 예시를示한다. SSA는 분류에 관련된 이미지 영역을 포착하고, 관련 없는 영역은 0(검은 영역)으로 만들어, 효과적이면서 이벤트 기반이며 에너지 효율적인 특성을 보인다.

---

## 그림 3

**그림 3: SSA의 어텐션 맵(attention map) 예시.** 검은색 영역은 0이다.

---

**CIFAR**는 해상도 32×32의 학습 이미지 50,000장과 테스트 이미지 10,000장을 제공한다. 배치 크기는 128로 설정하였다. 4블록 SPS(첫 두 블록에는 맥스 풀링을 포함하지 않음)는 이미지를 64개의 4×4 패치로 분할한다. 표 3은 CIFAR에서 Spikformer를 다른 모델들과 비교한 정확도를 보여준다.

---

## 표 3

**표 3: CIFAR10/100에서 기존 방법과의 성능 비교.** 제안 방법은 모든 과제에서 성능을 향상시킨다. `*`는 Deng et al. (2021)이 자체 구현(self-implementation)한 결과를 의미한다. Hybrid training (Rathi et al., 2020)은 CIFAR10에는 ResNet-20, CIFAR100에는 VGG-11을 사용한다는 점에 유의하라.

| Methods | Architecture | Param (M) | Time Step | CIFAR10 Acc | CIFAR100 Acc |
|---|---|---:|---:|---:|---:|
| Hybrid training (Rathi et al., 2020) | VGG-11 | 9.27 | 125 | 92.22 | 67.87 |
| Diet-SNN (Rathi & Roy, 2020) | ResNet-20 | 0.27 | 10/5 | 92.54 | 64.07 |
| STBP (Wu et al., 2018) | CIFARNet | 17.54 | 12 | 89.83 | - |
| STBP NeuNorm (Wu et al., 2019) | CIFARNet | 17.54 | 12 | 90.53 | - |
| TSSL-BP (Zhang & Li, 2020) | CIFARNet | 17.54 | 5 | 91.41 | - |
| STBP-tdBN (Zheng et al., 2021) | ResNet-19 | 12.63 | 4 | 92.92 | 70.86 |
| TET (Deng et al., 2021) | ResNet-19 | 12.63 | 4 | 94.44 | 74.47 |
| ANN ResNet-19* |  | 12.63 | 1 | 94.97 | 75.35 |
| Transformer-4-384 |  | 9.32 | 1 | 96.73 | 81.02 |
| Spikformer | Spikformer-4-256 | 4.15 | 4 | 93.94 | 75.96 |
|  | Spikformer-2-384 | 5.76 | 4 | 94.80 | 76.95 |
|  | Spikformer-4-384 | 9.32 | 4 | 95.19 | 77.86 |
|  | Spikformer-4-384 400E | 9.32 | 4 | 95.51 | 78.21 |

표 3에서 Spikformer-4-384는 CIFAR10에서 95.19% 정확도를 달성하여 TET(94.44%) 및 ResNet-19 ANN(94.97%)보다 우수하다. 임베딩 차원 또는 블록 수를 늘리면 성능이 향상된다. 구체적으로 Spikformer-4-384는 Spikformer-4-256보다 1.25%p, Spikformer-2-384보다 0.39%p 높다. 또한 학습 에폭을 400으로 늘리면 성능이 개선됨을 관찰했다(Spikformer-4-384 400E는 CIFAR10 및 CIFAR100에서 각각 0.32%p와 0.35%p 더 높음). 더 복잡한 데이터셋인 CIFAR100에서는 개선 폭이 더 크다. Spikformer-4-384(77.86%, 9.32M)는 ResNet-19 ANN(75.35%, 12.63M)보다 2.51%p 향상된다. ANN-Transformer 모델은 Spikformer-4-384보다 각각 1.54%p 및 3.16%p 높다. 부록 D.5에 보이듯, 사전학습된(pre-trained) Spikformer를 기반으로 한 전이학습(transfer learning)은 CIFAR에서 더 높은 성능을 달성할 수 있으며, 이는 Spikformer의 높은 전이 능력(transfer ability)을 보여준다.

### 4.2 뉴로모픽 데이터셋 분류(Neuromorphic Datasets Classification)

DVS128 Gesture는 29명의 개체가 3가지 조명 조건에서 수행한 11개 손동작(gesture) 범주를 포함하는 제스처 인식 데이터셋이다. CIFAR10-DVS는 정적 이미지 데이터셋을 이미지 이동(shifting)으로 DVS 카메라가 촬영하도록 변환한 뉴로모픽 데이터셋으로, 학습 샘플 9,000개와 테스트 샘플 1,000개를 제공한다.

위 두 데이터셋(이미지 크기 128×128)에 대해, 우리는 4블록 SPS를 사용했다. 패치 임베딩 차원은 256이고 패치 크기는 16×16이다. 트랜스포머 인코더 블록이 2개인 얕은(shallow) Spikformer를 사용하였다. SSA는 DVS128 Gesture에서는 8개 헤드, CIFAR10-DVS에서는 16개 헤드를 사용한다. 스파이킹 뉴런의 타임스텝은 10 또는 16이며, 학습 에폭은 DVS128 Gesture 200, CIFAR10-DVS 106이다. 최적화기는 AdamW, 배치 크기는 16으로 설정했다. 학습률은 0.1로 시작해 코사인 감쇠로 감소한다. CIFAR10-DVS에는 (Li et al., 2022)를 따라 데이터 증강을 적용했다. 또한 $QK^\top V$ 결과를 제어하기 위한 스케일링 계수(scaling factor)로 학습 가능한 파라미터를 사용했다.

뉴로모픽 데이터셋에서 Spikformer와 최신 모델들의 분류 성능은 표 4에示한다. 2.59M 모델만으로 두 데이터셋 모두에서 좋은 성능을 달성함을 볼 수 있다. DVS128 Gesture에서 우리는 16 타임스텝으로 98.2% 정확도를 얻어 SEW-ResNet(97.9%)보다 높다. 또한 순전파(forward propagation)에서 부동소수점 스파이크를 사용하는 TA-SNN(98.6%, 60 타임스텝)(Yao et al., 2021)과 비교해도 경쟁력 있다. CIFAR10-DVS에서는 이진 스파이크(binary spikes)로 10 타임스텝과 16 타임스텝을 사용했을 때, SOTA 방법인 DSR(77.3%)보다 각각 1.6%p 및 3.6%p 높은 정확도를 달성한다. TET은 아키텍처 기반이 아니라 손실 기반(loss-based) 방법으로, 300 에폭과 9.27M VGGSNN으로 83.2%를 달성하므로, 표에서는 비교하지 않았다.

---

## 표 4

**표 4: 두 뉴로모픽 데이터셋에서 최신(SOTA) 방법과의 성능 비교.** 굵은 글씨는 최고 성능을 의미하며, `*`는 데이터 증강(data augmentation)을 적용했음을 의미한다.

| Method | Spikes | CIFAR10-DVS T Step | CIFAR10-DVS Acc | DVS128 T Step | DVS128 Acc |
|---|:---:|---:|---:|---:|---:|
| LIAF-Net (Wu et al., 2021)TNNLS-2021 | ✗ | 10 | 70.4 | 60 | 97.6 |
| TA-SNN (Yao et al., 2021)ICCV-2021 | ✗ | 10 | 72.0 | 60 | 98.6 |
| Rollout (Kugele et al., 2020)Front. Neurosci-2020 | ✓ | 48 | 66.8 | 240 | 97.2 |
| DECOLLE (Kaiser et al., 2020)Front. Neurosci-2020 | ✓ | - | - | 500 | 95.5 |
| tdBN (Zheng et al., 2021)AAAI-2021 | ✓ | 10 | 67.8 | 40 | 96.9 |
| PLIF (Fang et al., 2021b)ICCV-2021 | ✓ | 20 | 74.8 | 20 | 97.6 |
| SEW-ResNet (Fang et al., 2021a)NeurIPS-2021 | ✓ | 16 | 74.4 | 16 | 97.9 |
| Dspike (Li et al., 2021)NeurIPS-2021 | ✓ | 10 | 75.4* | - | - |
| SALT (Kim & Panda, 2021)Neural Netw-2021 | ✓ | 20 | 67.1 | - | - |
| DSR (Meng et al., 2022)CVPR-2022 | ✓ | 10 | 77.3* | - | - |
| Spikformer | ✓ | 10 | 78.9* | 10 | 96.9 |
|  | ✓ | 16 | 80.9* | 16 | 98.3 |

### 4.3 소거 연구(Ablation Study)

**타임스텝(time step).** 스파이킹 뉴런의 시뮬레이션 타임스텝 수에 따른 정확도는 표 5에示하였다. 타임스텝이 1일 때, CIFAR10에서 $T=4$인 네트워크보다 1.87%p 낮다. 그럼에도 Spikformer-8-512는 타임스텝 1에서도 70.14%를 달성한다. 이러한 결과는 Spikformer가 저지연(low latency) 조건(타임스텝 수가 적음)에서도 견고함(robust)을 보여준다.

**SSA.** SSA의 장점을 더 분명히 확인하기 위해, SSA를 표준 VSA로 대체한 실험을 수행하였다. 밸류가 부동소수점 형태인 경우(Spikformer-L-D w VSA $V_F$)와 스파이크 형태인 경우(Spikformer-L-D w VSA)를 모두 테스트하였다. 또한 표 1을 따라 ImageNet에서 다양한 어텐션 변형도 테스트하였다. CIFAR10에서 SSA를 사용한 Spikformer는 VSA 및 VSA $V_F$와 견줄 만한 성능을 보이며, ImageNet에서는 Spikformer-8-512 w SSA가 Spikformer-8-512 w VSA보다 0.68%p 높다. CIFAR100과 ImageNet에서는 부동소수점 밸류를 사용하는 Spikformer-L-D w VSA $V_F$가 Spikformer보다 더 높을 수 있는데, 이는 밸류가 부동소수점 형태이기 때문이다. 한편 Spikformer-8-512 w I, Spikformer-8-512 w ReLU, Spikformer-8-512 w LeakyReLU가 수렴하지 않는 이유는, 쿼리/키/밸류의 내적 값(dot-product)이 커져 출력 스파이킹 뉴런 층의 대리 그래디언트(surrogate gradient)가 사라지기 때문이며, 자세한 내용은 부록 D.4에 제시한다. 반면 설계된 SSA의 내적 값은 희소한 스파이크 형태 $Q, K, V$에 의해 제어 가능한 범위에 있으며, 이로 인해 SSA를 사용한 Spikformer는 학습이 안정적이고 수렴하기 쉽다.

---

## 표 5

**표 5: SSA 및 타임스텝에 대한 소거 실험 결과.**

| Datasets | Models | Time Step | Top1-Acc (%) |
|---|---|---:|---:|
| CIFAR10/100 | Spikformer-4-384 w SSA | 1 | 93.51 / 74.36 |
|  |  | 2 | 93.59 / 76.28 |
|  |  | 4 | 95.19 / 77.86 |
|  |  | 6 | 95.34 / 78.61 |
|  | Spikformer-4-384 w VSA | 4 | 94.97 / 77.92 |
|  | Spikformer-4-384 w VSA $V_F$ | 4 | 95.17 / 78.37 |
| ImageNet | Spikformer-8-512 w I | 4 | ✗ |
|  | Spikformer-8-512 w ReLU | 4 | ✗ |
|  | Spikformer-8-512 w LeakyReLU | 4 | ✗ |
|  | Spikformer-8-512 w VSA | 4 | 72.70 |
|  | Spikformer-8-512 w VSA $V_F$ | 4 | 73.96 |
|  | Spikformer-8-512 w SSA | 1 | 70.14 |
|  |  | 2 | 71.09 |
|  |  | 4 | 73.38 |
|  |  | 6 | 73.70 |

---

## 5 결론(Conclusion)

본 연구에서는 스파이킹 뉴런 네트워크(Spiking Neuron Networks)에서 자기-어텐션 메커니즘과 트랜스포머를 구현할 수 있는지의 가능성을 탐구하고, 새로운 스파이킹 자기-어텐션(Spiking Self-Attention, SSA)에 기반한 Spikformer를 제안하였다. ANN에서의 표준 자기-어텐션과 달리, SSA는 SNN과 스파이크 데이터(spike data)에 특화되어 설계되었다. 우리는 SSA에서 복잡한 소프트맥스 연산을 제거하고, 스파이크 형태의 쿼리/키/밸류에 대해 행렬 내적을 직접 수행함으로써 효율적이며 곱셈을 회피하도록 했다. 이 단순한 자기-어텐션 메커니즘은 Spikformer가 정적 및 뉴로모픽 데이터셋 모두에서 매우 효과적으로 동작하게 한다. 스크래치부터의 직접 학습만으로도, Spikformer는 최신 SNN 모델들을 능가한다. 우리의 연구가 트랜스포머 기반 SNN 모델에 대한 향후 연구의 발판이 되기를 바란다.

---

## 재현성 성명(Reproducibility Statement)

우리의 코드는 오픈 소스 SNN 프레임워크인 SpikingJelly(Fang et al., 2020)와 Pytorch image models 라이브러리(Timm)(Wightman, 2019)를 기반으로 한다. 본 논문의 실험 결과는 재현 가능하다. 본문 및 부록에서 모델 학습과 데이터셋 증강의 상세 내용을 설명하였다. Spikformer 모델 코드는 보조 자료(supplementary material)로 업로드되어 있으며, 심사(review) 후 GitHub에서 공개될 예정이다.

---

## REFERENCES

Arnon Amir, Brian Taba, David Berg, Timothy Melano, Jeffrey McKinstry, Carmelo Di Nolfo, Tapan
Nayak, Alexander Andreopoulos, Guillaume Garreau, Marcela Mendoza, Jeff Kusnitz, Michael
Debole, Steve Esser, Tobi Delbruck, Myron Flickner, and Dharmendra Modha. A low power, fully
event-based gesture recognition system. In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR), pp. 7243–7252, 2017.

Tong Bu, Wei Fang, Jianhao Ding, PengLin Dai, Zhaofei Yu, and Tiejun Huang. Optimal ann-snn
conversion for high-accuracy and ultra-low-latency spiking neural networks. In International
Conference on Learning Representations (ICLR), 2021.

Yongqiang Cao, Yang Chen, and Deepak Khosla. Spiking deep convolutional neural networks for
energy-efficient object recognition. International Journal of Computer Vision, 113(1):54–66, 2015.

Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey
Zagoruyko. End-to-end object detection with transformers. In Proceedings of the European
Conference on Computer Vision (ECCV), pp. 213–229. Springer, 2020.

Charlotte Caucheteux and Jean-Remi King. Brains and algorithms partially converge in natural ´
language processing. Communications biology, 5(1):1–10, 2022.

Hanting Chen, Yunhe Wang, Tianyu Guo, Chang Xu, Yiping Deng, Zhenhua Liu, Siwei Ma, Chunjing
Xu, Chao Xu, and Wen Gao. Pre-trained image processing transformer. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 12299–12310,
2021.

Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas
Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, et al. Rethinking attention
with performers. arXiv preprint arXiv:2009.14794, 2020.

Xiangxiang Chu, Zhi Tian, Yuqing Wang, Bo Zhang, Haibing Ren, Xiaolin Wei, Huaxia Xia, and
Chunhua Shen. Twins: Revisiting the design of spatial attention in vision transformers. In
Proceedings of the International Conference on Neural Information Processing Systems (NeurIPS),
volume 34, pp. 9355–9366, 2021.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale
hierarchical image database. In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition (CVPR), pp. 248–255, 2009.

Shikuang Deng, Yuhang Li, Shanghang Zhang, and Shi Gu. Temporal Efficient Training of Spiking
Neural Network via Gradient Re-weighting. In International Conference on Learning Representa￾tions (ICLR), 2021.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas
Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image
is worth 16x16 words: Transformers for image recognition at scale. In International Conference
on Learning Representa- tions (ICLR), 2020.

Wei Fang, Yanqi Chen, Jianhao Ding, Ding Chen, Zhaofei Yu, Huihui Zhou, Yonghong Tian, and other
contributors. Spikingjelly. https://github.com/fangwei123456/spikingjelly,
2020. Accessed: YYYY-MM-DD.

Wei Fang, Zhaofei Yu, Yanqi Chen, Tiejun Huang, Timothee Masquelier, and Yonghong Tian. Deep ´
Residual Learning in Spiking Neural Networks. In Proceedings of the International Conference on
Neural Information Processing Systems (NeurIPS), volume 34, pp. 21056–21069, 2021a.

Wei Fang, Zhaofei Yu, Yanqi Chen, Timothee Masquelier, Tiejun Huang, and Yonghong Tian. ´
Incorporating learnable membrane time constant to enhance learning of spiking neural networks.
In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 2661–
2671, 2021b.

Bing Han, Gopalakrishnan Srinivasan, and Kaushik Roy. Rmp-snn: Residual membrane potential
neuron for enabling deeper high-accuracy and low-latency spiking neural network. In Proceedings
of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 13558–
13567, 2020.

Ali Hassani, Steven Walton, Nikhil Shah, Abulikemu Abuduweili, Jiachen Li, and Humphrey Shi.
Escaping the big data paradigm with compact transformers. arXiv preprint arXiv:2104.05704,
2021.

Mark Horowitz. 1.1 computing’s energy problem (and what we can do about it). In 2014 IEEE
International Solid-State Circuits Conference Digest of Technical Papers (ISSCC), pp. 10–14.
IEEE, 2014.

Yangfan Hu, Huajin Tang, and Gang Pan. Spiking deep residual networks. IEEE Transactions on
Neural Networks and Learning Systems, pp. 1–6, 2021a. doi: 10.1109/TNNLS.2021.3119238.

Yifan Hu, Yujie Wu, Lei Deng, and Guoqi Li. Advancing residual learning towards powerful deep
spiking neural networks. arXiv preprint arXiv:2112.08954, 2021b.

Eric Hunsberger and Chris Eliasmith. Spiking deep networks with lif neurons. arXiv preprint
arXiv:1510.08829, 2015.

Jacques Kaiser, Hesham Mostafa, and Emre Neftci. Synaptic Plasticity Dynamics for Deep
Continuous Local Learning (DECOLLE). Frontiers in Neuroscience, 14:424, 2020. doi:
10.3389/fnins.2020.00424.

Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and Franc¸ois Fleuret. Transformers are rnns:
Fast autoregressive transformers with linear attention. In Proceedings of the 37th International
Conference on Machine Learning (ICML), pp. 5156–5165, 2020.

Youngeun Kim and Priyadarshini Panda. Optimizing Deeper Spiking Neural Networks for Dynamic
Vision Sensing. Neural Networks, 144:686–698, 2021.

Alex Krizhevsky. Learning multiple layers of features from tiny images. 2009.

Alexander Kugele, Thomas Pfeil, Michael Pfeiffer, and Elisabetta Chicca. Efficient Processing of
Spatio-temporal Data Streams with Spiking Neural Networks. Frontiers in Neuroscience, 14:439,
2020.

Souvik Kundu, Gourav Datta, Massoud Pedram, and Peter A Beerel. Spike-thrift: Towards energy￾efficient deep spiking neural networks by limiting spiking activity via attention-guided compression.
In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV),
pp. 3953–3962, 2021a.

Souvik Kundu, Massoud Pedram, and Peter A Beerel. Hire-snn: Harnessing the inherent robustness of
energy-efficient deep spiking neural networks by training with crafted input noise. In Proceedings
of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 5209–5218, 2021b.

Chankyu Lee, Syed Shakib Sarwar, Priyadarshini Panda, Gopalakrishnan Srinivasan, and Kaushik Roy.
Enabling spike-based backpropagation for training deep neural network architectures. Frontiers in
neuroscience, 14:119, 2020.

Jun Haeng Lee, Tobi Delbruck, and Michael Pfeiffer. Training deep spiking neural networks using
backpropagation. Frontiers in neuroscience, 10:508, 2016.

Hongmin Li, Hanchao Liu, Xiangyang Ji, Guoqi Li, and Luping Shi. Cifar10-dvs: an event-stream
dataset for object classification. Frontiers in neuroscience, 11:309, 2017.

Yuhang Li, Yufei Guo, Shanghang Zhang, Shikuang Deng, Yongqing Hai, and Shi Gu. Differentiable
Spike: Rethinking Gradient-Descent for Training Spiking Neural Networks. In Proceedings of the
International Conference on Neural Information Processing Systems (NeurIPS), volume 34, pp.
23426–23439, 2021.

Yuhang Li, Youngeun Kim, Hyoungseob Park, Tamar Geller, and Priyadarshini Panda. Neuromorphic
data augmentation for training spiking neural networks. arXiv preprint arXiv:2203.06145, 2022.

Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo.
Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the
IEEE/CVF International Conference on Computer Vision (ICCV), pp. 10012–10022, 2021.

Ali Lotfi Rezaabad and Sriram Vishwanath. Long short-term memory spiking networks and their
applications. In Proceedings of the International Conference on Neuromorphic Systems 2020
(ICONS), pp. 1–9, 2020.

Wolfgang Maass. Networks of spiking neurons: the third generation of neural network models.
Neural networks, 10(9):1659–1671, 1997.

Qingyan Meng, Mingqing Xiao, Shen Yan, Yisen Wang, Zhouchen Lin, and Zhi-Quan Luo. Training
High-Performance Low-Latency Spiking Neural Networks by Differentiation on Spike Representa￾tion. ArXiv preprint arXiv:2205.00459, 2022.

Paul A Merolla, John V Arthur, Rodrigo Alvarez-Icaza, Andrew S Cassidy, Jun Sawada, Filipp
Akopyan, Bryan L Jackson, Nabil Imam, Chen Guo, Yutaka Nakamura, et al. A million spiking￾neuron integrated circuit with a scalable communication network and interface. Science, 345
(6197):668–673, 2014.

Etienne Mueller, Viktor Studenyak, Daniel Auge, and Alois Knoll. Spiking transformer networks:
A rate coded approach for processing sequential data. In 2021 7th International Conference on
Systems and Informatics (ICSAI), pp. 1–5. IEEE, 2021.

Emre O Neftci, Hesham Mostafa, and Friedemann Zenke. Surrogate gradient learning in spiking
neural networks: Bringing the power of gradient-based optimization to spiking neural networks.
IEEE Signal Processing Magazine, 36(6):51–63, 2019.

Priyadarshini Panda, Sai Aparna Aketi, and Kaushik Roy. Toward scalable, efficient, and accurate deep
spiking neural networks with backward residual connections, stochastic softmax, and hybridization.
Frontiers in Neuroscience, 14:653, 2020.

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor
Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style,
high-performance deep learning library. In Proceedings of the International Conference on Neural
Information Processing Systems (NeurIPS), volume 32, 2019.

Zhen Qin, Weixuan Sun, Hui Deng, Dongxu Li, Yunshen Wei, Baohong Lv, Junjie Yan, Ling￾peng Kong, and Yiran Zhong. cosformer: Rethinking softmax in attention. arXiv preprint
arXiv:2202.08791, 2022.

Yongming Rao, Wenliang Zhao, Benlin Liu, Jiwen Lu, Jie Zhou, and Cho-Jui Hsieh. Dynamicvit:
Efficient vision transformers with dynamic token sparsification. In Proceedings of the International
Conference on Neural Information Processing Systems (NeurIPS), volume 34, pp. 13937–13949,
2021.

Nitin Rathi and Kaushik Roy. Diet-snn: Direct input encoding with leakage and threshold optimization
in deep spiking neural networks. arXiv preprint arXiv:2008.03658, 2020.

Nitin Rathi, Gopalakrishnan Srinivasan, Priyadarshini Panda, and Kaushik Roy. Enabling deep
spiking neural networks with hybrid conversion and spike timing dependent backpropagation.
arXiv preprint arXiv:2005.01807, 2020.

Kaushik Roy, Akhilesh Jaiswal, and Priyadarshini Panda. Towards spike-based machine intelligence
with neuromorphic computing. Nature, 575(7784):607–617, 2019.

Bodo Rueckauer, Iulia-Alexandra Lungu, Yuhuang Hu, Michael Pfeiffer, and Shih-Chii Liu. Conver￾sion of continuous-valued deep networks to efficient event-driven networks for image classification.
Frontiers in neuroscience, 11:682, 2017.

Sumit B Shrestha and Garrick Orchard. Slayer: Spike layer error reassignment in time. In Proceedings
of the International Conference on Neural Information Processing Systems (NeurIPS), volume 31,
2018.

Jeong-geun Song. Ufo-vit: High performance linear vision transformer without softmax. arXiv
preprint arXiv:2109.14382, 2021.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz
Kaiser, and Illia Polosukhin. Attention is all you need. In Proceedings of the International
Conference on Neural Information Processing Systems (NeurIPS), volume 30, 2017.

Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo,
and Ling Shao. Pyramid vision transformer: A versatile backbone for dense prediction without
convolutions. In Proceedings of the IEEE/CVF International Conference on Computer Vision
(ICCV), pp. 568–578, 2021.

Yuchen Wang, Malu Zhang, Yi Chen, and Hong Qu. Signed neuron with memory: Towards simple,
accurate and high-efficient ann-snn conversion. In International Joint Conference on Artificial
Intelligence, 2022.

James C. R. Whittington, Joseph Warren, and Tim E.J. Behrens. Relating transformers to mod￾els and neural representations of the hippocampal formation. In International Conference on
Learning Representations (ICLR), 2022. URL https://openreview.net/forum?id=
B8DVo9B1YE0.

Ross Wightman. Pytorch image models. https://github.com/rwightman/
pytorch-image-models, 2019.

Yujie Wu, Lei Deng, Guoqi Li, Jun Zhu, and Luping Shi. Spatio-temporal backpropagation for
training high-performance spiking neural networks. Frontiers in neuroscience, 12:331, 2018.

Yujie Wu, Lei Deng, Guoqi Li, Jun Zhu, Yuan Xie, and Luping Shi. Direct Training for Spiking
Neural Networks: Faster, Larger, Better. In Proceedings of the AAAI Conference on Artificial
Intelligence (AAAI), pp. 1311–1318, 2019. doi: 10.1609/aaai.v33i01.33011311.

Zhenzhi Wu, Hehui Zhang, Yihan Lin, Guoqi Li, Meng Wang, and Ye Tang. LIAF-Net: Leaky
Integrate and Analog Fire Network for Lightweight and Efficient Spatiotemporal Information
Processing. IEEE Transactions on Neural Networks and Learning Systems, pp. 1–14, 2021. doi:
10.1109/TNNLS.2021.3073016.

Mingqing Xiao, Qingyan Meng, Zongpeng Zhang, Yisen Wang, and Zhouchen Lin. Training
feedback spiking neural networks by implicit differentiation on the equilibrium state. volume 34,
pp. 14516–14528, 2021a.

Tete Xiao, Mannat Singh, Eric Mintun, Trevor Darrell, Piotr Dollar, and Ross Girshick. Early ´
convolutions help transformers see better. In Proceedings of the International Conference on
Neural Information Processing Systems (NeurIPS), volume 34, pp. 30392–30400, 2021b.

Jianwei Yang, Chunyuan Li, Pengchuan Zhang, Xiyang Dai, Bin Xiao, Lu Yuan, and Jianfeng
Gao. Focal attention for long-range interactions in vision transformers. In Proceedings of the
International Conference on Neural Information Processing Systems (NeurIPS), volume 34, pp.
30008–30022, 2021.

Man Yao, Huanhuan Gao, Guangshe Zhao, Dingheng Wang, Yihan Lin, Zhaoxu Yang, and Guoqi Li.
Temporal-wise attention spiking neural networks for event streams classification. In Proceedings
of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 10221–10230, 2021.

Man Yao, Guangshe Zhao, Hengyu Zhang, Yifan Hu, Lei Deng, Yonghong Tian, Bo Xu, and Guoqi
Li. Attention spiking neural networks. arXiv preprint arXiv:2209.13929, 2022.

Bojian Yin, Federico Corradi, and Sander M Bohte. Accurate and efficient time-domain classification ´
with adaptive spiking recurrent neural networks. Nature Machine Intelligence, 3(10):905–913,
2021.

Li Yuan, Yunpeng Chen, Tao Wang, Weihao Yu, Yujun Shi, Zi-Hang Jiang, Francis EH Tay, Jiashi
Feng, and Shuicheng Yan. Tokens-to-token vit: Training vision transformers from scratch on
imagenet. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV),
pp. 558–567, 2021a.

Li Yuan, Qibin Hou, Zihang Jiang, Jiashi Feng, and Shuicheng Yan. Volo: Vision outlooker for visual
recognition. arXiv preprint arXiv:2106.13112, 2021b.

Jiqing Zhang, Bo Dong, Haiwei Zhang, Jianchuan Ding, Felix Heide, Baocai Yin, and Xin Yang.
Spiking transformers for event-based single object tracking. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition (CVPR), pp. 8801–8810, 2022a.

Jiyuan Zhang, Lulu Tang, Zhaofei Yu, Jiwen Lu, and Tiejun Huang. Spike transformer: Monocular
depth estimation for spiking camera. In Proceedings of the European Conference on Computer
Vision (ECCV), 2022b.

Wenrui Zhang and Peng Li. Temporal spike sequence learning via backpropagation for deep spiking
neural networks. In Proceedings of the International Conference on Neural Information Processing
Systems (NeurIPS), volume 33, pp. 12022–12033, 2020.

Hanle Zheng, Yujie Wu, Lei Deng, Yifan Hu, and Guoqi Li. Going Deeper With Directly-Trained
Larger Spiking Neural Networks. In Proceedings of the AAAI Conference on Artificial Intelligence
(AAAI), pp. 11062–11070, 2021.

Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, and Jifeng Dai. Deformable detr:
Deformable transformers for end-to-end object detection. arXiv preprint arXiv:2010.04159, 2020.

Zulun Zhu, Jiaying Peng, Jintang Li, Liang Chen, Qi Yu, and Siqiang Luo. Spiking graph convolu￾tional networks. In Proceedings of the Thirty-First International Joint Conference on Artificial
Intelligence (IJCAI), pp. 2434–2440, 2022. doi: 10.24963/ijcai.2022/338.

---

## 부록(Appendix)

### A 다중 헤드 스파이킹 자기-어텐션(Multihead Spiking Self Attention)

실제로는 $Q, K, V \in \mathbb{R}^{T \times N \times D}$를 다중 헤드 형태인 $\mathbb{R}^{T \times H \times N \times d}$로 재배치(reshape)하며, 여기서 $D = H \times d$이다. 그런 다음 $Q, K, V$를 $H$개의 부분으로 나누고, $H$개의 SSA 연산을 병렬로 수행하는데, 이를 $H$-헤드 SSA라고 한다. 다중 헤드 스파이킹 자기-어텐션(Multihead Spiking Self Attention, MSSA)은 다음과 같다.

$$
Q = (q_1, q_2, \ldots, q_H),\quad K = (k_1, k_2, \ldots, k_H),\quad V = (v_1, v_2, \ldots, v_H),\quad q,k,v \in \mathbb{R}^{T \times N \times d}. \qquad (17)
$$

$$
\mathrm{MSSA}_0(Q, K, V) = [\mathrm{SSA}^0_1(q_1, k_1, v_1);\ \mathrm{SSA}^0_2(q_2, k_2, v_2);\ \ldots;\ \mathrm{SSA}^0_H(q_H, k_H, v_H)]. \qquad (18)
$$

$$
\mathrm{MSSA}(Q, K, V) = SN\left(\mathrm{BN}(\mathrm{Linear}(\mathrm{MSSA}_0(Q, K, V)))\right). \qquad (19)
$$

### B 스파이킹 자기-어텐션과 타임스텝(Spiking Self Attention and Time Step)

실제로 $T$는 스파이킹 뉴런 층(spike neuron layer)에서 독립적인 차원(independent dimension)이다. 그 외의 층에서는 $T$가 배치 크기(batch size)와 병합(merge)된다.

### C 실험 상세(Experiment Details)

#### C.1 학습(Training)

표준 ViT와 달리, Spikformer에는 드롭아웃(dropout)과 드롭패스(droppath)를 적용하지 않는다. 또한 각 자기-어텐션 및 MLP 블록 이전의 레이어 정규화(layer norm)를 제거하고, 각 선형 층 이후에 배치 정규화(batch norm)를 추가한다. 모든 Spikformer 모델에서 MLP 블록의 은닉 차원(hidden dimension)은 $4 \times D$이며, 여기서 $D$는 임베딩 차원이다. 식 (20)과 같이, 우리는 $\alpha = 4$인 시그모이드(Sigmoid) 함수를 대리 함수(surrogate function)로 선택했다.

$$
\mathrm{Sigmoid}(x) = \frac{1}{1 + \exp(-\alpha x)}. \qquad (20)
$$

DVS128 Gesture의 경우, 데이터 밀도(density)를 높이기 위해 $Q$와 $K$ 뒤에 1D 맥스 풀링(1D max-pooling) 층을 추가하였으며, 이는 16 타임스텝에서 정확도를 97.9%에서 98.3%로 향상시킨다. 우리는 $QK^\top V \cdot s$ 뒤에 위치한 스파이킹 뉴런 층의 임계 전압(threshold voltage) $V_{\text{th}}$를 0.5로 설정하고, 그 외의 스파이킹 뉴런 층들은 1로 설정했다.

#### C.2 이론적 시냅스 연산과 에너지 소비량 계산(Theoretical Synaptic Operation and Energy Consumption Calculation)

이론적 에너지 소비량을 계산하려면 먼저 시냅스 연산(synaptic operations)을 계산해야 한다.

$$
\mathrm{SOPs}(l) = f_r \times T \times \mathrm{FLOPs}(l), \qquad (21)
$$

여기서 $l$은 Spikformer의 블록/층을 의미하며, $f_r$는 해당 블록/층 입력 스파이크 열(spike train)의 발화율(firing rate), $T$는 스파이킹 뉴런의 시뮬레이션 타임스텝이다. $\mathrm{FLOPs}(l)$는 층 $l$의 부동소수점 연산 수로, multiply-and-accumulate(MAC) 연산 수를 의미한다. SOPs는 spike-based accumulate(AC) 연산 수를 의미한다. 우리는 (Kundu et al., 2021b; Hu et al., 2021b; Horowitz, 2014; Kundu et al., 2021a; Yin et al., 2021; Panda et al., 2020; Yao et al., 2022)를 따라 Spikformer의 이론적 에너지 소비량을 추정한다. MAC과 AC 연산은 45nm 하드웨어에서 구현된다고 가정하며[12], 이때 $E_{\text{MAC}} = 4.6\text{pJ}$, $E_{\text{AC}} = 0.9\text{pJ}$이다. Spikformer의 이론적 에너지 소비량은 다음과 같다.

$$
E_{\text{Spikformer}} = E_{\text{MAC}} \times FL^{1}_{\text{SNN Conv}}
+ E_{\text{AC}} \times
\left(
\sum_{n=2}^{N} \mathrm{SOP}^{n}_{\text{SNN Conv}}
+ \sum_{m=1}^{M} \mathrm{SOP}^{m}_{\text{SNN FC}}
+ \sum_{l=1}^{L} \mathrm{SOP}^{l}_{\text{SSA}}
\right). \qquad (22)
$$

여기서 $FL^{1}_{\text{SNN Conv}}$는 정적 RGB 이미지를 스파이크 형태로 인코딩하는 첫 번째 층이다. 이후 $n$개의 SNN 컨볼루션 층, $m$개의 SNN 완전연결 층(fully connected layer, FC), 그리고 $l$개의 SSA의 SOPs를 모두 합한 뒤, $E_{\text{AC}}$를 곱한다. ANN의 경우, 블록 $b$의 이론적 에너지 소비량은 다음과 같이 계산한다.

$$
\mathrm{Power}(b) = 4.6\text{pJ} \times \mathrm{FLOPs}(b). \qquad (23)
$$

SNN에서는 다음과 같다.

$$
\mathrm{Power}(b) = 0.9\text{pJ} \times \mathrm{SOPs}(b). \qquad (24)
$$

---

## 그림 4

**그림 4: ImageNet 테스트셋에서 Spikformer-8-512의 각 블록에서 쿼리(Query), 키(Key), 밸류(Value)의 발화율(firing rate).**

---

## 그림 5

**그림 5: ImageNet에서의 학습 손실(training loss), 테스트 손실(testing loss), 테스트 정확도(test accuracy).**  
(a) 학습 손실 (b) 테스트 손실 (c) 테스트 정확도

---

### D 추가 결과(Additional Results)

#### D.1 쿼리/키/밸류의 발화율(Fire Rate of Query, Key and Value)

그림 4에서 보이듯, SSA의 쿼리(Query), 키(Key), 밸류(Value)는 매우 희소(sparse)하며, 이는 SSA 계산이 희소하게(sparse computation) 이루어지도록 한다.

#### D.2 ImageNet에서의 손실 및 정확도(Loss and Accuracy on ImageNet)

그림 5에 Spikformer의 학습 손실, 테스트 손실, 테스트 정확도를示하였다. Spikformer 블록 수가 증가하거나 임베딩 차원이 증가할수록 학습/테스트 손실이 감소한다.

---

## 표 6

**표 6: CIFAR10/100에서의 추가 결과(Additional result).** Spikformer-4-384 w IF는 적분-발화(Integrate-and-Fire, IF) 뉴런을 사용한다.

| Models | Time Step | Top1-Acc (%) |
|---|---:|---:|
| Spikformer-4-384 w I | 1 | 92.39 / 74.28 |
| Spikformer-4-384 w ReLU | 1 | 92.98 / 74.32 |
| Spikformer-4-384 w LeakyReLU | 1 | 92.88 / 74.31 |
| Spikformer-4-384 w VSA | 1 | 93.11 / 74.37 |
| Spikformer-4-384 w IF | 4 | 95.33 / 78.14 |

#### D.3 CIFAR에서의 추가 정확도 결과(Additional Accuracy Results on CIFAR)

표 6과 같이 CIFAR에서 추가 실험을 수행하였다.

#### D.4 ImageNet에서 수렴하지 않는 자기-어텐션 변형의 분석(Analysis of Self-Attention Variants Not Converging on ImageNet)

표 5에서 세 모델이 수렴하지 않는 이유는 다음과 같다. 그림 6(a)에서 보이듯, 시그모이드 대리 함수(sigmoid surrogate function)의 그래디언트는 평균 입력값 $V_i$와 발화 임계값 $V_{\text{th}}$의 차이가 너무 크거나 너무 작으면 소실(vanish)된다. 우리는 Spikformer-8-512 w I, Spikformer-8-512 w ReLU, Spikformer-8-512 w LeakyReLU, Spikformer-8-512 w SSA를 1 에폭 학습한 뒤, 식 (15)처럼 스파이킹 뉴런 층에 입력되는 $QK^\top V \cdot s$ 출력값을 수집하였다. 그림 6(b)에서 보이듯, 다른 세 변형에 비해 Spikformer-8-512 w SSA의 $QK^\top V \cdot s$ 값은 적절한 범위로 제어된다. 따라서 SSA는 학습 중 안정적인 대리 그래디언트를 가지며 수렴이 쉽다.

---

## 그림 6

**그림 6:** (a) 시그모이드 대리 함수(sigmoid surrogate function)와 그 그래디언트(gradient) 곡선. (b) $QK^\top V$의 값(value).

---

#### D.5 전이학습(Transfer Learning)

우리는 ImageNet에서 사전학습된 Spikformer를 CIFAR 다운스트림 데이터셋으로 전이하였다. ImageNet에서 사전학습된 Spikformer-4-384 및 Spikformer-8-384/512를 60 에폭 동안 파인튜닝(finetune)하였다. CIFAR의 입력 크기는 224×224이며, 나머지 하이퍼파라미터는 CIFAR을 직접 학습할 때와 동일하다. 표 7에示하듯 Spikformer는 높은 전이 능력을 보인다.

---

## 표 7

**표 7: CIFAR10/100에서의 전이학습(Transfer Learning).**

| Models | CIFAR10 | CIFAR100 |
|---|---:|---:|
| Spikformer-4-384 | 95.54 | 79.96 |
| Spikformer-8-384 | 96.64 | 82.09 |
| Spikformer-8-512 | 97.03 | 83.83 |
