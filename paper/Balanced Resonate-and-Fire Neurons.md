# 균형 잡힌 공진-발화 뉴런(Balanced Resonate-and-Fire Neurons)

Saya Higuchi, Sebastian Kairat, Sander M. Bohté, Sebastian Otte

1 적응형 AI 연구실(Adaptive AI Lab), 로보틱스 및 인지 시스템 연구소(Institute of Robotics and Cognitive Systems), 독일 뤼베크 대학교(University of Lübeck)  
2 머신러닝 그룹(Machine Learning Group), Centrum Wiskunde & Informatica(CWI), 네덜란드 암스테르담(Amsterdam)

교신저자(Correspondence): Saya Higuchi <saya.higuchi@student.uni-luebeck.de>, Sebastian Otte <sebastian.otte@uni-luebeck.de>

제41회 국제머신러닝학회(International Conference on Machine Learning) 발표 논문, 오스트리아 비엔나(Vienna, Austria), PMLR 235, 2024. 저작권은 저자에게 있다.

## 초록

20여 년 전에 제안된 공진-발화(resonate-and-fire, RF) 뉴런은 공진하는 막전위(membrane potential) 동역학 덕분에 시간 영역(time domain) 내의 주파수 패턴을 추출할 수 있는 단순하고 효율적이면서도 생물학적으로 그럴듯한 스파이킹(spiking) 뉴런 모델이다. 그러나 기존 RF 정식화(formulation)는 효과적인 학습을 제한하고 RF 뉴런의 원리적 장점을 충분히 활용하지 못하게 만드는 내재적 한계를 지닌다.

본 논문에서는 바닐라 RF 뉴런의 내재적 한계 일부를 완화하고 다양한 시퀀스 학습(sequence learning) 과제에서 순환 스파이킹 신경망(recurrent spiking neural networks, RSNNs) 내부에서의 유효성을 보이는 균형 RF(balanced RF, BRF) 뉴런을 소개한다. 우리는 BRF 뉴런으로 구성된 네트워크가 최신 RSNN과 비교했을 때 전반적으로 더 높은 과제 성능을 달성하고, 훨씬 적은 수의 스파이크를 생성하며, 유의하게 더 적은 파라미터를 필요로함을 보인다.

또한 BRF-RSNN은 시간 역전파(backpropagation through time, BPTT) 동안 수백 개의 시점(time step)을 거치는 경우에도 일관되게 훨씬 더 빠르고 안정적인 학습 수렴(convergence)을 제공한다. 이러한 결과는 BRF-RSNN이 향후 대규모 RSNN 아키텍처, 스파이킹 신경망(SNN) 방법론에 대한 추가 연구, 그리고 더 효율적인 하드웨어 구현을 위한 강력한 후보임을 보여준다.

키워드: 스파이킹 신경망(Spiking Neural Networks), 공진-발화(Resonate-and-Fire), 순환 신경망(Recurrent Neural Networks)

## 1. 서론

인공신경망(artificial neural networks, ANNs)은 최근 수년간 머신러닝 문제를 해결하는 주요 방법이 되었다(Goodfellow et al., 2016; Wu & Feng, 2018; Abiodun et al., 2019). 그러나 ANN은 대규모 실제 응용, 특히 엣지 컴퓨팅(edge computing) 영역에서 막대한 계산량과 에너지를 요구하므로 비효율적이다. 이는 데이터를 표현하기 위해 깊은 비선형 연속 추정기(deep non-linear continuous estimators)를 사용하기 때문이다(Pfeiffer & Pfeil, 2018).

스파이킹 신경망(spiking neural networks, SNNs)은 활동전위(action potentials), 즉 스파이크(spikes)의 정밀한 타이밍을 통해 정보를 처리함으로써 이러한 단점을 우회한다. SNN은 시스템 내부에서 스파이크가 전파될 때에만 계산이 필요하다는 사건 구동(event-driven) 특성 덕분에 ANN보다 잠재적으로 더 효율적일 수 있다(Paugam-Moisy & Bohté, 2012). 더 나아가 SNN은 뉴런이 시간에 따라 자신의 활동을 조절하는 동적 내부 상태(dynamic internal state)를 갖기 때문에 자기-순환(self-recurrent)적이며, 기존 ANN보다 생물학적으로 더 현실적이다. 생물학적 개연성이 증가하는 순서로 보면, 누설 적분-발화(leaky integrate-and-fire, LIF) 뉴런, 적응형 누설 적분-발화(adaptive leaky integrate-and-fire, ALIF) 뉴런(Bellec et al., 2018), Izhikevich 뉴런(Izhikevich, 2003), Hodgkin-Huxley(HH) 뉴런(Hodgkin & Huxley, 1952) 등이 대표적이다. HH 모델은 실제 적용에는 계산 비용이 지나치게 크지만, 더 단순한 모델에는 없는 풍부하고 복잡한 막 동역학을 보여준다.

머신러닝 응용에서 특히 주목할 만한 한 가지 동역학은 진동성 뉴런(oscillatory neurons)에서 나타나는 공진(resonating) 거동이다. 막전위의 문턱하 진동(subthreshold oscillations)은 전두엽 피질(frontal cortex) 뉴런(Llinas et al., 1991), 시상(thalamus)(Pedroarena & Llinás, 1997), 그리고 공간 정보 처리에 핵심적인 내측 내후각피질(medial entorhinal cortex) 제II층(Alonso & Llinás, 1989; Doeller et al., 2010) 등 다양한 포유류 신경계에서 관찰되었다. Izhikevich(2001)가 제안한 공진-발화(RF) 뉴런은 생물학적 뉴런의 이러한 감쇠 또는 지속 문턱하 진동을 모델링한다. RF 뉴런은 입력 스파이크의 주파수가 해당 뉴런의 감쇠 진동 주파수와 맞을 때 발화한다. 더 높은 주파수의 입력은 LIF 뉴런에서는 더 많은 발화를 유도하지만, 느리게 진동하는 RF 뉴런에서는 오히려 더 적은 발화를 유도한다. RF 뉴런은 LIF 뉴런과 유사한 수준의 계산 효율성을 가지므로 대규모 SNN에 적합할 수 있는 스파이킹 뉴런 모델이다(Izhikevich, 2001).

이전 연구들은 Intel Loihi 2(Davies et al., 2018)와 같은 뉴로모픽 프로세서(neuromorphic processor)에 구현된 RF 뉴런이 신호의 단시간 푸리에 변환(short-time Fourier transform, STFT)을 기존 방식보다 더 계산 효율적으로 수행할 수 있음을 보였다(Frady et al., 2022; Shrestha et al., 2023). 또한 RF 뉴런은 원시(raw) 데이터 신호를 스파이크 열(spike trains)로 성공적으로 변환하여, 이를 LIF 뉴런을 사용하는 SNN에 입력함으로써 탐지 및 분류 과제에 활용되었다(Shaaban et al., 2024; Hille et al., 2022; Lehmann et al., 2023). RF 뉴런은 (순방향) SNN 프레임워크 내에서 이미지 분류를 위한 조화 진동자(harmonic oscillators)로도 구현되었고(AlKhamissi et al., 2021), 광류(optical flow) 추정과 오디오 분류 과제에도 적용되었다(Frady et al., 2022). 그럼에도 RF 모델의 성능은 심층 LIF SNN(Frady et al., 2022)이나 LSTM 셀(AlKhamissi et al., 2021)을 유의미하게 넘어서지 못했으며, 이러한 연구들은 ALIF와 같은 최신 스파이킹 뉴런 모델과의 비교도 수행하지 않았다. 또한 의미 있는 파라미터 연구가 부족했고, 바닐라 RF 모델의 안정성 문제도 다루지 않았다. RF 뉴런의 장점을 고려하면, 이러한 결과는 RF 뉴런의 잠재력이 아직 충분히 탐구되지 않았음을 시사한다.

최근 순환 SNN(RSNN)을 학습하는 방법론의 발전은, 특히 BPTT와 결합될 때 시계열 학습(time series learning)에 대한 RSNN의 잠재력을 보여주었다(Bellec et al., 2018; Yin et al., 2021). 그럼에도 RSNN은 이른바 수렴 딜레마(convergence dilemma)에 직면한다. 즉, RSNN이 제대로 수렴하려면 보통 수백 에폭(epoch)이 필요하다(Yin et al., 2021; Fang et al., 2021; Zhang et al., 2023).

본 논문에서는 바닐라 RF 뉴런의 내재적 한계와 ALIF-RSNN 학습 중 발생하는 문제를 모두 극복하는 새로운 스파이킹 뉴런 모델, 즉 균형 RF 뉴런(BRF)을 제안한다. 그 결과, 제안하는 RF 변형은 ALIF 네트워크와 비슷하거나 더 높은 과제 성능을 달성할 뿐 아니라, 놀라울 정도로 빠르고 안정적으로 수렴하며, 최종 평균 정확도의 95%에 첫 다섯 에폭 안에 도달하면서도 ALIF 네트워크보다 최대 7배 적은 스파이크를 사용한다.

## 2. 공진-발화 뉴런(Resonate-and-Fire Neurons)

RF 뉴런의 막전위에서 나타나는 진동 거동은 다음의 두 개의 선형 미분방정식으로 정식화된다.

식 (1)

$$
\dot{x} = bx - \omega y + I
$$

식 (2)

$$
\dot{y} = \omega x + by
$$

이는 하나의 복소수 방정식으로 다음과 같이 쓸 수 있다.

식 (3)

$$
\dot{u} = (b + i\omega)u + I
$$

여기서 $u = (x + iy) \in \mathbb{C}$ 이고, $I$ 는 주입 전류(injected current)이다(Izhikevich, 2001). $\omega > 0$ 는 뉴런의 각주파수(angular frequency)로, 뉴런이 초당 몇 라디안(radians)을 진행하는지를 나타낸다. $b < 0$ 는 감쇠 계수(dampening factor)로, 진동을 지수적으로 감쇠시킨다. $b$ 가 더 작을수록 진동은 더 빠르게 휴지 상태(resting state)로 감쇠한다.

### Izhikevich RF 뉴런(Izhikevich RF Neuron)

우리는 식 (3)에 대해 시간 간격(time scale) $\delta$ 를 갖는 Euler 방법(Atkinson, 1989)을 적용하였다.

식 (4)

$$
u(t) = u(t - \delta) + \delta \left((b + i\omega)u(t - \delta) + I(t)\right)
$$

이는 그림 1의 RF 뉴런 시뮬레이션에 사용되었다. 그림 1은 유사한 주파수 타이밍을 갖는 입력이 각주파수 10 rad/s, 감쇠 계수 -1인 RF 뉴런에 주입될 때 나타나는 진동 거동을 보여준다.

그림 1. 두 RF 뉴런의 막 동역학(membrane dynamics). 네 개의 흥분성 입력 스파이크(excitatory input spikes)는 뉴런의 각주파수 $\omega = 10$ 과 위상(in phase)이 맞춰져 있다. 반 위상(half-phase)의 억제성 스파이크(inhibitory spike)는 뒤이어 오는 흥분성 입력에 대한 뉴런의 민감도를 높인다. 파라미터 설정에 따라, 뉴런은 $b = -0.3$, $\delta = 0.01$ 일 때 발산(divergence) 거동(위)을 보이거나, $b = -1$ 일 때 수렴(convergence) 거동(아래)을 보인다. $I(t) \in \mathbb{R}$ 과 $u_1(t), u_2(t) \in \mathbb{C}$ 는 각각 주입 전류와 뉴런의 막전위를 나타낸다.

### 조화 RF 뉴런(Harmonic RF Neuron)

조화 RF(harmonic RF, HRF) 뉴런에서는 막전위가 감쇠 조화 진동(dampened harmonic oscillation)의 동역학에 따라 변한다(AlKhamissi et al., 2021). 복소수 표현을 사용하는 대신, 막전위를 두 개의 상태(state)로 분리하여 다음과 같이 표현한다.

식 (5)

$$
\dot{u} = -2bu - \omega^2 v + I
$$

식 (6)

$$
\dot{v} = u
$$

Euler 적분(Euler integration)을 적용하면, 이산 시간(discrete time)에서 다음의 식을 얻는다.

식 (7)

$$
u(t) = u(t - \delta) + \delta\left(-2bu(t - \delta) - \omega^2 v(t - \delta) + I(t)\right)
$$

식 (8)

$$
v(t) = v(t - \delta) + \delta u(t - \delta)
$$

여기서 $b > 0$ 는 감쇠 계수이고, $\omega > 0$ 는 각주파수이다.

## 3. 균형 RF 모델(Balanced RF Models)

순차적 MNIST(sequential MNIST) 데이터셋(Deng, 2012)의 32개 샘플 무작위 부분집합을 이용한 RSNN 내 RF 뉴런의 초기 탐색(initial exploration)은 과도한 스파이킹, 발산 거동, 그리고 즉각적 공진을 방해하는 전통적인 하드/소프트 리셋(traditional hard and soft reset, 식 23)을 보여주었다.

발산은 연속 동적 시스템을 이산 단계(discrete steps)로 근사한 결과이며, 그림 1(위)에서 보이듯 $\omega$, $b$, 그리고 이산 시간 간격 $\delta$ 의 조합에 의존한다. 막전위는 발산하고 입력 신호와 무관한 스파이크를 계속 생성하며, 그 결과 원래의 주파수를 교란하는 잡음과 인공 신호가 만들어져 모델의 효과적인 학습을 방해할 수 있다.

### 균형 Izhikevich RF 뉴런(Balanced Izhikevich RF Neuron)

기본 RF 뉴런의 내재적 한계를 고려하여, 우리는 문턱값(threshold), 리셋 메커니즘(reset mechanism), 발산 경계(divergence boundary)를 변형한 균형 RF(BRF) 뉴런을 제안한다. 뉴런의 연속적인 스파이킹을 줄이고 스파이크 희소성(spiking sparsity)을 유도하기 위해, $q(t)$ 로 표기되는 불응기(refractory period)를 문턱값에 도입하여 뉴런이 발화한 뒤 문턱값이 증가하도록 하였다.

식 (9)

$$
\vartheta(t) = \vartheta_c + q(t)
$$

식 (10)

$$
z(t) = \Theta(\mathrm{Re}(u(t)) - \vartheta(t))
$$

여기서 $\vartheta_c$ 는 상수 문턱값(constant threshold), $z(t)$ 는 출력 스파이킹(output spiking)이다. 문턱값 메커니즘에는 즉각적인 반응을 유도하는 식 (3)의 실수부(real part)를 사용하였다. 불응기는 시간에 따라 지수적으로 감쇠한다.

식 (11)

$$
q(t) = \gamma q(t - \delta) + z(t - \delta)
$$

기본 불응기 상수(default refractory period constant)는 $\gamma = 0.9$ 이다.

기본 RF 모델의 또 다른 한계는 전통적인 리셋 메커니즘인데, 이 메커니즘은 진폭(amplitude)을 줄이기는 하지만 진동 자체를 교란한다. 이에 대한 대안으로, 뉴런이 발화한 뒤 불응기를 감쇠 항(dampening term)에 통합함으로써 진폭이 더 빨리 감쇠하도록 일시적으로 감쇠를 증가시키는 부드러운 리셋(smooth reset)을 제안한다.

식 (12)

$$
b(t) = b_c - q(t)
$$

여기서 $b_c$ 는 상수 감쇠 계수(constant dampening factor)이다.

불응기와 부드러운 리셋의 도입 효과는 그림 2에서 볼 수 있다. 단일 RF 뉴런은 $\omega = 10$, $b = -1$ 로 시뮬레이션되었고, 입력 신호는 주파수 타이밍이 맞도록 주어졌다. 불응기와 부드러운 리셋의 결합은 스파이크 수를 유의하게 줄였으며, 동시에 출력 스파이크는 뉴런 각주파수의 주기를 효과적으로 반영하였다. 두 메커니즘의 효과는 부록 A.8의 그림 10에서 더욱 강조된다.

그림 2. 주어진 입력 신호 $I(t)$ 에 대해, 불응기(refractory period, RP) 또는 부드러운 리셋(smooth reset, SR)이 없는 RF 뉴런(주황색)과 두 메커니즘을 모두 적용한 RF 뉴런(파란색)의 막전위 $u(t)$ 및 스파이킹 응답 $z(t)$.

발산 문제를 완화하기 위해, 우리는 수렴을 보장하는 $\delta$, $b_c$, $\omega$ 사이의 해석적으로 유도된 관계인 발산 경계 아래의 부분공간(subspace)으로 파라미터 공간(parameter space)을 제한할 것을 제안한다. RF 뉴런이 수렴하거나 지속 진동(sustained oscillation)을 보이기 위해서는, 입력 신호가 주어졌을 때 막전위의 크기(magnitude)가 시간에 따라 감소하거나 일정해야 한다.

식 (13)

$$
|u(t)| \le |u(t - \delta)|
$$

스파이크 시작(spike onset) 이후에는 크기가 수렴하지만 정확히 0이 되지는 않으므로, 즉 $|u(t - \delta)| \ne 0$ 이므로 양변을 $|u(t - \delta)|$ 로 나눌 수 있다.

식 (14)

$$
\frac{|u(t)|}{|u(t - \delta)|} \le 1
$$

또한 입력 스파이크 이후 막전위의 문턱하 진동 거동의 명시적 형태(explicit form)는 부록 A.3에서 유도되며 다음과 같다.

식 (15)

$$
u(t) = \delta \left(1 + \delta(b_c + i\omega)\right)^{\frac{t}{\delta} - 1}
$$

불응기와 리셋은 이 유도에서 고려하지 않았는데, 스파이크 이후 거동(post-spiking behavior)은 문턱하 거동을 모델링하는 데 중요하지 않기 때문이다. 이 명시적 형태를 위 부등식에 대입하여 단순화하면 다음을 얻는다.

$$
|1 + \delta(b_c + i\omega)| \le 1
\Leftrightarrow
\sqrt{(1 + \delta b_c)^2 + (\delta \omega)^2} \le 1
$$

피제곱수(radicand)가 양수이므로, 부등식의 양변을 제곱할 수 있다.

식 (16)

$$
(1 + \delta b_c)^2 + (\delta \omega)^2 - 1 \le 0
$$

감쇠 진동(damped oscillation)을 나타내기 위한 조건은 $b_c$ 에 대한 이차부등식(quadratic inequality)을 푸는 것으로 얻을 수 있으며, 상수 $\omega$ 가 주어졌을 때 $b_c$ 는 다음 해의 범위에 있어야 한다.

식 (17)

$$
\frac{-1 - \sqrt{1 - (\delta\omega)^2}}{\delta}
<
b_c
<
\frac{-1 + \sqrt{1 - (\delta\omega)^2}}{\delta}
$$

더 나아가 뉴런이 지속 진동자(sustained oscillator)가 되기 위한 조건은 다음과 같다.

식 (18)

$$
b_c = \frac{-1 \pm \sqrt{1 - (\delta\omega)^2}}{\delta}
$$

이는 $b_c \in \mathbb{R}_{<0}$ 이고 $\omega \in \mathbb{R}_{>0}$ 라는 점을 고려하면 뉴런에 대해 다음 조건을 이끌어낸다.

식 (19)

$$
\sqrt{1 - (\delta\omega)^2} > 0
\Rightarrow
\omega \le \frac{1}{\delta}
$$

기본값으로 $\delta = 0.01$ 을 사용하면, 뉴런이 공진할 수 있는 최고 주파수는 각주파수 100 rad/s에 해당하는 주파수이다. 우리는 이 각주파수를 상한 경계(upper boundary) $\omega_{ub}$ 로 정의한다. 지속 진동을 유도하는 $b_c$ 의 상한은 본 논문에서 발산 경계(divergence boundary)로도 간주되는 $p(\omega)$ 로 구현된다.

식 (20)

$$
p(\omega) = \frac{-1 + \sqrt{1 - (\delta\omega)^2}}{\delta}
$$

그리고 유연성과 수렴을 보장하기 위해 훈련 가능한(trainable) $b$-오프셋(offset) $b' > 0$ 를 결합하여, $b_c = p(\omega) - b'$ 로 둔다. 이는 하나의 시퀀스 길이(sequence length) 동안 상수로 유지된다. 그림 9는 여러 $\delta$ 값에 대한 예시 발산 경계를 보여준다. 유도된 감쇠 계수를 부드러운 리셋과 결합하면, 최적화에 적용되는 최종 $b(t)$ 식은 다음과 같다.

식 (21)

$$
b(t) = p(\omega) - b' - q(t)
$$

불응기, 부드러운 리셋, 발산 경계를 함께 구현하면 네 개의 데이터셋 모두에서 효율적인 학습과 희소한 스파이킹이 가능해진다.

### 균형 조화 RF 뉴런(Balanced Harmonic RF Neuron)

유사하게, 우리는 불응기, 부드러운 리셋, 그리고 $\omega$ 에 의존하는 $b$ 에 대한 맞춤형 발산 경계(tailored divergence boundary)를 가진 균형 HRF(balanced HRF, BHRF) 뉴런도 제안한다.

식 (22)

$$
p(\omega) = \frac{\omega^2}{200}
$$

$\delta = 0.01$ 이고 $b_c = p(\omega)$ 일 때, BHRF 뉴런은 지속 진동을 보인다.

### BRF 뉴런의 주파수 응답(Frequency Response of the BRF Neuron)

그림 3의 주파수 응답(frequency response) 플롯은 RF 뉴런이 특정 주파수에 얼마나 잘 반응하는지를 평가한다. 주파수 응답 플롯의 피크(peak)는 뉴런이 해당 입력 신호 주파수에 대해 높은 민감도를 가진다는 것을 의미한다(자세한 내용은 부록 A.4 참조). 그림은 응답 피크와 RF 뉴런의 $\omega$ 가 정렬됨을 보여주며, 이를 통해 공진 특성(resonance property)이 이산 경우(discrete case)에도 유지됨을 확인할 수 있다. 응답 피크와 RF 뉴런의 $\omega$ 사이의 약간의 오프셋(offset)은 막 동역학 미분방정식의 수치 적분(numerical integration)에서 생기는 인공물(artifact)이며, 시간 간격 $\delta$ 를 줄이면 사라진다.

그림 3. $\delta = 0.001$ 일 때, 예시적인 $\omega$ 및 $b'$ 조합에 대한 RF 뉴런 주파수 응답 플롯.

RF 뉴런은 협대역 대역통과 필터(narrow band pass filter)로도 볼 수 있으며, 필터링되는 특정 주파수 범위는 뉴런의 각주파수 $\omega$ 에 의해 결정된다. 여기에 이산 시간 간격 $\delta$ 와 감쇠 파라미터 $b'$ 가 결합되어, 대역통과 필터의 폭과 민감도가 결정된다. 전역적(global) 관점에서 RF 뉴런은 대응하는 각주파수 이하의 주파수만 필터링할 수 있으므로 저역통과 필터(low pass filter)로도 볼 수 있다(식 19). 또한 RF 뉴런은 자신의 각주파수의 저차 부분고조파(lower order subharmonics), 즉 민감도가 감소하는 형태로 $\frac{1}{2}\omega$, $\frac{1}{3}\omega$, $\frac{1}{4}\omega$ 등에 자연스럽게 민감하다.

우리는 BRF 뉴런에 초점을 맞추었고, BHRF는 아직 예비적인(preliminary) 탐색 단계에 머물렀다. 그림 8에서 보듯, BRF의 응답이 BHRF의 응답보다 입력 주파수에 더 민감했기 때문이다.

## 4. 네트워크 구현(Network Implementation)

우리는 RF, BRF, BHRF 뉴런을 RSNN 내부에 구현하여 여러 벤치마크 데이터셋에 대한 시뮬레이션에 적용하였다. BRF 및 BHRF 뉴런의 핵심 정식화는 각각 알고리즘 1과 알고리즘 2에 요약되어 있다. 알고리즘적 정식화에서는 표기법을 시간 이산 텐서 연산(time-discrete tensor operations)으로 바꾸고, 시간 인덱스를 위첨자로 표기하였다. 또한 $t$ 에서 $t+1$ 로의 전이는 $\delta$ 만큼의 시간 지연(time delay)으로 간주하였다.

우리는 이중 가우시안 함수(double-Gaussian function, 식 39; Yin et al., 2021)를 대리 기울기(surrogate gradient)로 사용하여 네트워크를 BPTT로 학습하였다(Bellec et al., 2018; Neftci et al., 2019). 네트워크에 대한 추가 세부사항은 부록 A.6에 기술한다.

알고리즘 1. BRF 순전파(BRF Forward Pass)

$$
b^t = p(\omega) - b' - q^{t-1}
$$

$$
u^t = u^{t-1} + \delta\left((b^t + i\omega)u^{t-1} + x^t\right)
$$

$$
\vartheta^t = \vartheta_c + q^{t-1}
$$

$$
z^t = \Theta(\mathrm{Re}(u^t) - \vartheta^t)
$$

$$
q^t = \gamma q^{t-1} + z^t
$$

$\vartheta_c = 1$, $\gamma = 0.9$, 그리고

$$
p(\omega) = \frac{-1 + \sqrt{1 - (\delta\omega)^2}}{\delta}
$$

$(\mathrm{Re}, \Theta, p)$ 는 성분별(component-wise)로 적용한다.

주 1. 소스 코드는 AdaptiveAILab/brf-neurons 저장소에서 이용할 수 있다.

알고리즘 2. BHRF 순전파(BHRF Forward Pass)

$$
b^t = p(\omega) - b' - q^{t-1}
$$

$$
u^t = u^{t-1} + \delta\left(-2b^t u^{t-1} - \omega^2 v^{t-1} + x^t\right)
$$

$$
v^t = v^{t-1} + \delta u^{t-1}
$$

$$
\vartheta^t = \vartheta_c + q^{t-1}
$$

$$
z^t = \Theta(u^t - \vartheta^t)
$$

$$
q^t = \gamma q^{t-1} + z^t
$$

$\vartheta_c = 1$, $\gamma = 0.9$, 그리고

$$
p(\omega) = \frac{\omega^2}{200}
$$

$(\mathrm{Re}, \Theta, p)$ 는 성분별(component-wise)로 적용한다.

그림 4. 데이터셋 개요. 예시적인 MNIST 이미지와 그에 대응하는 순차 표현(sequential representation) 및 순열화된 표현(permuted representation). MNIST 및 S-MNIST 샘플에서 공통 픽셀 행(common pixel row)은 빨간색으로 표시하였다. ECG 샘플은 레벨 크로스 인코딩(level-cross encoding) 이후의 형태이고, SHD 샘플은 전처리 후의 형태이다.

## 5. 실험(Experiments)

MNIST 데이터셋은 분류를 위한 $28 \times 28$ 크기의 회색조 필기 숫자 이미지로 구성된다. 순차적 MNIST(sequential-MNIST, S-MNIST)는 이미지를 $1 \times 784$ 길이의 시퀀스로 변환한 것으로, 54,000개의 학습 이미지, 6,000개의 검증 이미지, 10,000개의 테스트 이미지로 이루어져 있어 순차 모델 간 비교를 가능하게 하는 대표 벤치마크 데이터셋이다. 순열화된 S-MNIST(permuted S-MNIST, PS-MNIST) 변형에서는 픽셀 위치를 먼저 무작위로 섞은 다음, 그 순서를 고정한 채 순차적으로 네트워크에 입력한다.

심전도(electrocardiogram, ECG) 기록은 시간에 따른 전압으로 표현되며, P, PQ, QR, RS, ST, TP의 여섯 가지 특징적 파형(characteristic waveforms)으로 이루어진 주기적 활동(cyclic activity)을 포함한다. QT 데이터베이스는 분야 전문가가 시점별(per-time step) 라벨을 부여한 ECG 기록으로 구성된다(Laguna et al., 1997). 전체 기록은 105개였고, 각 기록은 두 개의 전극으로 15분 동안 측정되었다. 주석(annotation)이 없는 급사(sudden death) 기록 24개는 제외하였다. 그림 4에 나타낸 전처리에 대한 자세한 내용은 부록 A.7을 참조하라.

Spiking Heidelberg dataset(SHD)은 SNN을 위해 특별히 생성된 오디오-투-스파이크(audio-to-spike) 벤치마크 데이터셋이다(Cramer et al., 2020). 이 데이터셋은 영어와 독일어로 발화된 숫자 0부터 9까지의 10,420개 녹음으로 이루어진다. 녹음은 높은 충실도(high fidelity)로 이루어진 후, 정밀한 인공 내이(artificial inner ear) 모델을 통해 700개 채널의 스파이크 열로 변환되었다(Cramer et al., 2020). 이후 이 스파이크 열은 이산 시간 간격 $4e^{-3}$ s로 추가 처리되어, 0 패딩(zero-padding)을 포함한 길이 250의 시퀀스로 변환되었다. 이 전처리는 Yin et al. (2021)과 비교 가능하도록 하기 위해 수행되었다. 우리는 7,341개 시퀀스를 학습에, 815개를 검증에, 2,264개를 추론(inference)에 사용하였다.

## 6. 결과(Results)

네트워크 아키텍처, 파라미터 수, 생성된 스파이크 수에 대한 결과는 최상의 BRF-, RF-, BHRF-RSNN 및 기준선 ALIF-RSNN(부록 A.2 참조), 그리고 기타 최신 성능(state-of-the-art, SoTA) 모델들 사이에서 각 분류 과제별로 표 1에 비교되어 있다. 각 데이터셋에 대한 BRF- 및 BHRF-RSNN의 하이퍼파라미터(hyperparameters)는 부록 A.12의 표 3에 제시되어 있다. BHRF와 BRF 모델은 각각 네 개와 세 개의 과제에서 RSNN 최신 성능을 능가하였다. 정확도뿐 아니라, BRF 및 BHRF 네트워크는 ALIF 네트워크보다 훨씬 적은 파라미터를 사용하면서도 더 높은 희소성(sparseness)을 보여주었다. 또한 표 1의 RF 모델에는 리셋(reset)이 포함되지 않았다는 점에 유의해야 한다. 바닐라 RF 뉴런에 하드 리셋이나 소프트 리셋을 추가하면 과제 성능이 극적으로 변하며, 이는 부록 A.9의 그림 11에서 확인할 수 있다.

표 1. 다양한 순차 분류 과제에서 바닐라 RF, BRF, BHRF, ALIF(Yin et al., 2021), 그리고 기타 최신 성능 모델인 DCLS-Delays(DCLS-D)(Hammouamri et al., 2023) 및 RadLIF(Bittar & Garner, 2022)의 결과 비교. Architecture는 각 층(layer)의 뉴런 수를 의미한다. RF, BRF, BHRF 모델의 정확도는 5회 실행 평균이며, PS-MNIST BHRF는 제외하였다. SOPs는 평균 스파이크 연산(spike operations)이다. `*` 는 RSNN 최신 성능(RSNN-SoTa), `**` 는 SNN 최신 성능(SNN-SoTa)을 의미한다.

| Task | Model | Architecture | No. params. (↓) | Test Acc. (↑) | SOPs (↓) | SOPs/step (↓) | SOP Ratio (↓) |
|---|---|---:|---:|---:|---:|---:|---:|
| S-MNIST | ALIF* | 4,64,(256)$^2$,10 | 156,126 | 98.7 % | 70,810.8 | 90.32 | 1.00× |
| S-MNIST | RF | 1,256,10 | 68,874 | 98.0±0.4 % | 29,034.8 | 37.03 | 0.41× |
| S-MNIST | BRF | 1,256,10 | 68,874 | 99.0±0.1 % | 15,462.6 | 19.72 | 0.22× |
| S-MNIST | BHRF | 1,256,10 | 68,874 | 99.1±0.1 % | 21,565.7 | 25.51 | 0.30× |
| PS-MNIST | ALIF* | 4,64,(256)$^2$,10 | 156,126 | 94.3 % | 59,772.1 | 76.24 | 1.00× |
| PS-MNIST | RF | 1,256,10 | 68,874 | 9.9±0.8 % | 66,474.2 | 84.79 | 1.11× |
| PS-MNIST | BRF | 1,256,10 | 68,874 | 95.0±0.2 % | 27,839.7 | 35.51 | 0.46× |
| PS-MNIST | BHRF | 1,256,10 | 68,874 | 95.2 % | 24,564.2 | 33.33 | 0.41× |
| ECG | ALIF* | 4,36,6 | 1,776 | 85.9 % | 35,011.2 | 26.93 | 1.00× |
| ECG | RF | 4,36,6 | 1,734 | 85.5±0.7 % | 11,981.9 | 9.22 | 0.34× |
| ECG | BRF | 4,36,6 | 1,734 | 85.8±0.7 % | 6,307.7 | 4.85 | 0.18× |
| ECG | BHRF | 4,36,6 | 1,734 | 87.0±0.4 % | 6,233.8 | 4.80 | 0.18× |
| SHD | DCLS-D** | 700,(256)$^2$,20 | ≈200,000 | 95.1±0.2 % | - | - | - |
| SHD | RadLIF* | 700,(1024)$^3$,20 | 3,893,288 | 94.62 % | - | - | - |
| SHD | ALIF | 700,(128)$^2$,20 | 142,120 | 90.4 % | 24,690.0 | 98.76 | 1.00× |
| SHD | RF | 700,128,20 | 108,820 | 89.2±0.6 % | 4,750.2 | 19.00 | 0.19× |
| SHD | BRF | 700,128,20 | 108,820 | 91.7±0.8 % | 3,502.6 | 14.01 | 0.14× |
| SHD | BHRF | 700,128,20 | 108,820 | 92.7±0.7 % | 4,139.5 | 16.56 | 0.17× |

우리는 또한 RF 및 ALIF 모델의 이론적 에너지 효율(theoretical energy efficiency)을, 각각의 총 스파이크 연산(SOPs; 식 41 참조)과 시퀀스 단계당 SOPs를 계산하여 비교하였다. 이 두 지표 모두 BRF와 BHRF 모델에서 상당히 더 작았으므로, 이들 모델은 더 적은 계산으로 더 좋거나 비슷한 성능을 달성할 수 있음을 의미한다. 이는 뉴런이 주파수를 표현하는 데 더 적은 스파이크를 필요로 하는 공진적 스파이킹 거동(resonating spiking behavior)과, 연속적 스파이킹을 억제하는 불응기 및 부드러운 리셋 때문일 수 있다. 특히 SHD 과제에서는 BRF 모델이 ALIF 모델의 14% 수준의 스파이크만 발생시키는 큰 차이가 관찰되었다.

리셋이 없는 표준 RF 모델과 비교하면, 우리의 균형 변형들은 훨씬 더 높은 성능을 유지하면서도 유의하게 더 희소한 활동을 보였다. 특히 PS-MNIST의 경우 표준 RF 모델은 우연 수준(chance level)에 머문 반면, 균형 RF 변형들은 RSNN 최신 성능을 능가하였다. 이는 높은 발산 거동을 유발하는 감쇠 계수와 각주파수의 조합 때문으로 보인다. PS-MNIST 데이터셋은 증가한 무작위성을 상쇄하기 위해 더 넓은 범위의 공진 주파수를 요구하기 때문에 이러한 문제가 특히 크게 나타난다. 예를 들어, $\omega = 70$ rad/s 와 $b_c = -1$ 인 뉴런은 시퀀스 끝으로 갈수록 더 큰 크기(magnitude)를 갖는다. 이러한 뉴런이 많이 존재하면 시스템 전체가 발산하여 불안정해진다. 한편, BRF 모델의 성능은 파라미터 초기화(initialization)에 의존하며, 각주파수가 데이터셋의 기저 주파수(underlying frequencies)와 너무 멀리 떨어져 있을 때는 전반적으로 성능이 떨어진다.

전반적으로, BRF 또는 BHRF 뉴런으로 구성된 RSNN은 모든 분류 과제에서 기준선 ALIF-RSNN보다 더 나은 성능을 보였다. 표준 RF 뉴런과 비교할 때, 부드러운 리셋, 불응기, 발산 경계는 RF 파라미터의 안정성과 효율을 유의하게 향상시켰으며, 이는 모델의 공진 특성(resonant properties)을 더 잘 활용했기 때문일 수 있다. 이에 대해서는 아래에서 더 자세히 논의한다.

### 수렴(Convergence)

RF, BRF, ALIF 모델의 학습 곡선(learning curves)은 그림 5에 제시되어 있다. 우리는 ALIF 모델을 BRF 모델과 동일한 크기 및 학습 가능한 파라미터 수를 갖도록 줄였다. S-MNIST와 PS-MNIST 과제에서는 절단 단계(truncation step)가 50인 절단 BPTT(truncated BPTT, TBPTT)를 적용할 때만 ALIF 네트워크가 효과적으로 학습된다. 그럼에도 전체 역전파(full backpropagation)가 적용된 BRF 네트워크는 TBPTT 기반 ALIF 네트워크보다 훨씬 더 빠르게 수렴하며, 이는 정량적 수렴 결과를 보여주는 표 2에서도 분명히 확인된다.

ECG 데이터셋과 SHD에서도 유사하게 빠르고 안정적인 수렴 패턴이 관찰된다. SHD와 ECG에서 RF와 BRF의 수렴 양상이 비슷한 이유는, 바닐라 RF 뉴런이 더 낮은 주파수에서 공진하므로 발산 거동의 영향을 덜 받기 때문이다. 그림 5에 제시된 RF 모델은 리셋이 포함되지 않은 것이다. 소프트 리셋이나 하드 리셋을 추가하면 수렴이 더 느리고 불안정해지며, 이는 부록 A.9의 그림 11에서 확인할 수 있다. 그림 5는 데이터셋에 따라 RF 네트워크의 학습 양상이 크게 달라짐을 보여주며, 이러한 변동은 BRF 네트워크에 의해 완화된다.

그림 5. 위쪽 행: BRF, RF, ALIF 모델 간 S-MNIST, PS-MNIST, ECG, SHD의 학습 곡선. 각 곡선은 5회 실행에 대해 에폭별 평균(실선)과 표준편차(음영)을 나타낸다. 정확도 곡선 위의 점(dot)은 최종 정확도의 95%에 도달한 시점을 나타낸다. 아래쪽 행: S-MNIST, PS-MNIST, ECG, SHD에 대해 모든 실행에서의 초기 BRF 파라미터와 최적화된 BRF 파라미터 조합, 즉 각주파수 $\omega$ 와 $b$-오프셋 $b'$ 를 나타낸다. 점선은 발산 경계(divergence boundary)이다. RF 모델은 리셋 없이 시뮬레이션되었다. 전통적 리셋에서의 수렴은 부록 A.9를 보라.

표 2. 정량적 수렴 결과. 학습 곡선에서 최종 테스트 정확도의 95%, 98%, 100%에 도달한 뒤의 평균 에폭 수.

| Task | Model | 95 % (↓) | 98 % (↓) | 100 % (↓) |
|---|---|---:|---:|---:|
| S-MNIST | ALIF | 105 | 162 | 276 |
| S-MNIST | RF | 134 | 172 | 263 |
| S-MNIST | BRF | 3 | 29 | 246 |
| S-MNIST | BHRF | 5 | 12 | 119 |
| PS-MNIST | ALIF | 143 | 200 | 265 |
| PS-MNIST | BRF | 10 | 39 | 282 |
| PS-MNIST | BHRF | 6 | 19 | 123 |
| ECG | ALIF | 51 | 157 | 282 |
| ECG | RF | 8 | 29 | 112 |
| ECG | BRF | 6 | 32 | 75 |
| ECG | BHRF | 15 | 38 | 109 |
| SHD | ALIF | 14 | 15 | 16 |
| SHD | RF | 2 | 4 | 5 |
| SHD | BRF | 2 | 3 | 7 |
| SHD | BHRF | 3 | 5 | 8 |

(B)RF-RSNN의 학습률은 수치적으로 상당히 컸음에도 대부분의 데이터셋에서 안정적이고 빠른 수렴을 이끌어냈다. 이는 큰 학습률이 성능 저하로 이어지는 일반적인 SNN과 대조적이다. 여기서 우리가 사용한 이중 가우시안 대리 기울기 함수(Yin et al., 2021)는 발화 임계 근처에서 멀리 떨어진 뉴런에 대해서도 기울기 흐름(gradient flow)을 보장해 주었을 가능성이 있다. 높은 학습률과 결합되면서, 이는 발화하지 않는(non-spiking) 뉴런이 자신의 각주파수를 효과적으로 이동시키는 데 기여했을 수 있다.

ECG 과제는 작은 배치 크기(batch size)와 높은 학습률을 사용할 때 BRF와 BHRF 네트워크에서 가장 효과적으로 학습되었다. 작은 배치 크기는 데이터셋 내의 세밀한 변이를 더 잘 포착할 수 있으며, 특히 입력 스파이크의 타이밍을 잘 반영하는데, 이것이 시점별 파형 분류(per-step wave classification)에 중요했을 가능성이 있다.

현재 진행 중인 연구(Higuchi et al., 2024)는 오류 지형(error landscape)의 형상이 BRF 모델의 빠르고 안정적인 수렴에 큰 기여를 할 수 있음을 시사한다(부록 A.11의 그림 13 참조).

### 파라미터 분석(Parameter analysis)

우리는 그림 5에 보이듯, 뉴런이 어떻게 작동하는지에 대한 직관을 얻기 위해 학습 전후의 훈련 가능한 BRF 파라미터 $\omega$ 와 $b'$ 를 비교하였다. 최적화는 이 파라미터들을 크게 이동시켰으며, 이는 B(H)RF 네트워크 내부에서 기울기가 효과적으로 전파되었음을 보여준다.

S-MNIST 과제에서 $\omega-b_c$ (그림 5)를 그려보면, $\omega$ 값이 18~28 rad/s 부근과 40 rad/s 부근에 군집(cluster)을 이루며, $b_c$ 는 넓은 범위에 퍼져 있다. 감쇠 계수가 0에 가까운 것은 거의 지속 진동(near-sustained oscillation) 거동을 나타내며, 이는 오랜 기간 공진 진폭(resonance amplitude)을 유지하게 한다. $b_c$ 값의 군집 다양성은 B(H)RF 네트워크가 S-MNIST를 효과적으로 학습하기 위해 단기 기억(short-term memory)과 장기 기억(long-term memory)을 모두 필요로 함을 시사한다. 또한 S-MNIST 과제에서 $\omega$ 의 분포는 이봉 분포(bimodal distribution)를 보였는데, 가장 높은 피크는 23.3~25.3 rad/s, 두 번째 피크는 40.8~42.8 rad/s 부근에 있었다.

MNIST 숫자들은 모두 어떤 형태로든 대각선(diagonal line)을 포함하며, 이는 특히 숫자 1에서 두드러지게 보이고 폭은 대략 3~6 픽셀이다. 이를 행(row)-기반 방식으로 시퀀스로 변환하면, 대각선은 시퀀스 전반에 걸쳐 약 25 픽셀의 주기를 갖는 주기 신호(periodic signal)처럼 분포한다. 픽셀을 $\delta = 0.01$ 의 이산 시간 단계로 간주하면, 신호에서 자주 관찰되는 이론적 주기 $T$ 는 0.25 s이다. 따라서 신호의 기저 주파수에 해당하는 가장 빈번한 이론적 각주파수 $\omega'$ 는 다음과 같다.

$$
\omega' = \frac{2\pi}{0.25\ \mathrm{s}} = 25.13\ \mathrm{rad/s}
$$

이는 BRF 뉴런이 가장 자주 학습한 각주파수와 가깝다. 이 단순 계산은 BRF 뉴런이 S-MNIST 데이터셋의 기저 주파수를 의미 있게 학습했음을 시사한다. 반면 PS-MNIST에서는 S-MNIST의 특징적 주파수 구조가 순열화(permutation)로 사실상 가려지기 때문에, $\omega$ 와 $b_c$ 파라미터가 더 넓게 퍼져 있는 것을 볼 수 있다.

ECG 스파이크 열 샘플(그림 4)을 보면, 300 시간 단계 간격의 뚜렷한 주기 스파이크가 관찰되며, 이는 이론적 각주파수 $\omega' = 2.09$ rad/s 에 해당한다. 실제로 $\omega-b_c$ 플롯에는 약 2 rad/s 부근에 군집이 존재한다. 이론적 각주파수와 최적화된 각주파수가 정렬되어 있다는 점은 네트워크가 동적 신호(dynamical signal)를 학습했음을 보여준다.

그림 6. 학습 후 레이블 10을 갖는 SHD 샘플에 대한 네트워크 활동의 래스터 플롯(raster plot). BRF 뉴런들은 각주파수에 따라 정렬되어 있다. 검은 점은 출력 스파이크를 나타낸다.

### 가중치 희소성(Weight Sparsity)

우리는 S-MNIST와 PS-MNIST에 대해 비교 가능한 BRF 및 ALIF 네트워크 사이에서 순환 가중치(recurrent weights) 프루닝(pruning)의 영향을 절제 연구(ablation study)로 탐색하였고, 이에 대한 세부 하이퍼파라미터는 표 3에, 결과는 그림 7에 제시하였다. BRF 뉴런은 프루닝 전반에 걸쳐 일관된 성능을 제공하는 반면, ALIF 모델은 순환 가중치가 부족해지면 성능이 급격히 떨어진다. 이는 공진 특성(resonating properties)을 지닌 BRF 네트워크가 합리적인 수준으로 학습하는 데에는 명시적 순환 연결(explicit recurrencies)이 필수적이지 않은 반면, ALIF 네트워크에는 이러한 연결이 중요함을 보여준다. 또한 BRF 네트워크의 정확도는 전반적으로 ALIF 네트워크보다 표준편차가 더 작으며, 이는 BRF 네트워크의 성능이 특정하게 프루닝된 가중치에 덜 민감함을 시사한다.

그림 7. (a) S-MNIST 및 (b) PS-MNIST에서 프루닝 확률(pruning probability)에 따른 테스트 정확도. 프루닝 1.0은 순환 가중치가 없음을 의미한다.

## 7. 논의(Discussion)

BRF-RSNN 구현 결과는 RF 뉴런이 어떻게 작동하는지에 대해 상당한 통찰을 제공한다. 특히 각 데이터셋의 학습 곡선 및 파라미터 분석은 BRF 네트워크가 공진 거동(resonating behavior)을 유리한 방향으로 조정함으로써 의미 있는 파라미터를 학습하고, 아직 대규모 시뮬레이션에서 관찰되지 않았던 복잡한 동역학을 모델링함을 보여준다. 우리는 모든 데이터셋에서 래스터 플롯에 나타난 주기적 스파이크(periodic spikes)의 주파수가 더 높은 각주파수와 함께 증가하는, 막전위의 바람직한 진동 거동을 관찰하였다. SHD의 경우, 입력 신호의 0 패딩이 있음에도 시퀀스 후반부에 출력 스파이크가 존재한다. 대부분의 신호가 여전히 주기적이라는 점을 고려하면, 이는 시퀀스 끝까지 관련 주파수를 유지하는 제어된 진동과 스파이킹(controlled oscillation and spiking)을 시사한다.

모든 RF 모델에 대해 우리는 바이어스(bias)를 사용하지 않고 실험을 수행했는데, 이는 네트워크 동역학을 살아 있게 유지하고 과제를 해결하는 데 얼마나 많은 스파이크가 필요한지를 조사하기 위함이었다. 그 결과, 네트워크 내 BRF 뉴런 중 일부는 그림 6에서 보듯 시퀀스 전반에 걸쳐 지속적으로 발화하는 법을 학습하였고, 이는 사실상 네트워크에 일정한 전류를 주입하는 바이어스를 흉내낸 것이다. 반대로 ALIF 네트워크는 잘 작동하기 위해 명시적인 바이어스가 필요하다. 따라서 BRF 네트워크는 적분형 뉴런(integrator neurons)만으로는 학습할 수 없는 행동을 모델링하는 데 더 유연하다.

뉴런들은 각자 선호하는 주파수를 학습하며, 그 결과 개별 시간 스케일(individual time scales)에 집중한다. 이는 순환 네트워크 내부에서 분산적이고 집단적인 과제 해결(distributed, collective task solving)의 맥락에서 지역 정보 이득(local information gain)을 극대화하는 항상성 조절 과정(homeostatic regulatory process)으로 느슨하게 볼 수 있다.

사후 학습된 모델(post-trained models)에 대해 여러 유형의 잡음(noise) 및 수치 양자화(numerical quantization)를 추가로 탐색한 결과(Stromatias et al., 2015; Park et al., 2021), 부록 A.10의 그림 12에 자세히 제시된 바와 같이 BRF 네트워크는 ALIF 네트워크에 비해 일관되게 높은 강건성(robustness)을 보였다. 일부 잡음 변형에 대해서는 BRF 네트워크가 RF 네트워크보다도 약간 더 강건했으며, 이때에도 BRF 네트워크의 SOP는 지속적으로 낮게 유지되었다. 하드웨어 구현은 이러한 잡음에 취약한 경향이 있으므로(Stromatias et al., 2015), 이는 BRF 뉴런이 하드웨어 응용에 적합함을 시사한다. 우리는 또한 RF 뉴런 변형을 지원하는 Loihi 2 상에서 이들 뉴런으로 추론을 구현하는 것도 목표로 하고 있다(Davies et al., 2018; Shrestha et al., 2023).

네트워크 차원(network dimension)에 관해서는, 단일 은닉층(single hidden layer)에 공진 뉴런을 일정 수준 이상 추가하더라도 성능이 더 유의미하게 개선되지는 않았다.

BRF- 및 BHRF-RSNN의 한 가지 단점은 각주파수와 감쇠 오프셋(dampening factor offset)의 최적 초기 파라미터화를 찾기 어렵다는 점이다. 이 두 요소가 모델 성능을 모두 결정하기 때문이다. 파라미터 분석 결과는, 데이터셋을 최적으로 학습하기 위해 BRF 뉴런이 필요로 하는 각주파수 범위를 사전에 근사하기 위해 푸리에 변환(Fourier transform)을 적용할 가능성을 시사한다.

또한 BRF-RSNN의 계산 복잡도(computational complexity)는 단일 은닉층 ALIF 모델과 유사하다는 점을 언급할 필요가 있다. 그러나 수렴이 더 빠르므로 학습을 훨씬 더 일찍 중단할 수 있고, 결과적으로 학습 시간이 크게 줄어든다. 구현 측면에서는, 대리 기울기 함수를 모델링하기 위해 순방향 기울기 주입(forward gradient injection)(Otte, 2024)을 사용하고, 이를 PyTorch의 TorchScript와 같은 자동 모델 최적화 루틴과 결합함으로써 추가적인 속도 향상이 가능하다(일부 결과는 부록 A.13에 제시한다).

## 8. 결론(Conclusion)

우리는 표준 RF 뉴런으로 구성된 네트워크에서 나타나는 학습의 어려움을 해결하면서 내부 공진 상태(internal resonating state)가 일종의 장기 기억(long-term memory)으로서 효과적임을 보여주는, 원리적인 공진 스파이킹 뉴런 모델(principled resonating spiking neuron model)로서 BRF- 및 BHRF-RSNN을 제안하였다. 더 작은 네트워크 아키텍처와 더 희소한 스파이킹을 가진 BRF- 및 BHRF-RSNN은 심층 기준선 모델(deep baseline models)을 능가하였다. 학습 곡선은 BRF 네트워크가 ALIF 네트워크보다 훨씬 더 빠르게 수렴함을 보여주었다. 더 나아가 안정적인 수렴은 BRF 모델 성능의 좋은 재현성(reproducibility)을 시사한다. BRF 파라미터 분석은 네트워크가 입력 신호의 기저 주파수를 학습하고, ALIF 네트워크와 비교하여 더 희소한 순환 연결(sparser recurrent connectivity)에 대해서도 강건함을 보여주었다.

향후 연구는 이러한 시뮬레이션을 CIFAR(Krizhevsky & Hinton, 2009)나 Google Speech Command 데이터셋(Warden, 2018)처럼 훨씬 더 크고 복잡한 문제로 확장하는 방향으로 나아갈 수 있다. BRF 모델은 시간 영역 내의 주기 패턴(periodic pattern)을 추출하는 능력이 있으므로, 원시 오디오 처리(raw audio processing) 과제에 적용하는 것도 특히 흥미롭다.

또 다른 연구 방향은 BRF 뉴런을 e-prop(Bellec et al., 2020)과 같은 온라인 학습(online learning) 접근법과 결합하는 것이다. 이는 더 긴 시퀀스를 더 빠르게 처리하고 메모리 효율성(memory efficiency)을 높일 수 있으므로 응용 범위를 넓혀줄 것이다. 또한 보다 유연한 적응형 문턱화(adaptive thresholding)를 위해 훈련 가능한 불응기 감쇠(trainable refractory period decays)를 도입할 수 있고, 다중 시간 해상도 처리(multi-temporal resolution processing)를 촉진하기 위해 훈련 가능한 가변 시간 상수(simulated variable time constants)도 함께 도입할 수 있다.

본 연구는 순환 스파이킹 신경망의 맥락에서 안정적인 공진 뉴런(stable resonating neurons)의 작동을 조사하기 위한 초기 탐색(initial exploration)에 해당한다. 우리의 모델은 RSNN에 대한 최신 성능과 비슷하거나 그 이상을 보인다. 또한 구현을 가능한 한 단순한 형태로 유지했기 때문에, 향후 BRF-RSNN 변형에 대한 추가 연구를 위한 기초를 제공한다.

## 감사의 말(Acknowledgements)

실험의 일부는 튀빙겐 대학교(University of Tübingen)의 머신러닝 클러스터 탁월성 센터(Machine Learning Cluster of Excellence, EXC number 2064/1 – Project number 390727645)의 ML Cloud 인프라를 사용하여 수행되었다.

Sebastian Otte는 Alexander von Humboldt Foundation의 Feodor Lynen fellowship의 지원을 받았다.

## 영향 진술(Impact Statement)

본 연구의 목적은, 특히 스파이킹 신경망(SNN)에 초점을 맞추어 인공지능 시스템의 효과성과 에너지 효율을 증대시킴으로써 머신러닝 분야를 발전시키는 데 있다. 머신러닝 연구의 대부분의 발전과 마찬가지로, 우리의 연구 역시 다양한 사회적 영향을 가질 수 있다는 점은 주목할 만하다. 그러나 우리는 그 어떤 영향도 특별한 고려가 필요한 수준이라고 보지는 않는다.

# 부록 A

## A.1 전통적인 소프트 리셋과 하드 리셋(Traditional soft and hard reset)

전통적인 소프트 리셋(soft reset)과 하드 리셋(hard reset) 메커니즘은 각각 다음과 같이 정식화된다.

식 (23)

$$
u(t) = u'(t) - (1 + i)z(t)\vartheta
$$

식 (24)

$$
u(t) = [1 - (1 + i)z(t)]u'(t)
$$

여기서 $u'(t)$ 는 리셋 이전의 막전위(membrane potential)를 나타낸다.

## A.2 기준선 ALIF 뉴런(Baseline ALIF neuron)

기준선 ALIF 뉴런(Yin et al., 2021)의 정식화는 $\vartheta = 0.01$, $\beta = 1.8$ 일 때 다음과 같다.

식 (25)

$$
\vartheta_t = \vartheta + \beta a(t)
$$

식 (26)

$$
a(t) = \rho a(t - \delta) + (1 - \rho)z(t - \delta)
$$

식 (27)

$$
u'(t) = \alpha u(t - \delta) + (1 - \alpha)I(t)
$$

식 (28)

$$
z(t) = \Theta(u'(t) - \vartheta_t)
$$

식 (29)

$$
u(t) = u'(t) - z(t)\vartheta_t
$$

여기서 $\rho = e^{-\delta/\tau_a} \in (0, 1)$ 는 적응형 문턱값 감쇠 상수(adaptive threshold decay constant), $\tau_a$ 는 그 시간 상수(time constant), 그리고 $\alpha = e^{-\delta/\tau_m} \in (0, 1)$ 는 막전위 감쇠 상수(membrane potential decay constant)이다. $a(t)$ 는 뉴런의 스파이킹 거동이 누적된 활동량(accumulative activity)이다. 뉴런이 발화할 때 $(z(t)=1)$ 막전위 $u(t)$ 는 적응형 문턱값 $\vartheta_t$ 로 소프트 리셋된다.

## A.3 Izhikevich RF 뉴런의 명시적 형태(Explicit form of the Izhikevich RF neuron)

막전위 방정식은 다음과 같이 다시 쓸 수 있다.

$$
u(t) = (1 + \delta(b + i\omega))u(t - \delta) + \delta I(t)
$$

이때 이산 시간 간격 $\delta$ 와 $t = 0, \delta, 2\delta, 3\delta, \ldots, T\delta$ 를 고려하자. 또한 $t = \delta$ 에서만 스파이크가 주입되어 $I(\delta)=1$ 이고, 초기 막전위 $u(0)=0$ 이라고 가정하면 다음이 성립한다.

식 (30)

$$
u(\delta) = \delta
$$

식 (31)

$$
u(2\delta) = \delta(1 + \delta(b + i\omega))
$$

식 (32)

$$
u(3\delta) = (1 + \delta(b + i\omega))\left(\delta(1 + \delta(b + i\omega))\right)
$$

식 (33)

$$
u(3\delta) = \delta(1 + \delta(b + i\omega))^2
$$

식 (34)

$$
u(4\delta) = (1 + \delta(b + i\omega))\left(\delta(1 + \delta(b + i\omega))^2\right)
$$

식 (35)

$$
u(4\delta) = \delta(1 + \delta(b + i\omega))^3
$$

식 (36)

$$
\cdots
$$

식 (37)

$$
u(t) = \delta(1 + \delta(b + i\omega))^{\frac{t}{\delta} - 1}
$$

## A.4 주파수 응답 플롯 생성(Frequency response plot generation)

문턱하 응답(subthreshold responses)은 RF 뉴런에 대해 무작위로 선택한 각주파수 $\omega \in [0, 100)$ 과 $b' \in (0, 10)$ 를 사용하여 탐색하였다. 각주파수에 상대적인 $\{0.1, 0.2, \cdots, 100\}$ 의 주파수를 갖는 스파이킹 입력 신호를 $\delta = 0.001$ 의 이산 시간 간격으로 20초 동안 뉴런에 입력하였다. 양의 스파이크(positive spikes)는 각주파수에 해당하는 주기의 위상(in-phase)에 맞춰 입력되었다.

식 (38)

$$
T = \frac{2\pi}{\text{angular frequency}}
$$

전체 시퀀스에 대해 막전위의 평균 절대 크기(mean absolute magnitude)를 계산하고 이를 그려서 각 시험 주파수 신호에 대한 뉴런의 응답을 얻었다. 동일한 절차를 HRF 뉴런에 대해서도 수행하였다.

그림 8. $\delta = 0.001$ 일 때, 예시적인 각주파수 $\omega$ 와 $b$-오프셋 $b'$ 조합에 대한 HRF 뉴런 주파수 응답 플롯.

## A.5 발산 경계(Divergence Boundary)

이산 시간 간격 $\delta$ 는 발산 거동에 큰 영향을 미친다. 아래 그림은 기본값 $\delta = 0.01$ 보다 큰 경우와 작은 경우의 발산 경계 곡선을 보여준다. $\omega$ 와 $b_c$ 의 조합이 선 아래에 있으면 막전위는 수렴한다. 각주파수 $\omega$ 를 0부터 100 rad/s 범위로 볼 때, 지속 진동 곡선(sustained oscillation curve)의 기울기는 $\delta$ 가 작아질수록 더 완만해진다. 즉, 진동이 더 정밀하게 모델링될수록 시스템이 발산할 가능성은 줄어든다. 실제로, 더 작은 시간 간격으로 계산할수록 연속 미분방정식에 대한 이산 근사(discrete approximation)는 더 정밀해진다.

그림 9. $\delta \in \{0.002, \cdots, 0.02\}$ 에 따른 발산 경계.

## A.6 실험 설정(Experimental Setup)

입력 차원 $m$, 은닉 BRF 또는 BHRF 뉴런 $h$ 개, 그리고 누설 적분기(leaky integrator, LI) 출력 뉴런 $C$ 개를 갖는 완전 순환 RF-RSNN은 PyTorch(Paszke et al., 2017)로 구현되었다. 미니배치 크기 $B$ 를 사용하는 시간 역전파(BPTT) 알고리즘을 사용하였고, 네트워크와 과제에 따라 Adam(Kingma & Ba, 2014), RAdam(Liu et al., 2020), RMSprop(Hinton et al., 2012)을 사용하였다.

현재 시퀀스 단계의 입력 $x_t \in \mathbb{M}^{B \times m}$ 와 이전 시점의 순환 은닉 뉴런 출력 스파이킹 $z_{t-1} \in \mathbb{M}^{B \times h}$ 는 완전 연결 선형층(fully connected linear layer)을 구현함으로써 결합되었다. 이렇게 결합된 신호 $x_t \in \mathbb{M}^{B \times h}$ 는 익숙한 주입 전류 $I(t)$ 에 대응하지만, 미니배치 내 모든 은닉 뉴런을 표현하기 위해 행렬 형태로 표기되었다. 알고리즘에는 명시적으로 나타나지 않지만, $w_{in,rec} \in \mathbb{M}^{h \times (m+h)}$ 와 $w_{out} \in \mathbb{M}^{C \times h}$ 는 각각 입력-은닉(input-to-hidden), 은닉-은닉(hidden-to-hidden), 은닉-출력(hidden-to-output) 뉴런 사이의 연결 강도(strength of the connections)를 나타내며, 이들은 최적화되었다. 네트워크에서는 어떠한 바이어스도 사용하거나 학습하지 않았다. $\omega$ 와 $b' \in \mathbb{R}^h$ 를 갖는 B(H)RF 뉴런은 $B$ 개 데이터 샘플의 시간 시퀀스에 대해 $u^t$, $(v^t)$, $b^t$, $q^t$, $\vartheta^t \in \mathbb{M}^{B \times h}$ 와 함께 갱신되었다. LI 막전위 시간 감쇠 상수 $\tau_{m,out} \in \mathbb{R}^C_{>0}$ 또한 학습되었다. 우리의 기준 함수(criterion)로는 시뮬레이션에 따라 음의 로그우도(negative log-likelihood, NLL) 또는 교차 엔트로피(cross-entropy, CE) 손실을 사용하였다.

ALIF 구현의 경우, 적응형 문턱값 시간 감쇠 상수 $\tau_a \in \mathbb{R}^h_{>0}$ 가 막전위 시간 감쇠 상수 $\tau_m \in \mathbb{R}^h_{>0}$ 와 함께 추가로 학습된다.

평균 시퀀스 손실(average sequence loss)과 NLL을 사용하는 경우, 로그 소프트맥스(logarithmic softmax) 함수는 모든 출력 뉴런에 대해 시점별로 계산되었다.

$$
\hat{y}^t = \log(\mathrm{softmax}(u^{out}_t))
$$

이는 다음 손실 함수에 전달된다.

$$
L^t = \mathrm{criterion}(\hat{y}^t, y^t)
= \frac{1}{B}\sum_{c=1}^{C} - y_c^t \hat{y}_c^t
$$

여기서 $y^t \in \mathbb{M}^{B \times C}$ 는 시점 $t$ 의 목표 레이블(target label)을 원-핫 벡터(one-hot vectors)로 나타낸 것이다. 평균 손실은 시퀀스 길이 $T$ 에 대해 다음과 같이 계산된다.

$$
\mathcal{L} = \frac{1}{T}\sum_{t=1}^{T} L^t
$$

반면 label-last loss의 경우에는 마지막 시점에서의 손실만 역전파된다.

BPTT 알고리즘의 backward pass는 PyTorch의 자동 미분 엔진(automatic differentiation engine)으로 내부적으로 계산되었다(Paszke et al., 2017). B(H)RF 및 ALIF 뉴런에서 비미분 가능한 Heaviside 함수에 대해서는 multi-Gaussian 함수를 대리 기울기로 수동 구현하였다(Neftci et al., 2019; Yin et al., 2021).

식 (39)

$$
\frac{\partial \Theta}{\partial u}
=_{\mathrm{def}}
(1 + h)g(u, \vartheta, \sigma) - 2h\,g(u, \vartheta, s\sigma)
$$

여기서

식 (40)

$$
g(x, \mu, \sigma) = e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}
$$

이며, $h = 0.15$, $s = 6$, $\sigma = 0.5$ 로 설정하였다. 음의 기울기 값(negative gradient values)은 dying-ReLU 문제를 방지하는 누설 rectified linear unit 및 exponential linear unit에 의해 동기부여되었다(Lu et al., 2019). 이러한 값에서 multi-Gaussian 대리 기울기 함수는 초기 막전위가 낮더라도 뉴런이 스파이크를 내도록 유도하였고, 이는 효과적인 학습에 기여하였다.

가장 좋은 모델(best model)은 검증 손실(validation loss)에 대해 조기 종료(early stopping)를 수행했을 때, 처음 다섯 번의 실행에서 평균 테스트 손실(average test loss)이 가장 높았던 모델로 저장하였다. 학습, 검증, 테스트 세트의 손실과 정확도는 학습 패턴을 분석하기 위해 Tensorboard(Abadi et al., 2016)에 기록되었다. 최적 모델을 얻은 뒤에는, 다섯 개 저장 모델 전체에 대해 출력 스파이크의 총합 $z_{sum}$ 을 데이터 샘플 수 $N$ 으로 나누어 SOPs를 계산하였다.

식 (41)

$$
SOPs = \frac{z_{sum}}{N}
$$

결과는 기준선 ALIF-RSNN과 비교하였다.

모델 시뮬레이션과 실험 수행에는 NVIDIA GeForce RTX 2060, NVIDIA GeForce RTX 2080 Ti, NVIDIA GeForce RTX 3090, NVIDIA A100을 포함한 여러 딥러닝 가속기(deep learning accelerators)를 갖춘 시스템을 사용하였으며, PyTorch 2.0.1, Python 3.10.4, CUDA 11.7 환경에서 실행하였다.

## A.7 ECG-QT 데이터베이스 전처리(ECG-QT database preprocessing)

비교 가능한 결과를 얻기 위해 전처리된 QT 데이터는 Yin et al. (2021)로부터 가져왔다. 원래 시퀀스는 각 구간이 1300 ms의 기록을 포함하도록 더 작은 간격(intervals)으로 분할되었다. 이어서 두 ECG 신호는 정규화되었고, 레벨 크로스 인코딩(level-cross encoding)을 통해 두 개의 분리된 스파이크 열로 부호화되었다. 임계값 $L = 0.3$ 을 기준으로, 임계값보다 큰 양의 기울기와 작은 음의 기울기는 각각 다음과 같은 스파이크를 유도한다.

식 (42)

$$
s_+ =
\begin{cases}
1 & \text{if } x_t - x_{t-1} \ge L \\
0 & \text{otherwise}
\end{cases}
$$

식 (43)

$$
s_- =
\begin{cases}
1 & \text{if } x_t - x_{t-1} \le -L \\
0 & \text{otherwise}
\end{cases}
$$

목표 레이블(target labels)도 이에 맞추어 동일하게 분할되어 각 시점에서 예측되었다. 그림 4의 색상 blue, red, green, cyan, olive, purple는 각각 P, PQ, QR, RS, ST 라벨에 대응한다. 학습에는 557개 구간(segment), 검증에는 61개, 테스트에는 141개를 사용하였다.

## A.8 불응기와 부드러운 리셋의 효과(Effect of refractory period and smooth reset)

불응기(RP)와 부드러운 리셋(SmR)은 모두 스파이크 연산을 줄이는 효과를 내지만, 그 영향은 데이터셋에 따라 다르다. BRF 뉴런의 SOP는 일관되게 가장 낮은 값을 보인다. 이는 BRF 뉴런이 RF 뉴런에 비해 다양한 유형의 데이터를 더 효율적으로 학습할 수 있는 유연성을 가짐을 시사한다. ECG에서는 RP로 최적화한 경우가 SmR보다 더 적게 발화하고, SHD에서는 오히려 SmR이 더 많이 발화한다. 그러나 두 경우 모두 RP와 SmR을 함께 사용한 완전한 BRF가 가장 효율적이다. PS-MNIST는 리셋이 없을 때(no reset), RP만 있을 때, SmR만 있을 때, 그리고 RP와 SmR의 조합 모두에서 발산한다. 따라서 이러한 변형들은 유의하게 높은 SOP를 보인다. 이는 또한 발산 경계(DB)가 가져오는 안정성을 부각한다. RP만 사용해 학습한 S-MNIST는 불안정한 성능과 SOP를 보인다.

그림 10. 발산 경계(divergence boundary, DB), 불응기(refractory period, RP), 부드러운 리셋(smooth reset, SmR)을 적용한 RF 네트워크의 SOP 결과. 5회 실행 평균 SOP를 표준편차와 함께 도시하였다. DB, RP, SmR을 모두 적용한 경우가 완전한 BRF 뉴런을 의미한다.

## A.9 리셋 변형과 성능(Reset variation and performance)

표준 리셋 메커니즘(부록 A.1에 기술됨)이 공진자 네트워크(resonator network)에 미치는 영향을 탐색하기 위해, 부드러운 리셋(smooth), 소프트 리셋(soft), 하드 리셋(hard)을 적용한 RF 모델을 ECG와 SHD에 대해 최적화하였고, 그 결과는 그림 11에 제시하였다. 전통적인 리셋 메커니즘을 S-MNIST와 PS-MNIST에 적용한 경우에는 수렴에 실패했다는 점에 유의하라. 그림은 소프트 및 하드 리셋이 성능 저하와 더 느린 수렴을 유발함을 보여주는데, 이는 진동의 위상(phase)이 변형되고 연속적인 공진 스파이킹(continuous resonant spiking)이 어려워지기 때문으로 추정된다. 스파이크가 원래 신호의 주기의 두 배에 해당하는 시점에서 발생할 수도 있기 때문이다. 반면 리셋이 없는 결과에서는 수렴 양상이 BRF 뉴런과 거의 비슷한 수준인데, 이는 위상 자체가 보존되기 때문이다. 본문에서 언급했듯, 특히 ECG와 SHD에서는 리셋이 없는 RF 모델이 발산하지 않았는데, 이는 작은 각주파수가 시간에 따른 오차 누적(error accumulation)을 줄여주기 때문이다.

그림 11. RF 리셋 메커니즘에 따른 ECG 및 SHD의 수렴 비교.

## A.10 잡음 강건성(Noise robustness)

사후 학습된 모델의 잡음 강건성은 Stromatias et al. (2015)와 Park et al. (2021)이 사용한 방법으로 추가 조사하였다. SNN의 하드웨어 구현은 제한된 정밀도(restricted precision)나 신호 잡음 때문에 성능에 제약을 받는다. 여기서는 네 가지 서로 다른 제약 및 잡음 유형을 탐색한다. 양자화(quantization)는 표현의 비트 정밀도(bit precision)를 줄인다(그림 12 좌상). 입력 신호에 대한 잡음은 표준편차를 점진적으로 증가시키는 가우시안 잡음(Gaussian noise)으로 시뮬레이션하였다(그림 12 우상). 스파이크 삭제(spike deletion)(Park et al., 2021)의 경우, 각 시퀀스 단계에서 일정 비율의 스파이크를 네트워크에서 제거한다(그림 12 좌하). 시냅스 잡음(synaptic noise)은 가중치에 가우시안 잡음을 추가하는 방식이다.

그림 12. 저장된 최적 성능 모델에 대해 수행한 S-MNIST 잡음 강건성 연구. 실선은 성능(performance), 점선은 각 네트워크의 스파이크 연산(SOP)을 나타낸다.

전반적으로, 리셋이 없는 바닐라 RF 네트워크와 BRF 네트워크는 다양한 잡음에 직면했을 때 ALIF 네트워크보다 더 나은 성능을 보였다. 특히 입력에 도입된 잡음의 경우 RF와 BRF 네트워크는 성능을 유지한 반면, ALIF 네트워크는 작은 표준편차의 잡음만으로도 이미 실패하였다. BRF 네트워크는 RF나 ALIF 네트워크에 비해 매우 높은 스파이크 희소성을 가지면서도 이를 달성했다. RF 네트워크는 BRF나 ALIF보다 스파이크 삭제에 대해 더 강건한데, 이는 RF 네트워크의 중복 스파이킹(redundant spiking)의 효과일 수 있다. 이는 스파이크 삭제 강건성과 높은 스파이크 희소성 사이의 절충(trade-off)을 보여준다.

## A.11 오류 지형(Error landscape)

BRF-RSNN의 오류 지형(error landscape)은 매끄럽고 볼록(convex)-유사한 구조를 가지는 반면, RF-RSNN, 특히 ALIF-RSNN의 지형은 좁은 골짜기(narrow valley)를 가진 거친 표면을 보인다. 이러한 매끄러운 지형은 높은 일반화(generalization)와 직관적인 최적화(straightforward optimization)를 나타내며, 이것이 빠른 수렴을 설명한다.

우리는 또한 발산 경계를 구현하면 하나의 이산 시간 단계에 대해 계산된 막 상태 전이 사상(membrane state transition mapping)의 스펙트럴 반경(spectral radius)이 1 이하가 됨을 발견했다(Higuchi et al., 2024). 이는 기울기를 효과적으로 안정화하고 그 크기를 보존함으로써, 발산 경계가 빠르고 매끄러운 학습에 기여함을 시사한다.

그림 13. Higuchi et al. (2024)에서 가져온 S-MNIST 데이터셋의 RF, BRF, ALIF 네트워크에 대한 오류 지형 플롯. 위: 오류 표면 플롯(error surface plots). $x$ 축과 $y$ 축은 각각 $\alpha$ 와 $\beta$, 즉 파라미터 편차(parameter deviations)에 대응하고, $z$ 축은 $f(\alpha,\beta)$, 즉 오류(error)에 대응한다(Li et al., 2018). 아래: 오류 등고선 플롯(error contour plots). 시각화를 강화하기 위해 각 도표의 값 범위(value range)와 그에 대응하는 색상(coloring)은 서로 다르다는 점에 유의하라.

## A.12 하이퍼파라미터(Hyperparameters)

표 3. label-last loss(S-MNIST) 및 평균 시퀀스 손실(PS-MNIST, ECG, SHD), 그리고 프루닝(pruning)에서 TBPTT에 대해 사용한 truncation step 50 조건에서, 최적 성능을 보인 BRF, BHRF, ALIF 모델에 적용된 하이퍼파라미터. BRF- 및 RF-BPTT 모델에는 동일한 하이퍼파라미터를 적용했으며, RF-RSNN에서는 $b = b'$ 로 두었다.

### 표 3-a. S-MNIST

| 항목 | BRF-RSNN BPTT | BRF-RSNN TBPTT | BHRF-RSNN BPTT | ALIF-RSNN BPTT | ALIF-RSNN TBPTT |
|---|---|---|---|---|---|
| Network | 1 + 256 (fully recurrent) + 10 | 1 + 256 (fully recurrent) + 10 | 1 + 256 (fully recurrent) + 10 | 1 + 256 (fully recurrent) + 10 | 1 + 256 (fully recurrent) + 10 |
| Learning rate (Lr) | 0.1 | 0.006 | 0.1 | 0.001 | 0.001 |
| Loss function | NLL | NLL | CE | NLL | NLL |
| Minibatch size | 256 | 256 | 256 | 256 | 256 |
| Lr scheduling | LinearLR | LinearLR | LinearLR | LinearLR | LinearLR |
| Optimizer | Adam | Adam | RAdam | Adam | Adam |
| Epochs | 300 | 300 | 400 | 300 | 300 |
| Parameter initialization | $\omega : U(15, 50)$, $b' : U(0.1, 1)$, $\tau_{m,out} : N(20, 5)$ | $\omega : U(15, 50)$, $b' : U(0.1, 1)$, $\tau_{m,out} : N(20, 5)$ | $\omega : U(15, 35)$, $b' : U(0.1, 1)$, $\alpha : \mathrm{Sigmoid}(N(0, 0.1))$ | $\tau_m : N(20, 5)$, $\tau_a : N(200, 50)$, $\tau_{m,out} : N(20, 5)$ | $\tau_m : N(20, 5)$, $\tau_a : N(200, 50)$, $\tau_{m,out} : N(20, 5)$ |

### 표 3-b. PS-MNIST

| 항목 | BRF-RSNN BPTT | BRF-RSNN TBPTT | BHRF-RSNN BPTT | ALIF-RSNN BPTT | ALIF-RSNN TBPTT |
|---|---|---|---|---|---|
| Network | 1 + 256 (fully recurrent) + 10 | 1 + 256 (fully recurrent) + 10 | 1 + 256 (fully recurrent) + 10 | 1 + 256 (fully recurrent) + 10 | 1 + 256 (fully recurrent) + 10 |
| Learning rate (Lr) | 0.1 | 0.006 | 0.1 | 0.001 | 0.001 |
| Loss function | NLL | NLL | NLL | NLL | NLL |
| Minibatch size | 256 | 256 | 256 | 256 | 256 |
| Lr scheduling | LinearLR | LinearLR | LinearLR | LinearLR | LinearLR |
| Optimizer | Adam | Adam | RAdam | Adam | Adam |
| Epochs | 300 | 300 | 200 | 300 | 300 |
| Parameter initialization | $\omega : U(15, 85)$, $b' : U(0.1, 1)$, $\tau_{m,out} : N(20, 1)$ | $\omega : U(15, 85)$, $b' : U(0.1, 1)$, $\tau_{m,out} : N(20, 1)$ | $\omega : U(10, 50)$, $b' : U(1, 6)$, $\alpha : \mathrm{Sigmoid}(N(0, 0.1))$ | $\tau_m : N(20, 5)$, $\tau_a : N(200, 50)$, $\tau_{m,out} : N(20, 5)$ | $\tau_m : N(20, 5)$, $\tau_a : N(200, 50)$, $\tau_{m,out} : N(20, 5)$ |

### 표 3-c. ECG-QT

| 항목 | BRF-RSNN | BHRF-RSNN | ALIF-RSNN |
|---|---|---|---|
| Network | 4 + 36 (fully recurrent) + 6 | 4 + 36 (fully recurrent) + 6 | 4 + 36 (fully recurrent) + 6 |
| Learning rate (Lr) | 0.1 | 0.3 | 0.05 |
| Loss function | NLL | NLL | NLL |
| Minibatch size | 16 | 4 | 64 |
| Lr scheduling | LinearLR | LinearLR | LinearLR |
| Optimizer | Adam | RAdam | Adam |
| Epochs | 400 | 300 | 400 |
| Parameter initialization | $\omega : U(3, 5)$, $b' : U(0.1, 1)$, $\tau_{m,out} : N(20, 1)$ | $\omega : U(7, 11)$, $b' : U(0.1, 1)$, $\alpha : \mathrm{Sigmoid}(N(0, 0.1))$ | $\tau_m : N(20, 0.5)$, $\tau_a : N(7, 0.2)$, $\tau_{m,out} : N(20, 0.5)$ |
| Sub-seq. length | 0 | 0 | 10 |

### 표 3-d. SHD

| 항목 | BRF-RSNN | BHRF-RSNN | ALIF-RSNN |
|---|---|---|---|
| Network | 700 + 128 (fully recurrent) + 20 | 700 + 128 (fully recurrent) + 20 | 700 + 128 (fully recurrent) + 20 |
| Learning rate (Lr) | 0.075 | 0.1 | 0.075 |
| Loss function | NLL | NLL | NLL |
| Minibatch size | 32 | 32 | 32 |
| Lr scheduling | LinearLR | LinearLR | LinearLR |
| Optimizer | Adam | RMSprop | Adam |
| Epochs | 20 | 20 | 20 |
| Parameter initialization | $\omega : U(5, 10)$, $b' : U(2, 3)$, $\tau_{m,out} : N(20, 5)$ | $\omega : U(3, 10)$, $b' : U(0.1, 1)$, $\alpha : \mathrm{Sigmoid}(N(0, 0.1))$ | $\tau_m : N(20, 5)$, $\tau_a : N(150, 10)$, $\tau_{m,out} : N(20, 5)$ |
| Sub-seq. length | 0 | 0 | 10 |

BHRF-RSNN은 아직 예비 탐색 단계(preliminary phase of exploration)에 있었기 때문에, 우리는 누설 적분기 감쇠 상수(leaky integrator decay constant) $\alpha$ 에 대해 초기에는 sigmoid 함수를 사용하였다. 이는 감쇠가 0과 1 사이 범위에 있도록 보장해 주기 때문이다.

## A.13 순방향 기울기 주입(FGI)과 TorchScript를 이용한 학습 속도 향상(Learning speed-up)

추가 탐색으로서, 우리는 대리 기울기 함수를 모델링하기 위해 Otte(2024)의 순방향 기울기 주입(forward gradient injection, FGI)을 적용하였다. $x$ 를 계산 노드(computation node), $f$ 를 비미분 함수(non-differentiable function; 예를 들어 계단 함수(step function)), $g'$ 를 $f$ 의 도함수로 사용하고자 하는 함수, 그리고 $sg$ 를 기울기 정지(stop gradient) 연산자라고 하자. `backward()` 메서드를 재정의하는 대신, 우리는 다음과 같이 쓸 수 있다.

식 (44)

$$
h = x \cdot sg(g'(x))
$$

식 (45)

$$
y = h - sg(h) + sg(f(x))
$$

FGI와 TorchScript를 함께 사용하면, 표 4에서 보듯 모든 데이터셋에서 2배가 넘는 속도 향상을 달성하면서도 성능을 유지할 수 있다.

표 4. 표준 `backward()` 기준선 대비, 모델 최적화 방법에 따른 BRF-RSNN 학습 속도 향상.

| Optimization | Gradient | S-MNIST Time (hrs) | Ratio | PS-MNIST Time (hrs) | Ratio | ECG Time (hrs) | Ratio | SHD Time (min) | Ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| - | backward() | 23.8 | 1× | 31.0 | 1× | 8.5 | 1× | 40.9 | 1× |
| TorchScript | backward() | 11.3 | 2.1× | 13.8 | 2.2× | 4.1 | 2.1× | 16.4 | 2.5× |
| TorchScript | FGI | 10.2 | 2.3× | 12.8 | 2.4× | 3.5 | 2.4× | 14.0 | 2.9× |
