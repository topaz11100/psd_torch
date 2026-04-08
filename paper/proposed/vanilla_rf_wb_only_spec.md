# Vanilla Resonate-and-Fire (RF) 뉴런 구현/이론 명세서

## 0. 문서 목적

이 문서는 현재 프로젝트에서 사용하는 vanilla Resonate-and-Fire (RF) 뉴런의 구현 규칙과 해석 규칙을 정의한다. 핵심 목적은 두 가지다.

1. dense layer weight 를 포함한 실제 학습 가능한 RF 분류기를 명확히 규정한다.
2. 학습 후 RF 뉴런의 공명 특성을 필터 통계로 해석할 수 있게 한다.

이 문서에서 말하는 **intrinsic trainable parameter** 는 각 RF 뉴런 내부의 $\omega_i$, $b_i$ 이다. 다만 이것이 layer connection weight 를 학습하지 않는다는 뜻은 아니다. 레이어 간 연결 가중치와 bias 는 일반적인 dense parameter 로 학습한다.

---

## 1. 설계 원칙

### 1.1 해석 대상

RF 뉴런의 선형 서브스레시홀드 동역학은 감쇠 진동계로 해석된다. 따라서 학습이 끝난 뒤 뉴런 내부 파라미터를 적절한 필터 통계로 변환하면, 뉴런 집단이 어떤 주파수 대역에 민감하게 조직되었는지 분석할 수 있다.

### 1.2 학습되는 파라미터의 의미

이 프로젝트에서 학습되는 파라미터는 아래 두 부류다.

- 레이어 연결 가중치와 bias: dense 하게 학습되는 일반적인 네트워크 parameter
- 뉴런 내부 파라미터: 각 RF 뉴런의 $\omega_i$, $b_i$

즉, 의도는 "layer weight 는 학습한다" 와 "intrinsic neuron parameter 는 $\omega, b$ 두 개로 제한한다" 를 동시에 만족하는 것이다.

### 1.3 해석 시 기록하는 통계량

원시 파라미터 $\omega$, $b$ 를 그대로 저장하지 않고, 직접 해석되는 양으로 바꿔 저장한다.

$$
\rho_i = e^{b_i \Delta t}
$$

$$
f_{\mathrm{cyc/sample},i} = \frac{\omega_i \Delta t}{2\pi}
$$

여기서 $f_{\mathrm{cyc/sample}} \in [0, 0.5]$ 이며, $0.5$ 는 Nyquist 에 대응한다.

---

## 2. RF 동역학 정의

### 2.1 연속시간 식

뉴런 $i$ 의 상태를 $x_i(t), y_i(t)$ 로 두고 입력 전류를 $I_i(t)$ 라 하면, 연속시간 RF 뉴런은

$$
\dot{x}_i = b_i x_i - \omega_i y_i + I_i(t)
$$

$$
\dot{y}_i = \omega_i x_i + b_i y_i
$$

로 둔다.

복소수 상태 $z_i = x_i + j y_i$ 를 쓰면

$$
\dot{z}_i = (b_i + j\omega_i) z_i + I_i(t)
$$

이다. 여기서 $\omega_i > 0$ 는 고유 각주파수, $b_i < 0$ 는 감쇠 계수다.

### 2.2 스파이크 정의

스파이크는 membrane state $x_i$ 의 threshold crossing 으로 정의한다.

$$
s_i(t) = H(x_i(t) - \theta)
$$

여기서 $H(\cdot)$ 는 Heaviside 함수이고, $\theta$ 는 고정 임계값이다.

### 2.3 reset 정책

고전적인 RF / Izhikevich 계열 모델은 reset 을 포함하는 경우가 많다. Izhikevich 의 2001 RF 원전과 2003 simple spiking neuron 계열은 모두 spike 후 상태를 재설정하는 규칙을 포함한다. 즉, reset 이 원전의 자연스러운 출발점이다.

다만 최근 sequence modeling 및 long-horizon SNN/RF 문헌에서는 reset 이 내부 상태 정보를 끊고 병렬화나 장기 의존성 유지에 불리하다는 이유로, no-reset 또는 decoupled-reset 류 설계를 쓰는 경우가 늘었다. PRF 는 reset 이 병렬 학습을 어렵게 만든다고 보고 reset decoupling 을 제안하고, 최근 reset 메커니즘 재검토 논문도 reset 이 정보 손실과 장기 의존성 약화를 일으킬 수 있다고 정리한다. 따라서 no-reset 이 field-wide mainstream 이라서가 아니라, 최근 특정 목적의 구현에서 실용적 선택지로 자주 쓰이는 것으로 이해해야 한다.

현재 프로젝트의 기본값은 해석 우선 실험을 위한 `no_reset` 이다.

$$
(x_i, y_i) \leftarrow (x_i, y_i)
$$

이 설정은 서브스레시홀드 공명 동역학을 가장 직접적으로 보존한다. 동시에 원전 계열과의 연결을 위해 `soft_reset` 도 ablation 으로 유지한다.

$$
x_i \leftarrow x_i - \theta s_i
$$

즉, 이 프로젝트는 "reset 이 원전 쪽에 가깝고, no-reset 은 최근 해석/시퀀스 모델링 목적에서 채택되는 기본 실험 설정" 으로 문서화한다.

---

## 3. 아키텍처와 입력 결합

### 3.1 dense learned coupling

입력층 또는 이전 레이어의 활성 $u_t$ 가 들어오면, RF layer 로 들어가는 전류는 dense affine transform 으로 만든다.

$$
I_t = W u_t + c
$$

여기서 $W$ 와 $c$ 는 일반적인 학습 파라미터다. 따라서 이 프로젝트는 **고정 random projection** 을 기본 구조로 사용하지 않는다.

구조 분리 실험에서 mask 가 적용되더라도, 허용된 연결 내부에서는 여전히 dense weight 가 학습된다. 다시 말해, mask 는 연결 가능성의 구조를 제한할 뿐, 허용된 edge 는 모두 trainable 하다.

### 3.2 학습 파라미터 정리

한 RF dense layer 에서 학습되는 항목은 아래와 같다.

- dense weight $W$
- dense bias $c$
- 각 뉴런의 intrinsic parameter $\omega_i$, $b_i$

반대로 threshold, surrogate shape hyperparameter, reset policy choice 자체는 고정 실험 설정이다.

---

## 4. 이산시간 구현 규칙

### 4.1 exact ZOH 만 사용

현재 프로젝트의 vanilla RF 는 Euler branch 를 두지 않고 exact zero-order hold (ZOH) 만 사용한다.

입력 $I_{i,t}$ 가 한 step 동안 상수라고 두면

$$
z_{i,t+1} = \alpha_i z_{i,t} + \beta_i I_{i,t}
$$

$$
\alpha_i = e^{(b_i + j\omega_i)\Delta t} = \rho_i e^{j\phi_i}
$$

$$
\beta_i = \frac{e^{(b_i + j\omega_i)\Delta t} - 1}{b_i + j\omega_i}
$$

이고,

$$
\rho_i = e^{b_i \Delta t}, \qquad \phi_i = \omega_i \Delta t
$$

이다.

실수 상태로 쓰면

$$
\begin{bmatrix}
x_{i,t+1} \\
y_{i,t+1}
\end{bmatrix}
=
\rho_i
\begin{bmatrix}
\cos \phi_i & -\sin \phi_i \\
\sin \phi_i & \cos \phi_i
\end{bmatrix}
\begin{bmatrix}
x_{i,t} \\
y_{i,t}
\end{bmatrix}
+
\begin{bmatrix}
\beta_{x,i} \\
\beta_{y,i}
\end{bmatrix} I_{i,t}
$$

로 구현한다.

### 4.2 안정성 해석

ZOH 구현에서는 step 감쇠가 직접

$$
|\alpha_i| = \rho_i = e^{b_i \Delta t}
$$

로 해석된다. 따라서 $b_i < 0$ 이면 $\rho_i < 1$ 이고, 이산시간 구현에서도 안정하다.

### 4.3 clip 단위

RF clip 경계는 raw $\omega$ 가 아니라 Nyquist 상한이 $0.5$ 인 normalized frequency 단위로 입력한다.

$$
f_i \in [0, 0.5]
$$

$$
\omega_i = \frac{2\pi f_i}{\Delta t}
$$

clip model 은 CLI 에서 받은 $f$ 경계를 내부적으로 $\omega$ 로 변환한 뒤 적용한다. 저장과 통계는 다시 $f_{\mathrm{cyc/sample}}$ 로 남긴다.

### 4.4 동역학 학습 파라미터 초기화

vanilla / clip RF 의 동역학 학습 파라미터 초기화는 fan-based Kaiming/Xavier 가 아니라 **유효 물리 파라미터 구간에서의 bounded uniform sampling** 을 사용한다.

free RF 에서는

$$
f_i \in (0, 0.5)
$$

에서 normalized resonance frequency 를 bounded uniform sampling 으로 초기화하고, damping magnitude $|b_i|$ 도 별도의 기본 bounded range 안에서 bounded uniform sampling 으로 초기화한다. 실제 damping 은

$$
b_i = -|b_i|
$$

로 음수 영역에 둔다.

clip RF 에서는 위와 같은 bounded-uniform 규칙을 유지하되, resonance frequency $f_i$ 의 support 만 각 neuron 이 배정받은 clip interval 로 좁힌다. 현재 public RF clip CLI 는 frequency 만 제한하므로 damping magnitude 초기화는 free bounded-uniform 정책을 유지한다.

---

## 5. 실험 로그와 필터 통계

### 5.1 저장 항목

RF layer 는 매 epoch 마다 아래 필터 통계를 저장한다.

- layer 단위 bar plot
- model 단위 bar plot
- `summary_stats.csv`
- `all_layers_summary.csv`

### 5.2 CSV 에 포함되는 통계량

각 parameter 에 대해 최소 아래 열을 기록한다.

- `count`
- `mean`
- `variance`
- `std`
- `min`
- `q25`
- `q50`
- `q75`
- `max`

RF 에서는 parameter 이름을 `rho`, `f_cyc_per_sample` 로 통일한다. LIF 계열이면 `alpha` 를 사용한다.

### 5.3 bar plot 해석

bar plot 은 원시 뉴런 배열을 직접 나열하는 그림이 아니라, 집계 통계량을 요약하는 그림이다. 현재 프로젝트에서는 각 parameter 에 대해 평균, 분산, 사분위수를 한 그림에 저장한다.

---

## 6. 구현 대응 관계

현재 코드 기준 대응은 아래와 같다.

- `src/neurons/RF_neuron.py`
  - dense weight 와 bias 는 `fc.weight`, `fc.bias`
  - intrinsic parameter 는 `raw_b`, `raw_omega` 를 통해 $b$, $\omega$ 로 변환
  - 저장 통계는 `rho()` 와 `f_cyc_per_sample()`
- `src/common/psd_analysis_driver.py`
  - epoch 단위 RF / LIF 필터 통계 저장
  - `summary_stats.csv`, `all_layers_summary.csv` 생성
- `bash/run_psd.sh`, `bash/psd.sh`
  - RF reset mode, clip frequency, PSD 저장 규칙을 현재 CLI 와 맞춰 노출하고, `run_psd.sh` 는 병렬 scenario launcher, `psd.sh` 는 직접 실행하지 않는 내부 config template 으로 유지

---

## 7. 프로젝트 권장 설정

현재 프로젝트의 권장 기본값은 아래와 같다.

| 항목 | 기본값 | 설명 |
|---|---:|---|
| integration | exact ZOH | Euler 제거 |
| reset | no_reset | 해석 우선 기본값 |
| dense connection weight | learned | layer coupling 은 학습 |
| intrinsic RF parameter | $\omega, b$ | 뉴런 내부 자유도 |
| saved RF stats | `rho`, `f_cyc_per_sample` | 직접 해석량 저장 |
| frequency unit | cycle/sample | Nyquist upper bound = 0.5 |

---

## 8. 참고 문헌

1. Eugene M. Izhikevich, *Resonate-and-Fire Neurons*, Neural Networks, 2001.
2. Eugene M. Izhikevich, *Simple Model of Spiking Neurons*, IEEE Transactions on Neural Networks, 2003.
3. Jibin Wu et al., *PRF: Parallel Resonate and Fire Neuron for Long Sequence Learning in Spiking Neural Networks*, 2024.
4. *Revisiting Reset Mechanisms in Spiking Neural Networks for Long Sequence Processing*, 2025.
