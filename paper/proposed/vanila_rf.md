# Vanila Resonate-and-Fire (RF) 명세서

## 0. 목적

이 문서는 현재 프로젝트가 사용하는 baseline RF neuron 의 구현 규칙과 해석 규칙을 정의한다. 배경이 되는 RF / BRF 계열 논문은 `paper/Balanced Resonate-and-Fire Neurons.md` 와 RF lineage 문헌에 두되, 현재 공식 baseline 자체는 paper 원문을 그대로 감싼 wrapper 가 아니라 프로젝트 표준 exact-ZOH RF 경로다.

문서 분리는 아래와 같다.

- 본 문서: vanilla RF 동역학, spike 결정 규칙, 초기화, 저장 해석량
- `paper/proposed/vanila_scenario.md` : clip / structure / structclip 시나리오
- `paper/proposed/filter_analysis.md` : `rho`, `f_cyc_per_sample` 통계와 histogram 저장 규칙
- `paper/proposed/readout.md` : output layer readout 규칙

## 1. 설계 원칙

vanilla RF layer 에서 학습되는 파라미터는 두 부류다.

1. 레이어 연결 가중치
2. 각 neuron 내부의 RF 동역학 파라미터

즉 이 프로젝트는 고정 random projection 이 아니라 learned dense coupling 을 사용하면서도, RF neuron 내부 자유도는 감쇠와 공명 주파수에 집중시킨다. 다만 입력 결합에는 bias 를 두지 않으며, bias 는 학습 파라미터에도 포함되지 않는다.

RF 계열의 직접 해석량은 raw parameter 자체가 아니라 아래 두 값이다.

$$
\rho_i = e^{b_i \Delta t}
$$

$$
f_{\mathrm{cyc/sample},i} = \frac{\omega_i \Delta t}{2\pi}
$$

여기서 $f_{\mathrm{cyc/sample}} \in [0, 0.5]$ 이며, $0.5$ 는 Nyquist 에 대응한다. filter 통계 저장은 raw `b`, raw `omega` 대신 위 직접 해석량을 기준으로 한다.

## 2. RF 동역학

### 2.1 연속시간 식

뉴런 $i$ 의 상태를 $x_i(t), y_i(t)$ 로 두고 입력 전류를 $I_i(t)$ 라 하면, 연속시간 RF neuron 은

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

### 2.2 spike 결정 변수와 출력 spike

이 프로젝트에서 막전위 signal family 라고 부르는 것은 스파이크 생성함수 $O$ 로 실제로 들어가 spike 여부를 결정하는 변수다. RF 에서는 exact ZOH step 이후의 soma-side 상태 $x_{i,t}^{\mathrm{pre}}$ 가 spike 결정을 만든다. threshold shift 를 포함한 spike 결정 변수와 출력 spike 는

$$
m_{i,t} = x_{i,t}^{\mathrm{pre}} - \theta
$$

$$
o_{i,t} = O(m_{i,t})
$$

로 둔다.

따라서 membrane signal family 는 $O$ 직전의 spike 결정 변수 의미론을 가지며, spike signal family 는 뉴런의 출력 spike signal $o_{i,t}$ 다.

### 2.3 reset 정책

현재 프로젝트의 기본값은 해석 우선 실험을 위한 `no_reset` 이다.

$$
(x_{i,t}^{\mathrm{post}}, y_{i,t}^{\mathrm{post}}) = (x_{i,t}^{\mathrm{pre}}, y_{i,t}^{\mathrm{pre}})
$$

동시에 원전 계열과의 비교를 위해 `soft_reset` 도 남긴다.

$$
x_{i,t}^{\mathrm{post}} = x_{i,t}^{\mathrm{pre}} - \theta o_{i,t}, \qquad y_{i,t}^{\mathrm{post}} = y_{i,t}^{\mathrm{pre}}
$$

즉 reset 정책 choice 는 실험 설정이고, 본 프로젝트의 권장 baseline 은 `no_reset` 이다.

## 3. 입력 결합과 학습 파라미터

입력층 또는 이전 레이어의 활성 $u_t$ 가 들어오면, RF layer 로 들어가는 전류는 dense linear transform 으로 만든다.

$$
I_t = W u_t
$$

즉 vanilla RF baseline 에서는 bias term 을 두지 않는다.

`recurrent=True` 인 경우에는 이전 시점의 output spike 에 대한 project-side recurrent projection 이 current term 에 더해진다. 이 recurrent adapter 는 intrinsic RF state update 자체를 바꾸지 않고, 입력 전류를 만드는 경로에만 추가된다.

한 RF dense layer 에서 학습되는 항목은 아래와 같다.

- dense weight $W$
- 각 뉴런의 intrinsic parameter $b_i$, $\omega_i$

bias 는 사용하지 않으며 학습 파라미터에도 포함되지 않는다.

반대로 threshold, surrogate shape hyperparameter, reset policy choice 자체는 고정 실험 설정이다. output layer 도 같은 base RF neuron 을 사용하며, output neuron 뒤에 learned NN head 를 두지 않는다.

## 4. 이산시간 구현 규칙

### 4.1 exact ZOH 만 사용

현재 프로젝트의 vanilla RF 는 Euler branch 를 두지 않고 exact zero-order hold 만 사용한다. 입력 $I_{i,t}$ 가 한 step 동안 상수라고 두면

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
x_{i,t+1}^{\mathrm{pre}} \\
y_{i,t+1}^{\mathrm{pre}}
\end{bmatrix}
=
\rho_i
\begin{bmatrix}
\cos \phi_i & -\sin \phi_i \\
\sin \phi_i & \cos \phi_i
\end{bmatrix}
\begin{bmatrix}
x_{i,t}^{\mathrm{post}} \\
y_{i,t}^{\mathrm{post}}
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

## 5. 동역학 학습 파라미터 초기화

vanilla / clip RF 의 동역학 학습 파라미터 초기화는 fan-based Kaiming / Xavier 가 아니라 유효 물리 파라미터 구간에서의 bounded uniform sampling 을 사용한다.

free RF 에서는 아래 두 값을 각각 독립적으로 bounded uniform sampling 한다.

$$
f_i \in (0, 0.5)
$$

$$
|b_i| \in (0.1, 1.0)
$$

실제 damping 은

$$
b_i = -|b_i|
$$

로 음수 영역에 둔다.

clip RF 에서는 위 bounded-uniform 규칙을 유지하되, resonance frequency $f_i$ 의 support 만 각 neuron 이 배정받은 `w_clip_edges` interval 로 좁힌다. 현재 public RF clip CLI 는 frequency 만 제한하므로 damping magnitude 초기화는 free bounded-uniform 정책을 유지한다. 경계 끝점이 정확히 0 또는 0.5 인 경우에도 raw unconstrained parameter 가 무한대로 가지 않도록 구현에서는 작은 trim epsilon 을 둔 열린구간에서 샘플링할 수 있다.

## 6. 시나리오와 저장 링크

- `rf`, `rf_struct`, `rf_clip`, `rf_structclip` 의 그룹 구성과 `tear` 규칙은 `paper/proposed/vanila_scenario.md` 를 따른다.
- RF filter 통계와 histogram 저장은 `paper/proposed/filter_analysis.md` 를 따른다.
- output layer readout 정의는 `paper/proposed/readout.md` 를 따른다.
- output layer 는 same-base-neuron output layer 이며 추가 NN head 를 두지 않는다.

## 7. 구현 대응

현재 코드 기준 대응은 아래와 같다.

- `src/neurons/RF_neuron.py` : exact ZOH, `rho()`, `f_cyc_per_sample()`, `no_reset` / `soft_reset`
- `src/neurons/vanila_rf.py` : RF dense layer compatibility alias
- `src/util/psd_analysis_driver.py` : RF / LIF PSD variant builder, grouped block 저장, metadata 기록
