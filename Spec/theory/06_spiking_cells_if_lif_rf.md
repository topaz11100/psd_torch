# 06 Spiking Cells: IF, LIF, RF

## 1. 공통 관점

이 프로젝트의 학습·추론 실행 단위는 clocked sequence index `t` 이다. 따라서 구현과 분석의 1차 기준은 연속시간 ODE가 아니라 다음과 같은 **이산시간 recurrence** 이다.

```text
state[t+1] = F(state[t], input[t+1])
spike[t+1] = surrogate_step(readout(state[t+1]) - threshold)
```

연속시간 뉴런식은 문헌적 배경이나 parameterization prior 로만 사용한다. 실제 PSD/FFT/filter 통계는 저장된 discrete trace와 discrete pole parameter 에서 계산한다.

## 2. IF

IF cell은 누적기다.

\[
U_t = U_{t-1}+I_t
\]

분석 가능한 주요 통계는 threshold, spike rate, membrane trace PSD 이다.

## 3. LIF

LIF cell은 1차 이산 저역통과 pole을 갖는다.

\[
U_t = \alpha U_{t-1}+I_t
\]

여기서 `alpha` 는 sample 단위 pole radius/decay factor 이다. 이 값이 1에 가까울수록 기억 길이가 길고, 0에 가까울수록 현재 입력에 더 가까운 응답을 만든다.

## 4. RF

RF cell은 직접 이산 복소 pole로 정의한다.

\[
z_t=x_t+j y_t,
\qquad
z_{t+1}=a z_t + I_{t+1},
\qquad
a=\rho e^{j\phi}.
\]

실수 구현은 오일러 공식 \(e^{j\phi}=\cos\phi+j\sin\phi\) 를 사용한 2차원 회전-스케일 행렬이다.

\[
\begin{bmatrix}x_{t+1}\\y_{t+1}\end{bmatrix}
=
\rho
\begin{bmatrix}
\cos\phi & -\sin\phi\\
\sin\phi & \cos\phi
\end{bmatrix}
\begin{bmatrix}x_t\\y_t\end{bmatrix}
+
\begin{bmatrix}1\\0\end{bmatrix}I_{t+1}.
\]

- \(\rho=|a|\): sample 단위 감쇠/증폭 비율
- \(\phi=\arg(a)\): rad/sample 단위 공명 주파수
- \(f=\phi/(2\pi)\): cycles/sample 단위 중심 주파수

`rf_pole_radius_constrained=true` 이면 \(0\le\rho<\rho_{max}<1\) 로 제한한다. 이 경우 linear subthreshold RF는 안정한 causal IIR filter 로 해석된다. `false` 이면 \(\rho\) 는 positive softplus parameter 이며 1을 넘을 수 있다. 이 경우 모델은 finite-horizon reset-bounded resonant amplifier 로 사용할 수 있으나, 안정 필터 해석에서는 제외해야 한다.

## 5. spike/reset은 필터 해석과 분리

RF의 discrete pole 분석은 spike/reset 이전의 linear subthreshold dynamics에 대한 분석이다. threshold, surrogate spike, soft/hard reset이 추가되면 전체 시스템은 비선형이 된다. 따라서 `filter_stats_vectors()`는 pole radius, pole angle, frequency response, stability margin/excess 를 기록하고, PSD 분석은 별도로 실제 membrane/spike trace 에 대해 수행한다.
