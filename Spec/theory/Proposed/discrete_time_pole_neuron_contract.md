# Direct Discrete Pole Neuron Contract

## 1. 원칙

디지털 SNN에서 구현, 학습, 추론은 모두 finite sequence index 에서 수행된다. 따라서 RF 및 proposed resonant branch 의 정식 분석 단위는 연속시간 pole 이 아니라 이산시간 pole 이다.

\[
z_{t+1}=a z_t+R I_{t+1}
\]

\[
a=\rho e^{j\phi}
\]

- \(\rho=|a|\): per-step memory/decay/amplification
- \(\phi=\arg(a)\): resonant frequency in rad/sample
- \(R\): discrete input gain

## 2. 실수 구현

복소 dtype 없이도 다음 실수 recurrence 로 완전히 동일하게 구현된다.

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
\begin{bmatrix}R_x\\R_y\end{bmatrix}I_{t+1}.
\]

이것은 Euler discretization 이 아니라 Euler formula \(e^{j\phi}=\cos\phi+j\sin\phi\) 의 정확한 실수 전개다.

## 3. 적용 모델

| 모델 | 이산 pole | 설명 |
|---|---:|---|
| vanilla RF | \(\rho e^{j\phi}\) | single complex IIR resonator |
| CNN vanilla RF | \(\rho e^{j\phi}\) | conv input coupling + channel-wise RF pole |
| `my_D_RF` | \(\rho_{n,d}e^{j\phi_{n,d}}\) | branch-wise complex IIR bank |
| `my_DH_SNN` | \(\alpha_{n,d}\) | branch EMA pole |
| `my_R_DH_SNN` | \(\alpha_{n,d}\) | branch EMA pole + learned soma branch mixing |

## 4. 안정성 policy

\[
\rho<1
\]

이면 stable causal IIR filter 다. \(\rho>1\) 은 즉시 무한대가 아니라 finite horizon 에서 지수 증폭하는 amplifier 이다. 이 프로젝트는 vanilla RF 에 대해 `rf_pole_radius_constrained` 인수로 두 해석을 모두 허용한다.

## 5. time step 해석

프로젝트 내부 기본 단위는 sample index 이며 \(\Delta t=1\) 로 둔다. 실제 Hz 변환이 필요한 경우에만 데이터의 bin size 를 별도 metadata 로 사용한다.

\[
f_{Hz}=\frac{\phi}{2\pi\Delta t}.
\]

동일 데이터셋 안에서 binning 이 고정되어 있으면 `center_frequency=phi/(2*pi)` cycles/sample 만으로 충분하다.


## Soma-local reset and threshold contract

The proposed `my_*` families now use the same soma-facing reset and threshold contract as the project vanilla neurons.  The contract is deliberately applied **only to the soma**.  Dendritic EMA states, resonant branch states, branch masks, branch-count variables, and branch filter parameters are not reset by soma spikes.

For a soma pre-state `u_pre[t]` and a positive threshold vector `theta`, the spike is

```math
s[t] = H(u_pre[t] - theta).
```

The reset policy is then

```math
soft:\quad u[t] = u_pre[t] - theta s[t],
```

```math
hard:\quad u[t] = u_pre[t](1-s[t]),
```

```math
none:\quad u[t] = u_pre[t].
```

`v_th: ["fixed", value]` stores `theta` as a non-trainable soma buffer.  `v_th: ["train", value]` creates one positive trainable threshold per soma neuron using a softplus parameterization.  Filter statistics export both `threshold` and `v_threshold`, plus the tensor flags `soma_reset_enabled`, `soma_hard_reset`, and `soma_trainable_threshold`.

Implementation note: `my_DH_SNN` and `my_R_DH_SNN` compute a soma pre-state with their learned soma pole, spike from that pre-state, then store the post-reset soma state for the next step. `my_D_RF` uses the dendritic resonator sum as the soma pre-state and records a soma post-reset state, while leaving the resonator states untouched.

