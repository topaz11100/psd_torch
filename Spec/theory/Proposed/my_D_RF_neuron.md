# my_D_RF_neuron — Direct-Discrete Multi-Branch D-RF

## 1. 설계 전환

`my_D_RF`는 더 이상 연속 RF ODE의 ZOH/Euler 이산화로 정의하지 않는다. 구현, 학습, 분석의 기준은 처음부터 이산시간 복소 IIR branch 이다.

\[
z_{n,d}[t+1]=a_{n,d}z_{n,d}[t]+R_{n,d}I_n[t+1],
\qquad
a_{n,d}=\rho_{n,d}e^{j\phi_{n,d}}.
\]

- \(n\): 출력 뉴런 index
- \(d\): 사전 할당된 branch index
- \(z=u+jv\): branch state
- \(\rho\): per-sample pole radius
- \(\phi\): pole angle in rad/sample
- \(R=R_x+jR_y\): discrete input gain

복소 dtype 대신 실수 state 두 개를 사용한다.

\[
u_{t+1}=\rho(\cos\phi\,u_t-\sin\phi\,v_t)+R_x I_{t+1}
\]

\[
v_{t+1}=\rho(\sin\phi\,u_t+\cos\phi\,v_t)+R_y I_{t+1}
\]

이 식은 복소 곱을 실수 2D 회전행렬로 정확히 전개한 것이며 근사 이산화가 아니다.

## 2. branch 구조

사전 할당 branch 수는 `branch=D` 로 고정한다. 각 뉴런은 연속 branch-count parameter \(s_n\in[S_{min},S_{max}]\) 를 갖고, soft/hard prefix mask \(M_{n,d}(s_n)\) 로 활성 branch를 선택한다.

\[
H_n[t]=\frac{1}{s_n}\sum_{d=1}^{D}M_{n,d}(s_n)\,u_{n,d}[t].
\]

`soft_mask_epochs`, `ste_epochs`, `harden_epoch` 은 branch count 학습 schedule만 제어한다. 이것은 vanilla `scenario_mode=structure` 와 별개다.

## 3. soma threshold/reset

`my_D_RF`는 vanilla RF와 같은 soma threshold 설정을 사용한다.

```yaml
v_th: [fixed | train, initial_value]
reset: soft | hard | none
```

여기서 threshold는 branch별 값이 아니라 뉴런별 세포체 baseline threshold \(V_{base,n}\) 이다. `fixed`이면 buffer로 유지하고, `train`이면 `softplus(raw)+eps` 형태의 양수 trainable parameter 로 둔다.

branch resonator 상태 \(u_{n,d},v_{n,d}\) 는 reset하지 않는다. reset은 branch 합산 이후의 soma path 에만 적용된다.

| reset mode | 의미 |
|---|---|
| `soft` | spike 이후 soma post-reset 기록값에서 firing threshold를 subtract 하고 adaptive-threshold history를 갱신한다. |
| `hard` | spike 이후 soma post-reset 기록값을 0으로 만든다. branch resonator는 유지한다. |
| `none` | soma reset과 adaptive-threshold history 갱신을 비활성화한다. |

## 4. adaptive threshold

pre-indicator 기반 adaptive threshold를 유지한다.

\[
p_n[t]=\Theta(H_n[t]-V_{base,n})
\]

\[
V_{th,n}[t]=V_{base,n}+\sum_{k=1}^{K}a_kp_n[t-k],
\qquad a_k>0.
\]

출력 spike는

\[
S_n[t]=\Theta(H_n[t]-V_{th,n}[t])
\]

로 정의하며 학습 시 surrogate gradient 를 사용한다.

## 5. 안정성과 증폭

기본 구현은 \(\rho<1\) 로 초기화되고, `my_D_RF` layer 내부 기본값도 안정 pole 쪽으로 둔다. 안정 필터 해석에서는 \(\rho<1\) 이어야 impulse response가 수렴한다.

다만 SNN 전체는 finite horizon + threshold/reset 비선형계를 사용하므로 \(\rho>1\) 이 수학적으로 불가능한 것은 아니다. 이 경우 `stability_excess=max(rho-1,0)` 를 반드시 확인해야 하며, 해석은 stable filter 가 아니라 reset-bounded resonant amplifier 로 적어야 한다.

## 6. 필터 전달함수

branch 하나의 subthreshold transfer function은

\[
H_{n,d}(z)=\frac{R_{n,d}}{1-a_{n,d}z^{-1}}.
\]

soma 입력까지의 전달함수는

\[
H_n(z)=\frac{1}{s_n}\sum_{d=1}^{D}M_{n,d}(s_n)\frac{R_{n,d}}{1-a_{n,d}z^{-1}}.
\]

주파수 응답은 \(z=e^{j\Omega}\), \(\Omega\in[0,\pi]\) 에서 평가한다. `center_frequency` 는 \(\phi/(2\pi)\) cycles/sample 로 기록한다.

## 7. 기록 통계

`filter_stats_vectors()`는 다음 축을 기록한다.

- `pole_radius`, `pole_angle`, `pole_real`, `pole_imag`
- `center_frequency`, `sample_time_constant`
- `input_gain_real`, `input_gain_imag`
- `stability_margin=1-rho`, `stability_excess=max(rho-1,0)`
- `s_value`, `active_branch_count`, `branch_mask_mass`, `branch_utilization`
- `f_peak`, `f_low_3db`, `f_high_3db`, `bw_3db`, `dc_ratio`, `nyquist_ratio`, `filter_class_code`

기존 이름 `tau`, `omega`, `damping` 은 backward-compatible alias 로 남긴다. 여기서 `tau` 는 sample-domain memory constant, `omega` 는 pole angle, `damping` 은 sample-domain log damping `-log(rho)` 이며, 연속시간 ODE parameter 가 아니다.


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

