# Integrated my_* Neurons and Branch Statistics

## 1. 통합 관점

`my_*` 계열은 모두 직접 이산시간 모델이다. 학습과 분석의 기준은 sequence index `t` 이며, 연속시간 ODE 또는 ZOH/Euler 이산화는 정식 구현 명세가 아니다.

| 모델 | branch dynamics | 주요 통계 |
|---|---|---|
| `my_DH_SNN` | discrete EMA branch \(i_d[t]=\alpha_d i_d[t-1]+(1-\alpha_d)I_d[t]\) | `branch_pole_radius`, `soma_pole_radius`, branch count |
| `my_R_DH_SNN` | discrete EMA branch + learned branch mixing | EMA pole, positive/negative mix count, branch count |
| `my_D_RF` | complex discrete IIR branch \(z_d[t+1]=\rho_de^{j\phi_d}z_d[t]+R_dI[t+1]\) | `pole_radius`, `pole_angle`, input gain, branch count |

## 2. branch-count 통계

모든 `my_*` 모델은 사전 할당된 최대 branch 수 `D` 와 뉴런별 soft count \(s_n\) 를 사용한다.

\[
M_{n,d}(s_n)\in[0,1]
\]

분석 키:

- `s_value`
- `active_branch_count`
- `branch_mask_mass`
- `branch_utilization`
- `branch_mass_fraction`

이 통계는 neuron-level raw distribution, layer-level snapshot, model-level aggregate 에 모두 저장된다.

## 3. EMA 계열 통계

`my_DH_SNN`과 `my_R_DH_SNN`의 branch는 sample-domain EMA pole이다.

\[
H_d(z)=\frac{1-\alpha_d}{1-\alpha_d z^{-1}}.
\]

분석 키:

- `alpha`, `branch_pole_radius`
- `beta`, `soma_pole_radius`
- `f_peak`, `bw_3db`, `dc_ratio`, `nyquist_ratio`
- `positive_mix_weight_count`, `negative_mix_weight_count` for `my_R_DH_SNN`

## 4. D-RF 계열 통계

`my_D_RF`는 branch별 direct discrete pole을 갖는다.

\[
a_{n,d}=\rho_{n,d}e^{j\phi_{n,d}}.
\]

\[
H_{n,d}(z)=\frac{R_{n,d}}{1-a_{n,d}z^{-1}}.
\]

분석 키:

- `pole_radius`, `pole_angle`
- `pole_real`, `pole_imag`
- `input_gain_real`, `input_gain_imag`
- `center_frequency = pole_angle/(2*pi)`
- `sample_time_constant`
- `stability_margin`, `stability_excess`
- `pole_radius_mean/std`, `pole_angle_mean/std`

`tau`, `omega`, `damping` 이름은 backward-compatible alias 로만 유지한다. 새 문서와 새 실험에서는 `pole_radius`, `pole_angle` 을 우선 사용한다.

## 5. vanilla RF와의 정합성

vanilla RF도 같은 direct discrete pole contract 를 사용한다. vanilla RF는 single-branch resonator 로 볼 수 있고, `my_D_RF`는 branch bank mixture 로 볼 수 있다.

vanilla RF에서만 `rf_pole_radius_constrained` 인수로 \(\rho<1\) 안정 제약을 끄고 켤 수 있다. `my_D_RF`는 기본적으로 안정 branch bank 로 초기화되며, 필요하면 layer constructor 수준에서 별도 실험을 확장한다.

## 6. 해석 원칙

- stable filter claim: \(\rho<1\) 또는 EMA \(0<\alpha<1\) 필요
- finite-horizon amplifier claim: \(\rho>1\) 허용 가능하나 `stability_excess` 기록 필요
- spike/reset 포함 전체 시스템: nonlinear system 이므로 pole transfer function 과 분리해서 설명

## 7. soma-local threshold/reset 계약

`my_*` 계열은 vanilla IF/LIF/RF와 같은 threshold/reset 설정을 사용한다.

```yaml
model:
  reset: soft | hard | none
  v_th: [fixed | train, initial_value]
```

이 계약은 **세포체(soma)에만 적용**된다. `my_DH_SNN`과 `my_R_DH_SNN`에서는 branch EMA 상태 `d_state`가 reset되지 않고, 세포체 막전위 `mem`만 soft/hard/no-reset 정책을 따른다. `my_D_RF`에서는 branch resonator 상태 `(u,v)`가 reset되지 않고, branch 합산으로 얻은 soma drive와 adaptive-threshold history만 soma-local reset 정책의 영향을 받는다.

세부 의미는 다음과 같다.

| 설정 | soma 동작 | branch 상태 |
|---|---|---|
| `soft` | spike 후 threshold를 subtract | 변경 없음 |
| `hard` | spike 후 soma state를 zero | 변경 없음 |
| `none` / `no_reset` | soma reset 및 reset-history update 비활성화 | 변경 없음 |

`v_th: ["fixed", x]`는 고정 threshold buffer를 만들고, `v_th: ["train", x]`는 `softplus(raw)+eps`로 양수인 per-soma threshold parameter를 만든다. 따라서 trainable threshold 수는 layer의 soma neuron 수와 같고 branch 수와 무관하다.


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


### Membrane-only output readout and effective trainability

When the selected readout is membrane-only, the output neuron layer is configured with `emit_spike=false`.  In that case, output-layer threshold and adaptive-threshold-kernel parameters have no path to the supervised loss.  The implementation therefore keeps the user-facing threshold/reset policy for hidden soma dynamics, but freezes spike-only output-layer parameters under membrane-only readouts.  This is an execution-contract detail; the subthreshold membrane trajectory and the membrane logits are unchanged.
