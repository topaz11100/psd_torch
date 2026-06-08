# Filter Property Analysis — Direct Discrete Contract

## 1. 분석 대상

필터 분석은 spike/reset 이후의 전체 비선형 SNN이 아니라, 각 뉴런 내부의 **linear subthreshold path** 를 대상으로 한다. 저장된 trace PSD와 달리, filter-property analysis 는 학습된 parameter 만으로 계산된다.

## 2. 공통 주파수 축

정규화 주파수는 cycles/sample 로 둔다.

\[
f_i=\frac{i}{2(F-1)},\quad i=0,\ldots,F-1
\]

각주파수는

\[
\Omega_i=2\pi f_i\in[0,\pi]
\]

이며 response는 unit circle \(z=e^{j\Omega_i}\) 에서 평가한다.

## 3. EMA branch 계열

`my_DH_SNN`, `my_R_DH_SNN` 의 branch는 discrete EMA pole 이다.

\[
H_d(z)=\frac{1-\alpha_d}{1-\alpha_d z^{-1}},
\qquad 0<\alpha_d<1.
\]

- `branch_pole_radius = alpha`
- `soma_pole_radius = beta`
- `alpha`, `beta` 는 backward-compatible 이름으로 유지한다.

## 4. direct discrete RF branch

`my_D_RF` 와 vanilla RF는 복소 pole을 직접 사용한다.

\[
a=\rho e^{j\phi}
\]

\[
H_d(z)=\frac{R_d}{1-a_d z^{-1}}.
\]

branch mixture는

\[
H_n(z)=\frac{1}{s_n}\sum_d M_{n,d}(s_n)H_{n,d}(z)
\]

로 계산한다.

vanilla RF는 single-branch case 로 볼 수 있으며, 입력 gain은 layer input projection weight에 흡수된다. 따라서 filter statistics 는 `pole_radius`, `pole_angle`, `center_frequency`, `stability_margin`, `stability_excess` 를 우선 기록한다.

## 5. 안정성 해석

- \(\rho<1\): stable causal IIR filter
- \(\rho=1\): marginal oscillator
- \(\rho>1\): finite-horizon resonant amplifier

필터 주파수 응답을 엄밀한 stable LTI response 로 해석하려면 \(\rho<1\) 이어야 한다. 그러나 학습 run 자체에서 \(\rho>1\) 을 금지할지는 configuration policy 로 결정한다.

## 6. 저장 통계

공통 response 요약:

- `f_peak`
- `f_low_3db`
- `f_high_3db`
- `bw_3db`
- `dc_ratio`
- `nyquist_ratio`
- `filter_class_code`

pole/branch 요약:

- `pole_radius`, `pole_angle`, `pole_real`, `pole_imag`
- `branch_pole_radius`, `soma_pole_radius`
- `sample_time_constant`
- `stability_margin`, `stability_excess`
- `s_value`, `active_branch_count`, `branch_utilization`
