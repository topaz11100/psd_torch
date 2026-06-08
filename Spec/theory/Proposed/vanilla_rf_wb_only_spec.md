# Vanilla RF — Direct Discrete Pole Specification

## 1. 현재 프로젝트의 정식 정의

현재 vanilla RF는 연속시간 \(b,\omega\) 모델이 아니라 직접 이산 복소 pole 모델이다.

\[
z_{t+1}=a z_t + I_{t+1},
\qquad
a=\rho e^{j\phi}.
\]

여기서 \(z=x+jy\) 이며 실제 구현은 실수 상태 \((x,y)\) 두 개를 사용한다.

\[
x_{t+1}=\rho(\cos\phi\,x_t-\sin\phi\,y_t)+I_{t+1}
\]

\[
y_{t+1}=\rho(\sin\phi\,x_t+\cos\phi\,y_t)
\]

이 방식은 복소 pole을 직접 학습하는 discrete IIR resonator 이다. ZOH/Euler는 더 이상 모델의 정식 구현 구분이 아니다.

## 2. 학습 파라미터

- `pole_radius_raw` → \(\rho\)
- `pole_angle_raw` → \(\phi\)
- `input_weight` → layer input projection
- optional `recurrent_weight`
- optional trainable threshold

기존 `filter` 인수는 RF에서 center frequency 고정값으로 해석한다. 즉 `filter=0.25` 는 \(f=0.25\) cycles/sample, \(\phi=2\pi f\) 로 고정한다.

## 3. pole radius constraint

새 인수:

```yaml
model_training:
  model:
    discrete_dynamics:
      rf_pole_radius_constrained: true
      rf_pole_radius_max: 0.9999
```

`rf_pole_radius_constrained=true`:

\[
\rho=\rho_{max}\sigma(r),\qquad 0<\rho<\rho_{max}<1.
\]

`rf_pole_radius_constrained=false`:

\[
\rho=\operatorname{softplus}(r),\qquad \rho>0.
\]

두 번째 모드는 \(\rho>1\) 을 허용한다. 이 경우 unstable pole 이 아니라 finite-horizon resonant amplifier 로 해석해야 한다. 학습 안정성을 위해 gradient clipping, reset 정책, `stability_excess` monitoring 이 필요하다.

## 4. 분석 통계

`filter_stats_vectors()`는 다음을 출력한다.

- `pole_radius = rho`
- `pole_angle = phi`
- `center_frequency = phi/(2*pi)`
- `pole_real = rho*cos(phi)`
- `pole_imag = rho*sin(phi)`
- `stability_margin = 1-rho`
- `stability_excess = max(rho-1,0)`
- `threshold`

`damping` 은 backward-compatible alias 로 남지만, 현재 의미는 continuous damping 이 아니라 sample-domain log damping `-log(rho)` 이다. 직접 해석에서는 `pole_radius` 와 `pole_angle` 을 우선 사용한다.

## 5. 문헌식과의 관계

연속 RF ODE는 역사적 motivation 으로 남길 수 있다. 그러나 이 프로젝트의 학습/추론/분석 주장은 다음 문장으로 고정한다.

> The trained RF neuron is a discrete-time complex IIR resonator.  Its memory and frequency selectivity are characterized by the learned discrete pole `a`, where `|a|` is the per-sample decay/amplification factor and `arg(a)` is the resonant frequency in radians per sample.
