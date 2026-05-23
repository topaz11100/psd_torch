# 동역학 통계 이론

## 1. 목적

동역학 통계는 spectral artifact와 별개로 cell parameter와 내부 상태의 분포를 기록한다. 목적은 “왜 특정 layer의 PSD가 그렇게 생겼는가”를 parameter 관점에서 보조 설명하는 것이다.

## 2. 파라미터 벡터

모델 layer $\ell$의 parameter family $p$에 대해 벡터

$$
\theta_{\ell,p}\in\mathbb{R}^{d_\ell}
$$

를 수집한다. 예시는 다음이다.

- LIF membrane decay alpha.
- RF resonant frequency.
- RF damping magnitude.
- RF decay radius.
- threshold.

각 벡터는 값만이 아니라 다음 metadata를 갖는다.

```text
layer_index, layer_name, parameter_name, role, unit,
shape, trainable, lower_bound, upper_bound, group_ids,
scenario, constraint_hash
```

## 3. bounds와 group metadata

clip scenario에서 lower/upper bound는 parameter의 해석에 필수다. 예를 들어 같은 alpha 값이라도 어떤 bound interval 안에서 학습되었는지에 따라 의미가 달라진다.

clipstructure에서는 group id도 함께 필요하다.

$$
\theta_i \in [l_{g_i},u_{g_i}]
$$

## 4. 내부 상태 통계

internal state statistics는 spike rate, membrane distribution, RF real/imag state 같은 runtime trace summary를 다룬다. 현재 phase에서는 parameter vector collector와 최소 internal state stats interface를 유지한다.

## 5. spectral 분석과의 관계

동역학 통계는 PSD를 대체하지 않는다. PSD는 출력 신호의 시간 구조를 설명하고, dynamics는 그 구조를 만든 cell parameter와 constraint를 설명한다. 두 artifact는 run/checkpoint/layer metadata로 결합된다.
