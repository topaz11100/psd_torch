# IF/LIF/RF cell 동역학

## 1. 문서의 대상

MLP는 topology이고 IF/LIF/RF는 cell이다. topology는 layer 연결과 recurrent current를 정의하고, cell은 time state update를 정의한다. 같은 MLP skeleton 안에서 cell kind만 바꿔 비교할 수 있어야 한다.

## 2. 공통 표기

hidden layer $\ell$의 input current를

$$
I_t^{(\ell)}
$$

라 둔다. spike decision은 대체로 threshold를 뺀 값

$$
D_t = U_t^{-}-\theta
$$

이고 spike는

$$
S_t=H(D_t)
$$

이다. 학습에서는 surrogate gradient를 쓸 수 있지만 이론 정의는 hard threshold로 표현한다.

## 3. IF cell

IF는 leak이 없다.

$$
U_t^{-}=U_{t-1}^{+}+I_t
$$

$$
D_t=U_t^{-}-\theta
$$

$$
S_t=H(D_t)
$$

reset은 mode에 따라 다르다.

hard reset:

$$
U_t^{+}=(1-S_t)U_t^{-}+S_tU_{reset}
$$

soft reset:

$$
U_t^{+}=U_t^{-}-\theta S_t
$$

none:

$$
U_t^{+}=U_t^{-}
$$

IF의 clip 대상은 threshold뿐이다. decay나 resonance parameter가 없기 때문이다.

## 4. LIF cell

LIF는 membrane decay를 갖는다.

$$
U_t^{-}=\alpha U_{t-1}^{+}+I_t
$$

여기서 $0<\alpha<1$이다. clip bounds가 있을 때는 raw parameter $a$로부터

$$
\alpha = \alpha_{low}+(\alpha_{high}-\alpha_{low})\sigma(a)
$$

로 만든다. optimizer step 뒤 clamp하는 방식이 아니라 parameterization 자체가 범위를 보장한다.

## 5. RF cell

RF는 real/imag state를 갖는 2차원 resonant state다.

$$
z_t = r_t + j q_t
$$

연속계는 감쇠와 회전을 갖는 resonator로 볼 수 있다. discrete update는 Euler 근사가 아니라 closed-form exact ZOH를 사용한다. damping magnitude $\gamma>0$, angular frequency $\omega$에 대해 decay radius는

$$
\rho = \exp(-\gamma\Delta t)
$$

이고 rotation은

$$
R(\omega\Delta t)=
\begin{bmatrix}
\cos(\omega\Delta t) & -\sin(\omega\Delta t) \\
\sin(\omega\Delta t) & \cos(\omega\Delta t)
\end{bmatrix}
$$

이다. 기본 상태 update는

$$
\begin{bmatrix}r_t^{-}\\q_t^{-}\end{bmatrix}
=
\rho R(\omega\Delta t)
\begin{bmatrix}r_{t-1}^{+}\\q_{t-1}^{+}\end{bmatrix}
+
B(\omega,\gamma,\Delta t)I_t
$$

형태다. spike decision은 real state 기준이다.

$$
D_t=r_t^{-}-\theta
$$

RF frequency와 damping도 bounds가 있으면 bounded sigmoid로 만든다.

$$
f=f_{low}+(f_{high}-f_{low})\sigma(a_f)
$$

$$
\gamma=\gamma_{low}+(\gamma_{high}-\gamma_{low})\sigma(a_\gamma)
$$

## 6. reset mode

RF는 state가 2차원이므로 reset mode가 IF/LIF보다 세분화된다. `hard_state`, `hard_real`, `soft_real`, `scale_state`, `threshold_only`, `none`이 현재 정책이다. reset 정책은 trace에 real/imag pre/post가 남아야 검증 가능하다.

## 7. 분석 파라미터

동역학 분석은 다음 벡터를 수집한다.

- IF: threshold.
- LIF: membrane decay alpha, threshold, optional tau.
- RF: resonant frequency, damping magnitude, decay radius, threshold.

각 벡터는 trainable 여부, unit, bounds, group_ids를 metadata로 가져야 한다.
