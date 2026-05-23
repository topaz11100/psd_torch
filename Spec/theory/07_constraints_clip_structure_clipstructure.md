# clip / structure / clipstructure scenario

## 1. 목적

scenario는 MLP topology에 적용되는 실험 조건이다. cell kind나 reset mode가 아니다. 같은 topology와 cell을 두고 연결 구조 또는 cell parameter 범위를 제한해 spectral behavior가 어떻게 달라지는지 관찰한다.

현재 canonical scenario는 네 가지다.

```text
none
clip
structure
clipstructure
```

## 2. structure

structure는 hidden neuron을 group으로 나누고 같은 group 사이의 연결만 허용하는 mask를 만든다. target neuron $i$, source neuron $j$의 group을 각각 $g_i$, $g_j$라 하면 feedforward mask는

$$
M_{ij}^{ff}=\begin{cases}
1, & g_i=g_j \\
0, & g_i\ne g_j
\end{cases}
$$

이다. recurrent hidden layer에서는

$$
M_{ij}^{rec}=\begin{cases}
1, & g_i=g_j \\
0, & g_i\ne g_j
\end{cases}
$$

를 recurrent weight에 곱한다.

중요한 점은 recurrent current가 여전히 직전 spike에서만 온다는 것이다.

$$
I_t^{rec}=R S_{t-1}
$$

structure는 $R$의 effective mask를 바꿀 뿐, membrane이나 decision을 recurrent source로 바꾸지 않는다.

## 3. clip

clip은 cell parameter의 허용 범위를 제한한다. optimizer 후 clamp가 아니라 bounded parameterization을 사용한다.

LIF alpha:

$$
\alpha = l + (u-l)\sigma(a)
$$

RF frequency:

$$
f = l_f + (u_f-l_f)\sigma(a_f)
$$

RF damping:

$$
\gamma = l_\gamma + (u_\gamma-l_\gamma)\sigma(a_\gamma)
$$

threshold도 trainable이면 같은 방식으로 제한할 수 있다.

## 4. cell별 허용 clip 대상

- IF: threshold bounds만 허용.
- LIF: alpha와 threshold bounds 허용.
- RF: frequency, damping, threshold bounds 허용.

잘못된 조합은 validation error다. 예를 들어 LIF에 RF frequency bounds를 주거나 RF에 LIF alpha bounds를 주면 안 된다.

## 5. clipstructure

clipstructure는 group structure와 bounds를 동시에 적용한다. group별 bounds가 있으면 group id를 feature별 lower/upper tensor로 확장한다.

$$
l_i=l_{g_i}, \quad u_i=u_{g_i}
$$

이 경우 dynamics parameter vector에는 group_ids, lower_bound, upper_bound가 함께 남는다.

## 6. output layer 정책

현재 phase에서는 hidden layer constraint를 기본 지원 범위로 둔다. output layer에 constraint를 적용하는 것은 별도 정책이 필요하므로 현재는 의도적 미지원 또는 validator error로 처리한다.
