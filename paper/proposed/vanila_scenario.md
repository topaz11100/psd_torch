# Vanila RF / LIF 시나리오 명세서

## 0. 목적

이 문서는 vanilla RF / LIF 계열의 clip / structure / structclip 시나리오를 정의한다. 동역학 자체는 `paper/proposed/vanila_rf.md`, `paper/proposed/vanila_lif.md` 를 따르고, 본 문서는 variant token, 그룹 구성, `tear` 규칙, mask / clip 적용 위치, PSD 저장 구조를 모듈식으로 정리한다.

## 1. variant token 과 허용 suffix

공식 variant token 은 아래와 같다.

| token | base neuron | structure mask | clipped intrinsic parameter |
| --- | --- | --- | --- |
| `rf` | RF | no | no |
| `rf_struct` | RF | yes | no |
| `rf_clip` | RF | no | yes |
| `rf_structclip` | RF | yes | yes |
| `lif` | LIF | no | no |
| `lif_struct` | LIF | yes | no |
| `lif_clip` | LIF | no | yes |
| `lif_structclip` | LIF | yes | yes |

이들 variant 는 `_R` recurrent suffix 를 받을 수 있다. 다만 PSD clip / structure variant 는 `_R_<int>` branch-count suffix 를 허용하지 않는다.

## 2. 그룹 구성 규칙

grouped variant 에서는 hidden layer 마다 group id 배열을 만든다. hidden layer 의 폭을 $N_\ell$, group 수를 $G$ 라 하면 neuron $i$ 의 group id 를

$$
g_\ell(i) \in \{0,1,\dots,G-1\}
$$

로 둔다.

`band_neuron_ends` 는 hidden layer 마다 하나의 문자열 entry 를 가지며, 각 entry 는 cumulative group end index 를 쉼표로 적는다. group 수가 $G$ 이면 각 hidden layer entry 는 정확히 $G-1$ 개의 cumulative end index 를 가져야 한다.

group 수 결정 규칙은 아래와 같다.

- structure-only variant 에서 `band_neuron_ends` 가 없으면 기본 `num_groups = 2` 로 본다.
- clip variant 에서는 `num_groups = len(edges) - 1` 이다.
- grouped variant 인데 `band_neuron_ends` 가 없으면, 각 hidden layer 폭에 대해 균등 분할에 가까운 default cumulative ends 를 자동 생성한다.

`tear` 는 제약이 시작되는 destination hidden-layer index 를 뜻하는 1-based 정수다. 항상

$$
1 \le \text{tear} \le \text{num\_hidden\_layers}
$$

를 만족해야 한다.

## 3. structure 시나리오

structure mask 는 hidden-to-hidden projection 에만 적용한다. destination hidden-layer index 를 $\ell$ 라 두면, $\ell \ge 2$ 인 hidden-to-hidden projection 의 mask 는

$$
M_\ell(i,j) = \mathbf{1}\{g_\ell(i) = g_{\ell-1}(j)\}
$$

로 정의한다. 즉 같은 group 끼리만 연결을 허용한다.

중요한 구현 규칙은 아래와 같다.

- input stream 자체는 grouped object 가 아니므로 input -> hidden_1 은 `tear = 1` 이어도 구조 mask 를 적용하지 않는다.
- recurrent variant 에서는 recurrent projection 에 대해서도 같은 group 안 연결만 허용하는 recurrent mask 를 사용한다.

$$
R_\ell(i,j) = \mathbf{1}\{g_\ell(i) = g_\ell(j)\}
$$

- recurrent mask 역시 destination hidden-layer index 가 `tear` 이상일 때만 켠다.
- output layer 는 구조 mask 를 적용하지 않는다.
- mask 가 1 인 허용 edge 는 모두 trainable 하다. 즉 structure 는 연결 가능성만 제한하고, 허용된 edge 내부에서는 dense weight 학습을 유지한다.

## 4. clip 시나리오

clip 은 hidden layer neuron 을 group 으로 나눈 뒤 group 별 intrinsic parameter support 를 제한하는 시나리오다.

- RF 는 `w_clip_edges` 를 사용하며 단위는 Nyquist 상한이 0.5 인 cycle/sample 이다.
- LIF 는 `alpha_clip_edges` 를 사용하며 범위는 `[0, 1]` 이다.

경계를

$$
\beta_0 < \beta_1 < \dots < \beta_G
$$

라 두면 group $g$ 에 속한 neuron 은 자신의 intrinsic parameter 를

$$
[\beta_g, \beta_{g+1}]
$$

interval 안에서만 가질 수 있다. RF 에서는 이것이 resonance frequency support 를, LIF 에서는 decay coefficient support 를 뜻한다.

clip 적용 위치 규칙은 아래와 같다.

- clip 은 destination hidden-layer index 가 `tear` 이상인 hidden layer 에서만 켠다.
- 따라서 `tear = 1` 이면 hidden_1 부터 바로 clip 된다.
- output layer 는 clip 을 적용하지 않는다.
- 실제 bounded-uniform initialization 규칙은 `paper/proposed/vanila_rf.md`, `paper/proposed/vanila_lif.md` 를 따른다.

## 5. structclip 조합과 PSD 저장

`*_structclip` variant 는 structure mask 와 clip 제약을 동시에 적용한다. 즉 hidden-to-hidden connectivity 는 group 내부로 제한하고, 같은 group assignment 를 이용해 intrinsic parameter support 도 제한한다.

grouped hidden layer 는 PSD 저장에서 아래 원칙을 따른다.

- hidden layer 전체 PSD bundle 은 그대로 저장한다.
- 추가로 `hidden_n/block/block_k/` 아래 group 별 block PSD bundle 을 저장한다.
- hidden layer root 와 각 block root 에 `w_plot.png` 를 함께 저장한다.
- output layer 는 block subdirectory 를 만들지 않는다.

run metadata 에는 최소 아래 항목이 드러나야 한다.

- `num_groups`
- `band_neuron_ends`
- `group_cumulative_ends_per_hidden_layer`
- `group_ids_per_hidden_layer`
- `tear`
- `rf_clip_edges` 또는 `lif_clip_edges`
- `recurrent`
- `rf_reset_mode` 또는 `lif_reset_mode`
