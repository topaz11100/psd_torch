# PCA 대표화와 fixed-reference basis

## 1. 목적

PCA 대표화는 hidden-state map의 row 축을 소수의 component trajectory로 바꾼 뒤, 각 component의 시간 PSD를 분석하는 방법이다. 단순히 PC1만 보는 것이 아니라 $K$개 component를 설정할 수 있다. fixed-reference 모드는 checkpoint 간 비교에서 “좌표계가 바뀌는 문제”를 막기 위한 장치다.

## 2. 입력과 fit matrix

입력은 SignalMap

$$
X \in \mathbb{R}^{S \times R \times T}
$$

이다. PCA의 feature는 row 축 $R$이고 observation은 sample-time 쌍이다. 따라서 fit matrix는

$$
A \in \mathbb{R}^{(S T) \times R}
$$

로 만든다.

$$
A_{(s,t),r}=X_{s,r,t}
$$

centering이 켜져 있으면 feature mean

$$
\mu_r=\frac{1}{ST}\sum_{s,t}A_{(s,t),r}
$$

을 빼고 SVD를 수행한다.

## 3. basis 정의

PCA basis는

$$
W \in \mathbb{R}^{R \times K}
$$

이고 $K$는 component 수다. projection은 sample별로

$$
Y_s = (X_s^\top - \mu)W
$$

이다. 여기서 $X_s \in \mathbb{R}^{R \times T}$이고 $Y_s \in \mathbb{R}^{T \times K}$다. 분석용 shape는 다시

$$
Y \in \mathbb{R}^{S \times K \times T}
$$

로 둔다.

## 4. component 부호 규칙

PCA component의 부호는 수학적으로 임의다. 비교를 deterministic하게 만들기 위해 각 component $w_k$에서 절대값이 가장 큰 loading index를 찾고,

$$
i^* = \arg\max_i |w_{i,k}|
$$

만약 $w_{i^*,k}<0$이면 component 전체를 뒤집는다.

$$
w_k \leftarrow -w_k
$$

이를 `largest_abs_loading_positive` 규칙이라 한다.

## 5. fit mode

### fit_per_checkpoint

각 checkpoint가 자기 basis를 fit한다. 이 경우 component 좌표계가 checkpoint마다 달라질 수 있으므로 distance 비교는 원칙적으로 비활성화한다. 비교하려면 같은 basis id가 명시적으로 같아야 한다.

### fixed_reference

reference checkpoint, split, probe scope, layer, series에서 basis를 fit한다. target checkpoint는 같은 basis를 apply한다. 이때 target output은 reference 좌표계 위에 표현되므로 checkpoint 간 distance가 의미를 갖는다.

## 6. basis key와 record

basis key는 비교 가능한 PCA 좌표계를 식별한다. 최소 항목은 다음이다.

```text
reference_checkpoint_epoch
reference_split
reference_scope
reference_probe_family
layer_index, layer_name
signal_kind, series
n_components, row_count
centering, sign_convention
```

basis record는 key, basis id, mean, components, explained variance, explained variance ratio, artifact path를 갖는다.

## 7. basis id와 비교 규칙

basis id는 key와 basis tensor metadata에서 deterministic하게 만든다. PCA spectral artifact는 같은 non-empty basis id끼리만 비교할 수 있다.

$$
\operatorname{Comparable}(A,B)=
\left[pca\_basis\_id(A)=pca\_basis\_id(B)\ne\varnothing\right]
$$

basis id가 없거나 다르면 distance writer는 실패해야 한다. target에서 basis가 없다고 자동으로 새 basis를 fit하면 fixed-reference의 의미가 깨진다.

## 8. artifact

`pca_basis.csv`는 basis tensor 전체가 아니라 metadata와 component별 explained variance row를 저장한다. 실제 mean/components tensor는 `.pt` artifact로 저장한다. spectral output에는 `pca_basis_id`와 `component_id`가 반드시 남는다.
