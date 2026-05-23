# Probe family 이론

## 1. 목적

probe family는 “어떤 입력 표본으로 trace를 수집하는가”를 정의한다. spectral object는 probe 선택에 강하게 의존하므로, probe sampling은 artifact metadata의 일부다.

dataset split $D_s=\{(x_i,y_i)\}_{i=1}^{N_s}$가 있을 때 probe family는 index 집합

$$
\mathcal{I}_{s,\eta}\subseteq \{1,\ldots,N_s\}
$$

를 만든다. 여기서 $s$는 split, $\eta$는 probe family다.

## 2. balanced_global

label별 quota를 균등하게 잡는다. class 집합이 $\mathcal{C}$이고 sample count가 $M$이면 기본 quota는

$$
q_c \approx \frac{M}{|\mathcal{C}|}
$$

이다. 나머지는 deterministic rounding으로 배분한다. 목적은 class imbalance가 spectral signature를 지배하지 않게 하는 것이다.

## 3. distributed_set

split의 empirical label distribution을 반영한다.

$$
\hat{p}_c=\frac{n_c}{N_s}
$$

이고 quota는

$$
q_c \approx M\hat{p}_c
$$

로 둔다. class_counts, quotas, seed는 ProbeManifest에 남아야 한다.

## 4. label_set

지정한 label 집합에 속하는 sample만 사용한다.

$$
\mathcal{I}_{label\_set}(L)=\{i:y_i\in L\}
$$

단일 label 분석 또는 class subset 분석에 쓴다.

## 5. label_single

label별 하나의 sample을 deterministic하게 선택한다.

$$
|\{i\in\mathcal{I}:y_i=c\}|=1
$$

이다. 이 family는 특정 분석 전용이 아니라 모든 신호분석에서 쓸 수 있는 top-level probe family다.

exclusion family가 주어지면 먼저 제외 index 집합 $E$를 만들고,

$$
\mathcal{C}_c=\{i:y_i=c, i\notin E\}
$$

에서 하나를 고른다. 남은 후보가 없으면 실패한다. 이 방식은 대표 probe와 single-sample probe가 겹치는 것을 막는다.

## 6. ProbeManifest와 ProbeBatch

ProbeManifest는 선택 규칙을 재현 가능하게 만드는 metadata다.

```text
probe_family, split, scope, seed, selected_indices,
selected_labels, class_counts, quotas, exclusion_family,
excluded_indices, probe_manifest_id
```

ProbeBatch는 실제 model forward에 들어가는 batch다. sample_indices는 batch-local이 아니라 split 기준 index여야 한다. 그래야 trace와 artifact가 원래 데이터 표본에 연결된다.

## 7. 이 객체로 하는 일

probe family는 analysis scope를 정의한다. 같은 checkpoint라도 balanced probe와 distributed probe의 PSD는 다른 질문에 답한다. 따라서 distance 비교도 probe metadata가 같은지 확인해야 한다.
