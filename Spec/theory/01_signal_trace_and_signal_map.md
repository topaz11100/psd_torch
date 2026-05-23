# Trace와 SignalMap 이론

## 1. 대상 정의

SNN 분석에서 첫 번째 공식 객체는 layer trace다. 한 batch의 raw trace는

$$
Z \in \mathbb{R}^{B \times T \times D_1 \times \cdots \times D_m}
$$

로 둔다. $B$는 sample, $T$는 time, 뒤쪽 축은 feature 또는 spatial/channel 축이다. 이 저장 순서는 외부 contract다. 내부 구현이 다른 순서를 쓰더라도 외부 trace는 항상 `B,T,*`로 나온다.

분석용 객체는 SignalMap이다.

$$
X \in \mathbb{R}^{S \times R \times T}
$$

여기서 $S=B$, $R=\prod_i D_i$ 또는 의미 있는 row 축, $T$는 원래 time axis다. time axis는 자르거나 chunk하지 않는다.

## 2. 변환 규칙

MLP trace는

$$
Z \in \mathbb{R}^{B \times T \times F}
$$

이고 SignalMap은

$$
X_{b,r,t}=Z_{b,t,r}
$$

이므로 shape는 `B,F,T`가 된다.

image-like trace는

$$
Z \in \mathbb{R}^{B \times T \times C \times H \times W}
$$

이고 row를

$$
r = ((cH)+h)W+w
$$

로 flatten해

$$
X_{b,r,t}=Z_{b,t,c,h,w}
$$

로 둔다. 이 경우 row semantics는 spatial flattening을 명시해야 한다.

## 3. 공식 series

공식 series는 state의 물리적 의미를 드러내야 한다.

- `input_current`: cell에 들어간 전류.
- `membrane_pre`: reset 전 membrane 값.
- `decision`: threshold 판정용 값.
- `spike`: spike output.
- `membrane_post`: reset 후 membrane 값.
- `rf_real_pre`, `rf_imag_pre`: RF cell의 reset 전 real/imag state.
- `rf_real_post`, `rf_imag_post`: RF cell의 reset 후 real/imag state.
- `output_membrane_pre`, `output_decision`, `logits`: readout 계열 출력.

분석에서 series 이름은 신호의 의미를 보존한다. 모호한 단일 이름으로 여러 상태를 섞지 않는다.

## 4. metadata의 역할

SignalMap은 tensor만으로 충분하지 않다. 같은 $S,R,T$ shape라도 어떤 checkpoint, 어떤 probe, 어떤 layer인지에 따라 비교 가능성이 달라진다. 따라서 최소 metadata는 다음이다.

```text
run_id, checkpoint_epoch, split, scope, probe_family,
label, sample_indices, labels,
layer_index, layer_name, signal_kind, series,
scenario, constraint_hash
```

PCA projection 후에는 `pca_basis_id`, 2D FFT에는 `row_axis_semantics`, trace artifact에는 chunk path와 layout이 추가된다.

## 5. unavailable series 정책

일부 readout은 spike series를 내지 않는다. 이때 zero tensor를 만들어 저장하면 실제 spike가 없었다는 사실을 숨기게 된다. 따라서 unavailable series는 status와 reason으로 기록하거나 skip한다. fake tensor를 만들지 않는 것이 공식 정책이다.

## 6. 이 객체로 하는 일

SignalMap은 모든 신호분석의 공통 입력이다.

- PSD 대표화는 각 row의 time PSD를 만든다.
- PCA는 row 축 $R$을 component 축 $K$로 바꾼다.
- 2D FFT는 $R \times T$ matrix 전체를 spectral matrix로 바꾼다.
- trace writer는 raw `B,T,*` tensor를 sample-axis chunk로 저장한다.
