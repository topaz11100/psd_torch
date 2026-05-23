# 2D FFT 이론

## 1. 목적

2D FFT는 PSD 대표화가 아니다. SignalMap의 row-time matrix 전체를 주파수 영역으로 보내 row 방향 구조와 time 방향 구조를 동시에 관찰하는 독립 분석 방법이다.

입력은

$$
X \in \mathbb{R}^{S \times R \times T}
$$

이고 sample별 matrix는

$$
X_s \in \mathbb{R}^{R \times T}
$$

이다.

## 2. exact 2D FFT

각 sample에 대해 row axis에는 full FFT, time axis에는 one-sided real FFT를 적용한다.

$$
F_s = \operatorname{FFT}_{row}\left(\operatorname{RFFT}_{time}(X_s)\right)
$$

raw power matrix는

$$
P_s(i,k)=|F_s(i,k)|^2
$$

이다. exact output shape는

$$
F_{row}=R,
\quad
F_{time}=\left\lfloor\frac{T}{2}\right\rfloor+1
$$

이다.

sample 축 평균은

$$
\bar{P}(i,k)=\frac{1}{S}\sum_s P_s(i,k)
$$

이다.

## 3. row frequency와 time frequency

row frequency grid는

$$
f_i^{row}=\operatorname{fftfreq}(R)_i
$$

이고 time frequency grid는

$$
f_k^{time}=\operatorname{rfftfreq}(T)_k
$$

이다. row axis에 `fftshift`를 적용하면 frequency grid도 같이 shift해야 한다. shift 여부는 artifact metadata에 남는다.

## 4. userbin 정책

2D FFT userbin은 원본 $R,T$ trace를 pooling하지 않는다. 반드시 2D FFT 후 raw power matrix에서 frequency binning을 한다.

### time_frequency

시간 주파수 축만 binning한다.

$$
P_s \in \mathbb{R}^{F_{row}\times F_{time}}
\rightarrow
P_s^{bin} \in \mathbb{R}^{F_{row}\times B_{time}}
$$

### row_frequency

row 주파수 축만 binning한다.

$$
P_s \rightarrow P_s^{bin} \in \mathbb{R}^{B_{row}\times F_{time}}
$$

### both_frequency_axes

두 축을 모두 binning한다.

$$
P_s \rightarrow P_s^{bin} \in \mathbb{R}^{B_{row}\times B_{time}}
$$

bin reducer는 raw power에서 `mean` 또는 `median`이다. dB는 finalize 이후에만 적용한다.

## 5. row_axis_semantics

row frequency는 row ordering에 의존한다. 따라서 row 의미를 명시해야 한다.

- `unordered`: 일반 MLP neuron 순서. exact row FFT는 계산할 수 있지만 해석 경고가 필요하다. row-frequency userbin은 금지한다.
- `group_ordered`: structure scenario로 group order가 의미 있는 경우.
- `feature_ordered`: feature 순서가 외부 의미를 갖는 경우.
- `spatial_flattened`: image feature를 spatial order로 flatten한 경우.
- `channel_ordered`: channel 순서만 의미 있는 경우.
- `pca_component`: PC1, PC2 같은 component row.

## 6. artifact와 거리

2D FFT artifact type은 `spectral_matrix_2d`다. row/column axis sidecar가 frequency/bin metadata를 보존한다.

matrix distance는 `centered_l2`와 `diff_l2`만 사용한다. exact/userbin, userbin axes, bin edges, row semantics, shift policy가 다르면 비교하지 않는다.
