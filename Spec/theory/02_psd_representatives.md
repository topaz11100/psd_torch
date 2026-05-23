# PSD 대표화 이론

## 1. 목적

PSD 대표화는 layer trace의 시간 주파수 구조를 비교 가능한 curve 또는 matrix로 바꾸는 절차다. 핵심은 신호를 먼저 평균하지 않는다는 점이다. 공식 원칙은 다음이다.

$$
\text{대표 PSD 객체}
=
\operatorname{Rep}\left(\operatorname{PSD}(\text{row signal})\right)
$$

반대로

$$
\operatorname{PSD}\left(\operatorname{Rep}(\text{signal})\right)
$$

은 current main object가 아니다. row별 time signal이 먼저 power spectrum이 되고, 그 power 객체를 row/sample 축에서 대표화한다.

## 2. 입력 객체

입력 SignalMap은

$$
X \in \mathbb{R}^{S \times R \times T}
$$

이다. $S$는 sample, $R$은 row, $T$는 time이다. 각 sample $s$와 row $r$에 대한 time signal은

$$
x_{s,r}[t], \quad t=0,\ldots,T-1
$$

이다.

## 3. time FFT와 raw power

one-sided frequency grid는

$$
\mathcal{F}_T=
\left\{f_k=\frac{k}{T}: k=0,\ldots,\left\lfloor\frac{T}{2}\right\rfloor\right\}
$$

이다. window와 centering을 적용한 신호를 $\tilde{x}_{s,r}[t]$라 하면 raw power는

$$
P_{s,r,k}=\left|\operatorname{rFFT}(\tilde{x}_{s,r})_k\right|^2
$$

이다. 모든 userbin, dB, distance는 이 raw power 객체에서 출발한다.

## 4. exact spectral axis

exact mode는 $\mathcal{F}_T$의 native frequency bin을 그대로 사용한다. exact artifact는 같은 frequency identity를 가진 exact artifact하고만 비교할 수 있다.

## 5. userbin spectral axis

userbin mode는 사용자가 정한 edge

$$
0 \le b_0 < b_1 < \cdots < b_M \le 0.5
$$

로 exact frequency bin을 묶는다. bin $m$의 index 집합은

$$
\mathcal{I}_m=\{k: b_m \le f_k < b_{m+1}\}
$$

이고 대표값은 raw power에서 계산한다.

$$
P^{bin}_{s,r,m}=\operatorname{Reducer}\{P_{s,r,k}:k\in\mathcal{I}_m\}
$$

Reducer는 `mean` 또는 `median`이다. dB 변환은 이 단계 이후 finalize에서만 수행한다.

## 6. `mean` representative

row PSD를 평균해 sample별 curve를 만든다.

$$
C_{s,k}^{mean}=\frac{1}{R}\sum_{r=1}^{R}P_{s,r,k}
$$

run-level curve는 sample 축 평균이다.

$$
\bar{C}_k=\frac{1}{S}\sum_{s=1}^{S}C_{s,k}
$$

이 객체는 layer 전체의 평균 주파수 signature를 본다.

## 7. `median` representative

row axis에서 median을 취한다.

$$
C_{s,k}^{median}=\operatorname{median}_{r}\,P_{s,r,k}
$$

mean보다 outlier row에 덜 민감하다. userbin reducer의 median과 다르다. 여기서 median은 row 축 대표화이고, userbin median은 frequency bin 내부 대표화다.

## 8. `element_psd` representative

row axis를 보존한다.

$$
M_{r,k}=\frac{1}{S}\sum_{s=1}^{S}P_{s,r,k}
$$

결과는 `spectral_matrix_1d`이며 shape는 $R \times F$ 또는 $R \times B$다. 이 객체는 “어떤 neuron/channel row가 어떤 frequency에 반응하는지”를 보존한다.

## 9. `pca` representative

PCA는 row 축을 component 축으로 바꾼 뒤 component trajectory에 PSD를 적용한다. 즉

$$
X \in \mathbb{R}^{S \times R \times T}
\quad\rightarrow\quad
Y \in \mathbb{R}^{S \times K \times T}
$$

이고 이후 $Y_{s,k}[t]$에 PSD를 계산한다. PCA는 row 평균이 아니라 row-space projection이다.

## 10. dB와 distance

raw power summary $P$의 dB 표현은

$$
P_{dB}=10\log_{10}(P+\epsilon)
$$

이다. 이 변환은 update 또는 binning 전에 하지 않는다.

허용 distance는 두 개다.

$$
d_{centered}(a,b)=\left\|(a-\bar{a})-(b-\bar{b})\right\|_2
$$

$$
d_{diff}(a,b)=\left\|\Delta a-\Delta b\right\|_2
$$

matrix에서는 flatten 전에 global mean centering 또는 frequency-axis first difference를 적용한다. exact와 userbin은 직접 비교하지 않는다.
