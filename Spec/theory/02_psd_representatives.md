# PSD 대표화

## 기본 PSD

SignalMap $X \in \mathbb{R}^{S	imes R	imes T}$에서 sample $s$, row $r$의 time signal은 $x_{s,r}[t]$다. one-sided frequency bin은

$$
f_k = rac{k}{T}, \quad k=0,\ldots,\lfloor T/2floor
$$

이다. centered variant는 time 평균을 제거한 뒤 PSD를 계산한다.

## 대표 curve

row별 PSD $P_{s,r,k}$를 먼저 계산한 뒤 row/sample 축을 요약한다.

Mean representative:

$$
C^{mean}_k = rac{1}{S}\sum_s rac{1}{R}\sum_r P_{s,r,k}
$$

Median representative:

$$
C^{median}_k = rac{1}{S}\sum_s \operatorname{median}_r P_{s,r,k}
$$

## dispersion

분산과 MAD는 같은 frequency bin 안에서 row 축 분산성을 기록한다.

$$
V_k = rac{1}{S}\sum_s \operatorname{Var}_r(P_{s,r,k})
$$

MAD는 row median 기준 절대편차의 median이다.

## element PSD

Element PSD는 row 축을 보존한다.

$$
M_{r,k}=rac{1}{S}\sum_s P_{s,r,k}
$$

모델 element PSD는 hidden/output row에 대해서만 생성한다. input row에 대한 element PSD가 필요하면 dataset 분석 stage를 사용한다.

## scale

`raw`는 power 값을 그대로 저장한다. `db`는 raw power 집계가 끝난 뒤 dB로 변환한다. batch별 dB를 평균하지 않는다.
