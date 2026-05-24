# FFT와 2D FFT

## dataset FFT

`dataset_fft`는 prepared input SignalMap $X\in\mathbb{R}^{S	imes R	imes T}$에 대해 time axis rFFT power를 계산한다. row 축은 평균하여 dataset input의 time-frequency 구조를 요약한다.

$$
F_k=rac{1}{S}\sum_s rac{1}{R}\sum_r |\operatorname{rFFT}(x_{s,r})_k|^2
$$

## model 2D FFT

`2d_fft_analysis`는 hidden/output SignalMap의 row-time matrix 전체에 대해 2D FFT를 계산한다.

$$
Q_{u,v}=rac{1}{S}\sum_s |\operatorname{FFT2}(X_{s,:,:})_{u,v}|^2
$$

row frequency는 neuron/channel row 방향 구조를, time frequency는 시간 진동 구조를 나타낸다.

## variant와 scale

- `raw`: 원 신호 사용.
- `centered`: 분석 대상 축 평균 제거 후 계산.
- `raw` scale: power 저장.
- `db` scale: 집계된 power를 dB로 변환.

모델 2D FFT는 input을 포함하지 않는다.
