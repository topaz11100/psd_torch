# Two-Dimensional FFT

`element_psd`는 각 row의 시간 주파수만 본다. 하지만 CNN/MLP hidden map에서는 row index 자체에도 구조가 있을 수 있다. `2d_fft_analysis`는 row × time map 전체를 2D 신호로 보고 다음 변환을 적용한다.

\[
\hat{M}(k_r,k_t)=\sum_{r=0}^{R-1}\sum_{t=0}^{T-1}M(r,t)e^{-i2\pi(k_r r/R+k_t t/T)}.
\]

Power matrix는

\[
P(k_r,k_t)=|\hat{M}(k_r,k_t)|^2.
\]

sample 평균을 취하면 scope별 2D spectral signature가 된다.

## 해석

- \(k_t\)축: 시간 frequency. 일반 PSD와 동일한 temporal oscillation을 나타낸다.
- \(k_r\)축: neuron/channel/spatial row 방향의 진동성. 인접 neuron이나 spatial location 사이의 구조적 반복을 포착한다.
- 중심부: low-frequency smooth activation.
- 바깥 영역: 고주파, neuron별 급격한 변화, sparse event pattern.

2D FFT는 aggregate PSD가 같은 두 layer라도 row 방향 구조가 다를 수 있음을 보여준다.


## Element-wise FFT와의 구분

`element_fft`는 각 뉴런 row를 독립적인 1차원 시간 신호로 보고 complex FFT의 real, imaginary, magnitude, phase component를 저장한다. 반면 `2d_fft_analysis`는 row 축과 time 축을 동시에 변환해 row 간 배열 구조까지 포함한 spectral matrix를 만든다. 따라서 `element_fft`는 뉴런별 시간 필터 해석에, `2d_fft_analysis`는 row 간 구조 또는 spatially arranged activation map의 구조 해석에 사용한다.
