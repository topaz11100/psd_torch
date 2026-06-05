# PSD Representatives

## Periodogram

각 row 신호 \(m_{n,r}(t)\)의 one-sided FFT를

\[
\hat{m}_{n,r}(\omega_k)=\sum_{t=0}^{T-1}m_{n,r}(t)e^{-i2\pi kt/T}
\]

라고 할 때 PSD는

\[
P_{n,r}(\omega_k)=\frac{1}{T}|\hat{m}_{n,r}(\omega_k)|^2.
\]

프로젝트의 `exact` curve space는 이 periodogram을 직접 사용한다.

## Representative curve

Layer-level curve는 sample과 row 축을 reducer \(\rho\)로 줄인다.

\[
R^{(\ell)}(\omega_k)=\rho_{n,r}\big(P^{(\ell)}_{n,r}(\omega_k)\big).
\]

기본 reducer는 mean이다.

\[
R^{(\ell)}(\omega_k)=\frac{1}{NR}\sum_{n=1}^{N}\sum_{r=1}^{R}P^{(\ell)}_{n,r}(\omega_k).
\]

Median reducer는 outlier neuron이나 class imbalance가 강할 때 안정적인 대안이다.

## dB scale

Power scale의 dynamic range가 크므로 dB 변환을 지원한다.

\[
R_{\mathrm{dB}}(\omega)=10\log_{10}(R(\omega)+\epsilon).
\]

\(\epsilon\)은 numerical floor다. dB scale에서 L2 distance를 해석할 때는 절대 power 차이가 아니라 상대적인 spectral shape 차이에 가깝다.

## Distance

두 curve \(a,b\)의 centered L2는

\[
d_{\mathrm{centered}}(a,b)=\left\|\left(a-\bar{a}\right)-\left(b-\bar{b}\right)\right\|_2.
\]

`diff_l2`는 단순 차분

\[
d_{\mathrm{diff}}(a,b)=\|a-b\|_2
\]

을 사용한다. Centered distance는 전체 energy offset보다 shape 차이를 더 강조한다.
