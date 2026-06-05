# Element-wise FFT

`element_fft`는 hidden/output trace map

\[
M \in \mathbb{R}^{N\times R\times T}
\]

에서 각 sample \(n\)과 neuron/row \(r\)의 시간 신호 \(M_{n,r}(t)\)를 독립적으로 변환한다.

\[
\hat{M}_{n,r}(k)=\sum_{t=0}^{T-1}M_{n,r}(t)e^{-i2\pi kt/T}, \qquad k=0,\ldots,\lfloor T/2\rfloor.
\]

프로젝트 산출물은 sample 축을 평균한 뒤, neuron 축을 보존한다.

\[
E_r(k)=\frac{1}{N}\sum_{n=1}^{N}\hat{M}_{n,r}(k).
\]

저장 component는 `real`, `imag`, `magnitude`, `phase`다. PSD가 \(|\hat{M}|^2/T\)처럼 power만 남기는 반면, element FFT는 complex phase 정보를 별도 component로 남긴다. 이 차이 때문에 같은 power spectrum을 가진 두 뉴런도 phase alignment가 다르면 `element_fft`에서 구별될 수 있다.

`raw` variant는 DC component를 포함한다. `centered` variant는 각 neuron별 시간 평균을 제거하여 변화 성분에 대한 Fourier component를 강조한다.
