# Theory Overview

이 프로젝트의 과학적 질문은 다음과 같다.

> SNN의 시간 동역학은 입력 데이터의 주파수 구조를 어떤 방식으로 보존, 증폭, 변형 또는 소거하는가?

이를 위해 각 layer의 membrane/spike trace를 시간 신호로 보고, PSD와 2D FFT를 통해 신호의 spectral profile을 비교한다. 모델은 단순한 classifier가 아니라 시간-주파수 변환기를 내재한 동역학 시스템으로 취급한다.

## 기본 notation

입력 batch를 \(X\), layer \(\ell\)의 trace를 \(Y^{(\ell)}\)라 둔다.

\[
X \in \mathbb{R}^{B\times T\times D}, \qquad
Y^{(\ell)} \in \mathbb{R}^{B\times T\times C_\ell}.
\]

각 neuron/channel \(c\)의 시간 신호는

\[
y^{(\ell)}_{b,c}(t), \qquad t=0,1,\ldots,T-1.
\]

PSD는 다음 one-sided periodogram으로 정의한다.

\[
P^{(\ell)}_{b,c}(\omega_k)=\frac{1}{T}\left|\sum_{t=0}^{T-1} y^{(\ell)}_{b,c}(t)e^{-i2\pi kt/T}\right|^2.
\]

대표곡선은 sample과 channel 축을 줄여 layer-level spectral signature를 만든다.

## 해석 목표

- 입력과 hidden layer의 PSD shape distance를 측정한다.
- 인접 layer 사이의 spectral drift를 측정한다.
- fixed PCA basis로 epoch별 곡선 변화를 같은 좌표계에서 비교한다.
- `dataset_fft`로 데이터 자체의 spectral baseline을 따로 산출한다.
- `element_psd`, `element_fft`, `2d_fft_analysis`로 aggregate PSD가 놓치는 neuron별/공간별 이질성을 확인한다.
