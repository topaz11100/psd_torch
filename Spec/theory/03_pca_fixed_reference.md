# Fixed PCA Reference

PSD curve는 epoch와 layer마다 분포가 달라진다. epoch별 PCA를 따로 계산하면 좌표계가 계속 회전하므로 추세 비교가 어렵다. 따라서 프로젝트는 고정 기준 epoch \(e_0\)에서 PCA basis를 계산하고, 모든 checkpoint를 그 basis에 사영한다.

## Basis construction

layer \(\ell\)의 curve matrix를

\[
A^{(\ell)}_{e_0}\in\mathbb{R}^{N\times F}
\]

라 하자. \(F\)는 frequency bin 수다. 평균을 제거한 뒤 SVD를 계산한다.

\[
\bar{a}=\frac{1}{N}\sum_{n=1}^{N}A_n,\qquad
A_c=A-\mathbf{1}\bar{a}^\top,
\]

\[
A_c=U\Sigma V^\top.
\]

상위 \(K\)개 right singular vector가 fixed basis다.

\[
B_K = V_{1:K} \in \mathbb{R}^{K\times F}.
\]

## Projection

임의 epoch \(e\)의 curve \(a_e\)는

\[
z_e = B_K(a_e-\bar{a})
\]

로 projection된다. 같은 \(B_K\)와 \(\bar{a}\)를 사용하기 때문에 epoch 간 좌표 비교가 의미를 갖는다.

## Regularization use

학습 중 PSD-PCA regularizer는 input 또는 이전 layer의 spectral coordinates와 hidden layer coordinates 사이의 거리를 penalize한다.

\[
\mathcal{L}_{\mathrm{PCA}} = \lambda\sum_{\ell} \|z^{(\ell)} - z^{(\ell-1)}\|_2^2.
\]

이는 raw PSD 전체를 직접 맞추기보다 주요 spectral mode를 보존하도록 유도한다.
