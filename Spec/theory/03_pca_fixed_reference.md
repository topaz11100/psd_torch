# PCA 고정 기준(reference) 이론과 현재 구현

현재 root pipeline에는 PCA 확장이 **실제 구현**되어 있다. 핵심은 row/channel 집단 신호 \(X\in\mathbb{R}^{N\times R\times T}\)를 고정 기저로 투영해 모드 신호 \(Z\in\mathbb{R}^{N\times D\times T}\)를 만든 뒤, 스칼라 대표 PSD와는 다른 정보(모드별 PSD, 모드 간 스펙트럼 결합)를 분석/규제에 반영하는 것이다.

관측행렬은
\[
\widetilde{X}=\text{reshape}_{(N\cdot T,\;R)}(X)
\]
로 두고, 중심화
\[
\widehat{X}=\widetilde{X}-\mathbf{1}\mu^\top,\qquad \mu\in\mathbb{R}^{R}
\]
후 SVD(실패 시 covariance-eigh fallback)로 basis \(U_D\in\mathbb{R}^{R\times D}\)를 얻는다.
투영은
\[
Z_{n,:,t}=U_D^\top\left(X_{n,:,t}-\mu\right)
\]
로 정의한다.

## 고정 기준 원칙

1. 기준 checkpoint(`pca_ref_epoch`)에서 basis를 1회 적합한다.
2. 같은 run의 다른 checkpoint에는 같은 basis id를 재사용한다.
3. basis id가 다른 projection 결과끼리는 직접 distance를 섞지 않는다.
4. metadata에 dataset, checkpoint/epoch, split/scope, layer, family, row_count, dim, basis/centroid shape를 남긴다.

## mean/median 대표화와 PCA 대표화의 차이

- mean/median 대표화는 row 축 통계량
  \[
  \bar{x}_t=\frac1R\sum_{r=1}^R x_{r,t},\qquad \text{or median}_r(x_{r,t})
  \]
  만 남기므로 row 간 공분산 구조는 소실된다.
- PCA 대표화는 \(U_D\)를 통해 row 공분산의 주축 방향을 보존하므로, mode별 PSD와 mode 간 결합(MIMO/cross-spectrum)을 평가할 수 있다.

## DDP 제약(현재 상태)

PCA PSD regularization의 분산 동기화(broadcast) 경로는 아직 구현되지 않았다. 따라서 DDP + PCA penalty 조합은 fail-fast로 차단하는 것이 현재 정합한 계약이다.
