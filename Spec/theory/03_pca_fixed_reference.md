# PCA 확장 경계

현재 공식 root pipeline의 필수 단계는 PSD, element PSD, dataset FFT, model 2D FFT다. PCA 기반 fixed-reference 분석은 향후 확장 가능한 신호분석 방식으로 남긴다.

PCA를 추가할 경우 원칙은 다음이다.

1. reference checkpoint 또는 reference scope에서 basis를 fit한다.
2. 같은 basis id를 target checkpoint에 적용한다.
3. 서로 다른 basis id를 가진 projection 결과끼리는 직접 distance를 계산하지 않는다.
4. basis metadata에는 dataset, checkpoint, layer, series, row semantics, component 수를 남긴다.

현재 필수 CSV schema와 bash/config 단계에는 PCA stage를 포함하지 않는다.
