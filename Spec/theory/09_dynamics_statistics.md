# 동역학 통계

동역학 통계는 PSD/FFT와 별개로 cell parameter의 분포를 기록한다.

## 주요 parameter

- LIF: `alpha`
- RF: `center_frequency`, `damping`
- IF/LIF/RF: threshold

각 parameter는 layer, layer_index, statistic, value_unit을 갖는 `filter_snapshot` 또는 `filter_trend` category로 저장된다.

## statistic

기본 통계는 count, mean, std, min, q25, q50, q75, max다. count는 값의 개수이므로 다른 parameter value와 단위가 다르다.

## 해석

PSD/FFT가 신호 결과를 보여준다면 동역학 통계는 그 결과를 만든 parameter 조건을 보조 설명한다.
