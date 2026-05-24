# Artifact, distance, manifest

## CSV category

현재 주요 category는 다음이다.

- `dataset_curve`, `dataset_dispersion`, `dataset_fft`
- `training_metric`
- `analysis_curve`, `analysis_dispersion`
- `element_psd`, `analysis_2d_fft`
- `pair_distance`, `layer_distance_profile`, `layer_distance_trend`
- `layer_dispersion_profile`, `layer_dispersion_trend`
- `filter_snapshot`, `filter_trend`
- `analysis_manifest`, `dataset_psd_manifest`, `dataset_fft_manifest`, `plotting_manifest`

## Manifest

Manifest는 산출물 inventory다. 각 stage는 어떤 CSV를 만들었는지, 실패/경고가 있었는지, 어떤 scope에 해당하는지 기록해야 한다.

## Distance

Distance는 같은 spectral axis, 같은 scale, 같은 reducer, 같은 layer/series 의미를 가진 artifact 사이에서만 해석한다. input-reference distance는 모델 분석에서 생성하지 않는다.

## 금지 사항

- 모델 분석에서 input layer artifact를 생성하지 않는다.
- JSON 설정 외 YAML 설정을 공식 실행 계약에 넣지 않는다.
- stage가 다른 artifact를 같은 category로 섞지 않는다.
