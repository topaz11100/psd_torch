# 10 Artifacts Distance And Manifests

## 2026 개정 해설

Artifact CSV와 YAML manifest는 실험 재현성의 감사 로그다. CSV row는 어떤 checkpoint, dataset, scope, layer, signal, metric에서 나왔는지 추적 가능해야 하고, manifest는 산출물 inventory를 YAML로 기록한다.

# Artifact, distance, manifest

## CSV category

현재 주요 category는 다음이다.

- `dataset_curve`, `dataset_dispersion`, `dataset_fft`
- `training_metric`
- `analysis_curve`, `analysis_dispersion`
- `element_psd`, `element_fft`, `analysis_2d_fft`
- `pair_distance`, `layer_distance_profile`, `layer_distance_trend`
- `layer_dispersion_profile`, `layer_dispersion_trend`
- `filter_snapshot`, `filter_trend`
- `analysis_manifest`, `dataset_psd_manifest`, `dataset_fft_manifest`, `plotting_manifest`

## Manifest

Manifest는 YAML 산출물 inventory다. 각 stage는 어떤 CSV를 만들었는지, 실패/경고가 있었는지, 어떤 scope에 해당하는지 기록해야 한다. 체크포인트 내부 metadata는 JSON-compatible payload로 유지한다.

## Distance

Distance는 같은 spectral axis, 같은 scale, 같은 reducer, 같은 layer/series 의미를 가진 artifact 사이에서만 해석한다. input-reference distance는 모델 분석에서 생성하지 않는다.

## 금지 사항

- 모델 분석에서 input layer artifact를 생성하지 않는다.
- manifest가 아닌 실험/분석 수치 산출물을 YAML로 저장하지 않는다.
- stage가 다른 artifact를 같은 category로 섞지 않는다.
