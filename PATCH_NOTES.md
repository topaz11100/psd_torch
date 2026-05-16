# PSD analysis patch notes

이 zip은 기존 프로젝트 위에 덮어쓸 수 있는 코드 패치 묶음이다. 모델 학습 로직은 수정하지 않았다.

수정 범위:

- `src/util/csv_schema.py`
- `src/psd_analysis.py`
- `src/plotting.py`
- `src/neurons/RF_neuron.py`
- `src/neurons/cnn2d.py`
- `Spec/impl/spec/csv_schema.md`
- `Spec/impl/spec/psd_analysis.md`
- `Spec/impl/spec/plotting.md`

주요 변경:

1. `analysis_curve`, `analysis_dispersion` 에서 `series` 를 보존한다.
2. `pair_distance`, `pairwise_dependency_appendix` 에 `source_series`, `target_series`, `reducer` 를 보존한다.
3. `drift_distance` 는 같은 checkpoint 안의 input PSD 대비 대상 layer/series PSD shape distance trend 로 생성한다.
4. `drift_distance` representative curve distance 는 `mean`, `median` reducer 를 구분한다.
5. `drift_distance` dispersion variance/MAD 는 `reducer=none` 으로 생성한다.
6. `src/plotting.py` 는 업로드한 recursive plotter 기반으로 교체했고, drift plot 을 `curve_distance`, `dispersion_variance`, `dispersion_mad` 폴더로 분리한다.
7. RF 계열 filter snapshot 은 `damping`, `center_frequency` 를 기록한다.

기존 PSD output 은 `series`/`reducer` 정보가 누락된 상태일 수 있으므로 재사용하지 말고 삭제 후 재분석해야 한다.

## 2026-05-01 추가 패치

- PSD 대표 곡선과 dispersion의 dB 기준을 `mean(dB)` 계열이 아니라 raw domain 연산 완료 후 dB 변환 기준으로 고정했다.
- `psd_analysis.py`, `dataset_psd.py` 기본 실행 경로에서 userbin 분석, 저장, 산출을 비활성화했다.
- `mnist`, `cifar-10` prepared storage와 runtime views를 `(T,C,H,W)`, `T=4` static repeat direct input으로 바꿨다.

## 2026-05-01 data-flow and ResNet PSD patch

- `mnist`, `cifar-10` prepared storage now stores static direct input as `(T,C,H,W)` with `T=4`.
- `model_input`, `psd_input`, and `image_psd_view` return `(T,C,H,W)` for static image datasets.
- Dense SNN inputs are flattened only by `canonicalize_model_input_batch` into `(B,T,C*H*W)`.
- CNN models now require prepared `(B,T,C,H,W)` for static-repeat image input and no longer synthesize repeated frames inside `_prepare_input`.
- PSD flattening for static images happens only in signal analysis: `(B,T,C,H,W) -> (B,C*H*W,T)`.
- ResNet BasicBlock now records `residual_add` after the complete shortcut addition; `layer_input`, `membrane`, and `spike` are all post-add signals.
- ResNet residual layer names are unified as `*_residual_add` to prevent layer-index mismatch in PSD analysis.
