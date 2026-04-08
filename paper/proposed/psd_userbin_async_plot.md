# PSD userbin / async plot writer specification

## 1. 목적

이 문서는 PSD 관련 heatmap 표현 규칙과 plot writer 비동기 저장 규칙을 정의한다. 대상 실험은 `psd_analysis` 와 `dataset_psd` 다. `psd_analysis` 는 학습 loop 안에서 PNG 를 직접 그리지 않고 deferred numeric payload 를 저장한 뒤 학습 종료 후 렌더링하는 두 단계 전략을 사용하고, `dataset_psd` 는 학습 loop 가 없으므로 train/test 전체 입력 PSD 와 `probe_set_reference/` 저장 모두 직접 async plot writer 를 사용한다.

핵심 원칙은 아래와 같다.

1. waveform / mean spectrogram 은 exact bin 축으로 저장한다.
2. periodogram / spectrogram heatmap 은 userbin 표현 계층이다.
3. raw / centered 두 variant 를 모두 저장한다.
4. exact periodogram / exact spectrogram 공식 경로에는 taper window 를 적용하지 않는다.
5. dB 저장이 활성화되면 각 선형 power-like plot 에 대해 `_db.png` suffix 를 붙인 대응 dB plot 을 추가 저장한다.

## 2. userbin 을 쓰는 그림과 exact 를 쓰는 그림

### 2.1 exact 로 저장하는 그림

아래 그림은 exact 축을 그대로 사용한다.

- `mean_psd_waveform_exact_raw.png`
- `mean_psd_waveform_exact_centered.png`
- `mean_spectrogram_exact_raw.png`
- `mean_spectrogram_exact_centered.png`

원칙은 다음과 같다.

- waveform 과 mean spectrogram 은 exact 저장이 기본이다.
- exact 결과는 raw / centered 두 variant 를 모두 저장한다.
- exact 공식 경로는 windowless simple periodogram / sliding simple periodogram 이어야 한다.
- dB 저장이 활성화되면 exact plot 도 같은 exact 축을 유지한 채 $10 \log_{10}(x + 10^{-12})$ 를 적용한 `_db.png` 대응본을 함께 저장한다.

### 2.2 userbin 으로 저장하는 그림

아래 그림은 exact 결과를 userbin 으로 집계해 저장한다.

- `element_psd_heatmap_userbin_raw.png`
- `element_psd_heatmap_userbin_centered.png`
- `element_spectrogram_heatmap_userbin_raw.png`
- `element_spectrogram_heatmap_userbin_centered.png`

원칙은 다음과 같다.

- waveform 과 mean spectrogram 은 userbin 으로 저장하면 안 된다.
- userbin 은 element heatmap 가독성을 위한 표현 계층이다.
- spectrogram 도 periodogram 과 같은 원칙을 따른다. mean spectrogram 은 exact, element heatmap 만 userbin 이다.
- dB 저장이 활성화되면 userbin heatmap 도 같은 축을 유지한 채 $10 \log_{10}(x + 10^{-12})$ 를 적용한 `_db.png` 대응본을 함께 저장한다.

## 3. PSD heatmap 규칙

`element_psd_heatmap_userbin_raw.png`, `element_psd_heatmap_userbin_centered.png` 는 아래를 만족해야 한다.

- x축: userbin 중심 주파수, unit = cycle/sample
- y축: element index
- origin: `lower`
- 모든 칸에 수치 annotation 포함
- large canvas 사용
- row index 가 낮을수록 아래쪽에 위치

## 4. spectrogram heatmap 규칙

`element_spectrogram_heatmap_userbin_raw.png`, `element_spectrogram_heatmap_userbin_centered.png` 는 아래를 만족해야 한다.

- x축: frame center / frequency userbin 의 frame-major 열 순서
- y축: element index
- origin: `lower`
- annotation 없음
- exact spectrogram 을 userbin 으로 집계한 뒤 `(rows, bands, frames)` 를 frame-major 2차원 heatmap 으로 펼친다.
- row index 가 낮을수록 아래쪽에 위치

## 5. summary.json 규칙

`save_psd_bundle(...)` 가 `summary.json` 을 저장할 때에는 아래 의미가 드러나야 한다.

- `variants_saved = ["raw", "centered"]`
- `taper_window_applied = false`
- exact / userbin / frame-major 표현 규칙
- 저장된 PNG plot 파일명 목록. 선형만 저장하면 8개이고, dB 저장이 활성화되면 대응 `_db.png` 파일까지 함께 포함한다.
- raw / centered variant 별 scalar summary

## 6. plot writer 프로세스 규칙

PNG 저장은 main process 밖 plot writer process 로 처리한다. 구현은 아래를 만족해야 한다.

1. main process 는 수치 payload 만 준비하고 Matplotlib 렌더링은 writer 로 넘긴다.
2. plot writer 는 line plot, heatmap plot, bar plot 저장을 비동기로 처리한다.
3. `psd_analysis` 는 학습 loop 안에서 selected epoch hidden/output signal PSD bundle, grouped hidden-layer block PSD bundle, selected-epoch attenuation 통계 plot, selected-epoch hidden-layer `w_plot`, selected-epoch 시간영역 plot 을 직접 queue 에 enqueue 하지 않고 deferred numeric payload 를 process-local CPU 메모리에만 홀드해야 한다. 입력 probe reference bundle 저장은 `dataset_psd` 가 담당한다.
4. `psd_analysis` 는 학습 완료 후 별도 render pass 가 deferred payload 를 queue 에 enqueue 하고, tqdm 진행 로그를 남기며 성공한 payload 를 즉시 메모리에서 제거해야 한다.
5. `dataset_psd` 의 train/test 전체 입력 PSD 와 `probe_set_reference/` 저장, 그리고 `psd_analysis` 의 학습 완료 후 최종 accuracy / derivative / final attenuation plot 은 기존처럼 plot writer queue 를 직접 사용할 수 있다.
6. worker 수, queue 최대 길이, DPI, existing-file skip, start method 는 환경변수로 조정 가능해야 한다.
7. `--plot_epoch`(alias `--plot_epochs`) 가 주어지면 hidden/output signal PSD bundle payload, grouped hidden-layer block PSD payload, selected-epoch attenuation plot payload, selected-epoch hidden-layer `w_plot` payload 는 선택된 epoch 에서만 저장해야 한다. 선택되지 않은 epoch 에 대해서는 `epoch_<eeee>/` 디렉터리를 만들지 않는다. `train_test_accuracy.png` 와 `training_complete_stats/attenuation_stats/` 아래의 최종 attenuation 통계 plot 저장은 이 epoch 리스트로 필터링하지 않는다.

지원 환경변수 예시는 아래와 같다.

- `PSD_PLOT_WRITER_WORKERS`
- `PSD_PLOT_QUEUE_MAXSIZE`
- `PSD_PLOT_WRITER_DPI`
- `PSD_PLOT_SKIP_EXISTING`
- `PSD_PLOT_WRITER_START_METHOD`

## 7. dataset_psd 고정 설정

`dataset_psd` 는 main process 밖 별도 process, `spawn`, worker 1개, 작은 queue 를 사용한다. 기본값은 아래와 같다.

- `PSD_PLOT_WRITER_WORKERS=1`
- `PSD_PLOT_QUEUE_MAXSIZE=8`
- `PSD_PLOT_WRITER_START_METHOD=spawn`

## 8. flush 와 종료

모든 저장이 끝난 뒤 main process 는 반드시 아래 순서를 따른다.

1. `psd_analysis` 인 경우 `render_deferred_plot_tasks(...)` 를 먼저 호출한다.
2. `flush_plot_tasks()`
3. `shutdown_plot_worker(wait=True)`

이 순서를 통해 background writer queue 에 남아 있는 waveform, heatmap, spectrogram, bar plot, accuracy plot 저장 요청이 모두 반영되고, process-local 메모리에 홀드된 deferred payload 도 성공적으로 소진되도록 한다.
