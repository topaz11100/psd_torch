# PSD analysis implementation specification

## 1. 범위

이 문서는 `src/util/psd_analysis_driver.py` 중심의 현재 실험 dataset 공통 `psd_analysis` 구현 규칙을 정의한다. 사용 dataset 범위와 dataset 별 preprocessing 세부는 `paper/proposed/data_preprocessing.md` 를 따른다. 구현은 `paper/proposed/psd_analysis.md` 의 저장 정의를 코드 레벨에서 그대로 만족해야 한다.

핵심 구현 요구는 아래와 같다.

1. hidden layer 와 output layer 의 membrane / spike bundle 은 기본적으로 매 epoch 저장하되, `--plot_epoch`(alias `--plot_epochs`) 가 주어지면 지정한 epoch 에서만 epoch 디렉터리를 만들고 저장한다. 각 selected epoch 의 각 layer / family 디렉터리에는 `time_domain_heatmap.png`, `time_domain_element_mean.png` 를 함께 저장한다. 단 학습 loop 안에서는 PNG 를 직접 그리지 않고 plot 생성용 numeric payload 를 process-local CPU 메모리에 적재하며, 학습 완료 후 최종 PNG 를 렌더링한다.
2. waveform 은 **window 없는 exact simple periodogram** 이고 raw / centered 를 둘 다 저장한다.
3. periodogram heatmap 은 **exact periodogram 에 userbin 을 집계한 raw / centered 두 버전** 을 저장한다.
4. mean spectrogram 은 **window 없는 exact sliding simple periodogram** 이고 raw / centered 를 둘 다 저장한다.
5. spectrogram heatmap 은 **exact spectrogram 에 userbin 을 집계한 raw / centered 두 버전** 을 저장한다.
6. exact periodogram / exact spectrogram 공식 경로에는 taper window 를 적용하지 않는다.
7. accuracy 는 epoch 마다 CSV 에 누적 기록하고, 최종 plot 은 그 CSV 를 다시 읽어 생성한다.
8. 선택된 epoch root 에는 `probe_set_accuracy.txt`, `attenuation_stats/`, `all_layers_summary.csv`, hidden-layer `w_plot.png` 를 저장하고, grouped(`clip`, `structure`, `structclip`) hidden layer 에 대해서는 같은 hidden layer 아래 block PSD bundle / block weight plot 을 추가 저장한다. 학습 완료 시 최종 probe-set accuracy 를 `training_complete_stats/probe_set_accuracy/` 아래 별도로 저장한다.
9. output layer 는 실제 neuron layer 이며 output neuron 뒤에 learned NN head 를 두지 않는다.
10. curve-shape semi-metric 은 선택된 epoch hidden / output bundle 과 probe input reference 의 같은 family mean plot 사이에서만 추적하고, 저장 디렉터리 이름은 `shape_sim_metric/` 다.
11. curve-shape semi-metric 대상은 `element_*` 가 아니라 mean plot 8개뿐이며, output layer 도 반드시 포함한다.
12. `epochs = 0` 이면 학습과 selected-epoch artifact 생성은 수행하지 않는다.
13. freq_ 실험 모듈에 의존하지 않는다.

## 2. 모듈 구성

현재 구현은 아래 모듈 조합을 사용한다.

- 엔트리: `src/psd_analysis.py`
- 메인 드라이버: `src/util/psd_analysis_driver.py`
- dataset registry / preprocessing adapter: `src/data/registry.py` 와 각 dataset adapter(`src/data/*.py`)
- PSD bundle 생성 / 저장: `src/signal/psd_artifacts.py`
- 학습 loop 분리: `src/model/psd_training.py`
- PSD / spectrogram 수치 연산: `src/signal/psd_utils.py`
- readout 규칙: `paper/proposed/readout.md` 와 `src/readout/readout.py`
- first_spike timing loss wrapper: `src/model/first_spike_loss.py`
- output neuron layer 조립: `src/model/snn_builder.py`

예전 freq_ 실험 모듈 의존성은 남기지 않는다.

추가로 `src/util/psd_analysis_driver.py` 는 `plot_epochs` 정규화 helper 를 가져야 하며, 중복 epoch 는 dedupe 하고 모든 값이 `1 <= epoch <= epochs` 범위 안인지 검증해야 한다. 생략 시 effective list 는 `1..epochs` 전체다.

## 3. entry CLI 규칙

`src/psd_analysis.py` 는 현재 CLI 와 일치해야 한다.

- `--dataset` 은 단일 인수다. 유효한 canonical dataset token 집합은 현재 실험 범위와 동일하며 `paper/proposed/data_preprocessing.md` 를 따른다.
- `--model` 은 하나 이상 받을 수 있다.
- clip / structure PSD variant token 은 `rf`, `rf_struct`, `rf_clip`, `rf_structclip`, `lif`, `lif_struct`, `lif_clip`, `lif_structclip` 이다.
- 그 외 일반 neuron model 은 `src/model/model_registry.py` 의 alias 를 따르며, 공식 범위에는 `tc`, `ts`, `dh_snn`, `d_rf`, `my_dh_snn`, `my_r_dh_snn`, `my_d_rf` 와 그 canonical alias 가 포함된다.
- recurrent suffix `_R` 는 `lif`, `rf`, `tc_lif`, `ts_lif`, `dh_snn` 과 그 canonical alias 에만 허용한다.
- 고정 branch 수를 갖는 dendritic model 은 `dh_snn_2`, `d_rf_4` 처럼 `_정수` suffix 를 받을 수 있고, recurrent 고정-branch canonical 표기는 `dh_snn_R_4` 처럼 `_R_<정수>` 를 쓴다.
- `lif` / `rf` 의 recurrent variant 는 `clip`, `structure`, `structclip` 시나리오에서도 같은 clip interval 과 structure mask 규칙을 그대로 적용해야 한다.
- `my_*` variable-branch model 은 `_정수` suffix 를 쓰지 않고 `S_min`, `S_max` 로 범위를 지정한다.
- 여러 model token 을 주면 받은 순서대로 직렬 실행한다.
- 허용 mode 이름은 `final_membrane`, `first_spike`, `max_rate` 셋뿐이다.
- 여러 readout mode 를 주면 받은 순서대로 직렬 실행한다.
- `--model`, `--readout_mode` 에 여러 값을 주면 같은 dataset 안에서 model 바깥 루프, readout 안쪽 루프로 조합별 직렬 실행한다.
- grouped model/readout 병렬 시나리오 확장은 `bash/run_psd.sh` 가 담당한다.
- `final_membrane` 는 마지막 membrane 값을 raw pre-softmax score 로 사용하고, 이 mode 에서만 output layer 의 spike emission 과 spike-triggered reset path 를 끈다.
- `max_rate` 는 output spike sequence 평균 firing-rate 를 raw score 로 사용한다.
- `first_spike` 는 output spike / membrane record 를 사용한 released First-spike timing loss 를 공식 supervised path 로 사용한다.
- `--plot_epoch`(alias `--plot_epochs`) 는 hidden / output signal PSD bundle 저장 epoch 를 정하는 정수 리스트 인수다. 생략하면 전체 epoch 저장, 지정하면 명시된 epoch 에서만 epoch 디렉터리와 signal PSD bundle 최종 PNG / `summary.json` / epoch probe-set accuracy / epoch attenuation 통계 / hidden-layer weight visualization 을 저장한다. 구현은 학습 중 numeric payload 만 저장하고, 학습 완료 후 최종 PNG 를 렌더링해야 한다. grouped(`clip`, `structure`, `structclip`) hidden layer 에 대해서는 같은 selected epoch root 의 hidden layer 아래 block PSD bundle / block weight plot 을 추가 저장한다. 학습 완료 시 `training_complete_stats/` 아래의 최종 accuracy plot, 최종 probe-set accuracy, 최종 attenuation 통계, 최종 curve-shape semi-metric 요약은 이 인수와 독립적으로 저장하고, 해당 요약 저장 루트는 `shape_sim_metric/` 다.
- probe set selection 은 split + seed 로 만든 label별 canonical order 를 공유하고, `same_label` / `balanced_global` 은 각자 requested count 만큼 같은 canonical order 의 prefix 를 취해야 한다.
- `--psd_window`, `--psd_overlap` 은 spectrogram frame 길이 / overlap 을 정하는 인수다. waveform 은 항상 full-length exact periodogram 이다.
- `--window_fn` 은 **legacy compatibility 인수** 다. 현재 공식 exact periodogram / exact spectrogram 경로에서는 무시되며, taper window 를 적용해서는 안 된다.
- `--userbin_edges` 는 periodogram / spectrogram heatmap 표현용 userbin 경계를 정한다. waveform 과 mean spectrogram 은 exact bin 으로 유지한다.
- RF 는 exact ZOH 만 지원하므로 Euler 관련 인수는 노출하지 않는다.

## 4. common PSD bundle payload 규격

`src/signal/psd_artifacts.py` 의 `combined_exact_psd_payload_from_maps_torch(...)` 는 입력 `(S,R,T)` maps 에 대해 아래 payload 를 만든다.

- `exact_freqs`
- `spectrogram_freqs`
- `spectrogram_frame_centers`
- `set_mean_psd_exact_raw`
- `set_mean_psd_exact_centered`
- `set_mean_heatmap_user_raw`
- `set_mean_heatmap_user_centered`
- `set_mean_spectrogram_exact_raw`
- `set_mean_spectrogram_exact_centered`
- `set_mean_spectrogram_user_raw`
- `set_mean_spectrogram_user_centered`
- `periodogram_length`
- `spectrogram_window_length`
- `spectrogram_overlap_length`
- `num_samples`
- `num_rows`
- `variants_saved`
- `taper_window_applied`
- `legacy_window_fn_ignored`

여기서 의미는 다음과 같다.

- `set_mean_psd_exact_<variant>`: sample 평균 후 row 평균한 exact periodogram waveform
- `set_mean_heatmap_user_<variant>`: sample 평균 후 row 유지한 periodogram userbin heatmap
- `set_mean_spectrogram_exact_<variant>`: sample 평균 후 row 평균한 exact mean spectrogram
- `set_mean_spectrogram_user_<variant>`: sample 평균 후 row 유지한 spectrogram userbin tensor, shape = `(rows, bands, frames)`
- `<variant>` 는 `raw`, `centered` 두 값만 허용한다.

주파수 관련 payload (`exact_freqs`, `spectrogram_freqs`) 의 단위는 모두 Nyquist 상한이 0.5 인 cycle/sample 이다. `spectrogram_frame_centers` 는 spectrogram frame center time-step index 다.

`merge_exact_psd_payloads(...)` 는 `num_samples` 가중 평균으로 여러 batch payload 를 병합해야 하며, raw / centered 네 계열을 빠짐없이 모두 병합해야 한다.

## 5. bundle 저장 규칙

`psd_analysis` 경로에서 `save_psd_bundle(..., save_db_plots=True)` 는 아래 16개 PNG 와 선택적 `summary.json` 을 저장해야 한다.

1. `mean_psd_waveform_exact_raw.png`
2. `mean_psd_waveform_exact_centered.png`
3. `element_psd_heatmap_userbin_raw.png`
4. `element_psd_heatmap_userbin_centered.png`
5. `mean_spectrogram_exact_raw.png`
6. `mean_spectrogram_exact_centered.png`
7. `element_spectrogram_heatmap_userbin_raw.png`
8. `element_spectrogram_heatmap_userbin_centered.png`
9. `mean_psd_waveform_exact_raw_db.png`
10. `mean_psd_waveform_exact_centered_db.png`
11. `element_psd_heatmap_userbin_raw_db.png`
12. `element_psd_heatmap_userbin_centered_db.png`
13. `mean_spectrogram_exact_raw_db.png`
14. `mean_spectrogram_exact_centered_db.png`
15. `element_spectrogram_heatmap_userbin_raw_db.png`
16. `element_spectrogram_heatmap_userbin_centered_db.png`

세부 규칙은 다음과 같다.

- waveform 과 mean spectrogram 은 exact bin 축으로 저장한다.
- exact waveform / exact spectrogram 공식 경로에는 taper window 를 적용하지 않는다.
- PSD heatmap 은 userbin 주파수축을 갖고 **모든 칸에 수치 annotation** 을 넣는다.
- mean spectrogram 은 exact frequency bin 축을 갖고 annotation 을 넣지 않는다.
- spectrogram userbin heatmap 은 `(rows, bands, frames)` 를 frame-major 순서의 2차원 heatmap 으로 펼친다.
- spectrogram userbin heatmap 은 annotation 을 넣지 않는다.
- dB plot 은 선형 power-like 값 $x$ 에 대해 $10 \log_{10}(x + 10^{-12})$ 를 적용한 별도 PNG 다.
- row index 의 낮은 값이 아래쪽에 오도록 `origin="lower"` 를 사용한다.
- `save_summary_json=True` 이면 같은 디렉터리에 `summary.json` 을 저장한다.
- `summary.json` 은 raw / centered variant 별 scalar summary, 공통 metadata, plot file list, `taper_window_applied=false`, `variants_saved=["raw", "centered"]`, dB plot 저장 여부와 epsilon 을 포함해야 한다.

현재 공식 구현에서는 `psd_analysis` 와 `dataset_psd` 모두 `save_summary_json=True` 를 사용한다.

## 6. probe set reference 저장

`_probe_reference_payloads_for_split(...)` 는 deterministic probe batch 에 대한 입력 reference PSD payload 를 계산한다. `psd_analysis` 에서는 curve-shape semi-metric 내부 reference 로만 쓰고, 영구 저장 경로는 `dataset_psd` 가 담당한다.

이 경로는 입력 reference 전용이므로 model prediction 기반 `probe_set_accuracy.txt` 는 두지 않는다.

## 7. epoch 분석 저장

`_save_epoch_analysis(...)` 는 선택된 epoch 에 대해서만 epoch root 아래 probe-set accuracy text 와 signal PSD bundle 을 저장한다. 구현은 학습 중 최종 PNG 를 만들지 않고 PSD bundle 렌더링에 필요한 numeric payload 만 저장해야 한다. hidden layer 전체 PSD bundle 은 유지하되, grouped(`clip`, `structure`, `structclip`) hidden layer 에 대해서는 같은 hidden layer 아래 `block/block_k/<family>/` 경로로 block PSD bundle 을 추가 저장한다. 따라서 `--plot_epoch` 가 주어지면 선택되지 않은 epoch 에 대해서는 `epoch_<eeee>/` 디렉터리 자체를 만들지 않는다. 학습 완료 시 최종 probe-set accuracy snapshot 은 별도 `training_complete_stats/probe_set_accuracy/` 아래 저장한다.

- hidden layer membrane
- hidden layer spike
- output layer membrane
- output layer spike

저장 경로는

```text
epoch_<eeee>/<split>/<scope>/<layer_name>/<family>/
epoch_<eeee>/<split>/<scope>/<layer_name>/block/block_<k>/<family>/
```

형식을 따른다. 여기서 `family` 는 `membrane` 또는 `spike` 다. block 저장은 grouped(`clip`, `structure`, `structclip`) hidden layer 에만 적용하고 output layer 에는 block 하위 경로를 만들지 않는다.

probe-set accuracy 예측은 공식 supervised path 와 일치해야 한다. 즉 criterion 이 `requires_output_record = true` 인 경우에는 `logits.argmax(...)` 를 쓰지 말고 `criterion.analyze_output_record(...)` 와 `criterion.predictions_from_analysis(...)` 를 사용해야 한다. 현재 `first_spike` 가 여기에 해당한다.

선택된 epoch signal PSD payload 저장 시 curve-shape tracker 가 켜져 있으면 probe input reference payload 와 현재 hidden / output payload 를 비교해 curve-shape semi-metric bucket 을 즉시 갱신해야 한다. 이때 tracked key 는 mean plot 8개뿐이며, `element_*` heatmap 은 제외한다.

선택된 epoch 에 대해서는 각 고정 probe set root 에 아래 파일을 저장한다.

```text
epoch_<eeee>/<split>/<scope>/probe_set_accuracy.txt
```

학습 완료 시에는 최종 model 기준 probe-set accuracy 를 아래 경로에 추가 저장한다.

```text
training_complete_stats/probe_set_accuracy/<split>/<scope>/probe_set_accuracy.txt
```

이 텍스트 파일에는 최소 아래 필드를 사람이 읽을 수 있는 형식으로 기록한다.

- `epoch`
- `split`
- `probe_type`
- `label` (`same_label` 일 때만 값이 있고, 아니면 `none`)
- `correct`
- `total`
- `accuracy`

## 8. hidden-layer weight visualization 저장

선택된 epoch 에 대해서는 hidden layer 별 incoming-weight density plot(`w_plot.png`) 을 epoch root 바로 아래 저장한다. 이 plot 역시 학습 중에는 deferred plot payload 로만 기록하고, 학습 완료 후 렌더링한다. output layer 에 대해서는 이 시각화를 만들지 않는다. 저장 경로는 아래와 같다.

```text
epoch_<eeee>/<hidden_layer_name>/w_plot.png
epoch_<eeee>/<hidden_layer_name>/block/block_<k>/w_plot.png
```

규칙은 다음과 같다.

- `w_plot.png` 는 해당 hidden layer 로 들어오는 가중치 전체의 분포를 면적 1 density curve 로 시각화한다.
- block weight plot(`w_plot.png`) 은 grouped(`clip`, `structure`, `structclip`) hidden layer 에만 추가 저장한다.

## 9. accuracy 저장 구현

accuracy 저장은 아래 순서를 따른다.

1. epoch 종료 직후 `train_test_accuracy.csv` 에 row append
2. run 종료 시 최종 row 집합을 다시 CSV 로 재기록
3. 그 CSV 를 읽어 `train_test_accuracy.png` 생성
4. flush 이후 `training_complete_stats/train_test_accuracy.csv` 와 `training_complete_stats/train_test_accuracy.png` 복사본 저장

`train_test_accuracy.png` 저장은 `--plot_epoch` 와 무관하게 항상 run 종료 시 수행해야 한다.

즉 plot 이 직접 메모리 history 를 그리는 것이 아니라, 저장된 CSV 를 단일 소스로 사용해야 한다.

`probe_set_accuracy.txt` 계산도 같은 원칙을 따른다. `first_spike` 처럼 output-record criterion 을 쓰는 경우 train/test accuracy 와 probe accuracy 가 서로 다른 판정 규칙을 쓰면 안 된다.

## 10. 필터 통계 저장

감쇠 / 공명 관련 통계의 공식 구현 범위는 선택된 epoch attenuation snapshot 과 학습 완료 시점의 최종 attenuation snapshot 둘 다다. 저장 경로는 아래와 같다. selected epoch attenuation 저장은 `--plot_epoch` 로 선택된 epoch 에 대해서만 수행하고, 최종 snapshot 은 이 선택과 무관하게 항상 존재해야 한다. 선택 epoch 의 `summary_stats.csv` 와 `all_layers_summary.csv` 는 즉시 기록하되, 대응 bar / histogram PNG 는 학습 완료 후 deferred render pass 에서 process-local CPU 메모리에 홀드된 payload 로 생성할 수 있다.

- `epoch_<eeee>/attenuation_stats/layers/<layer_name>/summary_stats.csv`
- `epoch_<eeee>/attenuation_stats/layers/<layer_name>/*_stats_bar.png`
- `epoch_<eeee>/attenuation_stats/layers/<layer_name>/*_value_hist_bar.png`
- `epoch_<eeee>/attenuation_stats/model/summary_stats.csv`
- `epoch_<eeee>/attenuation_stats/model/*_stats_bar.png`
- `epoch_<eeee>/attenuation_stats/model/*_value_hist_bar.png`
- `epoch_<eeee>/all_layers_summary.csv`
- `training_complete_stats/attenuation_stats/layers/<layer_name>/summary_stats.csv`
- `training_complete_stats/attenuation_stats/layers/<layer_name>/*_stats_bar.png`
- `training_complete_stats/attenuation_stats/layers/<layer_name>/*_value_hist_bar.png`
- `training_complete_stats/attenuation_stats/model/summary_stats.csv`
- `training_complete_stats/attenuation_stats/model/*_stats_bar.png`
- `training_complete_stats/attenuation_stats/model/*_value_hist_bar.png`
- `training_complete_stats/all_layers_summary.csv`

구현은 backward compatibility 용으로 `training_complete_stats/attenuation_stats/all_layers_summary.csv` 복사본을 함께 둘 수 있다. RF 계열 저장 키는 raw parameter 가 아니라 `rho`, `f_cyc_per_sample` 이어야 한다. LIF 계열은 별도 `v_reset` 인터페이스 없이 $v_{\mathrm{th}}$ 기반 subtractive soft reset 을 사용한다. `*_value_hist_bar.png` 는 x축이 parameter value, y축이 neuron count 인 histogram bar plot 이어야 한다. `paper/proposed/filter_analysis.md` 의 현재 binding scope 도 이 selected-epoch + final attenuation 저장 규칙이다.

## 11. plot writer 규칙

모든 PNG 저장은 `src/plot/plotting.py` 의 plot writer 인터페이스를 사용한다. `psd_analysis` 는 두 단계 전략을 따른다. 학습 중에는 selected epoch signal PSD bundle, probe reference bundle, selected epoch attenuation PNG, hidden-layer `w_plot.png` 를 직접 렌더링하지 않고 `src/plot/deferred_plot_tasks.py` 를 통해 numeric payload 를 process-local CPU 메모리에만 적재한다. 학습 완료 후 `render_deferred_plot_tasks(...)` 가 tqdm 진행 로그를 남기며 payload 를 하나씩 렌더링하고, 성공적으로 PNG 저장이 끝난 payload 는 즉시 메모리에서 제거한다. run 종료 시에는 아래를 만족해야 한다.

- `psd_analysis` 는 deferred render pass 를 먼저 실행한 뒤 `flush_plot_tasks()` 와 `shutdown_plot_worker(wait=True)` 를 호출한다.
- `dataset_psd` 는 학습 loop 가 없으므로 전체 split 입력 PSD 와 probe-set reference 저장 모두 direct plot writer queue 경로를 사용하고 마지막에 `flush_plot_tasks()` 와 `shutdown_plot_worker(wait=True)` 를 호출한다.
- 환경변수 `PSD_PLOT_WRITER_WORKERS`, `PSD_PLOT_QUEUE_MAXSIZE`, `PSD_PLOT_WRITER_DPI`, `PSD_PLOT_SKIP_EXISTING`, `PSD_PLOT_WRITER_START_METHOD` 를 지원한다.

## 12. dataset_psd 와의 공통화

`dataset_psd` 는 학습을 하지 않지만, split 입력 reference 저장에는 동일한 `combined_exact_psd_payload_from_maps_torch(...)` 와 `save_psd_bundle(..., save_db_plots=True)` 를 재사용한다. 즉 두 실험의 PSD / spectrogram 정의가 달라지면 안 된다. 다만 `dataset_psd` 는 training loop 가 없으므로 deferred render 없이 직접 async plot writer 를 사용해도 된다.
