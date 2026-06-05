# Timestamped output directory policy

반복 실행 결과가 같은 root에 덮어써지지 않도록, 결과를 쓰는 entrypoint는 기본적으로 실제 산출물 경로에 실행시각 폴더를 한 단계 추가한다.

## 고정 정책

- 기본값은 `timestamped_output=true`이다.
- `run_timestamp`가 지정되지 않으면 Asia/Seoul 현재시각을 `YYYYmmdd_HHMMSS_microseconds` 형식으로 생성한다.
- 폴더명은 기본적으로 `run_<timestamp>` 형식이다.
- `model_training`은 DDP 실행 시 rank 0이 생성한 timestamp를 모든 rank에 broadcast하여 checkpoint와 metric 경로가 동일한 run id를 공유한다.
- `timestamped_output=false`를 명시하면 기존처럼 지정된 root에 직접 쓴다. 이 옵션은 과거 경로 호환 또는 테스트용이다.

## 적용 대상

다음 entrypoint는 `--run_timestamp`와 `--timestamped_output`을 지원한다.

```text
model_training.py
checkpoint_accuracy_analysis.py
checkpoint_accuracy_eval_plot.py
dataset_psd.py
dataset_fft.py
psd_analysis.py
2d_fft_analysis.py
element_psd.py
element_fft.py
plotting.py
```

`data_prep.py`는 prepared dataset bundle을 만드는 단계이므로 이 정책의 대상이 아니다. 기존 `force_overwrite` 계약으로 관리한다.

## 경로 예시

일반 output root:

```text
--output_root /home/yongokhan/workspace/result/base/psd_analysis
→ /home/yongokhan/workspace/result/base/psd_analysis/psd_analysis_20260530_181234_123456
```

학습 checkpoint/metric root:

```text
--checkpoint_root /home/yongokhan/workspace/result/model_training/checkpoints
--metric_root     /home/yongokhan/workspace/result/model_training/metrics

→ /home/yongokhan/workspace/result/model_training/run_20260530_181234_123456/checkpoints
→ /home/yongokhan/workspace/result/model_training/run_20260530_181234_123456/metrics
```

leaf name이 `checkpoints`, `metrics`, `train`이면 timestamp folder를 leaf 위에 삽입하여 같은 run의 관련 폴더들이 한 parent를 공유하게 한다.

## 출력 metadata

각 entrypoint는 최종 stdout status JSON line에 실제 `output_root` 또는 `checkpoint_root`/`metric_root`를 출력한다. `model_training` checkpoint payload와 `training_args`에는 `run_timestamp`, `timestamped_output`, `checkpoint_root`, `metric_root`를 기록한다.
