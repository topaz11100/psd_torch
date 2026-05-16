# model_training 구현 계약

## 1. 문서 역할

이 문서는 `src/model_training.py` 의 implementation contract 를 정의한다. 이 program 은 일반적인 supervised AI training entrypoint 다. model definition 과 prepared data bundle 을 받아 model 을 train 하고, evaluate 하며, 이후 signal analysis 에 사용할 selected checkpoint 를 저장한다.

이 program 은 model/layer PSD analysis 를 수행하지 않고 figure 를 render 하지 않는다.

## 2. official CLI

parser 는 아래 conceptual argument 를 노출해야 한다. 정확한 순서는 중요하지 않지만 name 은 중요하다.

| argument | required | 의미 |
| --- | --- | --- |
| `--dataset` | yes | canonical dataset token |
| `--prep_root` | yes | prepared dataset 들을 담은 root directory |
| `--model` | yes | 하나 이상의 official model token |
| `--hidden_spec` | model-dependent | dense hidden layout 또는 fixed-model placeholder |
| `--readout_mode` | yes | readout policy token |
| `--epochs` | yes | total training epochs |
| `--batch_size` | yes | supervised training batch size |
| `--lr` | yes | learning rate |
| `--seed` | yes | reproducibility seed |
| `--gpu_index` | optional | single process 사용 시 training 용 CUDA device |
| `--regularization_lambda1` | optional | 입력곡선과 모든 hidden layer 곡선 사이의 shape-distance 항 계수. 0 이상의 실수 |
| `--regularization_lambda2` | optional | 입력-첫 hidden 을 포함한 인접 layer 곡선 사이의 shape-distance 항 계수. 0 이상의 실수 |
| `--regularization_signal` | optional | 규제항에 사용할 단일 hidden-layer curve family. model spec 의 core hidden output family 중 하나이며 한 training case 안에서 하나만 선택한다. |
| `--regularization_curve_space` | optional | PSD 기반 규제항 curve space. `exact` 만 허용 |
| `--regularization_curve_scale` | optional | PSD 기반 규제항 curve scale. `raw` 또는 `db` |
| `--regularization_centering` | optional | PSD 계산 전 time-centering 여부. `raw` 또는 `centered` |
| `--regularization_reducer` | optional | row 축 대표 curve reducer. `mean` 또는 `median` |
| `--anal_epoch_list` | optional | launcher-level `ANAL_EPOCH_LIST` 에서 온 analysis checkpoint epoch list |
| `--checkpoint_root` | yes | selected `.pt` file 을 쓰는 clean directory |
| `--metric_root` | yes | training/evaluation CSV file 을 쓰는 directory |
| `--output_root` | optional | `checkpoint_root` 와 `metric_root` 를 자동 파생할 때 쓰는 parent run root |

금지 CLI:

1. 제거 대상 prepared-data direct-path interface
2. 제거 대상 FFT-length control
3. 제거 대상 scale-filter control
4. 제거 대상 curve-filter control
5. figure-rendering argument

Scenario 또는 multi-slot launcher 는 이 program 을 호출하기 전에 model/GPU 조합을 expand 할 수 있지만, program 자체 책임은 training-only 로 유지한다.

## 3. prepared data resolve

prepared data 는 아래 규칙으로만 resolve 한다.

```text
prepared dataset path = <prep_root>/<dataset>
```

제거 대상 prepared-data direct-path interface 로 단일 bundle 을 직접 지정하는 경로는 공식 구현에서 제거한다.

## 4. `ANAL_EPOCH_LIST` normalization

Launcher-level `ANAL_EPOCH_LIST` 는 Python argument `--anal_epoch_list` 로 mapping 된다.

Normalization rules:

1. list 가 없거나 비어 있으면 `[epochs]` 로 normalize 한다.
2. 값을 integer 로 convert 한다.
3. duplicate 를 제거한다.
4. ascending sort 한다.
5. `1 <= epoch <= epochs` 밖의 값은 reject 한다.
6. normalized epoch 에서만 checkpoint 를 저장한다.

이는 implicit all-epoch analysis schedule 을 피한다는 점에서 이전 selected checkpoint behavior 와 다르다.

## 5. hidden_spec normalization

`--hidden_spec` 은 model family 에 따라 해석한다.

| model family | valid `hidden_spec` | 의미 |
| --- | --- | --- |
| dense SNN | comma-separated positive integer list | hidden layer widths |
| recurrent dense SNN | comma-separated positive integer list | recurrent hidden layer widths |
| fixed CNN-SNN full token | `-`, empty string, or `default` | hidden width 를 사용하지 않음 |
| CNN-dense tail model | comma-separated positive integer list | CNN front-end 뒤 dense tail widths |
| SpikingSSM, Spikformer, SpikGRU | 각 model spec 문서가 정한 값 | model-family adapter contract 를 따름 |

Fixed CNN-SNN full token 에서는 `hidden_spec` 을 `None` 으로 normalize 해야 한다. dataset bundle 의 `default_hidden_sizes` 를 CNN builder 에 다시 주입하면 안 된다. 즉 사용자가 `--hidden_spec -` 를 줬는데 내부에서 `hidden_sizes=bundle.default_hidden_sizes` 로 되살리는 구현은 invalid 다.

## 6. training behavior

program 은 표준 training program 처럼 동작해야 한다.

필수 operation:

1. prepared data bundle 을 resolve 한다.
2. dataset loader 를 build 한다.
3. model token 과 model config 로 model 을 build 한다.
4. readout module 을 build 한다.
5. project-side training 용 Adam optimizer 를 build 한다.
6. training epoch 를 실행한다.
7. 기존 project rule 에 따라 validation 및/또는 test evaluation 을 실행한다.
8. training 과 evaluation metric 을 CSV 로 쓴다.
9. selected `.pt` checkpoint 를 저장한다.

금지 operation:

1. hidden layer 에 대한 PSD signal analysis 실행.
2. analysis artifact 용 layer signal map capture.
3. pairwise PSD distance 계산.
4. analysis artifact 용 filter histogram 계산.
5. figure rendering.
6. `psd_analysis.py` 가 소유하는 analysis CSV artifact 쓰기.

## 7. checkpoint directory policy

`--checkpoint_root` 는 strict analysis input directory 다.

규칙:

1. 성공적인 training 이후 selected `.pt` file 만 포함해야 한다.
2. `epochs > 0` 으로 training 하는 경우 최소 하나의 `.pt` file 을 포함해야 한다.
3. log, metric, JSON sidecar, figure, temporary file, lock file, subdirectory 를 포함하면 안 된다.
4. temporary checkpoint write 는 clean directory 안에 non-`.pt` file 을 남기지 않는 external temporary directory 또는 atomic rename strategy 를 사용해야 한다.
5. checkpoint write 가 fail 하면 program exit 전에 partial file 을 제거해야 한다.

권장 naming:

```text
epoch_000001.pt
epoch_000017.pt
epoch_000100.pt
```

zero padding width 는 `epochs` 에서 파생할 수 있지만 lexical sorting 이 epoch sorting 과 일치해야 한다.

## 8. checkpoint payload schema

각 selected checkpoint `.pt` 는 `torch.load` 로 load 가능해야 하며 top-level mapping 을 포함해야 한다.

최소 필수 field:

| field | type | 의미 |
| --- | --- | --- |
| `schema_version` | string | checkpoint schema version, 예: `psd_checkpoint_v1` |
| `epoch` | int | completed epoch index |
| `model_token` | string | canonical model token |
| `model_config` | mapping | reconstruction 에 필요한 architecture parameter |
| `state_dict` | mapping | model weight |
| `readout_config` | mapping | readout construction metadata |
| `dataset_token` | string | training 에 사용한 dataset token |
| `prep_root` | string | training 에 사용한 prepared root |
| `prepared_dataset_path` | string | `<prep_root>/<dataset_token>` 로 resolve 된 path |
| `axis_metadata_ref` | mapping | logical PSD view 구성에 필요한 metadata |
| `seed` | int | primary seed |
| `training_args` | mapping | normalized training CLI argument |
| `normalization_metadata` | mapping | input normalization 과 dtype semantics |
| `hidden_spec_normalized` | string or null | model family 에 맞게 normalize 된 hidden spec |

Optional field:

| field | 조건 |
| --- | --- |
| `optimizer_state_dict` | resume workflow 에만 허용, analysis 에 필요하지 않음 |
| `scheduler_state_dict` | resume workflow 에만 허용, analysis 에 필요하지 않음 |
| `metric_snapshot` | checkpoint epoch 의 latest scalar metric |
| `git_or_code_state` | reproducibility metadata |

`psd_analysis.py` 는 optimizer 또는 scheduler field 를 요구하면 안 된다.

## 9. metric CSV

Training 과 evaluation metric 은 `Spec/impl/spec/csv_schema.md` 의 `training_metric` category 를 사용하여 `--checkpoint_root` 밖에 쓴다.

권장 file:

```text
<metric_root>/training_metrics.csv
```

필수 semantic field:

| field | value |
| --- | --- |
| `category` | `training_metric` |
| `source_program` | `model_training` |
| `epoch` | epoch index |
| `scope` | `train`, `validation`, or `test` |
| `metric` | 예: `loss`, `accuracy`, `correct`, `total` |
| `value` | numeric value |
| `model_token` | model token |
| `dataset` | dataset token |
| `seed` | seed |

## 10. training-side regularization boundary

이 명세는 checkpoint analysis 의 curve extractor output 을 줄이는 argument 를 training 또는 analysis feature 로 사용하지 않는다. Training-time PSD curve-shape regularization 이 별도 실험 축으로 추가되는 경우에도 그 선택은 training-side regularization argument 로 정의해야 하며, checkpoint analysis output filtering 으로 구현하면 안 된다.

규칙:

1. regularization 은 training loss term 이며 model/layer analysis artifact 를 만들지 않는다.
2. regularization 은 checkpoint-analysis entrypoint 를 호출하지 않는다.
3. regularization metric 이 기록되는 경우 `category = training_metric` 을 사용한다.
4. analysis stage 는 regularization 선택과 무관하게 명세에 정의된 모든 curve extractor CSV output 을 모두 보존한다.

## 11. process 와 failure behavior

1. checkpoint write failure 는 training run 을 fail 한다.
2. metric CSV write failure 는 training run 을 fail 한다.
3. program 은 다음 training batch 또는 epoch 로 진행하기 전에 별도 analysis worker 를 기다리면 안 된다.
4. program 은 data loading 에 multiprocessing 을 사용할 수 있지만, queue response 는 training/evaluation work 범위에만 한정해야 한다.
5. training completion 은 signal analysis 가 수행되었음을 의미하지 않는다.

## 12. 구현된 training-side regularization argument

`model_training` 이 인정하는 training-side regularization 은 `Spec/theory/psd_analysis/signal.md` 의 $d_{\mathrm{shape}}(u,v)$ 로 정의되는 단일 PSD curve-shape regularization 뿐이다. 별도 regularization mode field 는 두지 않는다.

총 loss 는 아래처럼 계산한다.

```text
L_total = L_task + L_reg
```

$H$ 를 hidden layer 수라고 하고, $C_0$ 를 training minibatch 에서 첫 hidden layer 가 실제로 받는 입력곡선, $C_i$ 를 같은 curve selector 로 만든 hidden layer $i$ 의 곡선이라고 둔다. 인접 pair 집합은 $(C_0,C_1),(C_1,C_2),\dots,(C_{H-1},C_H)$ 이며 output layer 는 포함하지 않는다. 그러면 regularization loss 는 아래 식이다.

$$
L_{\mathrm{reg}}
=
\lambda_1
\sum_{i=1}^{H}
 d_{\mathrm{shape}}(C_0,C_i)
+
\lambda_2
\sum_{i=1}^{H}
 d_{\mathrm{shape}}(C_{i-1},C_i).
$$

규칙:

1. $\lambda_1$ 은 입력곡선과 모든 hidden layer 곡선 사이의 global alignment 항 계수다.
2. $\lambda_2$ 는 입력-첫 hidden 을 포함한 인접 layer 곡선 사이의 local continuity 항 계수다.
3. 두 합 모두 맨 마지막 hidden-output pair 를 포함하지 않는다.
4. 곡선 family 는 `--regularization_signal` 로 받은 단 하나만 사용한다. 한 regularization case 안에서 여러 hidden-output family 를 섞거나 여러 regularizer 를 병렬/직렬로 누적하지 않는다.
5. `--regularization_curve_space`, `--regularization_curve_scale`, `--regularization_centering`, `--regularization_reducer` 는 $C_i$ 를 만드는 단일 curve selector 다.
6. $\lambda_1 = 0$ 이고 $\lambda_2 = 0$ 이면 regularization branch 는 계산하지 않는다.

PSD 기반 규제항은 training forward trace 로부터 in-memory 로만 계산한다. 이 계산은 model/layer analysis artifact, pairwise distance CSV, plot PNG 를 만들면 안 된다.

`--regularization_curve_space userbin` 과 `--regularization_userbin_edges` 는 제거되었으며 사용하지 않는다.

Training metric CSV 는 최소한 `loss`, `task_loss`, `regularization_loss`, `regularization_global_loss`, `regularization_adjacent_loss`, `accuracy`, `correct`, `total` 을 train scope 에 기록할 수 있어야 한다.

### 12.1 bash `REGULARIZATION_SET` grammar

`bash/model_training.sh` 의 `REGULARIZATION_SET_RAW` 원소 문법은 아래 7-field pipe 형식이다.

```text
<lambda1>|<lambda2>|<signal>|<curve_space>|<curve_scale>|<centering>|<reducer>
```


기본값은 규제항을 끄는 아래 원소로 둔다.

```text
0|0|y_mem|exact|raw|raw|mean
```

사용 예시는 아래와 같다.

```text
REGULARIZATION_SET="1e-4|0|y_mem|exact|raw|raw|mean 1e-4|1e-5|y_spike|exact|db|centered|median"
```

각 원소는 하나의 training case 를 만든다. 여러 원소를 주는 것은 여러 training case sweep 을 뜻하며, 한 case 안에서 여러 regularizer 를 합성한다는 뜻이 아니다.

## 2026-05-02 수정 고정

- training-side PSD regularization 의 `curve_space` 는 `exact` 만 허용한다. `userbin` 및 `regularization_userbin_edges` 는 제거한다.
- regularization 기준 input curve 는 `ForwardResult.input_record` 로부터 만든다.
- trainable threshold 는 기존 parameter 이름 `v_threshold_param` 을 유지하면서 forward 에서 양수 effective threshold 로 사용한다.
- `spikegru` 는 내부적으로 `spikegru_max_over_time` readout 을 사용한다.
