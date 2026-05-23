# psd_analysis 구현 계약

## 1. 문서 역할

이 문서는 training-analysis-plotting split 이후 `src/psd_analysis.py` 의 implementation contract 를 정의한다.

`src/psd_analysis.py` 는 `src/model_training.py` 가 이미 생성한 checkpoint 에 대해 signal analysis 와 CSV production 을 수행한다. model 을 train 하지 않고 figure 를 render 하지 않는다.

## 2. official CLI

| argument | required | 의미 |
| --- | --- | --- |
| `--checkpoint` | yes | 단일 `.pt` file 또는 `.pt` file 만 포함하는 strict directory 하나 |
| `--dataset` | yes | 훈련에 사용한 canonical dataset token |
| `--prep_root` | yes | prepared dataset 들을 담은 root directory |
| `--output_root` | yes | analysis CSV output 용 root directory |
| `--anal_batch` | yes | 한 analysis forward pass 에서 GPU 로 보내는 sample 수의 최대값 |
| `--gpu_index` | yes | analysis 용 CUDA device index |
| `--enable_pairwise_dependency_appendix` | optional | optional appendix metric 활성화 |
| `--analysis_distance_metric` | optional | pair/layer shape distance metric. `centered_l2` 또는 `diff_l2` |
| `--seed` | optional | analysis seed, 기본값은 checkpoint seed |
| `--num_workers` | optional | probe loading 용 data-loader worker, plotting worker 아님 |
| `--low_vram` | optional | `1` 이면 trace 를 CPU 로 stage 하여 VRAM 사용량을 줄임 |

금지 CLI:

1. 제거 대상 prepared-data direct-path interface
2. train/test selector
3. 제거 대상 FFT-length control
4. 제거 대상 scale-filter control
5. 제거 대상 curve-filter control
6. figure-rendering argument

`--dataset` 은 checkpoint metadata 의 `dataset_token` 과 일치해야 한다. 일치하지 않으면 forward pass 전에 fail 한다.

## 2.1 exact-only extractor policy

현재 official analysis path 는 `psd_exact` 만 산출한다.

규칙:

1. `psd_userbin` 분석은 수행하지 않는다.
2. `psd_userbin` CSV 는 생성하지 않는다.
3. `--userbin_edges` 는 official CLI 에 포함하지 않는다. Python parser 에 호환성 인수로 남아 있더라도 실행 경로에서는 무시한다.
4. exact PSD 는 normalized frequency grid 를 그대로 사용한다.
5. 별도 FFT 길이 조정 개념은 official analysis path 에 존재하지 않는다.

## 2.2 curve extractor output policy

`psd_analysis.py` 는 `psd_exact` 만 CSV 로 기록한다.

필수 extractor:

| extractor | 의미 |
| --- | --- |
| `psd_exact` | exact one-sided periodogram 기반 representative PSD curve |

Analysis output 에는 value-scale 선택 CLI 가 없어야 한다. `psd_analysis.py` 는 `psd_exact` 에 대해 `raw` 와 `db` value-scale CSV 를 모두 기록해야 한다.

## 3. input path mode

### 3.1 file mode

`--checkpoint` 가 file 을 가리키는 경우:

1. file suffix 는 `.pt` 여야 한다.
2. 해당 checkpoint 하나만 분석한다.
3. output 은 file stem 에서 파생한 checkpoint-specific analysis directory 아래에 쓴다.

### 3.2 strict directory mode

`--checkpoint` 가 directory 를 가리키는 경우:

1. directory 는 suffix 가 `.pt` 인 regular file 을 하나 이상 포함해야 한다.
2. directory 바로 안의 모든 regular file 은 suffix 가 `.pt` 여야 한다.
3. subdirectory 는 invalid 다.
4. hidden non-`.pt` file 은 invalid 다.
5. file 은 checkpoint epoch 의 ascending order 로 분석한다. checkpoint 중 하나라도 epoch metadata 가 없으면 manifest CSV 에 warning row 를 기록한 뒤 lexical filename order 를 사용한다.

Directory mode 는 recursive 가 아니다. recursive traversal 은 checkpoint file 이 아니라 CSV file 에 대해 `plotting.py` 가 담당한다.

## 4. analysis lifecycle

각 checkpoint 에 대해 program 은 아래 sequence 를 수행한다.

```text
load checkpoint
  -> validate CLI dataset against checkpoint dataset metadata
  -> resolve prepared data with prep_root and dataset
  -> reconstruct model
  -> load state_dict
  -> move model to cuda:<gpu_index>
  -> build fixed probe loader for all analysis probe scopes
  -> run probe forward passes with sample count <= anal_batch
  -> capture layer signals
  -> compute PSD-first artifacts
  -> write category-based CSV outputs
```

program 은 다음을 수행하면 안 된다.

1. optimizer 를 step 하지 않는다.
2. checkpoint weight 를 modify 하지 않는다.
3. 새 model checkpoint 를 저장하지 않는다.
4. figure 를 render 하지 않는다.
5. training-time multiprocessing queue 에 의존하지 않는다.
6. checkpoint 에 저장된 optimizer 또는 scheduler state 를 요구하지 않는다.

## 5. scope 처리 정책

`psd_analysis.py` 는 train/test 중 하나를 고르는 train/test selector 를 받지 않는다. 또한 train/test 전체집합을 analysis scope 로 쓰지 않는다. Checkpoint analysis 는 prepared data 에 정의된 probe set 만 분석한다.

현재 구현의 필수 probe scope:

1. `train_balanced_global`
2. `test_balanced_global`

현재 `same_label` 및 `distribution_global` scope 는 checkpoint analysis 산출물로 생성하지 않는다.

금지 scope:

1. prepared train full split 전체집합 scope
2. prepared test full split 전체집합 scope


## 6. GPU 와 batching 의미론

`--anal_batch` 는 한 analysis forward pass 에서 model 을 통과하는 probe sample 수의 upper bound 다.

규칙:

1. `anal_batch >= 1` 이 필요하다.
2. scope sample 수가 `anal_batch` 보다 많으면 여러 analysis microbatch 로 split 한다.
3. microbatch output 은 동일 checkpoint-level CSV artifact 로 accumulate 한다.
4. checkpoint metadata 에 저장된 training batch size 는 `anal_batch` 를 override 하지 않는다.
5. program 은 model allocation 전에 `--gpu_index` 로 CUDA device 를 설정한다.
6. `--gpu_index` 가 요청되었는데 CUDA 를 사용할 수 없으면 execution 은 fail 한다.

## 7. 보존되는 analysis content

필수 category:

| category | 내용 |
| --- | --- |
| `analysis_curve` | 각 checkpoint, scope, layer, signal, series, extractor, reducer, variant, scale 의 representative PSD curve |
| `analysis_dispersion` | 각 checkpoint, scope, layer, signal, series, extractor, variant, scale 의 PSD-domain variance, MAD summary |
| `pair_distance` | 같은 layer, signal, series 에 대한 cross-scope PSD distance summary |
| `layer_distance_profile` | 특정 checkpoint 의 input-reference 및 adjacent layer shape-distance profile |
| `layer_distance_trend` | multiple checkpoint 분석 시 input-reference 및 adjacent layer shape-distance trend |
| `layer_dispersion_profile` | 특정 checkpoint 의 layer-wise variance/MAD scalar profile |
| `layer_dispersion_trend` | multiple checkpoint 분석 시 layer-wise variance/MAD scalar trend |
| `filter_snapshot` | applicable 한 경우 selected checkpoint filter/neuron parameter summary |
| `filter_trend` | multiple checkpoint 분석 시 checkpoint-axis trend |
| `accuracy_loss_join` | 기존 CSV 호환용. 새 산출물에서는 생성하지 않음 |
| `pairwise_dependency_appendix` | 활성화 시 optional representative curve correlation appendix metric |
| `analysis_manifest` | per-checkpoint artifact inventory 와 validation status |

모든 category 는 `Spec/impl/spec/csv_schema.md` 를 따르는 CSV 로 쓴다.

## 8. full curve-combination output rule

각 checkpoint 에서 아래 Cartesian product 를 가능한 범위에서 모두 생성한다. 여기서 `scope` 는 필수 probe scope 만 의미하며 train/test 전체집합 scope 는 포함하지 않는다.

```text
probe_scope
  x layer
  x signal_kind
  x extractor in {psd_exact}
  x reducer in {mean, median}
  x variant in {raw, centered}
  x scale in {raw, db}
```

적용되지 않는 signal 또는 layer 는 manifest 에 reason 을 기록하고 건너뛴다. extractor 또는 scale 을 사용자가 하나로 줄이는 기능은 없다.

Pair distance 는 가능한 extractor, reducer, variant, scale 조합마다 독립 CSV 로 저장한다. 레이어 간 curve distance 는 `layer_distance_profile` 과 `layer_distance_trend` 로 저장한다. 레이어 내부 dispersion 은 `layer_dispersion_profile` 과 `layer_dispersion_trend` 로 저장한다.

## 9. output path contract

권장 layout:

```text
<output_root>/
  analysis_manifest.csv
  checkpoint_epoch_000001/
    layers/
      layer_001__layer_01/
        analysis_curve/
        analysis_dispersion/
        filter_snapshot/
    layer_distance_profile/
      input_reference/
      adjacent/
    layer_dispersion_profile/
      variance/
      mad/
  checkpoint_epoch_000017/
    ...
  traces/
    layer_distance_trend/
      input_reference/
      adjacent/
    layer_dispersion_trend/
      variance/
      mad/
    filter_trend/
```

규칙:

1. 단일 `family_psd_curve.csv` 에 모든 곡선을 몰아넣지 않는다.
2. 각 category 와 좌표 조합마다 독립 CSV 를 쓴다.
3. CSV file name 은 stable lowercase snake case 다.
4. 이 program 은 figure file 을 쓰지 않는다.
5. 이 program 은 binary analysis bundle 을 쓰지 않는다.

## 10. checkpoint reconstruction

analysis program 은 checkpoint metadata 에서 model 을 reconstruct 한다.

규칙:

1. model token registry 는 `model_token` 을 resolve 해야 한다.
2. architecture parameter 는 `model_config` 에서 와야 한다.
3. readout construction 은 `readout_config` 에서 와야 한다.
4. axis metadata 는 checkpoint metadata 또는 prepared data manifest 에서 와야 한다.
5. required metadata 가 누락되면 forward pass 전에 fail 한다.
6. 가능하면 failure 를 `analysis_manifest.csv` 에 기록한다.

## 11. pairwise dependency appendix

appendix 는 optional 이며 `--enable_pairwise_dependency_appendix` 로만 제어한다. 현재 appendix metric 은 representative PSD curve correlation 이다.

규칙:

1. 더 이른 validation failure 가 없으면 core PSD-first analysis 는 항상 실행한다.
2. manifest 가 failure 를 기록한다면 appendix failure 는 완료된 core artifact 를 무효화하지 않는다.
3. appendix output 은 `category = pairwise_dependency_appendix` 를 사용한다.

## 12. error behavior

| condition | behavior |
| --- | --- |
| input path does not exist | analysis 전에 fail |
| file input is not `.pt` | analysis 전에 fail |
| directory input contains no `.pt` files | analysis 전에 fail |
| directory input contains any non-`.pt` regular file | analysis 전에 fail |
| directory input contains subdirectory | analysis 전에 fail |
| CLI dataset and checkpoint dataset mismatch | forward pass 전에 fail |
| checkpoint missing reconstruction metadata | forward pass 전에 fail |
| CUDA device invalid | model allocation 전에 fail |
| CSV write failure | current run fail |

program 은 output root creation 이후 만난 failure 에 대해 manifest row 를 쓸 수 있다.


## Patched psd_analysis constraints

1. `series` 는 모든 analysis curve/dispersion row 와 file name 에 보존한다.
2. `layer_distance_profile` 과 `layer_distance_trend` representative distance 는 `reducer in {mean, median}` 으로 구분한다.
3. dispersion variance/MAD 는 `layer_dispersion_profile` 과 `layer_dispersion_trend` 로 분리한다.
4. 새 산출물에서는 `drift_distance` 를 생성하지 않는다.

## 2026-05-01 구현 보정: exact-only, dB 후처리 기준

현재 구현은 checkpoint 기반 PSD 분석에서 `psd_exact` 만 산출한다. `psd_userbin` 계열 분석, 저장, CSV 산출은 비활성화한다. `--userbin_edges` 인수는 호환성을 위해 남길 수 있으나, 기본 실행 경로에서는 사용하지 않는다.

`scale=db` 는 모든 필요한 통계 연산을 먼저 raw power domain에서 수행한 뒤 마지막에 dB 변환을 적용한다. 대표 곡선은 row reducer와 sample 평균을 먼저 적용한 뒤 dB로 변환한다. dispersion 계열도 row variance 또는 row MAD와 sample 평균을 먼저 적용한 뒤 dB로 변환한다.

따라서 `scale=db` 의 대표 곡선은 `mean(dB(P))` 가 아니라 `dB(mean(P))` 이며, dispersion은 `var(dB(P))` 또는 `mad(dB(P))` 가 아니라 `dB(mean(var(P)))`, `dB(mean(mad(P)))` 기준이다.


## 2026-05-01 보정: exact-only PSD 산출

현재 구현은 PSD 분석 산출에서 `psd_exact` 만 생성한다. `psd_userbin` 분석, 저장, CSV 산출, plot 산출은 비활성화한다. bash 실행 스크립트는 `USERBIN_EDGES` 를 전달하지 않으며, Python CLI 의 `--userbin_edges` 는 과거 호환성용으로만 남아 있고 분석 경로에서는 무시된다.

## 2026-05-02 수정 고정

- 입력 기준 PSD 는 모델 forward 결과의 `input_record` 를 사용한다. 따라서 regularization 및 input-reference 분석 기준은 첫 hidden layer input 이 아니라 모델이 실제로 받은 prepared input 이다.
- `analysis_curve`, `analysis_dispersion`, `filter_snapshot` 은 `checkpoint_epoch_x/layers/layer_NNN__name/` 아래 저장한다.
- 새 레이어 거리 산출물은 `layer_distance_profile` 과 `layer_distance_trend` 이다. `relation_type` 은 `input_reference` 또는 `adjacent` 이다.
- 새 레이어 내부 퍼짐 산출물은 `layer_dispersion_profile` 과 `layer_dispersion_trend` 이다. `variance` 와 `mad` 는 layer distance 와 섞지 않는다.
- 새 산출물에서는 `drift_distance` 와 `accuracy_loss_join` 을 생성하지 않는다.
- `filter_trend` 는 `traces/filter_trend/layer_NNN__name/` 아래에서 parameter 및 statistic 별 epoch trend 로 저장한다.
