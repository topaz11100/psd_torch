# CSV schema 계약

## 1. 목적

PSD project 의 numeric artifact 는 CSV 로 저장한다. 이전처럼 모든 artifact 에 동일한 거대한 wide shared column 집합을 강제하지 않는다. 대신 모든 CSV 는 공통 prefix column 과 `category` 를 가진 뒤, category 별로 필요한 최소 column 만 가진다.

핵심 의도는 아래와 같다.

1. 한 CSV file 은 하나의 category 만 표현한다.
2. `category` 값만 보면 해당 CSV 의 column 양식을 알 수 있어야 한다.
3. plot 재구성에 필요한 좌표와 값은 CSV 안에 직접 들어 있어야 한다.
4. 필요 없는 column 을 빈칸으로 잔뜩 남기지 않는다.
5. PSD curve, dispersion, pair distance, drift distance 는 서로 다른 category 와 독립 CSV file 로 저장한다.

## 2. 공통 prefix column

모든 official CSV 는 맨 앞에 아래 column 을 이 순서로 둔다.

```text
schema_version
category
source_program
status
message
dataset
run_id
created_at
```

의미는 아래와 같다.

| column | 의미 |
| --- | --- |
| `schema_version` | CSV schema version. 기본값은 `psd_category_csv_v2` |
| `category` | category별 schema routing key |
| `source_program` | `data_prep`, `dataset_psd`, `model_training`, `psd_analysis`, `plotting`, `reinterpretation` 중 하나 |
| `status` | `ok`, `failed`, `rendered`, `skipped_unsupported_category`, `skipped_existing` 등 |
| `message` | optional warning, error, skip reason |
| `dataset` | canonical dataset token |
| `run_id` | run identifier |
| `created_at` | ISO-8601 timestamp |

## 3. category registry

Official category 는 아래를 포함한다.

| category | producer | file 단위 |
| --- | --- | --- |
| `training_metric` | `model_training.py` | metric table 하나 |
| `dataset_curve` | `dataset_psd.py` | scope, signal, extractor, reducer, variant, scale 조합 하나 |
| `dataset_dispersion` | `dataset_psd.py` | scope, signal, extractor, variant, scale, statistic 조합 하나 |
| `analysis_curve` | `psd_analysis.py` | checkpoint, layer, probe scope, signal, series, extractor, reducer, variant, scale 조합 하나 |
| `analysis_dispersion` | `psd_analysis.py` | checkpoint, layer, probe scope, signal, series, extractor, variant, scale, statistic 조합 하나 |
| `pair_distance` | `psd_analysis.py` | checkpoint, layer, source/target scope, source/target series, extractor, reducer, variant, scale 조합 하나 |
| `layer_distance_profile` | `psd_analysis.py` | 특정 checkpoint 의 layer relation curve distance |
| `layer_distance_trend` | `psd_analysis.py` | epoch 축 layer relation curve distance trend |
| `filter_snapshot` | `psd_analysis.py` | checkpoint, layer, filter/neuron parameter family 하나 |
| `filter_trend` | `psd_analysis.py` | layer, filter/neuron parameter family 하나 |
| `accuracy_loss_join` | `psd_analysis.py` | 기존 CSV 호환용. 새 산출물에서는 생성하지 않음 |
| `pairwise_dependency_appendix` | `psd_analysis.py` | optional appendix metric 하나 |
| `analysis_manifest` | `psd_analysis.py` | analysis artifact inventory 하나 |
| `dataset_psd_manifest` | `dataset_psd.py` | dataset baseline artifact inventory 하나 |
| `plotting_manifest` | `plotting.py` | rendering status table 하나 |
| `reinterpretation_metric` | `reinterpretation/driver.py` | paper experiment metric table 하나 |

Unsupported `category` 는 plotting 에서 global error 가 아니라 `skipped_unsupported_category` 로 기록한다.

## 4. category별 column

### 4.1 `training_metric`

```text
schema_version,category,source_program,status,message,dataset,run_id,created_at,model_token,model_family,readout_mode,seed,epoch,scope,metric,value,value_unit
```

`scope` 는 `train`, `validation`, `test` 처럼 supervised metric 이 평가된 범위를 기록한다. 이 scope 는 `psd_analysis` 나 `dataset_psd` 의 split 선택 인수가 아니다.

### 4.2 `dataset_curve`

```text
schema_version,category,source_program,status,message,dataset,run_id,created_at,scope,probe_family,label,signal_kind,extractor,reducer,variant,scale,frequency,frequency_unit,bin_left,bin_right,value,value_unit
```

규칙:

1. `scope` 는 `train_full`, `test_full`, `same_label_label_00`, `balanced_global`, `distribution_global` 같은 분석 범위를 기록한다.
2. `probe_family` 는 해당되는 경우 `same_label`, `balanced_global`, `distribution_global`, `full_dataset` 중 하나다.
3. `extractor` 는 현재 구현에서 `psd_exact` 만 사용한다.
4. `scale` 은 `raw` 또는 `db` 다.
5. `value_unit` 은 raw PSD curve 에서는 `power`, dB curve 에서는 `dB` 를 사용한다.
6. `psd_exact` row 는 `frequency` 와 `frequency_unit = normalized_frequency` 를 사용한다.
7. 현재 구현은 `psd_userbin` row 를 생성하지 않는다. `bin_left`, `bin_right` 는 exact-only 출력에서 비워둔다.

### 4.3 `dataset_dispersion`

```text
schema_version,category,source_program,status,message,dataset,run_id,created_at,scope,probe_family,label,signal_kind,extractor,variant,scale,statistic,frequency,frequency_unit,bin_left,bin_right,value,value_unit
```

`statistic` 은 `variance`, `mad`, `q25`, `q75` 등을 사용할 수 있다.

### 4.4 `analysis_curve`

```text
schema_version,category,source_program,status,message,dataset,run_id,created_at,model_token,model_family,readout_mode,seed,checkpoint_path,checkpoint_epoch,layer,layer_index,scope,probe_family,label,signal_kind,series,extractor,reducer,variant,scale,frequency,frequency_unit,bin_left,bin_right,value,value_unit
```

규칙:

1. 각 CSV file 은 checkpoint, layer, scope, signal_kind, series, extractor, reducer, variant, scale 조합 하나만 담는다.
2. `scope` 는 선택 인수가 아니라 결과 metadata 다.
3. analysis category 의 `scope` 는 prepared probe scope 만 기록한다. train/test full split 전체집합 scope 는 analysis category 에 쓰지 않는다.
4. 모든 가능한 명세 조합을 파일로 분리해 저장한다.

### 4.5 `analysis_dispersion`

```text
schema_version,category,source_program,status,message,dataset,run_id,created_at,model_token,model_family,readout_mode,seed,checkpoint_path,checkpoint_epoch,layer,layer_index,scope,probe_family,label,signal_kind,series,extractor,variant,scale,statistic,frequency,frequency_unit,bin_left,bin_right,value,value_unit
```

규칙:

1. `scope` 는 prepared probe scope 만 기록한다.
2. train/test full split 전체집합 scope 는 analysis dispersion 에 쓰지 않는다.

### 4.6 `pair_distance`

```text
schema_version,category,source_program,status,message,dataset,run_id,created_at,model_token,model_family,readout_mode,seed,checkpoint_epoch,layer,layer_index,source_scope,target_scope,source_signal_kind,source_series,target_signal_kind,target_series,extractor,reducer,variant,scale,distance_metric,value,value_unit
```

`distance_metric` 은 `l2`, `cosine_distance`, `area_distance` 등 구현이 지원하는 metric token 을 사용한다.

규칙:

1. `source_scope` 와 `target_scope` 는 prepared probe scope 만 기록한다.
2. train/test full split 전체집합 scope 는 pair distance 에 쓰지 않는다.

### 4.7 `layer_distance_profile` 와 `layer_distance_trend`

```text
schema_version,category,source_program,status,message,dataset,run_id,created_at,model_token,model_family,readout_mode,seed,checkpoint_epoch_a,checkpoint_epoch_b,layer,layer_index,scope,signal_kind,series,reference_signal_kind,reference_series,extractor,reducer,variant,scale,distance_metric,value,value_unit
```

Drift 는 같은 checkpoint 안에서 input PSD 를 reference 로 두고 대상 layer/series PSD 와의 개형 거리를 checkpoint 축으로 기록한다. 여러 checkpoint 가 있으면 epoch별 input-reference distance trend 를 만든다.

규칙:

1. `scope` 는 prepared probe scope 만 기록한다.
2. train/test full split 전체집합 scope 는 drift distance 에 쓰지 않는다.

### 4.8 `filter_snapshot`

```text
schema_version,category,source_program,status,message,dataset,run_id,created_at,model_token,model_family,seed,checkpoint_epoch,layer,layer_index,parameter_name,statistic,value,value_unit
```

### 4.9 `filter_trend`

```text
schema_version,category,source_program,status,message,dataset,run_id,created_at,model_token,model_family,seed,layer,layer_index,parameter_name,checkpoint_epoch,statistic,value,value_unit
```

### 4.10 `plotting_manifest`

```text
schema_version,category,source_program,status,message,dataset,run_id,created_at,input_csv_path,output_figure_path,render_seconds
```

## 5. 파일 이름 규칙

CSV file name 은 lowercase snake case 를 사용한다. 한 file 이 하나의 category 와 하나의 곡선 조합만 담도록 이름에 주요 좌표를 넣는다.

권장 pattern:

```text
<category>__<scope>__<signal_kind>__<series>__<extractor>__<reducer>__<variant>__<scale>.csv
<category>__epoch_<epoch>__layer_<n>__<scope>__<signal_kind>__<series>__<extractor>__<reducer>__<variant>__<scale>.csv
pair_distance__epoch_<epoch>__layer_<n>__<source_scope>__<source_signal_kind>_<source_series>__to__<target_scope>__<target_signal_kind>_<target_series>__<extractor>__<reducer>__<variant>__<scale>__<distance_metric>.csv
layer_distance_profile__epoch_<e>__<relation_type>__<scope>__<track_name>__<extractor>__<reducer>__<variant>__<scale>__<distance_metric>.csv
```

## 6. empty value policy

1. category schema 에 없는 column 은 만들지 않는다.
2. category schema 에 있는 column 이 특정 row 에서만 적용되지 않으면 empty cell 로 둔다.
3. numeric missing value 는 metric 이 `nan` semantics 를 명시적으로 정의하지 않는 한 empty cell 을 사용한다.
4. Boolean value 는 lowercase `true` 또는 `false` 로 encode 한다.
5. path 는 string 으로 저장한다. `run_id` 또는 `output_root` 에서 모호하지 않으면 relative path 를 권장한다.

## 7. 금지 output form

아래 항목은 official numeric artifact 가 아니다.

1. hidden multi-artifact binary bundle.
2. plot 재구성을 위해 별도 axes file 을 반드시 요구하는 matrix file.
3. numeric CSV 의 대체물인 figure file.
4. category column 이 없는 CSV.
5. 하나의 CSV 안에 서로 다른 category 를 섞은 파일.
6. official CSV output 으로서의 raw activation dump.

## 8. 예시

### 8.1 `analysis_curve`

```text
schema_version,category,source_program,status,message,dataset,run_id,created_at,model_token,model_family,readout_mode,seed,checkpoint_path,checkpoint_epoch,layer,layer_index,scope,probe_family,label,signal_kind,series,extractor,reducer,variant,scale,frequency,frequency_unit,bin_left,bin_right,value,value_unit
psd_category_csv_v2,analysis_curve,psd_analysis,ok,,shd,run001,2026-04-30T00:00:00Z,lif_soft_fixed,dense,mean,0,checkpoints/epoch_000017.pt,17,layer_1,1,balanced_global,balanced_global,,hidden,spike,psd_exact,mean,raw,raw,0.10,normalized_frequency,,,0.031,power
```

### 8.2 `pair_distance`

```text
schema_version,category,source_program,status,message,dataset,run_id,created_at,model_token,model_family,readout_mode,seed,checkpoint_epoch,layer,layer_index,source_scope,target_scope,source_signal_kind,source_series,target_signal_kind,target_series,extractor,reducer,variant,scale,distance_metric,value,value_unit
psd_category_csv_v2,pair_distance,psd_analysis,ok,,shd,run001,2026-04-30T00:00:00Z,lif_soft_fixed,dense,mean,0,17,layer_1,1,balanced_global,distribution_global,hidden,spike,hidden,spike,psd_exact,mean,raw,db,distance_raw,0.271,dimensionless
```


## Patched coordinate preservation

`analysis_curve` 와 `analysis_dispersion` 은 `signal_kind` 뒤에 `series` 를 보존한다. 따라서 `hidden/layer_input`, `hidden/membrane`, `hidden/spike` 같은 서로 다른 trace series 는 같은 CSV 파일에 섞이면 안 된다.

`layer_distance_profile` 과 `layer_distance_trend` 는 `relation_type=input_reference` 또는 `relation_type=adjacent` 의 representative PSD curve distance 를 기록한다. dispersion variance/MAD 는 `layer_dispersion_profile` 과 `layer_dispersion_trend` 로 분리한다.

`pair_distance` 와 `pairwise_dependency_appendix` 는 `source_series`, `target_series`, `reducer` 를 반드시 보존한다.

## 2026-05-01 구현 보정: userbin 비활성화와 dB 단위

현재 구현의 PSD 계열 CSV는 `extractor=psd_exact` 만 산출한다. `extractor=psd_userbin` 산출물은 생성하지 않는다.

`scale=db` 는 통계 연산 이후 dB 변환된 값을 의미한다. dispersion의 dB scale 값은 `dB^2` 가 아니라 dB로 표기한다. raw scale의 variance는 `power^2`, raw scale의 MAD는 `power` 를 유지한다.


## 2026-05-01 보정: exact-only PSD 산출

현재 구현은 PSD 분석 산출에서 `psd_exact` 만 생성한다. `psd_userbin` 분석, 저장, CSV 산출, plot 산출은 비활성화한다. bash 실행 스크립트는 `USERBIN_EDGES` 를 전달하지 않으며, Python CLI 의 `--userbin_edges` 는 과거 호환성용으로만 남아 있고 분석 경로에서는 무시된다.

## 2026-05-02 수정 고정

- `SCHEMA_VERSION` 은 `psd_category_csv_20260502` 이다.
- 모든 주요 PSD CSV category 는 `prep_profile`, `psd_axis_kind`, `psd_time_axis`, `psd_row_axes`, `psd_flatten_rule`, `psd_logical_shape`, `static_repeat_T` 를 포함한다.
- 새 category 는 `layer_distance_profile`, `layer_distance_trend`, `layer_dispersion_profile`, `layer_dispersion_trend` 이다.
- `drift_distance` 와 `accuracy_loss_join` 은 기존 CSV 읽기 호환용 schema 로만 남기고, 새 `psd_analysis` 산출물에서는 생성하지 않는다.
- `write_common_csv` 는 schema column 이 아닌 `extra_columns` 값도 버리지 않고 보존한다.
