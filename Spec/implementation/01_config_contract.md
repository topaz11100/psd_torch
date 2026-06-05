# YAML Config Contract

모든 entrypoint는 `--config <path.yaml>`을 지원하며, YAML root는 stage 이름을 key로 갖는 객체다. 공식 설정 파일 확장자는 `.yaml`/`.yml`만 사용한다.

```yaml
model_training:
  dataset: mnist
  prep_root: /home/yongokhan/workspace/data/prep_data
  seed: 0
```

## 공통 경로와 seed

- `prep_root`: `/home/yongokhan/workspace/data/prep_data`
- `raw_data` 또는 `raw_data_root`: `/home/yongokhan/workspace/data/raw_data` (`data_prep` 전용)
- 산출물: `/home/yongokhan/workspace/result/<stage-or-scenario>/<case>/...`
- `seed`: `0`

`raw_data`는 새 명세의 이름이고, `raw_data_root`는 기존 CLI 호환 alias다. 두 값은 `data_prep`에서만 의미가 있으며, 동시에 제공될 때 값이 다르면 오류다. 다른 stage는 prepared artifact만 읽으므로 raw-data 경로를 설정하지 않는다.

## Dataset 배열 계약

`data_prep`, `dataset_fft`, `dataset_psd`의 `dataset`은 문자열 또는 문자열 배열이다. 배열이면 다음처럼 직렬 실행한다.

```yaml
dataset_fft:
  dataset:
    - mnist
    - shd
  prep_root: /home/yongokhan/workspace/data/prep_data
  output_root: /home/yongokhan/workspace/result/dataset_fft
```

배열 길이가 2 이상이면 각 데이터셋의 산출물은 기본 `timestamped_output=true`에서 `output_root/<stage>_<timestamp>/<dataset>/...` 아래에 분리된다. `timestamped_output=false`일 때만 기존 `output_root/<dataset>/...` 경로를 쓴다.

## Scenario config

`config/ddp_train_scenario/<group>/<case>.yaml`마다 다음 분석 config가 대응된다.

- `config/psd_analysis_scenario/<group>/<case>.yaml`
- `config/fft2d_analysis_scenario/<group>/<case>.yaml`
- `config/2dfft_scenario/<group>/<case>.yaml` (별칭)
- `config/element_psd_scenario/<group>/<case>.yaml`
- `config/element_fft_scenario/<group>/<case>.yaml`
- `config/dataset_fft_scenario/<group>/<case>.yaml`

## Clean base config

`config/base/*.clean.yaml`은 실험 시나리오가 주입되지 않은 백지 설정이다. parser가 받는 모든 설정 인자를 포함하며, 사용자는 dataset/model/path만 바꿔 바로 실행할 수 있다.
