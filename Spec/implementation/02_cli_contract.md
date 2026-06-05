# CLI and Bash Execution Contract

## Config 병합 규칙

CLI 인자가 YAML 설정값보다 우선한다. `parse_args_with_config`는 다음 순서로 인자를 확정한다.

1. `--config`를 argv에서 추출한다.
2. stage key가 있으면 해당 객체만 읽는다.
3. parser가 모르는 YAML key가 있으면 즉시 실패한다.
4. YAML 값을 parser default로 설정한다.
5. 최종 CLI argv를 다시 parse한다.

이 규칙 때문에 설정 파일은 항상 실제 parser가 받는 key만 포함해야 한다.

## 2차원 bash config 그룹

다음 스크립트는 2D 그룹 계약을 따른다.

- `bash/model_training.sh`
- `bash/model_training_ddp.sh`
- `bash/psd_analysis.sh`
- `bash/fft2d_analysis.sh`
- `bash/element_psd.sh`
- `bash/element_fft.sh`
- `bash/plotting.sh`

예시는 다음과 같다.

```bash
CONFIG_GROUP_0=("a.yaml" "b.yaml")  # a,b 병렬
CONFIG_GROUP_1=("c.yaml")           # group0 완료 후 c 실행
CONFIG_GROUPS=(CONFIG_GROUP_0 CONFIG_GROUP_1)
```

각 config는 `nohup`으로 별도 로그 파일을 만들고, 그룹 내 모든 PID가 성공해야 다음 그룹으로 넘어간다. CLI로 config를 넘기면 인자 목록 전체가 하나의 병렬 그룹으로 해석된다.

## Dataset stage 예외

`data_prep.sh`, `dataset_fft.sh`, `dataset_psd.sh`는 config 내부 dataset 배열을 기준으로 직렬 실행한다. 같은 raw/prep directory에 대해 동시에 쓰기 작업이 겹치면 manifest 손상이 가능하기 때문이다.

## DDP 실행

`model_training_ddp.sh`는 기본적으로 `NPROC_PER_NODE=2`를 사용한다. 다른 GPU 수를 사용하려면 환경변수와 config의 `ddp_world_size`를 함께 맞춘다.

```bash
NPROC_PER_NODE=4 bash bash/model_training_ddp.sh config/base/model_training_ddp.clean.yaml
```
