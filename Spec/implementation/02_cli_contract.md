# CLI and Bash Execution Contract

## Config 병합 규칙

CLI 인자가 YAML 설정값보다 우선한다. `parse_args_with_config`는 다음 순서로 인자를 확정한다.

1. `--config`를 argv에서 추출한다.
2. stage key가 있으면 해당 객체만 읽는다.
3. 중첩 YAML을 leaf key 기준으로 flatten한다.
4. 값이 공란인 clean-template 항목은 parser default override에서 제외한다.
5. parser가 모르는 YAML key가 있으면 즉시 실패한다.
6. YAML 값을 parser default로 설정한다.
7. 최종 CLI argv를 다시 parse한다.
8. 필수 인자가 아직 없으면 명시적 missing-required 오류를 낸다.

이 규칙 때문에 설정 파일은 항상 실제 parser가 받는 key만 포함해야 한다. 각 clean template는 이 조건을 기준으로 검증한다.

## Bash wrapper 계약

루트 `bash/` 아래 script는 각 stage의 단일 launcher다. 첫 번째 인자는 config 경로이며, 생략하면 `config/<stage>.yaml`을 사용한다. 두 번째 이후 인자는 그대로 Python CLI로 전달한다.

```bash
bash bash/data_prep.sh config/data_prep.yaml --dataset mnist
bash bash/model_training.sh config/model_training.yaml --epochs 100
bash bash/psd_analysis.sh config/psd_analysis.yaml --checkpoint /path/to/model.pt
```

## DDP 실행

`model_training`은 `gpu_index` 배열 하나로 단일 GPU와 DDP를 결정한다. 별도의 DDP launcher는 없다.

```bash
# 단일 프로세스: cuda:0
bash bash/model_training.sh config/model_training.yaml --gpu_index 0

# 자동 DDP: physical cuda:0,1만 노출한 뒤 torchrun 재실행
bash bash/model_training.sh config/model_training.yaml --gpu_index 0 1

# comma form도 허용
bash bash/model_training.sh config/model_training.yaml --gpu_index 0,1,3
```

다중 원소 `gpu_index`를 감지하면 parent process가 `CUDA_VISIBLE_DEVICES`를 해당 목록으로 좁히고 `python -m torch.distributed.run --standalone --nproc_per_node=<len(gpu_index)> -m src.model_training ...` 형태로 자기 자신을 재실행한다. child rank에서는 `LOCAL_RANK`가 narrowed visible-device ordinal이므로 physical GPU 선택은 parent 단계에서만 수행된다.

## 병렬 scenario 그룹 제거

과거의 2차원 bash config group 실행 계약은 제거했다. 병렬 sweep이 필요하면 외부 job scheduler, shell loop, 또는 별도 generator가 clean template에서 resolved config를 만든 뒤 launcher를 반복 호출한다. 이 저장소에는 stage별 최소 launcher만 둔다.
