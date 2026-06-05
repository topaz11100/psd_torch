# Artifact Writer, Reader, and Plotting Contract

모든 분석 CSV는 `src/util/csv_schema.py::common_row`가 정의하는 공통 column superset을 사용한다. 없는 값은 빈 문자열로 남긴다.

## Manifest

각 stage는 `<stage>_manifest.yaml`를 쓴다. manifest row는 최소한 다음 정보를 포함한다.

- `source_program`
- `run_id`
- `dataset`
- `seed`
- `category`
- `artifact_name`
- `output_csv_path`
- `status`

분석 결과를 후처리할 때는 파일명 추론보다 manifest를 우선한다.

## Plotting

`plotting.py`는 `--input`이 파일이면 단일 CSV, 디렉터리면 재귀 CSV tree로 해석한다. `--output`을 지정하면 입력 tree와 같은 상대 경로 구조를 출력 디렉터리에 재현한다.

필터 통계에서 `count`는 단위가 다르므로 기본 plot에서는 제외한다. 필요한 경우 `--include_filter_count`를 명시한다.
