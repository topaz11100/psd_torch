# JSON 설정 안내
모든 실행 설정은 `config/*.json`을 사용합니다. JSON 주석은 지원하지 않으므로 설명은 이 문서에서 제공합니다.

## 공통 규칙
- `--config`는 `.json`만 허용합니다.
- CLI 인자가 JSON보다 우선합니다.
- 최상위는 객체여야 하며, 각 파일은 단계 키(`data_prep`, `dataset_psd` 등) 아래에 실제 설정을 둡니다.

## 주요 설정 항목
- `dataset`: 데이터셋 토큰, 문자열, 필수, 예: `mnist`, 사용: data_prep/dataset_psd/dataset_fft/model_training/분석 단계
- `prep_root`: prepared 루트 경로, 문자열, 필수, 예: `/ABS/PATH/TO/prepared`, 사용: 대부분 단계
- `output_root`: 출력 루트 경로, 문자열, 필수(해당 단계), 예: `/ABS/PATH/TO/output/...`
- `seed`: 전역 시드, 정수, 권장/대부분 필수
- `batch_size`: 배치 크기, 정수(1 이상)
- `gpu_index`: CUDA 장치 인덱스, 정수(0 이상)
- `num_workers`: DataLoader worker 수, 정수(0 이상)
- `checkpoint`: 체크포인트 파일/디렉터리 경로, 문자열, 모델 분석 단계 필수
- `anal_batch`: 분석 배치 크기, 정수(1 이상), 모델 분석 단계 필수
- `anal_epoch_list`: 분석 에폭 리스트, 정수 배열, model_training에서 선택
- `force_overwrite`: 기존 prepared 덮어쓰기 여부, 불리언 또는 불리언 문자열, data_prep
- `download`: 원본 데이터 다운로드 여부, 불리언 또는 불리언 문자열, data_prep
- `max_samples`: split별 최대 샘플 수, 정수(양수) 또는 null, data_prep
- `prep_profile`: 전처리 프로필, 문자열, 예: `project_standard`, data_prep
- `deap_label_axis`: DEAP 라벨 축, 문자열, 허용값 `valence|arousal`, data_prep
- `deap_num_classes`: DEAP 클래스 수, 정수, 허용값 `2|3`, data_prep
- `shd_dt_ms`/`ssc_dt_ms`: SHD/SSC 시간 해상도(ms), 양수 실수, data_prep
- `shd_max_time`/`ssc_max_time`: SHD/SSC 최대 시간(초), 양수 실수, data_prep
- `overwrite`: plot 출력 덮어쓰기, 불리언, plotting

## 파일별 entrypoint
- `data_prep.json` -> `src/data_prep.py`
- `dataset_psd.json` -> `src/dataset_psd.py`
- `dataset_fft.json` -> `src/dataset_fft.py`
- `model_training.json` -> `src/model_training.py`
- `psd_analysis.json` -> `src/psd_analysis.py`
- `element_psd.json` -> `src/element_psd.py`
- `fft2d_analysis.json` -> `src/2d_fft_analysis.py`
- `plotting.json` -> `src/plotting.py`
