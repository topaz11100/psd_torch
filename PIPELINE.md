# 실행 파이프라인

## 1. data_prep

- entrypoint: `src/data_prep.py`
- wrapper: `bash/data_prep.sh`
- config: `config/data_prep.json`
- 입력: 원본 데이터 루트, 데이터셋 토큰, 전처리 옵션
- 출력: `<prep_root>/<dataset>/manifest.json`, `train.npy`, `test.npy`, 필요 시 view별 `.npy`

`prep_profile`이 `project_standard`가 아니면 출력 루트가 `<prep_root>/<prep_profile>/<dataset>` 형태가 될 수 있다. downstream 단계의 `prep_root`는 manifest가 실제로 생성된 루트를 기준으로 맞춘다.

## 2. dataset_psd

- entrypoint: `src/dataset_psd.py`
- 입력: prepared bundle
- 출력: `dataset_curve/`, `dataset_dispersion/`, `dataset_psd_manifest.csv`
- 역할: 입력 데이터 자체의 PSD curve와 dispersion을 계산한다.

## 3. dataset_fft

- entrypoint: `src/dataset_fft.py`
- 입력: prepared bundle
- 출력: `dataset_fft/`, `dataset_fft_manifest.csv`
- 역할: 입력 데이터 자체의 FFT power를 계산한다.

## 4. model_training

- entrypoint: `src/model_training.py`
- 입력: prepared bundle, model/readout/training 설정
- 출력: selected `.pt` checkpoint, `training_metrics.csv`
- 역할: 모델 학습과 checkpoint 생성만 수행한다. 모델 신호분석과 plotting은 수행하지 않는다.

## 5. psd_analysis

- entrypoint: `src/psd_analysis.py`
- 입력: checkpoint 파일 또는 `.pt`만 포함하는 checkpoint 디렉터리
- 출력: `analysis_curve`, `analysis_dispersion`, `pair_distance`, layer distance/dispersion, filter 통계, `analysis_manifest.csv`
- 정책: hidden/output 계층만 분석한다. input 레이어는 분석하지 않는다.

## 6. element_psd

- entrypoint: `src/element_psd.py`
- 입력: checkpoint
- 출력: row/neuron별 PSD matrix CSV, `analysis_manifest.csv`
- 정책: hidden/output 계층만 분석한다.

## 7. fft2d_analysis

- entrypoint: `src/2d_fft_analysis.py`
- 입력: checkpoint
- 출력: layer별 2D FFT matrix CSV, `analysis_manifest.csv`
- 정책: hidden/output 계층만 분석한다.

## 8. plotting

- entrypoint: `src/plotting.py`
- 입력: 분석 CSV 디렉터리 또는 CSV 파일
- 출력: PNG figure와 plotting manifest
- 역할: CSV를 읽어 그림만 생성한다. 학습/분석 계산은 하지 않는다.

## 공통 실행 규칙

- 모든 단계는 `--config <json>`을 받는다.
- CLI 인자는 JSON 설정보다 우선한다.
- JSON에 모르는 key가 있으면 오류를 낸다.
- YAML 설정은 공식 실행 경로에서 지원하지 않는다.
- `--help`는 heavy dependency 없이 출력되어야 한다.


## 실행 래퍼/로그 정책

- 모든 `bash/*.sh` 래퍼는 `nohup` 백그라운드 실행을 사용한다.
- 로그는 `logs/<stage>/<timestamp>__<config_stem>.log` 경로에 자동 생성된다.
- 래퍼 실행 직후 `STAGE`, `CONFIG`, `LOG`, `PID`를 출력한다.
- `data_prep`은 `dataset` 리스트를 받아 다중 dataset을 직렬 전처리할 수 있다.
- `model_training`은 현재 단일 GPU 경로를 유지한다. 향후 DDP는 `model_training/checkpoint` 저장 단계 중심으로 적용할 예정이다.


## model_training 실행 모드
- 단일 GPU 학습: `bash/model_training.sh config/model_training.json`
- 2-GPU DDP 학습: `bash/model_training_ddp.sh config/model_training_ddp.json`
- DDP는 model_training/checkpoint 저장 stage에만 적용한다.
- dataset/model 분석 및 plotting stage에는 DDP를 적용하지 않는다.
- DDP 로그 경로: `logs/model_training_ddp/<timestamp>__<config_stem>.log`

- DDP 실제 실행 예시: `torchrun --standalone --nproc_per_node=2 src/model_training.py --config config/model_training_ddp.json --ddp true`
- DDP 스모크 검증 명령: `pytest -q tests/test_model_training_ddp_smoke.py`
