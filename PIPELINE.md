# 파이프라인 개요
1. data_prep: raw 데이터를 prepared bundle로 변환
2. dataset_psd: 입력 데이터 PSD 분석
3. dataset_fft: 입력 데이터 FFT 분석
4. model_training: 모델 학습 및 체크포인트 생성
5. psd_analysis: 모델 hidden/output PSD 분석
6. element_psd: 모델 hidden/output element PSD 분석
7. fft2d_analysis: 모델 hidden/output 2D FFT 분석
8. plotting: 분석 CSV 시각화

## 입력/출력 정책
- 모델 분석은 input 레이어를 분석하지 않습니다.
- 입력 데이터 자체 분석은 `dataset_psd`, `dataset_fft`에서 독립 수행합니다.
- 설정은 JSON으로 통일하며 상세 항목은 `config/README.md`를 참고합니다.

## 실행 예시
- `bash/data_prep.sh config/data_prep.json`
- `bash/dataset_psd.sh`
- `bash/dataset_fft.sh config/dataset_fft.json`
- `bash/model_training.sh config/model_training.json`

## data_prep 경로/프로필 주의사항
- `prep_profile`이 `project_standard`가 아니면 출력 경로가 `<prep_root>/<prep_profile>/<dataset>` 형태가 될 수 있습니다.
- downstream 단계의 `prep_root`는 data_prep 실행 시 실제 manifest가 생성된 루트를 기준으로 맞춰야 합니다.
