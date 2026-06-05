# PSD/SNN 신호분석 프로젝트

이 저장소는 SNN 모델과 입력 데이터의 시간 신호를 PSD, element-wise PSD, element-wise FFT, 2D FFT, 거리 지표, 시각화 artifact로 분리해 분석하는 프로젝트다. 실행 단위는 root `src/*.py` entrypoint, root `config/*.yaml`, root `bash/*.sh` wrapper로 고정한다.

## 공식 실행 단계

```text
raw data
  -> src/data_prep.py
  -> src/dataset_psd.py
  -> src/dataset_fft.py
  -> src/model_training.py
  -> src/psd_analysis.py
  -> src/element_psd.py
  -> src/element_fft.py
  -> src/2d_fft_analysis.py
  -> src/plotting.py
```

## 공식 디렉터리

```text
bash/       단계별 shell wrapper
config/     단계별 YAML 설정과 설정 설명서
Spec/       이론/구현 명세
src/        데이터 준비, 학습, 분석, 시각화 코드
tests/      현재 root pipeline 계약 테스트
paper/      논문/배경 정리 자료
Origin/     선택적 외부 저자 코드 어댑터가 참조하는 원천 코드
```

## 핵심 정책

- 모델 분석(`psd_analysis`, `element_psd`, `element_fft`, `2d_fft_analysis`)은 input 레이어를 분석하지 않는다.
- 입력 데이터 자체의 PSD/FFT는 `dataset_psd`, `dataset_fft`에서 독립적으로 분석한다.
- 설정 파일과 manifest는 YAML을 사용한다.
- 학습/분석 결과 산출물은 CSV로 저장하며 `src/util/csv_schema.py`의 category schema를 따른다.
- 모델 체크포인트 내부 metadata만 JSON-compatible payload로 유지한다.
- seed는 Python, NumPy, Torch, DataLoader worker/generator에 적용한다.
- deterministic mode는 성능을 위해 끈다.

## 기본 실행 예시

```bash
bash/data_prep.sh config/data_prep.yaml
bash/dataset_psd.sh config/dataset_psd.yaml
bash/dataset_fft.sh config/dataset_fft.yaml
bash/model_training.sh config/model_training.yaml
bash/psd_analysis.sh config/psd_analysis.yaml
bash/element_psd.sh config/element_psd.yaml
bash/element_fft.sh config/element_fft.yaml
bash/fft2d_analysis.sh config/fft2d_analysis.yaml
bash/plotting.sh config/plotting.yaml
```

설정 항목의 의미와 자료형은 `config/README.md`에 정리되어 있다.

## 명세

- `Spec/theory/`: 신호 객체, PSD/FFT 수식, probe, artifact 의미.
- `Spec/implementation/`: 현재 코드 경로, CLI/config 계약, 출력 schema.
