# PSD/SNN 분석 워크스페이스

이 저장소는 SNN 레이어 신호의 시간 구조를 PSD, PCA, 2D FFT, 거리 지표로 분석하기 위해 리팩터링한 작업 공간이다. 현재 공식 구현은 `src/psd_snn` 아래에 있으며, 학습, checkpoint 분석, 동역학 통계, artifact 입출력, plotting을 분리한다.

## 현재 공식 패키지

```text
src/psd_snn/
  cli/          # train, analyze_signal, analyze_fft2d, analyze_dynamics, plot_artifacts
  config/       # dataclass 기반 설정과 validator
  models/       # MLP topology, IF/LIF/RF cell, fixed topology smoke models
  analysis/     # probe, trace, signal map, PSD/PCA/FFT2D, distance
  artifacts/    # summary writer, trace writer, reader, plotting
  training/     # 최소 synthetic training/checkpoint smoke 경로
```

`src/psd_snn` 밖의 과거 구현은 현재 실행 계약이 아니다. 역사적 출처와 비교가 필요하면 archive/reference 디렉터리를 확인한다.

## 현재 CLI

```bash
PYTHONPATH=src python -m psd_snn.cli.train --help
PYTHONPATH=src python -m psd_snn.cli.analyze_signal --help
PYTHONPATH=src python -m psd_snn.cli.analyze_fft2d --help
PYTHONPATH=src python -m psd_snn.cli.analyze_dynamics --help
PYTHONPATH=src python -m psd_snn.cli.plot_artifacts --help
```

사용 예시는 `examples/`가 공식 진입점이다.

```bash
source examples/bash/00_env.sh
examples/bash/12_end_to_end_train_analyze_plot.sh
```

## 구현 완료 경계

현재 phase는 다음 범위를 완료 대상으로 본다.

- MLP topology와 IF/LIF/RF cell 분리.
- hidden layer의 spike-only SRNN recurrence.
- `none`, `clip`, `structure`, `clipstructure` scenario.
- `final_if`, `final_mem` readout.
- raw trace `B,T,*`와 SignalMap `S,R,T` 변환.
- PSD 대표화 `mean`, `median`, `element_psd`, `pca`.
- fixed-reference PCA basis fit/apply 및 `pca_basis_id` 비교 규칙.
- 독립 2D FFT 분석과 `spectral_matrix_2d` artifact.
- exact/userbin 축과 PCA basis를 고려한 strict spectral distance.
- trace tensor chunk와 manifest/summary CSV.
- synthetic training checkpoint smoke와 artifact reader/plotting 기본 경로.

## 문서 구조

- `Spec/README.md`: 현재 명세 index.
- `Spec/theory/`: 대상 객체, 수식, 분석 의도, 해석 기준.
- `Spec/implementation/`: 현재 코드 경로, 설정, CLI, artifact contract.
- `examples/README.md`: 실행 예시와 config template 안내.
- `docs/refactor_completion_report.md`: 완료 경계와 향후 작업.
- `docs/final_audit_report.md`: 최근 audit 결과.

## Archive/reference 디렉터리

다음 디렉터리는 현재 실행 계층이 아니라 보존 자료다.

```text
old/
Origin/
origin/
references/
```

## 향후 작업

- 실제 dataset ingest/preprocessing 연동.
- 논문 제출용 figure style refinement.
- multi-run launch packaging.
- 대규모 학습 orchestration.
- fixed topology의 논문 원형 충실도 확장.
