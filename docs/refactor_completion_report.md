# 리팩터링 완료 보고서

## 현재 상태

분석 코어 phase는 완료 판정 가능한 상태다. 현재 공식 구현은 `src/psd_snn`이며, 문서와 examples도 이 구조를 기준으로 정리되어 있다.

## 완료된 영역

- IF/LIF/RF cell과 MLP topology 분리.
- spike-only SRNN recurrence.
- `none`, `clip`, `structure`, `clipstructure` scenario.
- raw trace `B,T,*`와 SignalMap `S,R,T` 변환.
- PSD 대표화 `mean`, `median`, `element_psd`, `pca`.
- fixed-reference PCA basis fit/apply와 basis id 비교 규칙.
- 2D FFT 독립 분석과 `spectral_matrix_2d` artifact.
- strict spectral distance compatibility.
- checkpoint-mode `analyze_signal`, `analyze_fft2d` smoke 경로.
- trace `.pt` chunk와 manifest.
- artifact reader와 기본 plotting CLI.
- synthetic training/checkpoint/analyze smoke.
- canonical examples와 config template.

## 남은 future work

- 실제 dataset ingest/preprocessing.
- multi-run launch packaging.
- publication-ready plotting style.
- 대규모 training scheduler.
- fixed topology의 논문 원형 충실도 추가 확장.

## Archive 정책

`old/`, `Origin/`, `origin/`, `references/`는 보존 자료다. 현재 문서는 archive 코드를 active pipeline으로 안내하지 않는다.
