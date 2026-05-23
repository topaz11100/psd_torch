# PSD/SNN 현재 명세

이 디렉터리는 현재 `src/psd_snn` 구현을 설명하는 공식 명세다. 명세는 두 층으로 나뉜다.

1. `Spec/theory/`: 분석 대상, 수식 정의, 왜 이 객체를 계산하는지, 결과를 어떻게 해석하는지 설명한다.
2. `Spec/implementation/`: 위 이론 객체가 현재 코드, CLI, config, artifact로 어떻게 구현되는지 정의한다.

## Theory 문서

| 문서 | 내용 |
|---|---|
| `theory/00_overview.md` | 전체 분석 철학과 실행 단위 |
| `theory/01_signal_trace_and_signal_map.md` | trace와 SignalMap 객체 정의 |
| `theory/02_psd_representatives.md` | PSD-first 대표화와 거리 정의 |
| `theory/03_pca_fixed_reference.md` | PCA 대표화와 fixed-reference basis |
| `theory/04_fft2d.md` | row-time map의 독립 2D FFT 분석 |
| `theory/05_probe_families.md` | probe family와 sampling 의미 |
| `theory/06_spiking_cells_if_lif_rf.md` | IF/LIF/RF cell 동역학 |
| `theory/07_constraints_clip_structure_clipstructure.md` | clip/structure/clipstructure scenario |
| `theory/08_topologies_and_readout.md` | MLP, fixed topology, readout |
| `theory/09_dynamics_statistics.md` | 파라미터와 내부 상태 통계 |
| `theory/10_artifacts_distance_and_manifests.md` | artifact, manifest, distance identity |

## Implementation 문서

| 문서 | 내용 |
|---|---|
| `implementation/00_architecture.md` | package 구조와 책임 분리 |
| `implementation/01_config_contract.md` | dataclass config와 validation |
| `implementation/02_cli_contract.md` | CLI contract |
| `implementation/03_model_factory_and_checkpoints.md` | model factory와 checkpoint restore |
| `implementation/04_trace_signal_analysis_pipeline.md` | trace-to-analysis pipeline |
| `implementation/05_artifact_writer_reader_plotting.md` | artifact writer/reader/plotting |
| `implementation/06_examples_contract.md` | examples와 config template |
| `implementation/99_acceptance_audit.md` | acceptance audit 기준 |

## 상태 표기

- `구현됨`: 현재 코드 경로와 테스트 또는 smoke 검증이 있다.
- `의도적 미지원`: 현재 설계상 금지하거나 다음 phase로 분리했다.
- `향후 작업`: 현재 phase의 필수 완료 범위 밖이다.

## Archive 경계

과거 자료는 `old/`, `Origin/`, `origin/`, `references/`에 둔다. 현재 명세는 `src/psd_snn`과 `examples/`만 active layer로 설명한다.
