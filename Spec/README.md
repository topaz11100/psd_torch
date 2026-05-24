# PSD/SNN 명세

이 명세는 현재 root entrypoint 기반 파이프라인을 기준으로 한다. 공식 실행 경로는 `src/*.py`, `config/*.json`, `bash/*.sh`다.

## 이론 문서

| 문서 | 내용 |
|---|---|
| `theory/00_overview.md` | 전체 문제 정의, 단계 분리, input 분석 정책 |
| `theory/01_signal_trace_and_signal_map.md` | trace와 SignalMap의 축 의미 |
| `theory/02_psd_representatives.md` | PSD curve, dispersion, element PSD 수식 |
| `theory/03_pca_fixed_reference.md` | 현재 비활성 PCA 기준과 향후 확장 경계 |
| `theory/04_fft2d.md` | dataset/model FFT와 2D FFT 정의 |
| `theory/05_probe_families.md` | full, balanced_global, label_single probe 의미 |
| `theory/06_spiking_cells_if_lif_rf.md` | IF/LIF/RF 동역학과 분석 가능 신호 |
| `theory/07_constraints_clip_structure_clipstructure.md` | clip, structure, clipstructure 제약 의미 |
| `theory/08_topologies_and_readout.md` | MLP/CNN/보조 topology와 readout 정책 |
| `theory/09_dynamics_statistics.md` | filter/동역학 파라미터 통계 |
| `theory/10_artifacts_distance_and_manifests.md` | CSV category, distance, manifest 정책 |

## 구현 문서

| 문서 | 내용 |
|---|---|
| `implementation/00_architecture.md` | 현재 코드 구조와 책임 분리 |
| `implementation/01_config_contract.md` | JSON 설정 계약 |
| `implementation/02_cli_contract.md` | CLI와 bash wrapper 계약 |
| `implementation/03_model_factory_and_checkpoints.md` | model token, checkpoint, restore 정책 |
| `implementation/04_trace_signal_analysis_pipeline.md` | prepared data와 model trace 분석 흐름 |
| `implementation/05_artifact_writer_reader_plotting.md` | CSV/manifest/plotting schema |
| `implementation/06_examples_contract.md` | config와 bash 실행 예시 정책 |

## 핵심 불변 조건

1. 모델 신호분석은 input 레이어를 분석하지 않는다.
2. 입력 데이터 자체 분석은 `dataset_psd.py`와 `dataset_fft.py`에서만 수행한다.
3. 설정은 JSON만 사용한다.
4. seed는 고정하지만 deterministic mode는 끈다.
5. stage별 산출물은 CSV category와 manifest로 추적한다.
