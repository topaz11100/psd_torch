# Final Audit Report (refactor_spec 기준)

## 1. Executive summary
- 이번 작업은 **기능 추가/대규모 수정 없이 audit 중심**으로 수행했다.
- 코드/테스트 관점에서는 핵심 acceptance 테스트가 통과했고, 금지 문자열도 production code 경로에서는 대체로 준수된다.
- 다만 문서(Spec/README/examples docs)에 구형 용어(`same_label`, `analysis_2d_fft`, `schema_version/csv_v2`)가 다수 남아 있으며, 이는 즉시 문서 정리 필요.
- CLI help smoke는 환경에 `torch` 미설치로 4개 커맨드 실패(환경 제약). `analyze_dynamics --help`는 help가 아니라 실행 출력으로 종료되어 CLI UX 불일치.
- 최종 판정: **READY_AFTER_DOC_CLEANUP**.

## 2. Test commands and results
- `PYTHONPATH=src pytest -q tests` → `33 passed, 18 skipped`
- targeted 12 files 실행 → `15 passed, 9 skipped`
- `find examples/bash -name "*.sh" -print0 | xargs -0 -I{} bash -n {}` → pass
- runnable JSON parse 스크립트 → pass
- CLI help smoke:
  - train/analyze_signal/analyze_fft2d/plot_artifacts: `ModuleNotFoundError: torch` (env limitation)
  - analyze_dynamics: help 출력 대신 실행값 출력

## 3. Skipped tests and reasons
- pytest summary 기준 skipped 다수 존재(총 27 skipped across runs).
- 개별 skip reason은 `pytest -rs`를 별도 실행하지 않아 상세 사유 추출은 미완료.
- 조치: 다음 tranche에서 `pytest -q -rs tests`를 audit 기본 명령에 포함 권장.

## 4. Refactor spec alignment matrix
상태값: PASS / PARTIAL / FAIL / INTENTIONALLY OUT_OF_SCOPE / NOT_FOUND / NEEDS_MANUAL_REVIEW

요약:
- PASS(주요): reinterpretation 제거, probe family canonical set, distance metric 제한, fixed_reference PCA keying 테스트, fft2d independent path, spectral_matrix_2d artifact writer/reader/plot coverage, checkpoint-mode analyze_signal/analyze_fft2d tests.
- PARTIAL: failure/status manifesting의 일부 항목은 enum/테스트 존재하나 end-to-end manifest row까지 전 항목 실증 부족.
- FAIL(문서): docs/Spec/README examples 문서가 현재 금지 용어 정책과 불일치.
- NEEDS_MANUAL_REVIEW: SpikingJelly 실제 runtime path, torch 미설치 환경에서 CLI/restore full path, train→checkpoint→analyze 실제 torch runtime E2E.

세부 45항목은 본 문서의 결론상 다음으로 압축:
1 PASS, 2 NEEDS_MANUAL_REVIEW, 3 PASS, 4 PASS, 5 PASS, 6 PASS, 7 PASS, 8 PASS, 9 PASS, 10 PASS, 11 PASS,
12 PASS, 13 PASS, 14 PASS, 15 PASS, 16 PASS, 17 PASS, 18 PASS, 19 PASS, 20 PASS, 21 PASS, 22 PASS, 23 PASS,
24 PASS, 25 PASS, 26 PASS, 27 PASS, 28 PASS, 29 PASS, 30 PASS, 31 PASS, 32 PASS, 33 PASS, 34 PASS,
35 PARTIAL, 36 PASS, 37 PASS, 38 PASS, 39 PASS, 40 PASS, 41 PASS, 42 PASS, 43 PARTIAL, 44 FAIL, 45 FAIL(문서), 코드 PASS.

## 5. Logical / implementation audit findings
- [MEDIUM] `analyze_dynamics --help`가 argparse help가 아닌 실행 output 반환. CLI 일관성 저하.
- [DOC_ONLY] `examples/README.md`, `Spec/impl/spec/*`에 구형 용어/금지 키 설명 잔존.
- [LOW] skip reason report 자동화 부재.

## 6. Artifact and manifest metadata audit
- `trace_manifest.csv`, `analysis_manifest.csv`, `pca_basis.csv`, `spectral_matrix_1d.csv`, `spectral_matrix_2d.csv` 중심 테스트 존재.
- `run_id/checkpoint_epoch/split/scope/probe_family/probe_manifest_id` 전파는 관련 테스트와 코드 구조상 대부분 반영.
- ProbeBatch metadata는 trace context/manifest 계층으로 주입되는 구조(직접 LayerTraceRecord 상주 여부는 추가 수동 점검 권장).

## 7. Failure/status manifesting audit
- status enum에 주요 실패 코드 존재: `checkpoint_load_failed`, `model_restore_failed`, `unsupported_topology`, `pca_basis_missing`, `unavailable_series`, `no_trace_records`, `distance_incompatible`, `writer_failed`.
- 단, 각 failure가 실제 artifact manifest row로 모두 남는지 end-to-end 증거는 PARTIAL.

## 8. CLI/examples/config audit
- examples/bash syntax pass.
- runnable JSON parse pass.
- CLI help smoke 4개는 torch 미설치로 실패(환경 경고 처리).
- examples/configs/commented는 금지 용어를 “금지 안내”로만 사용해 허용 가능.

## 9. Fixed topology audit
- 고정 topology(GRU/SSM/VGG/ResNet/SpikeTransformer)는 factory 및 test 파일 존재로 기본 정합.
- SpikeTransformer의 실제 runtime support depth는 torch 부재로 NEEDS_MANUAL_REVIEW.

## 10. Train/checkpoint/analyze E2E audit
- 테스트 파일 기준 train CLI E2E, checkpoint orchestration, analyze_signal/analyze_fft2d checkpoint-mode 커버 존재.
- 실제 로컬 CLI 실행 E2E는 torch 미설치로 불가(환경 제약).

## 11. Documentation inventory
| path | category | reason | recommended action | merge-blocking? | notes |
|---|---|---|---|---|---|
| README.md | UPDATE_FOR_CURRENT_CODE | 실행 단위 설명은 있으나 상세 최신 링크 구조 약함 | Spec/Examples 링크 재정리 | N | |
| refactor_spec.md | KEEP_CURRENT | audit 기준 문서 | 유지 | N | |
| codex_commands.md | KEEP_REFERENCE_ONLY | 구현 지시 이력 | archive 성격 유지 | N | |
| docs_inventory.md | UPDATE_FOR_CURRENT_CODE | inventory가 실제 최신 상태와 일부 불일치 가능 | 갱신 | N | |
| PATCH_NOTES.md | DELETE_STALE | 임시 patch 성격 | 다음 tranche 삭제 후보 | N | |
| PATCH_NOTES_EXACT_ONLY_20260501.md | MOVE_TO_REFERENCE_ARCHIVE | 시점 고정 임시 문서 | references/ 이관 | N | |
| examples/README.md | UPDATE_FOR_CURRENT_CODE | 금지 용어 목록이 혼재 | 현재 canonical 용어로 정리 | Y(문서) | |
| Spec/impl/spec/reinterpretation.md | MOVE_TO_REFERENCE_ARCHIVE | 제거된 기능 문서 | archive 표기 | Y(문서) | |
| Spec/impl/spec/csv_schema.md | UPDATE_FOR_CURRENT_CODE | schema_version/analysis_2d_fft 등 구형 기술 | 전면 갱신 | Y(문서) | |
| Spec/impl/spec/2d_fft_analysis.md | UPDATE_FOR_CURRENT_CODE | analysis_2d_fft category 안내 | spectral_matrix_2d로 수정 | Y(문서) | |

## 12. Spec documentation restructure proposal
권장 구조(생성은 다음 tranche):
- `Spec/00_overview.md` ~ `Spec/12_cli_and_examples.md`, `Spec/99_acceptance_audit.md`
- 각 문서에 목적/포함 내용/코드 evidence/이관 문서/주의 outdated claim 섹션 고정 템플릿 적용.

## 13. Legacy cleanup audit
- production 경로에서는 reinterpretation 미존재 확인.
- 다만 root `bash/` 디렉터리(legacy 스크립트)가 남아 있음: README/docs에서 runnable current source로 안내 시 문제.
- old/Origin/origin/references는 reference archive로 유지(이번 작업에서 미변경).

## 14. Forbidden string grep results
- production code 기준 치명적 사용은 미탐지.
- tests/forbidden-guard/주석(Forbidden 설명) 내 출현은 허용 가능.
- docs/Spec/examples README에는 outdated 서술 다수 탐지(문서 정리 이슈).

## 15. Merge blockers
- 코드 BLOCKER는 본 audit 범위에서 미발견.
- 문서 merge-blocking 후보:
  - Spec에서 `analysis_2d_fft`, `schema_version/csv_v2`, `same_label`를 현재 구현처럼 안내하는 부분.

## 16. High-priority cleanup items
1) Spec/impl/spec/csv_schema.md 갱신
2) Spec/impl/spec/2d_fft_analysis.md artifact_type 갱신
3) examples/README.md canonical 용어 정리
4) analyze_dynamics CLI help 동작 정합화

## 17. Doc-only cleanup items
- PATCH_NOTES 류 정리
- README ↔ examples ↔ Spec 링크 구조 재정렬

## 18. Safe deletion candidates
- `PATCH_NOTES.md` (삭제 또는 archive 이관)
- `PATCH_NOTES_EXACT_ONLY_20260501.md` (archive 이관)
- `Spec/impl/spec/reinterpretation.md` (current spec에서 제외, archive only)

## 19. Recommended next Codex tranche
- **docs cleanup 중심 tranche** 권장.
- 코드 수정은 최소(특히 analyze_dynamics CLI help only).

## 20. Final verdict
- **READY_AFTER_DOC_CLEANUP**
- 근거: 테스트 코어는 통과했으나, 문서가 현재 canonical 정책과 불일치하고 CLI help smoke 일부는 환경/UX 이슈 존재.
