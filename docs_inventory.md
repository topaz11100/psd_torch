# 현재 문서 인벤토리

이 파일은 리팩터링 이후 현재 유효한 문서 지도를 기록한다. 과거 entrypoint 목록이 아니라 현재 `src/psd_snn` 기준 문서 index다.

## 현재 authoritative 문서

| 경로 | 역할 |
|---|---|
| `README.md` | 저장소 개요와 공식 CLI 안내 |
| `Spec/README.md` | 현재 theory/implementation 명세 index |
| `Spec/theory/` | 수학적 정의, 대상 객체, 분석 해설 |
| `Spec/implementation/` | 코드 경로, 설정, artifact, CLI contract |
| `Spec/traceability.md` | 명세와 구현 evidence 연결 |
| `Spec/conflict.md` | 현재 알려진 충돌과 비범위 항목 |
| `examples/README.md` | 사용자 실행 예시와 config template 안내 |
| `docs/final_audit_report.md` | 최근 최종 audit 결과 |
| `docs/refactor_completion_report.md` | 완료 경계와 future work |

## 현재 실행 계층

| 영역 | 현재 경로 |
|---|---|
| Python package | `src/psd_snn/` |
| CLI module | `src/psd_snn/cli/` |
| Bash examples | `examples/bash/` |
| 실행형 JSON config | `examples/configs/runnable/` |
| 주석형 YAML template | `examples/configs/commented/` |
| Tests | `tests/` |

## Archive/reference

```text
old/
Origin/
origin/
references/
```

위 경로는 provenance와 비교를 위한 보존 자료다. 현재 실행 계약이나 사용자-facing 예시는 아니다.
