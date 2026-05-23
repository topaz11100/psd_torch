# Current Documentation Inventory

This file records the current documentation map after the PSD/SNN refactor. It is not a legacy entrypoint list.

## Authoritative current docs

| Path | Role |
|---|---|
| `README.md` | Repository overview and current CLI entrypoints |
| `Spec/README.md` | Current specification index |
| `Spec/theory/` | Mathematical and conceptual definitions |
| `Spec/implementation/` | Code-path and artifact contracts |
| `Spec/traceability.md` | Spec-to-code coverage map |
| `Spec/conflict.md` | Known conflicts or intentionally unresolved items |
| `examples/README.md` | User-facing examples and runnable config guide |
| `docs/final_audit_report.md` | Last final-audit report |
| `docs/refactor_completion_report.md` | Completion boundary and remaining future work |

## Current runnable layer

| Area | Current path |
|---|---|
| Python package | `src/psd_snn/` |
| CLI modules | `src/psd_snn/cli/` |
| Runnable examples | `examples/bash/` |
| Runnable JSON configs | `examples/configs/runnable/` |
| Commented templates | `examples/configs/commented/` |
| Tests | `tests/` |

## Archive/reference material

The following directories are read-only historical/reference material:

```text
old/
Origin/
origin/
references/
```

They are useful for understanding source provenance, but they are not current executable contracts.

## Current cleanup boundary

Legacy root shell launchers and stale implementation spec files were removed from the current contract. If historical copies are needed, recover them from Git history or archive bundles rather than reintroducing them into the current root execution layer.
