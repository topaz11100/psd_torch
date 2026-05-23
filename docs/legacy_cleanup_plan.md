# Legacy Cleanup Plan

## Applied in this patch

- Root shell launchers are removed from the current runnable layer.
- Temporary patch-note documents are removed from the current docs layer.
- Stale implementation spec files under the old Spec implementation path are removed.
- Current README, examples guide, documentation inventory, and Spec index are replaced with current paths.

## Preserved archives

The following directories remain untouched and are intentionally available for reference:

```text
old/
Origin/
origin/
references/
```

## Deferred dependency-audited cleanup

The repository may still contain old root Python modules outside `src/psd_snn`. They should be removed only after a dependency audit confirms that current tests, examples, and package imports do not rely on them. The recommended rule is simple: current runnable code should import `psd_snn` modules, while historical implementations belong in archive/reference directories.

## Guard policy

Current docs and examples must not present deprecated root launchers or old artifact categories as official. Negative guard tests may contain deprecated tokens, but runnable examples and current specs should use the canonical names defined in `Spec/README.md`.
