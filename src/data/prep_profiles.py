"""전처리 프로필 이름 조회 전용 경량 모듈."""

from __future__ import annotations

PROJECT_STANDARD_PREP_PROFILE = "project_standard"
REFERENCE_PREP_PROFILES = (
    "need_high_cifar10_dvs_t16",
    "drf_shd_t250",
    "dh_snn_shd_t1000",
)


def available_prep_profiles() -> tuple[str, ...]:
    return (PROJECT_STANDARD_PREP_PROFILE, *sorted(REFERENCE_PREP_PROFILES))
