"""Model token parsing and canonical model registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


ALIASES: Dict[str, str] = {
    "lif": "lif",
    "rf": "rf",
    "tc": "tc_lif",
    "tc_lif": "tc_lif",
    "ts": "ts_lif",
    "ts_lif": "ts_lif",
    "dh_snn": "dh_snn",
    "d_rf": "d_rf",
    "my_dh_snn": "my_dh_snn",
    "my_r_dh_snn": "my_r_dh_snn",
    "my_d_rf": "my_d_rf",
}


@dataclass
class ModelSpec:
    """Canonical parsed model specification."""

    canonical: str
    recurrent: bool = False
    fixed_branches: Optional[int] = None
    variant: Optional[str] = None


def parse_model_token(token: str) -> ModelSpec:
    """Parse model token including recurrent and branch suffixes."""

    raw = token.strip()
    recurrent = "_R" in raw
    parts = raw.replace("_R", "").split("_")
    fixed = None
    if parts[-1].isdigit():
        fixed = int(parts[-1])
        parts = parts[:-1]
    base = "_".join(parts)
    canonical = ALIASES.get(base.lower(), base.lower())
    return ModelSpec(canonical=canonical, recurrent=recurrent, fixed_branches=fixed)
