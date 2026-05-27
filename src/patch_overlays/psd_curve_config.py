"""PSD curve-token parsing and normalized-frequency userbin helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

EXTRACTOR_ALIASES = {'exact': 'psd_exact', 'psd_exact': 'psd_exact', 'userbin': 'psd_userbin', 'psd_userbin': 'psd_userbin'}
REDUCERS = ('mean', 'median')
CENTERING_ALIASES = {'raw': 'raw', 'centered': 'centered', 'cen': 'centered'}
SCALES = ('raw', 'db')
USERBIN_REDUCERS = ('mean', 'median', 'sum')
DEFAULT_PSD_TOKEN = 'exact_mean_raw_raw'
ALL_DATASET_PSD_TOKENS = tuple(
    f'{extractor}_{reducer}_{centering}_{scale}'
    for extractor in ('exact', 'userbin')
    for reducer in REDUCERS
    for centering in ('raw', 'centered')
    for scale in SCALES
)


@dataclass(frozen=True)
class PSDCurveSpec:
    token: str
    extractor: str
    reducer: str
    centering: str
    scale: str

    @property
    def summary_centering(self) -> str:
        return 'cen' if self.centering == 'centered' else 'raw'


def _split_tokens(values: Sequence[str] | str | None, *, default: Sequence[str] | None = None) -> list[str]:
    if values is None:
        return list(default or [DEFAULT_PSD_TOKEN])
    if isinstance(values, str):
        raw = [values]
    else:
        raw = [str(v) for v in values]
    out: list[str] = []
    for item in raw:
        for chunk in str(item).replace(',', ' ').split():
            token = chunk.strip()
            if token:
                out.append(token)
    return out or list(default or [DEFAULT_PSD_TOKEN])


def parse_psd_curve_token(token: str) -> PSDCurveSpec:
    raw = str(token).strip().lower().replace('-', '_')
    parts = raw.split('_')
    if parts[:2] == ['psd', 'exact']:
        parts = ['exact'] + parts[2:]
    elif parts[:2] == ['psd', 'userbin']:
        parts = ['userbin'] + parts[2:]
    if len(parts) != 4:
        raise ValueError(
            f'Invalid PSD curve token {token!r}. Expected <exact|userbin>_<mean|median>_<raw|centered>_<raw|db>.'
        )
    extractor_raw, reducer, centering_raw, scale = parts
    extractor = EXTRACTOR_ALIASES.get(extractor_raw)
    if extractor is None:
        raise ValueError(f'Unsupported PSD token extractor {extractor_raw!r}.')
    if reducer not in REDUCERS:
        raise ValueError(f'Unsupported PSD token reducer {reducer!r}.')
    centering = CENTERING_ALIASES.get(centering_raw)
    if centering is None:
        raise ValueError(f'Unsupported PSD token centering {centering_raw!r}.')
    if scale not in SCALES:
        raise ValueError(f'Unsupported PSD token scale {scale!r}.')
    canonical_extractor = 'exact' if extractor == 'psd_exact' else 'userbin'
    canonical = f'{canonical_extractor}_{reducer}_{centering}_{scale}'
    return PSDCurveSpec(token=canonical, extractor=extractor, reducer=reducer, centering=centering, scale=scale)


def parse_psd_curve_tokens(values: Sequence[str] | str | None, *, default: Sequence[str] | None = None) -> tuple[PSDCurveSpec, ...]:
    specs: list[PSDCurveSpec] = []
    seen: set[str] = set()
    for token in _split_tokens(values, default=default):
        spec = parse_psd_curve_token(token)
        if spec.token not in seen:
            specs.append(spec)
            seen.add(spec.token)
    if not specs:
        specs.append(parse_psd_curve_token(DEFAULT_PSD_TOKEN))
    return tuple(specs)


def tokens_require_userbins(specs: Iterable[PSDCurveSpec]) -> bool:
    return any(spec.extractor == 'psd_userbin' for spec in specs)


def _float_values(values: Sequence[float] | Sequence[str] | str | float | int | None) -> list[float]:
    if values is None:
        return []
    if isinstance(values, str):
        raw = values.replace(',', ' ').split()
    elif isinstance(values, (float, int)):
        raw = [values]
    else:
        raw = list(values)
    return [float(v) for v in raw if str(v).strip() != '']


def normalize_userbin_reducer(value: str | None) -> str:
    token = str(value or 'mean').strip().lower()
    if token not in USERBIN_REDUCERS:
        raise ValueError(f'Unsupported userbin reducer {value!r}. Allowed: {USERBIN_REDUCERS}.')
    return token


def resolve_userbin_edges(
    *,
    edges: Sequence[float] | Sequence[str] | str | None = None,
    width: float | str | None = None,
    count: int | str | None = None,
    required: bool = False,
) -> list[float] | None:
    values = _float_values(edges)
    if len(values) == 1:
        width = values[0]
        values = []
    if values:
        resolved = values
    elif width not in (None, ''):
        step = float(width)
        if step <= 0.0 or step > 0.5:
            raise ValueError('userbin width must satisfy 0 < width <= 0.5 in normalized frequency.')
        resolved = [0.0]
        current = 0.0
        while current + step < 0.5:
            current += step
            resolved.append(float(current))
        if abs(resolved[-1] - 0.5) > 1.0e-12:
            resolved.append(0.5)
    elif count not in (None, ''):
        n = int(count)
        if n < 1:
            raise ValueError('userbin count must be >= 1.')
        resolved = np.linspace(0.0, 0.5, n + 1, dtype=np.float64).tolist()
    elif required:
        raise ValueError('userbin edges are required for userbin PSD tokens. Provide explicit analysis_userbin_edges or a single bin width.')
    else:
        return None
    if len(resolved) < 2:
        raise ValueError('userbin edges must contain at least two values.')
    prev = None
    for value in resolved:
        if value < -1.0e-12 or value > 0.5 + 1.0e-12:
            raise ValueError('userbin edges must lie in normalized frequency range [0.0, 0.5].')
        if prev is not None and value <= prev:
            raise ValueError('userbin edges must be strictly increasing.')
        prev = value
    resolved[0] = max(0.0, float(resolved[0]))
    resolved[-1] = min(0.5, float(resolved[-1]))
    return [float(v) for v in resolved]
