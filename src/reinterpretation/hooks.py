"""PSD hook helpers for paper-experiment reinterpretation runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from src.signal.family_spectral_analysis import compute_family_spectral_summary


@dataclass
class StreamingPSDHookAccumulator:
    """Accumulate detached signal batches and convert them to PSD summaries on demand."""

    window: int
    overlap: int
    userbin_edges: tuple[float, ...]
    max_batches: int | None = None
    _signals: dict[str, list[torch.Tensor]] = field(default_factory=dict)
    _missing: dict[str, str] = field(default_factory=dict)

    def add_signal(self, family: str, signal: torch.Tensor | None, *, reason_if_missing: str = 'hook unavailable') -> None:
        family = str(family)
        if signal is None:
            self._missing.setdefault(family, str(reason_if_missing))
            return
        if self.max_batches is not None and len(self._signals.get(family, [])) >= int(self.max_batches):
            return
        tensor = torch.as_tensor(signal).detach().cpu()
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(1)
        if tensor.ndim == 3:
            # Author code may expose either (B,T,C) or (B,R,T). Treat the longer
            # axis as time only when no explicit adapter metadata is available.
            if int(tensor.shape[1]) >= int(tensor.shape[2]):
                tensor = tensor.transpose(1, 2).contiguous()
        elif tensor.ndim == 4:
            batch = int(tensor.shape[0])
            tensor = tensor.reshape(batch, int(tensor.shape[1]), -1).contiguous()
        elif tensor.ndim == 5:
            batch, time_steps = int(tensor.shape[0]), int(tensor.shape[1])
            tensor = tensor.permute(0, 2, 3, 4, 1).reshape(batch, -1, time_steps).contiguous()
        else:
            self._missing.setdefault(family, f'unsupported signal rank {tensor.ndim}')
            return
        self._signals.setdefault(family, []).append(tensor)

    def summaries(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for family, chunks in self._signals.items():
            if not chunks:
                continue
            maps = torch.cat(chunks, dim=0)
            out[family] = compute_family_spectral_summary(
                maps,
                window=int(self.window),
                overlap=int(self.overlap),
                userbin_edges=list(float(v) for v in self.userbin_edges),
            )
        return out

    def missing_rows(self) -> list[dict[str, Any]]:
        observed = set(self._signals)
        rows = [{'family': family, 'status': 'observed', 'reason': ''} for family in sorted(observed)]
        rows.extend({'family': family, 'status': 'missing', 'reason': reason} for family, reason in sorted(self._missing.items()) if family not in observed)
        return rows
