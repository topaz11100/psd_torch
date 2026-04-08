from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import torch


RecordArg = bool | Sequence[str]


def normalize_record_keys(record: RecordArg, default_keys: Sequence[str]) -> Optional[Tuple[str, ...]]:
    """Normalize the public ``record`` argument into an explicit key tuple.

    Rules used across neuron wrappers:
    - ``False``: do not record signals (fast path)
    - ``True``: record a common default key list
    - sequence: record only user-requested keys
    """

    if record is False:
        return None
    if record is True:
        return tuple(str(k) for k in default_keys)
    return tuple(str(k) for k in record)


def rollout_sequence(
    x_seq: torch.Tensor,
    *,
    step_fn: Callable[[torch.Tensor, bool], torch.Tensor | tuple[torch.Tensor, Dict[str, torch.Tensor]]],
    record_keys: Optional[Sequence[str]],
) -> torch.Tensor | tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Roll out ``(B,T,*)`` input through a step function with optional recording.

    ``step_fn`` must accept ``(x_t, record_flag)`` and return:
    - ``y_t`` when ``record_flag=False``
    - ``(y_t, signal_dict)`` when ``record_flag=True``
    """

    _, steps, _ = x_seq.shape

    if record_keys is None:
        out_list: list[torch.Tensor] = []
        for t in range(int(steps)):
            y_t = step_fn(x_seq[:, t], False)
            if not torch.is_tensor(y_t):
                raise TypeError("step_fn must return a tensor when record=False")
            out_list.append(y_t)
        return torch.stack(out_list, dim=1)

    out_list = []
    rec_lists: Dict[str, list[torch.Tensor]] = {str(k): [] for k in record_keys}
    for t in range(int(steps)):
        item = step_fn(x_seq[:, t], True)
        if not (isinstance(item, tuple) and len(item) == 2):
            raise TypeError("step_fn must return (tensor, dict) when record=True")
        y_t, sig = item
        out_list.append(y_t)
        for key in record_keys:
            key = str(key)
            if key not in sig:
                available = ", ".join(sorted(sig.keys()))
                raise KeyError(f"Unknown record key: {key!r}. Available keys: [{available}]")
            rec_lists[key].append(sig[key])

    out_seq = torch.stack(out_list, dim=1)
    rec = {key: torch.stack(values, dim=1) for key, values in rec_lists.items()}
    return out_seq, rec
