from __future__ import annotations

"""Parameter-free output readout helpers.

The classifier output layer is an actual neuron layer that produces membrane and
spike sequences. Readout is applied functionally to those sequences; there is no
extra learned neural-network head after the output neurons.

Supported modes
---------------
- ``final_membrane``: use the last-step output membrane vector as class
  logits. In the classifier wrapper this mode disables output-layer spike
  emission and spike-triggered reset paths.
- ``earliest_spike``: use the exact first-to-spike rule with same-time membrane
  tie breaking. The forward rule is the same in training and evaluation; when
  gradients are required, a straight-through surrogate score supplies the
  backward path.
- ``max_rate``: use the output firing-rate vector as class logits.
"""

from typing import Optional

import torch


READOUT_ALIASES = {
    "final_membrane": "final_membrane",
    "last_membrane": "final_membrane",
    "last": "final_membrane",
    "final": "final_membrane",
    "membrane": "final_membrane",
    "earliest_spike": "earliest_spike",
    "first_spike": "earliest_spike",
    "fastest_spike": "earliest_spike",
    "earliest": "earliest_spike",
    "ttfs": "earliest_spike",
    "first_to_spike": "earliest_spike",
    "trainable_ttfs": "earliest_spike",
    "max_rate": "max_rate",
    "firing_rate": "max_rate",
    "rate": "max_rate",
    "highest_rate": "max_rate",
}

READOUT_CHOICES = ["final_membrane", "earliest_spike", "max_rate"]
TTFS_TIEBREAK_SCALE = 1e-3
TTFS_NO_SPIKE_MEMBRANE_SCALE = 1e-4
TTFS_PREFIX_DECAY = 0.25
# Backward-compatible aliases kept because other modules store these names.
TTFS_RATE_TIEBREAK_SCALE = TTFS_TIEBREAK_SCALE
TTFS_MEMBRANE_TIEBREAK_SCALE = TTFS_NO_SPIKE_MEMBRANE_SCALE
TTFS_TRAIN_PREFIX_DECAY = TTFS_PREFIX_DECAY
TTFS_FUTURE_DECAY = TTFS_PREFIX_DECAY


def normalize_readout_mode(mode: str) -> str:
    key = str(mode).strip().lower()
    if key not in READOUT_ALIASES:
        raise ValueError(
            "Unknown readout mode: "
            f"{mode}. Supported modes are: final_membrane, earliest_spike, max_rate"
        )
    return READOUT_ALIASES[key]


def membrane_readout_last(soma_seq: torch.Tensor) -> torch.Tensor:
    if soma_seq is None:
        raise ValueError("membrane_readout_last requires soma_seq (got None)")
    if soma_seq.dim() != 3:
        raise ValueError(f"soma_seq must be (B,T,C), got {tuple(soma_seq.shape)}")
    return soma_seq[:, -1, :].to(torch.float32)


def spike_rate_readout(spike_seq: torch.Tensor) -> torch.Tensor:
    if spike_seq is None:
        raise ValueError("spike_rate_readout requires spike_seq (got None)")
    if spike_seq.dim() != 3:
        raise ValueError(f"spike_seq must be (B,T,C), got {tuple(spike_seq.shape)}")
    return spike_seq.to(torch.float32).mean(dim=1)


def _ttfs_descending_weights(num_steps: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.arange(int(num_steps), 0, -1, device=device, dtype=dtype).view(1, -1, 1)


def first_spike_indicator(spike_seq: torch.Tensor) -> torch.Tensor:
    """Return a differentiable first-spike-only tensor.

    For binary forward spikes this equals the strict first-spike mask

    ``q[t] = o[t] * prod_{u<t} (1 - o[u])``

    while remaining trainable through surrogate-gradient spikes.
    """
    if spike_seq is None:
        raise ValueError("first_spike_indicator requires spike_seq (got None)")
    if spike_seq.dim() != 3:
        raise ValueError(f"spike_seq must be (B,T,C), got {tuple(spike_seq.shape)}")
    spk = spike_seq.to(torch.float32).clamp(0.0, 1.0)
    survival_inclusive = torch.cumprod(1.0 - spk, dim=1)
    survival_exclusive = torch.cat(
        [torch.ones_like(spk[:, :1, :]), survival_inclusive[:, :-1, :]],
        dim=1,
    )
    return spk * survival_exclusive


def _bounded_membrane_value(soma_seq: Optional[torch.Tensor], *, like: torch.Tensor) -> torch.Tensor:
    if soma_seq is None:
        return torch.zeros_like(like, dtype=torch.float32)
    return torch.tanh(soma_seq.to(torch.float32))


def _ttfs_prefix_scores(mem_value: torch.Tensor, *, decay: float) -> torch.Tensor:
    if mem_value.dim() != 3:
        raise ValueError(f"mem_value must be (B,T,C), got {tuple(mem_value.shape)}")
    gamma = float(decay)
    if not (0.0 <= gamma < 1.0):
        raise ValueError(f"TTFS prefix decay must satisfy 0 <= decay < 1, got {decay}")
    prefix = torch.zeros_like(mem_value)
    prefix[:, 0, :] = mem_value[:, 0, :]
    for t in range(1, int(mem_value.shape[1])):
        prefix[:, t, :] = mem_value[:, t, :] + gamma * prefix[:, t - 1, :]
    return prefix


def earliest_spike_surrogate_scores(
    spike_seq: torch.Tensor,
    soma_seq: Optional[torch.Tensor] = None,
    *,
    tiebreak_scale: float = TTFS_TIEBREAK_SCALE,
    membrane_tiebreak_scale: float = TTFS_NO_SPIKE_MEMBRANE_SCALE,
    prefix_decay: float = TTFS_PREFIX_DECAY,
) -> torch.Tensor:
    """Differentiable surrogate score for the exact earliest-spike rule.

    The primary term tracks first-spike time. The membrane tie-break term uses a
    discounted past prefix so that, at the first-spike step, the current
    membrane value dominates and earlier steps are consulted only when the current
    value ties.
    """
    if spike_seq is None:
        raise ValueError("earliest_spike_surrogate_scores requires spike_seq (got None)")
    if spike_seq.dim() != 3:
        raise ValueError(f"spike_seq must be (B,T,C), got {tuple(spike_seq.shape)}")

    spk = spike_seq.to(torch.float32).clamp(0.0, 1.0)
    first_only = first_spike_indicator(spk)
    weights = _ttfs_descending_weights(int(spk.shape[1]), device=spk.device, dtype=spk.dtype)
    first_score = (first_only * weights).sum(dim=1)

    mem_value = _bounded_membrane_value(soma_seq, like=spk)
    prefix_scores = _ttfs_prefix_scores(mem_value, decay=float(prefix_decay))
    tie_score = (first_only * prefix_scores).sum(dim=1)
    silent_prob = torch.cumprod(1.0 - spk, dim=1)[:, -1, :]
    silent_mem = silent_prob * mem_value[:, -1, :]

    return first_score + float(tiebreak_scale) * tie_score + float(membrane_tiebreak_scale) * silent_mem


def _strict_first_spike_times(spike_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    spk_bin = spike_seq.to(torch.float32) > 0.0
    B, T, C = spk_bin.shape
    time_idx = torch.arange(int(T), device=spike_seq.device, dtype=torch.int64).view(1, T, 1)
    first_t = torch.where(
        spk_bin,
        time_idx,
        torch.full((1, T, 1), int(T), device=spike_seq.device, dtype=torch.int64),
    ).amin(dim=1)
    has_spike = spk_bin.any(dim=1)
    return first_t, has_spike


def earliest_spike_exact_logits(
    spike_seq: torch.Tensor,
    soma_seq: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Exact first-to-spike decision rule.

    Decision order:
      1. earliest first spike
      2. if tied, compare membrane state at that first-spike step
      3. if still tied, compare membrane state step-by-step backward in time
      4. if no class spikes at all, compare last-step membrane state
    """
    if spike_seq is None:
        raise ValueError("earliest_spike_exact_logits requires spike_seq (got None)")
    if spike_seq.dim() != 3:
        raise ValueError(f"spike_seq must be (B,T,C), got {tuple(spike_seq.shape)}")
    if soma_seq is None:
        raise ValueError("earliest_spike exact evaluation requires soma_seq")
    if soma_seq.dim() != 3:
        raise ValueError(f"soma_seq must be (B,T,C), got {tuple(soma_seq.shape)}")

    first_t, _ = _strict_first_spike_times(spike_seq)
    mem_bct = soma_seq.to(torch.float32).transpose(1, 2).contiguous()
    B, C, T = int(spike_seq.shape[0]), int(spike_seq.shape[2]), int(spike_seq.shape[1])
    device = spike_seq.device

    min_t = first_t.amin(dim=1)
    no_spike_batches = min_t >= int(T)
    winners = torch.zeros(B, device=device, dtype=torch.long)

    if bool(no_spike_batches.any()):
        last_mem = mem_bct[:, :, -1]
        winners = torch.where(no_spike_batches, last_mem.argmax(dim=1), winners)

    active = ~no_spike_batches
    if bool(active.any()):
        candidate = first_t.eq(min_t.view(B, 1)) & active.view(B, 1)
        done = (~active).clone()
        neg_inf = torch.tensor(float('-inf'), device=device, dtype=mem_bct.dtype)
        for offset in range(int(T)):
            unresolved = (~done) & (min_t >= offset)
            if not bool(unresolved.any()):
                break
            t_idx = (min_t - int(offset)).clamp(min=0, max=int(T) - 1)
            gather_idx = t_idx.view(B, 1, 1).expand(-1, C, 1)
            vals = mem_bct.gather(2, gather_idx).squeeze(-1)
            masked = vals.masked_fill(~candidate, neg_inf)
            best_vals = masked.max(dim=1).values
            best_here = candidate & vals.eq(best_vals.view(B, 1)) & unresolved.view(B, 1)
            candidate = torch.where(unresolved.view(B, 1), best_here, candidate)
            unique = candidate.sum(dim=1) == 1
            just_done = unresolved & unique
            if bool(just_done.any()):
                winners = torch.where(just_done, candidate.to(torch.float32).argmax(dim=1).to(torch.long), winners)
                done = done | just_done
        remaining = (~done) & active
        if bool(remaining.any()):
            winners = torch.where(remaining, candidate.to(torch.float32).argmax(dim=1).to(torch.long), winners)

    out = torch.zeros((B, C), device=device, dtype=torch.float32)
    out[torch.arange(B, device=device), winners] = 1.0
    return out


def earliest_spike_readout(
    spike_seq: torch.Tensor,
    soma_seq: Optional[torch.Tensor] = None,
    *,
    training: bool = False,
    prefix_tiebreak_scale: float = TTFS_TIEBREAK_SCALE,
    membrane_tiebreak_scale: float = TTFS_NO_SPIKE_MEMBRANE_SCALE,
    prefix_decay: float = TTFS_PREFIX_DECAY,
) -> torch.Tensor:
    # ``training`` is kept in the public signature for compatibility with
    # existing callers, but the forward rule itself does not branch on it.
    exact = earliest_spike_exact_logits(spike_seq, soma_seq)
    if not torch.is_grad_enabled():
        return exact
    requires_grad = bool(getattr(spike_seq, "requires_grad", False)) or bool(
        False if soma_seq is None else getattr(soma_seq, "requires_grad", False)
    )
    if not requires_grad:
        return exact
    surrogate = earliest_spike_surrogate_scores(
        spike_seq,
        soma_seq,
        tiebreak_scale=float(prefix_tiebreak_scale),
        membrane_tiebreak_scale=float(membrane_tiebreak_scale),
        prefix_decay=float(prefix_decay),
    )
    return surrogate + (exact - surrogate).detach()


def apply_readout(
    *,
    soma_seq: Optional[torch.Tensor],
    spike_seq: Optional[torch.Tensor],
    mode: str = "final_membrane",
    training: bool = False,
    earliest_spike_rate_tiebreak: float = TTFS_RATE_TIEBREAK_SCALE,
    earliest_spike_membrane_tiebreak: float = TTFS_MEMBRANE_TIEBREAK_SCALE,
) -> torch.Tensor:
    mode_key = normalize_readout_mode(mode)
    if mode_key == "final_membrane":
        return membrane_readout_last(soma_seq)
    if mode_key == "earliest_spike":
        return earliest_spike_readout(
            spike_seq,
            soma_seq,
            training=bool(training),
            prefix_tiebreak_scale=float(earliest_spike_rate_tiebreak),
            membrane_tiebreak_scale=float(earliest_spike_membrane_tiebreak),
        )
    if mode_key == "max_rate":
        return spike_rate_readout(spike_seq)
    raise ValueError(f"Unknown readout mode: {mode}")
