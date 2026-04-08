from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn

from src.model.model_registry import builder_name_from_any
from src.readout.readout import (
    TTFS_MEMBRANE_TIEBREAK_SCALE,
    TTFS_RATE_TIEBREAK_SCALE,
    apply_readout,
    normalize_readout_mode,
)
from src.neurons.surrogate import SpikeFn


@dataclass
class SNNConfig:
    model_name: str
    input_dim: int
    hidden_dim: int
    num_classes: int
    hidden_dims: Optional[Sequence[int]] = None
    branch: int = 8
    S_min: float = 1.0
    S_max: Optional[float] = None
    th_len: int = 4
    v_th: float = 1.0
    v_pre: float = 1.0
    spike_surrogate: str = "mg"
    dt: float = 1.0
    rf_reset_mode: str = "no_reset"
    readout_mode: str = "final_membrane"
    recurrent: bool = False


def _set_layer_spiking_enabled(layer: nn.Module, enabled: bool) -> None:
    setter = getattr(layer, "set_spiking_enabled", None)
    if callable(setter):
        setter(bool(enabled))
        return
    if hasattr(layer, "spiking_enabled"):
        setattr(layer, "spiking_enabled", bool(enabled))
        if not bool(enabled):
            spk = getattr(layer, "spk", None)
            if torch.is_tensor(spk):
                spk.zero_()


class SNNClassifier(nn.Module):
    """Feed-forward SNN with a neuron-based output layer and functional readout.

    There is no extra learned NN head after the output neurons. The output layer
    itself produces membrane and spike sequences, and readout maps those
    sequences to class logits. Only ``final_membrane`` switches the output layer
    into a non-spiking, no-spike-reset configuration; the spike-based readouts
    keep the ordinary spiking output-neuron dynamics.
    """

    def __init__(
        self,
        hidden_layers: Sequence[nn.Module],
        output_layer: nn.Module,
        *,
        readout_mode: str = "final_membrane",
        earliest_spike_rate_tiebreak: float = TTFS_RATE_TIEBREAK_SCALE,
        earliest_spike_membrane_tiebreak: float = TTFS_MEMBRANE_TIEBREAK_SCALE,
    ):
        super().__init__()
        self.hidden_layers = nn.ModuleList(list(hidden_layers))
        self.output_layer = output_layer
        self.readout_mode = normalize_readout_mode(readout_mode)
        self.earliest_spike_rate_tiebreak = float(earliest_spike_rate_tiebreak)
        self.earliest_spike_membrane_tiebreak = float(earliest_spike_membrane_tiebreak)
        self._configure_output_layer_for_readout()

    @property
    def layers(self) -> nn.ModuleList:
        return self.hidden_layers

    def _configure_output_layer_for_readout(self) -> None:
        _set_layer_spiking_enabled(self.output_layer, self.readout_mode != "final_membrane")

    def set_readout_mode(self, mode: str) -> None:
        self.readout_mode = normalize_readout_mode(mode)
        self._configure_output_layer_for_readout()

    def _hidden_forward(self, x_seq: torch.Tensor, *, record_hidden: bool = False):
        h = x_seq
        hidden_recs = []
        for layer in self.hidden_layers:
            if record_hidden:
                h, rec = layer.forward_sequence(h, record=True)
                hidden_recs.append(rec)
            else:
                h = layer.forward_sequence(h, record=False)
        return h, hidden_recs

    def _output_logits_from_record(self, out_seq: torch.Tensor, out_rec: dict[str, torch.Tensor]) -> torch.Tensor:
        soma_seq = out_rec.get("soma_state")
        spike_seq = out_rec.get("output", out_seq)
        return apply_readout(
            soma_seq=soma_seq,
            spike_seq=spike_seq,
            mode=self.readout_mode,
            training=bool(self.training),
            earliest_spike_rate_tiebreak=float(self.earliest_spike_rate_tiebreak),
            earliest_spike_membrane_tiebreak=float(self.earliest_spike_membrane_tiebreak),
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        h, _ = self._hidden_forward(x_seq, record_hidden=False)
        out_seq, out_rec = self.output_layer.forward_sequence(h, record=("soma_state", "output"))
        return self._output_logits_from_record(out_seq, out_rec)

    def forward_output_sequence(self, x_seq: torch.Tensor):
        h, _ = self._hidden_forward(x_seq, record_hidden=False)
        out_seq, out_rec = self.output_layer.forward_sequence(h, record=("soma_state", "output"))
        logits = self._output_logits_from_record(out_seq, out_rec)
        return logits, out_rec.get("soma_state")

    def forward_output_record(self, x_seq: torch.Tensor):
        h, _ = self._hidden_forward(x_seq, record_hidden=False)
        out_seq, out_rec = self.output_layer.forward_sequence(h, record=("soma_state", "output"))
        logits = self._output_logits_from_record(out_seq, out_rec)
        out_rec.setdefault("spike", out_rec.get("output", out_seq))
        out_rec.setdefault("spk", out_rec.get("output", out_seq))
        out_rec.setdefault("membrane", out_rec.get("soma_state"))
        return logits, out_rec

    def forward_with_records(self, x_seq: torch.Tensor):
        h, hidden_recs = self._hidden_forward(x_seq, record_hidden=True)
        out_seq, out_rec = self.output_layer.forward_sequence(h, record=("soma_state", "output"))
        logits = self._output_logits_from_record(out_seq, out_rec)
        out_rec.setdefault("spike", out_rec.get("output", out_seq))
        out_rec.setdefault("spk", out_rec.get("output", out_seq))
        out_rec.setdefault("membrane", out_rec.get("soma_state"))
        return logits, hidden_recs, out_rec

    def regularization_loss(self, lambda_ortho: float = 0.0, lambda_s: float = 0.0) -> torch.Tensor:
        from src.model.model_utils import s_complexity_mean

        loss = None
        for layer in list(self.hidden_layers) + [self.output_layer]:
            if hasattr(layer, "regularization_loss") and callable(getattr(layer, "regularization_loss")):
                l = layer.regularization_loss(lambda_ortho=lambda_ortho, lambda_s=0.0)  # type: ignore[attr-defined]
                loss = l if loss is None else (loss + l)
        if loss is None:
            param = next(self.parameters(), None)
            device = None if param is None else param.device
            loss = torch.zeros((), device=device)
        if float(lambda_s) != 0.0:
            loss = loss + float(lambda_s) * s_complexity_mean(self)
        return loss


def build_layer(model_name: str, input_dim: int, output_dim: int, cfg: SNNConfig) -> nn.Module:
    """Build one dense SNN layer from a normalized model token.

    Beginner note:
    - 논문 + Origin 코드가 있는 모델(dh-snn, tc-lif, ts-lif, d-rf)은
      `src/neurons/*_neuron.py`의 thin wrapper를 통해 저자 코드를 그대로
      재사용한다.
    - proposed 모델(my_*)만 프로젝트 정의식을 사용한다.
    """
    name = builder_name_from_any(model_name)
    spike_fn = SpikeFn(name=cfg.spike_surrogate, lens=0.5, gamma=0.5)

    if name == "lif":
        from src.neurons.LIF_neuron import LIFDenseLayer
        return LIFDenseLayer(input_dim, output_dim, v_th=cfg.v_th, spike_fn=spike_fn, recurrent=bool(cfg.recurrent))
    if name == "rf":
        from src.neurons.RF_neuron import RFDenseLayer
        return RFDenseLayer(input_dim, output_dim, dt=cfg.dt, threshold=cfg.v_th, reset_mode=cfg.rf_reset_mode, spike_fn=spike_fn, recurrent=bool(cfg.recurrent))
    if name == "tc-lif":
        from src.neurons.TC_LIF_neuron import TCLIFDenseLayer
        return TCLIFDenseLayer(input_dim, output_dim, v_th=cfg.v_th, spike_fn=spike_fn, recurrent=bool(cfg.recurrent))
    if name == "ts-lif":
        from src.neurons.TS_LIF_neuron import TSLIFDenseLayer
        return TSLIFDenseLayer(input_dim, output_dim, v_th=cfg.v_th, spike_fn=spike_fn, recurrent=bool(cfg.recurrent))
    if name == "dh-snn":
        from src.neurons.DH_SNN_neuron import DHSNNDenseLayer
        return DHSNNDenseLayer(input_dim, output_dim, branch=cfg.branch, v_th=cfg.v_th, spike_fn=spike_fn, recurrent=bool(cfg.recurrent))
    if name == "d-rf":
        from src.neurons.D_RF_neuron import DRFDenseLayer
        return DRFDenseLayer(input_dim, output_dim, branch=cfg.branch, th_len=cfg.th_len, v_pre=cfg.v_pre, spike_fn=spike_fn)
    if name == "my-dh-snn":
        from src.neurons.my_DH_SNN_neuron import MyDHSNNDenseLayer
        return MyDHSNNDenseLayer(input_dim, output_dim, branch=cfg.branch, S_min=cfg.S_min, S_max=cfg.S_max, v_th=cfg.v_th, spike_fn=spike_fn)
    if name == "my-r-dh-snn":
        from src.neurons.my_R_DH_SNN_neuron import MyReverseDHSNNDenseLayer
        return MyReverseDHSNNDenseLayer(input_dim, output_dim, branch=cfg.branch, S_min=cfg.S_min, S_max=cfg.S_max, v_th=cfg.v_th, spike_fn=spike_fn)
    if name == "my-d-rf":
        from src.neurons.my_D_RF_neuron import MyDRFDenseLayer
        return MyDRFDenseLayer(input_dim, output_dim, branch=cfg.branch, S_min=cfg.S_min, S_max=cfg.S_max, th_len=cfg.th_len, v_pre=cfg.v_pre, spike_fn=spike_fn)
    raise KeyError(f"Unknown model_name: {model_name}")


def build_snn_classifier(cfg: SNNConfig) -> SNNClassifier:
    hidden_dims = [int(v) for v in (cfg.hidden_dims if cfg.hidden_dims is not None else [cfg.hidden_dim])]
    dims = [int(cfg.input_dim)] + hidden_dims
    hidden_layers = [build_layer(cfg.model_name, dims[i], dims[i + 1], cfg) for i in range(max(0, len(dims) - 1))]
    readout_in = int(hidden_dims[-1]) if hidden_dims else int(cfg.input_dim)
    output_layer = build_layer(cfg.model_name, readout_in, int(cfg.num_classes), cfg)
    return SNNClassifier(hidden_layers, output_layer, readout_mode=str(getattr(cfg, "readout_mode", "final_membrane")))
