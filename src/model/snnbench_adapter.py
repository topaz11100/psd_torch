from __future__ import annotations

from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn

from src.model.snnbench_backbones import (
    TDBatchNorm,
    VGG_SNN,
    make_snnbench_lif_node,
    make_snnbench_rf_node,
    sew_function,
    spiking_resnet18,
)


class SNNBenchCNNAdapter(nn.Module):
    def __init__(
        self,
        *,
        spec,
        input_dim: int,
        sequence_length: int,
        input_shape: Sequence[int],
        num_classes: int,
        v_th: float,
        layer_record_cls,
        forward_result_cls,
    ):
        super().__init__()

        self.spec = spec
        self.input_dim = int(input_dim)
        self.sequence_length = int(sequence_length)
        self.input_shape = tuple(int(v) for v in input_shape)
        self.num_classes = int(num_classes)
        self.output_sequence_length = int(sequence_length)

        self.layer_record_cls = layer_record_cls
        self.forward_result_cls = forward_result_cls

        if len(self.input_shape) != 4:
            raise ValueError(f"reference-topology CNN adapter expects input_shape=(T,C,H,W), got {self.input_shape}")

        _time_steps, in_channels, _height, _width = self.input_shape

        # Prepared frames are consumed directly. data_prep already represents
        # static images as repeated frames and event datasets as integrated
        # frame sequences; no additional stochastic encoding or fixed input adapter is inserted.
        self.output_sn = None

        if spec.family == "cnn_lif":
            self.neuron_tag = "lif"
            node_factory = make_snnbench_lif_node
        elif spec.family == "cnn_rf":
            self.neuron_tag = "rf"
            node_factory = make_snnbench_rf_node
        else:
            raise ValueError(f"reference-topology CNN adapter supports cnn_lif/cnn_rf, got {spec.family!r}")

        node_kwargs = {
            "v_threshold": float(v_th),
            "reset_mode": spec.reset_mode,
            "trainable_threshold": bool(getattr(spec, "trainable_threshold", False)),
        }

        if spec.backbone == "vgg11":
            self.core = VGG_SNN(
                in_channels=int(in_channels),
                num_classes=int(num_classes),
                depth=11,
                bias=True,
                single_step_neuron=node_factory,
                **node_kwargs,
            )

        elif spec.backbone == "resnet18":
            self.core = spiking_resnet18(
                pretrained=False,
                progress=True,
                norm_layer=TDBatchNorm,
                single_step_neuron=node_factory,
                num_classes=int(num_classes),
                in_channels=int(in_channels),
                **node_kwargs,
            )

            self.output_sn = node_factory(
                channels=int(num_classes),
                **node_kwargs,
            )

        else:
            raise ValueError(f"Unsupported reference-topology backbone: {spec.backbone!r}")

    def reset_state(self) -> None:
        for module in self.modules():
            if module is self:
                continue
            reset = getattr(module, "reset", None)
            if callable(reset):
                try:
                    reset()
                except TypeError:
                    pass


    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(x)

        if tensor.ndim != 5:
            raise ValueError(f"reference-topology CNN expects input shape (B,T,C,H,W), got {tuple(tensor.shape)}")

        expected = self.input_shape
        actual = tuple(int(v) for v in tensor.shape[1:])

        if actual != expected:
            raise ValueError(f"reference-topology CNN expected input_shape={expected}, got {actual}")

        return tensor.contiguous()

    def _maybe_encode_frame(self, frame: torch.Tensor) -> torch.Tensor:
        return frame.float()

    def _forward_vgg_one_step(self, frame: torch.Tensor, *, capture_hidden: bool):
        records = []

        x = frame
        lif_index = 0
        for feature_index, module in enumerate(self.core.features):
            before = x
            x = module(x)
            if getattr(module, "is_snnbench_spiking_node", False):
                lif_index += 1
                if capture_hidden:
                    records.append(
                        (
                            f"vgg11_{self.neuron_tag}_{lif_index:02d}",
                            getattr(module, "v", before),
                            x,
                            before,
                        )
                    )
            elif capture_hidden and isinstance(module, nn.MaxPool2d):
                records.append(
                    (
                        f"vgg11_layer_{feature_index + 1:02d}_maxpool",
                        x,
                        None,
                        None,
                        "hidden",
                        "module_output",
                    )
                )

        x = self.core.fixpool(x)

        classifier_spike_indices = [
            idx for idx, module in enumerate(self.core.classifier)
            if getattr(module, "is_snnbench_spiking_node", False)
        ]
        if not classifier_spike_indices:
            raise RuntimeError("VGG classifier must contain at least one spiking output node.")
        output_spike_index = classifier_spike_indices[-1]
        fc_spike_index = 0
        output_membrane = None
        output_spike = None

        for module_index, module in enumerate(self.core.classifier):
            before = x
            x = module(x)
            if getattr(module, "is_snnbench_spiking_node", False):
                fc_spike_index += 1
                membrane = getattr(module, "v", before)
                if module_index == output_spike_index:
                    output_membrane = membrane
                    output_spike = x
                elif capture_hidden:
                    records.append(
                        (
                            f"vgg11_fc{fc_spike_index}_{self.neuron_tag}",
                            membrane,
                            x,
                            before,
                        )
                    )

        if output_membrane is None or output_spike is None:
            raise RuntimeError("VGG classifier did not produce an output spike record.")
        return output_membrane, output_spike, records

    def _forward_resnet_block(self, block, x: torch.Tensor, *, block_name: str, capture_hidden: bool):
        records = []
        identity = x

        out = block.conv1(x)
        out = block.bn1(out)
        conv1_current = out
        out = block.sn1(out)

        if capture_hidden:
            records.append(
                (
                    f"{block_name}_conv1",
                    getattr(block.sn1, "v", conv1_current),
                    out,
                    conv1_current,
                )
            )

        out = block.conv2(out)
        out = block.bn2(out)
        conv2_current = out
        out = block.sn2(out)

        if block.downsample is not None:
            identity = block.downsample(x)
            if getattr(block, "downsample_sn", None) is not None:
                identity = block.downsample_sn(identity)

        merged = sew_function(out, identity, getattr(block, "cnf", "ADD"))

        if capture_hidden:
            records.append(
                (
                    f"{block_name}_residual_add",
                    getattr(block.sn2, "v", conv2_current),
                    merged,
                    conv2_current,
                )
            )

        return merged, records

    def _forward_resnet_one_step(self, frame: torch.Tensor, *, capture_hidden: bool):
        records = []
        x = frame

        block_index = 0
        for layer in (self.core.layer1, self.core.layer2, self.core.layer3, self.core.layer4):
            for block in layer:
                block_index += 1
                x, block_records = self._forward_resnet_block(
                    block,
                    x,
                    block_name=f"resnet18_block_{block_index:02d}",
                    capture_hidden=capture_hidden,
                )
                records.extend(block_records)

        x = self.core.avgpool(x)
        x = torch.flatten(x, 1)

        output_current = self.core.fc(x)

        if self.output_sn is None:
            raise RuntimeError("ResNet output spiking node is not initialized.")

        output_spike = self.output_sn(output_current)
        output_membrane = getattr(self.output_sn, "v", output_current)

        return output_membrane, output_spike, records

    def _forward_one_step(self, frame: torch.Tensor, *, capture_hidden: bool):
        if self.spec.backbone == "vgg11":
            return self._forward_vgg_one_step(frame, capture_hidden=capture_hidden)

        if self.spec.backbone == "resnet18":
            return self._forward_resnet_one_step(frame, capture_hidden=capture_hidden)

        raise ValueError(f"Unsupported backbone: {self.spec.backbone!r}")

    def forward(self, input_sequence: torch.Tensor, *, capture_hidden: bool = False):
        prepared_input = self._prepare_input(input_sequence)

        self.reset_state()

        output_membrane_steps = []
        output_spike_steps = []
        hidden_buckets = {}

        for t in range(int(prepared_input.shape[1])):
            frame = prepared_input[:, t]
            frame = self._maybe_encode_frame(frame)

            output_membrane, output_spike, records = self._forward_one_step(
                frame,
                capture_hidden=capture_hidden,
            )

            output_membrane_steps.append(output_membrane)
            output_spike_steps.append(output_spike)

            if capture_hidden:
                for record in records:
                    if len(record) == 4:
                        name, membrane, spike, layer_input = record
                        signal_kind = None
                        series = None
                    elif len(record) == 6:
                        name, membrane, spike, layer_input, signal_kind, series = record
                    else:
                        raise ValueError(f"Unexpected VGG record format: {len(record)} fields")

                    bucket = hidden_buckets.setdefault(
                        name,
                        {
                            "membrane": [],
                            "spike": [],
                            "layer_input": [],
                            "signal_kind": signal_kind,
                            "series": series,
                        },
                    )

                    bucket["membrane"].append(membrane)

                    if spike is not None:
                        bucket["spike"].append(spike)

                    if layer_input is not None:
                        bucket["layer_input"].append(layer_input)

        output_membrane_seq = torch.stack(output_membrane_steps, dim=1).contiguous()
        output_spike_seq = torch.stack(output_spike_steps, dim=1).contiguous()

        hidden_records = []
        if capture_hidden:
            for name, bucket in hidden_buckets.items():
                membrane_seq = torch.stack(bucket["membrane"], dim=1).contiguous()

                if bucket["spike"]:
                    spike_seq = torch.stack(bucket["spike"], dim=1).contiguous()
                else:
                    spike_seq = membrane_seq

                if bucket["layer_input"]:
                    layer_input_seq = torch.stack(bucket["layer_input"], dim=1).contiguous()
                else:
                    layer_input_seq = None

                record = self.layer_record_cls(
                    layer_name=name,
                    membrane=membrane_seq,
                    spike=spike_seq,
                    layer_input=layer_input_seq,
                )

                if bucket.get("signal_kind") and bucket.get("series"):
                    setattr(record, "signal_kind", bucket["signal_kind"])
                    setattr(record, "series", bucket["series"])

                hidden_records.append(record)

        output_record = self.layer_record_cls(
            layer_name="output",
            membrane=output_membrane_seq,
            spike=output_spike_seq,
            layer_input=output_membrane_seq,
        )
        setattr(output_record, "readout_mem", output_membrane_seq)

        return self.forward_result_cls(
            hidden_records=hidden_records,
            output_record=output_record,
            input_record=prepared_input,
        )

    def iter_named_layers(self) -> Iterable[tuple[str, nn.Module]]:
        if self.spec.backbone == "vgg11":
            tag = self.neuron_tag
            spike_index = 0
            for feature_index, module in enumerate(self.core.features):
                if isinstance(module, nn.Conv2d):
                    name = f"vgg11_layer_{feature_index + 1:02d}_conv"
                elif isinstance(module, TDBatchNorm):
                    name = f"vgg11_layer_{feature_index + 1:02d}_tdbn"
                elif isinstance(module, nn.MaxPool2d):
                    name = f"vgg11_layer_{feature_index + 1:02d}_maxpool"
                elif getattr(module, "is_snnbench_spiking_node", False):
                    spike_index += 1
                    name = f"vgg11_{tag}_{spike_index:02d}"
                else:
                    name = f"vgg11_layer_{feature_index + 1:02d}_{module.__class__.__name__.lower()}"
                yield name, module

            yield "vgg11_fixpool", self.core.fixpool
            classifier_spike_indices = [idx for idx, module in enumerate(self.core.classifier) if getattr(module, "is_snnbench_spiking_node", False)]
            output_spike_index = classifier_spike_indices[-1] if classifier_spike_indices else -1
            fc_spike_index = 0
            fc_linear_index = 0
            for module_index, module in enumerate(self.core.classifier):
                if isinstance(module, nn.Flatten):
                    name = "vgg11_flatten"
                elif isinstance(module, nn.Linear):
                    fc_linear_index += 1
                    name = "vgg11_output_fc" if module_index > output_spike_index else f"vgg11_fc{fc_linear_index}"
                elif getattr(module, "is_snnbench_spiking_node", False):
                    fc_spike_index += 1
                    name = "output" if module_index == output_spike_index else f"vgg11_fc{fc_spike_index}_{tag}"
                elif isinstance(module, nn.Dropout):
                    name = f"vgg11_dropout_{module_index:02d}"
                else:
                    name = f"vgg11_classifier_{module_index:02d}_{module.__class__.__name__.lower()}"
                yield name, module
            return

        if self.spec.backbone == "resnet18":
            block_index = 0
            for layer in (self.core.layer1, self.core.layer2, self.core.layer3, self.core.layer4):
                for block in layer:
                    block_index += 1
                    yield f"resnet18_block_{block_index:02d}_conv1", block.sn1
                    yield f"resnet18_block_{block_index:02d}_residual_add", block.sn2

            yield "resnet18_fc", self.core.fc
            if self.output_sn is not None:
                yield "output", self.output_sn
            return

        raise ValueError(f"Unsupported backbone: {self.spec.backbone!r}")


    def iter_named_hidden_layers(self) -> Iterable[tuple[str, nn.Module]]:
        if self.spec.backbone == "vgg11":
            tag = self.neuron_tag
            spike_index = 0
            for module in self.core.features:
                if getattr(module, "is_snnbench_spiking_node", False):
                    spike_index += 1
                    yield f"vgg11_{tag}_{spike_index:02d}", module
            classifier_spike_indices = [idx for idx, module in enumerate(self.core.classifier) if getattr(module, "is_snnbench_spiking_node", False)]
            output_spike_index = classifier_spike_indices[-1] if classifier_spike_indices else -1
            fc_spike_index = 0
            for module_index, module in enumerate(self.core.classifier):
                if getattr(module, "is_snnbench_spiking_node", False):
                    fc_spike_index += 1
                    if module_index != output_spike_index:
                        yield f"vgg11_fc{fc_spike_index}_{tag}", module
            return

        if self.spec.backbone == "resnet18":
            block_index = 0
            for layer in (self.core.layer1, self.core.layer2, self.core.layer3, self.core.layer4):
                for block in layer:
                    block_index += 1
                    yield f"resnet18_block_{block_index:02d}_conv1", block.sn1
                    yield f"resnet18_block_{block_index:02d}_residual_add", block.sn2

            return

        raise ValueError(f"Unsupported backbone: {self.spec.backbone!r}")

    def _lif_backend_name(self) -> str:
        for module in self.modules():
            if getattr(module, "is_snnbench_spiking_node", False) and getattr(module, "snnbench_neuron_tag", None) == "lif":
                return str(getattr(module, "psd_neuron_backend", module.__class__.__module__))
        return "not_applicable"

    def model_metadata(self) -> dict[str, Any]:
        return {
            "raw_model_token": self.spec.raw_token,
            "canonical_model_token": self.spec.canonical_token,
            "family": self.spec.family,
            "backbone": self.spec.backbone,
            "cnn_source": "reference_topology_only_project_adapter",
            "input_dim": self.input_dim,
            "sequence_length": self.sequence_length,
            "output_sequence_length": self.output_sequence_length,
            "num_classes": self.num_classes,
            "cnn_input_shape": list(self.input_shape),
            "arch_spec": f"reference_topology_{self.spec.backbone}",
            "hidden_spec": None,
            "cnn_head": "project_temporal_output_wrapper",
            "cnn_neuron": self.neuron_tag,
            "spiking_runtime": "pure_torch_by_default_spikingjelly_with_explicit_probe_or_force",
            "vgg_depth": 11 if self.spec.backbone == "vgg11" else None,
            "resnet_depth": 18 if self.spec.backbone == "resnet18" else None,
            "sew_resnet_connect_function": getattr(self.core, "cnf", None) if self.spec.backbone == "resnet18" else None,
            "reference_backbone_contract": "topology_only_from_reference_SNNs; input geometry comes from prep_data",
            "resnet_input_projection": "none_first_basicblock_consumes_prepared_frame_channels" if self.spec.backbone == "resnet18" else None,
            "vgg_input_policy": "first_vgg_conv_consumes_prepared_frame_channels_directly" if self.spec.backbone == "vgg11" else None,
            "cnn_lif_backend": self._lif_backend_name() if self.spec.family == "cnn_lif" else None,
            "cnn_rf_backend": "torch" if self.spec.family == "cnn_rf" else None,
            "note": "Reference topology is wrapped for PSD ForwardResult; prep_data frames are consumed directly with no reference input front-end.",
        }


def build_snnbench_cnn_classifier(
    *,
    spec,
    input_dim: int,
    sequence_length: int,
    num_classes: int,
    input_shape: Sequence[int] | None,
    v_th: float,
    layer_record_cls,
    forward_result_cls,
):
    if input_shape is None:
        raise ValueError("reference-topology CNN adapter requires input_shape=(T,C,H,W).")

    return SNNBenchCNNAdapter(
        spec=spec,
        input_dim=int(input_dim),
        sequence_length=int(sequence_length),
        input_shape=input_shape,
        num_classes=int(num_classes),
        v_th=float(v_th),
        layer_record_cls=layer_record_cls,
        forward_result_cls=forward_result_cls,
    )

