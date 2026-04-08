from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class ModelSpec:
    canonical: str
    builder_name: str
    aliases: Tuple[str, ...]
    signal_manifest: Mapping[str, Tuple[str, ...]]
    timing_factor_keys: Tuple[str, ...]
    weight_group_keys: Tuple[str, ...]
    tracked_state_kinds: Tuple[str, ...]
    filter_property_status: str = "unsupported"


_COMMON_INPUT_KEYS = ("dendrite_input", "soma_input")
_COMMON_SPIKE_KEYS = ("spk",)


def _manifest(
    *,
    neuron_state_keys: Tuple[str, ...],
    layer_state_keys: Tuple[str, ...],
    layer_membrane_keys: Tuple[str, ...] = (),
) -> Mapping[str, Tuple[str, ...]]:
    return {
        "neuron_input_keys": _COMMON_INPUT_KEYS,
        "neuron_state_keys": tuple(neuron_state_keys),
        "neuron_spike_keys": _COMMON_SPIKE_KEYS,
        "layer_input_keys": _COMMON_INPUT_KEYS,
        "layer_state_keys": tuple(layer_state_keys),
        "layer_membrane_keys": tuple(layer_membrane_keys),
        "layer_spike_keys": _COMMON_SPIKE_KEYS,
    }


MODEL_SPECS: Tuple[ModelSpec, ...] = (
    ModelSpec(
        canonical="LIF",
        builder_name="lif",
        aliases=("lif", "LIF", "lif_neuron"),
        signal_manifest=_manifest(
            neuron_state_keys=("mem",),
            layer_state_keys=("mem",),
            layer_membrane_keys=("mem",),
        ),
        timing_factor_keys=("alpha",),
        weight_group_keys=("fc_weight",),
        tracked_state_kinds=("mem",),
    ),
    ModelSpec(
        canonical="RF",
        builder_name="rf",
        aliases=("rf", "RF", "vanilla_rf", "vanilla-RF", "vanillaRF"),
        signal_manifest=_manifest(
            neuron_state_keys=("x", "y"),
            layer_state_keys=("x", "y"),
            layer_membrane_keys=("x",),
        ),
        timing_factor_keys=("rho", "f_cyc_per_sample"),
        weight_group_keys=("fc_weight",),
        tracked_state_kinds=("x", "y"),
        filter_property_status="available",
    ),
    ModelSpec(
        canonical="TC_LIF",
        builder_name="tc-lif",
        aliases=("tc-lif", "tc_lif", "tclif", "TC_LIF", "TC-LIF", "tc"),
        signal_manifest=_manifest(
            neuron_state_keys=("v1", "v2"),
            layer_state_keys=("v1", "v2"),
            layer_membrane_keys=("v1", "v2"),
        ),
        timing_factor_keys=("decay_factor_0", "decay_factor_1"),
        weight_group_keys=("fc_weight",),
        tracked_state_kinds=("v1", "v2"),
    ),
    ModelSpec(
        canonical="TS_LIF",
        builder_name="ts-lif",
        aliases=("ts-lif", "ts_lif", "tslif", "TS_LIF", "TS-LIF", "ts"),
        signal_manifest=_manifest(
            neuron_state_keys=("vd", "vs"),
            layer_state_keys=("vd", "vs"),
            layer_membrane_keys=("vd", "vs"),
        ),
        timing_factor_keys=("alpha1", "alpha2", "beta1", "beta2", "gamma1", "gamma2", "kappa"),
        weight_group_keys=("fc_weight",),
        tracked_state_kinds=("vd", "vs"),
    ),
    ModelSpec(
        canonical="DH_SNN",
        builder_name="dh-snn",
        aliases=("dh-snn", "dh_snn", "dhsnn", "DH_SNN", "DH-SNN", "dh"),
        signal_manifest=_manifest(
            neuron_state_keys=("d_state", "mem"),
            layer_state_keys=("d_state", "mem"),
            layer_membrane_keys=("mem",),
        ),
        timing_factor_keys=("tau_n", "tau_m"),
        weight_group_keys=("fc_weight",),
        tracked_state_kinds=("d_state", "mem"),
    ),
    ModelSpec(
        canonical="D_RF",
        builder_name="d-rf",
        aliases=("d-rf", "d_rf", "drf", "D_RF", "D-RF"),
        signal_manifest=_manifest(
            neuron_state_keys=("u", "v", "pre_hist", "V_th"),
            layer_state_keys=("u", "v", "pre_hist", "V_th"),
            layer_membrane_keys=("membrane", "u"),
        ),
        timing_factor_keys=("tau", "omega", "alpha_th"),
        weight_group_keys=("fc_weight", "C"),
        tracked_state_kinds=("u", "v", "pre_hist", "V_th"),
    ),
    ModelSpec(
        canonical="my_DH_SNN",
        builder_name="my-dh-snn",
        aliases=("my-dh-snn", "my_dh_snn", "my_DH_SNN", "my-DH-SNN", "my-dh", "my_dh"),
        signal_manifest=_manifest(
            neuron_state_keys=("d_state", "mem"),
            layer_state_keys=("d_state", "mem"),
            layer_membrane_keys=("mem",),
        ),
        timing_factor_keys=("alpha_branch", "beta_soma"),
        weight_group_keys=("W",),
        tracked_state_kinds=("d_state", "mem"),
    ),
    ModelSpec(
        canonical="my_R_DH_SNN",
        builder_name="my-r-dh-snn",
        aliases=("my-r-dh-snn", "my_r_dh_snn", "my_R_DH_SNN", "my-R-DH-SNN", "my-r-snn", "r-dh-snn", "r-snn", "my_r_dh", "my-r-dh"),
        signal_manifest=_manifest(
            neuron_state_keys=("d_state", "mem"),
            layer_state_keys=("d_state", "mem"),
            layer_membrane_keys=("mem",),
        ),
        timing_factor_keys=("alpha_branch",),
        weight_group_keys=("W_in", "W_mix"),
        tracked_state_kinds=("d_state", "mem"),
        filter_property_status="available",
    ),
    ModelSpec(
        canonical="my_D_RF",
        builder_name="my-d-rf",
        aliases=("my-d-rf", "my_d_rf", "my_D_RF", "my-D-RF", "my-drf", "my_drf"),
        signal_manifest=_manifest(
            neuron_state_keys=("u", "v", "p_hist", "V_th"),
            layer_state_keys=("u", "v", "p_hist", "V_th"),
            layer_membrane_keys=("membrane", "u"),
        ),
        timing_factor_keys=("tau", "omega", "a_kernel"),
        weight_group_keys=("fc_weight",),
        tracked_state_kinds=("u", "v", "p_hist", "V_th"),
        filter_property_status="available",
    ),
)


_ALIAS_TO_SPEC: Dict[str, ModelSpec] = {}
_CANONICAL_TO_SPEC: Dict[str, ModelSpec] = {}
for spec in MODEL_SPECS:
    _CANONICAL_TO_SPEC[spec.canonical] = spec
    _ALIAS_TO_SPEC[spec.canonical.lower()] = spec
    _ALIAS_TO_SPEC[spec.builder_name.lower()] = spec
    for alias in spec.aliases:
        _ALIAS_TO_SPEC[str(alias).lower()] = spec


def canonical_model_names() -> List[str]:
    return [spec.canonical for spec in MODEL_SPECS]


def all_builder_names() -> List[str]:
    return [spec.builder_name for spec in MODEL_SPECS]


def get_model_spec(name: str) -> ModelSpec:
    key = str(name).strip().lower()
    if key not in _ALIAS_TO_SPEC:
        raise KeyError(f"Unknown model name: {name}. Available: {canonical_model_names()}")
    return _ALIAS_TO_SPEC[key]


def resolve_model_name(name: str) -> str:
    return get_model_spec(name).canonical


def builder_name_from_any(name: str) -> str:
    return get_model_spec(name).builder_name


def normalize_model_list(models: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for name in models:
        canonical = resolve_model_name(name)
        if canonical not in seen:
            out.append(canonical)
            seen.add(canonical)
    return out


_SPIKE_DRIVING_MEMBRANE_KEY_BY_CANONICAL: Dict[str, str] = {
    "LIF": "mem",
    "RF": "x",
    "TC_LIF": "v2",
    "TS_LIF": "vs",
    "DH_SNN": "mem",
    "D_RF": "membrane",
    "my_DH_SNN": "mem",
    "my_R_DH_SNN": "mem",
    "my_D_RF": "membrane",
}


def spike_driving_membrane_key(name: str) -> str:
    canonical = resolve_model_name(name)
    try:
        return _SPIKE_DRIVING_MEMBRANE_KEY_BY_CANONICAL[canonical]
    except KeyError as exc:
        raise KeyError(f"No spike-driving membrane key registered for model: {name}") from exc


def is_variable_branch_model(name: str) -> bool:
    return resolve_model_name(name) in {"my_DH_SNN", "my_R_DH_SNN", "my_D_RF"}


def filter_property_status(name: str) -> str:
    return get_model_spec(name).filter_property_status
