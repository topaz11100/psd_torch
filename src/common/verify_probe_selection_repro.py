from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Allow direct execution via:
#   python src/common/verify_probe_selection_repro.py
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.common.probe_selection import (
    canonical_label_probe_order,
    flatten_scope_indices,
    probe_scope_signature,
    probe_union_indices,
    select_fixed_probe_scopes,
)


class ToyDataset:
    def __init__(self, labels: List[int]):
        self._labels = [int(v) for v in labels]

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int):
        return int(idx), int(self._labels[int(idx)])


@dataclass(frozen=True)
class Scenario:
    name: str
    model: str
    readout_mode: str
    exp_name: str
    timestamp: str
    out_root: str
    probe_plot: int
    extra: str


SCENARIOS: Tuple[Scenario, ...] = (
    Scenario("rf-final", "rf", "final_membrane", "psd_analysis_shd", "20260329_100000", "/tmp/out_a", 1, "baseline"),
    Scenario("rf-earliest", "rf", "earliest_spike", "psd_analysis_shd_alt", "20260329_110000", "/tmp/out_b", 0, "readout_swap"),
    Scenario("lif-rate", "lif_structclip", "max_rate", "other_name", "20260330_010000", "/tmp/out_c", 1, "variant_swap"),
    Scenario("canon-final", "my_D_RF", "final_membrane", "another_prefix", "manual_ts", "/tmp/out_d", 0, "canonical_model"),
    Scenario("rf-rate", "rf_clip", "max_rate", "repro_case", "20260401_090000", "/tmp/out_e", 1, "clip_variant"),
)


def _toy_dataset() -> ToyDataset:
    labels: List[int] = []
    labels.extend([0] * 7)
    labels.extend([1] * 6)
    labels.extend([2] * 5)
    labels.extend([3] * 4)
    labels.extend([4] * 3)
    return ToyDataset(labels)



def _snapshot(dataset: ToyDataset, *, split_name: str, seed: int, same_n: int, balanced_n: int, scenario: Scenario) -> Dict[str, object]:
    scopes = select_fixed_probe_scopes(
        dataset,
        5,
        split_name=str(split_name),
        base_seed=int(seed),
        same_label_n=int(same_n),
        balanced_n=int(balanced_n),
    )
    return {
        "scenario": scenario.name,
        "same_label": scopes["same_label"],
        "balanced_global": scopes["balanced_global"],
        "same_label_flat": flatten_scope_indices(scopes, "same_label"),
        "balanced_global_flat": flatten_scope_indices(scopes, "balanced_global"),
        "probe_union": probe_union_indices(scopes),
        "signature": probe_scope_signature(scopes),
    }



def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)



def main() -> None:
    dataset = _toy_dataset()
    seeds = (0, 7, 19)
    count_pairs = ((1, 1), (2, 3), (3, 2), (4, 4))
    scenario_comparisons = 0
    equal_count_scope_checks = 0
    prefix_checks = 0
    seed_effect_checks = 0

    for split_name in ("train", "test"):
        for seed in seeds:
            for same_n, balanced_n in count_pairs:
                baseline = _snapshot(dataset, split_name=split_name, seed=seed, same_n=same_n, balanced_n=balanced_n, scenario=SCENARIOS[0])
                for scenario in SCENARIOS[1:]:
                    current = _snapshot(dataset, split_name=split_name, seed=seed, same_n=same_n, balanced_n=balanced_n, scenario=scenario)
                    _assert(current["same_label"] == baseline["same_label"], f"same_label mismatch: split={split_name}, seed={seed}, same_n={same_n}, balanced_n={balanced_n}, scenario={scenario.name}")
                    _assert(current["balanced_global"] == baseline["balanced_global"], f"balanced_global mismatch: split={split_name}, seed={seed}, same_n={same_n}, balanced_n={balanced_n}, scenario={scenario.name}")
                    _assert(current["same_label_flat"] == baseline["same_label_flat"], f"same_label_flat mismatch: split={split_name}, seed={seed}, same_n={same_n}, balanced_n={balanced_n}, scenario={scenario.name}")
                    _assert(current["balanced_global_flat"] == baseline["balanced_global_flat"], f"balanced_global_flat mismatch: split={split_name}, seed={seed}, same_n={same_n}, balanced_n={balanced_n}, scenario={scenario.name}")
                    _assert(current["probe_union"] == baseline["probe_union"], f"probe_union mismatch: split={split_name}, seed={seed}, same_n={same_n}, balanced_n={balanced_n}, scenario={scenario.name}")
                    _assert(current["signature"] == baseline["signature"], f"signature mismatch: split={split_name}, seed={seed}, same_n={same_n}, balanced_n={balanced_n}, scenario={scenario.name}")
                    scenario_comparisons += 1

            # equal-count case: same_label and balanced_global must be identical per label
            for n in (1, 2, 3, 4):
                scopes = select_fixed_probe_scopes(dataset, 5, split_name=split_name, base_seed=seed, same_label_n=n, balanced_n=n)
                for label in sorted(scopes["same_label"].keys()):
                    _assert(
                        scopes["same_label"][label] == scopes["balanced_global"][label],
                        f"equal-count scope mismatch: split={split_name}, seed={seed}, n={n}, label={label}",
                    )
                    equal_count_scope_checks += 1

            # prefix stability: larger requested count must extend the same canonical order
            canonical = canonical_label_probe_order(dataset, 5, split_name=split_name, base_seed=seed)
            for small_n, large_n in ((1, 2), (2, 4), (1, 4)):
                scopes_small = select_fixed_probe_scopes(dataset, 5, split_name=split_name, base_seed=seed, same_label_n=small_n, balanced_n=small_n)
                scopes_large = select_fixed_probe_scopes(dataset, 5, split_name=split_name, base_seed=seed, same_label_n=large_n, balanced_n=large_n)
                for label, ordered in canonical.items():
                    expected_small = ordered[: min(small_n, len(ordered))]
                    expected_large = ordered[: min(large_n, len(ordered))]
                    _assert(scopes_small["same_label"][label] == expected_small, f"same_label prefix mismatch: split={split_name}, seed={seed}, label={label}, small_n={small_n}")
                    _assert(scopes_small["balanced_global"][label] == expected_small, f"balanced_global prefix mismatch: split={split_name}, seed={seed}, label={label}, small_n={small_n}")
                    _assert(scopes_large["same_label"][label] == expected_large, f"same_label large prefix mismatch: split={split_name}, seed={seed}, label={label}, large_n={large_n}")
                    _assert(scopes_large["balanced_global"][label] == expected_large, f"balanced_global large prefix mismatch: split={split_name}, seed={seed}, label={label}, large_n={large_n}")
                    _assert(expected_large[: len(expected_small)] == expected_small, f"non-prefix canonical order: split={split_name}, seed={seed}, label={label}, small_n={small_n}, large_n={large_n}")
                    prefix_checks += 1

    # seed must have an observable effect on the selected probe order.
    for split_name in ("train", "test"):
        sig_a = _snapshot(dataset, split_name=split_name, seed=seeds[0], same_n=4, balanced_n=4, scenario=SCENARIOS[0])["signature"]
        sig_b = _snapshot(dataset, split_name=split_name, seed=seeds[1], same_n=4, balanced_n=4, scenario=SCENARIOS[0])["signature"]
        _assert(sig_a != sig_b, f"different seeds unexpectedly produced the same signature for split={split_name}")
        seed_effect_checks += 1

    print(
        "probe_selection_verification_passed "
        f"scenario_comparisons={scenario_comparisons} "
        f"equal_count_scope_checks={equal_count_scope_checks} "
        f"prefix_checks={prefix_checks} "
        f"seed_effect_checks={seed_effect_checks}"
    )


if __name__ == "__main__":
    main()
