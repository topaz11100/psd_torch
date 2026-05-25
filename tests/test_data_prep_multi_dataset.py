import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data_prep

def _args(dataset):
    return argparse.Namespace(dataset=dataset, raw_data_root="/tmp/raw", prep_root="/tmp/prep", seed=7, force_overwrite=False, download=True, max_samples=11, prep_profile="project_standard", deap_label_axis="valence", deap_num_classes=3, shd_dt_ms=1.0, shd_max_time=1.2, ssc_dt_ms=1.0, ssc_max_time=1.0)

def test_validate_dataset_string_and_list():
    assert data_prep._validate_args(_args("mnist")).dataset == "mnist"
    assert data_prep._validate_args(_args(["mnist", "shd"])).dataset == ["mnist", "shd"]

def test_validate_dataset_list_errors():
    with pytest.raises(ValueError):
        data_prep._validate_args(_args([]))
    with pytest.raises(ValueError):
        data_prep._validate_args(_args(["mnist", 1]))

def test_run_data_prep_serial():
    calls = []
    def fake_prepare(dataset, **kwargs):
        calls.append((dataset, kwargs))
        return f"/tmp/prep/{dataset}"
    summary = data_prep.run_data_prep(_args(["mnist", "shd"]), fake_prepare)
    assert [c[0] for c in calls] == ["mnist", "shd"]
    assert calls[0][1]["seed"] == calls[1][1]["seed"] == 7
    assert summary["outputs"][0]["dataset"] == "mnist"
