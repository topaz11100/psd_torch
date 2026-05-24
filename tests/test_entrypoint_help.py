from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


CASES = [
    ('src/data_prep.py', 'config/data_prep.json'),
    ('src/model_training.py', 'config/model_training.json'),
    ('src/dataset_psd.py', 'config/dataset_psd.json'),
    ('src/dataset_fft.py', 'config/dataset_fft.json'),
    ('src/psd_analysis.py', 'config/psd_analysis.json'),
    ('src/element_psd.py', 'config/element_psd.json'),
    ('src/2d_fft_analysis.py', 'config/fft2d_analysis.json'),
    ('src/plotting.py', 'config/plotting.json'),
]


@pytest.mark.parametrize(('script', 'config_path'), CASES)
def test_help_entrypoint(script: str, config_path: str, tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env.setdefault('PYTHONHASHSEED', '0')
    out_path = tmp_path / 'help.out'
    err_path = tmp_path / 'help.err'
    with out_path.open('w', encoding='utf-8') as stdout, err_path.open('w', encoding='utf-8') as stderr:
        proc = subprocess.run(
            [sys.executable, '-S', script, '--config', config_path, '--help'],
            cwd=root,
            text=True,
            stdout=stdout,
            stderr=stderr,
            timeout=60,
            env=env,
        )
    output = (out_path.read_text(encoding='utf-8') + err_path.read_text(encoding='utf-8')).lower()
    assert proc.returncode == 0, (script, output)
    assert 'usage' in output
    assert 'modulenotfounderror' not in output
    assert "no module named 'torch'" not in output
    assert "no module named 'h5py'" not in output
