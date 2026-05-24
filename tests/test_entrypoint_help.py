from __future__ import annotations
import subprocess, sys
from pathlib import Path

CASES=[
("src/data_prep.py","config/data_prep.json"),
("src/model_training.py","config/model_training.json"),
("src/dataset_psd.py","config/dataset_psd.json"),
("src/dataset_fft.py","config/dataset_fft.json"),
("src/psd_analysis.py","config/psd_analysis.json"),
("src/element_psd.py","config/element_psd.json"),
("src/2d_fft_analysis.py","config/fft2d_analysis.json"),
("src/plotting.py","config/plotting.json"),
]

def test_help_entrypoints():
    root=Path(__file__).resolve().parents[1]
    for script,cfg in CASES:
        proc=subprocess.run([sys.executable,script,'--config',cfg,'--help'],cwd=root,text=True,capture_output=True)
        assert proc.returncode==0, (script,proc.stderr)
        out=(proc.stdout+proc.stderr).lower()
        assert 'usage' in out
        assert 'modulenotfounderror' not in out
        assert "no module named 'torch'" not in out
        assert "no module named 'h5py'" not in out
