import pytest

import json, subprocess, sys, os
from pathlib import Path


def test_examples_structure_and_json_parse():
    assert Path('examples/README.md').exists()
    assert Path('examples/bash').exists()
    assert Path('examples/configs/commented').exists()
    assert Path('examples/configs/runnable').exists()
    for p in Path('examples/configs/runnable').glob('*.json'):
        json.loads(p.read_text())


def test_bash_syntax_and_cli_guard():
    for p in Path('examples/bash').glob('*.sh'):
        cp=subprocess.run(['bash','-n',str(p)], capture_output=True)
        assert cp.returncode==0
        txt=p.read_text()
        assert 'python -m psd_snn.cli.' in txt or p.name=='00_env.sh'
        for bad in ['analysis_2d_fft','reinterpretation','label_single_excluding_balanced']:
            assert bad not in txt


def test_cli_help_smoke():
    pytest.importorskip("torch")
    for mod in ['psd_snn.cli.train','psd_snn.cli.analyze_signal','psd_snn.cli.analyze_fft2d','psd_snn.cli.plot_artifacts']:
        cp=subprocess.run([sys.executable,'-m',mod,'--help'], env={**os.environ,'PYTHONPATH':'src'}, capture_output=True)
        assert cp.returncode==0
