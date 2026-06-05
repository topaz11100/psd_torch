import json
import subprocess
import sys
from pathlib import Path
from src.util.config import load_structured


def test_ddp_compile_smoke_matrix_dry_run_generates_case_config(tmp_path: Path):
    output_root = tmp_path / 'smoke_out'
    cache_root = tmp_path / 'cache'
    prep_root = tmp_path / 'prepared'
    proc = subprocess.run(
        [
            sys.executable,
            'tools/smoke_ddp_compile_matrix.py',
            '--prep-root', str(prep_root),
            '--output-root', str(output_root),
            '--compile-cache-root', str(cache_root),
            '--experiment-name', 'expA',
            '--case', 'ssc:spikegru:spikegru_max_over_time',
            '--dry-run',
        ],
        cwd=Path.cwd(),
        check=True,
        text=True,
        capture_output=True,
    )
    lines = [json.loads(line) for line in proc.stdout.splitlines() if line.strip().startswith('{')]
    assert len(lines) == 1
    item = lines[0]
    assert item['env']['PSD_TORCH_COMPILE_CACHE_DIR'] == str((cache_root / 'expA' / 'ssc__spikegru__spikegru_max_over_time').resolve())
    assert 'torchrun' in item['cmd'][0]
    assert '--standalone' in item['cmd']
    assert '--nproc_per_node=2' in item['cmd']
    cfg = load_structured(Path(item['config']))['model_training']
    assert cfg['ddp'] is True
    assert cfg['compile'] is True
    assert cfg['signal_window'] == 'hann'
    assert 'output_root' not in cfg
    assert 'model' not in cfg
    assert cfg['neuron_type'] == 'spikegru'
    assert cfg['v_th'] == ['fixed', 1.0]
    assert cfg['filter'] == 'train'
    assert cfg['dataset'] == 'ssc'
