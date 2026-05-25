import json
from pathlib import Path
import pytest
import src.model_training as mt

def test_ddp_config_keys_present():
    base=json.loads(Path('config/model_training.json').read_text())['model_training']
    ddp=json.loads(Path('config/model_training_ddp.json').read_text())['model_training']
    assert base['ddp'] is False and base['ddp_world_size']==2 and base['batch_size_is_global'] is True
    assert ddp['ddp'] is True and ddp['ddp_world_size']==2 and ddp['batch_size_is_global'] is True

def test_ddp_wrapper_script_static():
    text=Path('bash/model_training_ddp.sh').read_text(encoding='utf-8')
    assert 'torchrun' in text and '--standalone' in text and '--nproc_per_node=2' in text and '--ddp true' in text
    assert 'nohup' in text and 'LOG_PATH' in text and 'PID=' in text

def test_batch_split_helper():
    off=mt.DDPContext(False,0,0,1,None,True)
    on=mt.DDPContext(True,0,0,2,None,True)
    class A: pass
    a=A(); a.batch_size=256
    assert mt._resolve_effective_batch_size(a, off)==256
    assert mt._resolve_effective_batch_size(a, on)==128
    a.batch_size=255
    with pytest.raises(ValueError): mt._resolve_effective_batch_size(a, on)

def test_ddp_env_validation_without_torchrun(monkeypatch):
    class A: pass
    a=A(); a.ddp=True; a.ddp_world_size=2; a.batch_size_is_global=True; a.gpu_index=0
    monkeypatch.delenv('LOCAL_RANK', raising=False)
    monkeypatch.delenv('RANK', raising=False)
    monkeypatch.delenv('WORLD_SIZE', raising=False)
    with pytest.raises(ValueError): mt._build_ddp_context(a)

def test_static_keywords_present():
    text=Path('src/model_training.py').read_text(encoding='utf-8')
    for token in ['DistributedDataParallel','DistributedSampler','all_reduce','LOCAL_RANK','WORLD_SIZE','rank==0','destroy_process_group']:
        assert token in text

def test_no_ddp_keywords_in_other_stages():
    for path in ['src/dataset_psd.py','src/dataset_fft.py','src/psd_analysis.py','src/element_psd.py','src/2d_fft_analysis.py','src/plotting.py']:
        text = Path(path).read_text(encoding='utf-8')
        for token in ['DistributedDataParallel', 'DistributedSampler', 'LOCAL_RANK', 'WORLD_SIZE', 'RANK']:
            assert token not in text
