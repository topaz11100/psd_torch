from pathlib import Path
import pytest
import src.model_training as mt
from src.util.config import load_structured

def test_ddp_config_keys_present():
    base=load_structured(Path('config/model_training.yaml'))['model_training']
    ddp=load_structured(Path('config/model_training_ddp.yaml'))['model_training']
    assert base['ddp'] is False and base['ddp_world_size']==2 and base['batch_size_is_global'] is True
    assert ddp['ddp'] is True and ddp['ddp_world_size']==2 and ddp['batch_size_is_global'] is True
    assert base.get('compile_cpu_threads') is None
    assert ddp.get('compile_cpu_threads') == 2

def test_ddp_wrapper_script_static():
    text=Path('bash/model_training_ddp.sh').read_text(encoding='utf-8')
    assert 'torchrun' in text and '--standalone' in text and '--nproc_per_node="${NPROC_PER_NODE:-2}"' in text and '--ddp true' in text
    assert 'nohup' in text and 'LOG_PATH' in text and 'PID=' in text
    assert 'CONFIG_GROUP_0=(' in text and 'CONFIG_GROUPS=(CONFIG_GROUP_0 CONFIG_GROUP_1)' in text and '.yaml' in text
    assert 'for CONFIG_PATH in "${GROUP[@]}"' in text

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


def test_ddp_eval_subset_is_strided_without_padding():
    ctx0 = mt.DDPContext(True, 0, 0, 2, None, True)
    ctx1 = mt.DDPContext(True, 1, 1, 2, None, False)
    ds = list(range(5))
    sub0, policy0 = mt._evaluation_dataset_for_rank(ds, ctx0)
    sub1, policy1 = mt._evaluation_dataset_for_rank(ds, ctx1)
    assert list(sub0.indices) == [0, 2, 4]
    assert list(sub1.indices) == [1, 3]
    assert policy0['policy'] == 'ddp_rank_strided_subset_no_padding'
    assert policy0['global_samples'] == 5 and policy1['global_samples'] == 5
    assert policy0['local_samples'] + policy1['local_samples'] == 5


def test_ddp_compile_cpu_threads_default_and_override(monkeypatch):
    ctx = mt.DDPContext(True, 0, 0, 2, None, True)
    class A: pass
    args = A(); args.compile_cpu_threads = None
    monkeypatch.delenv('PSD_TORCH_COMPILE_CPU_THREADS', raising=False)
    assert mt._compile_cpu_thread_count_from_args(args, ctx) == (2, 'ddp_default_2')
    args.compile_cpu_threads = 4
    assert mt._compile_cpu_thread_count_from_args(args, ctx) == (4, 'argument')
    args.compile_cpu_threads = None
    monkeypatch.setenv('PSD_TORCH_COMPILE_CPU_THREADS', '3')
    assert mt._compile_cpu_thread_count_from_args(args, ctx) == (3, 'env:PSD_TORCH_COMPILE_CPU_THREADS')


def test_ddp_wrapper_exposes_compile_cache_namespace_args():
    text = Path('bash/model_training_ddp.sh').read_text(encoding='utf-8')
    assert '--compile-cache-root' in text
    assert '--experiment-name' in text
    assert 'PSD_TORCH_COMPILE_CACHE_DIR="$CONFIG_CACHE_DIR_RESOLVED"' in text
    assert 'COMPILE_CACHE=${CONFIG_CACHE_DIR_RESOLVED}' in text
