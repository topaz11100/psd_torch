import types

import torch

import src.model_training as mt
from src.neurons import _common as neuron_common

class M:
    def state_dict(self):
        return {'w':1}


def test_maybe_compile_disabled():
    m,applied,policy,_kwargs=mt._maybe_compile_model(M(),requested=False)
    assert not applied and policy=='disabled'


def test_maybe_compile_calls_torch_compile(monkeypatch):
    called={'v':False}
    def fake_compile(model):
        called['v']=True
        return model
    monkeypatch.setattr(mt,'torch',types.SimpleNamespace(compile=fake_compile), raising=False)
    _,applied,_policy,_kwargs=mt._maybe_compile_model(M(),requested=True)
    assert applied and called['v']


def test_checkpoint_payload_state_dict():
    p=mt._checkpoint_payload(model=M(),epoch=1)
    assert p['state_dict']['w']==1



def test_train_one_epoch_uses_bf16_safe_amp_flag_without_outer_step_compile():
    source = __import__('pathlib').Path('src/model/training.py').read_text(encoding='utf-8')
    assert 'compile_step' not in source
    assert 'compile_kwargs' not in source
    assert 'amp_bf16_safe' in source
    assert 'compile_step' not in source


def test_sequence_loop_backend_is_compiled_sequence_prealloc():
    assert neuron_common.sequence_backend_name() == 'compiled_sequence_prealloc'


def test_sequence_buffer_mode_is_fixed_prealloc():
    assert neuron_common.sequence_buffer_mode() == 'prealloc'
    assert neuron_common.use_preallocated_sequence_buffers() is True


def test_fixed_policy_uses_cpu_eager_backend_and_calls_compile(monkeypatch):
    called = {'v': False, 'kwargs': None}

    def fake_compile(model, **kwargs):
        called['v'] = True
        called['kwargs'] = dict(kwargs)
        return model

    monkeypatch.setattr(mt, 'torch', types.SimpleNamespace(nn=torch.nn, compile=fake_compile), raising=False)
    module = torch.nn.Linear(2, 2)
    compiled, applied, policy, kwargs = mt._maybe_compile_model(
        module,
        requested=True,
        device=torch.device('cpu'),
    )
    assert applied
    assert called['v'] is True
    assert called['kwargs']['backend'] == 'eager'
    assert kwargs['backend'] == 'eager'
    assert 'fixed_backend=eager' in policy
    assert getattr(compiled, '_orig_mod', None) is module



def test_compile_startup_status_includes_visible_applied_state(capsys):
    ctx = mt.DDPContext(False, 0, 0, 1, torch.device('cpu'), True)
    module = torch.nn.Linear(2, 2)
    module._compiled_sequence_no_trace = lambda x: x
    module._compiled_sequence_policy = 'torch.compile_dummy_sequence(backend=eager,dynamic=False,fullgraph=True)'
    mt._emit_compile_startup_status(
        ctx,
        module,
        compile_requested=True,
        compile_applied=True,
        compile_policy='dummy_policy',
        compile_kwargs={'backend': 'eager', 'fullgraph': True, 'dynamic': False},
        compile_stance_policy='dummy_stance',
        compile_cache_policy={'enabled': False},
        amp_mode='off',
        amp_bf16_safe_active=False,
        tf32_policy={'enabled': True},
        drop_last_train=True,
    )
    out = capsys.readouterr().out
    assert 'compile_startup_status' in out
    assert '"compile": true' in out
    assert '"compile_applied": true' in out
    assert 'compiled_region_count' in out
    assert '"kind": "sequence"' in out


def test_timestamped_output_path_inserts_run_directory_for_result_roots():
    from src.util.paths import timestamped_output_path, timestamped_output_root

    assert str(timestamped_output_path('/tmp/case/checkpoints', timestamp='T', enabled=True)).endswith('/tmp/case/run_T/checkpoints')
    assert str(timestamped_output_path('/tmp/case/metrics', timestamp='T', enabled=True)).endswith('/tmp/case/run_T/metrics')
    assert str(timestamped_output_path('/tmp/out', timestamp='T', enabled=True)).endswith('/tmp/out/run_T')
    assert str(timestamped_output_root('/tmp/out', run_timestamp='T', prefix='stage', enabled=True)).endswith('/tmp/out/stage_T')
    assert str(timestamped_output_path('/tmp/out', timestamp='T', enabled=False)).endswith('/tmp/out')
