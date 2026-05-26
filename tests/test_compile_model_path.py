import types
import src.model_training as mt

class M:
    def state_dict(self):
        return {'w':1}


def test_maybe_compile_disabled():
    m,applied,policy=mt._maybe_compile_model(M(),requested=False)
    assert not applied and policy=='disabled'


def test_maybe_compile_calls_torch_compile(monkeypatch):
    called={'v':False}
    def fake_compile(model):
        called['v']=True
        return model
    monkeypatch.setattr(mt,'torch',types.SimpleNamespace(compile=fake_compile), raising=False)
    _,applied,_=mt._maybe_compile_model(M(),requested=True)
    assert applied and called['v']


def test_checkpoint_payload_state_dict():
    p=mt._checkpoint_payload(model=M(),epoch=1)
    assert p['state_dict']['w']==1
