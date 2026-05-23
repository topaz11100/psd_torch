from __future__ import annotations
import json, hashlib
from dataclasses import asdict
from psd_snn.config.specs import ExperimentConfig

def canonical_config_dict(config: ExperimentConfig):
    d=asdict(config)
    if d['model']['constraints']['scenario']['scenario']=='structclip':
        d['model']['constraints']['scenario']['scenario']='clipstructure'
    return d

def canonical_json_dumps(d: dict)->str:
    return json.dumps(d, sort_keys=True, separators=(',', ':'))

def stable_hash(d: dict)->str:
    return hashlib.sha256(canonical_json_dumps(d).encode()).hexdigest()[:16]

def model_metadata_from_config(config: ExperimentConfig)->dict:
    d=canonical_config_dict(config)
    m=d['model']
    return {'topology':{'kind':m['topology']['kind'],'hidden_widths':m['topology']['hidden_widths']},'cell':{'kind':m['cell']['kind'],'reset_mode':m['cell']['reset_mode'],'threshold_mode':m['cell']['threshold_mode']},'readout':{'mode':m['readout']['kind']},'config_hash':stable_hash(d)}

def constraint_metadata_from_plan(plan)->dict:
    return {'scenario':plan.scenario,'constraint_hash':plan.constraint_hash}

def reconstruct_config_from_metadata(metadata: dict)->dict:
    return json.loads(json.dumps(metadata))

def validate_metadata(metadata: dict):
    s = metadata.get('constraint',{}).get('scenario') if 'constraint' in metadata else None
    if s=='structclip': raise ValueError('structclip must not appear in output metadata')
