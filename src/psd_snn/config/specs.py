from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Literal, Optional
import json
try:
    import yaml  # type: ignore
except Exception:
    yaml = None

ScenarioName = Literal['none','clip','structure','clipstructure']

@dataclass
class ScenarioSpec:
    scenario: ScenarioName = 'none'
    apply_to_output: bool = False

@dataclass
class StructureSpec:
    enabled: bool = False
    group_count: int = 1
    band_neuron_ends: Optional[list[list[int]]] = None
    input_group_ids: Optional[list[int]] = None
    apply_first_layer: bool = False
    apply_recurrent: bool = True

@dataclass
class ClipSpec:
    enabled: bool = False
    lif_alpha_bounds: Optional[list[tuple[float,float]]] = None
    rf_frequency_bounds: Optional[list[tuple[float,float]]] = None
    rf_damping_bounds: Optional[list[tuple[float,float]]] = None
    threshold_bounds: Optional[list[tuple[float,float]]] = None

@dataclass
class ConstraintSpec:
    scenario: ScenarioSpec = field(default_factory=ScenarioSpec)
    structure: StructureSpec = field(default_factory=StructureSpec)
    clip: ClipSpec = field(default_factory=ClipSpec)

ProbeFamily = Literal['balanced_global', 'distributed_set', 'label_set', 'label_single']
SpectralAxisPolicy = Literal['exact', 'userbin']
DistanceMetric = Literal['centered_l2', 'diff_l2']
CellKind = Literal['if', 'lif', 'rf']
ThresholdMode = Literal['fixed', 'trainable']
TopologyKind = Literal['mlp_stack', 'vgg', 'resnet', 's4', 'gru', 'spike_transformer']
ReadoutKind = Literal['final_if', 'final_mem']

@dataclass
class PCARepresentativeSpec:
    n_components: int = 1
    basis_mode: Literal['fixed_reference', 'fit_per_checkpoint'] = 'fit_per_checkpoint'
    reference_checkpoint: Optional[int] = None
    reference_split: Optional[str] = None
    reference_probe_family: Optional[str] = None
    reference_scope: Optional[str] = None
    center: bool = True
    sign_convention: Literal['largest_abs_loading_positive'] = 'largest_abs_loading_positive'

@dataclass
class RepresentativeSpec:
    method: Literal['mean','median','element_psd','pca'] = 'mean'
    pca: PCARepresentativeSpec = field(default_factory=PCARepresentativeSpec)

@dataclass
class PSDAnalysisSpec:
    enabled: bool = True
    spectral_axis: SpectralAxisPolicy = 'exact'
    userbin_edges: Optional[list[float]] = None
    userbin_reducer: Literal['mean','median'] = 'mean'
    centering: Literal['raw','centered'] = 'raw'
    window: Literal['none','hann'] = 'none'
    representative: RepresentativeSpec = field(default_factory=RepresentativeSpec)
    distance_metrics: list[DistanceMetric] = field(default_factory=lambda:['centered_l2'])
    scale_outputs: Literal['raw','db','both'] = 'raw'
    db_eps: float = 1e-12
    allow_empty_bins: bool = False
    empty_bin_policy: Literal['error','nan','zero'] = 'error'

@dataclass
class LayerSelectionSpec:
    include: Optional[list[str]] = None
    exclude: Optional[list[str]] = None

@dataclass
class SeriesSelectionSpec:
    include: list[str] = field(default_factory=lambda:['spike'])

@dataclass
class TraceSaveSpec:
    enabled: bool = False
    series: list[str] = field(default_factory=lambda:['spike'])
    layers: LayerSelectionSpec = field(default_factory=LayerSelectionSpec)
    dtype: Literal['uint8','float16','float32'] = 'uint8'
    chunk_size: int = 32
    artifact_dir: str = 'artifacts'
    save_manifest: bool = True

@dataclass
class ArtifactSpec:
    output_dir: str = 'artifacts'


@dataclass
class FFT2DUserbinSpec:
    userbin_axes: Literal['time_frequency','row_frequency','both_frequency_axes'] = 'time_frequency'
    time_bin_edges: Optional[list[float]] = None
    row_bin_edges: Optional[list[float]] = None

@dataclass
class RowAxisSpec:
    row_axis_semantics: Literal['unordered','group_ordered','feature_ordered','spatial_flattened','channel_ordered','pca_component'] = 'unordered'

@dataclass
class FFT2DAnalysisSpec:
    enabled: bool = False
    spectral_axis: SpectralAxisPolicy = 'exact'
    userbin: FFT2DUserbinSpec = field(default_factory=FFT2DUserbinSpec)
    userbin_reducer: Literal['mean','median'] = 'mean'
    centering: Literal['none','time_mean','global_mean'] = 'none'
    window_time: Literal['none','hann'] = 'none'
    window_row: Literal['none','hann'] = 'none'
    row_axis: RowAxisSpec = field(default_factory=RowAxisSpec)
    scale_outputs: Literal['raw','db','both'] = 'raw'
    allow_empty_bins: bool = False
    empty_bin_policy: Literal['error','nan','zero'] = 'error'
    distance_metrics: list[DistanceMetric] = field(default_factory=lambda:['centered_l2'])
    diff_axis: Literal['time_frequency','row_frequency','both_frequency_axes'] = 'time_frequency'

@dataclass
class SignalAnalysisSpec:
    psd: PSDAnalysisSpec = field(default_factory=PSDAnalysisSpec)
    fft2d: FFT2DAnalysisSpec = field(default_factory=FFT2DAnalysisSpec)
    trace_save: TraceSaveSpec = field(default_factory=TraceSaveSpec)
    artifact: ArtifactSpec = field(default_factory=ArtifactSpec)

@dataclass
class TopologySpec:
    kind: TopologyKind = 'mlp_stack'; input_dim: int = 8; hidden_widths: list[int] = field(default_factory=lambda:[16]); output_dim: int = 2
@dataclass
class CellSpec:
    kind: CellKind = 'lif'; reset_mode: str = 'soft'; threshold_mode: ThresholdMode = 'fixed'; threshold_init: float = 1.0; recurrent: bool = False; alpha: float = 0.9; alpha_trainable: bool = False; rf_omega: float = 1.0; rf_damping: float = 0.1; dt: float = 1.0
@dataclass
class ReadoutSpec: kind: ReadoutKind = 'final_mem'
@dataclass
class ProbeSpec: family: ProbeFamily = 'balanced_global'; sample_count: int = 8; labels: Optional[list[int]] = None; seed: int = 0; exclusion_family: Optional[ProbeFamily] = None
@dataclass
class SpectralSpec: axis_policy: SpectralAxisPolicy = 'exact'; userbin_edges: Optional[list[float]] = None; userbin_reducer: Literal['mean','median'] = 'mean'; allow_empty_bins: bool = False; empty_bin_fill: Literal['nan','zero']='nan'; distance_metric: DistanceMetric='centered_l2'
@dataclass
class DynamicsSpec: include_parameter_stats: bool = True; include_internal_state_stats: bool = True
@dataclass
class ModelSpec: topology: TopologySpec = field(default_factory=TopologySpec); cell: CellSpec = field(default_factory=CellSpec); readout: ReadoutSpec = field(default_factory=ReadoutSpec); constraints: ConstraintSpec = field(default_factory=ConstraintSpec)
@dataclass
class ExperimentConfig:
    mode: Literal['train','analyze_psd','analyze_element_psd','analyze_fft2d','analyze_pca_psd','analyze_dynamics','analyze_signal']='analyze_psd'
    model: ModelSpec = field(default_factory=ModelSpec); probe: ProbeSpec = field(default_factory=ProbeSpec); spectral: SpectralSpec = field(default_factory=SpectralSpec); dynamics: DynamicsSpec = field(default_factory=DynamicsSpec); signal_analysis: SignalAnalysisSpec = field(default_factory=SignalAnalysisSpec)

def _norm_scenario(s:str)->str: return 'clipstructure' if s=='structclip' else s

def validate_signal_analysis(sa: SignalAnalysisSpec):
    psd = sa.psd
    if psd.spectral_axis not in {'exact','userbin'}: raise ValueError('spectral_axis must be exact|userbin')
    if psd.userbin_reducer not in {'mean','median'}: raise ValueError('userbin_reducer must be mean|median')
    if psd.representative.method not in {'mean','median','element_psd','pca'}: raise ValueError('invalid representative method')
    if any(m not in {'centered_l2','diff_l2'} for m in psd.distance_metrics): raise ValueError('invalid distance metric')
    if psd.representative.method == 'pca' and psd.representative.pca.n_components <= 0: raise ValueError('pca n_components must be >0')
    if psd.representative.pca.basis_mode == 'fixed_reference' and psd.representative.pca.reference_checkpoint is None: raise ValueError('fixed_reference requires reference_checkpoint')
    if sa.trace_save.dtype == 'uint8' and any(s != 'spike' for s in sa.trace_save.series): raise ValueError('uint8 trace dtype allowed only for spike series')
    f2 = sa.fft2d
    if f2.spectral_axis not in {'exact','userbin'}: raise ValueError('fft2d spectral_axis must be exact|userbin')
    if f2.userbin_reducer not in {'mean','median'}: raise ValueError('fft2d userbin_reducer must be mean|median')
    if any(m not in {'centered_l2','diff_l2'} for m in f2.distance_metrics): raise ValueError('invalid fft2d distance metric')
    if f2.spectral_axis=='userbin':
      ua=f2.userbin.userbin_axes
      if ua=='time_frequency' and not f2.userbin.time_bin_edges: raise ValueError('fft2d time_bin_edges required')
      if ua=='row_frequency' and not f2.userbin.row_bin_edges: raise ValueError('fft2d row_bin_edges required')
      if ua=='both_frequency_axes' and (not f2.userbin.row_bin_edges or not f2.userbin.time_bin_edges): raise ValueError('fft2d row/time edges required')
      if f2.row_axis.row_axis_semantics=='unordered' and ua in {'row_frequency','both_frequency_axes'}: raise ValueError('unordered row_axis_semantics forbids row_frequency/both_frequency_axes userbin')


def validate_config(cfg: ExperimentConfig)->None:
    t,c = cfg.model.topology,cfg.model.cell
    sc = cfg.model.constraints.scenario; st = cfg.model.constraints.structure; cl = cfg.model.constraints.clip
    sc.scenario = _norm_scenario(sc.scenario)
    if sc.apply_to_output: raise ValueError('apply_to_output=true is currently unsupported; only hidden layer constraints are supported')
    if sc.scenario not in {'none','clip','structure','clipstructure'}: raise ValueError('invalid scenario')
    if t.kind!='mlp_stack' and sc.scenario!='none': raise ValueError('scenario only for mlp_stack')
    if t.kind=='mlp_stack' and not t.hidden_widths: raise ValueError('hidden_widths empty')
    if sc.scenario=='none' and (st.enabled or cl.enabled): raise ValueError('none scenario cannot enable clip/structure')
    if sc.scenario=='clip' and not cl.enabled: raise ValueError('clip scenario requires clip.enabled')
    if sc.scenario=='structure' and not st.enabled: raise ValueError('structure scenario requires structure.enabled')
    if sc.scenario=='clipstructure' and (not st.enabled or not cl.enabled): raise ValueError('clipstructure requires both')
    if c.kind=='if' and cl.lif_alpha_bounds is not None: raise ValueError('IF cannot take lif_alpha bounds')
    if c.kind=='rf' and cl.lif_alpha_bounds is not None: raise ValueError('RF cannot take lif_alpha bounds')
    if c.kind=='lif' and (cl.rf_frequency_bounds is not None or cl.rf_damping_bounds is not None): raise ValueError('LIF cannot take rf bounds')
    validate_signal_analysis(cfg.signal_analysis)

def to_sanitized_dict(cfg: ExperimentConfig)->dict:
    d=asdict(cfg); d['model']['constraints']['scenario']['scenario']=_norm_scenario(d['model']['constraints']['scenario']['scenario']); return d

def _from_dict(d: dict[str,Any])->ExperimentConfig:
    cs = d.get('model',{}).get('constraints',{})
    sa = d.get('signal_analysis',{})
    cfg = ExperimentConfig(
      mode=d.get('mode','analyze_psd'),
      model=ModelSpec(topology=TopologySpec(**d.get('model',{}).get('topology',{})), cell=CellSpec(**d.get('model',{}).get('cell',{})), readout=ReadoutSpec(**d.get('model',{}).get('readout',{})), constraints=ConstraintSpec(scenario=ScenarioSpec(**cs.get('scenario',{})), structure=StructureSpec(**cs.get('structure',{})), clip=ClipSpec(**cs.get('clip',{})))),
      probe=ProbeSpec(**d.get('probe',{})), spectral=SpectralSpec(**d.get('spectral',{})), dynamics=DynamicsSpec(**d.get('dynamics',{})),
      signal_analysis=SignalAnalysisSpec(psd=PSDAnalysisSpec(**{k:v for k,v in sa.get('psd',{}).items() if k!='representative'}), fft2d=FFT2DAnalysisSpec(**{k:v for k,v in sa.get('fft2d',{}).items() if k not in {'userbin','row_axis'}}), trace_save=TraceSaveSpec(**sa.get('trace_save',{})), artifact=ArtifactSpec(**sa.get('artifact',{})))
    )
    rep=sa.get('psd',{}).get('representative',{})
    if rep:
      cfg.signal_analysis.psd.representative=RepresentativeSpec(method=rep.get('method','mean'), pca=PCARepresentativeSpec(**rep.get('pca',{})))
    f2=sa.get('fft2d',{})
    if f2:
      cfg.signal_analysis.fft2d.userbin=FFT2DUserbinSpec(**f2.get('userbin',{}))
      cfg.signal_analysis.fft2d.row_axis=RowAxisSpec(**f2.get('row_axis',{}))
    validate_config(cfg); return cfg

def load_experiment_config(path:str)->ExperimentConfig:
    text=open(path).read();
    if path.endswith('.json'): return _from_dict(json.loads(text))
    if path.endswith(('.yaml','.yml')):
      if yaml is None: raise RuntimeError('PyYAML is not installed; use JSON config in this environment')
      return _from_dict(yaml.safe_load(text))
    raise ValueError('config path must end with .json/.yaml/.yml')
