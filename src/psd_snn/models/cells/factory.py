from __future__ import annotations
from psd_snn.config.specs import CellSpec

def build_cell(cell: CellSpec, features: int, cell_bounds=None, group_ids=None, layer_index=None, layer_name=None):
    try:
        if cell.kind == 'if':
            from .if_cell import IFCell
            return IFCell(features=features, threshold=cell.threshold_init, threshold_trainable=(cell.threshold_mode == 'trainable'), reset_mode=cell.reset_mode, threshold_bounds=(cell_bounds.threshold_bounds if cell_bounds else None), group_ids=group_ids)
        if cell.kind == 'lif':
            from .lif_cell import LIFCell
            return LIFCell(features=features, alpha=cell.alpha, alpha_trainable=cell.alpha_trainable, threshold=cell.threshold_init, threshold_trainable=(cell.threshold_mode == 'trainable'), reset_mode=cell.reset_mode, alpha_bounds=(cell_bounds.lif_alpha_bounds if cell_bounds else None), threshold_bounds=(cell_bounds.threshold_bounds if cell_bounds else None), group_ids=group_ids)
        if cell.kind == 'rf':
            from .rf_cell import RFCell
            return RFCell(features=features, omega=cell.rf_omega, damping=cell.rf_damping, threshold=cell.threshold_init, threshold_trainable=(cell.threshold_mode == 'trainable'), reset_mode=cell.reset_mode, frequency_bounds=(cell_bounds.rf_frequency_bounds if cell_bounds else None), damping_bounds=(cell_bounds.rf_damping_bounds if cell_bounds else None), threshold_bounds=(cell_bounds.threshold_bounds if cell_bounds else None), group_ids=group_ids, dt=cell.dt)
        raise ValueError('unsupported cell kind')
    except ImportError as e:
        raise ImportError(f'Failed to build official `{cell.kind}` cell: {e}') from e
