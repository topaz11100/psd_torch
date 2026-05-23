from __future__ import annotations
import math

def _mean(v):
    return sum(v) / len(v)

def centered_l2(a, b):
    if len(a) != len(b):
        raise ValueError('length mismatch')
    ma, mb = _mean(a), _mean(b)
    return math.sqrt(sum(((x - ma) - (y - mb)) ** 2 for x, y in zip(a, b)))

def diff_l2(a, b):
    if len(a) != len(b):
        raise ValueError('length mismatch')
    if len(a) < 2:
        return 0.0
    da = [a[i + 1] - a[i] for i in range(len(a) - 1)]
    db = [b[i + 1] - b[i] for i in range(len(b) - 1)]
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(da, db)))

def distance_between(curve_a, curve_b, metric: str):
    if curve_a['axis_policy'] != curve_b['axis_policy']:
        raise ValueError('cannot compare exact and userbin spaces')
    if metric == 'centered_l2':
        return centered_l2(curve_a['power'], curve_b['power'])
    if metric == 'diff_l2':
        return diff_l2(curve_a['power'], curve_b['power'])
    raise ValueError('unsupported metric')
