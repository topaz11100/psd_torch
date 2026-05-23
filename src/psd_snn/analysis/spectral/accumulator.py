from __future__ import annotations
import math

class PSDAccumulator:
    def __init__(self, axis_policy='exact', userbin_edges=None, userbin_reducer='mean', allow_empty_bins=False, empty_bin_fill='nan'):
        self.axis_policy = axis_policy
        self.userbin_edges = userbin_edges
        self.userbin_reducer = userbin_reducer
        self.allow_empty_bins = allow_empty_bins
        self.empty_bin_fill = empty_bin_fill
        self._sum = None
        self._count = 0
        self._freq = None

    def update(self, freq, power_batch):
        # power_batch shape: S,F ; accumulate over S only
        if self._freq is None:
            self._freq = list(freq)
        if self._sum is None:
            self._sum = [0.0] * len(power_batch[0])
        for row in power_batch:
            for i, v in enumerate(row):
                self._sum[i] += float(v)
            self._count += 1

    def _reduce_userbin(self, freq, power):
        edges = self.userbin_edges
        if not edges or len(edges) < 2:
            raise ValueError('userbin requires at least 2 edges')
        out_f, out_p = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            vals = [p for f, p in zip(freq, power) if lo <= f < hi]
            out_f.append((lo + hi) * 0.5)
            if not vals:
                if not self.allow_empty_bins:
                    raise ValueError('empty userbin encountered')
                out_p.append(float('nan') if self.empty_bin_fill == 'nan' else 0.0)
            elif self.userbin_reducer == 'mean':
                out_p.append(sum(vals) / len(vals))
            elif self.userbin_reducer == 'median':
                sv = sorted(vals); n = len(sv)
                out_p.append(sv[n // 2] if n % 2 else (sv[n // 2 - 1] + sv[n // 2]) / 2.0)
            else:
                raise ValueError('invalid reducer')
        return out_f, out_p

    def finalize(self, to_db=False):
        if self._sum is None or self._count == 0:
            raise ValueError('empty accumulator')
        power = [v / self._count for v in self._sum]
        freq = list(self._freq)
        if self.axis_policy == 'userbin':
            freq, power = self._reduce_userbin(freq, power)
        elif self.axis_policy != 'exact':
            raise ValueError('axis_policy must be exact|userbin')
        if to_db:
            power = [10.0 * math.log10(max(v, 1e-12)) for v in power]
        return {'axis_policy': self.axis_policy, 'freq': freq, 'power': power, 'count': self._count}
