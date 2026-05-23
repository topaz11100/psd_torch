from __future__ import annotations

def btf_to_bft(x):
    b = len(x); t = len(x[0]); f = len(x[0][0])
    return [[[x[bi][ti][fi] for ti in range(t)] for fi in range(f)] for bi in range(b)]


def btchw_to_bflat_t(x):
    b = len(x); t = len(x[0]); c = len(x[0][0]); h = len(x[0][0][0]); w = len(x[0][0][0][0])
    out = []
    for bi in range(b):
        rows = []
        for ci in range(c):
            for hi in range(h):
                for wi in range(w):
                    rows.append([x[bi][ti][ci][hi][wi] for ti in range(t)])
        out.append(rows)
    return out


def bt_to_srt(x):
    return btf_to_bft(x)
