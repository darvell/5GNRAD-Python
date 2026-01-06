from __future__ import annotations

import numpy as np


def sinc_interp(x, v, xq, w):
    x = np.asarray(x, dtype=float).reshape(-1)
    v = np.asarray(v)
    if v.shape[0] != x.shape[0]:
        raise ValueError("The number of rows in v must match the length of x")

    order = np.argsort(x)
    x = x[order]
    v = v[order]

    ts, t = np.meshgrid(xq, x, indexing="ij")
    kernel = np.sinc((ts - t) * w)
    return kernel @ v
