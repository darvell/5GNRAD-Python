from __future__ import annotations

import numpy as np


def suppress_sidelobes(pk: np.ndarray, tol_near: float = 4, sizes=None, wrap_d=True, wrap_az=True, wrap_el=True, tol_same: float = 2):
    if pk is None or len(pk) == 0:
        return pk
    if sizes is None:
        sizes = [np.inf, np.inf, np.inf, np.inf]

    order = np.argsort(pk[:, 4])[::-1]
    pk = pk[order]
    k = pk.shape[0]
    keep = np.ones(k, dtype=bool)

    rsz, dsz, elsz, azsz = sizes

    for i in range(k):
        if not keep[i]:
            continue
        ri, di, eli, azi = pk[i, 0], pk[i, 1], pk[i, 2], pk[i, 3]
        for j in range(i + 1, k):
            if not keep[j]:
                continue
            dr = abs(pk[j, 0] - ri)
            dd = abs(pk[j, 1] - di)
            de = abs(pk[j, 2] - eli)
            da = abs(pk[j, 3] - azi)
            if wrap_d and np.isfinite(dsz):
                dd = min(dd, dsz - dd)
            if wrap_az and np.isfinite(azsz):
                da = min(da, azsz - da)
            if wrap_el and np.isfinite(elsz):
                de = min(de, elsz - de)

            diffs = np.array([dr, dd, de, da])
            near_same = diffs <= tol_same
            if np.sum(near_same) >= 3:
                keep[j] = False
    return pk[keep]
