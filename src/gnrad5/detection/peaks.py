from __future__ import annotations

import numpy as np

from gnrad5.detection.regional_max import imregionalmax_2d


def pick_peaks_nms(rdm_pow: np.ndarray, noise_est: np.ndarray, alpha: float, sup_rad: tuple[int, int]):
    b = rdm_pow > (alpha * noise_est)
    if not np.any(b):
        return np.zeros((0, 3))

    local_max = imregionalmax_2d(rdm_pow)
    candidates = local_max & b

    rows, cols = np.nonzero(candidates)
    vals = rdm_pow[rows, cols]
    order = np.argsort(vals)[::-1]
    rows = rows[order]
    cols = cols[order]
    vals = vals[order]

    suppressed = np.zeros_like(rdm_pow, dtype=bool)
    peaks = []
    for r, c, v in zip(rows, cols, vals):
        if suppressed[r, c]:
            continue
        peaks.append((r + 1, c + 1, v))
        r0 = max(0, r - sup_rad[0])
        r1 = min(rdm_pow.shape[0], r + sup_rad[0] + 1)
        c0 = max(0, c - sup_rad[1])
        c1 = min(rdm_pow.shape[1], c + sup_rad[1] + 1)
        suppressed[r0:r1, c0:c1] = True
    return np.asarray(peaks, dtype=float)
