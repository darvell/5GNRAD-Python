from __future__ import annotations

import numpy as np
from scipy.signal import convolve2d


def cfar_2d_fast(rdm: np.ndarray, guard_radius: tuple[int, int], training_radius: tuple[int, int], threshold: float):
    g_r, g_c = guard_radius
    t_r, t_c = training_radius
    ker_t = np.ones((2 * t_r + 1, 2 * t_c + 1), dtype=rdm.dtype)
    ker_g = np.ones((2 * g_r + 1, 2 * g_c + 1), dtype=rdm.dtype)

    sum_t = convolve2d(rdm, ker_t, mode="same")
    sum_g = convolve2d(rdm, ker_g, mode="same")
    k_cells = ker_t.size - ker_g.size
    if k_cells <= 0:
        noise_est = np.full_like(rdm, np.inf)
    else:
        noise_est = (sum_t - sum_g) / k_cells

    mask = np.ones_like(rdm, dtype=bool)
    pad_r = t_r + g_r
    pad_c = t_c + g_c
    if pad_r > 0:
        mask[:pad_r, :] = False
        mask[-pad_r:, :] = False
    if pad_c > 0:
        mask[:, :pad_c] = False
        mask[:, -pad_c:] = False

    detect = np.zeros_like(rdm, dtype=bool)
    detect[mask] = rdm[mask] > noise_est[mask] * threshold
    out = np.zeros_like(rdm)
    out[detect] = rdm[detect]
    return out, noise_est
