from __future__ import annotations

import numpy as np


def cfar_2d_fast_mod(rdm: np.ndarray, guard_radius: tuple[int, int], training_radius: tuple[int, int], threshold: float):
    g_r, g_c = guard_radius
    t_r, t_c = training_radius
    if t_r < g_r or t_c < g_c:
        raise ValueError("Training radii must be >= guard radii")

    m, n = rdm.shape
    ker_t = np.ones((2 * t_r + 1, 2 * t_c + 1), dtype=rdm.dtype)
    ker_g = np.ones((2 * g_r + 1, 2 * g_c + 1), dtype=rdm.dtype)

    pad = ((t_r, t_r), (t_c, t_c))
    rdm_pad = np.pad(rdm, pad, mode="edge")
    one_pad = np.pad(np.ones_like(rdm), pad, mode="edge")

    sum_t = _conv2_valid(rdm_pad, ker_t)
    cnt_t = _conv2_valid(one_pad, ker_t)

    sum_g_full = _conv2_valid(rdm_pad, ker_g)
    cnt_g_full = _conv2_valid(one_pad, ker_g)

    dr = t_r - g_r
    dc = t_c - g_c
    rows = slice(dr, dr + m)
    cols = slice(dc, dc + n)
    sum_g = sum_g_full[rows, cols]
    cnt_g = cnt_g_full[rows, cols]

    ring_sum = sum_t - sum_g
    k_map = np.maximum(cnt_t - cnt_g, 1)
    noise_est = ring_sum / k_map

    detect = rdm > noise_est * threshold
    y = np.zeros_like(rdm)
    y[detect] = rdm[detect]
    return y, noise_est


def _conv2_valid(a: np.ndarray, k: np.ndarray) -> np.ndarray:
    from scipy.signal import convolve2d

    return convolve2d(a, k, mode="valid")
