from __future__ import annotations

import numpy as np

from gnrad5.detection.cfar_fast import cfar_2d_fast


def cfar_2d(power: np.ndarray, guard: tuple[int, int], train: tuple[int, int], threshold: float):
    g_r, g_d = guard
    t_r, t_d = train
    rows, cols = power.shape
    mask = np.zeros_like(power, dtype=bool)
    noise = np.zeros_like(power, dtype=float)

    r_start = t_r + g_r
    d_start = t_d + g_d
    r_end = rows - (t_r + g_r)
    d_end = cols - (t_d + g_d)

    for r in range(r_start, r_end):
        r0 = r - (t_r + g_r)
        r1 = r + (t_r + g_r) + 1
        for d in range(d_start, d_end):
            d0 = d - (t_d + g_d)
            d1 = d + (t_d + g_d) + 1

            window = power[r0:r1, d0:d1]
            guard_window = power[r - g_r : r + g_r + 1, d - g_d : d + g_d + 1]
            training_sum = window.sum() - guard_window.sum()
            training_cells = window.size - guard_window.size
            if training_cells <= 0:
                continue
            noise_level = training_sum / training_cells
            noise[r, d] = noise_level
            if power[r, d] > noise_level * threshold:
                mask[r, d] = True
    return mask, noise


def detect_cfar_rd(rd_map: np.ndarray, sens_config):
    power = np.abs(rd_map) ** 2
    peak = power.max() if power.size else 0.0
    thr_lin = peak / (10 ** (sens_config.rda_threshold / 10.0))
    power = np.where(power >= thr_lin, power, 0.0)

    guard = (sens_config.cfar_grd_cell_range, sens_config.cfar_grd_cell_velocity)
    train = (sens_config.cfar_trn_cell_range, sens_config.cfar_trn_cell_velocity)
    det_power, noise = cfar_2d_fast(power, guard, train, sens_config.cfar_threshold)
    mask = det_power > 0
    detections = np.column_stack(np.nonzero(mask))
    return detections, noise, mask
