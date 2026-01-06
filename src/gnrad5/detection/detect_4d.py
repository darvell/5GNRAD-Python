from __future__ import annotations

import numpy as np

from gnrad5.detection.cfar_fast_mod import cfar_2d_fast_mod
from gnrad5.detection.cluster import cluster_peaks_4d
from gnrad5.detection.peaks import pick_peaks_nms
from gnrad5.detection.sidelobes import suppress_sidelobes


def detect_cfar_4d(rd_cube: np.ndarray, sens_config):
    if np.iscomplexobj(rd_cube):
        p = np.abs(rd_cube) ** 2
    else:
        p = rd_cube.copy()
    p = np.maximum(p, 0)
    peak_val = p.max() if p.size else 0.0
    thr_lin = peak_val / (10 ** (sens_config.rda_threshold / 10.0))
    p[p < thr_lin] = 0

    r, d, el, az = p.shape

    maps = {
        "RD": p.sum(axis=(2, 3)),
        "RAz": p.sum(axis=(1, 2)),
        "REl": p.sum(axis=(1, 3)),
        "DAz": p.sum(axis=(0, 2)),
        "DEl": p.sum(axis=(0, 3)),
        "AzEl": p.sum(axis=(0, 1)),
    }

    sens_config.cfar_trn_cell_azimuth = 8
    sens_config.cfar_trn_cell_elevation = 6
    sens_config.cfar_grd_cell_azimuth = 4
    sens_config.cfar_grd_cell_elevation = 3

    grd_rd = (sens_config.cfar_grd_cell_range, sens_config.cfar_grd_cell_velocity)
    trn_rd = (sens_config.cfar_trn_cell_range, sens_config.cfar_trn_cell_velocity)
    grd_raz = (sens_config.cfar_grd_cell_range, sens_config.cfar_grd_cell_azimuth)
    trn_raz = (sens_config.cfar_trn_cell_range, sens_config.cfar_trn_cell_azimuth)
    grd_rel = (sens_config.cfar_grd_cell_range, sens_config.cfar_grd_cell_elevation)
    trn_rel = (sens_config.cfar_trn_cell_range, sens_config.cfar_trn_cell_elevation)
    grd_daz = (sens_config.cfar_grd_cell_velocity, sens_config.cfar_grd_cell_azimuth)
    trn_daz = (sens_config.cfar_trn_cell_velocity, sens_config.cfar_trn_cell_azimuth)
    grd_del = (sens_config.cfar_grd_cell_velocity, sens_config.cfar_grd_cell_elevation)
    trn_del = (sens_config.cfar_trn_cell_velocity, sens_config.cfar_trn_cell_elevation)
    grd_azel = (sens_config.cfar_grd_cell_azimuth, sens_config.cfar_grd_cell_elevation)
    trn_azel = (sens_config.cfar_trn_cell_azimuth, sens_config.cfar_trn_cell_elevation)

    thr_cfar = sens_config.cfar_threshold

    det = {}
    det["RD"] = cfar_2d_fast_mod(maps["RD"], grd_rd, trn_rd, thr_cfar)
    det["RAz"] = cfar_2d_fast_mod(maps["RAz"], grd_raz, trn_raz, thr_cfar)
    det["REl"] = cfar_2d_fast_mod(maps["REl"], grd_rel, trn_rel, thr_cfar)
    det["DAz"] = cfar_2d_fast_mod(maps["DAz"], grd_daz, trn_daz, thr_cfar)
    det["DEl"] = cfar_2d_fast_mod(maps["DEl"], grd_del, trn_del, thr_cfar)
    det["AzEl"] = cfar_2d_fast_mod(maps["AzEl"], grd_azel, trn_azel, thr_cfar)

    a = np.zeros((r, d, el, az), dtype=np.uint8)

    peaks = pick_peaks_nms(det["RD"][0], det["RD"][1], thr_cfar, grd_rd)
    for rr, dd, _ in peaks:
        a[int(rr) - 1, int(dd) - 1, :, :] += 1

    peaks = pick_peaks_nms(det["RAz"][0], det["RAz"][1], thr_cfar, grd_raz)
    for rr, aa, _ in peaks:
        a[int(rr) - 1, :, :, int(aa) - 1] += 1

    peaks = pick_peaks_nms(det["REl"][0], det["REl"][1], thr_cfar, grd_rel)
    for rr, ee, _ in peaks:
        a[int(rr) - 1, :, int(ee) - 1, :] += 1

    peaks = pick_peaks_nms(det["DAz"][0], det["DAz"][1], thr_cfar, grd_daz)
    for dd, aa, _ in peaks:
        a[:, int(dd) - 1, :, int(aa) - 1] += 1

    peaks = pick_peaks_nms(det["DEl"][0], det["DEl"][1], thr_cfar, grd_del)
    for dd, ee, _ in peaks:
        a[:, int(dd) - 1, int(ee) - 1, :] += 1

    peaks = pick_peaks_nms(det["AzEl"][0], det["AzEl"][1], thr_cfar, grd_azel)
    for aa, ee, _ in peaks:
        a[:, :, int(ee) - 1, int(aa) - 1] += 1

    v = np.zeros_like(a, dtype=float)
    to_keep = a >= 3
    v[to_keep] = p[to_keep]
    pk = _nms4d_greedy_onv(v, sens_config.nms_radius, sens_config.nms_max_peaks)

    dims = {"R": r, "D": d, "El": el, "Az": az, "wrapD": True, "wrapAz": True, "wrapEl": True}
    opts = {
        "minVal": pk[0, 4] / 55 if pk.size else 0,
        "norm": [2, 2, 1, 1],
        "valWeight": 0.5,
        "valScale": None,
        "eps": 6,
        "minPts": 1,
    }
    cl = cluster_peaks_4d(pk, dims, opts)
    if cl["clusters"].size == 0:
        return np.zeros((0, 4), dtype=int)
    detections = cl["clusters"][:, [7, 8, 9, 10, 5]]
    detections = suppress_sidelobes(detections, np.inf, [r, d, el, az])
    detections = np.round(detections).astype(int)
    return detections[:, :4]


def _nms4d_greedy_onv(v, rad, max_peaks):
    min_peak_val = 1e-12
    a = np.abs(v) if np.iscomplexobj(v) else v
    mask = a > min_peak_val
    if not np.any(mask):
        return np.zeros((0, 5))

    r, d, el, az = a.shape
    lin_idx = np.nonzero(mask)
    vals = a[lin_idx]
    order = np.argsort(vals)[::-1]
    rr = lin_idx[0][order]
    dd = lin_idx[1][order]
    ee = lin_idx[2][order]
    aa = lin_idx[3][order]
    vals = vals[order]

    r_r, r_d, r_el, r_az = rad
    m = len(vals)
    alive = np.ones(m, dtype=bool)
    out = np.zeros((min(max_peaks, m), 5))
    np_out = 0

    i = 0
    while np_out < max_peaks and i < m:
        if not alive[i]:
            i += 1
            continue
        out[np_out, :] = [rr[i] + 1, dd[i] + 1, ee[i] + 1, aa[i] + 1, vals[i]]
        np_out += 1
        j_idx = np.where(alive)[0]
        r0, d0, e0, a0 = rr[i], dd[i], ee[i], aa[i]
        kill = (
            np.abs(rr[j_idx] - r0) <= r_r
        ) & (
            np.abs(dd[j_idx] - d0) <= r_d
        ) & (
            np.abs(ee[j_idx] - e0) <= r_el
        ) & (
            np.abs(aa[j_idx] - a0) <= r_az
        )
        alive[j_idx[kill]] = False
        i += 1
    return out[:np_out]
