from __future__ import annotations

import numpy as np

from gnrad5.constants import SPEED_OF_LIGHT
from gnrad5.geometry.angles import vector2angle
from gnrad5.geometry.planes import get_plane_coefficient, is_point_in_positive_half_space


def estimate_scattering_geometry(
    tx,
    rx,
    delay_ns,
    aoa_deg,
    tx_heading=(0.0, 0.0),
    rx_heading=(0.0, 0.0),
    keep_sector_az=None,
):
    cspeed = SPEED_OF_LIGHT
    tx = np.asarray(tx, dtype=float).reshape(3)
    rx = np.asarray(rx, dtype=float).reshape(3)
    delay_s = np.asarray(delay_ns, dtype=float).reshape(-1) / 1e9
    aoa = np.asarray(aoa_deg, dtype=float)

    azimuth = np.deg2rad(aoa[:, 0])
    elevation = np.deg2rad(aoa[:, 1])

    ray_length = delay_s * cspeed
    is_degenerating = ray_length < np.linalg.norm(np.abs(tx - rx))

    u_aoa = np.vstack(
        [np.cos(azimuth) * np.cos(elevation), np.sin(azimuth) * np.cos(elevation), np.sin(elevation)]
    )

    d_rx_tx = rx - tx
    numerator = (cspeed * delay_s) ** 2 - np.linalg.norm(d_rx_tx) ** 2
    denom = np.vstack(
        [
            2 * (d_rx_tx @ u_aoa - cspeed * delay_s),
            2 * (d_rx_tx @ (-u_aoa) - cspeed * delay_s),
        ]
    )
    alpha = np.abs(numerator / denom)

    p1 = (rx[:, None] + u_aoa * alpha[0]).T
    p2 = (rx[:, None] - u_aoa * alpha[1]).T
    dod1 = p1 - tx
    dod2 = p2 - tx
    az1, el1 = vector2angle(dod1)
    az2, el2 = vector2angle(dod2)

    if keep_sector_az is None:
        in1 = np.ones_like(az1, dtype=bool)
        in2 = np.ones_like(az2, dtype=bool)
    else:
        in1 = _in_interval_wrap(az1, keep_sector_az[0], keep_sector_az[1])
        in2 = _in_interval_wrap(az2, keep_sector_az[0], keep_sector_az[1])

    keep_p1 = is_point_in_positive_half_space(p1, get_plane_coefficient(tx, *tx_heading))
    keep_p1 &= is_point_in_positive_half_space(p1, get_plane_coefficient(rx, *rx_heading))
    keep_p2 = is_point_in_positive_half_space(p2, get_plane_coefficient(tx, *tx_heading))
    keep_p2 &= is_point_in_positive_half_space(p2, get_plane_coefficient(rx, *rx_heading))

    keep_p1 &= ~is_degenerating & in1
    keep_p2 &= ~is_degenerating & in2

    p_out = np.full((len(delay_s), 3), np.nan)
    aod_out = np.full((len(delay_s), 2), np.nan)
    p_out[keep_p1] = p1[keep_p1]
    p_out[keep_p2] = p2[keep_p2]
    aod_out[keep_p1] = np.column_stack([az1[keep_p1], el1[keep_p1]])
    aod_out[keep_p2] = np.column_stack([az2[keep_p2], el2[keep_p2]])
    return p_out, aod_out, p1, p2


def _in_interval_wrap(az, start, end):
    az = np.mod(az, 360)
    start = float(np.mod(start, 360))
    end = float(np.mod(end, 360))
    if start <= end:
        return (az >= start - np.finfo(float).eps) & (az <= end + np.finfo(float).eps)
    return (az >= start - np.finfo(float).eps) | (az <= end + np.finfo(float).eps)
