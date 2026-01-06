from __future__ import annotations

import numpy as np

from gnrad5.constants import SPEED_OF_LIGHT


def angle2vector(az_deg, el_deg, delay):
    az = np.asarray(az_deg, dtype=float)
    el = np.asarray(el_deg, dtype=float)
    delay = np.asarray(delay, dtype=float)
    angle_vector_norm = SPEED_OF_LIGHT * delay
    z = angle_vector_norm * np.cos(np.deg2rad(el))
    y = angle_vector_norm * np.sin(np.deg2rad(el)) * np.sin(np.deg2rad(az))
    x = angle_vector_norm * np.sin(np.deg2rad(el)) * np.cos(np.deg2rad(az))
    return np.column_stack([x, y, z])


def vector2angle(vec):
    vec = np.asarray(vec, dtype=float)
    az = np.mod(np.rad2deg(np.arctan2(vec[:, 1], vec[:, 0])), 360)
    el = np.rad2deg(np.arctan2(vec[:, 2], np.sqrt(vec[:, 0] ** 2 + vec[:, 1] ** 2)))
    return az, el


def wrap_to_180(angle):
    angle = np.asarray(angle, dtype=float)
    return (angle + 180) % 360 - 180
