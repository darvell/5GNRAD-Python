from __future__ import annotations

import numpy as np


def get_plane_coefficient(point, azimuth_deg, elevation_deg):
    azimuth_rad = np.deg2rad(azimuth_deg)
    elevation_rad = np.deg2rad(elevation_deg)

    l = np.cos(elevation_rad) * np.cos(azimuth_rad)
    m = np.cos(elevation_rad) * np.sin(azimuth_rad)
    n = np.sin(elevation_rad)

    point = np.asarray(point, dtype=float).reshape(3)
    x0, y0, z0 = point
    d = -(l * x0 + m * y0 + n * z0)
    return np.array([l, m, n, d], dtype=float)


def is_point_in_positive_half_space(points, plane_coefficients):
    points = np.asarray(points, dtype=float)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    plane = np.asarray(plane_coefficients, dtype=float).reshape(4)
    a, b, c, d = plane
    plane_value = a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d
    return plane_value > 0
