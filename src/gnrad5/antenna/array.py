from __future__ import annotations

import numpy as np

from gnrad5.constants import SPEED_OF_LIGHT


class URA:
    def __init__(self, shape: tuple[int, int], element_spacing: float):
        self.shape = shape
        self.element_spacing = element_spacing
        self.num_elements = shape[0] * shape[1]

    def element_positions(self):
        dv, dh = self.shape
        idx_v = np.arange(dv)
        idx_h = np.arange(dh)
        yy, zz = np.meshgrid(idx_h * self.element_spacing, idx_v * self.element_spacing)
        xx = np.zeros_like(yy)
        pos = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()])
        return pos


def antenna_sorted_index(array: URA):
    pos = array.element_positions()
    ypos = pos[1, :]
    zpos = pos[2, :]
    ylist = np.unique(ypos)
    zlist = np.unique(zpos)
    ny = len(ylist)
    nz = len(zlist)
    if ny * nz != len(ypos):
        raise ValueError("Element grid size mismatch")
    idx_grid = np.empty((nz, ny), dtype=int)
    for iz, zval in enumerate(zlist):
        for iy, yval in enumerate(ylist):
            idx = np.where((np.abs(zpos - zval) < 1e-9) & (np.abs(ypos - yval) < 1e-9))[0]
            if idx.size == 0:
                raise ValueError("Missing element in grid")
            idx_grid[iz, iy] = idx[0]
    return idx_grid.ravel()


def array_angle_grid(num_pts_vert: int, num_pts_horz: int, d_vert: float, d_horz: float, lamb: float):
    kz = np.arange(-num_pts_vert / 2, num_pts_vert / 2)
    ky = np.arange(-num_pts_horz / 2, num_pts_horz / 2)
    fz = kz / num_pts_vert
    fy = ky / num_pts_horz
    uz = (lamb / d_vert) * fz
    uy = (lamb / d_horz) * fy
    uz_grid, uy_grid = np.meshgrid(uz, uy, indexing="ij")
    vis = (uz_grid**2 + uy_grid**2) <= 1 + np.finfo(float).eps
    el_grid = np.rad2deg(np.arcsin(np.clip(uz_grid, -1, 1)))
    az_grid = np.rad2deg(np.arctan2(uy_grid, np.sqrt(np.maximum(0, 1 - uz_grid**2 - uy_grid**2))))
    return az_grid, el_grid, vis


def steering_vector(fc: float, az_el_deg: np.ndarray, array: URA):
    c = SPEED_OF_LIGHT
    lamb = c / fc
    k = 2 * np.pi / lamb
    az = np.deg2rad(az_el_deg[0, :])
    el = np.deg2rad(az_el_deg[1, :])

    pos = array.element_positions()
    yy = pos[1, :]
    zz = pos[2, :]

    ux = np.cos(el) * np.cos(az)
    uy = np.cos(el) * np.sin(az)
    uz = np.sin(el)
    phase = k * (yy[:, None] * uy + zz[:, None] * uz)
    return np.exp(1j * phase)


def build_partial_combiner(array: URA, fc: float, n_rf: int, beam_angles: np.ndarray):
    n_rx = array.num_elements
    if n_rx % n_rf != 0:
        raise ValueError("Nrx must be divisible by NRF")
    n_sub = n_rx // n_rf
    full = steering_vector(fc, beam_angles.T, array)
    w_rf = np.zeros((n_rx, n_rf), dtype=np.complex128)
    for r in range(n_rf):
        idx = slice(r * n_sub, (r + 1) * n_sub)
        w_phase = np.exp(-1j * np.angle(full[idx, r]))
        w_rf[idx, r] = w_phase / np.sqrt(n_sub)
    return w_rf


def beam_grid(
    az_min: float,
    az_max: float,
    el_min: float,
    el_max: float,
    n_az: int,
    n_el: int,
):
    if n_az <= 0 or n_el <= 0:
        raise ValueError("Beam grid sizes must be positive")
    az = np.linspace(az_min, az_max, n_az)
    el = np.linspace(el_min, el_max, n_el)
    az_grid, el_grid = np.meshgrid(az, el)
    return np.column_stack([az_grid.ravel(), el_grid.ravel()])
