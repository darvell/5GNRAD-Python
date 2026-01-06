from __future__ import annotations

import math
import numpy as np


def get_sigma_rcs(
    st: str,
    spst: str = "single",
    st_direction=None,
    angles_rcs=None,
    n: int = 1,
    return_large_scale: bool = False,
    rng: np.random.Generator | None = None,
):
    if rng is None:
        rng = np.random.default_rng()

    st = st.lower()
    if st_direction is None:
        st_direction = [0.0, 90.0]
    if angles_rcs is None:
        angles_rcs = [0.0, 0.0, 0.0, 0.0]

    theta_i, phi_i, theta_s, phi_s = np.deg2rad(angles_rcs)
    if theta_i == theta_s and phi_i == phi_s:
        sensing_mode = "monostatic"
        beta = 0.0
    else:
        sensing_mode = "bistatic"
        u_i = np.array([np.sin(theta_i) * np.cos(phi_i), np.sin(theta_i) * np.sin(phi_i), np.cos(theta_i)])
        u_s = np.array([np.sin(theta_s) * np.cos(phi_s), np.sin(theta_s) * np.sin(phi_s), np.cos(theta_s)])
        u_bis = u_i + u_s
        u_bis = u_bis / np.linalg.norm(u_bis)
        theta = math.acos(u_bis[2])
        phi = math.atan2(u_bis[1], u_bis[0])
        dot_prod = float(np.dot(u_i, u_s))
        beta = math.acos(dot_prod)

    if st == "uav-small":
        if sensing_mode == "bistatic":
            sigma_m = max(-12.81 - 3 * math.sin(beta / 2), -math.inf)
        else:
            sigma_m = -12.81
        sigma_d = 0.0
        sigma_s = _angular_independent_rcs(3.74, (1, n), rng)
    elif st == "human-model-1":
        sigma_m = -1.37
        sigma_d = 0.0
        sigma_s = _angular_independent_rcs(3.94, (1, n), rng)
    elif st == "vehicle":
        table = _rcs_table_vehicle(spst)
        if sensing_mode == "bistatic":
            sigma_m = _sigma_md(table, theta, phi, 6.05, 1.33, beta)
            sigma_d = 1.0
        else:
            sigma_m = -20.0
            sigma_d = _angular_dependent_rcs(st_direction[0], st_direction[1], table)
        sigma_s = _angular_independent_rcs(3.41, (np.atleast_1d(sigma_d).shape[0], n), rng)
    elif st == "agv":
        table = _rcs_table_agv(spst)
        if sensing_mode == "bistatic":
            sigma_m = _sigma_md(table, theta, phi, 6.05, 1.33, beta)
            sigma_d = 1.0
        else:
            sigma_m = -4.25
            sigma_d = _angular_dependent_rcs(st_direction[0], st_direction[1], table)
        sigma_s = _angular_independent_rcs(2.51, (np.atleast_1d(sigma_d).shape[0], n), rng)
    elif st == "uav-large":
        table = _rcs_table_uav_large()
        if sensing_mode == "bistatic":
            sigma_m = _sigma_md(table, theta, phi, 6.05, 1.33, beta)
            sigma_d = 1.0
        else:
            sigma_m = -5.85
            sigma_d = 0.0
        sigma_s = _angular_independent_rcs(2.5, (1, n), rng)
    else:
        raise ValueError(f"Unsupported target type: {st}")

    if return_large_scale:
        return sigma_m
    return sigma_m + sigma_d + sigma_s


def _angular_independent_rcs(sigma_s_db: float, shape, rng: np.random.Generator):
    return (-math.log(10) / 20) * sigma_s_db**2 + sigma_s_db * rng.standard_normal(shape)


def _angular_dependent_rcs(theta: float, phi: float, table):
    phi = 360 - phi if phi >= 315 else phi
    index = (table["phi_range"][:, 0] <= phi) & (table["phi_range"][:, 1] >= phi) & (
        table["theta_range"][:, 0] <= theta) & (table["theta_range"][:, 1] >= theta)
    if not np.any(index):
        return 0.0
    idx = np.where(index)[0][0]
    theta_center = table["theta_center"][idx]
    theta_3db = table["theta_3db"][idx]
    phi_center = table["phi_center"][idx]
    phi_3db = table["phi_3db"][idx]
    g_max = table["g_max"][idx]
    sigma_max = table["sigma_max"][idx]

    sigma_v = 12 * ((theta - theta_center) / theta_3db) ** 2
    sigma_v = -min(sigma_v, sigma_max)
    sigma_h = 12 * ((phi - phi_center) / phi_3db) ** 2
    sigma_h = -min(sigma_h, sigma_max)
    return g_max - min(-(sigma_v + sigma_h), sigma_max)


def _sigma_md(table, theta, phi, k1, k2, beta):
    sigma_v = -np.minimum(12 * ((theta - table["theta_center"]) / table["theta_3db"]) ** 2, table["sigma_max"])
    sigma_h = -np.minimum(12 * ((phi - table["phi_center"]) / table["phi_3db"]) ** 2, table["sigma_max"])
    term1 = table["g_max"] - np.minimum(-(sigma_v + sigma_h), table["sigma_max"])
    term2 = -k1 * math.sin(k2 * beta / 2) + 5 * math.log10(math.cos(beta / 2))
    term2 = term2 if np.isfinite(term2) else -math.inf
    return np.maximum.reduce([term1 + term2, table["g_max"] - table["sigma_max"], -math.inf * np.ones_like(term1)])


def _rcs_table_vehicle(spst: str):
    if spst == "multi":
        return {
            "phi_center": np.array([90, 180, 270, 0, np.nan]),
            "phi_3db": np.array([26.90, 36.32, 26.90, 40.54, np.nan]),
            "theta_center": np.array([79.70, 79.65, 79.70, 71.75, 0.0]),
            "theta_3db": np.array([44.42, 36.73, 44.42, 29.13, 18.13]),
            "g_max": np.array([20.60, 13.90, 20.60, 14.99, 21.12]),
            "sigma_max": np.array([20.52, 13.82, 20.52, 14.91, 21.05]),
            "theta_range": np.array([[0, 180], [0, 180], [0, 180], [0, 180], [0, 180]]),
            "phi_range": np.array([[0, 360], [0, 360], [0, 360], [0, 360], [0, 360]]),
        }
    tol = 1e-10
    return {
        "phi_center": np.array([90, 180, 270, 0, np.nan]),
        "phi_3db": np.array([26.90, 36.32, 26.90, 40.54, np.nan]),
        "theta_center": np.array([79.70, 79.65, 79.70, 71.75, 0.0]),
        "theta_3db": np.array([44.42, 36.73, 44.42, 29.13, 18.13]),
        "g_max": np.array([20.75, 14.56, 20.75, 15.52, 21.26]),
        "sigma_max": np.array([13.68, 7.50, 13.68, 8.45, 14.19]),
        "theta_range": np.array([[30, 180], [30, 180], [30, 180], [30, 180], [0, 30 - tol]]),
        "phi_range": np.array([[45 + tol, 135], [135 + tol, 225], [225 + tol, 315], [-45 + tol, 45], [0, 360 - tol]]),
    }


def _rcs_table_uav_large():
    return {
        "phi_center": np.array([90, 180, 270, 0, np.nan, np.nan]),
        "phi_3db": np.array([7.13, 10.09, 7.13, 14.19, np.nan, np.nan]),
        "theta_center": np.array([90, 90, 90, 90, 180, 0]),
        "theta_3db": np.array([8.68, 11.43, 8.68, 16.53, 4.93, 4.93]),
        "g_max": np.array([7.43, 3.99, 7.43, 1.02, 13.55, 13.55]),
        "sigma_max": np.array([14.30, 10.86, 14.30, 7.89, 20.42, 20.42]),
        "theta_range": np.array([[45, 135], [45, 135], [45, 135], [45, 135], [135, 180], [0, 45]]),
        "phi_range": np.array([[45, 135], [135, 225], [225, 315], [-45, 45], [0, 360], [0, 360]]),
    }


def _rcs_table_agv(spst: str):
    if spst == "multi":
        return {
            "phi_center": np.array([0, 90, 180, 270, np.nan]),
            "phi_3db": np.array([13.68, 15.53, 12.49, 15.53, np.nan]),
            "theta_center": np.array([90, 75, 90, 75, 0.0]),
            "theta_3db": np.array([13.68, 20.03, 11.89, 20.03, 11.44]),
            "g_max": np.array([13, 7.27, 10.98, 7.27, 11.77]),
            "sigma_max": np.array([30.26, 24.53, 28.24, 24.53, 29.03]),
            "theta_range": np.array([[0, 180], [0, 180], [0, 180], [0, 180], [0, 180]]),
            "phi_range": np.array([[0, 360], [0, 360], [0, 360], [0, 360], [0, 360]]),
        }
    return {
        "phi_center": np.array([0, 90, 180, 270, np.nan]),
        "phi_3db": np.array([13.68, 15.53, 12.49, 15.53, np.nan]),
        "theta_center": np.array([90, 75, 90, 75, 0.0]),
        "theta_3db": np.array([13.68, 20.03, 11.89, 20.03, 11.44]),
        "g_max": np.array([13.02, 7.33, 11.01, 7.33, 11.79]),
        "sigma_max": np.array([23.29, 17.60, 21.28, 17.60, 22.06]),
        "theta_range": np.array([[30, 180], [30, 180], [30, 180], [30, 180], [0, 30]]),
        "phi_range": np.array([[-45, 45], [45, 135], [135, 225], [225, 315], [0, 360]]),
    }
