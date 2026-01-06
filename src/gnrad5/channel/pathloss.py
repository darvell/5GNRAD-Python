from __future__ import annotations

import math
import numpy as np

from gnrad5.constants import SPEED_OF_LIGHT


def compute_los_probability(scenario: str, d2d: float, hut: float) -> float:
    if hut < 1.5:
        hut = 1.5
    if hut > 300:
        hut = 300

    if scenario == "RMaAV":
        if 1.5 <= hut <= 10:
            if d2d <= 10:
                return 1.0
            return math.exp(-((d2d - 10) / 100))
        if 10 < hut <= 40:
            p1 = max(15021 * math.log10(hut) - 16053, 1000)
            d1 = max(1350.8 * math.log10(hut) - 1602, 18)
            if d2d <= d1:
                return 1.0
            return (d1 / d2d) + math.exp(-d2d / p1) * (1 - d1 / d2d)
        if 40 < hut <= 300:
            return 1.0
        raise ValueError("hUT should be between 1.5m and 300")

    if scenario == "UMaAV":
        if 1.5 <= hut <= 22.5:
            if d2d <= 18:
                return 1.0
            if hut <= 13:
                c_prime = 0.0
            elif hut <= 23:
                c_prime = ((hut - 13) / 10) ** 1.5
            else:
                raise ValueError("hUT must be <= 23m")
            d2d_out = d2d
            base = (18 / d2d_out) + math.exp(-(d2d_out / 63)) * (1 - 18 / d2d_out)
            return base * (1 + c_prime * (5 / 4) * (d2d_out / 100) ** 3 * math.exp(-d2d_out / 150))
        if 22.5 < hut <= 100:
            p1 = 4300 * math.log10(hut) - 3800
            d1 = max(460 * math.log10(hut) - 700, 18)
            if d2d <= d1:
                return 1.0
            return (d1 / d2d) + math.exp(-d2d / p1) * (1 - d1 / d2d)
        if 100 < hut <= 300:
            return 1.0
        raise ValueError("hUT should be between 1.5m and 300")

    if scenario == "UMiAV":
        if 1.5 <= hut <= 22.5:
            d2d_out = d2d
            return (18 / d2d_out) + math.exp(-(d2d_out / 36)) * (1 - 18 / d2d_out)
        if 22.5 < hut <= 300:
            p1 = 233.98 * math.log10(hut) - 0.95
            d1 = max(294.05 * math.log10(hut) - 432.94, 18)
            return (d1 / d2d) + math.exp(-d2d / p1) * (1 - d1 / d2d)

    raise ValueError("Invalid scenario")


def compute_ut_effective_height(d2d: float, hut: float, scenario: str, rng: np.random.Generator) -> float:
    if scenario == "UMiAV":
        return 1.0
    if scenario != "UMaAV":
        raise ValueError("Invalid scenario for effective height")
    if hut < 1 or hut > 300:
        raise ValueError("hUT out of range: 1m ≤ hUT ≤ 300m")
    if d2d < 0:
        raise ValueError("d2D must be non-negative")
    if d2d <= 18:
        g_d2d = 0.0
    else:
        g_d2d = (5 / 4) * (d2d / 100) ** 3 * math.exp(-d2d / 150)
    if hut < 13:
        c_d2d = 0.0
    elif hut <= 23:
        c_d2d = ((hut - 13) / 10) ** 1.5 * g_d2d
    else:
        c_d2d = g_d2d
    probability = 1 / (1 + c_d2d)
    if rng.random() < probability:
        return 1.0
    if math.floor(hut - 1.5) < 12:
        return 1.0
    return rng.integers(12, math.floor(hut - 1.5) + 1)


def compute_shadow_fading(scenario: str, is_los: bool, hut: float, is_after_bp: bool | None = None) -> float:
    if hut < 1.5 or hut > 300:
        raise ValueError("hUT must be within 1.5m < hUT < 300m")

    if scenario == "RMaAV":
        if is_los:
            if hut <= 10:
                if is_after_bp:
                    return 6
                return 4
            return 4.2 * math.exp(-0.0046 * hut)
        if hut <= 10:
            return 8
        return 6

    if scenario == "UMaAV":
        if is_los:
            if hut <= 22.5:
                return 4
            return 4.64 * math.exp(-0.0066 * hut)
        if hut <= 22.5:
            return 4
        return 6

    if scenario == "UMiAV":
        if is_los:
            if hut <= 22.5:
                return 4
            return max(5 * math.exp(-0.01 * hut), 2)
        if hut <= 22.5:
            return 7.82
        return 8

    raise ValueError("Invalid scenario")


def compute_path_loss(scenario: str, fc: float, tx_pos: np.ndarray, rx_pos: np.ndarray, is_background: bool, rng: np.random.Generator):
    c = SPEED_OF_LIGHT
    fc_ghz = fc / 1e9
    d3d = float(np.linalg.norm(tx_pos - rx_pos))
    d2d = float(np.linalg.norm(tx_pos[:2] - rx_pos[:2]))
    hut = float(rx_pos[2])
    hbs = float(tx_pos[2])

    los_prob = compute_los_probability(scenario, d2d, hut)
    if is_background:
        is_los = False
    else:
        is_los = rng.random() < los_prob

    is_after_bp = None
    if hut < 1.5:
        hut = 1.5
    if hut > 300:
        hut = 300

    if scenario == "RMaAV":
        if 1.5 <= hut <= 10:
            h = 5
            d_bp = 2 * math.pi * hbs * hut * fc / c
            def pl1(d):
                return 20 * math.log10((40 * math.pi * d * fc_ghz) / 3) + min(0.03 * hut**1.72, 10) * math.log10(d) - min(0.044 * hut**1.72, 14.77) + 0.002 * math.log10(h) * d
            is_after_bp = d2d >= d_bp
            if is_after_bp:
                pl = pl1(d_bp) + 40 * math.log10(d3d / d_bp)
            else:
                pl = pl1(d3d)
            if not is_los:
                w = 10
                pl_nlos = 161.04 - 7.1 * math.log10(w) + 7.5 * math.log10(h) - (24.37 - 3.7 * (h / hbs) ** 2) * math.log10(hbs) + (43.42 - 3.11 * math.log10(hbs)) * (math.log10(d3d) - 3) + 20 * math.log10(fc) - (3.2 * (math.log10(11.75 * hut)) ** 2 - 4.97)
                pl = max(pl, pl_nlos)
        elif 10 < hut <= 300:
            pl = max(23.9 - 1.8 * math.log10(hut), 20) * math.log10(d3d) + 20 * math.log10((40 * math.pi * fc_ghz) / 3)
            if not is_los:
                pl_nlos = -12 + (35 - 5.3 * math.log10(hut)) * math.log10(d3d) + 20 * math.log10((4 * math.pi * fc_ghz) / 3)
                pl = max(pl, pl_nlos)
        else:
            raise ValueError("hUT out of range for RMa-AV")

    elif scenario == "UMaAV":
        if 1.5 <= hut <= 22.5:
            hbs1 = hbs - hut
            h_e = compute_ut_effective_height(d2d, hut, scenario, rng)
            hut1 = hut - h_e
            d_bp1 = 4 * hbs1 * hut1 * fc / c
            if 10 <= d2d <= d_bp1:
                pl = 28.0 + 22 * math.log10(d3d) + 20 * math.log10(fc_ghz)
            elif d2d > d_bp1 and d2d <= 5e3:
                pl = 28.0 + 40 * math.log10(d3d) + 20 * math.log10(fc_ghz) - 9 * math.log10(d_bp1**2 + (hbs - hut) ** 2)
            else:
                raise ValueError("UMaAV d2D out of range")
            if not is_los:
                pl_nlos = 13.54 + 39.08 * math.log10(d3d) + 20 * math.log10(fc_ghz) - 0.6 * (hut - 1.5)
                pl = max(pl_nlos, pl)
        elif 22.5 < hut < 300:
            if is_los:
                if d2d > 4e3:
                    raise ValueError("UMaAV d2D out of range")
                pl = 28.0 + 22 * math.log10(d3d) + 20 * math.log10(fc_ghz)
            else:
                pl = -17.5 + (46 - 7 * math.log10(hut)) * math.log10(d3d) + 20 * math.log10((40 * math.pi * fc_ghz) / 3)
        else:
            raise ValueError("UMaAV hUT out of range")

    elif scenario == "UMiAV":
        if d2d > 4e3:
            raise ValueError("UMiAV d2D out of range")
        if 1.5 <= hut <= 22.5:
            hbs1 = hbs - hut
            h_e = compute_ut_effective_height(d2d, hut, scenario, rng)
            hut1 = hut - h_e
            d_bp1 = 4 * hbs1 * hut1 * fc / c
            if 10 <= d2d <= d_bp1:
                pl = 32.4 + 21 * math.log10(d3d) + 20 * math.log10(fc_ghz)
            elif d2d > d_bp1 and d2d <= 5e3:
                pl = 32.4 + 40 * math.log10(d3d) + 20 * math.log10(fc_ghz) - 9.5 * math.log10(d_bp1**2 + (hbs - hut) ** 2)
            else:
                raise ValueError("UMiAV d2D out of range")
            if not is_los:
                pl_nlos = 35.3 * math.log10(d3d) + 22.4 + 21.3 * math.log10(fc_ghz) - 0.3 * (hut - 1.5)
                pl = max(pl_nlos, pl)
        elif 22.5 < hut < 300:
            pl1 = 92.4 + 20 * math.log10(fc_ghz) + 20 * math.log10(d3d * 1e-3)
            pl = max(pl1, 30.9 + (22.25 - 0.5 * math.log10(hut)) * math.log10(d3d) + 20 * math.log10(fc_ghz))
            if not is_los:
                pl_nlos = 32.4 + (43.2 - 7.6 * math.log10(hut)) + 20 * math.log10(fc)
                pl = max(pl, pl_nlos)
        else:
            raise ValueError("UMiAV hUT out of range")
    else:
        raise ValueError("Invalid scenario")

    sigma_sf = compute_shadow_fading(scenario, is_los, hut, is_after_bp)
    pl = pl + rng.standard_normal() * sigma_sf
    return pl, is_los
