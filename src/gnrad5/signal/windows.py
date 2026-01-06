from __future__ import annotations

import numpy as np


def get_dft_window(window_type: str, n: int, alpha: float | None = None):
    window_type = window_type.lower()
    if window_type in {"rect", "rectangle"}:
        return np.ones(n)
    if window_type == "hamming":
        return 0.54 + 0.46 * np.cos(2 * np.pi * (np.linspace(-n / 2, n / 2, n)) / n)
    if window_type == "blackmanharris":
        idx = np.arange(n)
        w = 2 * np.pi * idx / n
        a0, a1, a2, a3 = 0.35875, 0.48829, 0.14128, 0.01168
        return a0 - a1 * np.cos(w) + a2 * np.cos(2 * w) - a3 * np.cos(3 * w)
    if window_type == "gaussian":
        if alpha is None:
            alpha = 2.5
        x = np.linspace(-n / 2, n / 2, n) / (n / 2)
        return np.exp(-0.5 * (alpha * x) ** 2)
    if window_type == "kaiser":
        beta = 3.4 if alpha is None else alpha
        return np.kaiser(n, beta)
    raise ValueError(f"Unsupported window type: {window_type}")
