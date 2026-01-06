from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata


def _to_db(x: np.ndarray) -> np.ndarray:
    return 20 * np.log10(np.maximum(np.abs(x), 1e-12))


def plot_grid(grid: np.ndarray, title: str = "OFDM Grid Magnitude", db: bool = True):
    data = _to_db(grid) if db else np.abs(grid)
    fig, ax = plt.subplots()
    im = ax.imshow(data, aspect="auto", origin="lower")
    ax.set_title(title)
    ax.set_xlabel("Symbol Index")
    ax.set_ylabel("Subcarrier")
    fig.colorbar(im, ax=ax)
    return fig


def plot_rd_map(rd_map: np.ndarray, title: str = "Range-Doppler", db: bool = True):
    data = rd_map
    if data.ndim == 4:
        data = data.sum(axis=(2, 3))
    data = _to_db(data) if db else np.abs(data)
    fig, ax = plt.subplots()
    im = ax.imshow(data, aspect="auto", origin="lower")
    ax.set_title(title)
    ax.set_xlabel("Doppler Bin")
    ax.set_ylabel("Range Bin")
    fig.colorbar(im, ax=ax)
    return fig


def plot_cfar_mask(mask: np.ndarray, title: str = "CFAR Mask"):
    fig, ax = plt.subplots()
    im = ax.imshow(mask.astype(float), aspect="auto", origin="lower", cmap="gray")
    ax.set_title(title)
    ax.set_xlabel("Doppler Bin")
    ax.set_ylabel("Range Bin")
    fig.colorbar(im, ax=ax)
    return fig


def plot_detections(rd_map: np.ndarray, detections: np.ndarray, title: str = "Detections"):
    data = rd_map
    if data.ndim == 4:
        data = data.sum(axis=(2, 3))
    data = _to_db(data)
    fig, ax = plt.subplots()
    im = ax.imshow(data, aspect="auto", origin="lower")
    ax.set_title(title)
    ax.set_xlabel("Doppler Bin")
    ax.set_ylabel("Range Bin")
    if detections is not None and len(detections):
        rr = detections[:, 0] - 1
        dd = detections[:, 1] - 1
        ax.scatter(dd, rr, s=25, edgecolors="red", facecolors="none")
    fig.colorbar(im, ax=ax)
    return fig


def save_fig(fig, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_position_error_cell(
    x,
    y,
    error_values,
    x_miss,
    y_miss,
    cell_size=None,
    plot_interp: bool = False,
):
    x = np.asarray(x)
    y = np.asarray(y)
    error_values = np.asarray(error_values)
    if cell_size is None:
        cell_size = [float(np.min(x)), float(np.max(x)), float(np.min(y)), float(np.max(y))]

    fig, ax = plt.subplots()
    if plot_interp:
        xq = np.linspace(cell_size[0], cell_size[1], 200)
        yq = np.linspace(cell_size[2], cell_size[3], 200)
        xq_grid, yq_grid = np.meshgrid(xq, yq)
        err_grid = griddata((x, y), error_values, (xq_grid, yq_grid), method="cubic")
        contour = ax.contourf(xq_grid, yq_grid, err_grid, levels=3)
        fig.colorbar(contour, ax=ax, label="Position Error (m)")
        ax.plot(x_miss, y_miss, "ro", label="Miss-Detection")
    else:
        scatter = ax.scatter(x, y, s=40, c=error_values, edgecolors="k", alpha=1.0)
        fig.colorbar(scatter, ax=ax, label="Position Error (m)")
        ax.plot(x_miss, y_miss, "ro", label="Miss-Detection")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend()
    return fig
