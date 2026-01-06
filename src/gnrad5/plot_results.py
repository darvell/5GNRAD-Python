from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from gnrad5.visualize import plot_position_error_cell, save_fig


def _position_error_table(table: pd.DataFrame) -> np.ndarray:
    if "positionErrorH" in table.columns:
        return table["positionErrorH"].to_numpy()
    if "positionError" in table.columns:
        return table["positionError"].to_numpy()
    if all(col in table.columns for col in ("positionErrorX", "positionErrorY", "positionErrorZ")):
        err_h = np.column_stack([table["positionErrorX"], table["positionErrorY"]])
        return np.linalg.norm(err_h, axis=1)
    raise ValueError("Missing position error columns in error.csv")


def plot_scenarios(folder: str, plot_dir: str, eta: float = 3.4):
    base = Path(folder)
    plot_dir = Path(plot_dir)
    scenario_dirs = [d for d in base.iterdir() if d.is_dir()]

    error_mats = []
    tgt_mats = []
    names = []
    tables = []
    pos_err_full = []
    pos_err_masked = []
    masks = []
    for scenario in scenario_dirs:
        error_file = scenario / "Output" / "error.csv"
        tgt_file = scenario / "Input" / "targetConfig.txt"
        if not error_file.exists() or not tgt_file.exists():
            continue
        tgt = np.loadtxt(tgt_file)[:, :3]
        table = pd.read_csv(error_file)
        pos_err = _position_error_table(table)
        mask = pos_err != 0
        error_mats.append((table.to_numpy()[mask], pos_err[mask]))
        tgt_mats.append(tgt)
        names.append(scenario.name)
        tables.append(table)
        pos_err_full.append(pos_err)
        pos_err_masked.append(pos_err[mask])
        masks.append(mask)

    for idx, (scenario, (err_mat, pos_err)) in enumerate(zip(names, error_mats)):
        tgt = tgt_mats[idx]
        true_missed = pos_err > eta
        error_values = pos_err[~true_missed]
        fig = plot_position_error_cell(
            tgt[~true_missed, 0],
            tgt[~true_missed, 1],
            error_values,
            tgt[true_missed, 0],
            tgt[true_missed, 1],
            cell_size=[-500, 500, -500, 500],
        )
        save_fig(fig, plot_dir / f"{scenario}_pos_error.png")

    md_mat = []
    for err_mat, pos_err in error_mats:
        md_mat.append(int(np.sum(pos_err > eta)))
    if md_mat:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.bar(np.arange(len(md_mat)), md_mat)
        ax.set_xticks(np.arange(len(md_mat)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("Miss Detection Count")
        ax.set_xlabel("Scenario")
        ax.grid(axis="y")
        save_fig(fig, plot_dir / "miss_detection.png")

        fig, ax = plt.subplots()
        for (err_mat, pos_err), name in zip(error_mats, names):
            data = np.sort(pos_err[pos_err >= eta])
            if data.size:
                cdf = np.arange(1, len(data) + 1) / len(data)
                ax.plot(data, cdf, label=name)
        ax.set_xlabel("Position Error (m)")
        ax.set_ylabel("CDF")
        ax.legend()
        save_fig(fig, plot_dir / "cdf_position_error.png")

        fig, ax = plt.subplots()
        for table, name, pos_err, mask in zip(tables, names, pos_err_masked, masks):
            vel_err = np.array([])
            if "velocityError" in table.columns:
                vel_err = np.abs(table["velocityError"].to_numpy())[mask]
                vel_err = vel_err[pos_err <= eta]
            if vel_err.size:
                data = np.sort(vel_err)
                cdf = np.arange(1, len(data) + 1) / len(data)
                ax.plot(data, cdf, label=name)
        ax.set_xlabel("Velocity Error (m)")
        ax.set_ylabel("CDF")
        ax.legend()
        save_fig(fig, plot_dir / "cdf_velocity_error.png")


def main():
    parser = argparse.ArgumentParser(description="MainPlot equivalent for 5GNRad outputs")
    parser.add_argument("--folder", default="examples3GPP", help="Folder containing scenario subfolders")
    parser.add_argument("--plot-dir", default="plots_main", help="Directory for plots")
    parser.add_argument("--eta", type=float, default=3.4, help="Position error threshold")
    args = parser.parse_args()
    plot_scenarios(args.folder, args.plot_dir, args.eta)


if __name__ == "__main__":
    main()
