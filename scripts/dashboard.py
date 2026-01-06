from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

from gnrad5.config.models import Target
from gnrad5.paper_chain import _VIRTUAL_RX_PARAMS, run_paper_chain_debug
from gnrad5.paper_scenarios import build_paper_scenario
from gnrad5.signal import build_prs_grid


RUNS_DIR = Path("experiments")
RUNS_DIR.mkdir(parents=True, exist_ok=True)
META_NAME = "run.json"
LOG_NAME = "run.log"
OUT_NAME = "results.csv"


@dataclass
class RunInfo:
    run_id: str
    path: Path
    meta: dict[str, Any]

    @property
    def log_path(self) -> Path:
        return self.path / LOG_NAME

    @property
    def out_path(self) -> Path:
        return self.path / OUT_NAME

    @property
    def pid(self) -> int | None:
        pid = self.meta.get("pid")
        return int(pid) if pid is not None else None

    @property
    def pgid(self) -> int | None:
        pgid = self.meta.get("pgid")
        return int(pgid) if pgid is not None else None


def _parse_list(raw: str) -> list[float]:
    vals: list[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            vals.append(float(item))
        except ValueError:
            continue
    return vals


def _load_runs() -> list[RunInfo]:
    runs: list[RunInfo] = []
    for path in sorted(RUNS_DIR.iterdir()):
        meta_path = path / META_NAME
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        run_id = meta.get("run_id", path.name)
        runs.append(RunInfo(run_id=run_id, path=path, meta=meta))
    return sorted(runs, key=lambda r: r.meta.get("started_at", 0), reverse=True)


def _write_meta(run_path: Path, meta: dict[str, Any]) -> None:
    meta_path = run_path / META_NAME
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))


def _is_running(pid: int | None, pgid: int | None = None) -> bool:
    if pgid is not None and pgid > 0:
        try:
            if pgid != os.getpgrp():
                os.killpg(pgid, 0)
                return True
        except PermissionError:
            return True
        except OSError:
            pass
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _stop_process(pid: int | None) -> None:
    if pid is None or pid <= 0:
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return


def _stop_group(pgid: int | None, pid: int | None) -> None:
    if pgid is None or pgid <= 0:
        _stop_process(pid)
        return
    try:
        if pgid == os.getpgrp():
            _stop_process(pid)
            return
        os.killpg(pgid, signal.SIGTERM)
    except OSError:
        _stop_process(pid)


def _read_tail(path: Path, max_lines: int = 200) -> list[str]:
    if not path.exists():
        return []
    try:
        lines = path.read_text().splitlines()
    except Exception:
        return []
    if len(lines) <= max_lines:
        return lines
    return lines[-max_lines:]


def _parse_progress(lines: list[str]) -> tuple[dict[tuple[str, float], tuple[int, int]], str | None]:
    progress: dict[tuple[str, float], tuple[int, int]] = {}
    last_line = None
    pattern = re.compile(r"\[(?P<scen>\w+) h=(?P<alt>[\d.]+)m\] (?P<done>\d+)/(?P<total>\d+)")
    for line in lines:
        match = pattern.search(line)
        if not match:
            continue
        scen = match.group("scen").upper()
        alt = float(match.group("alt"))
        done = int(match.group("done"))
        total = int(match.group("total"))
        progress[(scen, alt)] = (done, total)
        last_line = line
    return progress, last_line


def _overall_progress(
    progress: dict[tuple[str, float], tuple[int, int]],
    scenarios: list[str],
    altitudes: list[float],
) -> float:
    if not scenarios or not altitudes:
        return 0.0
    total_tasks = 0
    completed = 0
    for scen in scenarios:
        for alt in altitudes:
            key = (scen.upper(), float(alt))
            done, total = progress.get(key, (0, 0))
            if total <= 0:
                total = max(done, 0)
            total_tasks += total
            completed += min(done, total)
    if total_tasks <= 0:
        return 0.0
    return completed / total_tasks


def _plot_results(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No results in CSV yet.")
        return
    metric = st.selectbox(
        "Metric",
        ["p_d", "p_fa", "p_miss", "mean_range_err_m", "mean_angle_err_deg", "mean_pos_err_m"],
        index=0,
    )
    eta_vals = sorted(df["eta_db"].unique().tolist())
    eta_sel = st.selectbox("η (dB)", eta_vals, index=eta_vals.index(3.4) if 3.4 in eta_vals else 0)
    view = df[df["eta_db"] == eta_sel]
    for scen in sorted(view["scenario"].unique()):
        sub = view[view["scenario"] == scen].sort_values("altitude_m")
        st.write(f"Scenario: {scen}")
        st.line_chart(sub.set_index("altitude_m")[metric])


# -----------------
# Learn-mode helpers
# -----------------


@st.cache_data(show_spinner=False)
def _simulate_single_drop(
    scenario: str,
    altitude_m: float,
    range_m: float,
    az_deg: float,
    target_absent: bool,
    n_rp: int,
    eirp_dbm: float,
    eta_db: float,
    sensing_symbols: int,
    beam_az: int,
    beam_el: int,
    beam_chunk: int,
    add_noise: bool,
    seed: int,
) -> tuple[object, dict[str, Any], dict[str, Any]]:
    spec = build_paper_scenario(
        scenario,
        altitude_m,
        drops=1,
        seed=seed,
        target_absent=target_absent,
        fixed_rcs_dbsm=-12.81,
        fixed_nrp=int(n_rp),
        eirp_dbm=float(eirp_dbm),
    )
    spec.sens.number_sensing_symbols = int(sensing_symbols)
    spec.sens.doppler_fft_len = int(sensing_symbols)

    if target_absent:
        target = Target(position=[], velocity=[])
    else:
        x = float(range_m) * float(np.cos(np.deg2rad(az_deg)))
        y = float(range_m) * float(np.sin(np.deg2rad(az_deg)))
        target = Target(position=[[x, y, float(altitude_m)]], velocity=[[0.0, 0.0, 0.0]])

    det, dbg = run_paper_chain_debug(
        spec.sim,
        spec.prs,
        spec.sens,
        spec.geometry,
        target,
        seed=seed,
        fixed_nrp=int(n_rp),
        eirp_dbm=float(eirp_dbm),
        eta_db=float(eta_db),
        codebook_id="grid",
        beam_az=int(beam_az),
        beam_el=int(beam_el),
        beam_chunk=int(beam_chunk),
        add_noise=bool(add_noise),
        max_range_bins=512,
    )
    return det, dbg, {
        "bs_pos": np.asarray(spec.geometry.tx[0], dtype=float),
        "target": target,
        "scenario": scenario,
        "altitude_m": float(altitude_m),
        "range_m": float(range_m),
        "az_deg": float(az_deg),
    }


@st.cache_data(show_spinner=False)
def _prs_grid_preview(scenario: str, comb_size: int, num_prs_symbols: int) -> np.ndarray:
    spec = build_paper_scenario(
        scenario,
        altitude_m=25.0,
        drops=1,
        seed=0,
        target_absent=True,
        fixed_nrp=3,
        eirp_dbm=75.0,
    )
    spec.prs.comb_size = int(comb_size)
    spec.prs.num_prs_symbols = int(num_prs_symbols)
    prs_grid = build_prs_grid(spec.sim, spec.prs, number_sensing_symbols=1)
    grid = prs_grid.grid

    nfft = grid.shape[0]
    n_sc = spec.sim.carrier_n_size_grid * 12
    start = (nfft - n_sc) // 2
    end = start + n_sc
    view = np.abs(grid[start:end, :14]) > 0
    return view.astype(np.uint8)


def _plot_prs_grid(view: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.imshow(view, aspect="auto", origin="lower", interpolation="nearest")
    ax.set_xlabel("OFDM symbol (within slot)")
    ax.set_ylabel("Subcarrier (active band)")
    ax.set_title("PRS resource mapping (non-zero REs)")
    st.pyplot(fig, clear_figure=True)


def _plot_virtual_rx_distributions(scenario_key: str, seed: int = 0) -> None:
    params = _VIRTUAL_RX_PARAMS.get(scenario_key)
    if not params:
        st.info("Virtual Rx distributions not available for this scenario")
        return
    alpha = float(params["alpha"])
    beta = float(params["beta"])
    mu = float(params["mu"])
    sigma = float(params["sigma"])
    rng = np.random.default_rng(seed)

    samples = 5000
    r = rng.gamma(alpha, beta, size=samples)
    phi = rng.normal(mu, sigma, size=samples)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
    ax1.hist(r, bins=60, density=True, alpha=0.7)
    x = np.linspace(0, np.percentile(r, 99.5), 400)
    ax1.plot(x, stats.gamma(a=alpha, scale=beta).pdf(x), linewidth=2)
    ax1.set_title("Virtual Rx range r")
    ax1.set_xlabel("meters")
    ax1.set_ylabel("PDF")

    ax2.hist(phi, bins=60, density=True, alpha=0.7)
    x2 = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 400)
    ax2.plot(x2, stats.norm(loc=mu, scale=sigma).pdf(x2), linewidth=2)
    ax2.set_title("Virtual Rx azimuth φ")
    ax2.set_xlabel("degrees")
    ax2.set_ylabel("PDF")
    st.pyplot(fig, clear_figure=True)

    st.caption(
        f"From 10592557: r ~ Gamma(α={alpha:.4g}, β={beta:.4g} m), φ ~ Normal(μ={mu:.4g}°, σ={sigma:.4g}°)."
    )


def _plot_paths(debug: dict[str, Any]) -> None:
    paths = debug.get("paths")
    if not isinstance(paths, dict):
        st.info("No path debug data available")
        return
    delays = np.asarray(paths.get("delays_s", []), dtype=float).reshape(-1)
    gains = np.asarray(paths.get("gains_lin", []), dtype=float).reshape(-1)
    aoa_az = np.asarray(paths.get("aoa_az_deg", []), dtype=float).reshape(-1)
    aoa_el = np.asarray(paths.get("aoa_el_deg", []), dtype=float).reshape(-1)
    if delays.size == 0:
        st.info("No paths")
        return
    gains_db = 10 * np.log10(np.maximum(gains, 1e-30))

    fig, ax = plt.subplots(figsize=(8, 3))
    sc = ax.scatter(delays * 1e6, gains_db, c=aoa_az, cmap="viridis", s=20)
    ax.set_xlabel("Delay (µs)")
    ax.set_ylabel("Path gain (dB, rel.)")
    ax.set_title("Multipath components (colored by AoA azimuth)")
    fig.colorbar(sc, ax=ax, label="AoA azimuth (deg)")
    st.pyplot(fig, clear_figure=True)

    with st.expander("Path table (top 15 by gain)"):
        order = np.argsort(-gains_db)
        rows = []
        for idx in order[:15]:
            rows.append(
                {
                    "delay_us": float(delays[idx] * 1e6),
                    "gain_db": float(gains_db[idx]),
                    "aoa_az": float(aoa_az[idx]) if aoa_az.size else float("nan"),
                    "aoa_el": float(aoa_el[idx]) if aoa_el.size else float("nan"),
                }
            )
        st.dataframe(pd.DataFrame.from_records(rows), use_container_width=True)


def _plot_beam_sweep(det, debug: dict[str, Any]) -> None:
    scores = debug.get("beam_scores")
    beams = debug.get("beam_grid")
    best_idx = debug.get("best_idx", -1)
    if scores is None or beams is None:
        st.info("Beam sweep debug not available")
        return
    scores = np.asarray(scores, dtype=float).reshape(-1)
    beams = np.asarray(beams, dtype=float)
    score_db = 10 * np.log10(np.maximum(scores, 1e-30))

    fig, ax = plt.subplots(figsize=(6, 4))
    sc = ax.scatter(beams[:, 0], beams[:, 1], c=score_db, cmap="magma", s=35)
    if isinstance(best_idx, int) and 0 <= best_idx < beams.shape[0]:
        ax.scatter([beams[best_idx, 0]], [beams[best_idx, 1]], s=200, facecolors="none", edgecolors="cyan")
    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("Elevation (deg)")
    ax.set_title("Beam sweep score")
    fig.colorbar(sc, ax=ax, label="Score (dB)")
    st.pyplot(fig, clear_figure=True)

    mode = debug.get("beam_score_mode", "?")
    st.caption(f"Beam scoring mode: `{mode}`. Best beam = ({det.best_az:.1f}°, {det.best_el:.1f}°).")


def _plot_range_profile(det, debug: dict[str, Any]) -> None:
    range_power = debug.get("range_power")
    if range_power is None:
        st.info("Range profile not available (no detection stage executed)")
        return
    rp = np.asarray(range_power, dtype=float).reshape(-1)
    rb = np.asarray(det.range_bins, dtype=float).reshape(-1)
    n = min(rp.size, rb.size)
    rp = rp[:n]
    rb = rb[:n]
    rp_db = 10 * np.log10(np.maximum(rp, 1e-30))

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(rb, rp_db)
    if np.isfinite(det.best_range_m):
        ax.axvline(det.best_range_m, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Power (dB)")
    ax.set_title("Range profile")
    st.pyplot(fig, clear_figure=True)


def _plot_geometry(meta: dict[str, Any], det) -> None:
    bs = np.asarray(meta["bs_pos"], dtype=float)
    tgt = meta["target"]
    est = det.best_pos
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter([bs[0]], [bs[1]], label="BS", marker="^", s=120)
    if tgt.position:
        p = np.asarray(tgt.position[0], dtype=float)
        ax.scatter([p[0]], [p[1]], label="Target (true)", s=80)
    if est is not None and np.all(np.isfinite(est)):
        ax.scatter([est[0]], [est[1]], label="Estimate", s=80)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Top-down geometry")
    ax.legend(loc="best")
    st.pyplot(fig, clear_figure=True)


def _render_learn() -> None:
    st.header("Learn the paper (interactive walkthrough)")
    st.markdown(
        """
This page is designed as a guided, visual explanation of the paper:

1. **PRS grid** — what the BS transmits and why comb mapping matters.
2. **Channel + clutter** — how background multipath and the UAV target are modeled.
3. **Beam sweep (AoA)** — why we pick the best beam rather than estimating a continuous angle.
4. **Range processing** — IFFT across PRS subcarriers.
5. **Detection + position** — PAR thresholding and 3D reconstruction.

Tip: start with the **Interactive (fast)** preset.
"""
    )

    with st.sidebar:
        st.subheader("Demo controls")
        preset = st.selectbox("Preset", ["Interactive (fast)", "Paper-like"], index=0)
        scenario = st.selectbox("Scenario", ["UMI", "UMA"], index=0)
        altitude = st.selectbox("UAV altitude (m)", [25.0, 50.0, 100.0, 200.0], index=0)
        target_absent = st.checkbox("Target absent (false alarm case)", value=False)
        seed = st.number_input("Seed", min_value=0, value=0, step=1)

        st.divider()
        st.subheader("Target geometry")
        range_m = st.slider("Target range (m)", min_value=10.0, max_value=250.0, value=120.0, step=1.0)
        az_deg = st.slider("Target azimuth (deg)", min_value=-60.0, max_value=60.0, value=10.0, step=1.0)

        st.divider()
        st.subheader("Environment")
        n_rp = st.slider("N_RP (background RPs)", min_value=1, max_value=10, value=3, step=1)
        add_noise = st.checkbox("Add noise", value=True)
        eirp_dbm = st.slider("EIRP (dBm/100MHz)", min_value=50.0, max_value=80.0, value=75.0, step=0.5)
        eta_db = st.slider("Detection threshold η (dB)", min_value=0.0, max_value=10.0, value=3.4, step=0.1)

        st.divider()
        st.subheader("Resolution / speed")
        if preset.startswith("Interactive"):
            sensing_symbols_default = 64
            beam_az_default = 17
            beam_el_default = 9
            beam_chunk_default = 8
        else:
            sensing_symbols_default = 256
            beam_az_default = 25
            beam_el_default = 13
            beam_chunk_default = 4

        sensing_symbols = st.selectbox(
            "Sensing symbols",
            [64, 128, 256],
            index=[64, 128, 256].index(sensing_symbols_default),
        )
        beam_az = st.slider("Beam az points", min_value=7, max_value=41, value=int(beam_az_default), step=2)
        beam_el = st.slider("Beam el points", min_value=5, max_value=25, value=int(beam_el_default), step=2)
        beam_chunk = st.slider("Beam chunk", min_value=1, max_value=32, value=int(beam_chunk_default), step=1)

        run = st.button("Run single-drop demo", type="primary")

    if not run:
        st.info("Set parameters in the sidebar and click **Run single-drop demo**.")
        return

    with st.spinner("Simulating one drop…"):
        det, dbg, meta = _simulate_single_drop(
            scenario,
            altitude,
            range_m,
            az_deg,
            target_absent,
            int(n_rp),
            float(eirp_dbm),
            float(eta_db),
            int(sensing_symbols),
            int(beam_az),
            int(beam_el),
            int(beam_chunk),
            bool(add_noise),
            int(seed),
        )

    st.subheader("Outcome")
    cols = st.columns(4)
    cols[0].metric("Detected", str(det.detected))
    cols[1].metric("Best PAR (dB)", f"{det.best_par_db:.2f}")
    cols[2].metric(
        "Estimated range (m)",
        f"{det.best_range_m:.2f}" if np.isfinite(det.best_range_m) else "nan",
    )
    cols[3].metric("Best beam (az, el)", f"({det.best_az:.1f}°, {det.best_el:.1f}°)")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "1) PRS",
            "2) Channel & clutter",
            "3) Beam sweep",
            "4) Range + detection",
            "5) Position",
        ]
    )

    with tab1:
        st.markdown(
            """
### What is PRS?

The paper uses **Positioning Reference Signals (PRS)** because they are standardized, known at the receiver,
and have good correlation properties.

A key detail is the **comb mapping**: only every \(K\)-th subcarrier is active.
That changes the effective frequency sampling and the achievable range resolution.
"""
        )
        view = _prs_grid_preview(scenario=scenario, comb_size=4, num_prs_symbols=4)
        _plot_prs_grid(view)

    with tab2:
        st.markdown(
            """
### Background clutter via virtual reference points

The background is modeled by dropping **virtual reference points (RPs)** around the BS.
The IWCMC'24 work (10592557) reports fitted distributions for RP range and azimuth.

Below you can see those distributions and the multipath components used in this drop.
"""
        )
        scenario_key = "UMiAV" if scenario == "UMI" else "UMaAV"
        _plot_virtual_rx_distributions(scenario_key, seed=int(seed))
        _plot_paths(dbg)

    with tab3:
        st.markdown(
            """
### AoA via beam sweep

We score each beam direction and pick the best one. This mimics NR beam management.
"""
        )
        _plot_beam_sweep(det, dbg)

    with tab4:
        st.markdown(
            """
### Range processing + PAR detection

After beam selection, we compute a range profile by taking an **IFFT across PRS subcarriers**.
Detection uses **Peak-to-Average Ratio (PAR)**: a strong target creates a sharp peak.
"""
        )
        _plot_range_profile(det, dbg)
        st.caption(f"Detected if PAR ≥ η. Here: PAR={det.best_par_db:.2f} dB, η={eta_db:.2f} dB")

    with tab5:
        st.markdown(
            """
### 3D position reconstruction

The paper reconstructs position as:

\[ \hat{\mathbf{x}} = \mathbf{x}_{BS} + \hat{R}\,\mathbf{u}(\hat{\theta}, \hat{\phi}) \]

where \(\mathbf{u}\) is the unit vector defined by the estimated azimuth/elevation.
"""
        )
        _plot_geometry(meta, det)


def _render_runs() -> None:
    with st.sidebar:
        st.header("New Run")
        scenario_choice = st.multiselect("Scenarios", ["UMI", "UMA"], default=["UMI", "UMA"])
        altitudes_raw = st.text_input("Altitudes (m, comma-separated)", "25,50,100,200")
        drops = st.number_input("Drops per altitude", min_value=1, value=4000, step=100)
        etas_raw = st.text_input("ETAs (dB, comma-separated)", "1.5,2.5,3.4,4.5,6.0")

        st.subheader("Beam/Codebook")
        codebook_mode = st.selectbox("Codebook", ["grid", "type1 (default)"])
        beam_az = st.number_input("Beam azimuth points", min_value=1, value=25, step=1)
        beam_el = st.number_input("Beam elevation points", min_value=1, value=13, step=1)
        beam_chunk = st.number_input("Beam chunk", min_value=1, value=4, step=1)

        st.subheader("Channel/Noise")
        fixed_nrp = st.number_input("N_RP", min_value=1, value=3, step=1)
        fixed_rcs_dbsm = st.number_input("Fixed RCS (dBsm)", value=-12.81, step=0.1, format="%.2f")
        eirp_dbm = st.number_input("EIRP (dBm/100MHz)", value=75.0, step=0.5)
        no_noise = st.checkbox("Disable noise", value=False)

        st.subheader("Runtime")
        workers = st.number_input("Workers", min_value=1, value=64, step=1)
        max_range_bins = st.number_input("Max range bins", min_value=32, value=256, step=32)
        progress_every = st.number_input("Progress every N", min_value=1, value=50, step=1)
        progress_interval = st.number_input("Progress interval (s)", min_value=1.0, value=10.0, step=1.0)
        omp_threads = st.number_input("OMP/MKL/OPENBLAS threads", min_value=1, value=1, step=1)

        start = st.button("Start run", type="primary")

    scenarios = scenario_choice if scenario_choice else ["UMI", "UMA"]
    altitudes = _parse_list(altitudes_raw)
    _ = _parse_list(etas_raw)

    if start:
        run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        run_path = RUNS_DIR / run_id
        run_path.mkdir(parents=True, exist_ok=True)
        log_path = run_path / LOG_NAME
        out_path = run_path / OUT_NAME

        cmd = [sys.executable, "scripts/validate_paper.py"]
        cmd += ["--scenario", "both" if len(scenarios) == 2 else scenarios[0]]
        cmd += ["--altitudes", altitudes_raw]
        cmd += ["--drops", str(int(drops))]
        cmd += ["--etas", etas_raw]
        cmd += ["--beam-az", str(int(beam_az))]
        cmd += ["--beam-el", str(int(beam_el))]
        cmd += ["--beam-chunk", str(int(beam_chunk))]
        cmd += ["--workers", str(int(workers))]
        cmd += ["--max-range-bins", str(int(max_range_bins))]
        cmd += ["--progress-every", str(int(progress_every))]
        cmd += ["--progress-interval", str(float(progress_interval))]
        cmd += ["--output", str(out_path)]
        cmd += ["--fixed-nrp", str(int(fixed_nrp))]
        cmd += ["--fixed-rcs-dbsm", str(float(fixed_rcs_dbsm))]
        cmd += ["--eirp-dbm", str(float(eirp_dbm))]
        if codebook_mode == "grid":
            cmd += ["--codebook-id", "grid"]
        if no_noise:
            cmd += ["--no-noise"]

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(int(omp_threads))
        env["MKL_NUM_THREADS"] = str(int(omp_threads))
        env["OPENBLAS_NUM_THREADS"] = str(int(omp_threads))

        log_file = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=Path.cwd(),
            start_new_session=True,
        )
        pgid = None
        try:
            pgid = os.getpgid(proc.pid)
        except OSError:
            pgid = None

        meta = {
            "run_id": run_id,
            "cmd": cmd,
            "pid": proc.pid,
            "pgid": pgid,
            "started_at": time.time(),
            "scenarios": scenarios,
            "altitudes": altitudes,
            "drops": int(drops),
        }
        _write_meta(run_path, meta)
        st.success(f"Started run {run_id} (pid {proc.pid})")

    st.header("Runs")
    runs = _load_runs()
    if not runs:
        st.info("No runs yet. Start one from the sidebar.")
        return

    run_labels = [
        f"{r.run_id} ({'running' if _is_running(r.pid, r.pgid) else 'done'})" for r in runs
    ]
    selected = st.selectbox("Select run", run_labels, index=0)
    sel_idx = run_labels.index(selected)
    run = runs[sel_idx]

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Status")
        running = _is_running(run.pid, run.pgid)
        st.write("PID:", run.pid)
        st.write("PGID:", run.pgid)
        st.write("Running:", running)
        if running and st.button("Stop run", type="secondary"):
            _stop_group(run.pgid, run.pid)
            st.warning("Stop signal sent")

        lines = _read_tail(run.log_path)
        progress, last_line = _parse_progress(lines)
        scenarios = run.meta.get("scenarios", [])
        altitudes = run.meta.get("altitudes", [])
        overall = _overall_progress(progress, scenarios, altitudes)
        st.metric("Overall progress", f"{overall:.1%}")
        if last_line:
            st.caption(f"Last: {last_line}")

    with col_b:
        st.subheader("Log tail")
        st.text("\n".join(lines[-30:]))

    if run.out_path.exists() and run.out_path.stat().st_size:
        st.subheader("Results")
        df = pd.read_csv(run.out_path)
        _plot_results(df)


def main() -> None:
    st.set_page_config(page_title="5GNRad Dashboard", layout="wide")
    st.title("5GNRad Dashboard")

    mode = st.sidebar.radio("Mode", ["Learn the paper", "Run experiments"], index=0)
    if mode == "Learn the paper":
        _render_learn()
    else:
        _render_runs()


if __name__ == "__main__":
    main()

