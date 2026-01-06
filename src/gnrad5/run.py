from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from gnrad5.antenna import URA, antenna_sorted_index, array_angle_grid
from gnrad5.channel import get_sensing_cdl, load_background_channel, load_target_channel
from gnrad5.config import load_scenario
from gnrad5.constants import SPEED_OF_LIGHT
from gnrad5.detection import detect_cfar_4d, detect_cfar_rd
from gnrad5.nr import Py3gppUnavailable
from gnrad5.signal import build_prs_grid, get_ofdm_params, range_doppler_map
from gnrad5.utils.memory_guard import guard_memory
from gnrad5.visualize import plot_cfar_mask, plot_detections, plot_grid, plot_rd_map, save_fig


def run_slice(
    scenario: str,
    repo_root: str | Path | None = None,
    seed: int | None = None,
    max_symbols: int | None = None,
    max_range_bins: int | None = None,
    doppler_fft_len: int | None = None,
    az_fft_len: int | None = None,
    el_fft_len: int | None = None,
    no_channel: bool = False,
    rda_chunk: int | None = None,
    prs_export: str | None = None,
    prs_compare: str | None = None,
    cfar_threshold: float | None = None,
    rda_threshold: float | None = None,
    target_power_boost: float = 0.0,
    no_background: bool = False,
    single_path: bool = False,
    log_stage_max: bool = False,
    skip_range_mean: bool = False,
    min_clutter_ratio: float | None = None,
    matlab_range_mean: bool = True,
    target_positions: np.ndarray | None = None,
    target_velocities: np.ndarray | None = None,
    background_channel: dict | None = None,
    target_channel: list[dict] | dict | None = None,
    add_noise: bool = True,
    rng: np.random.Generator | None = None,
):
    sim, target, prs, geometry, sens = load_scenario(scenario)
    repo_root = Path(repo_root or Path.cwd())
    if background_channel is None:
        background = load_background_channel(repo_root, scenario, sim.system_fc)
    else:
        background = background_channel
    if isinstance(background, list):
        background = background[0]
    if no_background:
        background = {"AveragePathGains": [], "PathDelays": [], "AnglesAoA": [], "AnglesZoA": [], "AnglesAoD": [], "AnglesZoD": [], "InitialPhases": []}
    if target_channel is None:
        target_channel = load_target_channel(repo_root, scenario, sim.system_fc)
    if target_channel is None:
        target_chan_entry = None
    elif isinstance(target_channel, list):
        target_chan_entry = target_channel
    else:
        target_chan_entry = target_channel

    if target_positions is not None:
        target.position = target_positions
    if target_velocities is not None:
        target.velocity = target_velocities

    prs_grid = build_prs_grid(sim, prs, sens.number_sensing_symbols)
    ofdm_params = get_ofdm_params(prs_grid.nr_objects.ofdm_info, prs_grid.nr_objects.carrier)
    nfft = ofdm_params.nfft
    cp_lengths = ofdm_params.cp_lengths
    symbol_indices = prs_grid.symbol_indices
    if max_symbols is not None:
        symbol_indices = symbol_indices[: max_symbols]

    if prs_export:
        export_prs(prs_grid, prs_export)
    if prs_compare:
        info_prs = compare_prs(prs_grid, prs_compare)
    else:
        info_prs = {}
    tx_symbol_lengths = [nfft + cp_lengths[int(sym_idx % len(cp_lengths))] for sym_idx in symbol_indices]

    tx_symbols = []
    for sym_idx in symbol_indices:
        freq_symbol = prs_grid.grid[:, sym_idx] * np.sqrt(max(1, int(prs.comb_size)))
        time_no_cp = np.fft.ifft(freq_symbol, nfft) * np.sqrt(nfft)
        cp_len = cp_lengths[int(sym_idx % len(cp_lengths))]
        cp = time_no_cp[-cp_len:]
        tx_symbols.append(np.concatenate([cp, time_no_cp]))
    tx_waveform = np.concatenate(tx_symbols) if tx_symbols else np.zeros(0, dtype=np.complex128)
    n_ant = sim.antenna_num_v * sim.antenna_num_h
    tx_array = URA((sim.antenna_num_v, sim.antenna_num_h), (SPEED_OF_LIGHT / sim.system_fc) / 2)
    rx_array = URA((sim.antenna_num_v, sim.antenna_num_h), (SPEED_OF_LIGHT / sim.system_fc) / 2)

    n_realization = int(symbol_indices.max()) + 1 if symbol_indices.size else 1
    ofdm_info = prs_grid.nr_objects.ofdm_info
    ofdm_symbol_lengths = getattr(ofdm_info, "SymbolLengths", None) if not isinstance(ofdm_info, dict) else None
    sample_rate = ofdm_params.sample_rate
    symbols_per_slot = int(getattr(prs_grid.nr_objects.carrier, "SymbolsPerSlot", 14))
    if ofdm_symbol_lengths is not None and sample_rate:
        ofdm_ts = float(np.mean(np.asarray(ofdm_symbol_lengths)) / sample_rate)
    else:
        ofdm_ts = 1.0 / sens.number_sensing_symbols

    c = SPEED_OF_LIGHT
    number_subcarriers = sim.carrier_n_size_grid * 12
    range_resolution = 1.0
    if sample_rate:
        range_resolution = 1 / (2 * sample_rate) * c
    comb = max(1, int(prs.comb_size))
    range_bins = np.arange(nfft * comb) * range_resolution / comb
    range_limit = len(range_bins)
    if sim.max_range_interest:
        idx = np.argmax(range_bins > sim.max_range_interest)
        if idx > 0:
            range_limit = idx

    scs_khz = sim.carrier_subcarrier_spacing
    mu = int(round(np.log2(scs_khz / 15))) if scs_khz else 0
    slots_per_subframe = 2**mu
    if ofdm_symbol_lengths is not None and sample_rate:
        slot_duration = float(np.sum(np.asarray(ofdm_symbol_lengths)) / sample_rate)
    else:
        slot_duration = float(ofdm_ts * symbols_per_slot)
    prs_periodicity = slot_duration / slots_per_subframe * int(prs.prs_resource_set_period[0])
    vosf = sens.doppler_fft_len / sens.number_sensing_symbols
    velocity_resolution = c / (2 * sens.number_sensing_symbols * prs_periodicity * sim.system_fc) / vosf
    velocity_bins = (np.arange(-sens.doppler_fft_len / 2, sens.doppler_fft_len / 2) * velocity_resolution)

    scanvector = _build_scanvector(sim.antenna_num_h, sim.antenna_num_v)
    if rng is None:
        rng = np.random.default_rng(seed)

    tgt_pos = np.asarray(target.position)
    tgt_vel = np.asarray(target.velocity)
    if tgt_pos.ndim == 1:
        tgt_pos = tgt_pos.reshape(1, -1)
    if tgt_vel.ndim == 1:
        tgt_vel = tgt_vel.reshape(1, -1)

    fast_no_channel = bool(no_channel and not add_noise)
    if no_channel:
        h_full = np.zeros((1, n_realization, n_ant), dtype=np.complex128)
        h_full[0, :, :] = 1.0 + 0j
        sensing_out = type("SensingOut", (), {"sync_offset": 0.0})()
        cir_ant = h_full
    else:
        if target_chan_entry is None and tgt_pos.shape[0] > 1:
            tgt_pos_use = tgt_pos[0]
            tgt_vel_use = tgt_vel[0]
        else:
            tgt_pos_use = tgt_pos
            tgt_vel_use = tgt_vel
        sensing_out, cir_ant, _ = get_sensing_cdl(
            np.array(geometry.tx[0]),
            tgt_pos_use,
            tgt_vel_use,
            sim.system_fc,
            ofdm_ts,
            background,
            target_channel=target_chan_entry,
            n_realization=n_realization,
            sample_rate=sample_rate,
            transmit_array=tx_array,
            receive_array=rx_array,
            scanvector=scanvector,
            angle_estimation="ideal",
            scenario=sim.channel_scenario,
            rng=rng,
        )
    rx_grids = None
    if fast_no_channel:
        rx_grids = [prs_grid.grid]
    if not fast_no_channel:
        h_full = cir_ant
        if single_path:
            h_full = np.zeros_like(h_full)
            h_full[0, :, :] = 1.0 + 0j
        if target_power_boost != 0.0:
            h_full = h_full * (10 ** (target_power_boost / 20))
        waveform_len = len(tx_waveform) + h_full.shape[0] - 1
        guard_memory("rx_waveform", extra_bytes=waveform_len * n_ant * np.dtype(np.complex128).itemsize)
        rx_waveform = np.zeros((waveform_len, n_ant), dtype=np.complex128)
        pointer = 0
        for i, sym_len in enumerate(tx_symbol_lengths):
            this_symbol = tx_waveform[pointer : pointer + sym_len]
            for ant in range(n_ant):
                h_sym = h_full[:, i, ant]
                rx_sym = np.convolve(this_symbol, h_sym)
                rx_waveform[pointer : pointer + len(rx_sym), ant] += rx_sym
            pointer += sym_len

        if add_noise:
            k = 1.380649e-23
            t_k = 297.0
            nf = 10 ** (sim.system_nf / 10)
            noise_power = k * t_k * sim.system_bw * nf
            max_power = 52 - 30 + 8
            p_lin = 10 ** (max_power / 10)
            snr_db = 10 * np.log10(p_lin / noise_power)
            snr_var = 10 ** (snr_db / 10)
            noise = (rng.standard_normal(rx_waveform.shape) + 1j * rng.standard_normal(rx_waveform.shape))
            rx_waveform = rx_waveform + np.sqrt(1 / (2 * snr_var)) * noise

        rx_grids = []
        grid_bytes = prs_grid.grid.size * prs_grid.grid.itemsize
        guard_memory("rx_grids", extra_bytes=grid_bytes * n_ant)
        for ant in range(n_ant):
            pointer = 0
            rx_grid = np.zeros_like(prs_grid.grid)
            for i, sym_idx in enumerate(symbol_indices):
                sym_len = tx_symbol_lengths[i]
                cp_len = cp_lengths[int(sym_idx % len(cp_lengths))]
                this_symbol = rx_waveform[pointer : pointer + sym_len, ant]
                pointer += sym_len
                no_cp = this_symbol[cp_len:]
                freq_symbol = np.fft.fft(no_cp, nfft) / np.sqrt(nfft)
                rx_grid[:, sym_idx] = freq_symbol
            rx_grids.append(rx_grid)
    if doppler_fft_len is not None:
        sens.doppler_fft_len = doppler_fft_len
    if az_fft_len is not None:
        sens.az_fft_len = az_fft_len
    if el_fft_len is not None:
        sens.el_fft_len = el_fft_len
    if cfar_threshold is not None:
        sens.cfar_threshold = cfar_threshold
    if rda_threshold is not None:
        sens.rda_threshold = rda_threshold
    if matlab_range_mean:
        effective_skip_mean = bool(skip_range_mean)
        effective_clutter_ratio = None
    else:
        effective_skip_mean = bool(skip_range_mean or no_background or single_path)
        effective_clutter_ratio = min_clutter_ratio
    if symbol_indices.size:
        sym0 = int(symbol_indices[0])
        sym_window = symbol_indices - sym0
        num_symbols_total = int(sym_window[-1]) + 1
    else:
        sym0 = 0
        sym_window = symbol_indices
        num_symbols_total = 0
    tx_grid_window = prs_grid.grid[:, sym0 : sym0 + num_symbols_total]

    rd_maps = []
    rd_bytes = int(range_limit * sens.doppler_fft_len) * np.dtype(np.complex128).itemsize
    guard_memory("rd_maps", extra_bytes=rd_bytes * n_ant)
    prs_stats = {}
    if fast_no_channel:
        rd_map = range_doppler_map(
            tx_grid_window,
            tx_grid_window,
            sym_window,
            sim,
            prs,
            sens.doppler_fft_len,
            window="hamming",
            symbols_per_slot=symbols_per_slot,
            log_stats=prs_stats,
            skip_mean=effective_skip_mean,
            min_clutter_ratio=effective_clutter_ratio,
        )
        if max_range_bins is not None:
            rd_map = rd_map[:max_range_bins]
        else:
            rd_map = rd_map[:range_limit]
        rd_stack = np.repeat(rd_map[:, :, None], n_ant, axis=2)
    else:
        for ant in range(n_ant):
            rd_map = range_doppler_map(
                tx_grid_window,
                rx_grids[ant][:, sym0 : sym0 + num_symbols_total],
                sym_window,
                sim,
                prs,
                sens.doppler_fft_len,
                window="hamming",
                symbols_per_slot=symbols_per_slot,
                log_stats=prs_stats if ant == 0 else None,
                skip_mean=effective_skip_mean,
                min_clutter_ratio=effective_clutter_ratio,
            )
            if max_range_bins is not None:
                rd_map = rd_map[:max_range_bins]
            else:
                rd_map = rd_map[:range_limit]
            rd_maps.append(rd_map)
        rd_stack = np.stack(rd_maps, axis=2)
    idx_vec = antenna_sorted_index(rx_array)
    rd_stack = rd_stack[:, :, idx_vec]
    rd_cube = rd_stack.reshape(rd_stack.shape[0], rd_stack.shape[1], sim.antenna_num_v, sim.antenna_num_h)

    rda_bytes = (
        rd_cube.shape[0]
        * rd_cube.shape[1]
        * sens.el_fft_len
        * sens.az_fft_len
        * np.dtype(np.complex128).itemsize
    )
    guard_memory("rda", extra_bytes=rda_bytes)

    if rda_chunk is None:
        rda = np.fft.fftshift(
            np.fft.fftshift(
                np.fft.fft(np.fft.fft(rd_cube, sens.el_fft_len, axis=2), sens.az_fft_len, axis=3),
                axes=2,
            ),
            axes=3,
        )
    else:
        rda = np.zeros((rd_cube.shape[0], rd_cube.shape[1], sens.el_fft_len, sens.az_fft_len), dtype=np.complex128)
        for start in range(0, rd_cube.shape[0], rda_chunk):
            stop = min(rd_cube.shape[0], start + rda_chunk)
            chunk = rd_cube[start:stop]
            chunk_fft = np.fft.fftshift(
                np.fft.fftshift(
                    np.fft.fft(np.fft.fft(chunk, sens.el_fft_len, axis=2), sens.az_fft_len, axis=3),
                    axes=2,
                ),
                axes=3,
            )
            rda[start:stop] = chunk_fft

    detections_4d = detect_cfar_4d(rda, sens)
    rd_map = rd_stack[:, :, 0]
    detections, _, cfar_mask = detect_cfar_rd(rd_map, sens)

    diff = rx_grids[0] - prs_grid.grid
    col_mask = np.zeros(prs_grid.grid.shape[1], dtype=bool)
    col_mask[symbol_indices] = True
    nonzero = (np.abs(prs_grid.grid) > 0) & col_mask[None, :]
    if np.any(nonzero):
        evm = np.sqrt(np.mean(np.abs(diff[nonzero]) ** 2)) / (np.sqrt(np.mean(np.abs(prs_grid.grid[nonzero]) ** 2)) + 1e-12)
        max_err = float(np.max(np.abs(diff[nonzero])))
    else:
        evm = 0.0
        max_err = 0.0

    stage_max = {}
    if log_stage_max:
        stage_max = {
            "rd_max": float(np.max(np.abs(rd_stack)) if rd_stack.size else 0.0),
            "rda_max": float(np.max(np.abs(rda)) if rda.size else 0.0),
            "rx_grid_max": float(np.max(np.abs(rx_grids[0])) if rx_grids else 0.0),
            "g_tilde_max": float(np.max(np.abs(rx_grids[0] * np.conj(prs_grid.grid))) if rx_grids else 0.0),
        }
        stage_max.update(prs_stats)

    info = {
        "evm": float(evm),
        "rd_shape": rd_map.shape,
        "rda_shape": rda.shape,
        "detections": int(detections.shape[0]),
        "detections_4d": int(detections_4d.shape[0]),
        "sync_offset": float(sensing_out.sync_offset),
        "background_loaded": background is not None,
        "target_channel_loaded": target_channel is not None,
        "num_targets": len(target.position),
        "num_bs": len(geometry.tx),
        "max_prs_err": max_err,
    }
    info.update(stage_max)
    info.update(info_prs)
    az_grid, el_grid, _ = array_angle_grid(
        sens.el_fft_len,
        sens.az_fft_len,
        rx_array.element_spacing,
        rx_array.element_spacing,
        c / sim.system_fc,
    )

    debug = {
        "prs_grid": prs_grid.grid,
        "rd_map": rd_map,
        "rda": rda,
        "detections_4d": detections_4d,
        "cfar_mask": cfar_mask,
        "sens": sens,
        "sensing": sensing_out,
        "range_bins": range_bins[: rd_map.shape[0]],
        "velocity_bins": velocity_bins,
        "az_grid": az_grid,
        "el_grid": el_grid,
    }
    return info, debug


def export_prs(prs_grid, prefix: str):
    idx = np.where(np.abs(prs_grid.grid) > 0)
    lin = (idx[0] + 1) + idx[1] * prs_grid.grid.shape[0]
    np.savetxt(f"{prefix}_indices.csv", lin, fmt="%d", delimiter=",")
    np.savetxt(f"{prefix}_symbols.csv", prs_grid.grid[idx], fmt="%.18e", delimiter=",")


def compare_prs(prs_grid, prefix: str):
    try:
        matlab_idx = np.loadtxt(f"{prefix}_indices.csv", delimiter=",").astype(int)
        matlab_sym = np.loadtxt(f"{prefix}_symbols.csv", delimiter=",")
    except Exception:
        return {"prs_compare": "missing"}

    idx = np.where(np.abs(prs_grid.grid) > 0)
    lin = (idx[0] + 1) + idx[1] * prs_grid.grid.shape[0]
    sym = prs_grid.grid[idx]

    idx_ok = np.array_equal(lin, matlab_idx)
    if matlab_sym.ndim == 1:
        matlab_sym = matlab_sym.astype(np.complex128)
    sym_err = float(np.max(np.abs(sym - matlab_sym))) if sym.size and matlab_sym.size else 0.0
    return {"prs_idx_match": idx_ok, "prs_sym_err": sym_err}


def _build_scanvector(n_ant_h: int, n_ant_v: int):
    hpbw_h = 0.886 * 2 / n_ant_h * 180 / np.pi
    hpbw_v = 0.886 * 2 / n_ant_v * 180 / np.pi
    scan_step_h = max(1, int(np.floor(hpbw_h)))
    scan_step_v = max(1, int(np.floor(hpbw_v)))
    azimuth_range = np.arange(-180, 181, scan_step_h)
    elevation_range = np.arange(-90, 91, scan_step_v)
    az, el = np.meshgrid(azimuth_range, elevation_range)
    return np.column_stack([az.ravel(), el.ravel()])


def main():
    parser = argparse.ArgumentParser(description="5GNRad Python port: OFDM + RD slice")
    parser.add_argument("--scenario", required=True, help="Scenario path (e.g., examples3GPP/UMa-Av200-8x8-30)")
    parser.add_argument("--repo-root", default=None, help="Repo root (defaults to CWD)")
    parser.add_argument("--plot", action="store_true", help="Save visualization plots")
    parser.add_argument("--plot-dir", default="plots", help="Directory to save plots")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    parser.add_argument("--max-symbols", type=int, default=None, help="Limit PRS symbols for quick validation")
    parser.add_argument("--max-range-bins", type=int, default=None, help="Limit range bins for quick validation")
    parser.add_argument("--doppler-fft-len", type=int, default=None, help="Override doppler FFT length")
    parser.add_argument("--az-fft-len", type=int, default=None, help="Override azimuth FFT length")
    parser.add_argument("--el-fft-len", type=int, default=None, help="Override elevation FFT length")
    parser.add_argument("--no-channel", action="store_true", help="Bypass channel to validate PRS parity")
    parser.add_argument("--rda-chunk", type=int, default=None, help="Process RDA in range chunks")
    parser.add_argument("--prs-export", default=None, help="Path prefix to export PRS indices/symbols CSV")
    parser.add_argument("--prs-compare", default=None, help="Path prefix to compare against MATLAB PRS CSV")
    parser.add_argument("--cfar-threshold", type=float, default=None, help="Override CFAR threshold")
    parser.add_argument("--rda-threshold", type=float, default=None, help="Override RDA pre-threshold (dB)")
    parser.add_argument("--target-power-boost", type=float, default=0.0, help="Temporary boost (dB) to target path gains")
    parser.add_argument("--no-background", action="store_true", help="Disable background channel paths")
    parser.add_argument("--single-path", action="store_true", help="Force single-path CIR for debugging")
    parser.add_argument("--log-stage-max", action="store_true", help="Log max power per stage")
    parser.add_argument("--skip-range-mean", action="store_true", help="Skip range FFT mean subtraction for debugging")
    parser.add_argument("--min-clutter-ratio", type=float, default=1e-2, help="When not using MATLAB range-mean, skip subtraction if clutter ratio falls below this threshold")
    parser.add_argument("--no-matlab-range-mean", action="store_true", help="Disable strict MATLAB range-mean subtraction to allow conditional skipping")
    args = parser.parse_args()

    try:
        info, debug = run_slice(
            args.scenario,
            args.repo_root,
            args.seed,
            args.max_symbols,
            args.max_range_bins,
            args.doppler_fft_len,
            args.az_fft_len,
            args.el_fft_len,
            args.no_channel,
            args.rda_chunk,
            args.prs_export,
            args.prs_compare,
            args.cfar_threshold,
            args.rda_threshold,
            args.target_power_boost,
            args.no_background,
            args.single_path,
            args.log_stage_max,
            args.skip_range_mean,
            args.min_clutter_ratio,
            not args.no_matlab_range_mean,
        )
    except Py3gppUnavailable as exc:
        raise SystemExit(str(exc)) from exc

    print("OFDM + RD slice OK")
    for key, value in info.items():
        print(f"{key}: {value}")

    if args.plot:
        plot_dir = Path(args.plot_dir)
        grid_fig = plot_grid(debug["prs_grid"])
        save_fig(grid_fig, plot_dir / "grid.png")
        rd_fig = plot_rd_map(debug["rd_map"])
        save_fig(rd_fig, plot_dir / "rd_map.png")
        rda_fig = plot_rd_map(debug["rda"], title="RDA Sum")
        save_fig(rda_fig, plot_dir / "rda_sum.png")
        det_fig = plot_detections(debug["rd_map"], debug["detections_4d"])
        save_fig(det_fig, plot_dir / "detections.png")
        cfar_mask_fig = plot_cfar_mask(debug["cfar_mask"], title="CFAR Mask (RD)")
        save_fig(cfar_mask_fig, plot_dir / "cfar_mask.png")


if __name__ == "__main__":
    main()
