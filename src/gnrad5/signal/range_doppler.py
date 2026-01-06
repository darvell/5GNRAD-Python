from __future__ import annotations

import numpy as np

from gnrad5.config.models import PrsConfig, SimulationConfig
from gnrad5.signal.prs_destagger import prs_destagger
from gnrad5.signal.windows import get_dft_window


def range_doppler_map(
    tx_grid: np.ndarray,
    rx_grid: np.ndarray,
    symbol_indices: np.ndarray,
    sim: SimulationConfig,
    prs_cfg: PrsConfig,
    doppler_fft_len: int,
    window: str = "hamming",
    symbols_per_slot: int = 14,
    log_stats: dict | None = None,
    skip_mean: bool = False,
    min_clutter_ratio: float | None = None,
):
    nfft = tx_grid.shape[0]
    number_subcarriers = sim.carrier_n_size_grid * 12
    start_idx = (nfft - number_subcarriers) // 2
    end_idx = start_idx + number_subcarriers

    g_tilde = rx_grid * np.conj(tx_grid)
    subcarriers = g_tilde[start_idx:end_idx, symbol_indices]

    if window == "rect":
        win = get_dft_window("rect", number_subcarriers)
    elif window == "hamming":
        win = get_dft_window("hamming", number_subcarriers)
    elif window == "blackmanharris":
        win = get_dft_window("blackmanharris", number_subcarriers)
    elif window == "gaussian":
        win = get_dft_window("gaussian", number_subcarriers)
    elif window == "kaiser":
        win = get_dft_window("kaiser", number_subcarriers)
    else:
        win = get_dft_window("hamming", number_subcarriers)
    win = win[:, None]

    comb = int(prs_cfg.comb_size)
    if comb <= 0:
        comb = 1
    num_prs_sub = number_subcarriers // comb
    if subcarriers.shape[0] < num_prs_sub:
        num_prs_sub = subcarriers.shape[0]

    symbol_in_slot = np.mod(symbol_indices, symbols_per_slot)
    prs_slot = np.zeros((num_prs_sub, subcarriers.shape[1]), dtype=subcarriers.dtype)
    for col, sym in enumerate(symbol_in_slot):
        re_offset = _freq_offset(comb, prs_cfg.symbol_start, int(sym))
        k_init = (prs_cfg.re_offset + re_offset) % comb
        prs_slot[:, col] = subcarriers[k_init::comb, col][:num_prs_sub]

    destaggered = prs_destagger(prs_slot, prs_cfg, symbol_in_slot)
    destaggered = destaggered.reshape(num_prs_sub, -1)
    if log_stats is not None:
        log_stats["prs_slot_max"] = float(np.max(np.abs(prs_slot)) if prs_slot.size else 0.0)
        log_stats["destaggered_max"] = float(np.max(np.abs(destaggered)) if destaggered.size else 0.0)

    if log_stats is not None:
        log_stats["pre_ifft_max"] = float(np.max(np.abs(destaggered)) if destaggered.size else 0.0)
    range_fft = (
        np.fft.ifft(destaggered * win[: destaggered.shape[0]], n=nfft * comb, axis=0)
        * np.sqrt(nfft * comb)
    )

    if log_stats is not None:
        log_stats["post_ifft_max"] = float(np.max(np.abs(range_fft)) if range_fft.size else 0.0)

    did_mean = False
    skip_reason = None
    if skip_mean:
        skip_reason = "skip_mean_flag"
    elif min_clutter_ratio is not None:
        max_mag = float(np.max(np.abs(range_fft))) if range_fft.size else 0.0
        if max_mag > 0.0:
            clutter_mag = float(np.mean(np.abs(range_fft.mean(axis=1))))
            clutter_ratio = clutter_mag / max_mag
            if clutter_ratio < float(min_clutter_ratio):
                skip_reason = "low_clutter"
            if log_stats is not None:
                log_stats["clutter_ratio"] = float(clutter_ratio)
        else:
            skip_reason = "zero_energy"
    if skip_reason is None:
        range_fft = range_fft - range_fft.mean(axis=1, keepdims=True)
        did_mean = True
    if log_stats is not None:
        log_stats["did_mean_sub"] = bool(did_mean)
        log_stats["mean_skip_reason"] = skip_reason or "none"
    if log_stats is not None:
        log_stats["post_mean_max"] = float(np.max(np.abs(range_fft)) if range_fft.size else 0.0)
    rd = (
        np.fft.fftshift(
            np.fft.fft(range_fft, n=doppler_fft_len, axis=1) / np.sqrt(doppler_fft_len),
            axes=1,
        )
    )
    return rd


def _freq_offset(comb_size: int, start_symbol: int, symbol: int) -> int:
    table = {
        2: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        4: [0, 2, 1, 3, 0, 2, 1, 3, 0, 2, 1, 3],
        6: [0, 3, 1, 4, 2, 5, 0, 3, 1, 4, 2, 5],
        12: [0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11],
    }
    row = table.get(comb_size, table[12])
    if symbol < start_symbol:
        raise ValueError("Invalid symbol index")
    return row[symbol - start_symbol]
