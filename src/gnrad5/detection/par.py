from __future__ import annotations

import numpy as np

from gnrad5.signal.prs_destagger import prs_destagger
from gnrad5.signal.windows import get_dft_window


def par_detector(
    tx_grid: np.ndarray,
    rx_grid: np.ndarray,
    symbol_indices: np.ndarray,
    sim,
    prs_cfg,
    eta_db: float,
    window: str = "blackmanharris",
    symbols_per_slot: int = 14,
    clutter_mean: bool = True,
    max_range_bins: int | None = None,
):
    range_profile = _range_profile(
        tx_grid,
        rx_grid,
        symbol_indices,
        sim,
        prs_cfg,
        window=window,
        symbols_per_slot=symbols_per_slot,
        clutter_mean=clutter_mean,
        max_range_bins=max_range_bins,
    )
    if range_profile.size == 0:
        return {
            "par_db": -np.inf,
            "range_idx": -1,
            "range_power": range_profile,
            "detected": False,
        }
    mean_mag = float(np.mean(range_profile)) + 1e-12
    peak_mag = float(np.max(range_profile))
    par = peak_mag / mean_mag
    # Paper defines PAR on |r(d)|, but reports the threshold in dB.
    # We use 10*log10 to keep thresholds consistent with the rest of the chain.
    par_db = 10 * np.log10(par + 1e-12)
    range_idx = int(np.argmax(range_profile))
    return {
        "par_db": par_db,
        "range_idx": range_idx,
        "range_power": range_profile,
        "detected": par_db >= float(eta_db),
    }


def _range_profile(
    tx_grid: np.ndarray,
    rx_grid: np.ndarray,
    symbol_indices: np.ndarray,
    sim,
    prs_cfg,
    window: str = "blackmanharris",
    symbols_per_slot: int = 14,
    clutter_mean: bool = True,
    max_range_bins: int | None = None,
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
    if clutter_mean and destaggered.size:
        destaggered = destaggered - destaggered.mean(axis=1, keepdims=True)

    range_fft = (
        np.fft.ifft(destaggered * win[: destaggered.shape[0]], n=nfft, axis=0)
        * np.sqrt(nfft)
    )
    range_mag = np.mean(np.abs(range_fft), axis=1) if range_fft.size else np.zeros(0)
    if max_range_bins is not None:
        range_mag = range_mag[: max_range_bins]
    return range_mag


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
