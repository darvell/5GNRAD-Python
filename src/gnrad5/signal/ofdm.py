from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


class OfdmInfoError(RuntimeError):
    pass


@dataclass
class OfdmParams:
    nfft: int
    cp_lengths: list[int]
    sample_rate: float | None


def _as_int_list(value: Iterable[int] | np.ndarray) -> list[int]:
    arr = np.asarray(value).astype(int).tolist()
    if isinstance(arr, int):
        return [int(arr)]
    return [int(x) for x in arr]


def get_ofdm_params(ofdm_info, carrier=None) -> OfdmParams:
    if isinstance(ofdm_info, dict):
        nfft = ofdm_info.get("Nfft")
        cp_lengths = ofdm_info.get("CyclicPrefixLengths")
        sample_rate = ofdm_info.get("SampleRate")
    else:
        nfft = getattr(ofdm_info, "Nfft", None)
        cp_lengths = getattr(ofdm_info, "CyclicPrefixLengths", None)
        sample_rate = getattr(ofdm_info, "SampleRate", None)

    if nfft is None:
        raise OfdmInfoError("OFDM info missing Nfft")

    if cp_lengths is None:
        if carrier is None:
            raise OfdmInfoError("OFDM info missing CyclicPrefixLengths")
        scs_khz = getattr(carrier, "SubcarrierSpacing", None)
        if scs_khz is None:
            raise OfdmInfoError("Missing SubcarrierSpacing for CP computation")
        mu = int(round(np.log2(scs_khz / 15)))
        symbols_per_slot = int(getattr(carrier, "SymbolsPerSlot", 14))
        cp_lengths = []
        for sym in range(symbols_per_slot):
            base = 144 >> mu
            extra_symbol = 7 * (2**mu)
            extra = 16 if sym == 0 or sym == extra_symbol else 0
            cp_kappa = base + extra
            cp_samples = int(round(cp_kappa * (nfft * (2**mu) / 2048)))
            cp_lengths.append(cp_samples)
    if sample_rate is None:
        if carrier is None:
            sample_rate = None
        else:
            scs_khz = getattr(carrier, "SubcarrierSpacing", None)
            if scs_khz is None:
                sample_rate = None
            else:
                sample_rate = float(nfft * scs_khz * 1000)

    return OfdmParams(nfft=int(nfft), cp_lengths=_as_int_list(cp_lengths), sample_rate=sample_rate)


def ofdm_modulate(grid: np.ndarray, symbol_indices: np.ndarray, ofdm_info) -> np.ndarray:
    params = get_ofdm_params(ofdm_info)
    nfft = params.nfft
    cp_lengths = params.cp_lengths

    symbol_indices = np.asarray(symbol_indices).astype(int)
    num_symbols = len(symbol_indices)
    base_cp = cp_lengths[1] if len(cp_lengths) > 1 else cp_lengths[0]
    stride = nfft + base_cp
    tx = np.zeros(stride * num_symbols, dtype=np.complex128)

    for i, sym_idx in enumerate(symbol_indices):
        freq_symbol = grid[:, sym_idx]
        time_no_cp = np.fft.ifft(freq_symbol, nfft) * np.sqrt(nfft)
        cp_len = cp_lengths[sym_idx % len(cp_lengths)]
        cp = time_no_cp[-cp_len:]
        time_with_cp = np.concatenate([cp, time_no_cp])
        start = i * stride
        tx[start : start + nfft + cp_len] = time_with_cp
    return tx


def ofdm_demodulate(
    rx: np.ndarray, symbol_indices: np.ndarray, ofdm_info
) -> np.ndarray:
    params = get_ofdm_params(ofdm_info)
    nfft = params.nfft
    cp_lengths = params.cp_lengths
    symbol_indices = np.asarray(symbol_indices).astype(int)

    total_symbols = int(symbol_indices.max()) + 1
    grid = np.zeros((nfft, total_symbols), dtype=np.complex128)

    pointer = 0
    base_cp = cp_lengths[1] if len(cp_lengths) > 1 else cp_lengths[0]
    stride = nfft + base_cp
    for sym_idx in symbol_indices:
        cp_len = cp_lengths[sym_idx % len(cp_lengths)]
        symbol_len = nfft + cp_len
        if pointer + symbol_len > len(rx):
            raise OfdmInfoError("RX waveform too short for symbol demodulation")
        this_symbol = rx[pointer : pointer + symbol_len]
        pointer += stride
        no_cp = this_symbol[cp_len:]
        freq_symbol = np.fft.fft(no_cp, nfft) / np.sqrt(nfft)
        grid[:, sym_idx] = freq_symbol
    return grid
