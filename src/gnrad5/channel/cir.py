from __future__ import annotations

import numpy as np


class ChannelCIRError(RuntimeError):
    pass


def build_cir_from_background(background: dict, sample_rate: float) -> tuple[np.ndarray, int]:
    if background is None:
        return np.array([1.0 + 0.0j]), 0
    if sample_rate is None or sample_rate <= 0:
        raise ChannelCIRError("Sample rate required to build CIR")

    delays = np.asarray(background.get("PathDelays", []), dtype=float).reshape(-1)
    gains = np.asarray(background.get("AveragePathGains", []), dtype=float).reshape(-1)
    phases = np.asarray(background.get("InitialPhases", []), dtype=float).reshape(-1)

    if delays.size == 0:
        return np.array([1.0 + 0.0j]), 0
    if gains.size == 0:
        gains = np.ones_like(delays)
    if phases.size == 0:
        phases = np.zeros_like(delays)

    if gains.size != delays.size:
        raise ChannelCIRError("AveragePathGains length mismatch")
    if phases.size != delays.size:
        phases = np.resize(phases, delays.shape)

    sample_delays = np.round(delays * sample_rate).astype(int)
    sample_delays = np.maximum(sample_delays, 0)
    sync_offset = int(sample_delays.min())
    sample_delays = sample_delays - sync_offset

    length = int(sample_delays.max()) + 1
    cir = np.zeros(length, dtype=np.complex128)
    taps = np.sqrt(np.abs(gains)) * np.exp(1j * phases)
    for tap, idx in zip(taps, sample_delays):
        cir[idx] += tap
    if np.allclose(cir, 0):
        cir[0] = 1.0 + 0.0j
    return cir, sync_offset


def apply_cir(waveform: np.ndarray, cir: np.ndarray) -> np.ndarray:
    return np.convolve(waveform, cir, mode="full")
