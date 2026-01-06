from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence


@dataclass
class SimulationConfig:
    system_fc: float = 30e9
    system_nf: float = 7.0
    system_bw: float = 100e6
    channel_scenario: str = "UMiAV"
    antenna_num_h: int = 32
    antenna_num_v: int = 32
    antenna_coupling_efficiency: float = 0.7
    carrier_subcarrier_spacing: int = 120
    carrier_n_size_grid: int = 66
    n_st_drop: int = 1
    max_range_interest: float = 400.0


@dataclass
class PrsConfig:
    prs_resource_set_period: Sequence[int] = field(default_factory=lambda: [1, 0])
    prs_resource_offset: int = 0
    prs_resource_repetition: int = 1
    prs_resource_time_gap: int = 1
    num_rb: int = 66
    rb_offset: int = 0
    comb_size: int = 2
    re_offset: int = 0
    n_prs_id: int = 0
    num_prs_symbols: int = 2
    symbol_start: int = 0


@dataclass
class SensConfig:
    doppler_fft_len: int = 64
    window: str = "blackmanharris"
    window_len: int | None = None
    window_overlap: float = 0.5
    number_sensing_symbols: int = 256
    cfar_grd_cell_range: int = 0
    cfar_grd_cell_velocity: int = 0
    cfar_trn_cell_range: int = 0
    cfar_trn_cell_velocity: int = 0
    cfar_trn_cell_azimuth: int = 8
    cfar_trn_cell_elevation: int = 6
    cfar_grd_cell_azimuth: int = 4
    cfar_grd_cell_elevation: int = 3
    cfar_threshold: float = 3.0
    az_fft_len: int = 64
    el_fft_len: int = 64
    rda_threshold: float = 20.0
    nms_radius: Sequence[int] = field(default_factory=lambda: [2, 2, 1, 1])
    nms_max_peaks: int = 200


@dataclass
class Geometry:
    tx: list[list[float]]
    rx: list[list[float]]


@dataclass
class Target:
    position: list[list[float]]
    velocity: list[list[float]]


def apply_overrides(model, overrides: Mapping[str, object]):
    for key, value in overrides.items():
        if hasattr(model, key):
            setattr(model, key, value)
    return model


def require_sequence(value: Iterable[float] | float, length: int) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)] * length
    seq = list(value)
    if len(seq) != length:
        raise ValueError(f"Expected sequence length {length}, got {len(seq)}")
    return [float(x) for x in seq]
