from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np

from gnrad5.config.models import PrsConfig, SimulationConfig
from gnrad5.nr import build_nr_objects, prs_indices, prs_symbols


class PrsGrid:
    def __init__(self, grid: np.ndarray, symbol_indices: np.ndarray, nr_objects):
        self.grid = grid
        self.symbol_indices = symbol_indices
        self.nr_objects = nr_objects


_PRS_CACHE: "OrderedDict[tuple[Any, ...], PrsGrid]" = OrderedDict()
_PRS_CACHE_MAX = 8


def _model_key(model) -> tuple[Any, ...]:
    items = []
    for key, value in sorted(model.__dict__.items()):
        if isinstance(value, (list, tuple)):
            items.append((key, tuple(value)))
        else:
            items.append((key, value))
    return tuple(items)


def _prs_cache_key(sim: SimulationConfig, prs_cfg: PrsConfig, number_sensing_symbols: int) -> tuple[Any, ...]:
    return (
        number_sensing_symbols,
        _model_key(sim),
        _model_key(prs_cfg),
    )


def build_prs_grid(
    sim: SimulationConfig,
    prs_cfg: PrsConfig,
    number_sensing_symbols: int,
) -> PrsGrid:
    key = _prs_cache_key(sim, prs_cfg, number_sensing_symbols)
    cached = _PRS_CACHE.get(key)
    if cached is not None:
        _PRS_CACHE.move_to_end(key)
        return cached

    nr_objects = build_nr_objects(sim, prs_cfg)
    carrier = nr_objects.carrier
    prs = nr_objects.prs
    ofdm_info = nr_objects.ofdm_info

    number_subcarriers = sim.carrier_n_size_grid * 12
    if isinstance(ofdm_info, dict):
        ofdm_fft_len = int(ofdm_info.get("Nfft", number_subcarriers))
    else:
        ofdm_fft_len = int(getattr(ofdm_info, "Nfft", number_subcarriers))
    num_sym_per_slot = int(getattr(carrier, "SymbolsPerSlot", 14))

    prs_period = int(prs_cfg.prs_resource_set_period[0])
    num_slots = number_sensing_symbols * prs_period

    start_idx0 = (ofdm_fft_len - number_subcarriers) // 2

    grid = np.zeros((ofdm_fft_len, num_sym_per_slot * num_slots), dtype=np.complex128)
    symbol_cols = []

    for slot_idx in range(0, num_slots, prs_period):
        setattr(carrier, "NSlot", slot_idx)
        ind_cell = prs_indices(carrier, prs)
        sym_cell = prs_symbols(carrier, prs)
        idx = np.asarray(ind_cell[0]).reshape(-1)
        val = np.asarray(sym_cell[0]).reshape(-1)

        idx0 = idx.astype(np.int64) - 1
        row = idx0 % number_subcarriers
        col = idx0 // number_subcarriers

        row = row + start_idx0
        col = col + slot_idx * num_sym_per_slot

        grid[row, col] = val
        symbol_cols.extend(col.tolist())

    symbol_indices = np.unique(np.asarray(symbol_cols, dtype=np.int64))
    grid.setflags(write=False)
    symbol_indices.setflags(write=False)
    prs_grid = PrsGrid(grid=grid, symbol_indices=symbol_indices, nr_objects=nr_objects)

    _PRS_CACHE[key] = prs_grid
    _PRS_CACHE.move_to_end(key)
    if len(_PRS_CACHE) > _PRS_CACHE_MAX:
        _PRS_CACHE.popitem(last=False)
    return prs_grid
