from __future__ import annotations

from types import SimpleNamespace
from typing import Sequence

import numpy as np


def build_prs_config(**kwargs):
    return SimpleNamespace(**kwargs)


def prs_indices_local(carrier, prs):
    n_symb_slot = int(getattr(carrier, "SymbolsPerSlot", 14))
    num_rb = int(getattr(prs, "NumRB"))
    rb_offset = int(getattr(prs, "RBOffset", 0))
    comb_size = int(getattr(prs, "CombSize"))
    comb_offset = int(getattr(prs, "REOffset", 0))
    sym_start = int(getattr(prs, "SymbolStart", 0))
    num_prs_symbols = int(getattr(prs, "NumPRSSymbols", 1))

    num_subcarriers = num_rb * 12
    indices = []

    for l in range(sym_start, sym_start + num_prs_symbols):
        re_offset = _freq_offset(comb_size, sym_start, l)
        k_init = rb_offset * 12 + (comb_offset + re_offset) % comb_size
        nof_symbols_prb = 12 // comb_size
        for rb in range(num_rb):
            base = (rb * 12) + k_init
            for i in range(nof_symbols_prb):
                sc = base + comb_size * i
                if sc < num_subcarriers:
                    idx = sc + 1 + l * num_subcarriers
                    indices.append(idx)
    return np.asarray(indices, dtype=np.int64)


def prs_symbols_local(carrier, prs):
    import py3gpp

    n_symb_slot = int(getattr(carrier, "SymbolsPerSlot", 14))
    slot_idx = int(getattr(carrier, "NSlot", 0))
    n_id_prs = int(getattr(prs, "NPRSID", 0))
    num_rb = int(getattr(prs, "NumRB"))
    rb_offset = int(getattr(prs, "RBOffset", 0))
    comb_size = int(getattr(prs, "CombSize"))
    sym_start = int(getattr(prs, "SymbolStart", 0))
    num_prs_symbols = int(getattr(prs, "NumPRSSymbols", 1))

    nof_symbols_prb = 12 // comb_size
    total_re = nof_symbols_prb * num_rb
    symbols = []

    for l in range(sym_start, sym_start + num_prs_symbols):
        c_init = (
            (1 << 22) * (n_id_prs // 1024)
            + (1 << 10) * (n_symb_slot * slot_idx + l + 1) * (2 * (n_id_prs % 1024) + 1)
            + (n_id_prs % 1024)
        ) % (1 << 31)

        seq_len = 2 * nof_symbols_prb * (rb_offset + num_rb)
        bits = np.asarray(py3gpp.nrPRBS(c_init, seq_len), dtype=np.int8)
        if rb_offset > 0:
            bits = bits[2 * nof_symbols_prb * rb_offset :]
        bits = bits[: 2 * total_re]
        c0 = bits[0::2]
        c1 = bits[1::2]
        sym = (1 - 2 * c0 + 1j * (1 - 2 * c1)) / np.sqrt(2)
        symbols.append(sym)
    return np.concatenate(symbols)


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
