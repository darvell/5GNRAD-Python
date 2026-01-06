from __future__ import annotations

import numpy as np

from gnrad5.config.models import PrsConfig


_K_PRIME_TABLE = {
    2: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    4: [0, 2, 1, 3, 0, 2, 1, 3, 0, 2, 1, 3],
    6: [0, 3, 1, 4, 2, 5, 0, 3, 1, 4, 2, 5],
    12: [0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11],
}


def prs_destagger(y: np.ndarray, prs: PrsConfig, symbol_indices: np.ndarray) -> np.ndarray:
    if y.size == 0:
        return y
    comb = int(prs.comb_size)
    if comb not in _K_PRIME_TABLE:
        raise ValueError(f"Unsupported comb size {comb}")
    l_prs_start = int(prs.symbol_start)
    k_primes = _K_PRIME_TABLE[comb]

    symbol_indices = np.asarray(symbol_indices).astype(int)
    sym_offsets = symbol_indices - l_prs_start
    foffset = []
    for sym in sym_offsets:
        if sym < 0 or sym >= len(k_primes):
            raise ValueError("Invalid symbol index")
        kprime = k_primes[sym]
        foffset.append((prs.re_offset + kprime) % comb)
    order = np.argsort(foffset)
    return y[:, order]
