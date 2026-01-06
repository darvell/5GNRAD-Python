import numpy as np

from gnrad5.config.models import PrsConfig, SimulationConfig
from gnrad5.detection import par_detector


def test_par_detector_detects_impulse():
    sim = SimulationConfig(carrier_n_size_grid=1)
    prs = PrsConfig(num_rb=1, comb_size=2, num_prs_symbols=1, symbol_start=0, re_offset=0)
    tx_grid = np.ones((16, 4), dtype=np.complex128)
    rx_grid = tx_grid.copy()
    symbol_indices = np.array([0, 1, 2, 3])

    res = par_detector(
        tx_grid,
        rx_grid,
        symbol_indices,
        sim,
        prs,
        eta_db=0.0,
        window="hamming",
        symbols_per_slot=14,
        clutter_mean=False,
    )

    assert res["detected"]
    assert res["range_idx"] >= 0
