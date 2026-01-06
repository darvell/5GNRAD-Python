import numpy as np

from gnrad5.config.models import PrsConfig, SimulationConfig
from gnrad5.nr import build_nr_objects, prs_indices_local, prs_symbols_local


def test_prs_sequence_repeatability():
    sim = SimulationConfig(carrier_subcarrier_spacing=30, carrier_n_size_grid=66)
    prs_cfg = PrsConfig(
        prs_resource_set_period=[1, 0],
        prs_resource_offset=0,
        prs_resource_repetition=1,
        prs_resource_time_gap=1,
        num_rb=66,
        rb_offset=0,
        comb_size=2,
        re_offset=0,
        n_prs_id=0,
        num_prs_symbols=2,
        symbol_start=0,
    )
    nr = build_nr_objects(sim, prs_cfg)
    nr.carrier.NSlot = 0

    indices = prs_indices_local(nr.carrier, nr.prs)
    symbols = prs_symbols_local(nr.carrier, nr.prs)

    assert indices.ndim == 1
    assert symbols.ndim == 1
    assert len(indices) == len(symbols)
    assert np.isfinite(symbols).all()
