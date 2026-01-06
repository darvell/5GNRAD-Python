from gnrad5.paper_chain import run_paper_chain
from gnrad5.paper_scenarios import build_paper_scenario


def test_paper_chain_shapes_no_channel():
    spec = build_paper_scenario("UMI", 25, drops=1, seed=0, target_absent=True)
    det = run_paper_chain(
        spec.sim,
        spec.prs,
        spec.sens,
        spec.geometry,
        spec.target,
        target_absent=True,
        eta_db=3.4,
        codebook_id="grid",
        beam_az=3,
        beam_el=3,
        no_channel=True,
        add_noise=False,
        max_symbols=4,
        max_range_bins=64,
    )

    assert det.par_db.shape[0] == 9
    assert det.beam_grid.shape[0] == 9
    assert det.range_bins.size > 0
