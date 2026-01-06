from pathlib import Path

import os
import numpy as np
import pytest
from scipy.io import loadmat

from gnrad5.config import load_scenario
from gnrad5.signal import build_prs_grid
from gnrad5.run import run_slice


def _fixture_paths():
    root = Path(__file__).resolve().parents[1]
    fixtures = []
    for folder in ("examples", "examples3GPP"):
        base = root / folder
        if not base.exists():
            continue
        for scenario in base.iterdir():
            out_dir = scenario / "Output"
            if not out_dir.exists():
                continue
            prs_idx = out_dir / "matlab_prs_indices.csv"
            prs_sym = out_dir / "matlab_prs_symbols.csv"
            rd_mat = out_dir / "matlab_rd.mat"
            rda_mat = out_dir / "matlab_rda.mat"
            fixtures.append((scenario, prs_idx, prs_sym, rd_mat, rda_mat))
    return fixtures


@pytest.mark.parametrize("scenario,prs_idx,prs_sym,rd_mat,rda_mat", _fixture_paths())
def test_matlab_prs_fixture(scenario, prs_idx, prs_sym, rd_mat, rda_mat):
    if not (prs_idx.exists() and prs_sym.exists()):
        pytest.skip("Missing MATLAB PRS fixtures")

    sim, target, prs, _, sens = load_scenario(scenario)
    prs_grid = build_prs_grid(sim, prs, sens.number_sensing_symbols)
    idx = np.where(np.abs(prs_grid.grid) > 0)
    lin = (idx[0] + 1) + idx[1] * prs_grid.grid.shape[0]
    matlab_idx = np.loadtxt(prs_idx, delimiter=",").astype(int)
    matlab_sym = np.loadtxt(prs_sym, delimiter=",")
    if matlab_sym.ndim == 2 and matlab_sym.shape[1] == 2:
        matlab_sym = matlab_sym[:, 0] + 1j * matlab_sym[:, 1]
    elif matlab_sym.ndim == 2 and matlab_sym.shape[1] == 1:
        matlab_sym = matlab_sym[:, 0]
    assert np.array_equal(lin, matlab_idx)
    if matlab_sym.ndim == 1:
        matlab_sym = matlab_sym.astype(np.complex128)
    assert np.max(np.abs(prs_grid.grid[idx] - matlab_sym)) < 1e-6


@pytest.mark.parametrize("scenario,prs_idx,prs_sym,rd_mat,rda_mat", _fixture_paths())
def test_matlab_rd_rda_fixture(scenario, prs_idx, prs_sym, rd_mat, rda_mat):
    if not rd_mat.exists():
        pytest.skip("Missing MATLAB RD fixtures")
    if os.environ.get("GN5_VALIDATE_RD") != "1":
        pytest.skip("Set GN5_VALIDATE_RD=1 to enable RD parity check")

    max_range_bins = os.environ.get("GN5_RDA_MAX_RANGE_BINS")
    max_range_bins = int(max_range_bins) if max_range_bins else None

    rd = loadmat(rd_mat)["rd"]
    info, debug = run_slice(
        str(scenario),
        repo_root=None,
        no_channel=True,
        add_noise=False,
        log_stage_max=False,
        max_range_bins=max_range_bins,
    )
    rd_py = debug["rd_map"]
    assert rd.shape == rd_py.shape
    diff = np.max(np.abs(rd - rd_py))
    assert diff < 1e-6

    if rda_mat.exists():
        rda = loadmat(rda_mat)["rda"]
        rda_py = debug["rda"]
        assert rda.shape == rda_py.shape
        diff_rda = np.max(np.abs(rda - rda_py))
        assert diff_rda < 1e-6
