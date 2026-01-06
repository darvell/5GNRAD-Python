from pathlib import Path

import pytest

from gnrad5.run import run_slice


def _scenario_dirs():
    root = Path(__file__).resolve().parents[1]
    scenarios = []
    for folder in ("examples", "examples3GPP"):
        base = root / folder
        if not base.exists():
            continue
        for scenario in base.iterdir():
            if (scenario / "Input" / "simulationConfig.txt").exists():
                scenarios.append(str(scenario))
    return scenarios


@pytest.mark.parametrize("scenario", _scenario_dirs())
def test_scenario_smoke(scenario):
    info, _ = run_slice(
        scenario,
        repo_root=None,
        max_symbols=4,
        max_range_bins=128,
        doppler_fft_len=16,
        az_fft_len=8,
        el_fft_len=8,
        no_channel=True,
        rda_chunk=64,
        skip_range_mean=True,
        add_noise=False,
    )
    assert info["rd_shape"][0] <= 128
