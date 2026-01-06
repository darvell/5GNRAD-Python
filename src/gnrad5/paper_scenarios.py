from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gnrad5.config.models import Geometry, PrsConfig, SensConfig, SimulationConfig, Target


@dataclass
class PaperScenario:
    sim: SimulationConfig
    prs: PrsConfig
    sens: SensConfig
    geometry: Geometry
    target: Target
    prs_period_slots: int
    eirp_dbm: float
    fixed_rcs_dbsm: float
    fixed_nrp: int
    target_absent: bool


def build_paper_scenario(
    scenario: str,
    altitude_m: float,
    drops: int,
    seed: int | None = None,
    target_absent: bool = False,
    fixed_rcs_dbsm: float = -12.81,
    fixed_nrp: int = 3,
    eirp_dbm: float = 75.0,
    prs_repetition: int = 1,
    prs_period_slots: int = 32,
    sector_deg: float = 120.0,
    target_speed_mps: float = 10.0,
):
    scenario = scenario.upper()
    if scenario not in {"UMI", "UMA"}:
        raise ValueError("scenario must be UMi or UMa")

    if scenario == "UMI":
        bs_height = 10.0
        isd = 200.0
        channel_scenario = "UMiAV"
    else:
        bs_height = 25.0
        isd = 500.0
        channel_scenario = "UMaAV"

    sim = SimulationConfig(
        system_fc=30e9,
        system_nf=7.0,
        system_bw=100e6,
        channel_scenario=channel_scenario,
        antenna_num_h=32,
        antenna_num_v=32,
        antenna_coupling_efficiency=0.7,
        carrier_subcarrier_spacing=120,
        carrier_n_size_grid=66,
        n_st_drop=1,
        max_range_interest=400.0,
    )

    prs = PrsConfig(
        # Keep the grid compact (period=1) and let the processing chain apply the
        # physical periodicity via `prs_period_slots`.
        prs_resource_set_period=[1, 0],
        prs_resource_offset=0,
        prs_resource_repetition=int(prs_repetition),
        prs_resource_time_gap=1,
        num_rb=66,
        rb_offset=0,
        comb_size=4,
        re_offset=0,
        n_prs_id=0,
        num_prs_symbols=4,
        symbol_start=0,
    )

    sens = SensConfig(
        doppler_fft_len=256,
        window="blackmanharris",
        window_len=256,
        window_overlap=0.0,
        number_sensing_symbols=256,
        cfar_grd_cell_range=0,
        cfar_grd_cell_velocity=0,
        cfar_trn_cell_range=0,
        cfar_trn_cell_velocity=0,
        cfar_trn_cell_azimuth=8,
        cfar_trn_cell_elevation=6,
        cfar_grd_cell_azimuth=4,
        cfar_grd_cell_elevation=3,
        cfar_threshold=3.0,
        az_fft_len=64,
        el_fft_len=64,
        rda_threshold=20.0,
        nms_radius=[2, 2, 1, 1],
        nms_max_peaks=200,
    )

    geometry = Geometry(tx=[[0.0, 0.0, bs_height]], rx=[[0.0, 0.0, bs_height]])

    rng = np.random.default_rng(seed)
    positions = []
    velocities = []
    if not target_absent:
        min_r = 10.0
        max_r = isd / 2.0
        half_sector = np.deg2rad(float(sector_deg)) / 2.0
        for _ in range(int(drops)):
            r = rng.uniform(min_r, max_r)
            theta = rng.uniform(-half_sector, half_sector)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            positions.append([x, y, altitude_m])
            heading = rng.uniform(-np.pi, np.pi)
            speed = float(target_speed_mps)
            velocities.append([speed * np.cos(heading), speed * np.sin(heading), 0.0])

    target = Target(position=positions, velocity=velocities)
    return PaperScenario(
        sim=sim,
        prs=prs,
        sens=sens,
        geometry=geometry,
        target=target,
        prs_period_slots=int(prs_period_slots),
        eirp_dbm=eirp_dbm,
        fixed_rcs_dbsm=fixed_rcs_dbsm,
        fixed_nrp=fixed_nrp,
        target_absent=target_absent,
    )
