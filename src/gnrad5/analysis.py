from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gnrad5.config import load_scenario
from gnrad5.constants import SPEED_OF_LIGHT
from gnrad5.geometry import estimate_scattering_geometry, set_st_min_distance_constraint
from gnrad5.metrics import score_associations_pos
from gnrad5.run import run_slice
from gnrad5.signal import build_prs_grid, get_ofdm_params


@dataclass
class RunOutputs:
    results: pd.DataFrame
    det_stats: pd.DataFrame
    info: dict


def run5gnrad(
    scenario: str,
    repo_root: str | Path | None = None,
    strict_matlab_range_mean: bool = True,
):
    sim, target, prs, geometry, sens = load_scenario(scenario)
    st_position = np.asarray(target.position)
    st_velocity = np.asarray(target.velocity)

    n_st = st_position.shape[0]
    n_drop = int(np.floor(n_st / sim.n_st_drop))

    index = set_st_min_distance_constraint(st_position, sim.n_st_drop, 10)
    st_position = st_position[index]
    st_velocity = st_velocity[index]

    time_index = []
    position_error_x = []
    position_error_y = []
    position_error_z = []
    position_error_h = []
    position_error_v = []
    range_error = []
    velocity_error = []
    azimuth_error = []
    elevation_error = []
    snr_vals = []

    true_positive = np.zeros(n_drop, dtype=int)
    false_negative = np.zeros(n_drop, dtype=int)
    false_positive = np.zeros(n_drop, dtype=int)
    false_alarm_prob = np.zeros(n_drop, dtype=float)

    tx_pos = np.asarray(geometry.tx[0]) if geometry.tx else np.zeros(3)
    repo_root = Path(repo_root or Path.cwd())
    background_channel = None
    target_channel = None
    try:
        from gnrad5.channel import load_background_channel, load_target_channel

        background_channel = load_background_channel(repo_root, scenario, sim.system_fc)
        target_channel = load_target_channel(repo_root, scenario, sim.system_fc)
    except Exception:
        background_channel = None
        target_channel = None

    c = SPEED_OF_LIGHT
    scs_khz = sim.carrier_subcarrier_spacing
    mu = int(round(np.log2(scs_khz / 15))) if scs_khz else 0
    slots_per_subframe = 2**mu
    prs_grid = build_prs_grid(sim, prs, sens.number_sensing_symbols)
    ofdm_info = prs_grid.nr_objects.ofdm_info
    ofdm_params = get_ofdm_params(ofdm_info, prs_grid.nr_objects.carrier)
    ofdm_symbol_lengths = getattr(ofdm_info, "SymbolLengths", None) if not isinstance(ofdm_info, dict) else None
    if ofdm_symbol_lengths is not None and ofdm_params.sample_rate:
        slot_duration = float(np.sum(np.asarray(ofdm_symbol_lengths)) / ofdm_params.sample_rate)
    else:
        slot_duration = 1.0
    prs_periodicity = slot_duration / slots_per_subframe * int(prs.prs_resource_set_period[0])
    cpi = prs_periodicity * sens.number_sensing_symbols

    for q in range(n_drop):
        target_idx = slice(q * sim.n_st_drop, (q + 1) * sim.n_st_drop)
        st_pos_q = st_position[target_idx]
        st_vel_q = st_velocity[target_idx]
        bg_entry = background_channel[q] if isinstance(background_channel, list) else background_channel
        tgt_entry = None
        if isinstance(target_channel, list):
            tgt_entry = target_channel[target_idx]
        elif target_channel is not None:
            tgt_entry = target_channel

        info, debug = run_slice(
            scenario,
            repo_root=repo_root,
            max_symbols=None,
            max_range_bins=None,
            doppler_fft_len=None,
            az_fft_len=None,
            el_fft_len=None,
            no_channel=False,
            rda_chunk=None,
            prs_export=None,
            prs_compare=None,
            cfar_threshold=None,
            rda_threshold=None,
            target_power_boost=0.0,
            no_background=False,
            single_path=False,
            log_stage_max=False,
            skip_range_mean=not strict_matlab_range_mean,
            min_clutter_ratio=None if strict_matlab_range_mean else 1e-2,
            matlab_range_mean=strict_matlab_range_mean,
            target_positions=st_pos_q,
            target_velocities=st_vel_q,
            background_channel=bg_entry,
            target_channel=tgt_entry,
            add_noise=True,
        )

        detections = debug["detections_4d"].astype(int)
        if detections.size:
            az_grid = debug.get("az_grid")
            el_grid = debug.get("el_grid")
            az_est = az_grid[detections[:, 2] - 1, detections[:, 3] - 1] if az_grid is not None else np.zeros(len(detections))
            el_est = el_grid[detections[:, 2] - 1, detections[:, 3] - 1] if el_grid is not None else np.zeros(len(detections))
            rng_bins = debug.get("range_bins", np.zeros(0))
            vel_bins = debug.get("velocity_bins", np.zeros(0))
            rng_est = rng_bins[detections[:, 0] - 1] if rng_bins.size else np.zeros(len(detections))
            vel_est = vel_bins[detections[:, 1] - 1] if vel_bins.size else np.zeros(len(detections))
        else:
            az_est = np.zeros(0)
            el_est = np.zeros(0)
            rng_est = np.zeros(0)
            vel_est = np.zeros(0)

        if detections.size:
            st_pos_est, *_ = estimate_scattering_geometry(
                tx_pos,
                tx_pos,
                2 * rng_est / c * 1e9,
                np.column_stack([az_est, el_est]),
                keep_sector_az=[-60, 60],
            )
            valid_mask = np.all(np.isfinite(st_pos_est), axis=1)
            st_pos_est = st_pos_est[valid_mask]
            vel_est = vel_est[valid_mask]
        else:
            st_pos_est = np.zeros((0, 3))
            vel_est = np.zeros(0)

        gt_vel = np.sum(
            (-(tx_pos) + st_pos_q)
            / np.linalg.norm(tx_pos - st_pos_q, axis=1, keepdims=True)
            * st_vel_q,
            axis=1,
        )

        metrics, _ = score_associations_pos(st_pos_q, st_pos_est, gt_vel, vel_est, tx_pos)
        stats = metrics["stats"]
        time_index.append(np.full(sim.n_st_drop, q + 1))
        position_error_x.append(stats["pos"]["errXYZ"][:, 0])
        position_error_y.append(stats["pos"]["errXYZ"][:, 1])
        position_error_z.append(stats["pos"]["errXYZ"][:, 2])
        position_error_h.append(np.linalg.norm(stats["pos"]["errXYZ"][:, :2], axis=1))
        position_error_v.append(stats["pos"]["errXYZ"][:, 2])
        range_error.append(stats["pos"]["range_err"])
        velocity_error.append(stats["vel"]["vr_err"])
        azimuth_error.append(stats["pos"]["az_err_deg"])
        elevation_error.append(stats["pos"]["el_err_deg"])
        true_positive[q] = metrics["TP"]
        false_negative[q] = metrics["FN"]
        false_positive[q] = metrics["FP"]
        false_alarm_prob[q] = metrics["FPR"]
        if debug.get("sensing") is not None:
            tgt_pg = debug["sensing"].tgt_pg
            k = 1.380649e-23
            t_k = 297.0
            nf = 10 ** (sim.system_nf / 10)
            noise_power = k * t_k * sim.system_bw * nf
            max_power = 52 - 30 + 8
            p_lin = 10 ** (max_power / 10)
            snr_var = p_lin / noise_power
            if tgt_pg.size and tgt_pg.size % sim.n_st_drop == 0:
                tgt_pg_lin = 10 ** (tgt_pg / 10)
                reshape_pg = tgt_pg_lin.reshape(-1, sim.n_st_drop)
                snr_vals.append(10 * np.log10(np.sum(snr_var * reshape_pg, axis=0)))
            else:
                snr_vals.append(np.full(sim.n_st_drop, 10 * np.log10(snr_var)))
        else:
            snr_vals.append(np.full(sim.n_st_drop, 0.0))

    results = pd.DataFrame(
        {
            "timeIndex": np.concatenate(time_index) if time_index else np.array([]),
            "positionErrorX": np.concatenate(position_error_x) if position_error_x else np.array([]),
            "positionErrorY": np.concatenate(position_error_y) if position_error_y else np.array([]),
            "positionErrorZ": np.concatenate(position_error_z) if position_error_z else np.array([]),
            "rangeError": np.concatenate(range_error) if range_error else np.array([]),
            "velocityError": np.concatenate(velocity_error) if velocity_error else np.array([]),
            "azimuthError": np.concatenate(azimuth_error) if azimuth_error else np.array([]),
            "elevationError": np.concatenate(elevation_error) if elevation_error else np.array([]),
            "positionErrorH": np.concatenate(position_error_h) if position_error_h else np.array([]),
            "positionErrorV": np.concatenate(position_error_v) if position_error_v else np.array([]),
            "snr": np.concatenate(snr_vals) if snr_vals else np.array([]),
        }
    )
    det_stats = pd.DataFrame(
        {
            "truePositive": true_positive,
            "falseNegative": false_negative,
            "falsePositve": false_positive,
            "falseAlarmProb": false_alarm_prob,
        }
    )
    info = {
        "K": int(np.sum(true_positive >= 1)),
        "cpi": float(cpi),
        "isLos": np.zeros(n_st),
    }
    return RunOutputs(results=results, det_stats=det_stats, info=info)


def write_outputs(outputs: RunOutputs, scenario_path: str | Path):
    out_dir = Path(scenario_path) / "Output"
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs.results.to_csv(out_dir / "error.csv", index=False)
    outputs.det_stats.to_csv(out_dir / "detStats.csv", index=False)
