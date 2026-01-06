from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from gnrad5.antenna import URA, steering_vector
from gnrad5.channel.pathloss import compute_path_loss
from gnrad5.channel.rcs import get_sigma_rcs
from gnrad5.constants import SPEED_OF_LIGHT
from gnrad5.geometry import angle2vector, wrap_to_180
from gnrad5.signal.sinc_interp import sinc_interp


@dataclass
class SensingCdlOutput:
    delay: np.ndarray
    path_gain: np.ndarray
    aoa_az: np.ndarray
    aoa_el: np.ndarray
    aod_az: np.ndarray
    aod_el: np.ndarray
    angle_estimate: np.ndarray
    los: np.ndarray
    sync_offset: float
    tgt_pg: np.ndarray


def get_sensing_cdl(
    bs_pos: np.ndarray,
    tgt_pos: np.ndarray,
    tgt_vel: np.ndarray,
    fc: float,
    pri: float,
    background_channel: dict,
    target_channel: list[dict] | None = None,
    n_realization: int = 1,
    sample_rate: float = 100e6,
    transmit_array: URA | None = None,
    receive_array: URA | None = None,
    scanvector: np.ndarray | None = None,
    angle_estimation: str = "ideal",
    scenario: str = "UMiAV",
    fixed_rcs_dbsm: float | None = None,
    allow_no_target: bool = False,
    rng: np.random.Generator | None = None,
):
    c = SPEED_OF_LIGHT
    lamb = c / fc
    time_vector = np.arange(n_realization) * pri
    if transmit_array is None:
        transmit_array = URA((64, 1), lamb / 2)
    if receive_array is None:
        receive_array = URA((64, 1), lamb / 2)
    if scanvector is None:
        scanvector = np.zeros((1, 2))

    tgt_pos = np.asarray(tgt_pos) if tgt_pos is not None else np.zeros((0, 3))
    tgt_vel = np.asarray(tgt_vel) if tgt_vel is not None else np.zeros((0, 3))

    tgt_paths = []
    tgt_delays = []
    tgt_aoa_az = []
    tgt_aoa_el = []
    tgt_aod_az = []
    tgt_aod_el = []
    tgt_pg = []
    tgt_phase = []
    has_los_list = []

    if target_channel:
        for idx, entry in enumerate(target_channel):
            delays = np.asarray(entry.get("PathDelays", []), dtype=float)
            aoa_az = wrap_to_180(np.asarray(entry.get("AnglesAoA", []), dtype=float))
            aoa_el = 90 - np.asarray(entry.get("AnglesZoA", []), dtype=float)
            aod_az = wrap_to_180(np.asarray(entry.get("AnglesAoD", []), dtype=float))
            aod_el = 90 - np.asarray(entry.get("AnglesZoD", []), dtype=float)
            gains = np.asarray(entry.get("AveragePathGains", []), dtype=float)
            phases = np.asarray(entry.get("InitialPhases", []), dtype=float)

            if delays.size == 0:
                continue

            vel = tgt_vel[idx] if tgt_vel.ndim > 1 else tgt_vel
            dod_vec = angle2vector(aod_az, 90 - aod_el, 1.0)
            doa_vec = angle2vector(aoa_az, 90 - aoa_el, 1.0)
            dod_vec = dod_vec / np.linalg.norm(dod_vec, axis=1, keepdims=True)
            doa_vec = doa_vec / np.linalg.norm(doa_vec, axis=1, keepdims=True)
            v_rel = dod_vec @ vel + doa_vec @ vel
            f_d = v_rel / lamb
            doppler_phase = np.exp(1j * 2 * np.pi * f_d[:, None] * time_vector)
            if phases.size:
                doppler_phase = doppler_phase * np.exp(1j * phases[:, None])

            tgt_paths.append(doppler_phase)
            tgt_delays.append(delays)
            tgt_aoa_az.append(aoa_az)
            tgt_aoa_el.append(aoa_el)
            tgt_aod_az.append(aod_az)
            tgt_aod_el.append(aod_el)
            tgt_pg.append(10 * np.log10(np.abs(gains)))
            tgt_phase.append(doppler_phase)
            has_los_list.append(np.asarray(entry.get("HasLOSCluster", []), dtype=float))

    env_pg_lin = np.asarray(background_channel.get("AveragePathGains", []), dtype=float)
    env_pg = 10 * np.log10(np.abs(env_pg_lin)) if env_pg_lin.size else np.zeros(0)
    env_delays = np.asarray(background_channel.get("PathDelays", []), dtype=float)
    env_aoa_az = np.asarray(background_channel.get("AnglesAoA", []), dtype=float)
    env_aoa_el = np.asarray(background_channel.get("AnglesZoA", []), dtype=float)
    env_aod_az = np.asarray(background_channel.get("AnglesAoD", []), dtype=float)
    env_aod_el = np.asarray(background_channel.get("AnglesZoD", []), dtype=float)
    env_phases = np.asarray(background_channel.get("InitialPhases", []), dtype=float)

    if not target_channel:
        no_targets = tgt_pos.size == 0
        if allow_no_target and no_targets:
            pass
        else:
            if rng is None:
                rng = np.random.default_rng()
            tgt_rcs = (
                float(fixed_rcs_dbsm)
                if fixed_rcs_dbsm is not None
                else float(get_sigma_rcs("uav-small", return_large_scale=True, rng=rng))
            )
            tgt_range = float(np.linalg.norm(tgt_pos - bs_pos))
            tgt_delay = np.array([2 * tgt_range / c])
            bs2tgt_vec = (tgt_pos - bs_pos) / tgt_range
            v_rel = float(np.dot(tgt_vel, bs2tgt_vec))
            f_d = 2 * (v_rel / lamb)
            pl, has_los_flag = compute_path_loss(scenario, fc, bs_pos, tgt_pos, False, rng)
            tgt_pg_val = np.array([-(2 * pl - tgt_rcs + 10 * math.log10(c**2 / (4 * math.pi * fc**2)))])
            doppler_phase = np.exp(1j * 2 * np.pi * f_d * time_vector)[None, :]

            vx, vy, vz = bs2tgt_vec
            aoa_az = np.rad2deg(np.arctan2(vy, vx))
            aoa_el = np.rad2deg(np.arctan2(vz, np.sqrt(vx**2 + vy**2)))
            tgt_delays = [tgt_delay]
            tgt_pg = [tgt_pg_val]
            tgt_aoa_az = [np.array([aoa_az])]
            tgt_aoa_el = [np.array([aoa_el])]
            tgt_aod_az = [np.array([aoa_az])]
            tgt_aod_el = [np.array([aoa_el])]
            tgt_phase = [doppler_phase]
            has_los_list = [np.array([has_los_flag])]
    tgt_delay = np.concatenate(tgt_delays) if tgt_delays else np.zeros(0)
    tgt_pg = np.concatenate(tgt_pg) if tgt_pg else np.zeros(0)
    tgt_aoa_az = np.concatenate(tgt_aoa_az) if tgt_aoa_az else np.zeros(0)
    tgt_aoa_el = np.concatenate(tgt_aoa_el) if tgt_aoa_el else np.zeros(0)
    tgt_aod_az = np.concatenate(tgt_aod_az) if tgt_aod_az else np.zeros(0)
    tgt_aod_el = np.concatenate(tgt_aod_el) if tgt_aod_el else np.zeros(0)

    if tgt_pg.size:
        keep_path = tgt_pg > (tgt_pg.max() - 40)
    else:
        keep_path = np.zeros_like(tgt_pg, dtype=bool)

    doppler_phase = np.concatenate(tgt_phase, axis=0) if tgt_phase else np.zeros((0, n_realization), dtype=complex)
    tgt_phase_mat = np.angle(doppler_phase)
    if tgt_phase_mat.size:
        tgt_phase_mat = tgt_phase_mat[keep_path]
    env_phase_mat = np.tile(env_phases[:, None], (1, n_realization)) if env_phases.size else np.zeros((0, n_realization))
    phases_combined = np.concatenate([tgt_phase_mat, env_phase_mat], axis=0).T

    path_gain_combined = np.concatenate([tgt_pg[keep_path], env_pg])
    delays_combined = np.concatenate([tgt_delay[keep_path], env_delays])
    aoa_az_combined = np.concatenate([tgt_aoa_az[keep_path], env_aoa_az])
    aoa_el_combined = np.concatenate([tgt_aoa_el[keep_path], env_aoa_el])
    aod_az_combined = np.concatenate([tgt_aod_az[keep_path], env_aod_az])
    aod_el_combined = np.concatenate([tgt_aod_el[keep_path], env_aod_el])

    if delays_combined.size == 0:
        delays_sorted = np.array([0.0])
        path_gain_sorted = np.array([0.0])
        phases_sorted = np.zeros((n_realization, 1))
        aoa_az_sorted = np.zeros(1)
        aoa_el_sorted = np.zeros(1)
        aod_az_sorted = np.zeros(1)
        aod_el_sorted = np.zeros(1)
    else:
        order = np.argsort(delays_combined)
        delays_sorted = delays_combined[order]
        path_gain_sorted = path_gain_combined[order]
        aoa_az_sorted = aoa_az_combined[order]
        aoa_el_sorted = aoa_el_combined[order]
        aod_az_sorted = aod_az_combined[order]
        aod_el_sorted = aod_el_combined[order]
        phases_sorted = phases_combined[:, order]

    drop_ray = (aod_el_sorted >= 90) | (aoa_el_sorted >= 90) | (aod_el_sorted <= -90) | (aoa_el_sorted <= -90)
    if np.any(drop_ray):
        keep = ~drop_ray
        delays_sorted = delays_sorted[keep]
        path_gain_sorted = path_gain_sorted[keep]
        aoa_az_sorted = aoa_az_sorted[keep]
        aoa_el_sorted = aoa_el_sorted[keep]
        aod_az_sorted = aod_az_sorted[keep]
        aod_el_sorted = aod_el_sorted[keep]
        phases_sorted = phases_sorted[:, keep]

    n_samples = int(np.ceil((delays_sorted[-1] - delays_sorted[0]) * sample_rate) + 10)
    time_sampling = np.arange(n_samples) / sample_rate
    cir = (np.sqrt(10 ** (path_gain_sorted / 10.0)) * np.exp(1j * phases_sorted)).T

    tx_pv = steering_vector(fc, np.vstack([aod_az_sorted, aod_el_sorted]), transmit_array)
    rx_pv = steering_vector(fc, np.vstack([aoa_az_sorted, aoa_el_sorted]), receive_array)

    aoa_az_los = aoa_az_sorted[:1]
    aoa_el_los = aoa_el_sorted[:1]
    if angle_estimation == "scan":
        rx_bf = np.conj(steering_vector(fc, scanvector.T, receive_array))
        rx_gain = rx_bf.T @ rx_pv
        rx_gain = np.transpose(rx_gain, (2, 1, 0))
        cir_tg = cir - np.mean(cir, axis=1, keepdims=True)
        cir_tg = np.repeat(cir_tg.T[:, :, None], scanvector.shape[0], axis=2) * rx_gain
        rx_power_per_beam = np.sum(np.abs(cir_tg), axis=(0, 1))
        if aoa_az_los.size:
            aoa_az_val = aoa_az_los[0]
            if abs(aoa_az_val - 90) < 1e-6:
                rx_power_per_beam[np.sin(np.deg2rad(scanvector[:, 0])) < 0] = 0
            elif abs(aoa_az_val + 90) < 1e-6:
                rx_power_per_beam[np.sin(np.deg2rad(scanvector[:, 0])) > 0] = 0
            else:
                if np.cos(np.deg2rad(aoa_az_val)) > np.finfo(float).eps:
                    rx_power_per_beam[np.cos(np.deg2rad(scanvector[:, 0])) < 0] = 0
                else:
                    rx_power_per_beam[np.cos(np.deg2rad(scanvector[:, 0])) > 0] = 0
        idx = np.argmax(rx_power_per_beam)
        angle_estimate = scanvector[idx]
    elif angle_estimation == "nearest":
        tgt = np.array([aoa_az_sorted[0], aoa_el_sorted[0]])
        idx = np.argmin(np.sum(np.abs(scanvector - tgt), axis=1))
        angle_estimate = scanvector[idx]
    else:
        angle_estimate = np.array([aoa_az_los.mean() if aoa_az_los.size else 0.0, aoa_el_los.mean() if aoa_el_los.size else 0.0])

    tx_bf = np.sum(np.conj(steering_vector(fc, angle_estimate.reshape(2, 1), transmit_array)), axis=1)
    tx_bf = tx_bf / max(1, len(np.atleast_1d(aoa_az_los)))
    rx_bf = np.sum(np.conj(steering_vector(fc, angle_estimate.reshape(2, 1), receive_array)), axis=1)
    rx_bf = rx_bf / max(1, len(np.atleast_1d(aoa_az_los)))

    tx_gain = tx_pv.T @ tx_bf
    rx_gain = rx_bf.T @ rx_pv
    cir_beam = (cir.T * tx_gain).T
    cir_beam = (cir_beam.T * rx_gain).T

    a = cir.T[:, None, :]
    b = tx_gain[None, None, :]
    c_gain = rx_pv[None, :, :]
    cir_antenna_port = np.transpose(a * b * c_gain, (2, 0, 1))

    if sample_rate:
        cir_int = sinc_interp(delays_sorted - delays_sorted[0], cir_beam, time_sampling, sample_rate)
        out_cir = cir_int
        out_cir_ant = np.zeros((n_samples, n_realization, receive_array.num_elements), dtype=complex)
        for i in range(receive_array.num_elements):
            cir_int_ant = sinc_interp(delays_sorted - delays_sorted[0], cir_antenna_port[:, :, i], time_sampling, sample_rate)
            out_cir_ant[:, :, i] = cir_int_ant
    else:
        out_cir = cir_beam
        out_cir_ant = cir_antenna_port

    sync_offset = delays_sorted[0] * c / 2

    if has_los_list:
        los = np.concatenate([np.atleast_1d(x) for x in has_los_list])
    else:
        los = np.zeros(0)

    out = SensingCdlOutput(
        delay=delays_sorted,
        path_gain=out_cir,
        aoa_az=aoa_az_sorted,
        aoa_el=aoa_el_sorted,
        aod_az=aod_az_sorted,
        aod_el=aod_el_sorted,
        angle_estimate=angle_estimate,
        los=los,
        sync_offset=sync_offset,
        tgt_pg=tgt_pg,
    )
    return out, out_cir_ant, time_sampling
