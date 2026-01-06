from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gnrad5.antenna import (
    URA,
    beam_grid,
    build_type1_codebook_registry,
    default_type1_codebook_id,
    expand_codebook_to_elements,
    steering_vector,
)
from gnrad5.channel import compute_path_loss, get_sigma_rcs, load_background_channel
from gnrad5.constants import SPEED_OF_LIGHT
from gnrad5.detection import par_detector
from gnrad5.geometry import wrap_to_180
from gnrad5.signal import build_prs_grid, get_ofdm_params


@dataclass
class PaperDetection:
    detected: bool
    best_par_db: float
    best_range_idx: int
    best_range_m: float
    best_az: float
    best_el: float
    best_pos: np.ndarray | None
    range_bins: np.ndarray
    par_db: np.ndarray
    beam_grid: np.ndarray
    beam_scores: np.ndarray


_CODEBOOK_CACHE: dict[tuple[int, bool], dict[str, object]] = {}
_BACKGROUND_CACHE: dict[tuple[str, float, str], object] = {}
_GRID_BEAM_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
_CODEBOOK_WEIGHT_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
_VIRTUAL_RX_PARAMS = {
    "UMaAV": {"alpha": 0.8548, "beta": 75.9277, "mu": -0.2150, "sigma": 32.0331},
    "UMiAV": {"alpha": 0.8166, "beta": 22.1965, "mu": -0.4384, "sigma": 33.7821},
}


def _get_codebook_registry(p_csi_rs: int) -> dict[str, object]:
    key = (int(p_csi_rs), True)
    cached = _CODEBOOK_CACHE.get(key)
    if cached is None:
        cached = build_type1_codebook_registry(int(p_csi_rs), include_multi_panel=True, ranks=(1, 2, 3, 4))
        _CODEBOOK_CACHE[key] = cached
    return cached


def _get_background_channel(repo_root: Path, scenario_hint: str, fc_hz: float):
    key = (scenario_hint, float(fc_hz), str(repo_root))
    cached = _BACKGROUND_CACHE.get(key)
    if cached is None:
        cached = load_background_channel(repo_root, scenario_hint, fc_hz)
        _BACKGROUND_CACHE[key] = cached
    return cached


def _get_grid_beams(
    fc: float,
    array: URA,
    az_limits: tuple[float, float],
    el_limits: tuple[float, float],
    beam_az: int,
    beam_el: int,
) -> tuple[np.ndarray, np.ndarray]:
    key = (
        "grid",
        float(fc),
        tuple(array.shape),
        float(array.element_spacing),
        az_limits,
        el_limits,
        int(beam_az),
        int(beam_el),
    )
    cached = _GRID_BEAM_CACHE.get(key)
    if cached is not None:
        return cached
    beams = beam_grid(az_limits[0], az_limits[1], el_limits[0], el_limits[1], beam_az, beam_el)
    w_mat = steering_vector(fc, beams.T, array) / np.sqrt(array.num_elements)
    _GRID_BEAM_CACHE[key] = (beams, w_mat)
    return beams, w_mat


def _get_codebook_weights(
    fc: float,
    array: URA,
    cb,
    port_n1: int,
    port_n2: int,
    v_block: int,
    h_block: int,
) -> tuple[np.ndarray, np.ndarray]:
    key = (
        "type1",
        cb.id,
        float(fc),
        tuple(array.shape),
        float(array.element_spacing),
        int(port_n1),
        int(port_n2),
    )
    cached = _CODEBOOK_WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached
    weights = expand_codebook_to_elements(array, cb, port_n1=port_n1, port_n2=port_n2)
    beams = _rescale_beam_angles(cb.beam_angles, v_block, h_block)
    _CODEBOOK_WEIGHT_CACHE[key] = (beams, weights)
    return beams, weights


def _type1_panel_grid(n_g: int) -> tuple[int, int]:
    if n_g == 1:
        return (1, 1)
    if n_g == 2:
        return (2, 1)
    if n_g == 4:
        return (2, 2)
    raise ValueError(f"Unsupported N_g={n_g}; expected 1,2,4")


def _type1_port_grid(codebook) -> tuple[int, int]:
    if codebook.family == "typeI-multi-panel":
        n_g1, n_g2 = _type1_panel_grid(codebook.n_g)
        return codebook.n1 * n_g1, codebook.n2 * n_g2
    return codebook.n1, codebook.n2


def _rescale_beam_angles(beam_angles: np.ndarray, v_block: int, h_block: int) -> np.ndarray:
    if v_block <= 0 or h_block <= 0:
        raise ValueError("Beam rescale blocks must be positive")
    if v_block == 1 and h_block == 1:
        return beam_angles
    az = np.deg2rad(beam_angles[:, 0])
    el = np.deg2rad(beam_angles[:, 1])
    uz = np.sin(el) / v_block
    uy = np.cos(el) * np.sin(az) / h_block
    el_new = np.rad2deg(np.arcsin(np.clip(uz, -1.0, 1.0)))
    az_new = np.rad2deg(
        np.arctan2(uy, np.sqrt(np.maximum(0.0, 1.0 - uz**2 - uy**2)))
    )
    return np.column_stack([az_new, el_new])


def run_paper_chain(
    sim,
    prs,
    sens,
    geometry,
    target,
    repo_root: str | Path | None = None,
    seed: int | None = None,
    background_channel: dict | None = None,
    target_absent: bool = False,
    fixed_rcs_dbsm: float | None = None,
    fixed_nrp: int | None = None,
    eirp_dbm: float = 75.0,
    eta_db: float = 3.4,
    codebook_id: str | None = None,
    p_csi_rs: int = 32,
    beam_chunk: int | None = None,
    prs_period_slots: int | None = None,
    beam_az: int = 25,
    beam_el: int = 13,
    az_limits: tuple[float, float] = (-60.0, 60.0),
    el_limits: tuple[float, float] = (-30.0, 30.0),
    add_noise: bool = True,
    no_channel: bool = False,
    max_symbols: int | None = None,
    max_range_bins: int | None = None,
):
    det, _ = _run_paper_chain_impl(
        sim,
        prs,
        sens,
        geometry,
        target,
        repo_root=repo_root,
        seed=seed,
        background_channel=background_channel,
        target_absent=target_absent,
        fixed_rcs_dbsm=fixed_rcs_dbsm,
        fixed_nrp=fixed_nrp,
        eirp_dbm=eirp_dbm,
        eta_db=eta_db,
        codebook_id=codebook_id,
        p_csi_rs=p_csi_rs,
        beam_chunk=beam_chunk,
        prs_period_slots=prs_period_slots,
        beam_az=beam_az,
        beam_el=beam_el,
        az_limits=az_limits,
        el_limits=el_limits,
        add_noise=add_noise,
        no_channel=no_channel,
        max_symbols=max_symbols,
        max_range_bins=max_range_bins,
        return_debug=False,
    )
    return det


def run_paper_chain_debug(
    sim,
    prs,
    sens,
    geometry,
    target,
    repo_root: str | Path | None = None,
    seed: int | None = None,
    background_channel: dict | None = None,
    target_absent: bool = False,
    fixed_rcs_dbsm: float | None = None,
    fixed_nrp: int | None = None,
    eirp_dbm: float = 75.0,
    eta_db: float = 3.4,
    codebook_id: str | None = None,
    p_csi_rs: int = 32,
    beam_chunk: int | None = None,
    prs_period_slots: int | None = None,
    beam_az: int = 25,
    beam_el: int = 13,
    az_limits: tuple[float, float] = (-60.0, 60.0),
    el_limits: tuple[float, float] = (-30.0, 30.0),
    add_noise: bool = True,
    no_channel: bool = False,
    max_symbols: int | None = None,
    max_range_bins: int | None = None,
) -> tuple[PaperDetection, dict[str, Any]]:
    return _run_paper_chain_impl(
        sim,
        prs,
        sens,
        geometry,
        target,
        repo_root=repo_root,
        seed=seed,
        background_channel=background_channel,
        target_absent=target_absent,
        fixed_rcs_dbsm=fixed_rcs_dbsm,
        fixed_nrp=fixed_nrp,
        eirp_dbm=eirp_dbm,
        eta_db=eta_db,
        codebook_id=codebook_id,
        p_csi_rs=p_csi_rs,
        beam_chunk=beam_chunk,
        prs_period_slots=prs_period_slots,
        beam_az=beam_az,
        beam_el=beam_el,
        az_limits=az_limits,
        el_limits=el_limits,
        add_noise=add_noise,
        no_channel=no_channel,
        max_symbols=max_symbols,
        max_range_bins=max_range_bins,
        return_debug=True,
    )


def _run_paper_chain_impl(
    sim,
    prs,
    sens,
    geometry,
    target,
    repo_root: str | Path | None = None,
    seed: int | None = None,
    background_channel: dict | None = None,
    target_absent: bool = False,
    fixed_rcs_dbsm: float | None = None,
    fixed_nrp: int | None = None,
    eirp_dbm: float = 75.0,
    eta_db: float = 3.4,
    codebook_id: str | None = None,
    p_csi_rs: int = 32,
    beam_chunk: int | None = None,
    prs_period_slots: int | None = None,
    beam_az: int = 25,
    beam_el: int = 13,
    az_limits: tuple[float, float] = (-60.0, 60.0),
    el_limits: tuple[float, float] = (-30.0, 30.0),
    add_noise: bool = True,
    no_channel: bool = False,
    max_symbols: int | None = None,
    max_range_bins: int | None = None,
    return_debug: bool = False,
) -> tuple[PaperDetection, dict[str, Any]]:
    debug: dict[str, Any] = {}
    repo_root = Path(repo_root or Path.cwd())
    rng = np.random.default_rng(seed)

    if background_channel is None:
        scenario_hint = "UMi" if sim.channel_scenario == "UMiAV" else "UMa"
        background_channel = _get_background_channel(repo_root, scenario_hint, sim.system_fc)
        if return_debug:
            debug["background_source"] = "cache"
    elif return_debug:
        debug["background_source"] = "override"

    prs_grid = build_prs_grid(sim, prs, sens.number_sensing_symbols)
    comb = max(1, int(prs.comb_size))
    prs_scale = np.sqrt(comb)
    prs_grid_full = prs_grid.grid * prs_scale
    ofdm_params = get_ofdm_params(prs_grid.nr_objects.ofdm_info, prs_grid.nr_objects.carrier)
    nfft = ofdm_params.nfft
    symbol_indices = prs_grid.symbol_indices
    if max_symbols is not None:
        symbol_indices = symbol_indices[: max_symbols]

    if symbol_indices.size:
        sym0 = int(symbol_indices[0])
        sym_window = symbol_indices - sym0
        num_symbols_total = int(sym_window[-1]) + 1
    else:
        sym0 = 0
        sym_window = symbol_indices
        num_symbols_total = 0

    tx_grid_window = prs_grid_full[:, sym0 : sym0 + num_symbols_total]
    tx_grid_prs = prs_grid_full[:, symbol_indices]
    symbols_per_slot = int(getattr(prs_grid.nr_objects.carrier, "SymbolsPerSlot", 14))

    c = SPEED_OF_LIGHT
    scs_hz = float(sim.carrier_subcarrier_spacing) * 1e3
    number_subcarriers = int(sim.carrier_n_size_grid) * 12
    start_idx = (nfft - number_subcarriers) // 2
    end_idx = start_idx + number_subcarriers
    sub_idx = np.arange(start_idx, end_idx)
    freqs = (sub_idx - nfft / 2) * scs_hz

    sample_rate = ofdm_params.sample_rate
    ofdm_info = prs_grid.nr_objects.ofdm_info
    ofdm_symbol_lengths = getattr(ofdm_info, "SymbolLengths", None) if not isinstance(ofdm_info, dict) else None
    if ofdm_symbol_lengths is not None and sample_rate:
        ofdm_ts = float(np.mean(np.asarray(ofdm_symbol_lengths)) / sample_rate)
        slot_duration = float(np.sum(np.asarray(ofdm_symbol_lengths)) / sample_rate)
    else:
        ofdm_ts = 1.0 / max(1, sens.number_sensing_symbols)
        slot_duration = ofdm_ts * symbols_per_slot

    if sym_window.size:
        symbol_in_slot = np.mod(sym_window, symbols_per_slot)
        slot_idx = sym_window // symbols_per_slot
        if prs_period_slots is not None:
            base_period = int(getattr(prs, "prs_resource_set_period", [1])[0])
            desired_period = int(prs_period_slots)
            if base_period <= 0 or desired_period <= 0:
                raise ValueError("PRS periodicity slots must be positive")
            scale = desired_period / base_period
            time_vector = slot_idx * scale * slot_duration + symbol_in_slot * ofdm_ts
        else:
            time_vector = slot_idx * slot_duration + symbol_in_slot * ofdm_ts
    else:
        time_vector = sym_window.astype(float)
    if return_debug:
        debug["time_vector_s"] = time_vector
        debug["ofdm_ts_s"] = ofdm_ts
        debug["slot_duration_s"] = slot_duration

    range_resolution = 1.0
    if sample_rate:
        range_resolution = 1 / (2 * sample_rate) * c
    # PRS tones are spaced by K*Δf, so range bin spacing is scaled by 1/K.
    range_bins = np.arange(nfft) * range_resolution / comb
    if sim.max_range_interest:
        idx = np.argmax(range_bins > sim.max_range_interest)
        if idx > 0:
            range_bins = range_bins[:idx]
    if max_range_bins is not None:
        range_bins = range_bins[:max_range_bins]
    if return_debug:
        debug["range_bins_m"] = range_bins

    n_ant = sim.antenna_num_v * sim.antenna_num_h
    array = URA((sim.antenna_num_v, sim.antenna_num_h), (c / sim.system_fc) / 2)

    beams: np.ndarray
    if codebook_id == "grid":
        beams, w_mat = _get_grid_beams(
            sim.system_fc,
            array,
            az_limits,
            el_limits,
            beam_az,
            beam_el,
        )
    else:
        registry = _get_codebook_registry(p_csi_rs)
        if codebook_id is None:
            codebook_id = default_type1_codebook_id(p_csi_rs, rank=1, mode=1)
        if codebook_id not in registry:
            raise ValueError(f"Unknown codebook id: {codebook_id}")
        cb = registry[codebook_id]
        if cb.rank != 1:
            raise ValueError(
                f"Type I codebook rank {cb.rank} not supported in run_paper_chain; use rank-1"
            )
        port_n1, port_n2 = _type1_port_grid(cb)
        v_block = array.shape[0] // port_n1
        h_block = array.shape[1] // port_n2
        beams, weights = _get_codebook_weights(
            sim.system_fc,
            array,
            cb,
            port_n1,
            port_n2,
            v_block,
            h_block,
        )
        w_mat = weights[:, :, 0].T

    if return_debug:
        debug["beam_grid"] = beams
        debug["beam_grid_shape"] = (beam_el, beam_az) if codebook_id == "grid" else None

    if no_channel:
        res = par_detector(
            tx_grid_window,
            tx_grid_window,
            sym_window,
            sim,
            prs,
            eta_db=eta_db,
            window=str(sens.window).lower() if sens.window else "blackmanharris",
            symbols_per_slot=symbols_per_slot,
            clutter_mean=True,
            max_range_bins=len(range_bins),
        )
        par_db = np.full(beams.shape[0], res["par_db"], dtype=float)
        range_idx = np.full(beams.shape[0], res["range_idx"], dtype=int)
        beam_scores = np.zeros(beams.shape[0], dtype=float)
        det = _finalize_detection(
            beams,
            range_bins,
            par_db,
            range_idx,
            eta_db,
            beam_scores=beam_scores,
            best_idx=int(np.argmax(beam_scores)) if beam_scores.size else -1,
            bs_pos=geometry.tx[0] if geometry.tx else None,
        )
        if return_debug:
            debug["range_power"] = res.get("range_power")
        return det, debug

    delays, gains_lin, aoa_az, aoa_el, aod_az, aod_el, phase0, doppler = _collect_paths(
        background_channel,
        geometry,
        target,
        sim.system_fc,
        sim.channel_scenario,
        fixed_rcs_dbsm,
        fixed_nrp,
        target_absent,
        rng,
    )

    if return_debug:
        debug["paths"] = {
            "delays_s": delays,
            "gains_lin": gains_lin,
            "aoa_az_deg": aoa_az,
            "aoa_el_deg": aoa_el,
            "aod_az_deg": aod_az,
            "aod_el_deg": aod_el,
            "phase0": phase0,
            "doppler_hz": doppler,
        }

    if delays.size == 0:
        par_db = np.full(beams.shape[0], -np.inf, dtype=float)
        range_idx = np.full(beams.shape[0], -1, dtype=int)
        beam_scores = np.full(beams.shape[0], -np.inf, dtype=float)
        det = _finalize_detection(
            beams,
            range_bins,
            par_db,
            range_idx,
            eta_db,
            beam_scores=beam_scores,
            best_idx=-1,
            bs_pos=geometry.tx[0] if geometry.tx else None,
        )
        return det, debug

    a_tx = steering_vector(sim.system_fc, np.vstack([aod_az, aod_el]), array)
    a_rx = steering_vector(sim.system_fc, np.vstack([aoa_az, aoa_el]), array)

    tx_resp = w_mat.conj().T @ a_tx
    rx_resp = w_mat.conj().T @ a_rx

    p_tx_lin = 10 ** ((eirp_dbm - 30.0) / 10.0)
    p_tx_lin = p_tx_lin * max(0.0, float(sim.antenna_coupling_efficiency))
    amp = np.sqrt(np.maximum(gains_lin, 0.0)) * np.sqrt(p_tx_lin)
    phase = np.exp(1j * phase0)

    freq_phase = np.exp(-1j * 2 * np.pi * freqs[:, None] * delays[None, :])
    doppler_phase = np.exp(1j * 2 * np.pi * doppler[:, None] * time_vector[None, :])

    tx_grid_sub = tx_grid_prs[start_idx:end_idx, :]
    par_db = np.full(beams.shape[0], -np.inf, dtype=float)
    range_idx = np.full(beams.shape[0], -1, dtype=int)
    beam_scores_cs = np.full(beams.shape[0], -np.inf, dtype=float)
    beam_scores_raw = np.full(beams.shape[0], -np.inf, dtype=float)
    best_idx_cs = -1
    best_score_cs = -np.inf
    best_h_sub_cs = None
    best_idx_raw = -1
    best_score_raw = -np.inf
    best_h_sub_raw = None

    noise_power = 0.0
    if add_noise:
        k = 1.380649e-23
        t_k = 297.0
        nf = 10 ** (sim.system_nf / 10)
        noise_power = k * t_k * sim.system_bw * nf

    if beam_chunk is None:
        beam_chunk = 16
    beam_chunk = max(1, int(beam_chunk))

    n_beams = beams.shape[0]
    if n_beams:
        path_weight_base = amp * phase
        for start in range(0, n_beams, beam_chunk):
            end = min(start + beam_chunk, n_beams)
            # For monostatic sensing, the effective beamformed channel uses
            # (w^H a_rx) (a_tx^H f). With w=f from the same codebook,
            # a_tx^H f = conj(f^H a_tx).
            beam_gain = rx_resp[start:end] * np.conj(tx_resp[start:end])
            path_weight = path_weight_base[None, :] * beam_gain
            path_weight_time = path_weight[:, :, None] * doppler_phase[None, :, :]
            h_sub = np.einsum("sp,bpt->bst", freq_phase, path_weight_time, optimize=True)

            if h_sub.size:
                h_sub_cs = h_sub - h_sub.mean(axis=2, keepdims=True)
                score_cs = np.sum(np.abs(h_sub_cs) ** 2, axis=(1, 2))
                score_raw = np.sum(np.abs(h_sub) ** 2, axis=(1, 2))
            else:
                score_cs = np.full(end - start, -np.inf, dtype=float)
                score_raw = np.full(end - start, -np.inf, dtype=float)

            beam_scores_cs[start:end] = score_cs
            beam_scores_raw[start:end] = score_raw

            local_best_cs = int(np.argmax(score_cs)) if score_cs.size else -1
            if local_best_cs >= 0 and score_cs[local_best_cs] > best_score_cs:
                best_score_cs = float(score_cs[local_best_cs])
                best_idx_cs = start + local_best_cs
                best_h_sub_cs = h_sub[local_best_cs]

            local_best_raw = int(np.argmax(score_raw)) if score_raw.size else -1
            if local_best_raw >= 0 and score_raw[local_best_raw] > best_score_raw:
                best_score_raw = float(score_raw[local_best_raw])
                best_idx_raw = start + local_best_raw
                best_h_sub_raw = h_sub[local_best_raw]

    beam_scores = beam_scores_cs
    best_idx = best_idx_cs
    best_h_sub = best_h_sub_cs
    beam_score_mode = "cs"
    if beam_scores_cs.size:
        max_raw = float(np.max(beam_scores_raw))
        if np.isfinite(max_raw) and max_raw > 0.0:
            max_cs = float(np.max(beam_scores_cs))
            flat_cs = np.ptp(beam_scores_cs) <= np.finfo(float).eps * max_raw
            zeroed_cs = max_cs <= np.finfo(float).eps * max_raw
            if flat_cs or zeroed_cs:
                beam_scores = beam_scores_raw
                best_idx = best_idx_raw
                best_h_sub = best_h_sub_raw
                beam_score_mode = "raw"

    if return_debug:
        debug["beam_scores_cs"] = beam_scores_cs
        debug["beam_scores_raw"] = beam_scores_raw
        debug["beam_scores"] = beam_scores
        debug["beam_score_mode"] = beam_score_mode

    if best_idx >= 0 and best_h_sub is not None:
        rx_grid_sub = best_h_sub * tx_grid_sub
        if add_noise and noise_power > 0.0:
            noise = (
                rng.standard_normal(rx_grid_sub.shape) + 1j * rng.standard_normal(rx_grid_sub.shape)
            )
            rx_grid_sub = rx_grid_sub + np.sqrt(noise_power / 2) * noise

        beam_rx = np.zeros_like(tx_grid_window)
        beam_rx[start_idx:end_idx, sym_window] = rx_grid_sub
        res = par_detector(
            tx_grid_window,
            beam_rx,
            sym_window,
            sim,
            prs,
            eta_db=eta_db,
            window=str(sens.window).lower() if sens.window else "blackmanharris",
            symbols_per_slot=symbols_per_slot,
            clutter_mean=True,
            max_range_bins=len(range_bins),
        )
        par_db[best_idx] = res["par_db"]
        range_idx[best_idx] = res["range_idx"]
        if return_debug:
            debug["range_power"] = res.get("range_power")

    det = _finalize_detection(
        beams,
        range_bins,
        par_db,
        range_idx,
        eta_db,
        beam_scores=beam_scores,
        best_idx=best_idx,
        bs_pos=geometry.tx[0] if geometry.tx else None,
    )

    if return_debug:
        debug["best_idx"] = best_idx
        debug["detected"] = det.detected
        debug["best_par_db"] = det.best_par_db
        debug["best_range_m"] = det.best_range_m
        debug["best_az"] = det.best_az
        debug["best_el"] = det.best_el
        debug["best_pos"] = det.best_pos

    return det, debug


def _finalize_detection(
    beams,
    range_bins,
    par_db,
    range_idx,
    eta_db,
    beam_scores=None,
    best_idx: int | None = None,
    bs_pos=None,
):
    if best_idx is None:
        best_idx = int(np.argmax(par_db)) if par_db.size else -1
    if best_idx < 0:
        best_par_db = -np.inf
        best_range_idx = -1
        best_az = 0.0
        best_el = 0.0
    else:
        best_par_db = float(par_db[best_idx])
        best_range_idx = int(range_idx[best_idx])
        best_az = float(beams[best_idx, 0])
        best_el = float(beams[best_idx, 1])
    if best_range_idx >= 0 and range_bins.size:
        best_range_m = float(range_bins[best_range_idx])
    else:
        best_range_m = float("nan")

    best_pos = None
    if bs_pos is not None and best_range_idx >= 0 and np.isfinite(best_range_m):
        unit = _unit_vector(best_az, best_el)
        best_pos = np.asarray(bs_pos, dtype=float) + best_range_m * unit

    if beam_scores is None:
        beam_scores = np.zeros_like(par_db, dtype=float)
    return PaperDetection(
        detected=best_par_db >= float(eta_db),
        best_par_db=best_par_db,
        best_range_idx=best_range_idx,
        best_range_m=best_range_m,
        best_az=best_az,
        best_el=best_el,
        best_pos=best_pos,
        range_bins=range_bins,
        par_db=par_db,
        beam_grid=beams,
        beam_scores=beam_scores,
    )


def _collect_paths(
    background_channel,
    geometry,
    target,
    fc: float,
    scenario: str,
    fixed_rcs_dbsm: float | None,
    fixed_nrp: int | None,
    target_absent: bool,
    rng: np.random.Generator,
):
    delays = []
    gains_lin = []
    aoa_az = []
    aoa_el = []
    aod_az = []
    aod_el = []
    phase0 = []
    doppler = []

    (
        bg_delays,
        bg_gains,
        bg_aoa_az,
        bg_aoa_el,
        bg_aod_az,
        bg_aod_el,
        bg_phase0,
    ) = _merge_background_entries(background_channel, fixed_nrp, rng, scenario, fc, geometry)

    if bg_delays.size:
        delays.append(bg_delays)
        gains_lin.append(bg_gains)
        aoa_az.append(bg_aoa_az)
        aoa_el.append(bg_aoa_el)
        aod_az.append(bg_aod_az)
        aod_el.append(bg_aod_el)
        phase0.append(bg_phase0)
        doppler.append(np.zeros_like(bg_delays))

    if not target_absent and target.position:
        bs_pos = np.asarray(geometry.tx[0])
        tgt_pos = np.asarray(target.position[0])
        tgt_vel = np.asarray(target.velocity[0]) if target.velocity else np.zeros(3)
        c = SPEED_OF_LIGHT
        lamb = c / fc
        tgt_range = float(np.linalg.norm(tgt_pos - bs_pos))
        if tgt_range > 0:
            unit_vec = (tgt_pos - bs_pos) / tgt_range
        else:
            unit_vec = np.array([1.0, 0.0, 0.0])
        v_rel = float(np.dot(tgt_vel, unit_vec))
        f_d = 2 * v_rel / lamb

        pl, _ = compute_path_loss(scenario, fc, bs_pos, tgt_pos, False, rng)
        if fixed_rcs_dbsm is None:
            tgt_rcs = float(get_sigma_rcs("uav-small", return_large_scale=True, rng=rng))
        else:
            tgt_rcs = float(fixed_rcs_dbsm)
        tgt_pg_db = -(2 * pl - tgt_rcs + 10 * np.log10(c**2 / (4 * np.pi * fc**2)))
        tgt_gain_lin = 10 ** (tgt_pg_db / 10)

        az = np.rad2deg(np.arctan2(unit_vec[1], unit_vec[0]))
        el = np.rad2deg(np.arctan2(unit_vec[2], np.sqrt(unit_vec[0] ** 2 + unit_vec[1] ** 2)))

        delays.append(np.array([2 * tgt_range / c]))
        gains_lin.append(np.array([tgt_gain_lin]))
        aoa_az.append(np.array([az]))
        aoa_el.append(np.array([el]))
        aod_az.append(np.array([az]))
        aod_el.append(np.array([el]))
        phase0.append(np.array([0.0]))
        doppler.append(np.array([f_d]))

    if not delays:
        empty = np.zeros(0)
        return empty, empty, empty, empty, empty, empty, empty, empty

    return (
        np.concatenate(delays),
        np.concatenate(gains_lin),
        np.concatenate(aoa_az),
        np.concatenate(aoa_el),
        np.concatenate(aod_az),
        np.concatenate(aod_el),
        np.concatenate(phase0),
        np.concatenate(doppler),
    )


def _merge_background_entries(
    background_channel,
    fixed_nrp: int | None,
    rng: np.random.Generator,
    scenario: str | None,
    fc: float | None,
    geometry,
):
    entries: list[dict] = []
    if background_channel is None:
        entries = []
    elif isinstance(background_channel, list):
        if background_channel:
            n_rp = 1 if fixed_nrp is None or fixed_nrp <= 0 else int(fixed_nrp)
            n_rp = min(n_rp, len(background_channel))
            idx = rng.choice(len(background_channel), size=n_rp, replace=False)
            entries = [background_channel[int(i)] for i in np.atleast_1d(idx)]
    else:
        entries = [background_channel]

    if not entries:
        empty = np.zeros(0)
        return empty, empty, empty, empty, empty, empty, empty

    rp_positions = None
    rp_az_offsets = None
    if scenario in _VIRTUAL_RX_PARAMS and geometry is not None:
        bs_pos = np.asarray(geometry.tx[0]) if geometry.tx else np.zeros(3)
        rp_positions, rp_az_offsets = _sample_virtual_rx_positions(scenario, bs_pos, rng, len(entries))

    n_rp_eff = len(entries)
    scale = 1.0 / max(1, n_rp_eff)

    delays = []
    gains = []
    aoa_az = []
    aoa_el = []
    aod_az = []
    aod_el = []
    phase0 = []
    for idx, entry in enumerate(entries):
        d = np.asarray(entry.get("PathDelays", []), dtype=float)
        g = np.asarray(entry.get("AveragePathGains", []), dtype=float)
        if rp_positions is not None and d.size:
            ref_delay = float(np.min(d))
            if ref_delay > 0.0:
                r_new = float(np.linalg.norm(rp_positions[idx] - bs_pos))
                scale_delay = r_new / (SPEED_OF_LIGHT * ref_delay)
                if np.isfinite(scale_delay) and scale_delay > 0.0:
                    d = d * scale_delay
        if g.size:
            g = np.maximum(g, 0.0)
            power = float(np.sum(g))
            if power > 0.0:
                g = g / power
            if rp_positions is not None and fc is not None and scenario is not None:
                pl_db, _ = compute_path_loss(scenario, float(fc), bs_pos, rp_positions[idx], True, rng)
                g = g * (10 ** (-pl_db / 10.0))
            g = g * scale
        delays.append(d)
        gains.append(g)
        aoa_az_entry = wrap_to_180(np.asarray(entry.get("AnglesAoA", []), dtype=float))
        aoa_el.append(90 - np.asarray(entry.get("AnglesZoA", []), dtype=float))
        aod_az_entry = wrap_to_180(np.asarray(entry.get("AnglesAoD", []), dtype=float))
        aod_el.append(90 - np.asarray(entry.get("AnglesZoD", []), dtype=float))
        if rp_az_offsets is not None:
            offset = float(rp_az_offsets[idx])
            aoa_az_entry = wrap_to_180(aoa_az_entry + offset)
            aod_az_entry = wrap_to_180(aod_az_entry + offset)
        aoa_az.append(aoa_az_entry)
        aod_az.append(aod_az_entry)
        phase0.append(np.asarray(entry.get("InitialPhases", []), dtype=float))

    return (
        np.concatenate(delays),
        np.concatenate(gains),
        np.concatenate(aoa_az),
        np.concatenate(aoa_el),
        np.concatenate(aod_az),
        np.concatenate(aod_el),
        np.concatenate(phase0),
    )


def _sample_virtual_rx_positions(
    scenario: str,
    bs_pos: np.ndarray,
    rng: np.random.Generator,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    params = _VIRTUAL_RX_PARAMS[scenario]
    alpha = float(params["alpha"])
    beta = float(params["beta"])
    mu = float(params["mu"])
    sigma = float(params["sigma"])
    r = rng.gamma(alpha, beta, size=count)
    max_range = 4e3 if scenario == "UMiAV" else 5e3
    min_range = 10.0 + 1e-3
    r = np.clip(r, min_range, max_range)
    phi = rng.normal(mu, sigma, size=count)
    phi_rad = np.deg2rad(phi)
    x = bs_pos[0] + r * np.cos(phi_rad)
    y = bs_pos[1] + r * np.sin(phi_rad)
    z = np.full_like(x, max(1.5, bs_pos[2] - 1.0))
    positions = np.column_stack([x, y, z])
    return positions, phi


def _unit_vector(az_deg: float, el_deg: float) -> np.ndarray:
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)
    return np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)], dtype=float)
