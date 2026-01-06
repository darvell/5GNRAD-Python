from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from gnrad5.antenna.array import URA


@dataclass(frozen=True)
class TypeICodebook:
    id: str
    family: str
    rank: int
    mode: int
    n_g: int
    n1: int
    n2: int
    o1: int
    o2: int
    precoders: np.ndarray  # (n_entries, n_ports, rank)
    beam_angles: np.ndarray  # (n_entries, 2)


_VLM_CACHE: dict[tuple[int, int, int, int], np.ndarray] = {}


def build_type1_codebook_registry(
    p_csi_rs: int,
    include_multi_panel: bool = True,
    ranks: Iterable[int] = (1, 2, 3, 4),
    modes: Iterable[int] = (1, 2),
):
    registry: dict[str, TypeICodebook] = {}
    for rank in ranks:
        for mode in modes:
            for n1, n2, o1, o2 in _single_panel_configs(p_csi_rs):
                cb = _build_type1_single_panel(n1, n2, o1, o2, rank=rank, mode=mode)
                registry[cb.id] = cb
            if include_multi_panel:
                for n_g, n1, n2, o1, o2 in _multi_panel_configs(p_csi_rs):
                    if n_g == 4 and mode != 1:
                        continue
                    cb = _build_type1_multi_panel(n_g, n1, n2, o1, o2, rank=rank, mode=mode)
                    registry[cb.id] = cb
    return registry


def default_type1_codebook_id(p_csi_rs: int, rank: int = 1, mode: int = 1) -> str:
    n1, n2, o1, o2 = _single_panel_configs(p_csi_rs)[0]
    return _codebook_id("typeI-single-panel", rank, mode, 1, n1, n2, o1, o2)


def _single_panel_configs(p_csi_rs: int):
    # TS 38.214 Table 5.2.2.2.1-2 (subset for supported 32-port case)
    if p_csi_rs == 32:
        return [
            (4, 4, 4, 4),
            (8, 2, 4, 4),
            (16, 1, 4, 1),
        ]
    raise ValueError(f"Unsupported CSI-RS port count {p_csi_rs}; expected 32.")


def _multi_panel_configs(p_csi_rs: int):
    # TS 38.214 Table 5.2.2.2.2-1 (32-port entries)
    configs: list[tuple[int, int, int, int, int]] = []
    if p_csi_rs != 32:
        return configs
    # (N_g, N1, N2) with (O1, O2)
    configs.extend(
        [
            (2, 8, 1, 4, 1),
            (4, 4, 1, 4, 1),
            (2, 4, 2, 4, 4),
            (4, 2, 2, 4, 4),
        ]
    )
    return configs


def _codebook_id(family: str, rank: int, mode: int, n_g: int, n1: int, n2: int, o1: int, o2: int) -> str:
    return (
        f"{family}-32p-n_g={n_g}-n1={n1}-n2={n2}-o1={o1}-o2={o2}-mode={mode}-ri={rank}"
    )


def _phi_set() -> np.ndarray:
    # ϕ_n = exp(j π n / 2), n = 0..3
    return np.exp(1j * np.pi * np.arange(4) / 2)


def _theta_set() -> np.ndarray:
    # θ_p = exp(j π p / 4), p = 0..3
    return np.exp(1j * np.pi * np.arange(4) / 4)


def _dft_vector(n: int, o: int, idx: int) -> np.ndarray:
    if n == 1:
        return np.ones(1, dtype=np.complex128)
    k = np.arange(n)
    return np.exp(1j * 2 * np.pi * idx * k / (o * n)) / np.sqrt(n)


def _v_lm(n1: int, n2: int, o1: int, o2: int, l_idx: int, m_idx: int) -> np.ndarray:
    key = (n1, n2, o1, o2)
    table = _VLM_CACHE.get(key)
    if table is None:
        l_count = n1 * o1
        m_count = n2 * o2
        v_l = np.array([_dft_vector(n1, o1, l) for l in range(l_count)])
        u_m = np.array([_dft_vector(n2, o2, m) for m in range(m_count)])
        table = (v_l[:, None, :, None] * u_m[None, :, None, :]).reshape(l_count, m_count, n1 * n2)
        _VLM_CACHE[key] = table
    return table[l_idx, m_idx]


def _beam_angles_for_grid(n1: int, n2: int, o1: int, o2: int) -> np.ndarray:
    grid_v = n1 * o1
    grid_h = n2 * o2
    k_v = np.arange(grid_v)
    k_h = np.arange(grid_h)
    uz = 2 * k_v / grid_v
    uy = 2 * k_h / grid_h
    uz = (uz + 1) % 2 - 1
    uy = (uy + 1) % 2 - 1
    uz_grid, uy_grid = np.meshgrid(uz, uy, indexing="ij")
    el_grid = np.rad2deg(np.arcsin(np.clip(uz_grid, -1.0, 1.0)))
    az_grid = np.rad2deg(
        np.arctan2(uy_grid, np.sqrt(np.maximum(0.0, 1.0 - uz_grid**2 - uy_grid**2)))
    )
    return np.column_stack([az_grid.ravel(), el_grid.ravel()])


def _build_type1_single_panel(n1: int, n2: int, o1: int, o2: int, rank: int, mode: int) -> TypeICodebook:
    precoders, beam_angles = _build_type1_precoders(n1, n2, o1, o2, rank, mode, n_g=1, multi_panel=False)
    codebook_id = _codebook_id("typeI-single-panel", rank, mode, 1, n1, n2, o1, o2)
    return TypeICodebook(
        id=codebook_id,
        family="typeI-single-panel",
        rank=rank,
        mode=mode,
        n_g=1,
        n1=n1,
        n2=n2,
        o1=o1,
        o2=o2,
        precoders=precoders,
        beam_angles=beam_angles,
    )


def _build_type1_multi_panel(n_g: int, n1: int, n2: int, o1: int, o2: int, rank: int, mode: int) -> TypeICodebook:
    precoders, beam_angles = _build_type1_precoders(
        n1, n2, o1, o2, rank, mode, n_g=n_g, multi_panel=True
    )
    codebook_id = _codebook_id("typeI-multi-panel", rank, mode, n_g, n1, n2, o1, o2)
    return TypeICodebook(
        id=codebook_id,
        family="typeI-multi-panel",
        rank=rank,
        mode=mode,
        n_g=n_g,
        n1=n1,
        n2=n2,
        o1=o1,
        o2=o2,
        precoders=precoders,
        beam_angles=beam_angles,
    )


def _panel_grid(n_g: int) -> tuple[int, int]:
    if n_g == 1:
        return (1, 1)
    if n_g == 2:
        return (2, 1)
    if n_g == 4:
        return (2, 2)
    raise ValueError(f"Unsupported N_g={n_g}; expected 1,2,4")


def _panel_phase_terms(p_idx: int, n_idx: int) -> tuple[np.ndarray, np.ndarray]:
    # a_p = exp(jπ/4) * exp(jπ p/2)
    # b_n = exp(-jπ/4) * exp(jπ n/2)
    a_p = np.exp(1j * np.pi / 4) * np.exp(1j * np.pi * p_idx / 2)
    b_n = np.exp(-1j * np.pi / 4) * np.exp(1j * np.pi * n_idx / 2)
    return a_p, b_n


def _panel_vector(n_g: int, p_idx: int, n_idx: int) -> np.ndarray:
    n_g1, n_g2 = _panel_grid(n_g)
    a_p, b_n = _panel_phase_terms(p_idx, n_idx)
    phases = []
    for g1 in range(n_g1):
        for g2 in range(n_g2):
            phases.append((a_p**g1) * (b_n**g2))
    phases = np.asarray(phases, dtype=np.complex128)
    return phases / np.sqrt(len(phases))


def _build_type1_precoders(
    n1: int,
    n2: int,
    o1: int,
    o2: int,
    rank: int,
    mode: int,
    n_g: int,
    multi_panel: bool,
):
    phi = _phi_set()
    theta = _theta_set()
    l_vals = range(n1 * o1)
    m_vals = range(n2 * o2)
    base_angles = _beam_angles_for_grid(n1, n2, o1, o2)

    precoders: list[np.ndarray] = []
    beam_angles: list[np.ndarray] = []
    rank = int(rank)
    if rank < 1 or rank > 4:
        raise ValueError("Type I codebook supports rank 1..4")

    phase_set = theta if (mode == 2 and n2 > 1) else np.array([1.0 + 0j])
    panel_indices = range(n_g) if multi_panel else (0,)
    scale = 1.0 / np.sqrt(rank)
    signs = _sign_pattern(rank)
    offsets = _rank_offsets(rank, n1, n2, o1, o2, multi_panel=multi_panel, n_g=n_g)

    for l_idx in l_vals:
        for m_idx in m_vals:
            base = _v_lm(n1, n2, o1, o2, l_idx, m_idx)
            base_angle = base_angles[l_idx * (n2 * o2) + m_idx]
            for n_idx, phi_n in enumerate(phi):
                for theta_p in phase_set:
                    for p_idx in panel_indices:
                        panel_vec = _panel_vector(n_g, p_idx, n_idx)
                        base_panel = np.kron(panel_vec, base) if multi_panel else base
                        for k1, k2 in offsets:
                            cols = _build_rank_columns(
                                base_panel,
                                l_idx,
                                m_idx,
                                n1,
                                n2,
                                o1,
                                o2,
                                k1,
                                k2,
                                rank,
                                panel_vec=panel_vec if multi_panel else None,
                            )
                            upper = np.column_stack(cols)
                            lower = phi_n * theta_p * upper * signs
                            prec = np.vstack([upper, lower]) * scale
                            precoders.append(prec)
                            beam_angles.append(base_angle)

    return np.asarray(precoders), np.asarray(beam_angles)


def _build_rank_columns(
    base: np.ndarray,
    l_idx: int,
    m_idx: int,
    n1: int,
    n2: int,
    o1: int,
    o2: int,
    k1: int,
    k2: int,
    rank: int,
    panel_vec: np.ndarray | None = None,
):
    def make_vec(l_val: int, m_val: int) -> np.ndarray:
        v = _v_lm(n1, n2, o1, o2, l_val, m_val)
        if panel_vec is not None:
            return np.kron(panel_vec, v)
        return v

    cols = [base]
    if rank == 1:
        return cols
    l2 = (l_idx + k1) % (n1 * o1) if n1 * o1 > 0 else 0
    m2 = (m_idx + k2) % (n2 * o2) if n2 * o2 > 0 else 0
    v2 = make_vec(l2, m2)
    if rank == 2:
        return [base, v2]

    # rank 3/4: split offsets along each dimension when possible
    l3 = (l_idx + k1) % (n1 * o1) if n1 * o1 > 0 else 0
    m3 = m_idx
    l4 = l_idx
    m4 = (m_idx + k2) % (n2 * o2) if n2 * o2 > 0 else 0
    v3 = make_vec(l3, m3)
    v4 = make_vec(l4, m4)
    if rank == 3:
        return [base, v3, v4]

    v5 = make_vec(l2, m2)
    return [base, v3, v4, v5]


def _sign_pattern(rank: int) -> np.ndarray:
    if rank == 1:
        return np.ones(1)
    if rank == 2:
        return np.array([1.0, -1.0])
    if rank == 3:
        return np.array([1.0, 1.0, -1.0])
    if rank == 4:
        return np.array([1.0, 1.0, -1.0, -1.0])
    raise ValueError("Unsupported rank")


def _rank_offsets(
    rank: int,
    n1: int,
    n2: int,
    o1: int,
    o2: int,
    multi_panel: bool,
    n_g: int,
) -> list[tuple[int, int]]:
    if rank == 1:
        return [(0, 0)]
    if rank == 2:
        return _rank2_offsets(n1, n2, o1, o2)
    if multi_panel and n_g > 1:
        return _rank34_offsets_multi(n1, n2, o1, o2)
    return _rank34_offsets(n1, n2, o1, o2)


def _rank2_offsets(n1: int, n2: int, o1: int, o2: int) -> list[tuple[int, int]]:
    case = "n1_gt_n2"
    if n2 == 1:
        case = "n1_eq_2_n2_1" if n1 == 2 else "n1_gt_2_n2_1"
    elif n1 == n2:
        case = "n1_eq_n2"

    table = {
        "n1_gt_n2": [(0, 0), (o1, 0), (0, o2), (2 * o1, 0)],
        "n1_eq_n2": [(0, 0), (o1, 0), (0, o2), (o1, o2)],
        "n1_eq_2_n2_1": [(0, 0), (o1, 0)],
        "n1_gt_2_n2_1": [(0, 0), (o1, 0), (2 * o1, 0), (3 * o1, 0)],
    }
    return table[case]


def _rank34_offsets(n1: int, n2: int, o1: int, o2: int) -> list[tuple[int, int]]:
    key = (n1, n2)
    table = {
        (2, 1): [(o1, 0)],
        (4, 1): [(o1, 0), (2 * o1, 0), (3 * o1, 0)],
        (6, 1): [(o1, 0), (2 * o1, 0), (3 * o1, 0), (4 * o1, 0)],
        (2, 2): [(o1, 0), (0, o2), (o1, o2)],
        (3, 2): [(o1, 0), (0, o2), (o1, o2), (2 * o1, 0)],
    }
    if key in table:
        return table[key]
    return _rank2_offsets(n1, n2, o1, o2)


def _rank34_offsets_multi(n1: int, n2: int, o1: int, o2: int) -> list[tuple[int, int]]:
    key = (n1, n2)
    table = {
        (2, 1): [(o1, 0)],
        (4, 1): [(o1, 0), (2 * o1, 0), (3 * o1, 0)],
        (8, 1): [(o1, 0), (2 * o1, 0), (3 * o1, 0), (4 * o1, 0)],
        (2, 2): [(o1, 0), (0, o2), (o1, o2)],
        (4, 2): [(o1, 0), (0, o2), (o1, o2), (2 * o1, 0)],
    }
    if key in table:
        return table[key]
    return _rank2_offsets(n1, n2, o1, o2)


def expand_codebook_to_elements(
    array: URA,
    codebook: TypeICodebook,
    port_n1: int,
    port_n2: int,
):
    n_ant = array.num_elements
    if codebook.family == "typeI-multi-panel":
        n_g1, n_g2 = _panel_grid(codebook.n_g)
        expected_port_n1 = codebook.n1 * n_g1
        expected_port_n2 = codebook.n2 * n_g2
    else:
        n_g1, n_g2 = (1, 1)
        expected_port_n1 = codebook.n1
        expected_port_n2 = codebook.n2

    if port_n1 != expected_port_n1 or port_n2 != expected_port_n2:
        raise ValueError("Port grid does not match codebook configuration")
    if array.shape[0] % port_n1 != 0 or array.shape[1] % port_n2 != 0:
        raise ValueError("Array dimensions must be divisible by port grid")
    v_block = array.shape[0] // port_n1
    h_block = array.shape[1] // port_n2
    n_ports = port_n1 * port_n2
    n_ports_panel = codebook.n1 * codebook.n2
    n_panels = codebook.n_g if codebook.family == "typeI-multi-panel" else 1

    # Collapse dual-pol ports onto single-pol physical subarrays.
    prec = codebook.precoders
    if prec.shape[1] == 2 * n_ports:
        if codebook.family == "typeI-multi-panel" and n_panels > 1:
            panel_span = 2 * n_ports_panel
            pol0_list = []
            pol1_list = []
            for g in range(n_panels):
                start = g * panel_span
                pol0_list.append(prec[:, start : start + n_ports_panel, :])
                pol1_list.append(prec[:, start + n_ports_panel : start + panel_span, :])
            pol0 = np.concatenate(pol0_list, axis=1)
            pol1 = np.concatenate(pol1_list, axis=1)
        else:
            pol0 = prec[:, :n_ports, :]
            pol1 = prec[:, n_ports:, :]

        combined = (pol0 + pol1) / np.sqrt(2)
        energy = np.linalg.norm(combined, axis=1)
        if np.any(energy <= np.finfo(float).eps):
            mask = (energy <= np.finfo(float).eps)[:, None, :]
            combined = np.where(mask, pol0, combined)
        prec = combined
    elif prec.shape[1] != n_ports:
        raise ValueError("Port count mismatch for codebook expansion")

    if codebook.family == "typeI-multi-panel" and n_panels > 1:
        idx_map: list[int] = []
        for v_global in range(codebook.n1 * n_g1):
            g1, v_local = divmod(v_global, codebook.n1)
            for h_global in range(codebook.n2 * n_g2):
                g2, h_local = divmod(h_global, codebook.n2)
                p_panel = (g1 * n_g2 + g2) * n_ports_panel + v_local * codebook.n2 + h_local
                idx_map.append(p_panel)
        prec = prec[:, np.asarray(idx_map, dtype=int), :]

    weights = np.zeros((prec.shape[0], n_ant, prec.shape[2]), dtype=np.complex128)
    for p_idx in range(n_ports):
        v_start = (p_idx // port_n2) * v_block
        h_start = (p_idx % port_n2) * h_block
        rows = np.arange(v_start, v_start + v_block)
        cols = np.arange(h_start, h_start + h_block)
        grid = np.array([r * array.shape[1] + c for r in rows for c in cols])
        weights[:, grid, :] = prec[:, p_idx : p_idx + 1, :] / np.sqrt(len(grid))
    return weights
