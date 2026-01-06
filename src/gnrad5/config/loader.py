from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from .models import Geometry, PrsConfig, SensConfig, SimulationConfig, Target, apply_overrides


class ConfigLoadError(RuntimeError):
    pass


_NUM_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")


def _parse_scalar(token: str) -> Any:
    token = token.strip()
    if token == "":
        return token
    if token.lower() in {"true", "false"}:
        return token.lower() == "true"
    if _NUM_RE.match(token):
        val = float(token)
        if val.is_integer():
            return int(val)
        return val
    return token


def _parse_value(raw: str) -> Any:
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if inner == "":
            return []
        parts = re.split(r"[\s,]+", inner)
        return [_parse_scalar(p) for p in parts if p]
    if " " in raw or "," in raw:
        parts = re.split(r"[\s,]+", raw)
        return [_parse_scalar(p) for p in parts if p]
    return _parse_scalar(raw)


def _read_kv_table(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigLoadError(f"Missing config file: {path}")
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rows.append(line)
    if not rows:
        return {}
    data = {}
    for row in rows[1:]:
        parts = row.split("\t")
        if len(parts) < 2:
            continue
        key = parts[0].strip()
        value = _parse_value(parts[1])
        data[key] = value
    return data


def _read_matrix(path: Path) -> list[list[float]]:
    if not path.exists():
        raise ConfigLoadError(f"Missing matrix file: {path}")
    text = path.read_text().strip()
    if not text:
        return []
    delimiter = "," if "," in text else None
    matrix = np.loadtxt(path, delimiter=delimiter)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    return matrix.tolist()


def load_simulation_config(scenario_path: str | Path) -> SimulationConfig:
    path = Path(scenario_path) / "Input" / "simulationConfig.txt"
    raw = _read_kv_table(path)
    mapping = {
        "systemFc": "system_fc",
        "systemNF": "system_nf",
        "systemBw": "system_bw",
        "channelScenario": "channel_scenario",
        "antennaNumH": "antenna_num_h",
        "antennaNumV": "antenna_num_v",
        "antennaCouplingEfficiency": "antenna_coupling_efficiency",
        "carrierSubcarrierSpacing": "carrier_subcarrier_spacing",
        "carrierNSizeGrid": "carrier_n_size_grid",
        "nStDrop": "n_st_drop",
        "maxRangeInterest": "max_range_interest",
    }
    overrides = {mapping[k]: v for k, v in raw.items() if k in mapping}
    return apply_overrides(SimulationConfig(), overrides)


def load_prs_config(scenario_path: str | Path, sim_config: SimulationConfig) -> PrsConfig:
    path = Path(scenario_path) / "Input" / "prsConfig.txt"
    raw = _read_kv_table(path)
    mapping = {
        "PRSResourceSetPeriod": "prs_resource_set_period",
        "PRSResourceOffset": "prs_resource_offset",
        "PRSResourceRepetition": "prs_resource_repetition",
        "PRSResourceTimeGap": "prs_resource_time_gap",
        "NumRB": "num_rb",
        "RBOffset": "rb_offset",
        "CombSize": "comb_size",
        "REOffset": "re_offset",
        "NPRSID": "n_prs_id",
        "NumPRSSymbols": "num_prs_symbols",
        "SymbolStart": "symbol_start",
    }
    overrides = {mapping[k]: v for k, v in raw.items() if k in mapping}
    prs = apply_overrides(PrsConfig(), overrides)
    prs.num_rb = sim_config.carrier_n_size_grid
    return prs


def load_sens_config(scenario_path: str | Path) -> SensConfig:
    path = Path(scenario_path) / "Input" / "sensConfig.txt"
    raw = _read_kv_table(path)
    mapping = {
        "dopplerFftLen": "doppler_fft_len",
        "window": "window",
        "windowLen": "window_len",
        "windowOverlap": "window_overlap",
        "numberSensingSymbols": "number_sensing_symbols",
        "cfarGrdCellRange": "cfar_grd_cell_range",
        "cfarGrdCellVelocity": "cfar_grd_cell_velocity",
        "cfarTrnCellRange": "cfar_trn_cell_range",
        "cfarTrnCellVelocity": "cfar_trn_cell_velocity",
        "cfarTrnCellAzimuth": "cfar_trn_cell_azimuth",
        "cfarTrnCellElevation": "cfar_trn_cell_elevation",
        "cfarGrdCellAzimuth": "cfar_grd_cell_azimuth",
        "cfarGrdCellElevation": "cfar_grd_cell_elevation",
        "cfarThreshold": "cfar_threshold",
        "azFftLen": "az_fft_len",
        "elFftLen": "el_fft_len",
        "rdaThreshold": "rda_threshold",
        "nmsRadius": "nms_radius",
        "nmsMaxPeaks": "nms_max_peaks",
    }
    overrides = {mapping[k]: v for k, v in raw.items() if k in mapping}
    sens = apply_overrides(SensConfig(), overrides)
    if sens.window_len in (None, ""):
        sens.window_len = sens.doppler_fft_len
    return sens


def load_geometry(scenario_path: str | Path) -> Geometry:
    path = Path(scenario_path) / "Input" / "bsConfig.txt"
    matrix = _read_matrix(path)
    return Geometry(tx=matrix, rx=matrix)


def load_target(scenario_path: str | Path) -> Target:
    path = Path(scenario_path) / "Input" / "targetConfig.txt"
    matrix = _read_matrix(path)
    position = [row[:3] for row in matrix]
    velocity = [row[3:6] for row in matrix]
    return Target(position=position, velocity=velocity)


def load_scenario(scenario_path: str | Path):
    sim = load_simulation_config(scenario_path)
    prs = load_prs_config(scenario_path, sim)
    sens = load_sens_config(scenario_path)
    target = load_target(scenario_path)
    geometry = load_geometry(scenario_path)
    return sim, target, prs, geometry, sens
