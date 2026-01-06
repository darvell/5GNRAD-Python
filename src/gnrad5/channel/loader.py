from __future__ import annotations

import json
from pathlib import Path


class ChannelLoadError(RuntimeError):
    pass


def _scenario_type(path: str, for_target: bool) -> str:
    lower = path.lower()
    if "uma" in lower:
        return "UMaAV" if for_target else "3GPP_38.901_UMa_NLOS"
    if "umi" in lower:
        return "UMiAV" if for_target else "3GPP_38.901_UMi_NLOS"
    if "rma" in lower:
        return "RMaAV" if for_target else "3GPP_38.901_RMa_NLOS"
    raise ChannelLoadError(f"Unknown scenario type in scenarioPath: {path}")


def _load_json(path: Path):
    if not path.exists():
        raise ChannelLoadError(f"Channel file not found: {path}")
    return json.loads(path.read_text())


def load_background_channel(repo_root: str | Path, scenario_path: str, fc_hz: float):
    scenario_type = _scenario_type(scenario_path, for_target=False)
    fc_ghz = fc_hz / 1e9
    file_name = f"backgroundChannel_{fc_ghz:g}GHz_{scenario_type}.json"
    path = Path(repo_root) / "channel" / file_name
    return _load_json(path)


def load_target_channel(repo_root: str | Path, scenario_path: str, fc_hz: float):
    scenario_type = _scenario_type(scenario_path, for_target=True)
    fc_ghz = fc_hz / 1e9
    file_name = f"targetChannel_{fc_ghz:g}GHz_{scenario_type}.json"
    path = Path(repo_root) / "channel" / file_name
    if not path.exists():
        return None
    return _load_json(path)
