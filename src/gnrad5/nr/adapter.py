from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from gnrad5.config.models import PrsConfig, SimulationConfig
from gnrad5.nr.prs_local import build_prs_config, prs_indices_local, prs_symbols_local


class Py3gppUnavailable(RuntimeError):
    pass


def _load_py3gpp():
    try:
        import py3gpp  # type: ignore
    except Exception as exc:  # pragma: no cover - import error surface
        raise Py3gppUnavailable(
            "py3gpp is required for NR PRS/OFDM generation. Install via `uv add py3gpp`."
        ) from exc
    return py3gpp


def _resolve_fn(py3gpp: Any, name: str) -> Callable[..., Any]:
    for root in (py3gpp, getattr(py3gpp, "nr", None)):
        if root is None:
            continue
        fn = getattr(root, name, None)
        if fn is not None:
            return fn
    raise Py3gppUnavailable(f"py3gpp function `{name}` not found; API mismatch.")


def _set_attr(obj: Any, name: str, value: Any) -> None:
    if hasattr(obj, name) or obj.__class__.__name__ == "SimpleNamespace":
        setattr(obj, name, value)
        return
    if isinstance(obj, dict):
        obj[name] = value
        return
    raise AttributeError(f"Cannot set attribute {name} on {type(obj).__name__}")


def _call_with_format(fn: Callable[..., Any], *args, **kwargs) -> Any:
    try:
        return fn(*args, **kwargs)
    except TypeError:
        if "OutputResourceFormat" in kwargs:
            alt = kwargs.copy()
            alt["output_resource_format"] = alt.pop("OutputResourceFormat")
            return fn(*args, **alt)
        raise


@dataclass
class NrObjects:
    carrier: Any
    prs: Any
    ofdm_info: Any


def build_nr_objects(sim: SimulationConfig, prs_cfg: PrsConfig) -> NrObjects:
    py3gpp = _load_py3gpp()
    carrier_fn = _resolve_fn(py3gpp, "nrCarrierConfig")
    try:
        prs_fn = _resolve_fn(py3gpp, "nrPRSConfig")
    except Py3gppUnavailable:
        prs_fn = None
    ofdm_fn = _resolve_fn(py3gpp, "nrOFDMInfo")

    carrier = carrier_fn()
    _set_attr(carrier, "SubcarrierSpacing", sim.carrier_subcarrier_spacing)
    _set_attr(carrier, "NSizeGrid", sim.carrier_n_size_grid)

    if prs_fn is None:
        prs = build_prs_config()
    else:
        prs = prs_fn()
    _set_attr(prs, "PRSResourceSetPeriod", list(prs_cfg.prs_resource_set_period))
    _set_attr(prs, "PRSResourceOffset", prs_cfg.prs_resource_offset)
    _set_attr(prs, "PRSResourceRepetition", prs_cfg.prs_resource_repetition)
    _set_attr(prs, "PRSResourceTimeGap", prs_cfg.prs_resource_time_gap)
    _set_attr(prs, "NumRB", prs_cfg.num_rb)
    _set_attr(prs, "RBOffset", prs_cfg.rb_offset)
    _set_attr(prs, "CombSize", prs_cfg.comb_size)
    _set_attr(prs, "REOffset", prs_cfg.re_offset)
    _set_attr(prs, "NPRSID", prs_cfg.n_prs_id)
    _set_attr(prs, "NumPRSSymbols", prs_cfg.num_prs_symbols)
    _set_attr(prs, "SymbolStart", prs_cfg.symbol_start)

    ofdm_info = ofdm_fn(carrier)
    return NrObjects(carrier=carrier, prs=prs, ofdm_info=ofdm_info)


def prs_indices(carrier: Any, prs: Any):
    py3gpp = _load_py3gpp()
    try:
        fn = _resolve_fn(py3gpp, "nrPRSIndices")
        return _call_with_format(fn, carrier, prs, OutputResourceFormat="cell")
    except Py3gppUnavailable:
        return [prs_indices_local(carrier, prs)]


def prs_symbols(carrier: Any, prs: Any):
    py3gpp = _load_py3gpp()
    try:
        fn = _resolve_fn(py3gpp, "nrPRS")
        return _call_with_format(fn, carrier, prs, OutputResourceFormat="cell")
    except Py3gppUnavailable:
        return [prs_symbols_local(carrier, prs)]
