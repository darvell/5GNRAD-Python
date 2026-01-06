from __future__ import annotations

from typing import Any

import numpy as np


def field_to_num(params: dict, field: str, valid_values=None, default_value=None, step: float = 0.0):
    if field in params:
        value = params[field]
        if isinstance(value, str):
            try:
                value = float(value)
                if value.is_integer():
                    value = int(value)
            except ValueError:
                try:
                    value = np.array(eval(value, {}), dtype=float).tolist()
                except Exception:
                    pass
        params[field] = value
    else:
        params[field] = default_value

    if valid_values is None:
        return params

    value = np.asarray(params[field])
    if step == 0:
        if not np.all(np.isin(value, np.asarray(valid_values))):
            raise ValueError(f"Invalid value {value} for field {field}")
    else:
        if step == np.finfo(float).eps:
            lo, hi = valid_values
            if not (np.all(value >= lo) and np.all(value <= hi)):
                raise ValueError(f"Invalid value {value} for field {field}")
        else:
            lo, hi = valid_values
            allowed = np.arange(lo, hi + step / 2, step)
            if not np.all(np.isin(value, allowed)):
                raise ValueError(f"Invalid value {value} for field {field}")
    return params
