from __future__ import annotations

import numpy as np


def set_st_min_distance_constraint(positions, n_per_group, min_dist):
    positions = np.asarray(positions, dtype=float)
    n = positions.shape[0]
    groups = []
    i = 0
    while i < n:
        this = np.zeros(n_per_group, dtype=int)
        this[0] = i
        count = 1
        j = i + 1
        while j < n and count < n_per_group:
            p = positions[j]
            d = np.linalg.norm(positions[this[:count]] - p, axis=1)
            if np.all(d >= min_dist):
                this[count] = j
                count += 1
            j += 1
        if count < n_per_group:
            raise ValueError(
                f"Could not form full group {len(groups)+1}: only {count}/{n_per_group} points satisfy {min_dist:.1f} m spacing."
            )
        groups.append(this)
        i += n_per_group
    return np.concatenate(groups, axis=0)
