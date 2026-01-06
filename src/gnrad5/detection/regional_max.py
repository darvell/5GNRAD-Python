from __future__ import annotations

import numpy as np


def imregionalmax_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    max_val = arr.max() if arr.size else 0
    if arr.size == 0:
        return np.zeros_like(arr, dtype=bool)

    visited = np.zeros(arr.shape, dtype=bool)
    output = np.zeros(arr.shape, dtype=bool)

    rows, cols = arr.shape
    for r in range(rows):
        for c in range(cols):
            if visited[r, c]:
                continue
            visited[r, c] = True
            val = arr[r, c]
            if np.isnan(val):
                continue
            stack = [(r, c)]
            plateau = [(r, c)]
            is_max = True
            while stack:
                pr, pc = stack.pop()
                for rr in range(pr - 1, pr + 2):
                    for cc in range(pc - 1, pc + 2):
                        if rr < 0 or cc < 0 or rr >= rows or cc >= cols:
                            continue
                        if rr == pr and cc == pc:
                            continue
                        if np.isnan(arr[rr, cc]):
                            continue
                        if arr[rr, cc] > val:
                            is_max = False
                        if not visited[rr, cc] and arr[rr, cc] == val:
                            visited[rr, cc] = True
                            stack.append((rr, cc))
                            plateau.append((rr, cc))
            if is_max:
                for rr, cc in plateau:
                    output[rr, cc] = True
    return output
