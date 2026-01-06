from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

from gnrad5.geometry.angles import wrap_to_180


def munkres(cost_matrix, big_cost=None):
    cost = np.asarray(cost_matrix, dtype=float)
    n_rows, n_cols = cost.shape
    valid = np.isfinite(cost)
    if big_cost is None:
        if np.any(valid):
            big_cost = 10 ** (np.ceil(np.log10(np.sum(cost[valid]))).astype(int) + 1)
        else:
            big_cost = 1e9
    cost = cost.copy()
    cost[~valid] = big_cost
    n = max(n_rows, n_cols)
    padded = np.full((n, n), big_cost, dtype=float)
    padded[:n_rows, :n_cols] = cost
    row_ind, col_ind = linear_sum_assignment(padded)

    assignment = np.zeros(n_rows, dtype=int)
    for r, c in zip(row_ind, col_ind, strict=False):
        if r < n_rows and c < n_cols:
            if cost[r, c] < big_cost:
                assignment[r] = c + 1
    assigned_cols = assignment[assignment > 0] - 1
    assigned_rows = np.nonzero(assignment > 0)[0]
    total_cost = float(np.sum(cost[assigned_rows, assigned_cols])) if assigned_cols.size else 0.0
    return assignment, total_cost


def eval_tp_pos_vel(gt_pos, est_pos, gt_vel, est_vel, sensor_pos):
    gt_pos = np.asarray(gt_pos, dtype=float)
    est_pos = np.asarray(est_pos, dtype=float)
    sensor_pos = np.asarray(sensor_pos, dtype=float).reshape(1, 3)
    gt_vel = np.asarray(gt_vel, dtype=float).reshape(-1) if gt_vel is not None else np.zeros(len(gt_pos))
    est_vel = np.asarray(est_vel, dtype=float).reshape(-1) if est_vel is not None else np.zeros(len(est_pos))

    d_pos = est_pos - gt_pos
    r_gt_vec = gt_pos - sensor_pos
    r_est_vec = est_pos - sensor_pos

    az_gt = np.rad2deg(np.arctan2(r_gt_vec[:, 1], r_gt_vec[:, 0]))
    el_gt = np.rad2deg(
        np.arctan2(r_gt_vec[:, 2], np.sqrt(r_gt_vec[:, 0] ** 2 + r_gt_vec[:, 1] ** 2))
    )
    az_est = np.rad2deg(np.arctan2(r_est_vec[:, 1], r_est_vec[:, 0]))
    el_est = np.rad2deg(
        np.arctan2(r_est_vec[:, 2], np.sqrt(r_est_vec[:, 0] ** 2 + r_est_vec[:, 1] ** 2))
    )

    range_gt = np.linalg.norm(r_gt_vec, axis=1)
    range_est = np.linalg.norm(r_est_vec, axis=1)
    range_err = range_est - range_gt

    az_err = wrap_to_180(az_est - az_gt)
    el_err = wrap_to_180(el_est - el_gt)

    return {
        "pos": {
            "errXYZ": d_pos,
            "absXYZ": np.abs(d_pos),
            "errMag": np.linalg.norm(d_pos, axis=1),
            "range_gt": range_gt,
            "range_est": range_est,
            "range_err": range_err,
            "az_err_deg": az_err,
            "el_err_deg": el_err,
        },
        "vel": {"vr_err": est_vel - gt_vel},
    }


def score_associations_pos(
    gt_pos,
    det_pos,
    gt_vel,
    det_vel,
    sensor_pos,
    sigma_xyz=6.0,
    cov=None,
    p_gate=0.9973,
    big_cost=1e9,
):
    gt_pos = np.asarray(gt_pos, dtype=float)
    det_pos = np.asarray(det_pos, dtype=float)
    n_gt = gt_pos.shape[0]
    n_det = det_pos.shape[0]

    if n_gt == 0:
        metrics = {
            "TP": 0,
            "FN": 0,
            "FP": int(n_det),
            "TPR": 0.0,
            "FNR": 0.0,
            "FPR": 1.0 if n_det else 0.0,
            "stats": eval_tp_pos_vel(gt_pos, np.zeros((0, 3)), gt_vel, det_vel, sensor_pos),
        }
        assoc = {
            "gtToDet": np.zeros(0, dtype=int),
            "detToGt": np.zeros(n_det, dtype=int),
            "costMat": np.zeros((0, n_det)),
            "gate2": np.nan,
            "SigmaXYZ": sigma_xyz,
            "Cov": cov,
            "InvCov": None,
        }
        return metrics, assoc

    if cov is None:
        sigma = np.array([sigma_xyz] * 3, dtype=float) if np.isscalar(sigma_xyz) else np.asarray(sigma_xyz)
        cov_use = np.diag(sigma**2)
    else:
        cov_use = np.asarray(cov, dtype=float)
    cov_use = (cov_use + cov_use.T) / 2
    eigvals, eigvecs = np.linalg.eigh(cov_use)
    eigvals = np.maximum(eigvals, 1e-12 * np.max(eigvals))
    inv_cov = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T

    cost_mat = np.zeros((n_gt, n_det), dtype=float)
    for j in range(n_det):
        d = gt_pos - det_pos[j]
        t = d @ inv_cov
        cost_mat[:, j] = np.sum(t * d, axis=1)

    gate2 = chi2.ppf(p_gate, 3)
    cost_gated = cost_mat.copy()
    cost_gated[cost_gated > gate2] = big_cost

    assign, _ = munkres(cost_gated, big_cost=big_cost)
    gt_to_det = assign.astype(int)
    det_to_gt = np.zeros(n_det, dtype=int)
    for i, j in enumerate(gt_to_det, start=0):
        if j > 0 and det_to_gt[j - 1] == 0:
            det_to_gt[j - 1] = i + 1

    valid = np.zeros(n_gt, dtype=bool)
    for i, j in enumerate(gt_to_det):
        if j > 0:
            if cost_gated[i, j - 1] < big_cost:
                valid[i] = True

    tp = int(np.sum(valid))
    fn = int(n_gt - tp)
    fp = int(n_det - tp)

    det_pos_aligned = np.full((n_gt, 3), np.nan)
    det_vel_aligned = np.full((n_gt,), np.nan)
    if np.any(valid):
        idx = np.where(valid)[0]
        det_idx = gt_to_det[idx] - 1
        det_pos_aligned[idx] = det_pos[det_idx]
        if det_vel is not None:
            det_vel_aligned[idx] = np.asarray(det_vel, dtype=float)[det_idx]

    stats = eval_tp_pos_vel(gt_pos, det_pos_aligned, gt_vel, det_vel_aligned, sensor_pos)
    metrics = {
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TPR": tp / max(n_gt, 1),
        "FNR": fn / max(n_gt, 1),
        "FPR": fp / max(n_det, 1),
        "stats": stats,
    }
    assoc = {
        "gtToDet": gt_to_det,
        "detToGt": det_to_gt,
        "costMat": cost_gated,
        "gate2": gate2,
        "SigmaXYZ": sigma_xyz,
        "Cov": cov_use,
        "InvCov": inv_cov,
    }
    return metrics, assoc
