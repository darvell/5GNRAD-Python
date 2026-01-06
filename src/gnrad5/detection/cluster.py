from __future__ import annotations

import numpy as np


def cluster_peaks_4d(pk: np.ndarray, dims: dict, opts: dict):
    if pk is None or len(pk) == 0:
        return empty_out()

    wrap_d = dims.get("wrapD", True)
    wrap_az = dims.get("wrapAz", True)
    wrap_el = dims.get("wrapEl", True)

    min_val = opts.get("minVal", 0)
    norm = np.array(opts.get("norm", [1, 1, 1, 1]), dtype=float)
    val_weight = opts.get("valWeight", 0.0)
    eps = opts.get("eps", 2.5)
    min_pts = opts.get("minPts", 2)

    keep = pk[:, 4] >= min_val
    pk2 = pk[keep]
    if pk2.size == 0:
        return empty_out()

    r = pk2[:, 0]
    d = pk2[:, 1]
    el = pk2[:, 2]
    az = pk2[:, 3]
    v = pk2[:, 4]

    if "valScale" not in opts or opts["valScale"] is None:
        medv = np.median(v)
        madv = np.median(np.abs(v - medv)) + np.finfo(float).eps
        v_scale = 1.4826 * madv
    else:
        v_scale = max(opts["valScale"], np.finfo(float).eps)

    labels = _dbscan_wrap(
        pk2[:, :4],
        v,
        dims,
        norm,
        val_weight,
        v_scale,
        eps,
        min_pts,
        wrap_d,
        wrap_az,
        wrap_el,
    )

    k_max = labels.max() if labels.size else 0
    clusters = np.zeros((k_max, 11))
    members = []
    for k in range(1, k_max + 1):
        idx = np.where(labels == k)[0]
        members.append(idx)
        rr = r[idx]
        dd = d[idx]
        ee = el[idx]
        aa = az[idx]
        vv = v[idx]

        wsum = np.sum(vv) + np.finfo(float).eps
        w = vv / wsum

        rc = np.sum(w * rr)
        elc = np.sum(w * ee)

        dc = circ_mean_w_bins(dd, w, dims["D"]) if wrap_d else np.sum(w * dd)
        azc = circ_mean_w_bins(aa, w, dims["Az"]) if wrap_az else np.sum(w * aa)

        rmed = np.median(rr)
        elmed = np.median(ee)
        dmed = circ_median_w_bins(dd, vv, dims["D"]) if wrap_d else np.median(dd)
        azmed = circ_median_w_bins(aa, vv, dims["Az"]) if wrap_az else np.median(aa)

        clusters[k - 1, :] = [
            rc,
            dc,
            elc,
            azc,
            np.max(vv),
            np.sum(vv),
            len(idx),
            rmed,
            dmed,
            elmed,
            azmed,
        ]

    return {
        "labels": labels,
        "clusters": clusters,
        "members": members,
        "pk_kept": pk2,
    }


def _dbscan_wrap(xsp, v, dims, sig, wv, v_scale, eps_rad, min_pts, wrap_d, wrap_az, wrap_el):
    n = xsp.shape[0]
    labels = np.zeros(n, dtype=int)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    sr, sd, se, sa = sig
    d_size = dims["D"]
    az_size = dims["Az"]
    el_size = dims["El"]

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        nbrs = _region_query(xsp, v, i, sig, wv, v_scale, eps_rad, d_size, az_size, el_size, wrap_d, wrap_az, wrap_el)
        if len(nbrs) < min_pts:
            labels[i] = 0
            continue
        cluster_id += 1
        labels[i] = cluster_id
        s = list(nbrs)
        j = 0
        while j < len(s):
            q = s[j]
            if not visited[q]:
                visited[q] = True
                nbrs2 = _region_query(xsp, v, q, sig, wv, v_scale, eps_rad, d_size, az_size, el_size, wrap_d, wrap_az, wrap_el)
                if len(nbrs2) >= min_pts:
                    for item in nbrs2:
                        if item not in s:
                            s.append(item)
            if labels[q] == 0:
                labels[q] = cluster_id
            j += 1
    return labels


def _region_query(xsp, v, idx, sig, wv, v_scale, eps_rad, d_size, az_size, el_size, wrap_d, wrap_az, wrap_el):
    sr, sd, se, sa = sig
    dr = np.abs(xsp[:, 0] - xsp[idx, 0])
    dd = np.abs(xsp[:, 1] - xsp[idx, 1])
    de = np.abs(xsp[:, 2] - xsp[idx, 2])
    da = np.abs(xsp[:, 3] - xsp[idx, 3])
    if wrap_d:
        dd = np.minimum(dd, d_size - dd)
    if wrap_az:
        da = np.minimum(da, az_size - da)
    if wrap_el:
        de = np.minimum(de, el_size - de)
    dv = np.abs(v - v[idx])
    dist2 = (dr / sr) ** 2 + (dd / sd) ** 2 + (de / se) ** 2 + (da / sa) ** 2 + (
        wv * dv / v_scale
    ) ** 2
    return np.where(dist2 <= eps_rad**2)[0]


def circ_mean_w_bins(x, w, p):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    w = w / (np.sum(w) + np.finfo(float).eps)
    ang = 2 * np.pi * (np.mod(x, p) / p)
    z = np.sum(w * np.exp(1j * ang))
    if np.abs(z) < 1e-12:
        return circ_median_w_bins(x, w, p)
    a = np.angle(z)
    return np.mod(p * (a / (2 * np.pi)), p)


def circ_median_w_bins(x, w, p):
    x = (np.mod(np.asarray(x, dtype=float) - 1, p) + 1)
    w = np.asarray(w, dtype=float)
    w = w / (np.sum(w) + np.finfo(float).eps)
    ang = 2 * np.pi * (x / p)
    z = np.mean(np.exp(1j * ang))
    if np.abs(z) < 1e-12:
        anchor = x[np.argmax(w)]
    else:
        anchor = np.mod(p * (np.angle(z) / (2 * np.pi)), p)

    y = x - anchor
    y = y - p * np.round(y / p)
    order = np.argsort(y)
    ys = y[order]
    ws = w[order]
    cs = np.cumsum(ws)
    j = np.searchsorted(cs, 0.5, side="left")
    ym = ys[j]
    return np.mod(ym + anchor - 1, p) + 1


def empty_out():
    return {
        "labels": np.zeros((0,), dtype=int),
        "clusters": np.zeros((0, 11)),
        "members": [],
        "pk_kept": np.zeros((0, 5)),
    }
