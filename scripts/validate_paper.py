from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from gnrad5.config.models import Target
from gnrad5.antenna import build_type1_codebook_registry
from gnrad5.paper_chain import run_paper_chain
from gnrad5.paper_scenarios import build_paper_scenario


def _parse_float_list(raw: str):
    return [float(x) for x in raw.split(",") if x.strip()]


def _parse_int_list(raw: str):
    return [int(x) for x in raw.split(",") if x.strip()]


def _run_drop(args):
    (
        sim,
        prs,
        sens,
        geometry,
        pos,
        vel,
        seed,
        target_absent,
        fixed_rcs_dbsm,
        fixed_nrp,
        eirp_dbm,
        eta_db,
        codebook_id,
        prs_period_slots,
        beam_az,
        beam_el,
        max_range_bins,
        no_noise,
        repo_root,
        beam_chunk,
    ) = args

    if target_absent:
        target = Target(position=[], velocity=[])
    else:
        target = Target(position=[pos], velocity=[vel])

    det = run_paper_chain(
        sim,
        prs,
        sens,
        geometry,
        target,
        repo_root=repo_root,
        seed=seed,
        target_absent=target_absent,
        fixed_rcs_dbsm=fixed_rcs_dbsm,
        fixed_nrp=fixed_nrp,
        eirp_dbm=eirp_dbm,
        eta_db=eta_db,
        codebook_id=codebook_id,
        prs_period_slots=prs_period_slots,
        beam_az=beam_az,
        beam_el=beam_el,
        add_noise=not no_noise,
        max_range_bins=max_range_bins,
        beam_chunk=beam_chunk,
    )

    return {
        "present": not target_absent,
        "best_par_db": det.best_par_db,
        "best_range_m": det.best_range_m,
        "best_az": det.best_az,
        "best_el": det.best_el,
        "best_pos": det.best_pos,
        "true_pos": pos if not target_absent else None,
    }


def _init_worker(codebook_id: str | None, p_csi_rs: int):
    from gnrad5.paper_chain import _get_codebook_registry

    _get_codebook_registry(p_csi_rs)


def _format_eta(seconds: float) -> str:
    if not np.isfinite(seconds):
        return "?"
    if seconds < 60:
        return f"{seconds:.0f}s"
    return f"{seconds / 60.0:.1f}m"


def _log_progress(prefix: str, completed: int, total: int, start_time: float) -> None:
    elapsed = max(0.0, time.perf_counter() - start_time)
    rate = completed / elapsed if elapsed > 0 else 0.0
    remaining = (total - completed) / rate if rate > 0 else float("inf")
    frac = completed / total if total else 0.0
    print(
        f"{prefix} {completed}/{total} ({frac:6.2%}) elapsed {elapsed:.0f}s ETA {_format_eta(remaining)}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Reduced-sweep validation for arXiv 2505.24763v1 (Python-only)")
    parser.add_argument("--scenario", default="both", help="UMi, UMa, or both")
    parser.add_argument("--altitudes", default="25,50,100,200", help="Comma-separated UAV altitudes (m)")
    parser.add_argument("--drops", type=int, default=200, help="Drops per altitude")
    parser.add_argument("--etas", default="1.5,2.5,3.4,4.5,6.0", help="Comma-separated PAR thresholds (dB)")
    parser.add_argument("--beam-az", type=int, default=25, help="Azimuth beams")
    parser.add_argument("--beam-el", type=int, default=13, help="Elevation beams")
    parser.add_argument("--codebook-id", default=None, help="Type I codebook id (use 'grid' for legacy sweep)")
    parser.add_argument("--list-codebooks", action="store_true", help="List available codebook ids and exit")
    parser.add_argument("--workers", type=int, default=0, help="Number of worker processes (0=auto)")
    parser.add_argument("--beam-chunk", type=int, default=16, help="Beam batch size for sweep")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--output", default="paper_validation.csv", help="Output CSV path")
    parser.add_argument("--repo-root", default=None, help="Repo root (defaults to CWD)")
    parser.add_argument("--max-range-bins", type=int, default=1024, help="Limit range bins for memory")
    parser.add_argument("--no-noise", action="store_true", help="Disable noise")
    parser.add_argument("--progress-every", type=int, default=200, help="Print progress every N drops (0 to disable)")
    parser.add_argument("--progress-interval", type=float, default=30.0, help="Minimum seconds between progress prints")
    parser.add_argument("--eirp-dbm", type=float, default=75.0, help="EIRP in dBm/100MHz")
    parser.add_argument("--fixed-rcs-dbsm", type=float, default=-12.81, help="Fixed RCS in dBsm")
    parser.add_argument("--fixed-nrp", type=int, default=3, help="Number of background reference points (N_RP)")
    parser.add_argument("--prs-repetition", type=int, default=1, help="PRS resource repetition")
    parser.add_argument(
        "--prs-period-slots",
        type=int,
        default=32,
        help="PRS periodicity in slots (mu=3: 32 slots = 4ms)",
    )
    args = parser.parse_args()

    if args.list_codebooks:
        registry = build_type1_codebook_registry(32, include_multi_panel=True, ranks=(1, 2, 3, 4))
        for key in sorted(registry.keys()):
            print(key)
        return

    scenarios = ["UMI", "UMA"] if args.scenario.lower() == "both" else [args.scenario.upper()]
    altitudes = _parse_float_list(args.altitudes)
    etas = _parse_float_list(args.etas)

    workers = int(args.workers)
    if workers <= 0:
        workers = os.cpu_count() or 1

    records = []
    for scen in scenarios:
        for alt in altitudes:
            spec = build_paper_scenario(
                scen,
                alt,
                args.drops,
                seed=args.seed,
                target_absent=False,
                fixed_rcs_dbsm=args.fixed_rcs_dbsm,
                fixed_nrp=args.fixed_nrp,
                eirp_dbm=args.eirp_dbm,
                prs_repetition=args.prs_repetition,
                prs_period_slots=args.prs_period_slots,
            )
            spec_absent = build_paper_scenario(
                scen,
                alt,
                args.drops,
                seed=args.seed + 1,
                target_absent=True,
                fixed_rcs_dbsm=args.fixed_rcs_dbsm,
                fixed_nrp=args.fixed_nrp,
                eirp_dbm=args.eirp_dbm,
                prs_repetition=args.prs_repetition,
                prs_period_slots=args.prs_period_slots,
            )

            tp_counts = {eta: 0 for eta in etas}
            fn_counts = {eta: 0 for eta in etas}
            fp_counts = {eta: 0 for eta in etas}
            range_err = {eta: [] for eta in etas}
            angle_err = {eta: [] for eta in etas}
            pos_err = {eta: [] for eta in etas}
            pos_err_h = {eta: [] for eta in etas}
            pos_err_v = {eta: [] for eta in etas}

            bs_pos = np.asarray(spec.geometry.tx[0])
            tasks = []
            for idx, pos in enumerate(spec.target.position):
                vel = spec.target.velocity[idx]
                tasks.append(
                    (
                        spec.sim,
                        spec.prs,
                        spec.sens,
                        spec.geometry,
                        pos,
                        vel,
                        args.seed + idx,
                        False,
                        spec.fixed_rcs_dbsm,
                        spec.fixed_nrp,
                        spec.eirp_dbm,
                        etas[0],
                        args.codebook_id,
                        spec.prs_period_slots,
                        args.beam_az,
                        args.beam_el,
                        args.max_range_bins,
                        args.no_noise,
                        args.repo_root,
                        args.beam_chunk,
                    )
                )

            for idx in range(args.drops):
                tasks.append(
                    (
                        spec_absent.sim,
                        spec_absent.prs,
                        spec_absent.sens,
                        spec_absent.geometry,
                        None,
                        None,
                        args.seed + 1000 + idx,
                        True,
                        spec_absent.fixed_rcs_dbsm,
                        spec_absent.fixed_nrp,
                        spec_absent.eirp_dbm,
                        etas[0],
                        args.codebook_id,
                        spec_absent.prs_period_slots,
                        args.beam_az,
                        args.beam_el,
                        args.max_range_bins,
                        args.no_noise,
                        args.repo_root,
                        args.beam_chunk,
                    )
                )

            total_tasks = len(tasks)
            progress_prefix = f"[{scen} h={alt:g}m]"
            start_time = time.perf_counter()
            last_log = start_time
            completed = 0
            results = []
            if workers == 1:
                results_iter = map(_run_drop, tasks)
                for result in results_iter:
                    completed += 1
                    if args.progress_every > 0 and completed % args.progress_every == 0:
                        _log_progress(progress_prefix, completed, total_tasks, start_time)
                        last_log = time.perf_counter()
                    elif args.progress_interval > 0 and time.perf_counter() - last_log >= args.progress_interval:
                        _log_progress(progress_prefix, completed, total_tasks, start_time)
                        last_log = time.perf_counter()
                    results.append(result)
            else:
                with ProcessPoolExecutor(
                    max_workers=workers,
                    initializer=_init_worker,
                    initargs=(args.codebook_id, 32),
                ) as executor:
                    futures = [executor.submit(_run_drop, t) for t in tasks]
                    for fut in as_completed(futures):
                        results.append(fut.result())
                        completed += 1
                        if args.progress_every > 0 and completed % args.progress_every == 0:
                            _log_progress(progress_prefix, completed, total_tasks, start_time)
                            last_log = time.perf_counter()
                        elif args.progress_interval > 0 and time.perf_counter() - last_log >= args.progress_interval:
                            _log_progress(progress_prefix, completed, total_tasks, start_time)
                            last_log = time.perf_counter()
            results_iter = iter(results)

            for result in results_iter:
                if result["present"]:
                    pos = result["true_pos"]
                    true_vec = np.asarray(pos) - bs_pos
                    true_range = float(np.linalg.norm(true_vec))
                    true_az = np.rad2deg(np.arctan2(true_vec[1], true_vec[0]))
                    true_el = np.rad2deg(
                        np.arctan2(true_vec[2], np.sqrt(true_vec[0] ** 2 + true_vec[1] ** 2))
                    )
                    best_range = result["best_range_m"] if np.isfinite(result["best_range_m"]) else np.nan
                    est_pos = result["best_pos"]

                    for eta in etas:
                        detected = result["best_par_db"] >= eta
                        if detected:
                            tp_counts[eta] += 1
                            range_err[eta].append(abs(best_range - true_range))
                            angle_err[eta].append(abs(result["best_az"] - true_az) + abs(result["best_el"] - true_el))
                            if est_pos is not None and np.all(np.isfinite(est_pos)):
                                err_vec = est_pos - np.asarray(pos)
                                pos_err[eta].append(float(np.linalg.norm(err_vec)))
                                pos_err_h[eta].append(float(np.linalg.norm(err_vec[:2])))
                                pos_err_v[eta].append(float(abs(err_vec[2])))
                        else:
                            fn_counts[eta] += 1
                else:
                    for eta in etas:
                        if result["best_par_db"] >= eta:
                            fp_counts[eta] += 1

            for eta in etas:
                p_d = tp_counts[eta] / max(1, args.drops)
                p_fa = fp_counts[eta] / max(1, args.drops)
                p_miss = fn_counts[eta] / max(1, args.drops)
                rec = {
                    "scenario": scen,
                    "altitude_m": alt,
                    "eta_db": eta,
                    "p_d": p_d,
                    "p_fa": p_fa,
                    "p_miss": p_miss,
                    "mean_range_err_m": float(np.mean(range_err[eta])) if range_err[eta] else np.nan,
                    "mean_angle_err_deg": float(np.mean(angle_err[eta])) if angle_err[eta] else np.nan,
                    "mean_pos_err_m": float(np.mean(pos_err[eta])) if pos_err[eta] else np.nan,
                    "mean_pos_err_h_m": float(np.mean(pos_err_h[eta])) if pos_err_h[eta] else np.nan,
                    "mean_pos_err_v_m": float(np.mean(pos_err_v[eta])) if pos_err_v[eta] else np.nan,
                    "drops": args.drops,
                    "beam_az": args.beam_az,
                    "beam_el": args.beam_el,
                }
                records.append(rec)

    df = pd.DataFrame.from_records(records)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
