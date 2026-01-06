from __future__ import annotations

import argparse
from pathlib import Path

from gnrad5.analysis import run5gnrad, write_outputs


def main():
    parser = argparse.ArgumentParser(description="5GNRad Python full run (run5GNRad parity)")
    parser.add_argument("--scenario", required=True, help="Scenario path (e.g., examples3GPP/UMa-Av200-8x8-30)")
    parser.add_argument("--repo-root", default=None, help="Repo root (defaults to CWD)")
    parser.add_argument(
        "--no-matlab-range-mean",
        action="store_true",
        help="Disable strict MATLAB range-mean subtraction (for debugging only)",
    )
    args = parser.parse_args()

    outputs = run5gnrad(
        args.scenario,
        args.repo_root,
        strict_matlab_range_mean=not args.no_matlab_range_mean,
    )
    write_outputs(outputs, args.scenario)
    print(f"Wrote error.csv and detStats.csv to {Path(args.scenario) / 'Output'}")


if __name__ == "__main__":
    main()
