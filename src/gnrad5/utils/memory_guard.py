from __future__ import annotations

import os
import sys
import resource


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _total_memory_bytes() -> int | None:
    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().total)
    except Exception:
        pass
    if hasattr(os, "sysconf"):
        names = os.sysconf_names
        if "SC_PHYS_PAGES" in names and "SC_PAGE_SIZE" in names:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return int(pages * page_size)
    return None


def _process_rss_bytes() -> int:
    try:
        import psutil  # type: ignore

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        pass
    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as handle:
            parts = handle.read().strip().split()
        if len(parts) >= 2:
            rss_pages = int(parts[1])
            page_size = os.sysconf("SC_PAGE_SIZE")
            return rss_pages * page_size
    except Exception:
        pass
    try:
        import subprocess

        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True)
        rss_kb = int(out.strip().split()[0])
        return rss_kb * 1024
    except Exception:
        pass
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(rss)
    return int(rss) * 1024


def guard_memory(stage: str, extra_bytes: int = 0) -> None:
    max_fraction = _env_float("GN5_MAX_RAM_FRACTION", 0.85)
    max_rss_mb = _env_float("GN5_MAX_RSS_MB", 11264.0)

    rss = _process_rss_bytes()
    total = _total_memory_bytes()

    limit_bytes = None
    if max_rss_mb > 0:
        limit_bytes = int(max_rss_mb * 1024 * 1024)
    elif total is not None and max_fraction > 0:
        limit_bytes = int(total * max_fraction)

    if limit_bytes is None:
        return

    projected = rss + max(0, int(extra_bytes))
    if projected > limit_bytes:
        rss_mb = rss / (1024 * 1024)
        limit_mb = limit_bytes / (1024 * 1024)
        raise MemoryError(
            f"Memory guard tripped at {stage}: RSS {rss_mb:.1f} MB, "
            f"projected {projected / (1024 * 1024):.1f} MB > limit {limit_mb:.1f} MB. "
            "Set GN5_MAX_RAM_FRACTION or GN5_MAX_RSS_MB to tune."
        )
