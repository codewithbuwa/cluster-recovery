from __future__ import annotations

import os
from pathlib import Path


def configure_matplotlib_cache() -> None:
    """Point Matplotlib/fontconfig caches at writable workspace-safe paths."""

    cache_root = Path(os.environ.get("CPO_PLOT_CACHE_DIR", "/private/tmp/cpo_plot_cache"))
    mpl_dir = cache_root / "matplotlib"
    xdg_dir = cache_root / "xdg"
    fontconfig_dir = xdg_dir / "fontconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    fontconfig_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_dir))
    os.environ.setdefault("FC_CACHEDIR", str(fontconfig_dir))
