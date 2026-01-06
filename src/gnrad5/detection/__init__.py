from .cfar import cfar_2d, detect_cfar_rd
from .cfar_fast import cfar_2d_fast
from .cfar_fast_mod import cfar_2d_fast_mod
from .detect_4d import detect_cfar_4d
from .par import par_detector
from .peaks import pick_peaks_nms
from .cluster import cluster_peaks_4d
from .sidelobes import suppress_sidelobes
from .regional_max import imregionalmax_2d

__all__ = [
    "cfar_2d",
    "detect_cfar_rd",
    "cfar_2d_fast",
    "cfar_2d_fast_mod",
    "detect_cfar_4d",
    "par_detector",
    "pick_peaks_nms",
    "cluster_peaks_4d",
    "suppress_sidelobes",
    "imregionalmax_2d",
]
