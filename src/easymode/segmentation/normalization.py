"""Input normalization, shared by extraction and inference so both measure the statistic the
same way. center = mean; scale = MAD*1.4826 (global_mad, default) or std (global_std, legacy).
The global schemes normalize the whole volume once, before any rescale, and never re-normalize
per box/tile. local does the opposite: every box, and at inference every tile, is scaled by its
own statistic."""
import numpy as np

NORM_GLOBAL_MAD = "global_mad"
NORM_GLOBAL_STD = "global_std"   # legacy: what pre-2026 published models expect
NORM_LOCAL = "local"             # per box in training, per tile at inference

# What a user picks at training time. global_std is not offered: it exists only so inference can
# still read the tag on pre-2026 models, not as a scheme to train a new model with.
SCHEMES = {"global": NORM_GLOBAL_MAD, "local": NORM_LOCAL}


def global_stats(volume, method=NORM_GLOBAL_MAD, n_slices=32):
    # (center, scale) over the central XY region of n_slices evenly-spaced Z-slices.
    # 1.4826 makes MAD a consistent estimator of sigma. volume may be a memmap.
    z, k, l = volume.shape
    zidx = np.linspace(0, z - 1, min(n_slices, z)).astype(int)
    mk = min(int(0.2 * k), 64)
    ml = min(int(0.2 * l), 64)
    region = np.asarray(volume[zidx][:, mk:(k - mk) if mk else k, ml:(l - ml) if ml else l], dtype=np.float32)
    center = float(np.mean(region))
    if method == NORM_GLOBAL_STD:
        scale = float(np.std(region))
    else:
        med = np.median(region)
        scale = float(1.4826 * np.median(np.abs(region - med)))
    return center, scale + 1e-7
