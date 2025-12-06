from typing import Tuple, List
import numpy as np
from scipy.interpolate import interp1d
from scipy import ndimage

def hypotenuse_ndim(axes: List[np.ndarray], offset: float = 0.5) -> np.ndarray:
    """Function to compute the hypotenuse for n-dimensional axes.

    This is used to compute the distance of each voxel to the center point.
    Each 2D hypotenuse is computed by a^2 + b^2 = c^2
    Thus, this also gives the Euclidean distance of a 2D point.
    This is extended recursively to any dimension.

    Parameters
    ----------
    axes : List[np.ndarray]
        List of axes in n-dimensional space.
    offset : float, optional
        Offset to apply before calculating the hypotenuse, by default 0.5.


    Returns
    -------
    np.ndarray
        An n-dimensional numpy array representing the computed hypotenuse. The shape of
        the returned array is determined by the shape of the input axes.


    Notes
    -----
    The reason to use this function instead of simply computing the Euclidean distance
    directly is to calculate the hypotenuse with respect to an offset from the center
    point of the axes. This offset is subtracted from the maximum of each axis shape
    before the hypotenuse is calculated.
    """
    if len(axes) == 2:
        return np.hypot(
            axes[0] - max(axes[0].shape) * offset,
            axes[1] - max(axes[1].shape) * offset,
        )
    else:
        return np.hypot(
            hypotenuse_ndim(axes[1:], offset),
            axes[0] - max(axes[0].shape) * offset,
        )

def rotational_kernel(arr: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Create a rotational kernel from an input array.

    This function uses given input array and extends its values
    symmetrically by rotating it to the desired shape

    Parameters
    ----------
    arr : np.ndarray
        Input array used to create the rotational kernel.
        This should be a 1D array as output by radial_average()
    shape : Tuple[int, ...]
        Shape of the desired rotational kernel.

    Returns
    -------
    np.ndarray
        The created rotational kernel.
    """
    func = interp1d(np.arange(len(arr)), arr, bounds_error=False, fill_value=0)

    axes = np.ogrid[tuple(slice(0, np.ceil(s / 2)) for s in shape)]
    kernel = hypotenuse_ndim(axes, offset=0).astype("f4")
    kernel = func(kernel).astype("f4")
    for idx, s in enumerate(shape):
        padding = [(0, 0)] * len(shape)
        padding[idx] = (int(np.floor(s / 2)), 0)

        mode = "reflect" if s % 2 else "symmetric"
        kernel = np.pad(kernel, padding, mode=mode)
    return kernel