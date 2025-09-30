import numpy as np


EPS = np.finfo(float).eps


def get_arrays_tol(*arrays):
    """
    Get a relative tolerance for a set of arrays. Borrowed from COBYQA

    Parameters
    ----------
    *arrays: tuple
        Set of `numpy.ndarray` to get the tolerance for.

    Returns
    -------
    float
        Relative tolerance for the set of arrays.

    Raises
    ------
    ValueError
        If no array is provided.
    """
    if len(arrays) == 0:
        raise ValueError("At least one array must be provided.")
    size = max(array.size for array in arrays)
    weight = max(
        np.max(np.abs(array[np.isfinite(array)]), initial=1.0)
        for array in arrays
    )
    return 10.0 * EPS * max(size, 1.0) * weight
