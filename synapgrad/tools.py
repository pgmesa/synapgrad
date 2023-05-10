import random
import numpy as np


def manual_seed(seed:int):
    """ 
    Set the seed for the random number generators. This function should be called 
    at the beginning of the program in orderto ensure reproducibility.

    Parameters
    ----------
    seed : int
        The seed.

    Examples
    --------
    >>> manual_seed(42)
    
    """
    np.random.seed(seed)
    random.seed(seed)


def is_floating_point(array) -> bool:
    """
    Check if the given array is of a floating point type.

    Parameters
    ----------
    array : array-like
        The array to check. Must have attribute .dtype (numpy dtype)

    Returns
    -------
    bool
        True if the array is of a floating point type, False otherwise.

    Examples
    --------
    >>> is_floating_point([1, 2, 3])
    False
    >>> is_floating_point([1.0, 2.0, 3.0])
    True
    """
    return array.dtype == np.float16 or array.dtype == np.float32 or array.dtype == np.float64


def pretty_numpy(array:np.ndarray, precision=4, separator=',') -> str:
    """
    Simple function to personalized numpy array printing.

    Parameters
    ----------
    array : array-like
        The array to print.
    precision : int, optional
        The number of digits after the decimal point.
    separator : str, optional
        The separator between the elements.

    Returns
    -------
    str
        The string representation of the array.

    Examples
    --------
    >>> pretty_numpy(np.array([1, 2, 3]))
    '[1, 2, 3]'
    >>> pretty_numpy(np.array([1.0, 2.0, 3.0]), precision=2)
    '[1.00, 2.00, 3.00]'
    >>> pretty_numpy(np.array([1.0, 2.0, 3.0]), precision=3, separator=' ')
    '[1.000 2.000 3.000]'
    """
    data_str = np.array2string(array, precision=precision, separator=separator)
    return data_str


def get_selected_from_indices(values:np.ndarray, indices:np.ndarray, dim) -> np.ndarray:
    """
    Returns a boolean array of the same shape as the input array, where each element is True if the corresponding
    element in the input array is the selected value in the corresponding dimension.

    Parameters
    ----------
    values : np.ndarray
        The values in the input array.
    indices : np.ndarray
        The indices of the selected values in the input array.
    dim : int
        The dimension of the input array.

    Returns
    -------
    np.ndarray
        The boolean array.
    """
    def get_indices(array, dim):
        indices = []
        if dim == -1: dim = array.ndim - 1
        for i in range(dim):
            s = [None,] * dim
            s[i] = slice(None)
            s = tuple(s)
            #print(s)
            indices.append(np.arange(array.shape[i])[s])
        return tuple(indices)

    selected = np.zeros_like(values)
    
    if dim is not None:
        slices = get_indices(values, dim)
        slices = list(slices)
        slices.append(indices)
        slices = tuple(slices)
    else:
        slices = np.unravel_index(indices, values.shape)

    selected[slices] = 1
    
    return selected