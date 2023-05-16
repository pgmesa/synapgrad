import random
import numpy as np
from synapgrad.tensor import Tensor

# **************************
# ******* Conv tools *******
# **************************

def get_conv1d_output_size(input_length:int, kernel_size:int, stride:int, padding:int, dilation:int) -> int:
    """
    Computes the output size of a 1d convolution.

    Args:
        input_length (int): Length of the sequence
        kernel_size (int): Spatial size of the kernel
        stride (int): Stride of the convolution
        padding (int): Padding of the convolution
        dilation (int): Spacing between kernel elements

    Returns:
        int: Output size
    """
    length_padded = input_length + 2 * padding
    num_windows = int(np.floor((length_padded - dilation * (kernel_size - 1) - 1) / stride + 1))
    return num_windows


def get_arr2col_indices(arr_shape, kernel_size, dilation=1, stride=1, padding=0) -> tuple:
    """
    Compute the indices of the input array for the 1d convolution operation.

    Parameters
    ----------
    arr_shape : tuple
        Shape of the input array.
    kernel_size : int
        Size of the kernel used in the convolution operation.
    dilation : int
        Dilation rate used in the convolution operation.
    stride : int
        Stride value used in the convolution operation.
    padding : int
        Padding value or tuple of values used in the convolution operation.

    Returns
    -------
    tuple
        Tuple of 1d arrays of indices.

    Example
    -------
    >>> arr_shape = (1,1,9)
    >>> get_arr2col_indices(arr_shape, kernel_size=3, stride=2, padding=0)
    Output:
    (array([0, 1, 2]), array([2, 3, 4]), array([4, 5, 6]), array([6, 7, 8]),)
    """
    if padding > kernel_size / 2:
            raise ValueError("Invalid padding: pad should be smaller than or equal to half " +
                    "of kernel size, but got pad = {}, kernel_size = {}.".format(padding, kernel_size))
    
    L = get_conv1d_output_size(arr_shape[2], kernel_size, stride, padding, dilation)
    
    if L == 0:
        raise RuntimeError('Cannot get indices of an array with zero or negative spatial size (L='+str(L)+
            ') for kernel_size='+str(kernel_size)+', stride='+str(stride)+', padding='+str(padding)+
            ', dilation='+str(dilation)+' and shape='+str(arr_shape))
    
    window_size = kernel_size + (dilation - 1) * (kernel_size - 1)
    indices = tuple([np.arange(i*stride, (i*stride)+window_size, dilation) for i in range(L)])
    return indices
    
    
def arr2col(arr, kernel_size, dilation=1, stride=1, padding=0, pad_value=0, unf_indices=None, return_indices=False):
    """
    Convert a 3D dimension array to a 4D dimension array by unfolding the 3nd dimension.

    Parameters
    ----------
    arr : array-like
        Input array
    kernel_size : int
        Size of the kernel used in the convolution operation.
    dilation : int
        Dilation rate used in the convolution operation.
    stride : int
        Stride value used in the convolution operation.
    padding : int
        Padding value or tuple of values used in the convolution operation.
    pad_value : scalar, optional (default=0)
        Value used for padding the input array.
    unf_indices : tuple
        Precomputed indices of the input array.
    return_indices : bool
        If True, the function returns the indices of the input array.

    Returns
    -------
    array
        An array representing the unfolded array.

    Example
    -------
    >>> arr = np.array([[[1, 2, 3, 4, 5, 6, 7, 8, 9]]])
    >>> arr2col(arr, kernel_size=3, dilation=1, stride=2, padding=0)
    Output:
    array([[[[1, 2, 3],
             [3, 4, 5],
             [5, 6, 7],
             [7, 8, 9]]]])
    """
    if len(arr.shape) != 3:
        raise ValueError("array must be a 3D array, but got shape = {}".format(arr.shape))
    
    if not unf_indices:
        unf_indices = get_arr2col_indices(arr.shape, kernel_size, dilation, stride, padding)
    
    if padding > 0:
        arr = np.pad(arr, ((0,0), (0,0), (padding, padding)), mode='constant', constant_values=pad_value)
    
    unfolded = np.take(arr, unf_indices, axis=2)
    
    out = unfolded if not return_indices else (unfolded, unf_indices)
    return out


def col2arr(unfolded, arr_shape, kernel_size, dilation=1, stride=1, padding=0, unf_indices=None, return_indices=False):
    """
    Converts a 4D array to a 3D array by folding the 3rd dimension

    Parameters
    ----------
    unfolded : array-like
        4D input array to fold
    arr_shape : tuple
        Shape of the output array
    kernel_size : int
        Size of the kernel used in the convolution operation.
    dilation : int
        Dilation rate used in the convolution operation.
    stride : int
        Stride value used in the convolution operation.
    padding : int
        Padding value or tuple of values used in the convolution operation.
    unf_indices : tuple
        Precomputed indices in arr2col.
    return_indices : bool
        If True, the function returns the indices of the input array.

    Returns
    -------
    array
        An array representing the folded array.

    Example
    -------
    >>> unfolded = np.array([[1, 2, 3],
    ...                      [3, 4, 5],
    ...                      [4, 6, 7],
    ...                      [7, 8, 9]])
    >>> indices = (array([0, 1, 2]), array([2, 3, 4]), array([4, 5, 6]), array([6, 7, 8]),)
    >>> col2arr(unfolded, (1,1,9), kernel_size=3, dilation=1, stride=2, padding=0, unf_indices=indices)
    Output:
    array([[[1, 2, 3+3, 4, 5+5, 6, 7+7, 8, 9]]])
    """
    if len(unfolded.shape) != 4:
        raise ValueError("Input array must be 4D, but got shape="+str(unfolded.shape))
    
    if len(arr_shape) != 3:
        raise ValueError("Output array must be 3D, but got shape="+str(arr_shape))
    
    output = np.zeros((arr_shape[0], arr_shape[1], arr_shape[2] + 2 * padding))
  
    if unf_indices is None:
        unf_indices = get_arr2col_indices(arr_shape, kernel_size, dilation, stride, padding)
    
    np.add.at(output, (slice(None), slice(None), unf_indices), unfolded)
    
    if padding > 0:
        output = output[:, :, padding:-padding]
    
    out = output if not return_indices else (output, unf_indices)
    return out


def get_conv2d_output_size(shape:tuple, kernel_size, dilation, stride, padding) -> tuple:
    """
    Calculate the output size of a 2D convolution operation.

    Parameters
    ----------
    shape : tuple
        Shape of the input tensor (N, C, H, W).
    kernel_size : int or tuple
        Size of the kernel used in the convolution operation.
    dilation : int or tuple
        Dilation rate used in the convolution operation.
    stride : int or tuple
        Stride value used in the convolution operation.
    padding : int or tuple
        Padding value or tuple of values used in the convolution operation.

    Returns
    -------
    tuple
        A tuple (lH, lW) representing the output size of the convolution operation.

    Example
    -------
    >>> shape = (1, 1, 5, 5)
    >>> get_conv2d_output_size(shape, kernel_size=3, dilation=1, stride=2, padding=0)
    Output:
    (2, 2)
    """
    N, C, H, W = shape
    
    kernel_size = np.broadcast_to(kernel_size, 2)
    dilation = np.broadcast_to(dilation, 2)
    padding = np.broadcast_to(padding, 2)
    stride = np.broadcast_to(stride, 2)
    
    H_with_pad = H + 2 * padding[0]
    W_with_pad = W + 2 * padding[1]
    
    lH = int(np.floor((H_with_pad - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1))
    lW = int(np.floor((W_with_pad - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1))
    
    return lH, lW
    

def get_im2col_indices(a_shape:tuple, kernel_size, dilation=1, stride=1, padding=0):
    """
    Calculate the indices for the im2col operation.

    Parameters
    ----------
    a_shape : tuple
        Shape of the input tensor (N, C, H, W).
    kernel_size : int or tuple
        Size of the kernel used in the im2col operation.
    dilation : int or tuple, optional (default=1)
        Dilation rate used in the im2col operation.
    stride : int or tuple, optional (default=1)
        Stride value used in the im2col operation.
    padding : int or tuple, optional (default=0)
        Padding value or tuple of values used in the im2col operation.

    Returns
    -------
    tuple
        A tuple (k, i, j) containing the indices for the im2col operation.

    Raises
    ------
    RuntimeError
        If the spatial size of the tensor is zero or negative for the given parameters.
    """
    N, C, H, W = a_shape
    
    kernel_size = np.broadcast_to(kernel_size, 2)
    dilation = np.broadcast_to(dilation, 2)
    padding = np.broadcast_to(padding, 2)
    stride = np.broadcast_to(stride, 2)
    
    lH, lW = get_conv2d_output_size(a_shape, kernel_size, dilation, stride, padding)
    L = lH * lW
    
    if L <= 0:
        raise RuntimeError('Cannot get indices of a tensor with zero or negative spatial size (L='+str(L)+
            ') for kernel_size='+str(kernel_size)+', stride='+str(stride)+', padding='+str(padding)+
            ', dilation='+str(dilation)+' and shape='+str(a_shape))

    i0 = np.repeat(np.arange(0, kernel_size[0]*dilation[0], dilation[0]), kernel_size[1])
    i0 = np.tile(i0, C)
    i1 = stride[0] * np.repeat(np.arange(lH), lW)
    j0 = np.tile(np.arange(0, kernel_size[1]*dilation[1], dilation[1]), kernel_size[0] * C)
    j1 = stride[1] * np.tile(np.arange(lW), lH)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), kernel_size[0] * kernel_size[1]).reshape(-1, 1)

    return (k, i, j)


def im2col(a:np.ndarray, kernel_size, dilation=1, stride=1, padding=0, pad_value=0, col_indices=None, return_indices=False, as_unfold=False) -> np.ndarray:
    """
    Maps an input matrix to a column matrix using a specified kernel size.

    Parameters
    ----------
    a : np.ndarray
        Input array of shape (N, C, H, W).
    kernel_size : int or tuple
        Size of the kernel for the mapping. If an integer is provided, the kernel will be a square of size (kernel_size, kernel_size). If a tuple is provided, it should specify the height and width of the kernel (kernel_size[0], kernel_size[1]).
    dilation : int or tuple, optional (default=1)
        Dilation rate for the kernel. It specifies the spacing between kernel elements.
    stride : int or tuple, optional (default=1)
        Stride value for the kernel. It specifies the step size between successive kernel positions.
    padding : int or tuple, optional (default=0)
        Padding value or tuple of values for the input array. If an integer is provided, the same padding value will be applied to all sides. If a tuple is provided, it should specify the padding value for the top, bottom, left, and right sides, respectively.
    pad_value : scalar, optional (default=0)
        Value used for padding the input array.
    col_indices : tuple, optional
        Precomputed indices for the column mapping. If not provided, they will be calculated using the input shape and other parameters.
    return_indices : bool, optional (default=False)
        If True, the function returns the column matrix and the computed indices used for mapping.
    as_unfold : bool, optional (default=False)
        If True, the output matrix will have shape (N, C*kH*kW, L), where L is the number of mapped positions

    Returns
    -------
    np.ndarray
        The mapped column matrix. If `return_indices` is True, a tuple `(cols, col_indices)` is returned.

    Output size
    -----------
    - 2D: The usual shape of the im2col output matrix.
    - 3D: If `as_unfold` is True, the output shape is (N, C*kH*kW, L), where L is the number of mapped positions.

    Example
    -------
    >>> a = np.array([
            [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]]
        ])
    >>> im2col(a, kernel_size=2, stride=2)
    Output:
    array([[1., 2., 4., 5.],
           [2., 3., 5., 6.],
           [4., 5., 7., 8.],
           [5., 6., 8., 9.]])
    """
    
    padding = np.broadcast_to(padding, 2)
    # Pad input
    x_padded = np.pad(
        a, ((0, 0), (0, 0)) + tuple((padding[d], padding[d]) for d in range(2)),
        mode='constant', constant_values=pad_value)
    
    if col_indices is None:
        col_indices = get_im2col_indices(a.shape, kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride)
        
    k, i, j = col_indices
    cols = x_padded[:, k, i, j]
    
    if not as_unfold:
        C = a.shape[1]
        cols = cols.transpose(1, 2, 0).reshape(kernel_size[0] * kernel_size[1] * C, -1)
    
    out = cols if not return_indices else (cols, col_indices)
    return out


def col2im(a:np.ndarray, output_shape, kernel_size, dilation, stride, padding, col_indices=None, return_indices=False):
    """
    Maps a column matrix back to the original input matrix shape.

    Parameters
    ----------
    a : np.ndarray
        Column matrix of shape 2D or 3D=(N, C*kH*kW, L).
    output_shape : tuple
        Desired shape of the output matrix.
    kernel_size : int or tuple
        Size of the kernel used in the original im2col operation.
    dilation : int
        Dilation rate used in the original im2col operation.
    stride : int
        Stride value used in the original im2col operation.
    padding : int or tuple
        Padding value or tuple of values used in the original im2col operation.
    col_indices : tuple, optional
        Precomputed indices used in the im2col operation. If not provided, they will be calculated.
    return_indices : bool, optional (default=False)
        If True, the function returns the output matrix and the computed indices used for mapping.

    Returns
    -------
    np.ndarray
        The output matrix. If `return_indices` is True, a tuple `(output, col_indices)` is returned.

    Raises
    ------
    ValueError
        If the shape of the input tensor is invalid (should be 2 or 3-dimensional).

    Example
    -------
    >>> a = np.array([
        [a b c d]
        [e f g h]  
        [i j k l]
        [m n o p]
    ])
    >>> col2im(a, output_shape=(1, 1, 3, 3), kernel_size=2, dilation=1, stride=2, padding=0)
    Output:
    array([[[[ a      e+b      f ]
             [i+c  m+j+k+g+d  n+h]
             [ k      o+l      p ]]]])
    """
    kernel_size = np.broadcast_to(kernel_size, 2)
    padding = np.broadcast_to(padding, 2)
    
    if len(a.shape) == 2:
        mode = 'col2im'
        N, C, H, W = output_shape
    elif len(a.shape) == 3:
        mode = 'fold'
        if len(output_shape) == 2:
            N, CkHkW, L = a.shape
            C = CkHkW // np.prod(kernel_size)
            H, W = output_shape
        else:
            N, C, H, W = output_shape
    else:
        raise ValueError('Invalid shape of input tensor (should be 2 or 3-dimensional)')
        
    H_with_pad = H + 2 * padding[0]
    W_with_pad = W + 2 * padding[1]
    
    # Initialize output tensor
    output = np.zeros((N, C, H_with_pad, W_with_pad), dtype=a.dtype)
    
    if col_indices is None:
        col_indices = get_im2col_indices((N, C, H, W), kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride)
    
    k, i, j = col_indices 
    
    cols_reshaped = a
    if mode == 'col2im':
        cols_reshaped = a.reshape(C * kernel_size[0] * kernel_size[1], -1, N)
        cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    
    np.add.at(output, (slice(None), k, i, j), cols_reshaped) # This is the bottleneck of this function
    
    if padding[0] > 0 or padding[1] > 0:
        output = output[:, :, padding[0]:H_with_pad-padding[0], padding[1]:W_with_pad-padding[1]]

    out = output if not return_indices else (output, col_indices)
    return out

# *****************************
# ******* Generic tools *******
# *****************************

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


def recursively_seek_tensors(*inputs):
    tensors = []
    for arg in inputs:
        if isinstance(arg, Tensor): tensors.append(arg)
        elif isinstance(arg, list) or isinstance(arg, tuple):
            tensors += recursively_seek_tensors(*arg)
    return tensors 