import numpy as np


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


# ******************************
# ******* Sliding window *******
# ******************************

def extract_windows(a, kernel_size, step, padding, dilation=1, pad_value=0):
    """ 
    Create a sliding window view over a 3D or 4D tensor
    
    - if `a` is 4D tensor (N, C, H, W) --> (lH, lH, N, C, kH, kW) (6D)
    - if `a` is 3D tensor (N, C, W) --> (lW, N, C, kW) (4D)
    
    Args:
        - a (np.ndarray): input array
        - kernel_size (int or tuple): size of the kernel
        - step (int or tuple): step value
        - padding (int or tuple): padding value
        - dilation (int or tuple): dilation value
        - pad_value (float): padding value
    
    Returns:
        - np.ndarray: sliding window view
    
    Reference: 
        - https://mygrad.readthedocs.io/en/latest/_modules/mygrad/nnet/layers/utils.html
    """
    if len(a.shape) == 3: dims = 1
    elif len(a.shape) == 4: dims = 2
    else: raise ValueError("`a` must be 3D or 4D") 
    
    if not a.flags["C_CONTIGUOUS"]:
        a = np.ascontiguousarray(a)

    kernel_size = np.broadcast_to(kernel_size, dims)
    dilation = np.broadcast_to(dilation, dims)
    step = np.broadcast_to(step, dims)
    padding = np.broadcast_to(padding, dims)
    
    if dims == 1: sizes = get_conv1d_output_size(a.shape[-1], kernel_size, step, padding, dilation)
    elif dims == 2: sizes = get_conv2d_output_size(a.shape, kernel_size, dilation, step, padding)
    L = int(np.prod(sizes))
    
    if L == 0:
        raise RuntimeError('Cannot create windows on a tensor with zero or negative spatial size (L='+str(L)+
            ') for kernel_size='+str(kernel_size)+', stride='+str(stride)+', padding='+str(padding)+
            ', dilation='+str(dilation)+' and tensor shape='+str(a.shape))
    
    a = np.pad(
        a, ((0, 0), (0, 0)) + tuple((padding[d], padding[d]) for d in range(dims)),
        mode='constant', constant_values=pad_value)

    in_shape = np.array(a.shape[-len(step) :])  # (x, ... , z)
    nbyte = a.strides[-1]  # size, in bytes, of element in `arr`

    # per-byte strides required to fill a window
    win_stride = tuple(np.cumprod(a.shape[:0:-1])[::-1]) + (1,)

    # per-byte strides required to advance the window
    step_stride = tuple(win_stride[-len(step) :] * step)

    # update win_stride to accommodate dilation
    win_stride = np.array(win_stride)
    win_stride[-len(step) :] *= dilation
    win_stride = tuple(win_stride)

    stride = tuple(int(nbyte * i) for i in step_stride + win_stride)

    # number of window placements along x-dim: X = (x - (Wx - 1)*Dx + 1) // Sx + 1
    out_shape = tuple((in_shape - ((kernel_size - 1) * dilation + 1)) // step + 1)

    # ([X, (...), Z], ..., [Wx, (...), Wz])
    out_shape = out_shape + a.shape[: -len(step)] + tuple(kernel_size)
    out_shape = tuple(int(i) for i in out_shape)

    return np.lib.stride_tricks.as_strided(a, shape=out_shape, strides=stride, writeable=False)


def place_windows(windows, out_shape, kernel_size, step, padding, dilation=1):
    """ 
    Places back the windows extracted (in `extract_windows` function) into an array of shape=out_shape.
    
    - if `windows` is 6D tensor (lH, lH, N, C, kH, kW) --> (N, C, H, W) (4D)
    - if `windows` is 4D tensor (lW, N, C, kW) --> (N, C, W) (3D)
    
    Args:
        - windows (np.ndarray): extracted windows
        - out_shape (tuple): output shape
        - kernel_size (int or tuple): size of the kernel
        - step (int or tuple): step value
        - padding (int or tuple): padding value
        - dilation (int or tuple): dilation value
    
    Returns:
        - np.ndarray: output array
   
    Example:
        - windows = extract_windows(a, kernel_size, step, padding, dilation)
        - output = place_windows(windows, a.shape, kernel_size, step, padding, dilation)    
    """
    if len(windows.shape) == 4:
        dims = 1
        N, C, W = out_shape
    elif len(windows.shape) == 6:
        dims = 2
        N, C, H, W = out_shape
        
    kernel_size = np.broadcast_to(kernel_size, dims)
    dilation = np.broadcast_to(dilation, dims)
    step = np.broadcast_to(step, dims)
    padding = np.broadcast_to(padding, dims)
    
    windows = np.moveaxis(windows, dims, 0)
    
    if dims == 1: 
        sizes = get_conv1d_output_size(W, kernel_size, step, padding, dilation)
        W_with_pad = W + 2 * padding[0]
        # Initialize output tensor
        output = np.zeros((N, C, W_with_pad), dtype=windows.dtype)
    elif dims == 2: 
        sizes = get_conv2d_output_size(out_shape, kernel_size, dilation, step, padding)
        H_with_pad = H + 2 * padding[0]
        W_with_pad = W + 2 * padding[1]
        # Initialize output tensor
        output = np.zeros((N, C, H_with_pad, W_with_pad), dtype=windows.dtype)
    
    for ind in np.ndindex(sizes):
        slices = tuple(
            slice(i * s, i * s + w * d, d)
            for i, w, s, d in zip(ind, kernel_size, step, dilation)
        )
        output[(..., *slices)] += windows[(slice(None), *ind, ...)]
        
    # remove padding from dx
    if sum(padding):
        no_pads = tuple(slice(p, -p if p else None) for p in padding)
        output = output[(..., *no_pads)]
    
    return output

# **************************
# ******* im2col fns *******
# **************************

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
    
    (N, C, H, W) -> im2col -> (C * kH * kW, N * L) or (N, C * kH * kW, L) where L = lH * lW.

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
    - 2D: The usual shape of the im2col output matrix (C * kH * kW, N * L) where L = lH * lW.
    - 3D: If `as_unfold` is True, the output shape is (N, C * kH * kW, L), where L is the number of mapped positions.

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
    
    (C * kH * kW, N * L) or (N, C * kH * kW, L) -> col2im -> (N, C, H, W)
    
    where L = lH * lW. 

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


def im2col_v2(a:np.ndarray, kernel_size, dilation=1, stride=1, padding=0, pad_value=0, as_unfold=False):
    """ 
    Another version of im2col
    """
    assert len(a.shape) == 4, "Input tensor must be of shape (N, C, H, W)"
    N, C, H, W = a.shape
    kernel_size = np.broadcast_to(kernel_size, 2)
    dilation = np.broadcast_to(dilation, 2)
    padding = np.broadcast_to(padding, 2)
    stride = np.broadcast_to(stride, 2)

    # Calculate output spatial size
    lH = int(np.floor((H + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1))
    lW = int(np.floor((W + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1))
    L = lH*lW
    
    if L <= 0:
        raise RuntimeError('Cannot unfold a tensor with zero or negative spatial size (L='+str(L)+
            ') for kernel_size='+str(kernel_size)+', stride='+str(stride)+', padding='+str(padding)+
            ', dilation='+str(dilation)+' and tensor shape='+str(a.shape))
    
    # Pad input
    padded_input = np.pad(
        a, ((0, 0), (0, 0)) + tuple((padding[d], padding[d]) for d in range(2)),
        mode='constant', constant_values=pad_value)
    
    # Initialize output tensor
    output_size = (N, C * np.prod(kernel_size), L)
    output = np.zeros(output_size, dtype=a.dtype)
    
    # Extract sliding window for each input channel and put it in the output tensor
    for i in range(lH):
        for j in range(lW):
            # Height parameters
            h_start = i * stride[0]
            h_end = i * stride[0] + kernel_size[0] + (dilation[0] - 1) * (kernel_size[0] - 1)
            h_step = dilation[0]
            # Width parameters
            w_start = j * stride[1]
            w_end = j * stride[1] + kernel_size[1] + (dilation[1] - 1) * (kernel_size[1] - 1)
            w_step = dilation[1]
            # Extract sliding window
            window = padded_input[:, :, h_start:h_end:h_step, w_start:w_end:w_step]
            output[:, :, i*lW + j] = window.ravel().reshape(output[:, :, i*lW + j].shape)
            
    if not as_unfold:
        C = a.shape[1]
        output = output.transpose(1, 2, 0).reshape(kernel_size[0] * kernel_size[1] * C, -1)
            
    return output


def col2im_v2(a:np.ndarray, output_shape, kernel_size, dilation, stride, padding):
    """ 
    Another version of col2im
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
    
    dilation = np.broadcast_to(dilation, 2)
    stride = np.broadcast_to(stride, 2)
    
    if mode == 'col2im':
        a = a.reshape(C * kernel_size[0] * kernel_size[1], -1, N)
        a = a.transpose(2, 0, 1)
    
    H_with_pad = H + 2 * padding[0]
    W_with_pad = W + 2 * padding[1]
    
    # Calculate input spatial size
    lH = int(np.floor((H_with_pad - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1))
    lW = int(np.floor((W_with_pad - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1))
    L = lW*lH

    # Initialize output tensor
    output = np.zeros((N, C, H_with_pad, W_with_pad), dtype=a.dtype)

    # Reshape input tensor to match the expected shape of the output tensor
    a = a.reshape((N, C, np.prod(kernel_size), L))

    # Undo the sliding window operation and place the values back in the output tensor
    for i in range(lH):
        for j in range(lW):
            h_start = i * stride[0]
            h_end = i * stride[0] + kernel_size[0] + (dilation[0] - 1) * (kernel_size[0] - 1)
            h_step = dilation[0]
            w_start = j * stride[1]
            w_end = j * stride[1] + kernel_size[1] + (dilation[1] - 1) * (kernel_size[1] - 1)
            w_step = dilation[1]
            # Calculate the output window
            o = output[:, :, h_start:h_end:h_step, w_start:w_end:w_step]
            window = a[:, :, :, i*lW + j].reshape(o.shape)
            output[:, :, h_start:h_end:h_step, w_start:w_end:w_step] = o + window

    # Remove padding if necessary
    if padding[0] > 0 or padding[1] > 0:
        output = output[:, :, padding[0]:H_with_pad-padding[0], padding[1]:W_with_pad-padding[1]]

    return output


def im2col_fast(a:np.ndarray, kernel_size, dilation=1, stride=1, padding=0, pad_value=0, as_unfold=False):
    """ 
    Fast implementation of im2col
    """
    assert len(a.shape) == 4, "Input tensor must be of shape (N, C, H, W)"
    N, C, H, W = a.shape
    
    windows = extract_windows(a, kernel_size=kernel_size, step=stride, padding=padding, dilation=dilation, pad_value=pad_value)
    
    # Calculate output spatial size
    L = int(np.prod(windows.shape[:2]))
    
    if as_unfold:
        cols = np.moveaxis(windows.reshape(L, N, C * kernel_size[0] * kernel_size[1]), 0, 2)
    else:
        cols = windows.reshape(N * L, C * kernel_size[0] * kernel_size[1]).T
    
    return cols


def col2im_fast(a:np.ndarray, output_shape, kernel_size, dilation, stride, padding):
    """ 
    Fast implementation of col2im
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
        
    lW, lH = get_conv2d_output_size(output_shape, kernel_size, dilation, stride, padding)
    
    if mode == 'col2im':
        windows = a.T.reshape(lH, lW, N, C, kernel_size[0], kernel_size[1])
    elif mode == 'fold':
        windows = np.moveaxis(a, 2, 0).reshape(lH, lW, N, C, kernel_size[0], kernel_size[1])
    
    output = place_windows(windows, output_shape, kernel_size, stride, padding, dilation)
    
    return output
    