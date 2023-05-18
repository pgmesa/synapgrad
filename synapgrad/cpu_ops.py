import numpy as np
from synapgrad.tools import (
    recursively_seek_tensors,
    get_conv2d_output_size, get_conv1d_output_size,
    im2col, col2im, arr2col, col2arr
)

epsilon = 1e-12

# ----------------------------- Helpers -----------------------------
# -------------------------------------------------------------------
def unbroadcast(grad:np.ndarray, shape:tuple) -> np.ndarray:
    if len(grad.shape) < len(shape):
        grad = np.zeros(shape) + grad
    else:
        while len(grad.shape) > len(shape):
            grad = grad.sum(axis=0)
        for i in range(len(shape)):
            if grad.shape[i] != shape[i]:
                grad = grad.sum(axis=i, keepdims=True)
    return grad

def check_inputs(func):
    
    def check(*args, **kwargs):
        found_tensors = recursively_seek_tensors(*args)
        if len(found_tensors) > 0:
            raise TypeError(f"Tensor is not supported as input in cpu_ops.py")
        return func(*args, **kwargs)
    
    return check

# --------------------------- Operations ----------------------------
# -------------------------------------------------------------------

# *************************
# ******* Basic ops *******
# *************************

# @check_inputs
def add_forward(a:np.ndarray, b:np.ndarray):
    return a + b

# @check_inputs
def add_backward(grad:np.ndarray, a_shape:tuple, b_shape:tuple):
    grad_a = np.ones(a_shape, dtype=grad.dtype) * grad
    grad_b = np.ones(b_shape, dtype=grad.dtype) * grad
    return unbroadcast(grad_a, a_shape), unbroadcast(grad_b, b_shape)


# @check_inputs
def mul_forward(a:np.ndarray, b:np.ndarray):
    return a * b

# @check_inputs
def mul_backward(grad:np.ndarray, a:np.ndarray, b:np.ndarray):
    grad_a = grad * b
    grad_b = grad * a
    return unbroadcast(grad_a, a.shape), unbroadcast(grad_b, b.shape)


# @check_inputs
def matmul_forward(a:np.ndarray, b:np.ndarray):
    return a @ b

# @check_inputs
def matmul_backward(grad:np.ndarray, a:np.ndarray, b:np.ndarray):
    grad_a = grad @ np.swapaxes(b, -2, -1)
    grad_b = np.swapaxes(a, -2, -1) @ grad
    return unbroadcast(grad_a, a.shape), unbroadcast(grad_b, b.shape)


# @check_inputs
def addmm_forward(a:np.ndarray, b:np.ndarray, c:np.ndarray):
    return a + (b @ c)

# @check_inputs
def addmm_backward(grad:np.ndarray, a:np.ndarray, b:np.ndarray, c:np.ndarray):
    grad_a, grad_mm = add_backward(grad, a.shape, (b.shape[0], c.shape[1],))
    grad_b, grad_c = matmul_backward(grad_mm, b, c)
    return grad_a, grad_b, grad_c


# @check_inputs
def pow_forward(a:np.ndarray, n:'int | float'):
    return a ** n

# @check_inputs
def pow_backward(grad:np.ndarray, a:np.ndarray, n:'int | float'):
    return n * (a ** (n - 1)) * grad


# @check_inputs
def rpow_forward(a:np.ndarray, n:'int | float'):
    return n ** a

# @check_inputs
def rpow_backward(grad:np.ndarray, exp_n_a:np.ndarray, n:'int | float'):
    return (exp_n_a * np.log(n)) * grad


# @check_inputs
def neg_forward(a:np.ndarray):
    return -a

# @check_inputs
def neg_backward(grad:np.ndarray):
    return -grad


# @check_inputs
def slice_forward(a:np.ndarray, s:tuple):
    return a[s]

# @check_inputs
def slice_backward(grad:np.ndarray, a_shape:tuple, s:tuple):
    grad_a = np.zeros(a_shape, dtype=grad.dtype)
    grad_a[s] = grad
    return grad_a

# **********************************
# ******* Array manipulation *******
# **********************************

# @check_inputs
def concat_forward(a:np.ndarray, axis:int):
    return np.concatenate(a, axis=axis)

# @check_inputs
def concat_backward(grad:np.ndarray, sections:list, axis:int):
    grads = np.split(grad, indices_or_sections=sections, axis=axis)
    return grads


# @check_inputs
def stack_forward(a:np.ndarray, axis:int):
    return np.stack(a, axis=axis)

# @check_inputs
def stack_backward(grad:np.ndarray, axis:int):
    return unbind_forward(grad, axis)

  
# @check_inputs
def unbind_forward(a:np.ndarray, axis:int):
    return np.rollaxis(a, axis=axis)

# @check_inputs
def unbind_backward(grad:np.ndarray, a_shape:tuple, axis:int, index:int):
    slice_grad = np.zeros(a_shape, dtype=grad.dtype)
    if axis < 0: axis = len(a_shape) + axis
    axes = tuple([slice(None) if i != axis else index for i in range(len(a_shape))])
    slice_grad[axes] = grad
    return slice_grad

# *************************
# ******* Other ops *******
# *************************

# @check_inputs
def clone_forward(a:np.ndarray):
    return a.copy()

# @check_inputs
def clone_backward(grad:np.ndarray):
    return grad


# @check_inputs
def exp_forward(a:np.ndarray):
    return np.exp(a)

# @check_inputs
def exp_backward(grad:np.ndarray, exp_a:np.ndarray):
    return grad * exp_a


# @check_inputs
def log_forward(a:np.ndarray):
    return np.log(a + epsilon)

# @check_inputs
def log_backward(grad:np.ndarray, a:np.ndarray):
    return grad / (a + epsilon)


# @check_inputs
def sqrt_forward(a:np.ndarray):
    return np.sqrt(a)

# @check_inputs
def sqrt_backward(grad:np.ndarray, sqrt_a:np.ndarray):
    return grad / (2 * sqrt_a)
    
    
# @check_inputs
def sum_forward(a:np.ndarray, axis:'None| int | tuple', keepdims:bool):
    return np.sum(a, axis=axis, keepdims=keepdims)

# @check_inputs
def sum_backward(grad:np.ndarray, a_shape:tuple, axis:'None| int | tuple', keepdims:bool):
    out_grad = np.zeros(a_shape, dtype=grad.dtype)
    if not keepdims and axis is not None:
        grad = unsqueeze_forward(grad, axis)

    out_grad = out_grad + grad

    return out_grad


# @check_inputs
def mean_forward(a:np.ndarray, axis:'None| int | tuple', keepdims:bool):
    return np.mean(a, axis=axis, keepdims=keepdims)

# @check_inputs
def mean_backward(grad:np.ndarray, a_shape:tuple, axis:'None| int | tuple', keepdims:bool):
    out_grad = np.zeros(a_shape, dtype=grad.dtype)
    if not keepdims and axis is not None:
        grad = unsqueeze_forward(grad, axis)
    
    if axis is None: axis = range(len(a_shape))
    if isinstance(axis, int): 
        if axis < 0: axis = len(a_shape) + axis
        axis = [axis]
    n_samples = np.prod([a_shape[i] for i in range(len(a_shape)) if i in axis])

    out_grad = out_grad + grad

    return out_grad / n_samples


# @check_inputs
def max_forward(a, axis, keepdims):
    return np.max(a, axis=axis, keepdims=keepdims)

# @check_inputs
def max_backward(grad, a, axis, keepdims, max_indices=None):
    # Create mask of ones and zeros, where the maximum value is 1 
    mask = np.zeros_like(a)
    if max_indices is None:
        max_indices = np.argmax(a, axis=axis, keepdims=True)
    if axis is None:
        unr_indices = np.unravel_index(max_indices, a.shape)
        mask[unr_indices] = 1
    else:
        np.put_along_axis(mask, max_indices, 1, axis=axis)
    
    if not keepdims and axis is not None:
        grad = unsqueeze_forward(grad, axis)
    
    return grad * mask


# @check_inputs
def min_forward(a, axis, keepdims):
    return np.min(a, axis=axis, keepdims=keepdims)

# @check_inputs
def min_backward(grad, a, axis, keepdims):
    # Create mask of ones and zeros, where the minimum value is 1 
    mask = np.zeros_like(a)
    indices_min = np.argmin(a, axis=axis, keepdims=True)
    if axis is None:
        unr_indices = np.unravel_index(indices_min, a.shape)
        mask[unr_indices] = 1
    else:
        np.put_along_axis(mask, indices_min, 1, axis=axis)
    
    if not keepdims and axis is not None:
        grad = unsqueeze_forward(grad, axis)
    
    return grad * mask


# @check_inputs
def squeeze_forward(a:np.ndarray, axis:'None | int | tuple'):
    out = a
    can_apply = len(a.shape) > 0 and (axis is None or a.shape[axis] == 1)
    if can_apply: out = np.squeeze(a, axis)
    return out

# @check_inputs
def squeeze_backward(grad:np.ndarray, a_shape:tuple):
    return grad.reshape(a_shape)


# @check_inputs
def unsqueeze_forward(a:np.ndarray, axis:'int | tuple'):
    return np.expand_dims(a, axis)

# @check_inputs
def unsqueeze_backward(grad:np.ndarray, axis:'int | tuple'):
    return np.squeeze(grad, axis)


# @check_inputs
def reshape_forward(a:np.ndarray, shape:tuple):
    return a.reshape(shape)

# @check_inputs
def reshape_backward(grad:np.ndarray, a_shape:tuple):
    return grad.reshape(a_shape)


# @check_inputs
def movedim_forward(a:np.ndarray, source:int, destination:int):
    return np.moveaxis(a, source, destination)

# @check_inputs
def movedim_backward(grad:np.ndarray, source:int, destination:int):
    return np.moveaxis(grad, source, destination)


# @check_inputs
def transpose_forward(a:np.ndarray, axis0:int, axis1:int):
    return np.swapaxes(a, axis0, axis1)

# @check_inputs
def transpose_backward(grad:np.ndarray, axis0:int, axis1:int):
    return np.swapaxes(grad, axis0, axis1)
    

# @check_inputs
def unfold_dim_forward(a:np.ndarray, dimension:int, size:int, step:int):
    dim_size = a.shape[dimension]
    # check that the size is smaller than or equal to the size of the dimension
    if size > dim_size:
        raise ValueError(f"Size ({size}) must be smaller than or equal to the size of the specified dimension ({dim_size})")
    slices = [slice(None)] * a.ndim; slices[dimension] = slice(None, None, step)
    slices = tuple(slices)
    out_array = np.lib.stride_tricks.sliding_window_view(a, size, axis=dimension, writeable=True)[slices]
        
    return out_array
    
# @check_inputs
def unfold_dim_backward(grad:np.ndarray, a_shape:tuple, dimension:int, size:int, step:int):
    a_grad = np.zeros(a_shape)
    for i in range(grad.shape[dimension]):
        start = i * step
        end = start + size
        s1 = [slice(None)] * (dimension + 1); s1[dimension] = slice(start, end)
        s2 = [slice(None)] * (dimension + 1); s2[dimension] = i
        s1 = tuple(s1); s2 = tuple(s2)
        a_grad[s1] += np.moveaxis(grad[s2], -1, dimension).reshape(a_grad[s1].shape)
        
    return a_grad

# ********************************
# ******* Activation ops *********
# ********************************

# @check_inputs
def relu_forward(a:np.ndarray) -> np.ndarray:
    return np.maximum(0, a)

# @check_inputs
def relu_backward(grad:np.ndarray, a:np.ndarray) -> np.ndarray:
    return grad * (a > 0)


# @check_inputs
def tanh_forward(a:np.ndarray) -> np.ndarray:
    return np.tanh(a)

# @check_inputs
def tanh_backward(grad:np.ndarray, tanh_a:np.ndarray) -> np.ndarray:
    return grad * (1 - tanh_a**2)


# @check_inputs
def sigmoid_forward(a:np.ndarray) -> np.ndarray:
    return 1/(1 + np.exp(-a))

# @check_inputs
def sigmoid_backward(grad:np.ndarray, sigmoid_a:np.ndarray) -> np.ndarray:
    return grad * sigmoid_a * (1 - sigmoid_a)


# @check_inputs
def softmax_forward(a:np.ndarray, axis:int) -> np.ndarray:
    # Shift to make it numerically stable (with large values 'inf' appears)
    shiftx = a - a.max(axis=axis, keepdims=True) 
    exps = np.exp(shiftx)
    exp_sums = exps.sum(axis=axis, keepdims=True)
    return exps / exp_sums

# @check_inputs
def softmax_backward(grad:np.ndarray, softmax_a:np.ndarray, axis:int) -> np.ndarray:
    """ 
    References: 
    - https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
    - https://aimatters.wordpress.com/2019/06/17/the-softmax-function-derivative/
    """
    jacobians = np.stack([np.diag(y) - np.outer(y, y) for y in softmax_a])
    out_grad = np.expand_dims(grad, axis=axis)
    a_grad = (out_grad @ jacobians).sum(axis=axis)
    return a_grad


# @check_inputs
def log_softmax_forward(a:np.ndarray, axis:int) -> np.ndarray:
    max_val = a.max(axis=axis, keepdims=True)
    substract = a - max_val
    exp = np.exp(substract)
    lse = max_val + np.log(exp.sum(axis=axis, keepdims=True))
    log_softmax = a - lse
    return log_softmax

# @check_inputs
def log_softmax_backward(grad:np.ndarray, log_softmax_a:np.ndarray, axis:int) -> np.ndarray:
    softmax = np.exp(log_softmax_a)
    jacobians = np.stack([np.diag(y) - np.outer(y, y) for y in softmax])
    dlog_dsoftmax = (1/(softmax + epsilon)) * grad
    dlog_dsoftmax = np.expand_dims(dlog_dsoftmax, axis=axis)
    a_grad = (dlog_dsoftmax @ jacobians).sum(axis=axis)
    return a_grad

# **************************
# ******* Loss ops *********
# **************************

# @check_inputs
def mse_loss_forward(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    loss = (y_pred - y_true)**2
    return loss

# @check_inputs
def mse_loss_backward(grad: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    return grad * 2 * (y_pred - y_true)
    
    
# @check_inputs
def nll_loss_forward(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    loss = -y_pred[range(len(y_pred)), y_true].reshape((-1, 1))
    return loss

# @check_inputs
def nll_loss_backward(grad: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    loss_grad = np.zeros(y_pred.shape)
    loss_grad[range(len(y_pred)), y_true] = -1.0
    return grad * loss_grad


# @check_inputs
def bce_loss_forward(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    loss = - (y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))
    # For compatibility with pytorch (returns 100 when y_pred=0 and y_true=1; vice versa)
    loss = np.where(loss == -np.log(epsilon), 100, loss) 
    return loss

# @check_inputs
def bce_loss_backward(grad: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    term_0 = -(1 - y_true + epsilon) / ((1 - y_pred) + epsilon)
    term_1 = (y_true + epsilon) / (y_pred + epsilon)
    loss_grad = -(term_0 + term_1) * grad
    return loss_grad


# @check_inputs
def bce_with_logits_loss_forward(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    tn = -relu_forward(y_pred)
    loss = (1-y_true) * y_pred + tn + np.log(np.exp(-tn) + np.exp((-y_pred-tn)))
    return loss

# @check_inputs
def bce_with_logits_loss_backward(grad: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    tn = -relu_forward(y_pred)
    dtn = np.where(tn == 0, 0, -1)
    div1 = -dtn*np.exp(-tn) + (-1-dtn)*np.exp((-y_pred-tn))
    div2 = np.exp(-tn) + np.exp((-y_pred-tn))
    loss_grad = (1 - y_true) + dtn + (div1/(div2 + epsilon))
    return grad * loss_grad


# @check_inputs
def cross_entropy_loss_forward(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    softmax = np.log(softmax_forward(y_pred, 1) + epsilon)
    log_likelihood = nll_loss_forward(softmax, y_true)
    return log_likelihood

# @check_inputs
def cross_entropy_loss_backward(grad: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    dlogits = softmax_forward(y_pred, 1)
    n = y_pred.shape[0]
    dlogits[range(n), y_true] -= 1
    return  dlogits * grad

# ************************
# ******* Conv ops *******
# ************************

# @check_inputs
def conv1d_forward(a, weight, bias, stride, padding, dilation):
    N, C, W = a.shape
    C_out, C_in, kW = weight.shape
    kernel_size = kW
    
    L = get_conv1d_output_size(a.shape[2], kernel_size, stride, padding, dilation)
    cols, indices = arr2col(a, kernel_size, dilation, stride, padding, return_indices=True)
    
    weight_cols = weight.reshape(C_out, kW * C_in).T
    cols = cols.transpose(0, 2, 1, 3).reshape(N, L, -1)
    
    out = (cols @ weight_cols)
    if bias is not None:
        out += bias 
    out = out.transpose(0, 2, 1).reshape(N, C_out, L)

    return out, cols, indices
    
# @check_inputs
def conv1d_backward(grad, a_shape, weight, bias, stride, padding, dilation, cols, unf_indices):
    N, C, W = a_shape
    C_out, C_in, kW = weight.shape
    kernel_size = kW
    
    L = get_conv1d_output_size(a_shape[2], kernel_size, stride, padding, dilation)
    
    grad = grad.reshape(N, C_out, -1).transpose(0, 2, 1)
    weight_cols = weight.reshape(C_out, kW * C_in).T
    cols_grad, weight_grad = matmul_backward(grad, cols, weight_cols)

    weight_grad = weight_grad.T.reshape(weight.shape)
    bias_grad = None
    if bias is not None:
        bias_grad = grad.sum(axis=(0, 1))
        
    cols_grad = cols_grad.reshape(N, L, C_in, kW).transpose(0, 2, 1, 3)
    a_grad = col2arr(cols_grad, a_shape, kernel_size, dilation, stride, padding, unf_indices=unf_indices)
    
    return a_grad, weight_grad, bias_grad
    
    
# @check_inputs
def conv2d_forward(a, weight, bias, stride, padding, dilation):
    N, C, H, W = a.shape
    C_out, C_in, kH, kW = weight.shape
    kernel_size = (kH, kW)
    
    lH,lW = get_conv2d_output_size(a.shape, kernel_size, dilation, stride, padding)
    cols, indices = im2col(a, kernel_size, dilation, stride, padding, return_indices=True, as_unfold=True)
    
    weight_cols = weight.reshape(C_out, kH * kW * C_in).T
    out = (cols.transpose(0, 2, 1) @ weight_cols)
    
    if bias is not None:
        out += bias
    out = out.transpose(0, 2, 1).reshape(N, C_out, lH, lW)
    
    return out, cols, indices
    
# @check_inputs
def conv2d_backward(grad, a_shape, weight, bias, stride, padding, dilation, cols, col_indices):
    N, C, H, W = a_shape
    C_out, C_in, kH, kW = weight.shape
    kernel_size = (kH, kW)
    
    grad = grad.reshape(N, C_out, -1).transpose(0, 2, 1)
    weight_cols = weight.reshape(C_out, kH * kW * C_in).T
    cols_grad, weight_grad = matmul_backward(grad, cols.transpose(0, 2, 1), weight_cols)
    
    weight_grad = weight_grad.T.reshape(weight.shape)
    bias_grad = None
    if bias is not None:
        bias_grad = grad.sum(axis=(0, 1))
    a_grad = col2im(cols_grad.transpose(0, 2, 1), a_shape, kernel_size, dilation, stride, padding, col_indices=col_indices)
    
    return a_grad, weight_grad, bias_grad
    
# ************************
# ******* Pool ops *******
# ************************
    
# @check_inputs
def max_pool1d_forward(a, kernel_size, stride, padding, dilation):
    unfolded, indices = arr2col(a, kernel_size, dilation, stride, padding, pad_value=-np.inf, return_indices=True)
    out = unfolded.max(axis=3)

    return out, unfolded, indices
    
# @check_inputs
def max_pool1d_backward(grad, kernel_size, stride, padding, dilation, a_shape, unfolded, unf_indices):
    max_grad = max_backward(grad, unfolded, 3, False)
    out_grad = col2arr(max_grad, a_shape, kernel_size, dilation, stride, padding, unf_indices=unf_indices)
    
    return out_grad
    
    
# @check_inputs
def avg_pool1d_forward(a, kernel_size, stride, padding, dilation):
    unfolded, indices = arr2col(a, kernel_size, dilation, stride, padding, pad_value=0, return_indices=True)
    out = unfolded.mean(axis=3)

    return out, unfolded, indices

# @check_inputs
def avg_pool1d_backward(grad, kernel_size, stride, padding, dilation, a_shape, unfolded, unf_indices):
    mean_grad = mean_backward(grad, unfolded.shape, 3, False)
    out_grad = col2arr(mean_grad, a_shape, kernel_size, dilation, stride, padding, unf_indices=unf_indices)
    
    return out_grad


# @check_inputs
def max_pool2d_forward(a, kernel_size, stride, padding, dilation):
    N, C, H, W = a.shape
    lH, lW = get_conv2d_output_size(a.shape, kernel_size, dilation, stride, padding)
    
    x_split = a.reshape(N * C, 1, H, W)
    x_cols, col_indices = im2col(x_split, kernel_size, dilation=dilation, stride=stride,
                                            padding=padding, pad_value=-np.inf, return_indices=True)
    
    x_cols_argmax = np.argmax(x_cols, axis=0, keepdims=True)
    x_cols_max = x_cols[x_cols_argmax, np.arange(x_cols.shape[1])]
    out = x_cols_max.reshape(lH, lW, N, C).transpose(2,3,0,1)
    return out, a.shape, x_cols, x_cols_argmax, col_indices

# @check_inputs
def max_pool2d_backward(grad, kernel_size, stride, padding, dilation, a_shape, x_cols, max_indices, col_indices):
    # flatten the gradient
    grad = grad.transpose(2,3,0,1).ravel()
    out_grad = max_backward(grad, x_cols, 0, False, max_indices=max_indices)
    # Another way to do it: out_grad = np.zeros_like(x_cols); out_grad[max_indices, range(max_indices.size)] = grad
    
    N, C, H, W = a_shape
    # get the original X_reshaped structure from col2im
    shape = (N*C, 1, H, W)
    out_grad = col2im(out_grad, shape, kernel_size, dilation, stride, padding, col_indices=col_indices)
    out_grad = out_grad.reshape(N, C, H, W)
    return out_grad
    

# @check_inputs
def avg_pool2d_forward(a, kernel_size, stride, padding, dilation):
    N, C, H, W = a.shape
    lH, lW = get_conv2d_output_size(a.shape, kernel_size, dilation, stride, padding)
    
    x_split = a.reshape(N * C, 1, H, W)
    x_cols, col_indices = im2col(x_split, kernel_size, dilation=dilation, stride=stride,
                                            padding=padding, pad_value=0, return_indices=True)
    
    x_cols_avg = np.mean(x_cols, axis=0)
    out = x_cols_avg.reshape(lH, lW, N, C).transpose(2,3,0,1)
    return out, a.shape, x_cols.shape, col_indices

# @check_inputs
def avg_pool2d_backward(grad, kernel_size, stride, padding, dilation, a_shape, x_cols_shape, col_indices):
    # flatten the gradient
    grad = grad.transpose(2,3,0,1).ravel()
    out_grad = mean_backward(grad, x_cols_shape, 0, False)
    
    N, C, H, W = a_shape
    # get the original X_reshaped structure from col2im
    shape = (N*C, 1, H, W)
    out_grad = col2im(out_grad, shape, kernel_size, dilation, stride, padding, col_indices=col_indices)
    out_grad = out_grad.reshape(N, C, H, W)
    return out_grad
    
# ******************************
# ******* Batch norm ops *******
# ******************************