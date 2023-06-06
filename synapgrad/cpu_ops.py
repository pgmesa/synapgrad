import numpy as np
from synapgrad.conv_tools import extract_windows, place_windows

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
# --------------------------- Operations ----------------------------
# -------------------------------------------------------------------

# *************************
# ******* Basic ops *******
# *************************

def add_forward(a:np.ndarray, b:np.ndarray):
    return a + b

def add_backward(grad:np.ndarray, a_shape:tuple, b_shape:tuple):
    grad_a = np.ones(a_shape, dtype=grad.dtype) * grad
    grad_b = np.ones(b_shape, dtype=grad.dtype) * grad
    return unbroadcast(grad_a, a_shape), unbroadcast(grad_b, b_shape)


def mul_forward(a:np.ndarray, b:np.ndarray):
    return a * b

def mul_backward(grad:np.ndarray, a:np.ndarray, b:np.ndarray):
    grad_a = grad * b
    grad_b = grad * a
    return unbroadcast(grad_a, a.shape), unbroadcast(grad_b, b.shape)


def matmul_forward(a:np.ndarray, b:np.ndarray):
    return a @ b

def matmul_backward(grad:np.ndarray, a:np.ndarray, b:np.ndarray):
    grad_a = grad @ np.swapaxes(b, -2, -1)
    grad_b = np.swapaxes(a, -2, -1) @ grad
    return unbroadcast(grad_a, a.shape), unbroadcast(grad_b, b.shape)


def addmm_forward(a:np.ndarray, b:np.ndarray, c:np.ndarray):
    return a + (b @ c)

def addmm_backward(grad:np.ndarray, a:np.ndarray, b:np.ndarray, c:np.ndarray):
    grad_a, grad_mm = add_backward(grad, a.shape, (b.shape[0], c.shape[1],))
    grad_b, grad_c = matmul_backward(grad_mm, b, c)
    return grad_a, grad_b, grad_c


def pow_forward(a:np.ndarray, n:'int | float'):
    return a ** n

def pow_backward(grad:np.ndarray, a:np.ndarray, n:'int | float'):
    return n * (a ** (n - 1)) * grad


def rpow_forward(a:np.ndarray, n:'int | float'):
    return n ** a

def rpow_backward(grad:np.ndarray, exp_n_a:np.ndarray, n:'int | float'):
    return (exp_n_a * np.log(n)) * grad


def neg_forward(a:np.ndarray):
    return -a

def neg_backward(grad:np.ndarray):
    return -grad


def slice_forward(a:np.ndarray, s:tuple):
    return a[s]

def slice_backward(grad:np.ndarray, a_shape:tuple, s:tuple):
    grad_a = np.zeros(a_shape, dtype=grad.dtype)
    grad_a[s] = grad
    return grad_a

# **********************************
# ******* Array manipulation *******
# **********************************

def concat_forward(a:np.ndarray, axis:int):
    return np.concatenate(a, axis=axis)

def concat_backward(grad:np.ndarray, sections:list, axis:int):
    grads = np.split(grad, indices_or_sections=sections, axis=axis)
    return grads


def stack_forward(a:np.ndarray, axis:int):
    return np.stack(a, axis=axis)

def stack_backward(grad:np.ndarray, axis:int):
    return unbind_forward(grad, axis)

  
def unbind_forward(a:np.ndarray, axis:int):
    return np.rollaxis(a, axis=axis)

def unbind_backward(grad:np.ndarray, a_shape:tuple, axis:int, index:int):
    slice_grad = np.zeros(a_shape, dtype=grad.dtype)
    if axis < 0: axis = len(a_shape) + axis
    axes = tuple([slice(None) if i != axis else index for i in range(len(a_shape))])
    slice_grad[axes] = grad
    return slice_grad

# *************************
# ******* Other ops *******
# *************************

def clone_forward(a:np.ndarray):
    return a.copy()

def clone_backward(grad:np.ndarray):
    return grad


def exp_forward(a:np.ndarray):
    return np.exp(a)

def exp_backward(grad:np.ndarray, exp_a:np.ndarray):
    return grad * exp_a


def log_forward(a:np.ndarray):
    return np.log(a + epsilon)

def log_backward(grad:np.ndarray, a:np.ndarray):
    return grad / (a + epsilon)


def sqrt_forward(a:np.ndarray):
    return np.sqrt(a)

def sqrt_backward(grad:np.ndarray, sqrt_a:np.ndarray):
    return grad / (2 * sqrt_a)
    
    
def sum_forward(a:np.ndarray, axis:'None| int | tuple', keepdims:bool):
    return np.sum(a, axis=axis, keepdims=keepdims)

def sum_backward(grad:np.ndarray, a_shape:tuple, axis:'None| int | tuple', keepdims:bool):
    out_grad = np.zeros(a_shape, dtype=grad.dtype)
    if not keepdims and axis is not None:
        grad = unsqueeze_forward(grad, axis)

    out_grad = out_grad + grad

    return out_grad


def mean_forward(a:np.ndarray, axis:'None| int | tuple', keepdims:bool):
    return np.mean(a, axis=axis, keepdims=keepdims)

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


def max_forward(a, axis, keepdims):
    return np.max(a, axis=axis, keepdims=keepdims)

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


def min_forward(a, axis, keepdims):
    return np.min(a, axis=axis, keepdims=keepdims)

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


def squeeze_forward(a:np.ndarray, axis:'None | int | tuple'):
    out = a
    can_apply = len(a.shape) > 0 and (axis is None or a.shape[axis] == 1)
    if can_apply: out = np.squeeze(a, axis)
    return out

def squeeze_backward(grad:np.ndarray, a_shape:tuple):
    return grad.reshape(a_shape)


def unsqueeze_forward(a:np.ndarray, axis:'int | tuple'):
    return np.expand_dims(a, axis)

def unsqueeze_backward(grad:np.ndarray, axis:'int | tuple'):
    return np.squeeze(grad, axis)

# *************************
# ******* View ops ********
# *************************

def reshape_forward(a:np.ndarray, shape:tuple):
    return a.reshape(shape)

def reshape_backward(grad:np.ndarray, a_shape:tuple):
    return grad.reshape(a_shape)


def movedim_forward(a:np.ndarray, source:int, destination:int):
    return np.moveaxis(a, source, destination)

def movedim_backward(grad:np.ndarray, source:int, destination:int):
    return np.moveaxis(grad, source, destination)


def transpose_forward(a:np.ndarray, axis0:int, axis1:int):
    return np.swapaxes(a, axis0, axis1)

def transpose_backward(grad:np.ndarray, axis0:int, axis1:int):
    return np.swapaxes(grad, axis0, axis1)
    

def unfold_dim_forward(a:np.ndarray, dimension:int, size:int, step:int):
    dim_size = a.shape[dimension]
    # check that the size is smaller than or equal to the size of the dimension
    if size > dim_size:
        raise ValueError(f"Size ({size}) must be smaller than or equal to the size of the specified dimension ({dim_size})")
    slices = [slice(None)] * a.ndim; slices[dimension] = slice(None, None, step)
    slices = tuple(slices)
    out_array = np.lib.stride_tricks.sliding_window_view(a, size, axis=dimension, writeable=True)[slices]
        
    return out_array
    
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

def relu_forward(a:np.ndarray) -> np.ndarray:
    return np.maximum(0, a)

def relu_backward(grad:np.ndarray, a:np.ndarray) -> np.ndarray:
    return grad * (a > 0)


def leaky_relu_forward(a:np.ndarray, neg_slope:float) -> np.ndarray:
    return np.maximum(neg_slope * a, a)

def leaky_relu_backward(grad:np.ndarray, a:np.ndarray, neg_slope:float) -> np.ndarray:
    return grad * ((a > 0) + neg_slope * (a <= 0))


def selu_forward(a:np.ndarray, alpha:float, scale:float) -> np.ndarray:
    return scale * (np.maximum(0, a) + np.minimum(0, alpha * (np.exp(a) - 1)))

def selu_backward(grad:np.ndarray, a:np.ndarray, alpha:float, scale:float) -> np.ndarray:
    return scale * grad *((a > 0) +  alpha * np.exp(a) * (a <= 0))


def tanh_forward(a:np.ndarray) -> np.ndarray:
    return np.tanh(a)

def tanh_backward(grad:np.ndarray, tanh_a:np.ndarray) -> np.ndarray:
    return grad * (1 - tanh_a**2)


def sigmoid_forward(a:np.ndarray) -> np.ndarray:
    return 1/(1 + np.exp(-a))

def sigmoid_backward(grad:np.ndarray, sigmoid_a:np.ndarray) -> np.ndarray:
    return grad * sigmoid_a * (1 - sigmoid_a)


def softmax_forward(a:np.ndarray, axis:int) -> np.ndarray:
    # Shift to make it numerically stable (with large values 'inf' appears)
    shiftx = a - a.max(axis=axis, keepdims=True) 
    exps = np.exp(shiftx)
    exp_sums = exps.sum(axis=axis, keepdims=True)
    return exps / exp_sums

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


def log_softmax_forward(a:np.ndarray, axis:int) -> np.ndarray:
    max_val = a.max(axis=axis, keepdims=True)
    substract = a - max_val
    exp = np.exp(substract)
    lse = max_val + np.log(exp.sum(axis=axis, keepdims=True))
    log_softmax = a - lse
    return log_softmax

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

def mse_loss_forward(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    loss = (y_pred - y_true)**2
    return loss

def mse_loss_backward(grad: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    return grad * 2 * (y_pred - y_true)
    
    
def nll_loss_forward(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    loss = -y_pred[range(len(y_pred)), y_true].reshape((-1, 1))
    return loss

def nll_loss_backward(grad: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    loss_grad = np.zeros(y_pred.shape)
    loss_grad[range(len(y_pred)), y_true] = -1.0
    return grad * loss_grad


def bce_loss_forward(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    loss = - (y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))
    # For compatibility with pytorch (returns 100 when y_pred=0 and y_true=1; vice versa)
    loss = np.where(loss == -np.log(epsilon), 100, loss) 
    return loss

def bce_loss_backward(grad: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    term_0 = -(1 - y_true + epsilon) / ((1 - y_pred) + epsilon)
    term_1 = (y_true + epsilon) / (y_pred + epsilon)
    loss_grad = -(term_0 + term_1) * grad
    return loss_grad


def bce_with_logits_loss_forward(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    tn = -relu_forward(y_pred)
    loss = (1-y_true) * y_pred + tn + np.log(np.exp(-tn) + np.exp((-y_pred-tn)))
    return loss

def bce_with_logits_loss_backward(grad: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    tn = -relu_forward(y_pred)
    dtn = np.where(tn == 0, 0, -1)
    div1 = -dtn*np.exp(-tn) + (-1-dtn)*np.exp((-y_pred-tn))
    div2 = np.exp(-tn) + np.exp((-y_pred-tn))
    loss_grad = (1 - y_true) + dtn + (div1/(div2 + epsilon))
    return grad * loss_grad


def cross_entropy_loss_forward(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    softmax = np.log(softmax_forward(y_pred, 1) + epsilon)
    log_likelihood = nll_loss_forward(softmax, y_true)
    return log_likelihood

def cross_entropy_loss_backward(grad: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    dlogits = softmax_forward(y_pred, 1)
    n = y_pred.shape[0]
    dlogits[range(n), y_true] -= 1
    return  dlogits * grad
    
# ************************
# ******* Pool ops *******
# ************************
    
def max_pool1d_forward(a, kernel_size, stride, padding, dilation):
    windows = extract_windows(a, kernel_size, stride, padding, dilation, pad_value=-np.inf)
    maxed_windows = windows.max(axis=-1).transpose(1, 2, 0)
    return maxed_windows, a.shape, windows
    
def max_pool1d_backward(grad, kernel_size, stride, padding, dilation, a_shape, windows):
    grad = grad.transpose(2, 0, 1)
    windows_grad = max_backward(grad, windows, -1, False)
    out_grad = place_windows(windows_grad, a_shape, kernel_size, stride, padding, dilation)
    return out_grad
    
    
def avg_pool1d_forward(a, kernel_size, stride, padding, dilation):
    windows = extract_windows(a, kernel_size, stride, padding, dilation, pad_value=0)
    averaged_windows = windows.mean(axis=-1).transpose(1, 2, 0)
    return averaged_windows, a.shape, windows

def avg_pool1d_backward(grad, kernel_size, stride, padding, dilation, a_shape, windows):
    grad = grad.transpose(2, 0, 1)
    windows_grad = mean_backward(grad, windows.shape, -1, False)
    out_grad = place_windows(windows_grad, a_shape, kernel_size, stride, padding, dilation)
    return out_grad


def max_pool2d_forward(a, kernel_size, stride, padding, dilation):
    windows = extract_windows(a, kernel_size, stride, padding, dilation, pad_value=-np.inf)
    maxed_windows = windows.reshape(*windows.shape[:-2], -1).max(axis=-1).transpose(2, 3, 0, 1)
    return maxed_windows, a.shape, windows

def max_pool2d_backward(grad, kernel_size, stride, padding, dilation, a_shape, windows):
    grad = grad.transpose(2, 3, 0, 1)
    windows_grad = max_backward(grad, windows.reshape(*windows.shape[:-2], -1), -1, False)
    windows_grad = windows_grad.reshape(windows.shape)
    out_grad = place_windows(windows_grad, a_shape, kernel_size, stride, padding, dilation)
    return out_grad
    

def avg_pool2d_forward(a, kernel_size, stride, padding, dilation):
    windows = extract_windows(a, kernel_size, stride, padding, dilation, pad_value=0)
    averaged_windows = windows.reshape(*windows.shape[:-2], -1).mean(axis=-1).transpose(2, 3, 0, 1)
    return averaged_windows, a.shape, windows

def avg_pool2d_backward(grad, kernel_size, stride, padding, dilation, a_shape, windows):
    grad = grad.transpose(2, 3, 0, 1)
    windows_grad = mean_backward(grad, windows.reshape(*windows.shape[:-2], -1).shape, -1, False)
    windows_grad = windows_grad.reshape(windows.shape)
    out_grad = place_windows(windows_grad, a_shape, kernel_size, stride, padding, dilation)
    return out_grad

# ************************
# ******* Conv ops *******
# ************************

def conv1d_forward(a, weight, bias, stride, padding, dilation):
    C_out, C_in, kW = weight.shape
    kernel_size = kW
    
    windows = extract_windows(a, kernel_size, stride, padding, dilation)
    
    conv_out = np.tensordot(weight, windows, axes=[(1,2), (2,3)])
    if bias is not None: conv_out += bias.reshape(-1, 1, 1)
    out = np.moveaxis(conv_out, source=-1, destination=0)
    
    return out, windows
    
def conv1d_backward(grad, a_shape, weight, bias, stride, padding, dilation, windows):
    C_out, C_in, kW = weight.shape
    kernel_size = kW
    
    # input grad
    a_grad_windows = np.tensordot(grad, weight, axes=[[1], [0]])
    a_grad_windows = np.moveaxis(a_grad_windows, source=0, destination=1)
    a_grad = place_windows(a_grad_windows, a_shape, kernel_size, stride, padding, dilation)
    
    # weight grad
    weight_grad = np.tensordot(grad, windows, axes=[(2,0), (0,1)])
    # bias grad
    bias_grad = None
    if bias is not None:
        bias_grad = grad.sum(axis=(0,2))
        
    return a_grad, weight_grad, bias_grad
    
    
def conv2d_forward(a, weight, bias, stride, padding, dilation):
    """ 
    Reference to N dimensional convolution:
        - https://github.com/rsokl/MyGrad/blob/master/src/mygrad/nnet/layers/conv.py
    """
    C_out, C_in, kH, kW = weight.shape
    kernel_size = (kH, kW)
    
    windows = extract_windows(a, kernel_size, stride, padding, dilation)

    conv_out = np.tensordot(weight, windows, axes=[(1,2,3), (3,4,5)])
    if bias is not None: conv_out += bias.reshape(-1, 1, 1, 1)
    out = np.moveaxis(conv_out, source=-1, destination=0)
            
    return out, windows

def conv2d_backward(grad, a_shape, weight, bias, stride, padding, dilation, windows):
    C_out, C_in, kH, kW = weight.shape
    kernel_size = (kH, kW)
    
    # input grad
    a_grad_windows = np.tensordot(grad, weight, axes=[[1], [0]])
    a_grad_windows = np.moveaxis(a_grad_windows, source=0, destination=2)
    a_grad = place_windows(a_grad_windows, a_shape, kernel_size, stride, padding, dilation)
    
    # weight grad
    weight_grad = np.tensordot(grad, windows, axes=[(2,3,0), (0,1,2)])
    # bias grad
    bias_grad = None
    if bias is not None:
        bias_grad = grad.sum(axis=(0,2,3))
    
    return a_grad, weight_grad, bias_grad
    
# ******************************
# ******* Batch norm ops *******
# ******************************

def batch_norm_forward(x, gamma, beta, running_mean, running_var, training, momentum, eps):
    normed_dims = tuple(i for i in range(x.ndim) if i != 1)
    keepdims_shape = tuple(1 if n != 1 else d for n, d in enumerate(x.shape))
    n = x.size / x.shape[1]
    
    # normalize x
    mean = running_mean if running_mean is not None and not training else x.mean(axis=normed_dims)
    var = running_var if running_var is not None and not training else x.var(axis=normed_dims)
    std = np.sqrt(var + eps)
    
    x_norm = (x - mean.reshape(keepdims_shape)) / std.reshape(keepdims_shape)
    # optional affine transformation
    if gamma is not None:
        x_norm *= gamma.reshape(keepdims_shape)
    if beta is not None:
        x_norm += beta.reshape(keepdims_shape)
        
    if running_mean is not None and training:
        running_mean = mean * momentum + running_mean * (1 - momentum)
    if running_var is not None and training:
        unbiased_var = var * (n / (n - 1))
        running_var = unbiased_var * momentum + running_var * (1 - momentum)
    
    return x_norm, running_mean, running_var, mean, var

def batch_norm_backward(grad, x, gamma, beta, track_running_stats, training, eps, mean, variance):
    """ 
    References:
        - https://stackoverflow.com/questions/67968913/derivative-of-batchnorm2d-in-pytorch
        - https://github.com/renan-cunha/BatchNormalization/blob/master/src/feed_forward/layers.py
    """
    normed_dims = tuple(i for i in range(x.ndim) if i != 1)
    keepdims_shape = tuple(1 if n != 1 else d for n, d in enumerate(x.shape))
    n = x.size / x.shape[1]
    
    mean = mean.reshape(keepdims_shape)
    variance = variance.reshape(keepdims_shape)
    x_norm = (x - mean) / np.sqrt(variance + eps)
    
    dL_dxi_hat = grad
    dL_dgamma = None
    if gamma is not None:
        dL_dxi_hat = grad * gamma.reshape(keepdims_shape)
        dL_dgamma = (grad * x_norm).sum(normed_dims)
        
    dL_dbeta = None
    if beta is not None:
        dL_dbeta = grad.sum(normed_dims)
    
    if training or not track_running_stats:
        dL_dvar = (-0.5 * dL_dxi_hat * (x - mean)).sum(normed_dims, keepdims=True)  * ((variance + eps) ** -1.5)
        dL_davg = (-1.0 / np.sqrt(variance + eps) * dL_dxi_hat).sum(normed_dims, keepdims=True) + (dL_dvar * (-2.0 * (x - mean)).sum(normed_dims, keepdims=True) / n)
        dL_dxi = (dL_dxi_hat / np.sqrt(variance + eps)) + (2.0 * dL_dvar * (x - mean) / n) + (dL_davg / n)
    else:
        dL_dxi = dL_dxi_hat
    
    return dL_dxi, dL_dgamma, dL_dbeta