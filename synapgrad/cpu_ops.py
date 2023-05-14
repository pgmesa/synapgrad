import numpy as np
from synapgrad.tensor import Tensor
from synapgrad.tools import recursively_seek_tensors, get_selected_from_indices

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

@check_inputs
def add_forward(a:np.ndarray, b:np.ndarray):
    return a + b

@check_inputs
def add_backward(grad:np.ndarray, a_shape:tuple, b_shape:tuple):
    grad_a = np.ones(a_shape, dtype=grad.dtype) * grad
    grad_b = np.ones(b_shape, dtype=grad.dtype) * grad
    return unbroadcast(grad_a, a_shape), unbroadcast(grad_b, b_shape)


@check_inputs
def mul_forward(a:np.ndarray, b:np.ndarray):
    return a * b

@check_inputs
def mul_backward(grad:np.ndarray, a:np.ndarray, b:np.ndarray):
    grad_a = grad * b
    grad_b = grad * a
    return unbroadcast(grad_a, a.shape), unbroadcast(grad_b, b.shape)


@check_inputs
def matmul_forward(a:np.ndarray, b:np.ndarray):
    return a @ b

@check_inputs
def matmul_backward(grad:np.ndarray, a:np.ndarray, b:np.ndarray):
    grad_a = grad @ np.swapaxes(b, -2, -1)
    grad_b = np.swapaxes(a, -2, -1) @ grad
    return unbroadcast(grad_a, a.shape), unbroadcast(grad_b, b.shape)


@check_inputs
def pow_forward(a:np.ndarray, n:'int | float'):
    return a ** n

@check_inputs
def pow_backward(grad:np.ndarray, a:np.ndarray, n:'int | float'):
    return n * (a ** (n - 1)) * grad


@check_inputs
def rpow_forward(a:np.ndarray, n:'int | float'):
    return n ** a

@check_inputs
def rpow_backward(grad:np.ndarray, exp_n_a:np.ndarray, n:'int | float'):
    return (exp_n_a * np.log(n)) * grad


@check_inputs
def neg_forward(a:np.ndarray):
    return -a

@check_inputs
def neg_backward(grad:np.ndarray):
    return -grad


@check_inputs
def slice_forward(a:np.ndarray, s:tuple):
    return a[s]

@check_inputs
def slice_backward(grad:np.ndarray, a_shape:tuple, s:tuple):
    grad_a = np.zeros(a_shape, dtype=grad.dtype)
    grad_a[s] = grad
    return grad_a

# **********************************
# ******* Array manipulation *******
# **********************************

@check_inputs
def concat_forward(a:np.ndarray, axis:int):
    return np.concatenate(a, axis=axis)

@check_inputs
def concat_backward(grad:np.ndarray, sections:list, axis:int):
    sections = [s if i == 0 else sections[i-1] + s for i, s in enumerate(sections)]
    grads = np.split(grad, indices_or_sections=sections, axis=axis)
    return grads


@check_inputs
def stack_forward(a:np.ndarray, axis:int):
    return np.stack(a, axis=axis)

@check_inputs
def stack_backward(grad:np.ndarray, axis:int):
    return unbind_forward(grad, axis)

  
@check_inputs
def unbind_forward(a:np.ndarray, axis:int):
    return np.rollaxis(a, axis=axis)

@check_inputs
def unbind_backward(grad:np.ndarray, a_shape:tuple, axis:int, index:int):
    slice_grad = np.zeros(a_shape, dtype=grad.dtype)
    if axis < 0: axis = len(a_shape) + axis
    axes = tuple([slice(None) if i != axis else index for i in range(len(a_shape))])
    slice_grad[axes] = grad
    return slice_grad

# *************************
# ******* Other ops *******
# *************************

@check_inputs
def clone_forward(a:np.ndarray):
    return a.copy()

@check_inputs
def clone_backward(grad:np.ndarray):
    return grad


@check_inputs
def exp_forward(a:np.ndarray):
    return np.exp(a)

@check_inputs
def exp_backward(grad:np.ndarray, exp_a:np.ndarray):
    return grad * exp_a


@check_inputs
def log_forward(a:np.ndarray):
    return np.log(a + epsilon)

@check_inputs
def log_backward(grad:np.ndarray, a:np.ndarray):
    return grad / (a + epsilon)


@check_inputs
def sqrt_forward(a:np.ndarray):
    return np.sqrt(a)

@check_inputs
def sqrt_backward(grad:np.ndarray, sqrt_a:np.ndarray):
    return grad / (2 * sqrt_a)
    
    
@check_inputs
def sum_forward(a:np.ndarray, axis:'None| int | tuple', keepdims:bool):
    return np.sum(a, axis=axis, keepdims=keepdims)

@check_inputs
def sum_backward(grad:np.ndarray, a_shape:tuple, axis:'None| int | tuple', keepdims:bool):
    out_grad = np.zeros(a_shape, dtype=grad.dtype)
    if not keepdims and axis is not None:
        grad = unsqueeze_forward(grad, axis)

    out_grad = out_grad + grad

    return out_grad


@check_inputs
def mean_forward(a:np.ndarray, axis:'None| int | tuple', keepdims:bool):
    return np.mean(a, axis=axis, keepdims=keepdims)

@check_inputs
def mean_backward(grad:np.ndarray, a_shape:tuple, axis:'None| int | tuple', keepdims:bool):
    out_grad = np.zeros(a_shape, dtype=grad.dtype)
    if not keepdims and axis is not None:
        grad = unsqueeze_forward(grad, axis)
    
    if axis is None: axis = range(len(a_shape))
    if isinstance(axis, int): axis = [axis]
    n_samples = np.prod([a_shape[i] for i in range(len(a_shape)) if i in axis])

    out_grad = out_grad + grad

    return out_grad / n_samples


@check_inputs
def max_forward(a, axis, keepdims):
    return np.max(a, axis=axis, keepdims=keepdims)

@check_inputs
def max_backward(grad, a, axis, keepdims):
    # Create mask of ones and zeros, where the maximum value is 1 
    mask = np.zeros_like(a)
    indices_max = np.argmax(a, axis=axis, keepdims=True)
    if axis is None:
        unr_indices = np.unravel_index(indices_max, a.shape)
        mask[unr_indices] = 1
    else:
        np.put_along_axis(mask, indices_max, 1, axis=axis)
    
    if not keepdims and axis is not None:
        grad = unsqueeze_forward(grad, axis)
    
    return grad * mask


@check_inputs
def min_forward(a, axis, keepdims):
    return np.min(a, axis=axis, keepdims=keepdims)

@check_inputs
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


@check_inputs
def squeeze_forward(a:np.ndarray, axis:'None | int | tuple'):
    return np.squeeze(a, axis)

@check_inputs
def squeeze_backward(grad:np.ndarray, a_shape:tuple):
    return grad.reshape(a_shape)


@check_inputs
def unsqueeze_forward(a:np.ndarray, axis:'int | tuple'):
    return np.expand_dims(a, axis)

@check_inputs
def unsqueeze_backward(grad:np.ndarray, axis:'int | tuple'):
    return np.squeeze(grad, axis)


@check_inputs
def reshape_forward(a:np.ndarray, shape:tuple):
    return a.reshape(shape)

@check_inputs
def reshape_backward(grad:np.ndarray, a_shape:tuple):
    return grad.reshape(a_shape)


@check_inputs
def transpose_forward(a:np.ndarray, axis0:int, axis1:int):
    return np.swapaxes(a, axis0, axis1)

@check_inputs
def transpose_backward(grad:np.ndarray, axis0:int, axis1:int):
    return np.swapaxes(grad, axis0, axis1)
    

@check_inputs
def unfold_forward(a:np.ndarray, dimension:int, size:int, step:int):
    dim_size = a.shape[dimension]
    # check that the size is smaller than or equal to the size of the dimension
    if size > dim_size:
        raise ValueError(f"Size ({size}) must be smaller than or equal to the size of the specified dimension ({dim_size})")
    # calculate the size of the output dimension
    out_size = int((dim_size - size) / step) + 1
    # create an output array with the appropriate shape
    out_shape = list(a.shape)
    out_shape[dimension] = out_size
    out_shape.append(size)
    out_array = np.zeros(out_shape, dtype=a.dtype)
    # fill the output array with the unfolded slices
    for i in range(out_size):
        start = i * step
        end = start + size
        window = np.take(a, np.arange(start, end), axis=dimension)
        window = np.moveaxis(window, dimension, -1)
        out_array = np.delete(out_array, i, axis=dimension)
        out_array = np.insert(out_array, i, window, axis=dimension)
        
    return out_array
    
@check_inputs
def unfold_backward(grad:np.ndarray, a_shape:tuple, dimension:int, size:int, step:int):
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

@check_inputs
def relu_forward(a:np.ndarray) -> np.ndarray:
    return np.maximum(0, a)

@check_inputs
def relu_backward(grad:np.ndarray, a:np.ndarray) -> np.ndarray:
    return grad * (a > 0)


@check_inputs
def tanh_forward(a:np.ndarray) -> np.ndarray:
    return np.tanh(a)

@check_inputs
def tanh_backward(grad:np.ndarray, tanh_a:np.ndarray) -> np.ndarray:
    return grad * (1 - tanh_a**2)


@check_inputs
def sigmoid_forward(a:np.ndarray) -> np.ndarray:
    return 1/(1 + np.exp(-a))

@check_inputs
def sigmoid_backward(grad:np.ndarray, sigmoid_a:np.ndarray) -> np.ndarray:
    return grad * sigmoid_a * (1 - sigmoid_a)


@check_inputs
def softmax_forward(a:np.ndarray, axis:int) -> np.ndarray:
    # Shift to make it numerically stable (with large values 'inf' appears)
    shiftx = a - a.max(axis=axis, keepdims=True) 
    exps = np.exp(shiftx)
    exp_sums = exps.sum(axis=axis, keepdims=True)
    return exps / exp_sums

@check_inputs
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


@check_inputs
def log_softmax_forward(a:np.ndarray, axis:int) -> np.ndarray:
    max_val = a.max(axis=axis, keepdims=True)
    substract = a - max_val
    exp = np.exp(substract)
    lse = max_val + np.log(exp.sum(axis=axis, keepdims=True))
    log_softmax = a - lse
    return log_softmax

@check_inputs
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

@check_inputs
def mse_loss_forward(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    loss = (y_pred - y_true)**2
    return loss

@check_inputs
def mse_loss_backward(grad: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    return grad * 2 * (y_pred - y_true)
    
    
@check_inputs
def nll_loss_forward(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    loss = -y_pred[range(len(y_pred)), y_true].reshape((-1, 1))
    return loss

@check_inputs
def nll_loss_backward(grad: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    loss_grad = np.zeros(y_pred.shape)
    loss_grad[range(len(y_pred)), y_true] = -1.0
    return grad * loss_grad


@check_inputs
def bce_loss_forward(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    loss = - (y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))
    # For compatibility with pytorch (returns 100 when y_pred=0 and y_true=1; vice versa)
    loss = np.where(loss == -np.log(epsilon), 100, loss) 
    return loss

@check_inputs
def bce_loss_backward(grad: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    term_0 = -(1 - y_true + epsilon) / ((1 - y_pred) + epsilon)
    term_1 = (y_true + epsilon) / (y_pred + epsilon)
    loss_grad = -(term_0 + term_1) * grad
    return loss_grad


@check_inputs
def bce_with_logits_loss_forward(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    tn = -relu_forward(y_pred)
    loss = (1-y_true) * y_pred + tn + np.log(np.exp(-tn) + np.exp((-y_pred-tn)))
    return loss

@check_inputs
def bce_with_logits_loss_backward(grad: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    tn = -relu_forward(y_pred)
    dtn = np.where(tn == 0, 0, -1)
    div1 = -dtn*np.exp(-tn) + (-1-dtn)*np.exp((-y_pred-tn))
    div2 = np.exp(-tn) + np.exp((-y_pred-tn))
    loss_grad = (1 - y_true) + dtn + (div1/(div2 + epsilon))
    return grad * loss_grad


@check_inputs
def cross_entropy_loss_forward(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    softmax = np.log(softmax_forward(y_pred, 1) + epsilon)
    log_likelihood = nll_loss_forward(softmax, y_true)
    return log_likelihood

@check_inputs
def cross_entropy_loss_backward(grad: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    dlogits = softmax_forward(y_pred, 1)
    n = y_pred.shape[0]
    dlogits[range(n), y_true] -= 1
    return  dlogits * grad

# ************************
# ******* Pool ops *******
# ************************



# ************************
# ******* Conv ops *******
# ************************

def unfold4d_forward(tensor:np.ndarray, kernel_size, stride=1, padding=0, dilation=1, pad_value=0) -> np.ndarray:
    """
    Unfold a tensor of shape (N, C, H, W) to a tensor in the shape of (N, C*kH*kW, L)
    
    Reference: 
        https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html

    Args:
        tensor (numpy.ndarray): Input tensor of shape (N, C, H, W).
        kernel_size (int or tuple): Size of the sliding window.
        stride (int or tuple, optional): Stride size. Defaults to 1.
        padding (int or tuple, optional): Padding size. Defaults to 0.
        dilation (int or tuple, optional): Dilation factor. Defaults to 1.

    Returns:
        numpy.ndarray: Output tensor of shape (N, C*kH*kW, L).
    """
    assert len(tensor.shape) == 4, "Input tensor must be of shape (N, C, H, W)"
    N, C, H, W = tensor.shape
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
            ', dilation='+str(dilation)+' and tensor shape='+str(tensor.shape))
    
    # Pad input
    padded_input = np.pad(
        tensor, ((0, 0), (0, 0)) + tuple((padding[d], padding[d]) for d in range(2)),
        mode='constant', constant_values=pad_value)
    
    # Initialize output tensor
    output_size = (N, C * np.prod(kernel_size), L)
    output = np.zeros(output_size, dtype=tensor.dtype)
    
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
            
    return output


def fold4d_forward(tensor:np.ndarray, output_size, kernel_size, stride=1, padding=0, dilation=1):
    """
    Fold a tensor of shape (N, C*kH*kW, L) to a tensor in the shape of (N, C, H, W).
    
    Reference:
        https://pytorch.org/docs/stable/generated/torch.nn.Fold.html

    Args:
        tensor (numpy.ndarray): Input tensor of shape (N, C*kH*kW, L).
        kernel_size (int or tuple): Size of the sliding window.
        output_size (tuple): Desired output size of the folded tensor, in the form of (H, W).
        stride (int or tuple, optional): Stride size. Defaults to 1.
        padding (int or tuple, optional): Padding size. Defaults to 0.
        dilation (int or tuple, optional): Dilation factor. Defaults to 1.

    Returns:
        numpy.ndarray: Output tensor of shape (N, C, H, W).
    """
    N, CkHkW, L = tensor.shape
    kernel_size = np.broadcast_to(kernel_size, 2)
    dilation = np.broadcast_to(dilation, 2)
    padding = np.broadcast_to(padding, 2)
    stride = np.broadcast_to(stride, 2)

    C = CkHkW // np.prod(kernel_size)
    H, W = output_size
    
    H_with_pad = H + 2 * padding[0]
    W_with_pad = W + 2 * padding[1]
    
    # Calculate input spatial size
    lH = int(np.floor((H_with_pad - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1))
    lW = int(np.floor((W_with_pad - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1))

    # Initialize output tensor
    output = np.zeros((N, C, H_with_pad, W_with_pad), dtype=tensor.dtype)

    # Reshape input tensor to match the expected shape of the output tensor
    tensor = tensor.reshape((N, C, np.prod(kernel_size), L))

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
            window = tensor[:, :, :, i*lW + j].reshape(o.shape)
            output[:, :, h_start:h_end:h_step, w_start:w_end:w_step] = o + window

    # Remove padding if necessary
    if padding[0] > 0 or padding[1] > 0:
        output = output[:, :, padding[0]:H_with_pad-padding[0], padding[1]:W_with_pad-padding[1]]

    return output