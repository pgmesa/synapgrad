import numpy as np
from . import Tensor

epsilon = 1e-12

# ----------------------------- Helpers -----------------------------
# -------------------------------------------------------------------
def unbroadcast(grad:np.ndarray, shape:tuple, to_keep:int=0) -> np.ndarray:
    if len(grad.shape) < len(shape):
        grad = np.zeros(shape) + grad
    else:
        while len(grad.shape) > len(shape):
            grad = grad.sum(axis=0)
        for i in range(len(shape) - to_keep):
            if grad.shape[i] != shape[i]:
                grad = grad.sum(axis=i, keepdims=True)
    return grad


def check_inputs(func):
    
    def check(*args, **kwargs):
        for arg in args:
            if isinstance(arg, Tensor):
                raise TypeError(f"Tensor is not supported as input in ops_cpu.py")
        return func(*args, **kwargs)
    
    return check

# --------------------------- Operations ----------------------------
# -------------------------------------------------------------------

# *************************
# ******* Basic ops *******
# *************************

@check_inputs
def add_forward(a, b):
    return a + b

def add_backward(grad, a_shape, b_shape):
    grad_a = np.ones(a_shape) * grad.data
    grad_b = np.ones(b_shape) * grad.data
    return unbroadcast(grad_a, a_shape), unbroadcast(grad_b, b_shape)


@check_inputs
def mul_forward(a, b):
    return a * b

def mul_backward(grad, a, b):
    grad_a = grad * b
    grad_b = grad * a
    return unbroadcast(grad_a, a.shape), unbroadcast(grad_b, b.shape)


@check_inputs
def matmul_forward(a, b):
    return a @ b

def matmul_backward(grad, a, b):
    print(grad.shape, a.shape, b.shape)
    grad_a = grad @ np.moveaxis(b, -2, -1)
    grad_b = np.moveaxis(a, -2, -1) @ grad
    return unbroadcast(grad_a, a.shape), unbroadcast(grad_b, b.shape)


@check_inputs
def pow_forward(a, n):
    return a ** n

def pow_backward(grad, a, n):
    return n * (a ** (n - 1)) * grad


@check_inputs
def rpow_forward(a, n):
    return n ** a

def rpow_backward(grad, exp_n_a, n):
    return (exp_n_a * np.log(n)) * grad


@check_inputs
def neg_forward(a):
    return -a

def neg_backward(grad):
    return -grad


@check_inputs
def slice_forward(a, s):
    return a[s]

def slice_backward(grad, a_shape, s):
    grad_a = np.zeros(a_shape)
    grad_a[s] = grad
    return grad_a

# *************************
# ******* Other ops *******
# *************************

@check_inputs
def log_forward(a):
    return np.log(a)

def log_backward(grad, a):
    return grad / a


@check_inputs
def exp_forward(a):
    return np.exp(a)

def exp_backward(grad, exp_a):
    return grad * exp_a

############################

@check_inputs
def slice_forward(a, indices):
    ...
    #return inner_slice(a, indices)


@check_inputs
def unsqueeze_forward(a, axis):
    return np.expand_dims(a, axis)


@check_inputs
def squeeze_forward(a, axis):
    return np.squeeze(a, axis)


@check_inputs
def slice_forward(a, slice):
    return a[slice]


@check_inputs
def transpose_forward(a):
    return a.T


@check_inputs
def reshape_forward(a, shape):
    return a.reshape(shape)


@check_inputs
def max_forward(a, axis):
    out = np.amax(a, axis=None if axis is None else tuple(axis), keepdims=True) 
    if axis is not None:
        out = out.reshape([a.shape[i] for i in range(len(a.shape)) if i not in axis])
    return out


@check_inputs
def min_forward(a, axis):
    out = np.amin(a, axis=None if axis is None else tuple(axis), keepdims=True)
    if axis is not None:
        out = out.reshape([a.shape[i] for i in range(len(a.shape)) if i not in axis])
    return out


@check_inputs
def sum_forward(a, axis):
    if axis is None:
        return np.array(a.sum())
    return a.sum(axis=axis)

# ************************
# **** Activations *******
# ************************

def relu_forward(data:np.ndarray) -> np.ndarray:
    return np.maximum(0, data)


def tanh_forward(data:np.ndarray) -> np.ndarray:
    return np.tanh(data)


def sigmoid_forward(data:np.ndarray) -> np.ndarray:
    return 1/(1 + np.exp(-data))


def softmax_forward(data:np.ndarray, dim:int) -> np.ndarray:
    # Shift to make it numerically stable (with large values 'inf' appears)
    shiftx = data - data.max(axis=dim, keepdims=True) 
    exps = np.exp(shiftx)
    exp_sums = exps.sum(axis=dim, keepdims=True)
    return exps / exp_sums


def log_softmax_forward(data:np.ndarray, dim:int) -> np.ndarray:
    # Using log-sum-exp trick for numerical stability
    max_val = data.max(axis=dim, keepdims=True)
    substract = data - max_val
    exp = np.exp(substract)
    lse = max_val + np.log(exp.sum(axis=dim, keepdims=True))
    log_softmax = data - lse
    return log_softmax

# *******************
# **** Losses *******
# *******************

def mse_loss_forward(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    loss = (y_pred - y_true)**2
    return loss
    
    
def nll_loss_forward(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    loss = -y_pred[range(len(y_pred)), y_true].reshape((-1, 1))
    return loss


def bce_loss_forward(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    assert y_pred.max() <= 1 and y_pred.min() >= 0, "BCELoss inputs must be between 0 and 1"
    loss = - (y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))
    # For compatibility with pytorch (returns 100 when y_pred=0 and y_true=1; vice versa)
    loss = np.where(loss == -np.log(epsilon), 100, loss) 
    return loss


def bce_with_logits_loss_forward(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    tn = -relu_forward(y_pred)
    loss = (1-y_true) * y_pred + tn + np.log(np.exp(-tn) + np.exp((-y_pred-tn)))
    return loss


def cross_entropy_loss_forward(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    softmax = np.log(softmax_forward(y_pred, 1) + epsilon)
    log_likelihood = nll_loss_forward(softmax, y_true)
    return log_likelihood

# *********************
# **** Conv ops *******
# *********************

def unfold_forward(tensor:np.ndarray, kernel_size, stride=1, padding=0, dilation=1, pad_value=0) -> np.ndarray:
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


def fold_forward(tensor:np.ndarray, output_size, kernel_size, stride=1, padding=0, dilation=1):
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

# ---------------------------- Backward -----------------------------
# -------------------------------------------------------------------
