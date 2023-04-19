import numpy as np


epsilon = 1e-12

# ---------------------------- Functions ----------------------------
# -------------------------------------------------------------------
def relu_fn(data:np.ndarray) -> np.ndarray:
    return np.maximum(0, data)


def tanh_fn(data:np.ndarray) -> np.ndarray:
    return np.tanh(data)


def sigmoid_fn(data:np.ndarray) -> np.ndarray:
    return 1/(1 + np.exp(-data))


def softmax_fn(data:np.ndarray, dim:int) -> np.ndarray:
    # Shift to make it numerically stable (with large values 'inf' appears)
    shiftx = data - data.max(axis=dim, keepdims=True) 
    exps = np.exp(shiftx)
    exp_sums = exps.sum(axis=dim, keepdims=True)
    return exps / exp_sums


def log_softmax_fn(data:np.ndarray, dim:int) -> np.ndarray:
    # Using log-sum-exp trick for numerical stability
    max_val = data.max(axis=dim, keepdims=True)
    substract = data - max_val
    exp = np.exp(substract)
    lse = max_val + np.log(exp.sum(axis=dim, keepdims=True))
    log_softmax = data - lse
    return log_softmax

# ----------------------------- Losses ------------------------------
# -------------------------------------------------------------------
def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    loss = (y_pred - y_true)**2
    return loss
    
    
def nll_loss(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    loss = -y_pred[range(len(y_pred)), y_true].reshape((-1, 1))
    return loss


def bce_loss(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    assert y_pred.max() <= 1 and y_pred.min() >= 0, "BCELoss inputs must be between 0 and 1"
    loss = - (y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))
    # For compatibility with pytorch (returns 100 when y_pred=0 and y_true=1; vice versa)
    loss = np.where(loss == -np.log(epsilon), 100, loss) 
    return loss


def bce_with_logits_loss(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    tn = -relu_fn(y_pred)
    loss = (1-y_true) * y_pred + tn + np.log(np.exp(-tn) + np.exp((-y_pred-tn)))
    return loss


def cross_entropy_loss(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    softmax = np.log(softmax_fn(y_pred, 1) + epsilon)
    log_likelihood = nll_loss(softmax, y_true)
    return log_likelihood

# ----------------------- Special Functions -------------------------
# -------------------------------------------------------------------
def unfold(tensor:np.ndarray, kernel_size, stride=1, padding=0, dilation=1, pad_value=0) -> np.ndarray:
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


def fold(tensor:np.ndarray, output_size, kernel_size, stride=1, padding=0, dilation=1):
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