import numpy as np

from synapgrad import cpu_ops
from synapgrad.tensor import Tensor
from synapgrad.autograd import Function, Context
from synapgrad.device import Device


# ************************************
# ******* Activation functions *******
# ************************************

class ReLU(Function):
    
    @staticmethod
    def forward(ctx:Context, x:Tensor):
        if not isinstance(x, Tensor):
            raise TypeError(f"Expected x to be a Tensor but got {type(x)}")
        
        if x.device == Device.CPU:
            out_data = cpu_ops.relu_forward(x.data)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {x.device} not supported")

        out = Tensor(out_data, device=x.device)

        ctx.save_for_backward(x)
        
        return out
    
    @staticmethod
    def backward(ctx:Context, grad_output:Tensor):
        x, = ctx.saved_tensors
        
        if grad_output.device == Device.CPU:
            a_grad = cpu_ops.relu_backward(grad_output.data, x.data)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {grad_output.device} not supported")
        
        x_grad = Tensor(a_grad, device=grad_output.device)
        
        return x_grad


def relu(x:Tensor):
    """ 
    ReLU activation function. 
    
    The ReLU activation function is defined as:
    f(x) = max(0, x)

    Args:
        x (Tensor): tensor

    Returns:
        Tensor: result
    """
    return ReLU.apply(x)


class Tanh(Function):
    
    @staticmethod
    def forward(ctx:Context, x:Tensor):
        if not isinstance(x, Tensor):
            raise TypeError(f"Expected x to be a Tensor but got {type(x)}")
        
        if x.device == Device.CPU:
            out_data = cpu_ops.tanh_forward(x.data)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {x.device} not supported")

        out = Tensor(out_data, device=x.device)

        ctx.save_for_backward(out)
        
        return out
    
    @staticmethod
    def backward(ctx:Context, grad_output:Tensor):
        out, = ctx.saved_tensors
        
        if grad_output.device == Device.CPU:
            a_grad = cpu_ops.tanh_backward(grad_output.data, out.data)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {grad_output.device} not supported")
        
        x_grad = Tensor(a_grad, device=grad_output.device)
        
        return x_grad
    

def tanh(x:Tensor):
    """ 
    Tanh activation function.

    Args:
        x (Tensor): tensor

    Returns:
        Tensor: result
    """
    return Tanh.apply(x)


class Sigmoid(Function):
    
    @staticmethod
    def forward(ctx:Context, x:Tensor):
        if not isinstance(x, Tensor):
            raise TypeError(f"Expected x to be a Tensor but got {type(x)}")
        
        if x.device == Device.CPU:
            out_data = cpu_ops.sigmoid_forward(x.data)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {x.device} not supported")

        out = Tensor(out_data, device=x.device)

        ctx.save_for_backward(out)
        
        return out
        
    @staticmethod
    def backward(ctx:Context, grad_output:Tensor):
        out, = ctx.saved_tensors
        
        if grad_output.device == Device.CPU:
            a_grad = cpu_ops.sigmoid_backward(grad_output.data, out.data)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {grad_output.device} not supported")
        
        x_grad = Tensor(a_grad, device=grad_output.device)
        
        return x_grad
    
    
def sigmoid(x:Tensor):
    """ 
    Sigmoid activation function. 
    
    The Sigmoid activation function is defined as:
    f(x) = 1 / (1 + exp(-x))

    Args:
        x (Tensor): tensor

    Returns:
        Tensor: result
    """
    return Sigmoid.apply(x)


class Softmax(Function):
    
    @staticmethod
    def forward(ctx:Context, x:Tensor, dim:int):
        if not isinstance(x, Tensor):
            raise TypeError(f"Expected x to be a Tensor but got {type(x)}")
        
        if x.device == Device.CPU:
            out_data = cpu_ops.softmax_forward(x.data, dim)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {x.device} not supported")

        out = Tensor(out_data, device=x.device)

        ctx.save_for_backward(out)
        ctx.dim = dim
        
        return out
        
    @staticmethod
    def backward(ctx:Context, grad_output:Tensor):
        out, = ctx.saved_tensors
        dim = ctx.dim
        
        if grad_output.device == Device.CPU:
            a_grad = cpu_ops.softmax_backward(grad_output.data, out.data, dim)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {grad_output.device} not supported")
        
        x_grad = Tensor(a_grad, device=grad_output.device)
        
        return x_grad
    
    
def softmax(x:Tensor, dim:int):
    """ 
    Softmax activation function. 
    
    The Softmax activation function is defined as:
    f(x) = exp(x) / sum(exp(x))

    Args:
        x (Tensor): tensor

    Returns:
        Tensor: result
    """
    return Softmax.apply(x, dim)


class LogSoftmax(Function):
    
    @staticmethod
    def forward(ctx:Context, x:Tensor, dim:int):
        if not isinstance(x, Tensor):
            raise TypeError(f"Expected x to be a Tensor but got {type(x)}")
        
        if x.device == Device.CPU:
            out_data = cpu_ops.log_softmax_forward(x.data, dim)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {x.device} not supported")

        out = Tensor(out_data, device=x.device)

        ctx.save_for_backward(out)
        ctx.dim = dim
        
        return out
        
    @staticmethod
    def backward(ctx:Context, grad_output:Tensor):
        out, = ctx.saved_tensors
        dim = ctx.dim
        
        if grad_output.device == Device.CPU:
            a_grad = cpu_ops.log_softmax_backward(grad_output.data, out.data, dim)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {grad_output.device} not supported")
        
        x_grad = Tensor(a_grad, device=grad_output.device)
        
        return x_grad


def log_softmax(x:Tensor, dim:int):
    """ 
    LogSoftmax activation function. 
    
    The LogSoftmax activation function is defined as:
    f(x) = log(exp(x) / sum(exp(x))) = log(softmax(x))

    Args:
        x (Tensor): tensor

    Returns:
        Tensor: result
    """
    return LogSoftmax.apply(x, dim)

# ******************************
# ******* Loss functions *******
# ******************************

class MSELoss(Function):
    
    @staticmethod
    def forward(ctx:Context, y_pred:Tensor, y_true:Tensor):
        if not isinstance(y_pred, Tensor):
            raise TypeError(f"Expected y_pred to be a Tensor but got {type(y_pred)}")
        if not isinstance(y_true, Tensor):
            raise TypeError(f"Expected y_true to be a Tensor but got {type(y_true)}")
        
        if not y_pred.matches_shape(y_true):
            raise ValueError(f"Inputs shape don't match y_pred={y_pred.shape}, y_true={y_true.shape}")
        
        if y_pred.device == Device.CPU:
            loss_data = cpu_ops.mse_loss_forward(y_pred.data, y_true.data)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {y_pred.device} not supported")

        loss = Tensor(loss_data, device=y_pred.device)

        ctx.save_for_backward(y_pred, y_true)
        
        return loss
    
    @staticmethod
    def backward(ctx:Context, grad_output:Tensor):
        y_pred, y_true = ctx.saved_tensors
        
        if grad_output.device == Device.CPU:
            loss_grad_data = cpu_ops.mse_loss_backward(grad_output.data, y_pred.data, y_true.data)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {grad_output.device} not supported")
        
        loss_grad = Tensor(loss_grad_data, device=grad_output.device)
        
        return loss_grad
    

def mse_loss(y_pred:Tensor, y_true:Tensor):
    """ 
    Mean Squared Error loss function.

    Args:
        - y_pred (Tensor): tensor
        - y_true (Tensor): tensor

    Returns:
        Tensor: result
    """
    return MSELoss.apply(y_pred, y_true)


class NLLLoss(Function):
    
    @staticmethod
    def forward(ctx:Context, y_pred:Tensor, y_true:Tensor):
        if not isinstance(y_pred, Tensor):
            raise TypeError(f"Expected y_pred to be a Tensor but got {type(y_pred)}")
        if not isinstance(y_true, Tensor):
            raise TypeError(f"Expected y_true to be a Tensor but got {type(y_true)}")
        
        if y_pred.device == Device.CPU:
            loss_data = cpu_ops.nll_loss_forward(y_pred.data, y_true.data)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {y_pred.device} not supported")

        loss = Tensor(loss_data, device=y_pred.device)

        ctx.save_for_backward(y_pred, y_true)
        
        return loss
    
    @staticmethod
    def backward(ctx:Context, grad_output:Tensor):
        y_pred, y_true = ctx.saved_tensors
        
        if grad_output.device == Device.CPU:
            loss_grad_data = cpu_ops.nll_loss_backward(grad_output.data, y_pred.data, y_true.data)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {grad_output.device} not supported")
        
        loss_grad = Tensor(loss_grad_data, device=grad_output.device)
        
        return loss_grad
    

def nll_loss(y_pred:Tensor, y_true:Tensor):
    """ 
    Negative Log Likelihood loss function.

    Args:
        - y_pred (Tensor): tensor
        - y_true (Tensor): tensor

    Returns:
        Tensor: result
    """
    return NLLLoss.apply(y_pred, y_true)


class BCELoss(Function):
    
    @staticmethod
    def forward(ctx:Context, y_pred:Tensor, y_true:Tensor):
        if not isinstance(y_pred, Tensor):
            raise TypeError(f"Expected y_pred to be a Tensor but got {type(y_pred)}")
        if not isinstance(y_true, Tensor):
            raise TypeError(f"Expected y_true to be a Tensor but got {type(y_true)}")
        
        if y_pred.device == Device.CPU:
            loss_data = cpu_ops.bce_loss_forward(y_pred.data, y_true.data)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {y_pred.device} not supported")

        loss = Tensor(loss_data, device=y_pred.device)

        ctx.save_for_backward(y_pred, y_true)
        
        return loss
    
    @staticmethod
    def backward(ctx:Context, grad_output:Tensor):
        y_pred, y_true = ctx.saved_tensors
        
        if grad_output.device == Device.CPU:
            loss_grad_data = cpu_ops.bce_loss_backward(grad_output.data, y_pred.data, y_true.data)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {grad_output.device} not supported")
        
        loss_grad = Tensor(loss_grad_data, device=grad_output.device)
        
        return loss_grad
    

def binary_cross_entropy(y_pred:Tensor, y_true:Tensor):
    """ 
    Binary Cross Entropy loss function.

    Args:
        - y_pred (Tensor): tensor
        - y_true (Tensor): tensor

    Returns:
        Tensor: result
    """
    return BCELoss.apply(y_pred, y_true)


class BCEWithLogitsLoss(Function):
    
    @staticmethod
    def forward(ctx:Context, y_pred:Tensor, y_true:Tensor):
        if not isinstance(y_pred, Tensor):
            raise TypeError(f"Expected y_pred to be a Tensor but got {type(y_pred)}")
        if not isinstance(y_true, Tensor):
            raise TypeError(f"Expected y_true to be a Tensor but got {type(y_true)}")
        
        if y_pred.device == Device.CPU:
            loss_data = cpu_ops.bce_with_logits_loss_forward(y_pred.data, y_true.data)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {y_pred.device} not supported")

        loss = Tensor(loss_data, device=y_pred.device)

        ctx.save_for_backward(y_pred, y_true)
        
        return loss
    
    @staticmethod
    def backward(ctx:Context, grad_output:Tensor):
        y_pred, y_true = ctx.saved_tensors
        
        if grad_output.device == Device.CPU:
            loss_grad_data = cpu_ops.bce_with_logits_loss_backward(grad_output.data, y_pred.data, y_true.data)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {grad_output.device} not supported")
        
        loss_grad = Tensor(loss_grad_data, device=grad_output.device)
        
        return loss_grad
    

def binary_cross_entropy_with_logits(y_pred:Tensor, y_true:Tensor):
    """ 
    Binary Cross Entropy with Logits loss function.

    Args:
        - y_pred (Tensor): tensor
        - y_true (Tensor): tensor

    Returns:
        Tensor: result
    """
    return BCEWithLogitsLoss.apply(y_pred, y_true)


class CrossEntropyLoss(Function):
    
    @staticmethod
    def forward(ctx:Context, y_pred:Tensor, y_true:Tensor):
        if not isinstance(y_pred, Tensor):
            raise TypeError(f"Expected y_pred to be a Tensor but got {type(y_pred)}")
        if not isinstance(y_true, Tensor):
            raise TypeError(f"Expected y_true to be a Tensor but got {type(y_true)}")
        
        if y_pred.device == Device.CPU:
            loss_data = cpu_ops.cross_entropy_loss_forward(y_pred.data, y_true.data)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {y_pred.device} not supported")

        loss = Tensor(loss_data, device=y_pred.device)

        ctx.save_for_backward(y_pred, y_true)
        
        return loss
    
    @staticmethod
    def backward(ctx:Context, grad_output:Tensor):
        y_pred, y_true = ctx.saved_tensors
        
        if grad_output.device == Device.CPU:
            loss_grad_data = cpu_ops.cross_entropy_loss_backward(grad_output.data, y_pred.data, y_true.data)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {grad_output.device} not supported")
        
        loss_grad = Tensor(loss_grad_data, device=grad_output.device)
        
        return loss_grad
    

def cross_entropy(y_pred:Tensor, y_true:Tensor):
    """ 
    Cross Entropy loss function.

    Args:
        - y_pred (Tensor): tensor
        - y_true (Tensor): tensor

    Returns:
        Tensor: result
    """
    return CrossEntropyLoss.apply(y_pred, y_true)

# *********************************
# ******* Linear functions ********
# *********************************

class Linear(Function):
    
    @staticmethod
    def forward(ctx:Context, x:Tensor, weight:Tensor, bias:Tensor=None):
        if not isinstance(x, Tensor):
            raise TypeError(f"Expected x to be a Tensor but got {type(x)}")
        if not isinstance(weight, Tensor):
            raise TypeError(f"Expected weight to be a Tensor but got {type(weight)}")
        if bias is not None and not isinstance(bias, Tensor):
            raise TypeError(f"Expected not None bias to be a Tensor but got {type(bias)}")
        
        if x.device == Device.CPU:
            if bias is not None:
                out_data = cpu_ops.addmm_forward(bias.data, x.data, weight.data.T)
            else:
                out_data = cpu_ops.matmul_forward(x.data, weight.data.T)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {x.device} not supported")

        out = Tensor(out_data, device=x.device, requires_grad=True)

        if bias is not None:
            ctx.save_for_backward(x, weight, bias)
        else:
            ctx.save_for_backward(x, weight)
        
        return out
    
    def backward(ctx:Context, grad_output:Tensor):
        if len(ctx.saved_tensors) == 3:
            x, weight, bias = ctx.saved_tensors
        else:
            x, weight = ctx.saved_tensors
            bias = None
        
        if grad_output.device == Device.CPU:
            if bias is not None:
                bias_grad, x_grad, weight_grad = cpu_ops.addmm_backward(grad_output.data, bias.data, x.data, weight.data.T)
            else:
                x_grad, weight_grad = cpu_ops.matmul_backward(grad_output.data, x.data, weight.data.T)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {grad_output.device} not supported")
        
        grad_input = Tensor(x_grad, device=grad_output.device)
        grad_weight = Tensor(weight_grad.T, device=grad_output.device)
        
        out = [grad_input, grad_weight]
        
        if bias is not None:
            grad_bias = Tensor(bias_grad, device=grad_output.device)
            out.append(grad_bias)
                   
        return tuple(out)
    
    
def linear(x:Tensor, weight:Tensor, bias:Tensor=None):
    """ 
    Linear function. x @ w.T + b

    Args:
        - x (Tensor): tensor
        - weight (Tensor): tensor
        - bias (Tensor): tensor. (default=None)

    Returns:
        Tensor: result
    """
    return Linear.apply(x, weight, bias)

# *******************************
# ******* Conv functions ********
# *******************************

class Unfold(Function):
    
    @staticmethod
    def forward(ctx:Context, x:Tensor, kernel_size, dilation=1, stride=1, padding=0, pad_value=0):
        if not isinstance(x, Tensor):
            raise TypeError(f"Expected x to be a Tensor but got {type(x)}")
        
        if len(x.shape) != 4:
            raise ValueError(f"Input tensor must be of shape (N, C, H, W), but got {x.shape}")
        
        if x.device == Device.CPU:
            out_data, col_indices = \
                cpu_ops.im2col(x.data, kernel_size, dilation, stride, padding, pad_value, return_indices=True, as_unfold=True)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {x.device} not supported")

        out = Tensor(out_data, device=x.device)

        ctx.im_size = x.shape[2:]
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.col_indices = col_indices
        
        return out
    
    @staticmethod
    def backward(ctx:Context, grad_output:Tensor):
        im_size = ctx.im_size
        kernel_size = ctx.kernel_size
        dilation = ctx.dilation
        stride = ctx.stride
        padding = ctx.padding
        col_indices = ctx.col_indices    
        
        if not isinstance(grad_output, Tensor):
            raise TypeError(f"Expected grad_output to be a Tensor but got {type(grad_output)}")
        
        if grad_output.device == Device.CPU:
            grad_input_data = cpu_ops.col2im(grad_output.data, im_size, kernel_size,
                                    dilation, stride, padding, col_indices=col_indices)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {grad_output.device} not supported")
        
        grad_input = Tensor(grad_input_data, device=grad_output.device)
        
        return grad_input
    

def unfold(x:Tensor, kernel_size:'int | tuple', dilation:'int | tuple'=1, stride:'int | tuple'=1, padding:'int | tuple'=0, pad_value=0) -> Tensor:
    """
    Unfold a tensor of shape (N, C, H, W) to a tensor in the shape of (N, C*kH*kW, L)
    
    Reference: 
        https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html

    Args:
        - tensor (numpy.ndarray): Input tensor of shape (N, C, H, W).
        - kernel_size (int or tuple): Size of the sliding window.
        - dilation (int or tuple, optional): Dilation factor. Defaults to 1.
        - stride (int or tuple, optional): Stride size. Defaults to 1.
        - padding (int or tuple, optional): Padding size. Defaults to 0.

    Returns:
        numpy.ndarray: Output tensor of shape (N, C*kH*kW, L).
    """
    return Unfold.apply(x, kernel_size, dilation, stride, padding, pad_value)
    
    
class Fold(Function):
    
    @staticmethod
    def forward(ctx:Context, x:Tensor, im_size, kernel_size, dilation=1, stride=1, padding=0):
        if not isinstance(x, Tensor):
            raise TypeError(f"Expected x to be a Tensor but got {type(x)}")
        
        if len(x.shape) != 3:
            raise ValueError(f"Input tensor must be of shape (N, C*kH*kW, L), but got {x.shape}")
        
        if x.device == Device.CPU:
            out_data, col_indices = \
                cpu_ops.col2im(x.data, im_size, kernel_size, dilation, stride, padding, return_indices=True)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {x.device} not supported")

        out = Tensor(out_data, device=x.device)

        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.col_indices = col_indices
        
        return out
    
    @staticmethod
    def backward(ctx:Context, grad_output:Tensor):
        kernel_size = ctx.kernel_size
        dilation = ctx.dilation
        stride = ctx.stride
        padding = ctx.padding
        col_indices = ctx.col_indices    
    
        if not isinstance(grad_output, Tensor):
            raise TypeError(f"Expected grad_output to be a Tensor but got {type(grad_output)}")
        
        if grad_output.device == Device.CPU:
            grad_input_data = cpu_ops.im2col(grad_output.data, kernel_size, dilation,
                                    stride, padding, 0, col_indices=col_indices, as_unfold=True)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {grad_output.device} not supported")
        
        grad_input = Tensor(grad_input_data, device=grad_output.device)
        
        return grad_input
    
    
def fold(x:Tensor, output_size:tuple, kernel_size:'int | tuple', dilation:'int | tuple'=1, stride:'int | tuple'=1, padding:'int | tuple'=0) -> Tensor:
    """
    Fold a tensor of shape (N, C*kH*kW, L) to a tensor in the shape of (N, C, H, W).
    
    Reference:
        https://pytorch.org/docs/stable/generated/torch.nn.Fold.html

    Args:
        - tensor (numpy.ndarray): Input tensor of shape (N, C*kH*kW, L).
        - output_size (tuple): Desired output size of the folded tensor, in the form of (H, W).
        - kernel_size (int or tuple): Size of the sliding window.
        - dilation (int or tuple, optional): Dilation factor. Defaults to 1.
        - stride (int or tuple, optional): Stride size. Defaults to 1.
        - padding (int or tuple, optional): Padding size. Defaults to 0.

    Returns:
        numpy.ndarray: Output tensor of shape (N, C, H, W).
    """
    return Fold.apply(x, output_size, kernel_size, dilation, stride, padding)


class Conv1d(Function):
    
    @staticmethod
    def forward(ctx:Context, x:Tensor, weight:Tensor, bias:Tensor=None, stride:int=1, padding:int=0, dilation:int=1):
        if not isinstance(x, Tensor):
            raise TypeError(f"Expected x to be a Tensor but got {type(x)}")
        
        if not isinstance(weight, Tensor):
            raise TypeError(f"Expected weight to be a Tensor but got {type(weight)}")
        
        if bias is not None and not isinstance(bias, Tensor):
            raise TypeError(f"Expected bias to be a Tensor but got {type(bias)}")
        
        if x.device == Device.CPU:
            bias_data = bias.data if bias is not None else None
            out_data, *bw_data = cpu_ops.conv1d_forward(x.data, weight.data, bias_data, stride, padding, dilation)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {x.device} not supported")

        out = Tensor(out_data, device=x.device, requires_grad=True)
        
        ctx.x_shape = x.shape
        ctx.weight = weight
        ctx.bias = bias
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.bw_data = bw_data
        
        return out
    
    @staticmethod
    def backward(ctx:Context, grad_output:Tensor):
        x_shape = ctx.x_shape
        weight = ctx.weight
        bias = ctx.bias
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        bw_data = ctx.bw_data
        
        if not isinstance(grad_output, Tensor):
            raise TypeError(f"Expected grad_output to be a Tensor but got {type(grad_output)}")
        
        if grad_output.device == Device.CPU:
            bias_data = bias.data if bias is not None else None
            gradients = cpu_ops.conv1d_backward(grad_output.data, x_shape, weight.data,
                                                        bias_data, stride, padding, dilation, *bw_data)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {grad_output.device} not supported")
        
        grad_tensors = [Tensor(g, device=grad_output.device) for g in gradients]
        
        return grad_tensors
    
    
def conv1d(x:Tensor, weight:Tensor, bias:Tensor=None, stride:int=1, padding:int=0, dilation:int=1) -> Tensor:
    """
    1D convolution.
    
    Reference:
        https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html

    Args:
        - x (Tensor): Input tensor of shape (N, C, W).
        - weight (Tensor): Weight tensor of shape (C_out, C, kW).
        - bias (Tensor, optional): Bias tensor of shape (C_out,). Defaults to None.
        - stride (int, optional): Stride size. Defaults to 1.
        - padding (int, optional): Padding size. Defaults to 0.
        - dilation (int, optional): Dilation factor. Defaults to 1.

    Returns:
        numpy.ndarray: Output tensor of shape (N, C_out, L).
    """
    return Conv1d.apply(x, weight, bias, stride, padding, dilation)


class Conv2d(Function):
    
    @staticmethod
    def forward(ctx:Context, x:Tensor, weight:Tensor, bias:Tensor=None, stride=1, padding=0, dilation=1):
        if not isinstance(x, Tensor):
            raise TypeError(f"Expected x to be a Tensor but got {type(x)}")
        
        if not isinstance(weight, Tensor):
            raise TypeError(f"Expected weight to be a Tensor but got {type(weight)}")
        
        if bias is not None and not isinstance(bias, Tensor):
            raise TypeError(f"Expected bias to be a Tensor but got {type(bias)}")
        
        if x.device == Device.CPU:
            bias_data = bias.data if bias is not None else None
            out_data, *bw_data = cpu_ops.conv2d_forward(x.data, weight.data, bias_data, stride, padding, dilation)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {x.device} not supported")

        out = Tensor(out_data, device=x.device, requires_grad=True)
        
        ctx.x_shape = x.shape
        ctx.weight = weight
        ctx.bias = bias
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.bw_data = bw_data

        return out
    
    @staticmethod
    def backward(ctx:Context, grad_output:Tensor):
        x_shape = ctx.x_shape
        weight = ctx.weight
        bias = ctx.bias
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        bw_data = ctx.bw_data
        
        if not isinstance(grad_output, Tensor):
            raise TypeError(f"Expected grad_output to be a Tensor but got {type(grad_output)}")
        
        if grad_output.device == Device.CPU:
            bias_data = bias.data if bias is not None else None
            gradients = cpu_ops.conv2d_backward(grad_output.data, x_shape, weight.data,
                                                        bias_data, stride, padding, dilation, *bw_data)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {grad_output.device} not supported")
        
        grad_tensors = [Tensor(g, device=grad_output.device) for g in gradients]
        
        return grad_tensors


def conv2d(x:Tensor, weight:Tensor, bias:Tensor=None, stride=1, padding=0, dilation=1) -> Tensor:
    """
    Applies 2D convolution to an input tensor.
    
    Reference:
        https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

    Args:
        - x (Tensor): Input tensor of shape (N, C, H, W).
        - weight (Tensor): Weight tensor of shape (C_out, C, kH, kW).
        - bias (Tensor, optional): Bias tensor of shape (C_out). Defaults to None.
        - stride (tuple | int, optional): Stride size. Defaults to 1.
        - padding (tuple | int, optional): Padding size. Defaults to 0.
        - dilation (tuple | int, optional): Dilation factor. Defaults to 1.

    Returns:
        Tensor: Output tensor of shape (N, C_out, lH, lW).
    """
    return Conv2d.apply(x, weight, bias, stride, padding, dilation)

# *******************************
# ******* Pool functions ********
# *******************************

class MaxPool1d(Function):
    
    @staticmethod
    def forward(ctx:Context, x:Tensor, kernel_size, stride=None, padding=0, dilation=1):
        if not isinstance(x, Tensor):
            raise TypeError(f"Expected x to be a Tensor but got {type(x)}")
        
        if len(x.shape) != 3:
            raise ValueError(f"Input tensor must be of shape (N, C, L), but got {x.shape}")
        
        if x.device == Device.CPU:
            out_data, *bw_data = cpu_ops.max_pool1d_forward(x.data, kernel_size, stride, padding, dilation)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {x.device} not supported")

        out = Tensor(out_data, device=x.device, requires_grad=True)
        
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.x_shape = x.shape
        ctx.bw_data = bw_data

        return out
    
    @staticmethod
    def backward(ctx:Context, grad_output:Tensor):
        kernel_size = ctx.kernel_size
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        x_shape = ctx.x_shape
        bw_data = ctx.bw_data
        
        if not isinstance(grad_output, Tensor):
            raise TypeError(f"Expected grad_output to be a Tensor but got {type(grad_output)}")
        
        if grad_output.device == Device.CPU:
            grad_input_data = \
                cpu_ops.max_pool1d_backward(grad_output.data, kernel_size, stride, padding, dilation, x_shape, *bw_data)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {grad_output.device} not supported")
        
        grad_input = Tensor(grad_input_data, device=grad_output.device)
        
        return grad_input
    
    
def max_pool1d(x:Tensor, kernel_size, stride=None, padding=0, dilation=1):
    """
    Max-pooling operation for 1D data.
    
    Reference:
        https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html

    Args:
        - x (Tensor): Input tensor of shape (N, C, W).
        - kernel_size (int): Size of the sliding window.
        - stride (int, optional): Stride size. Defaults to kernel_size.
        - padding (int, optional): Padding size. Defaults to 0.
        - dilation (int, optional): Dilation factor. Defaults to 1.

    Returns:
        Tensor: Output tensor of shape (N, C, L).
    """
    if stride is None: stride = kernel_size
    return MaxPool1d.apply(x, kernel_size, stride, padding, dilation)


class MaxPool2d(Function):
    
    @staticmethod
    def forward(ctx:Context, x:Tensor, kernel_size, stride=None, padding=0, dilation=1):
        if not isinstance(x, Tensor):
            raise TypeError(f"Expected x to be a Tensor but got {type(x)}")
        
        if len(x.shape) != 4:
            raise ValueError(f"Input tensor must be of shape (N, C, H, W), but got {x.shape}")
        
        if x.device == Device.CPU:
            out_data, *bw_data = cpu_ops.max_pool2d_forward(x.data, kernel_size, stride, padding, dilation)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {x.device} not supported")

        out = Tensor(out_data, device=x.device, requires_grad=True)
        
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.bw_data = bw_data

        return out
    
    @staticmethod
    def backward(ctx:Context, grad_output:Tensor):
        kernel_size = ctx.kernel_size
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        bw_data = ctx.bw_data
        
        if not isinstance(grad_output, Tensor):
            raise TypeError(f"Expected grad_output to be a Tensor but got {type(grad_output)}")
        
        if grad_output.device == Device.CPU:
            grad_input_data = cpu_ops.max_pool2d_backward(grad_output.data, kernel_size, stride, padding, dilation, *bw_data)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {grad_output.device} not supported")
        
        grad_input = Tensor(grad_input_data, device=grad_output.device)
        
        return grad_input


def max_pool2d(x:Tensor, kernel_size, stride=None, padding=0, dilation=1):
    """ 
    Max-pooling function.
    
    Reference:
        https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html

    Args:
        - tensor (numpy.ndarray): Input tensor of shape (N, C, H, W).
        - kernel_size (int or tuple): Size of the sliding window.
        - stride (int or tuple, optional): Stride size. Defaults to kernel_size.
        - padding (int or tuple, optional): Padding size. Defaults to 0.
        - dilation (int or tuple, optional): Dilation factor. Defaults to 1.

    Returns:
        numpy.ndarray: Output tensor of shape (N, C, lH, lW).
    """
    if stride is None: stride = kernel_size
    return MaxPool2d.apply(x, kernel_size, stride, padding, dilation)


class AvgPool1d(Function):
    
    @staticmethod
    def forward(ctx:Context, x:Tensor, kernel_size, stride=None, padding=0, dilation=1):
        if not isinstance(x, Tensor):
            raise TypeError(f"Expected x to be a Tensor but got {type(x)}")
        
        if len(x.shape) != 3:
            raise ValueError(f"Input tensor must be of shape (N, C, L), but got {x.shape}")
        
        if x.device == Device.CPU:
            out_data, *bw_data = cpu_ops.avg_pool1d_forward(x.data, kernel_size, stride, padding, dilation)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {x.device} not supported")

        out = Tensor(out_data, device=x.device, requires_grad=True)
        
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.x_shape = x.shape
        ctx.bw_data = bw_data

        return out
    
    @staticmethod
    def backward(ctx:Context, grad_output:Tensor):
        kernel_size = ctx.kernel_size
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        x_shape = ctx.x_shape
        bw_data = ctx.bw_data
        
        if not isinstance(grad_output, Tensor):
            raise TypeError(f"Expected grad_output to be a Tensor but got {type(grad_output)}")
        
        if grad_output.device == Device.CPU:
            grad_input_data = cpu_ops.avg_pool1d_backward(grad_output.data, kernel_size, stride, padding, dilation, x_shape, *bw_data)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {grad_output.device} not supported")
        
        grad_input = Tensor(grad_input_data, device=grad_output.device)
        
        return grad_input
    

def avg_pool1d(x:Tensor, kernel_size, stride=None, padding=0, dilation=1):
    """ 
    Average-pooling function.
    
    Reference:
        https://pytorch.org/docs/stable/generated/torch.nn.AvgPool1d.html

    Args:
        - tensor (numpy.ndarray): Input tensor of shape (N, C, W).
        - kernel_size (int or tuple): Size of the sliding window.
        - stride (int or tuple, optional): Stride size. Defaults to kernel_size.
        - padding (int or tuple, optional): Padding size. Defaults to 0.
        - dilation (int or tuple, optional): Dilation factor. Defaults to 1.

    Returns:
        numpy.ndarray: Output tensor of shape (N, C, L).
    """
    if stride is None: stride = kernel_size
    return AvgPool1d.apply(x, kernel_size, stride, padding, dilation)
    


class AvgPool2d(Function):
    
    @staticmethod
    def forward(ctx:Context, x:Tensor, kernel_size, stride=None, padding=0, dilation=1):
        if not isinstance(x, Tensor):
            raise TypeError(f"Expected x to be a Tensor but got {type(x)}")
        
        if len(x.shape) != 4:
            raise ValueError(f"Input tensor must be of shape (N, C, H, W), but got {x.shape}")
        
        if x.device == Device.CPU:
            out_data, *bw_data = cpu_ops.avg_pool2d_forward(x.data, kernel_size, stride, padding, dilation)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {x.device} not supported")

        out = Tensor(out_data, device=x.device, requires_grad=True)
        
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.bw_data = bw_data

        return out
        
    @staticmethod
    def backward(ctx:Context, grad_output:Tensor):
        kernel_size = ctx.kernel_size
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        bw_data = ctx.bw_data
        
        if not isinstance(grad_output, Tensor):
            raise TypeError(f"Expected grad_output to be a Tensor but got {type(grad_output)}")
        
        if grad_output.device == Device.CPU:
            grad_input_data = cpu_ops.avg_pool2d_backward(grad_output.data, kernel_size, stride, padding, dilation, *bw_data)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {grad_output.device} not supported")
        
        grad_input = Tensor(grad_input_data, device=grad_output.device)
        
        return grad_input
    

def avg_pool2d(x:Tensor, kernel_size, stride=None, padding=0, dilation=1):
    """ 
    Average-pooling function.
    
    Reference:
        https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html

    Args:
        - tensor (numpy.ndarray): Input tensor of shape (N, C, H, W).
        - kernel_size (int or tuple): Size of the sliding window.
        - stride (int or tuple, optional): Stride size. Defaults to kernel_size.
        - padding (int or tuple, optional): Padding size. Defaults to 0.
        - dilation (int or tuple, optional): Dilation factor. Defaults to 1.

    Returns:
        numpy.ndarray: Output tensor of shape (N, C, lH , lW).
    """
    if stride is None: stride = kernel_size
    return AvgPool2d.apply(x, kernel_size, stride, padding, dilation)