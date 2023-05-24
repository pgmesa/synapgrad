import numpy as np

from synapgrad.tensor import Tensor
from synapgrad import cpu_ops
from synapgrad.device import Device
from synapgrad.functional import BackwardFunction


# ************************************
# ******* Activation functions *******
# ************************************

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
    if not isinstance(x, Tensor):
        raise TypeError(f"Expected x to be a Tensor but got {type(x)}")
    
    if x.device == Device.CPU:
        out_data = cpu_ops.relu_forward(x.data)
    else:
        raise RuntimeError(f"{x.device} not supported")

    out = Tensor(out_data, device=x.device, children=(x,), requires_grad=x.requires_grad, operation="Relu")
    
    def backward():
        grad_output = out.grad
        if grad_output.device == Device.CPU:
            a_grad = cpu_ops.relu_backward(grad_output.data, x.data)
        else:
            raise RuntimeError(f"{grad_output.device} not supported")
        
        if x.requires_grad: x._grad += a_grad 
    
    if out.requires_grad: out.grad_fn = BackwardFunction(backward, out._operation)
        
    return out


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
    if not isinstance(x, Tensor):
        raise TypeError(f"Expected x to be a Tensor but got {type(x)}")
    
    if x.device == Device.CPU:
        out_data = cpu_ops.log_softmax_forward(x.data, dim)
    else:
        raise RuntimeError(f"{x.device} not supported")

    out = Tensor(out_data, device=x.device, children=(x,), requires_grad=x.requires_grad, operation="LogSoftmax")

    def backward():
        grad_output = out.grad
        if grad_output.device == Device.CPU:
            a_grad = cpu_ops.log_softmax_backward(grad_output.data, out.data, dim)
        else:
            raise RuntimeError(f"{grad_output.device} not supported")
        
        if x.requires_grad: x._grad += a_grad 
    
    if out.requires_grad: out.grad_fn = BackwardFunction(backward, out._operation)

    return out

# ******************************
# ******* Loss functions *******
# ******************************

def nll_loss(y_pred:Tensor, y_true:Tensor):
    """ 
    Negative Log Likelihood loss function.

    Args:
        - y_pred (Tensor): tensor
        - y_true (Tensor): tensor

    Returns:
        Tensor: result
    """
    if not isinstance(y_pred, Tensor):
        raise TypeError(f"Expected y_pred to be a Tensor but got {type(y_pred)}")
    if not isinstance(y_true, Tensor):
        raise TypeError(f"Expected y_true to be a Tensor but got {type(y_true)}")
    
    if y_pred.device == Device.CPU:
        loss_data = cpu_ops.nll_loss_forward(y_pred.data, y_true.data)
    else:
        raise RuntimeError(f"{y_pred.device} not supported")

    loss = Tensor(loss_data, device=y_pred.device, children=(y_pred, y_true), requires_grad=True, operation="NLLLoss")
    
    def backward():
        grad_output = loss.grad
        if grad_output.device == Device.CPU:
            loss_grad_data = cpu_ops.nll_loss_backward(grad_output.data, y_pred.data, y_true.data)
        else:
            raise RuntimeError(f"{grad_output.device} not supported")
        
        if y_pred.requires_grad: y_pred._grad += loss_grad_data
    
    if loss.requires_grad: loss.grad_fn = BackwardFunction(backward, loss._operation)
    
    return loss

def cross_entropy(y_pred:Tensor, y_true:Tensor):
    """ 
    Cross Entropy loss function.

    Args:
        - y_pred (Tensor): tensor
        - y_true (Tensor): tensor

    Returns:
        Tensor: result
    """
    if not isinstance(y_pred, Tensor):
        raise TypeError(f"Expected y_pred to be a Tensor but got {type(y_pred)}")
    if not isinstance(y_true, Tensor):
        raise TypeError(f"Expected y_true to be a Tensor but got {type(y_true)}")
    
    if y_pred.device == Device.CPU:
        loss_data = cpu_ops.cross_entropy_loss_forward(y_pred.data, y_true.data)
    else:
        raise RuntimeError(f"{y_pred.device} not supported")

    loss = Tensor(loss_data, device=y_pred.device, children=(y_pred, y_true), requires_grad=True, operation="CrossEntropyLoss")
    
    def backward():
        grad_output = loss.grad
        if grad_output.device == Device.CPU:
            loss_grad_data = cpu_ops.cross_entropy_loss_backward(grad_output.data, y_pred.data, y_true.data)
        else:
            raise RuntimeError(f"{grad_output.device} not supported")
        
        if y_pred.requires_grad: y_pred._grad += loss_grad_data
    
    if loss.requires_grad: loss.grad_fn = BackwardFunction(backward, loss._operation)
    
    return loss

# *********************************
# ******* Linear functions ********
# *********************************

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
    if not isinstance(x, Tensor):
        raise TypeError(f"Expected x to be a Tensor but got {type(x)}")
    if not isinstance(weight, Tensor):
        raise TypeError(f"Expected weight to be a Tensor but got {type(weight)}")
    if bias is not None and not isinstance(bias, Tensor):
        raise TypeError(f"Expected not None bias to be a Tensor but got {type(bias)}")
    
    if x.device == Device.CPU:
        if bias:
            out_data = cpu_ops.addmm_forward(bias.data, x.data, weight.data.T)
        else:
            out_data = cpu_ops.matmul_forward(x.data, weight.data.T)
    else:
        raise RuntimeError(f"{x.device} not supported")
    
    if bias: inputs = (x, weight, bias)
    else: inputs = (x, weight)
        
    out = Tensor(out_data, device=x.device, children=inputs, requires_grad=True, operation="Linear")
        
    def backward():
        grad_output = out.grad
        if out.device == Device.CPU:
            if bias:
                bias_grad, x_grad, weight_grad = cpu_ops.addmm_backward(grad_output.data, bias.data, x.data, weight.data.T)
            else:
                x_grad, weight_grad = cpu_ops.matmul_backward(grad_output.data, x.data, weight.data.T)
        else:
            raise RuntimeError(f"{out.device} not supported")
        
        if x.requires_grad: 
            x._grad += x_grad
        if weight.requires_grad: 
            weight._grad += weight_grad.T
        if bias and bias.requires_grad: 
            bias._grad += bias_grad
            
    if out.requires_grad: out.grad_fn = BackwardFunction(backward, out._operation)
    
    return out

# *******************************
# ******* Conv functions ********
# *******************************

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
    
    if not isinstance(x, Tensor):
        raise TypeError(f"Expected x to be a Tensor but got {type(x)}")
    
    if len(x.shape) != 4:
        raise ValueError(f"Input tensor must be of shape (N, C, H, W), but got {x.shape}")
    
    if x.device == Device.CPU:
        out_data, *bw_data = cpu_ops.max_pool2d_forward(x.data, kernel_size, stride, padding, dilation)
    else:
        raise RuntimeError(f"{x.device} not supported")

    out = Tensor(out_data, device=x.device, children=(x,), requires_grad=True, operation="MaxPool2d")
    
    def backward():
        grad_output = out.grad
        if grad_output.device == Device.CPU:
            x_grad = cpu_ops.max_pool2d_backward(grad_output.data, kernel_size, stride, padding, dilation, *bw_data)
        else:
            raise RuntimeError(f"{grad_output.device} not supported")
        
        if x.requires_grad: x._grad += x_grad
            
    if out.requires_grad: out.grad_fn = BackwardFunction(backward, out._operation)
    
    return out


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
        raise RuntimeError(f"{x.device} not supported")

    if bias: inputs = (x, weight, bias)
    else: inputs = (x, weight)
    out = Tensor(out_data, device=x.device, children=inputs, requires_grad=True, operation="Conv2d")
    
    def backward():
        grad_output = out.grad
        if grad_output.device == Device.CPU:
            bias_data = bias.data if bias is not None else None
            x_grad, weight_grad, bias_grad = cpu_ops.conv2d_backward(grad_output.data, x.shape, weight.data,
                                                        bias_data, stride, padding, dilation, *bw_data)
        else:
            raise RuntimeError(f"{grad_output.device} not supported")
        
        if x.requires_grad:
            x._grad += x_grad
        if weight.requires_grad:
            weight._grad += weight_grad    
        if bias and bias.requires_grad:
            bias._grad += bias_grad
            
    if out.requires_grad: out.grad_fn = BackwardFunction(backward, out._operation)
    
    return out