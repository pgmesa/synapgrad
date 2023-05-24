from typing import Any
from synapgrad.tensor import Tensor
from synapgrad import cpu_ops
from synapgrad.device import Device

"""
Tensor autograd functions.
"""

class BackwardFunction:
    
    def __init__(self, backward:callable, operation:str) -> None:
        self.backward = backward
        self.operation = operation
    
    def __call__(self) -> Any:
        return self.backward()
    
    def name(self) -> str:
        return self.operation + "Backward"
    
    def __repr__(self) -> str:
        string:str = super().__repr__()
        idx =string.find(" ")
        string = "<" + self.name() + string[idx:]
        return string

# *******************************
# ******* Basic functions *******
# *******************************

def add(x1:Tensor, x2:Tensor):
    """ 
    Add two tensors.

    Args:
        x1 (Tensor): First tensor.
        x2 (Tensor): Second tensor.

    Returns:
        Tensor: The result of the addition.
    """
    if not isinstance(x1, Tensor):
        raise TypeError(f"Expected x1 to be a Tensor but got {type(x1)}")
    if not isinstance(x2, Tensor):
        raise TypeError(f"Expected x2 to be a Tensor but got {type(x2)}")
    
    if x1.device == Device.CPU:
        out_data = cpu_ops.add_forward(x1.data, x2.data)
    else:
        raise RuntimeError(f"{x1.device} not supported")

    inputs = (x1, x2)
    req_grad = any(inp.requires_grad for inp in inputs)
    out = Tensor(out_data, device=x1.device, children=inputs, requires_grad=req_grad, operation="Add")
    
    def backward():
        grad_output = out.grad
        if grad_output.device == Device.CPU:
            a_grad, b_grad = cpu_ops.add_backward(grad_output.data, x1.shape, x2.shape)
        else:
            raise RuntimeError(f"{grad_output.device} not supported")
        
        if x1.requires_grad: x1._grad += a_grad 
        if x2.requires_grad: x2._grad += b_grad
    
    if out.requires_grad: out.grad_fn = BackwardFunction(backward, out._operation)
    
    return out


def mul(x1:Tensor, x2:Tensor):
    """ 
    Multiply two tensors.

    Args:
        x1 (Tensor): First tensor.
        x2 (Tensor): Second tensor.

    Returns:
        Tensor: The result of the multiplication.
    """
    if not isinstance(x1, Tensor):
        raise TypeError(f"Expected x1 to be a Tensor but got {type(x1)}")
    if not isinstance(x2, Tensor):
        raise TypeError(f"Expected x2 to be a Tensor but got {type(x2)}")
    
    if x1.device == Device.CPU:
        out_data = cpu_ops.mul_forward(x1.data, x2.data)
    else:
        raise RuntimeError(f"{x1.device} not supported")
    
    inputs = (x1, x2)
    req_grad = any(inp.requires_grad for inp in inputs)
    out = Tensor(out_data, device=x1.device, children=inputs, requires_grad=req_grad, operation="Mul")
    
    def backward():
        grad_output = out.grad
        if grad_output.device == Device.CPU:
            a_grad, b_grad = cpu_ops.mul_backward(grad_output.data, x1.data, x2.data)
        else:
            raise RuntimeError(f"{grad_output.device} not supported")
        
        if x1.requires_grad: x1._grad += a_grad 
        if x2.requires_grad: x2._grad += b_grad
    
    if out.requires_grad: out.grad_fn = BackwardFunction(backward, out._operation)
    
    return out


def matmul(x1:Tensor, x2:Tensor):
    """ 
    Matrix multiplication between tensors. To multiply 2 different n dimensional tensors, with n > 2,
    dimension -1 of first tensor and -2 of second tensor must agree. Furthermore, dimensions [0 to -3]
    must also agree or one of them be equal to 1. If tensors are not of the same shape, the missing 
    dimensions are filled with 1.
    
    Examples:
        [2,2,4,5] @ [1,5,20] = [2,2,4,20]
        but
        [2,2,4,5] @ [3,1,5,20] = Error (3 != 2 at dimension 0)

    Args:
        x1 (Tensor): First tensor.
        x2 (Tensor): Second tensor.

    Returns:
        Tensor: The result of the matrix multiplication.
    """
    if not isinstance(x1, Tensor):
        raise TypeError(f"Expected x1 to be a Tensor but got {type(x1)}")
    if not isinstance(x2, Tensor):
        raise TypeError(f"Expected x2 to be a Tensor but got {type(x2)}")
    
    if x1.ndim < 2 or x2.ndim < 2:
        raise ValueError(f"At least two dimensions are required for each input")

    if x1.device == Device.CPU:
        out_data = cpu_ops.matmul_forward(x1.data, x2.data)
    else:
        raise RuntimeError(f"{x1.device} not supported")
        
    inputs = (x1, x2)
    req_grad = any(inp.requires_grad for inp in inputs)
    out = Tensor(out_data, device=x1.device, children=inputs, requires_grad=req_grad, operation="Matmul")

    def backward():
        grad_output = out.grad
        if grad_output.device == Device.CPU:
            a_grad, b_grad = cpu_ops.matmul_backward(grad_output.data, x1.data, x2.data)
        else:
            raise RuntimeError(f"{grad_output.device} not supported")
        
        if x1.requires_grad: x1._grad += a_grad 
        if x2.requires_grad: x2._grad += b_grad
    
    if out.requires_grad: out.grad_fn = BackwardFunction(backward, out._operation)
    
    return out


def addmm(x1:Tensor, x2:Tensor, x3:Tensor):
    """ 
    Performs de operation a + x2 @ x3

    Args:
        x1 (Tensor): First tensor.
        x2 (Tensor): Second tensor.
        x3 (Tensor): Third tensor.

    Returns:
        Tensor: The result of the matrix multiplication.
    """
    if not isinstance(x1, Tensor):
        raise TypeError(f"Expected x1 to be a Tensor but got {type(x1)}")
    if not isinstance(x2, Tensor):
        raise TypeError(f"Expected x2 to be a Tensor but got {type(x2)}")
    if not isinstance(x3, Tensor):
        raise TypeError(f"Expected x3 to be a Tensor but got {type(x3)}")
    
    if x1.device == Device.CPU:
        out_data = cpu_ops.addmm_forward(x1.data, x2.data, x3.data)
    else:
        raise RuntimeError(f"{x1.device} not supported")
    
    inputs = (x1, x2, x3)
    req_grad = any(inp.requires_grad for inp in inputs)
    out = Tensor(out_data, device=x1.device, children=inputs, requires_grad=req_grad, operation="Addmm")
    
    def backward():
        grad_output = out.grad
        if grad_output.device == Device.CPU:
            a_grad, b_grad, c_grad = cpu_ops.addmm_backward(grad_output.data, x1.data, x2.data, x3.data)
        else:
            raise RuntimeError(f"{grad_output.device} not supported")
        
        if x1.requires_grad: x1._grad += a_grad 
        if x2.requires_grad: x2._grad += b_grad
        if x3.requires_grad: x3._grad += c_grad
    
    if out.requires_grad: out.grad_fn = BackwardFunction(backward, out._operation)
    
    return out


def pow(x:Tensor, n:'int | float'):
    """ 
    Power of a tensor x ** n

    Args:
        x (Tensor): First tensor.
        n (int | float): Power.

    Returns:
        Tensor: The result of the power.
    """
    if not isinstance(x, Tensor):
        raise TypeError(f"Expected x to be a Tensor but got {type(x)}")

    if not isinstance(n, (int, float)):
        raise ValueError(f"Power of type '{type(n)}' not supported. Only int and float are supported.")
    
    if x.device == Device.CPU:
        out_data = cpu_ops.pow_forward(x.data, n)
    else:
        raise RuntimeError(f"{x.device} not supported")
        
    out = Tensor(out_data, device=x.device, children=(x,), requires_grad=x.requires_grad, operation="Pow")
    
    def backward():
        grad_output = out.grad
        if grad_output.device == Device.CPU:
            a_grad = cpu_ops.pow_backward(grad_output.data, x.data, n)
        else:
            raise RuntimeError(f"{grad_output.device} not supported")
        
        if x.requires_grad: x._grad += a_grad 
    
    if out.requires_grad: out.grad_fn = BackwardFunction(backward, out._operation)
        
    return out


def sqrt(x:Tensor) -> 'Tensor':
    """ 
    Calculate the square root of a tensor (sqrt(x))

    Args:
        x (Tensor): First tensor.

    Returns:
        Tensor: The result of the square root.
    """
    if not isinstance(x, Tensor):
        raise TypeError(f"Expected x to be a Tensor but got {type(x)}")
    
    if x.device == Device.CPU:
        out_data = cpu_ops.sqrt_forward(x.data)
    else:
        raise RuntimeError(f"{x.device} not supported")
    
    out = Tensor(out_data, device=x.device, children=(x,), requires_grad=x.requires_grad, operation="Sqrt")
    
    def backward():
        grad_output = out.grad
        if grad_output.device == Device.CPU:
            a_grad = cpu_ops.sqrt_backward(grad_output.data, out.data)
        else:
            raise RuntimeError(f"{grad_output.device} not supported")
        
        if x.requires_grad: x._grad += a_grad 
    
    if out.requires_grad: out.grad_fn = BackwardFunction(backward, out._operation)
        
    return out

# ***********************************
# ******* Tensor manipulation *******
# ***********************************



# *******************************
# ******* Other functions *******
# *******************************

def reshape(x:Tensor, shape:'tuple') -> 'Tensor':
    """ 
    Reshape a tensor.

    Args:
        x (Tensor): First tensor.
        shape (tuple): New shape.

    Returns:
        Tensor: The result of the reshape.
    """
    if not isinstance(x, Tensor):
        raise TypeError(f"Expected x to be a Tensor but got {type(x)}")
    
    if x.device == Device.CPU:
        out_data = cpu_ops.reshape_forward(x.data, shape)
    else:
        raise RuntimeError(f"{x.device} not supported")
    
    out = Tensor(out_data, device=x.device, children=(x,), requires_grad=x.requires_grad, operation="Reshape")
        
    def backward():
        grad_output = out.grad
        if grad_output.device == Device.CPU:
            a_grad = cpu_ops.reshape_backward(grad_output.data, x.shape)
        else:
            raise RuntimeError(f"{grad_output.device} not supported")
        
        if x.requires_grad: x._grad += a_grad 
    
    if out.requires_grad: out.grad_fn = BackwardFunction(backward, out._operation)
        
    return out


def sum(x:Tensor, dim:'int | tuple'=None, keepdims:bool=False) -> 'Tensor':
    """ 
    Calculate the sum of a tensor.

    Args:
        x (Tensor): First tensor.
        dim (int or tuple, optional): Dimension to sum over. Defaults to None.
        keepdims (bool, optional): Keep dimensions. Defaults to False.

    Returns:
        Tensor: The result of the sum.
    """
    if not isinstance(x, Tensor):
        raise TypeError(f"Expected x to be a Tensor but got {type(x)}")
    
    if x.device == Device.CPU:
        out_data = cpu_ops.sum_forward(x.data, dim, keepdims)
    else:
        raise RuntimeError(f"{x.device} not supported")
    
    out = Tensor(out_data, device=x.device, children=(x,), requires_grad=x.requires_grad, operation="Sum")
    
    def backward():
        grad_output = out.grad
        if grad_output.device == Device.CPU:
            a_grad = cpu_ops.sum_backward(grad_output.data, x.shape, dim, keepdims)
        else:
            raise RuntimeError(f"{grad_output.device} not supported")
                
        if x.requires_grad: x._grad += a_grad
        
    if out.requires_grad: out.grad_fn = BackwardFunction(backward, out._operation)
    
    return out


def mean(x:Tensor, dim:'int | tuple'=None, keepdims:bool=False) -> 'Tensor':
    """ 
    Calculate the mean of a tensor.

    Args:
        x (Tensor): First tensor.
        dim (int or tuple, optional): Dimension to mean over. Defaults to None.
        keepdims (bool, optional): Keep dimensions. Defaults to False.

    Returns:
        Tensor: The result of the mean.
    """
    if not isinstance(x, Tensor):
        raise TypeError(f"Expected x to be a Tensor but got {type(x)}")
    
    if x.device == Device.CPU:
        out_data = cpu_ops.mean_forward(x.data, dim, keepdims)
    else:
        raise RuntimeError(f"{x.device} not supported")
    
    out = Tensor(out_data, device=x.device, children=(x,), requires_grad=x.requires_grad, operation="Mean")
    
    def backward():
        grad_output = out.grad
        if grad_output.device == Device.CPU:
            a_grad = cpu_ops.mean_backward(grad_output.data, x.shape, dim, keepdims)
        else:
            raise RuntimeError(f"{grad_output.device} not supported")
        
        if x.requires_grad: x._grad += a_grad
    
    if out.requires_grad: out.grad_fn = BackwardFunction(backward, out._operation)
    
    return out


def flatten(x:Tensor, start_dim:int=0, end_dim:int=-1) -> 'Tensor':
    """ 
    Flatten a tensor.

    Args:
        x (Tensor): First tensor.
        start_dim (int, optional): Start dimension. Defaults to -1.
        end_dim (int, optional): End dimension. Defaults to -1.

    Returns:
        Tensor: The result of the flatten.
    """
    if not isinstance(x, Tensor):
        raise TypeError(f"Expected x to be a Tensor but got {type(x)}")
    
    shape = x.shape
    start = start_dim if start_dim != -1 else len(shape)
    end = end_dim if end_dim != -1 else len(shape)
    if start > end:
        raise RuntimeError("flatten() has invalid args: start_dim cannot come after end_dim")
    if start < end:
        shape = shape[:start] + (-1,) + shape[end+1:]
    
    if x.device == Device.CPU:
        out_data = cpu_ops.reshape_forward(x.data, shape)
    else:
        raise RuntimeError(f"{x.device} not supported")
    
    out = Tensor(out_data, device=x.device, children=(x,), requires_grad=x.requires_grad, operation="Flatten")
    
    def backward():
        grad_output = out.grad
        if grad_output.device == Device.CPU:
            a_grad = cpu_ops.reshape_backward(grad_output.data, x.shape)
        else:
            raise RuntimeError(f"{grad_output.device} not supported")
        
        if x.requires_grad: x._grad += a_grad
    
    if out.requires_grad: out.grad_fn = BackwardFunction(backward, out._operation)
    
    return out


def squeeze(x:Tensor, dim:'int | tuple'=None) -> 'Tensor':
    """ 
    Remove dimensions with size 1.

    Args:
        x (Tensor): First tensor.
        dim (int or tuple, optional): Dimension to squeeze. Defaults to None.

    Returns:
        Tensor: The result of the squeeze.
    """
    if not isinstance(x, Tensor):
        raise TypeError(f"Expected x to be a Tensor but got {type(x)}")
    
    if x.device == Device.CPU:
        out_data = cpu_ops.squeeze_forward(x.data, dim)
    else:
        raise RuntimeError(f"{x.device} not supported")
    
    out = Tensor(out_data, device=x.device, children=(x,), requires_grad=x.requires_grad, operation="Squeeze")
    
    def backward():
        grad_output = out.grad
        if grad_output.device == Device.CPU:
            a_grad = cpu_ops.squeeze_backward(grad_output.data, x.shape)
        else:
            raise RuntimeError(f"{grad_output.device} not supported")

        x._grad += a_grad
    
    if out.requires_grad: out.grad_fn = BackwardFunction(backward, out._operation)
    
    return out