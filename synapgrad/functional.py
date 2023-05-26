from typing import Any
from synapgrad.tensor import Tensor
from synapgrad import cpu_ops
from synapgrad.device import Device

"""
Tensor autograd functions.
"""

class BackwardFunction:
    
    def __init__(self, backward:callable, operation:str, *args, **kwargs) -> None:
        self.backward = backward
        self.operation = operation
        self.args = args
        self.kwargs = kwargs
    
    def __call__(self) -> Any:
        if not self.args and not self.kwargs:
            return self.backward()
        elif self.args and not self.kwargs:
            return self.backward(*self.args)
        elif not self.args and self.kwargs:
            return self.backward(**self.kwargs)
        else:
            return self.backward(*self.args, **self.kwargs)
    
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
    
    if not x1.device == x2.device:
        raise RuntimeError(f"x1 and x2 must be on the same device")
    
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
    
    if not x1.device == x2.device:
        raise RuntimeError(f"x1 and x2 must be on the same device")
    
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
    
    if not x1.device == x2.device:
        raise RuntimeError(f"x1 and x2 must be on the same device")
    
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
    Performs de operation x1 + x2 @ x3

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
    
    if not (x1.device == x2.device == x3.device):
        raise RuntimeError(f"x1 and x2 and x3 must be on the same device")
    
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


def rpow(x:Tensor, n:'int | float'):
    """ 
    Reciprocal power of a tensor n ** x

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
        out_data = cpu_ops.rpow_forward(x.data, n)
    else:
        raise RuntimeError(f"{x.device} not supported")
        
    out = Tensor(out_data, device=x.device, children=(x,), requires_grad=x.requires_grad, operation="RPow")
    
    def backward():
        grad_output = out.grad
        if grad_output.device == Device.CPU:
            x_grad = cpu_ops.rpow_backward(grad_output.data, out.data, n)
        else:
            raise RuntimeError(f"{grad_output.device} not supported")
        
        if x.requires_grad: x._grad += x_grad 
    
    if out.requires_grad: out.grad_fn = BackwardFunction(backward, out._operation)
    
    return out


def neg(x:Tensor):
    """ 
    Negate a tensor

    Args:
        x (Tensor): First tensor.

    Returns:
        Tensor: The result of the negation.
    """
    if not isinstance(x, Tensor):
        raise TypeError(f"Expected x to be a Tensor but got {type(x)}")
    
    if x.device == Device.CPU:
        out_data = cpu_ops.neg_forward(x.data)
    else:
        raise RuntimeError(f"{x.device} not supported")
        
    out = Tensor(out_data, device=x.device, children=(x,), requires_grad=x.requires_grad, operation="Neg")
    
    def backward():
        grad_output = out.grad
        if grad_output.device == Device.CPU:
            x_grad = cpu_ops.neg_backward(grad_output.data)
        else:
            raise RuntimeError(f"{grad_output.device} not supported")
        
        if x.requires_grad: x._grad += x_grad 
    
    if out.requires_grad: out.grad_fn = BackwardFunction(backward, out._operation)
    
    return out


def slice(x:Tensor, s:slice):
    """ 
    Slice a tensor

    Args:
        x (Tensor): First tensor.
        s (slice): Slice.

    Returns:
        Tensor: The result of the slice.
    """
    if not isinstance(x, Tensor):
        raise TypeError(f"Expected x to be a Tensor but got {type(x)}")
    
    if x.device == Device.CPU:
        out_data = cpu_ops.slice_forward(x.data, s)
    else:
        raise RuntimeError(f"{x.device} not supported")
        
    out = Tensor(out_data, device=x.device, children=(x,), requires_grad=x.requires_grad, operation="Slice")
    
    def backward():
        grad_output = out.grad
        if grad_output.device == Device.CPU:
            x_grad = cpu_ops.slice_backward(grad_output.data, x.shape, s)
        else:
            raise RuntimeError(f"{grad_output.device} not supported")
        
        if x.requires_grad: x._grad += x_grad 
    
    if out.requires_grad: out.grad_fn = BackwardFunction(backward, out._operation)
    
    return out

# ***********************************
# ******* Tensor manipulation *******
# ***********************************

def concat(x:list['Tensor'], dim:int) -> 'Tensor':
    """ 
    Concatenate tensors along a given dimension

    Args:
        x (list['Tensor']): List of tensors.
        dim (int): Dimension.
        
    Examples:
        >>>     x = synapgrad.tensor([1, 2, 3])
        >>>     y = synapgrad.concat([x, x, x], dim=0)
        >>>     print(y)
        >>>     # [1, 2, 3, 1, 2, 3, 1, 2, 3]

    Returns:
        Tensor: The result of the concatenation.
    """
    if not isinstance(x, list) and not isinstance(x, tuple):
        raise TypeError(f"Expected x to be a list or tuple but got {type(x)}")
    
    if not all(isinstance(t, Tensor) for t in x):
        raise TypeError(f"Expected all elements of x to be Tensors but got {type(x)}")
    
    if not all(t.device == x[0].device for t in x):
        raise RuntimeError(f"All tensors must be on the same device")
    
    if not isinstance(dim, int):
        raise TypeError(f"Expected dim to be an int but got {type(dim)}")
    
    if x[0].device == Device.CPU:
        out_data = cpu_ops.concat_forward([t.data for t in x], dim)
    else:
        raise RuntimeError(f"{x[0].device} not supported")
    
    inputs = tuple(x)
    req_grad = any([t.requires_grad for t in x])
    out = Tensor(out_data, device=x[0].device, children=inputs, requires_grad=req_grad, operation="Concat")
    
    sections = []
    for t in x[:-1]:
        s = t.shape[dim]
        if len(sections) > 0: s += sections[-1]
        sections.append(s)
    
    def backward():
        grad_output = out.grad
        if grad_output.device == Device.CPU:
            gradients = cpu_ops.concat_backward(grad_output.data, sections, dim)
        else:
            raise RuntimeError(f"{grad_output.device} not supported")
        
        for inp, grad in zip(inputs, gradients):
            if inp.requires_grad: inp._grad += grad
    
    if out.requires_grad: out.grad_fn = BackwardFunction(backward, out._operation)
    
    return out


def stack(x:list['Tensor'], dim:int=0) -> 'Tensor':
    """ 
    Stack tensors along a given dimension

    Args:
        x (list['Tensor']): List of tensors.
        dim (int): Dimension.
        
    Examples:
        >>>     x = synapgrad.tensor([1, 2, 3])
        >>>     y = synapgrad.stack([x, x, x], dim=0)
        >>>     print(y)
        Tensor(([1, 2, 3], [1, 2, 3], [1, 2, 3]])
        >>>     y = synapgrad.stack([x, x, x], dim=1)
        >>>     print(y)
        Tensor([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        
    Returns:
        Tensor: The result of the concatenation.
    """
    if not isinstance(x, list) and not isinstance(x, tuple):
        raise TypeError(f"Expected x to be a list or tuple but got {type(x)}")
    
    if not all(isinstance(t, Tensor) for t in x):
        raise TypeError(f"Expected all elements of x to be Tensors but got {type(x)}")
    
    if not all(t.device == x[0].device for t in x):
        raise RuntimeError(f"All tensors must be on the same device")
    
    if not isinstance(dim, int):
        raise TypeError(f"Expected dim to be an int but got {type(dim)}")
    
    if x[0].device == Device.CPU:
        out_data = cpu_ops.stack_forward([t.data for t in x], dim)
    else:
        raise RuntimeError(f"{x[0].device} not supported")
    
    inputs = tuple(x)
    req_grad = any([t.requires_grad for t in x])
    out = Tensor(out_data, device=x[0].device, children=inputs, requires_grad=req_grad, operation="Stack")
    
    def backward():
        grad_output = out.grad
        if grad_output.device == Device.CPU:
            slices_grad = cpu_ops.stack_backward(grad_output.data, dim)
        else:
            raise RuntimeError(f"{grad_output.device} not supported")
        
        for inp, grad in zip(inputs, slices_grad):
            if inp.requires_grad: inp._grad += grad
            
    if out.requires_grad: out.grad_fn = BackwardFunction(backward, out._operation)
    
    return out


def unbind(x:Tensor, dim:int=0) -> list['Tensor']:
    """ 
    Unbind a tensor. The inverse operation to `synapgrad.stack`.

    Args:
        x (Tensor): First tensor.
        dim (int): Dimension.
        
    Examples:
        >>>     x = synapgrad.tensor([[1, 2], [3, 4]])
        >>>     y = synapgrad.unbind(x, dim=0)
        >>>     print(y)
        [Tensor([1, 2]), Tensor([3, 4])]
        >>>     y = synapgrad.unbind(x, dim=1)
        >>>     print(y)
        [Tensor([1, 3]), Tensor([2, 4])]
        
    Returns:
        list['Tensor']: The result of the unbind.
    """
    if not isinstance(x, Tensor):
        raise TypeError(f"Expected x to be a Tensor but got {type(x)}")
    
    if x.device == Device.CPU:
        slices_data = cpu_ops.unbind_forward(x.data, dim)
    else:
        raise RuntimeError(f"{x.device} not supported")
    
    out = tuple(Tensor(o, device=x.device, children=(x,), requires_grad=x.requires_grad, operation="Unbind") for o in slices_data)
    
    def backward(out_index):
        grad_output = out[out_index].grad
        if grad_output.device == Device.CPU:
            x_grad = cpu_ops.unbind_backward(grad_output.data, x.shape, dim, out_index)
        else:
            raise RuntimeError(f"{grad_output.device} not supported")
        
        if x.requires_grad: x._grad += x_grad
    
    for i, o in enumerate(out):
        if o.requires_grad: o.grad_fn = BackwardFunction(backward, o._operation, i)     
    
    return out

# *******************************
# ******* Other functions *******
# *******************************

def clone(x:Tensor):
    """ 
    Clone a tensor

    Args:
        x (Tensor): First tensor.

    Returns:
        Tensor: The result of the clone.
    """
    if not isinstance(x, Tensor):
        raise TypeError(f"Expected x to be a Tensor but got {type(x)}")
    
    if x.device == Device.CPU:
        out_data = cpu_ops.clone_forward(x.data)
    else:
        raise RuntimeError(f"{x.device} not supported")
    
    out = Tensor(out_data, device=x.device, children=(x,), requires_grad=x.requires_grad, operation="Clone")
    
    def backward():
        grad_output = out.grad
        if grad_output.device == Device.CPU:
            a_grad = cpu_ops.clone_backward(grad_output.data)
        else:
            raise RuntimeError(f"{grad_output.device} not supported")
        
        if x.requires_grad: x._grad += a_grad 
    
    if out.requires_grad: out.grad_fn = BackwardFunction(backward, out._operation)
    
    return out


def exp(x:Tensor) -> 'Tensor':
    """ 
    Calculate the exponential of a tensor (e^x)

    Args:
        x (Tensor): First tensor.

    Returns:
        Tensor: The result of the exponential.
    """
    if not isinstance(x, Tensor):
        raise TypeError(f"Expected x to be a Tensor but got {type(x)}")
    
    if x.device == Device.CPU:
        out_data = cpu_ops.exp_forward(x.data)
    else:
        raise RuntimeError(f"{x.device} not supported")
    
    out = Tensor(out_data, device=x.device, children=(x,), requires_grad=x.requires_grad, operation="Exp")
    
    def backward():
        grad_output = out.grad
        if grad_output.device == Device.CPU:
            a_grad = cpu_ops.exp_backward(grad_output.data, out.data)
        else:
            raise RuntimeError(f"{grad_output.device} not supported")
        
        if x.requires_grad: x._grad += a_grad 
    
    if out.requires_grad: out.grad_fn = BackwardFunction(backward, out._operation)
    
    return out


def log(x:Tensor) -> 'Tensor':
    """ 
    Calculate the natural logarithm of a tensor (ln(x))

    Args:
        x (Tensor): First tensor.

    Returns:
        Tensor: The result of the natural logarithm.
    """
    if not isinstance(x, Tensor):
        raise TypeError(f"Expected x to be a Tensor but got {type(x)}")
    
    if x.device == Device.CPU:
        out_data = cpu_ops.log_forward(x.data)
    else:
        raise RuntimeError(f"{x.device} not supported")
    
    out = Tensor(out_data, device=x.device, children=(x,), requires_grad=x.requires_grad, operation="Log")
    
    def backward():
        grad_output = out.grad
        if grad_output.device == Device.CPU:
            a_grad = cpu_ops.log_backward(grad_output.data, x.data)
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


def max(x:Tensor, dim:int, keepdims=False) -> 'Tensor':
    """ 
    Calculate the maximum of a tensor.

    Args:
        x (Tensor): First tensor.
        dim (int): Dimension to max over.
        keepdims (bool, optional): Keep dimensions. Defaults to False.

    Returns:
        Tensor: The result of the max.
    """
    if not isinstance(x, Tensor):
        raise TypeError(f"Expected x to be a Tensor but got {type(x)}")
    
    if x.device == Device.CPU:
        out_data = cpu_ops.max_forward(x.data, dim, keepdims)
    else:
        raise RuntimeError(f"{x.device} not supported")
    
    out = Tensor(out_data, device=x.device, children=(x,), requires_grad=x.requires_grad, operation="Max")
    
    def backward():
        grad_output = out.grad
        if grad_output.device == Device.CPU:
            a_grad = cpu_ops.max_backward(grad_output.data, x.data, dim, keepdims)
        else:
            raise RuntimeError(f"{grad_output.device} not supported")
        
        if x.requires_grad: x._grad += a_grad
    
    if out.requires_grad: out.grad_fn = BackwardFunction(backward, out._operation)
    
    return out


def min(x:Tensor, dim:int, keepdims:bool=False) -> 'Tensor':
    """ 
    Calculate the minimum of a tensor.

    Args:
        x (Tensor): First tensor.
        dim (int): Dimension to min over.
        keepdims (bool, optional): Keep dimensions. Defaults to False.

    Returns:
        Tensor: The result of the min.
    """
    if not isinstance(x, Tensor):
        raise TypeError(f"Expected x to be a Tensor but got {type(x)}")
    
    if x.device == Device.CPU:
        out_data = cpu_ops.min_forward(x.data, dim, keepdims)
    else:
        raise RuntimeError(f"{x.device} not supported")
    
    out = Tensor(out_data, device=x.device, children=(x,), requires_grad=x.requires_grad, operation="Min")
    
    def backward():
        grad_output = out.grad
        if grad_output.device == Device.CPU:
            a_grad = cpu_ops.min_backward(grad_output.data, x.data, dim, keepdims)
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

        if x.requires_grad: x._grad += a_grad
    
    if out.requires_grad: out.grad_fn = BackwardFunction(backward, out._operation)
    
    return out


def unsqueeze(x:Tensor, dim:'int | tuple') -> 'Tensor':
    """ 
    Add an extra dimension to a tensor.

    Args:
        x (Tensor): First tensor.
        dim (int or tuple): Dimension to unsqueeze

    Returns:
        Tensor: The result of the unsqueeze.
    """
    if not isinstance(x, Tensor):
        raise TypeError(f"Expected x to be a Tensor but got {type(x)}")
    
    if x.device == Device.CPU:
        out_data = cpu_ops.unsqueeze_forward(x.data, dim)
    else:
        raise RuntimeError(f"{x.device} not supported")
    
    out = Tensor(out_data, device=x.device, children=(x,), requires_grad=x.requires_grad, operation="Unsqueeze")
    
    def backward():
        grad_output = out.grad
        if grad_output.device == Device.CPU:
            a_grad = cpu_ops.unsqueeze_backward(grad_output.data, dim)
        else:
            raise RuntimeError(f"{grad_output.device} not supported")

        if x.requires_grad: x._grad += a_grad
    
    if out.requires_grad: out.grad_fn = BackwardFunction(backward, out._operation)
    
    return out

# ***************************************
# ******* Tensor view functions *********
# ***************************************

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


def movedim(x:Tensor, source:int, destination:int) -> 'Tensor':
    """ 
    Move a dimension.

    Args:
        x (Tensor): First tensor.
        source (int): Source dimension.
        destination (int): Destination dimension.

    Returns:
        Tensor: The result of the movedim.
    """
    if not isinstance(x, Tensor):
        raise TypeError(f"Expected x to be a Tensor but got {type(x)}")
    
    if x.device == Device.CPU:
        out_data = cpu_ops.movedim_forward(x.data, source, destination)
    else:
        raise RuntimeError(f"{x.device} not supported")
    
    out = Tensor(out_data, device=x.device, children=(x,), requires_grad=x.requires_grad, operation="Movedim")
        
    def backward():
        grad_output = out.grad
        if grad_output.device == Device.CPU:
            a_grad = cpu_ops.movedim_backward(grad_output.data, source, destination)
        else:
            raise RuntimeError(f"{grad_output.device} not supported")
        
        if x.requires_grad: x._grad += a_grad 
    
    if out.requires_grad: out.grad_fn = BackwardFunction(backward, out._operation)
        
    return out


def transpose(x:Tensor, dim0:int, dim1:int) -> 'Tensor':
    """ 
    Transpose a tensor.

    Args:
        x (Tensor): First tensor.
        dim0 (int): First dimension.
        dim1 (int): Second dimension.

    Returns:
        Tensor: The result of the transpose.
    """
    if not isinstance(x, Tensor):
        raise TypeError(f"Expected x to be a Tensor but got {type(x)}")
    
    if x.device == Device.CPU:
        out_data = cpu_ops.transpose_forward(x.data, dim0, dim1)
    else:
        raise RuntimeError(f"{x.device} not supported")
    
    out = Tensor(out_data, device=x.device, children=(x,), requires_grad=x.requires_grad, operation="Transpose")
        
    def backward():
        grad_output = out.grad
        if grad_output.device == Device.CPU:
            a_grad = cpu_ops.transpose_backward(grad_output.data, dim0, dim1)
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


def unfold_dim(x:Tensor, dimension:int, size:int, step:int):
    """
    Unfold a tensor along a specified dimension.

    Parameters
    ----------
    x : Tensor
        The tensor to unfold.
    dimension : int
        The dimension to unfold.
    size : int
        The size of the unfolding window.
    step : int
        The step between each unfolding window.

    Returns
    -------
    Tensor
        The unfolding result.

    Raises
    ------
    ValueError
        If the specified dimension is invalid, if the size or step are not positive integers, or if the size is
        larger than the size of the specified dimension.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.arange(10)
    >>> unfold(a, 0, 3, 1)
    array([[[0, 1, 2],
            [3, 4, 5],
            [6, 7, 8]],
        
            [[9, 0, 1],
            [2, 3, 4],
            [5, 6, 7]]])
    """
    if not isinstance(x, Tensor):
        raise TypeError(f"Expected x to be a Tensor but got {type(x)}")
    
    # check that the specified dimension is valid
    if dimension >= x.ndim or dimension < -x.ndim:
        raise ValueError(f"Dimension out of range for tensor with {x.ndim} dimensions: {dimension}")
    if dimension < 0:
        dimension += x.ndim
    # check that the size and step are positive integers
    if not isinstance(size, int) or size <= 0:
        raise ValueError(f"Invalid size: {size}")
    if not isinstance(step, int) or step <= 0:
        raise ValueError(f"Invalid step: {step}")
    
    if x.device == Device.CPU:
        out_data = cpu_ops.unfold_dim_forward(x.data, dimension, size, step)
    else:
        raise RuntimeError(f"{x.device} not supported")
    
    out = Tensor(out_data, device=x.device, children=(x,), requires_grad=x.requires_grad, operation="UnfoldDim")
    
    def backward():
        grad_output = out.grad
        if grad_output.device == Device.CPU:
            a_grad = cpu_ops.unfold_dim_backward(grad_output.data, x.shape, dimension, size, step)
        else:
            raise RuntimeError(f"{grad_output.device} not supported")
        
        if x.requires_grad: x._grad += a_grad
    
    if out.requires_grad: out.grad_fn = BackwardFunction(backward, out._operation)
    
    return out