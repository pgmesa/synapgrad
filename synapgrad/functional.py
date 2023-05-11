from synapgrad import cpu_ops
from synapgrad.tensor import Tensor
from synapgrad.autograd import Function
from synapgrad.device import Device

"""
Tensor autograd functions.
"""

# *************************
# ******* Basic ops *******
# *************************

class Add(Function):

    @staticmethod
    def forward(ctx, x1:Tensor, x2:Tensor):
        if x1.device == Device.CPU:
            out_data = cpu_ops.add_forward(x1.data, x2.data)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {x1.device} not supported")

        out = Tensor(out_data, device=x1.device)

        ctx.x1_shape = x1.shape
        ctx.x2_shape = x2.shape
        
        return out
        
    @staticmethod
    def backward(ctx, grad_output:Tensor):
        x1_shape, x2_shape = ctx.x1_shape, ctx.x2_shape
        
        if grad_output.device == Device.CPU:
            a_grad, b_grad = cpu_ops.add_backward(grad_output.data, x1_shape, x2_shape)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {grad_output.device} not supported")
        
        x1_grad = Tensor(a_grad, device=grad_output.device)
        x2_grad = Tensor(b_grad, device=grad_output.device)
        
        return x1_grad, x2_grad


def add(x1:Tensor, x2:Tensor):
    """ 
    Add two tensors.

    Args:
        x1 (Tensor): First tensor.
        x2 (Tensor): Second tensor.

    Returns:
        Tensor: The result of the addition.
    """
    return Add.apply(x1, x2)


class Mul(Function):

    @staticmethod
    def forward(ctx, x1:Tensor, x2:Tensor):
        if x1.device == Device.CPU:
            out_data = cpu_ops.mul_forward(x1.data, x2.data)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {x1.device} not supported")
            
        out = Tensor(out_data, device=x1.device)

        ctx.save_for_backward(x1, x2)
        
        return out

    @staticmethod
    def backward(ctx, grad_output:Tensor):
        x1, x2 = ctx.saved_tensors
        
        if grad_output.device == Device.CPU:
            a_grad, b_grad = cpu_ops.mul_backward(grad_output.data, x1.data, x2.data)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {grad_output.device} not supported")
        
        x1_grad = Tensor(a_grad, device=grad_output.device)
        x2_grad = Tensor(b_grad, device=grad_output.device)

        return x1_grad, x2_grad

    
def mul(x1:Tensor, x2:Tensor):
    """ 
    Multiply two tensors.

    Args:
        x1 (Tensor): First tensor.
        x2 (Tensor): Second tensor.

    Returns:
        Tensor: The result of the multiplication.
    """
    return Mul.apply(x1, x2)


class MatMul(Function):

    @staticmethod
    def forward(ctx, x1:Tensor, x2:Tensor):
        if x1.device == Device.CPU:
            out_data = cpu_ops.matmul_forward(x1.data, x2.data)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {x1.device} not supported")
            
        out = Tensor(out_data, device=x1.device)
        
        ctx.save_for_backward(x1, x2)
        
        return out

    @staticmethod
    def backward(ctx, grad_output:Tensor):
        x1, x2 = ctx.saved_tensors
        
        if grad_output.device == Device.CPU:
            a_grad, b_grad = cpu_ops.matmul_backward(grad_output.data, x1.data, x2.data)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {grad_output.device} not supported")
            
        x1_grad = Tensor(a_grad, device=grad_output.device)
        x2_grad = Tensor(b_grad, device=grad_output.device)

        return x1_grad, x2_grad
    
    
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
    return MatMul.apply(x1, x2)


class Pow(Function):

    @staticmethod
    def forward(ctx, x:Tensor, n:'int | float'):
        if not isinstance(n, (int, float)):
            raise ValueError(f"Power of type '{type(n)}' not supported. Only int and float are supported.")
        
        if x.device == Device.CPU:
            out_data = cpu_ops.pow_forward(x.data, n)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {x.device} not supported")
            
        out = Tensor(out_data, device=x.device)
        
        ctx.save_for_backward(x)
        ctx.n = n
        
        return out

    @staticmethod
    def backward(ctx, grad_output:Tensor):
        x, = ctx.saved_tensors
        n = ctx.n
        
        if grad_output.device == Device.CPU:
            a_grad = cpu_ops.pow_backward(grad_output.data, x.data, n)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {grad_output.device} not supported")
            
        x_grad = Tensor(a_grad, device=grad_output.device)

        return x_grad


def pow(x:Tensor, n:'int | float'):
    """ 
    Power of a tensor x ** n

    Args:
        x (Tensor): First tensor.
        n (int | float): Power.

    Returns:
        Tensor: The result of the power.
    """
    return Pow.apply(x, n)


class RPow(Function):

    @staticmethod
    def forward(ctx, x:Tensor, n:'int | float'):
        if not isinstance(n, (int, float)):
            raise ValueError(f"Power of type '{type(n)}' not supported. Only int and float are supported.")
        
        if x.device == Device.CPU:
            out_data = cpu_ops.rpow_forward(x.data, n)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {x.device} not supported")
            
        out = Tensor(out_data, device=x.device)
        
        ctx.save_for_backward(out)
        ctx.n = n
        
        return out

    @staticmethod
    def backward(ctx, grad_output:Tensor):
        out, = ctx.saved_tensors
        n = ctx.n
        
        if grad_output.device == Device.CPU:
            a_grad = cpu_ops.rpow_backward(grad_output.data, out.data, n)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {grad_output.device} not supported")
            
        x_grad = Tensor(a_grad, device=grad_output.device)

        return x_grad
    
    
def rpow(x:Tensor, n:'int | float'):
    """ 
    Reciprocal power of a tensor n ** x

    Args:
        x (Tensor): First tensor.
        n (int | float): Power.

    Returns:
        Tensor: The result of the power.
    """
    return RPow.apply(x, n)


class Neg(Function):
     
    @staticmethod
    def forward(ctx, x:Tensor):
        if x.device == Device.CPU:
            out_data = cpu_ops.neg_forward(x.data)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {x.device} not supported")
            
        out = Tensor(out_data, device=x.device)
        
        return out

    @staticmethod
    def backward(ctx, grad_output:Tensor):
        if grad_output.device == Device.CPU:
            a_grad = cpu_ops.neg_backward(grad_output.data)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {grad_output.device} not supported")
            
        x_grad = Tensor(a_grad, device=grad_output.device)

        return x_grad
    
    
def neg(x:Tensor):
    """ 
    Negate a tensor

    Args:
        x (Tensor): First tensor.

    Returns:
        Tensor: The result of the negation.
    """
    return Neg.apply(x)


class Slice(Function):
    
    @staticmethod
    def forward(ctx, x:Tensor, s:slice):
        if x.device == Device.CPU:
            out_data = cpu_ops.slice_forward(x.data, s)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {x.device} not supported")
            
        out = Tensor(out_data, device=x.device)
        
        ctx.x_shape = x.shape
        ctx.slice = s
        
        return out

    @staticmethod
    def backward(ctx, grad_output:Tensor):
        x_shape = ctx.x_shape
        s = ctx.slice
        
        if grad_output.device == Device.CPU:
            a_grad = cpu_ops.slice_backward(grad_output.data, x_shape, s)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {grad_output.device} not supported")
            
        x_grad = Tensor(a_grad, device=grad_output.device)

        return x_grad


def slice(x:Tensor, s:slice):
    """ 
    Slice a tensor

    Args:
        x (Tensor): First tensor.
        s (slice): Slice.

    Returns:
        Tensor: The result of the slice.
    """
    return Slice.apply(x, s)

# ***********************************
# ******* Tensor manipulation *******
# ***********************************

class Concat(Function):
    ...
    
    
class Stack(Function):
    ...
    
    
class Unbind(Function):
    ...

# *************************
# ******* Other ops *******
# *************************

class Clone(Function):
    
    @staticmethod
    def forward(ctx, x:Tensor):
        if x.device == Device.CPU:
            out_data = cpu_ops.clone_forward(x.data)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {x.device} not supported")
        
        out = Tensor(out_data, device=x.device)
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output:Tensor):
        if grad_output.device == Device.CPU:
            a_grad = cpu_ops.clone_backward(grad_output.data)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {grad_output.device} not supported")
        
        x_grad = Tensor(a_grad, device=grad_output.device)
        
        return x_grad
   
    
def clone(x:Tensor):
    """ 
    Clone a tensor

    Args:
        x (Tensor): First tensor.

    Returns:
        Tensor: The result of the clone.
    """
    return Clone.apply(x)