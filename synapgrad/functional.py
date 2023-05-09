from .autograd import Function, Tensor
from . import ops_cpu
from .device import Device


# *************************
# ******* Basic ops *******
# *************************

class Add(Function):

    @staticmethod
    def forward(ctx, x1:Tensor, x2:Tensor):
        if x1.device != x2.device:
            raise RuntimeError("Add: x and y must be on the same device")

        requires_grad = x1.requires_grad or x2.requires_grad

        if x1.device == Device.CPU:
            out_data = ops_cpu.add_forward(x1.data, x2.data)
        else:
            raise RuntimeError(f"Add: {x1.device} not supported")

        out = Tensor(out_data, device=x1.device, requires_grad=requires_grad, _children=(x1,x2), _operation="<Add>")

        ctx.x1_shape = x1.shape
        ctx.x2_shape = x2.shape
        
        return out
        
    @staticmethod
    def backward(ctx, grad_output:Tensor):
        x1_shape, x2_shape = ctx.x1_shape, ctx.x2_shape
        
        if grad_output.device == Device.CPU:
            a_grad, b_grad = ops_cpu.add_backward(grad_output.data, x1_shape, x2_shape)
        else:
            raise RuntimeError(f"Add: {grad_output.device} not supported")
        
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
        if x1.device != x2.device:
            raise RuntimeError("Mul: x and y must be on the same device")
        
        requires_grad = x1.requires_grad or x2.requires_grad
        
        if x1.device == Device.CPU:
            out_data = ops_cpu.mul_forward(x1.data, x2.data)
        else:
            raise RuntimeError(f"Mul: {x1.device} not supported")
            
        out = Tensor(out_data, device=x1.device, requires_grad=requires_grad, _children=(x1,x2), _operation="<Mul>")

        ctx.save_for_backward(x1, x2)
        
        return out

    @staticmethod
    def backward(ctx, grad_output:Tensor):
        x1, x2 = ctx.saved_tensors
        
        if grad_output.device == Device.CPU:
            a_grad, b_grad = ops_cpu.mul_backward(grad_output.data, x1.data, x2.data)
        else:
            raise RuntimeError(f"Mul: {grad_output.device} not supported")
        
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
        if x1.device != x2.device:
            raise RuntimeError("Matmul: x and y must be on the same device")
        
        requires_grad = x1.requires_grad or x2.requires_grad
        
        if x1.device == Device.CPU:
            out_data = ops_cpu.matmul_forward(x1.data, x2.data)
        else:
            raise RuntimeError(f"MatMul: {x1.device} not supported")
            
        out = Tensor(out_data, device=x1.device, requires_grad=requires_grad, _children=(x1,x2), _operation="<Matmul>")
        
        ctx.save_for_backward(x1, x2)
        
        return out

    @staticmethod
    def backward(ctx, grad_output:Tensor):
        x1, x2 = ctx.saved_tensors
        
        if grad_output.device == Device.CPU:
            a_grad, b_grad = ops_cpu.matmul_backward(grad_output.data, x1.data, x2.data)
        else:
            raise RuntimeError(f"MatMul: {grad_output.device} not supported")
            
        x1_grad = Tensor(a_grad, device=grad_output.device)
        x2_grad = Tensor(b_grad, device=grad_output.device)

        return x1_grad, x2_grad
    
    
def matmul(x1:Tensor, x2:Tensor):
    """ 
    Matrix multiplication between tensors

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
            out_data = ops_cpu.pow_forward(x.data, n)
        else:
            raise RuntimeError(f"Pow: {x.device} not supported")
            
        out = Tensor(out_data, device=x.device, requires_grad=x.requires_grad, _children=(x,), _operation="<Pow>")
        
        ctx.save_for_backward(x)
        ctx.n = n
        
        return out

    @staticmethod
    def backward(ctx, grad_output:Tensor):
        x, = ctx.saved_tensors
        n = ctx.n
        
        if grad_output.device == Device.CPU:
            a_grad = ops_cpu.pow_backward(grad_output.data, x.data, n)
        else:
            raise RuntimeError(f"Pow: {grad_output.device} not supported")
            
        x_grad = Tensor(a_grad, device=grad_output.device)

        return x_grad, None


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
            out_data = ops_cpu.rpow_forward(x.data, n)
        else:
            raise RuntimeError(f"RPow: {x.device} not supported")
            
        out = Tensor(out_data, device=x.device, requires_grad=x.requires_grad, _children=(x,), _operation="<RPow>")
        
        ctx.save_for_backward(out)
        ctx.n = n
        
        return out

    @staticmethod
    def backward(ctx, grad_output:Tensor):
        out, = ctx.saved_tensors
        n = ctx.n
        
        if grad_output.device == Device.CPU:
            a_grad = ops_cpu.rpow_backward(grad_output.data, out.data, n)
        else:
            raise RuntimeError(f"RPow: {grad_output.device} not supported")
            
        x_grad = Tensor(a_grad, device=grad_output.device)

        return x_grad, None
    
    
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
            out_data = ops_cpu.neg_forward(x.data)
        else:
            raise RuntimeError(f"Neg: {x.device} not supported")
            
        out = Tensor(out_data, device=x.device, requires_grad=x.requires_grad, _children=(x,), _operation="<RPow>")
        
        return out

    @staticmethod
    def backward(ctx, grad_output:Tensor):
        if grad_output.device == Device.CPU:
            a_grad = ops_cpu.neg_backward(grad_output.data)
        else:
            raise RuntimeError(f"Neg: {grad_output.device} not supported")
            
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
            out_data = ops_cpu.slice_forward(x.data, s)
        else:
            raise RuntimeError(f"Slice: {x.device} not supported")
            
        out = Tensor(out_data, device=x.device, requires_grad=x.requires_grad, _children=(x,), _operation="<RPow>")
        
        ctx.x_shape = x.shape
        ctx.slice = s
        
        return out

    @staticmethod
    def backward(ctx, grad_output:Tensor):
        x_shape = ctx.x_shape
        s = ctx.slice
        
        if grad_output.device == Device.CPU:
            a_grad = ops_cpu.slice_backward(grad_output.data, x_shape, s)
        else:
            raise RuntimeError(f"Neg: {grad_output.device} not supported")
            
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


# *************************
# ******* Other ops *******
# *************************