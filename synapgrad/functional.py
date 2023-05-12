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
        if not isinstance(x1, Tensor):
            raise TypeError(f"Expected x1 to be a Tensor but got {type(x1)}")
        if not isinstance(x2, Tensor):
            raise TypeError(f"Expected x2 to be a Tensor but got {type(x2)}")
        
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
        if not isinstance(x1, Tensor):
            raise TypeError(f"Expected x1 to be a Tensor but got {type(x1)}")
        if not isinstance(x2, Tensor):
            raise TypeError(f"Expected x2 to be a Tensor but got {type(x2)}")
        
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
        if not isinstance(x1, Tensor):
            raise TypeError(f"Expected x1 to be a Tensor but got {type(x1)}")
        if not isinstance(x2, Tensor):
            raise TypeError(f"Expected x2 to be a Tensor but got {type(x2)}")
    
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
        if not isinstance(x, Tensor):
            raise TypeError(f"Expected x to be a Tensor but got {type(x)}")

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
        if not isinstance(x, Tensor):
            raise TypeError(f"Expected x to be a Tensor but got {type(x)}")
        
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
        if not isinstance(x, Tensor):
            raise TypeError(f"Expected x to be a Tensor but got {type(x)}")
        
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
        if not isinstance(x, Tensor):
            raise TypeError(f"Expected x to be a Tensor but got {type(x)}")
        
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
    
    @staticmethod
    def forward(ctx, x:list['Tensor'], dim:int):
        if not isinstance(x, list) and not isinstance(x, tuple):
            raise TypeError(f"Expected x to be a list or tuple but got {type(x)}")
        
        if not all(isinstance(t, Tensor) for t in x):
            raise TypeError(f"Expected all elements of x to be Tensors but got {type(x)}")
        
        if not isinstance(dim, int):
            raise TypeError(f"Expected dim to be an int but got {type(dim)}")
        
        if x[0].device == Device.CPU:
            out_data = cpu_ops.concat_forward([t.data for t in x], dim)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {x[0].device} not supported")
            
        out = Tensor(out_data, device=x[0].device)
        
        ctx.num_inputs = len(x)
        ctx.dim = dim
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output:Tensor):
        num_inputs, dim = ctx.num_inputs, ctx.dim
        
        if grad_output.device == Device.CPU:
            a_grad = cpu_ops.concat_backward(grad_output.data, num_inputs, dim)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {grad_output.device} not supported")
            
        x_grad = [Tensor(g, device=grad_output.device) for g in a_grad]

        return x_grad
    

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
    return Concat.apply(x, dim)
    
    
class Stack(Function):
    
    @staticmethod
    def forward(ctx, x:list['Tensor'], dim:int):
        if not isinstance(x, list) and not isinstance(x, tuple):
            raise TypeError(f"Expected x to be a list or tuple but got {type(x)}")
        
        if not all(isinstance(t, Tensor) for t in x):
            raise TypeError(f"Expected all elements of x to be Tensors but got {type(x)}")
        
        if not isinstance(dim, int):
            raise TypeError(f"Expected dim to be an int but got {type(dim)}")
        
        if x[0].device == Device.CPU:
            out_data = cpu_ops.stack_forward([t.data for t in x], dim)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {x[0].device} not supported")
            
        out = Tensor(out_data, device=x[0].device)
    
        ctx.dim = dim
        
        return out
    
    def backward(ctx, grad_output:Tensor):
        dim = ctx.dim
        
        if grad_output.device == Device.CPU:
            slices_grad = cpu_ops.stack_backward(grad_output.data, dim)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {grad_output.device} not supported")
            
        x_grad = [Tensor(g, device=grad_output.device) for g in slices_grad]

        return x_grad
    

def stack(x:list['Tensor'], dim:int) -> 'Tensor':
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
    return Stack.apply(x, dim)
    
    
class Unbind(Function):
    
    @staticmethod
    def forward(ctx, x:Tensor, dim:int):
        if not isinstance(x, Tensor):
            raise TypeError(f"Expected x to be a Tensor but got {type(x)}")
        
        if x.device == Device.CPU:
            slices_data = cpu_ops.unbind_forward(x.data, dim)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {x.device} not supported")
        
        out = tuple(Tensor(o, device=x.device) for o in slices_data)
        
        ctx.x_shape = x.shape
        ctx.dim = dim
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output:Tensor):
        x_shape, dim = ctx.x_shape, ctx.dim
        out_index = ctx.forward_out_index
        
        if grad_output.device == Device.CPU:
            a_grad = cpu_ops.unbind_backward(grad_output.data, x_shape, dim, out_index)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {grad_output.device} not supported")
            
        x_grad = Tensor(a_grad, device=grad_output.device)

        return x_grad
    

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
    return Unbind.apply(x, dim)

# *************************
# ******* Other ops *******
# *************************

class Clone(Function):
    
    @staticmethod
    def forward(ctx, x:Tensor):
        if not isinstance(x, Tensor):
            raise TypeError(f"Expected x to be a Tensor but got {type(x)}")
        
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