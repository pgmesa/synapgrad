from synapgrad import cpu_ops
from synapgrad.tensor import Tensor
from synapgrad.autograd import Function, Context
from synapgrad.device import Device

"""
Tensor autograd functions.
"""

# *******************************
# ******* Basic functions *******
# *******************************

class Add(Function):

    @staticmethod
    def forward(ctx:Context, x1:Tensor, x2:Tensor):
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
    def backward(ctx:Context, grad_output:Tensor):
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
    def forward(ctx:Context, x1:Tensor, x2:Tensor):
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
    def backward(ctx:Context, grad_output:Tensor):
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
    def forward(ctx:Context, x1:Tensor, x2:Tensor):
        if not isinstance(x1, Tensor):
            raise TypeError(f"Expected x1 to be a Tensor but got {type(x1)}")
        if not isinstance(x2, Tensor):
            raise TypeError(f"Expected x2 to be a Tensor but got {type(x2)}")
        
        if x1.ndim < 2 or x2.ndim < 2:
            raise ValueError(f"{ctx.fn_name}: At least two dimensions are required for each input")
    
        if x1.device == Device.CPU:
            out_data = cpu_ops.matmul_forward(x1.data, x2.data)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {x1.device} not supported")
            
        out = Tensor(out_data, device=x1.device)
        
        ctx.save_for_backward(x1, x2)
        
        return out

    @staticmethod
    def backward(ctx:Context, grad_output:Tensor):
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
    def forward(ctx:Context, x:Tensor, n:'int | float'):
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
    def backward(ctx:Context, grad_output:Tensor):
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
    def forward(ctx:Context, x:Tensor, n:'int | float'):
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
    def backward(ctx:Context, grad_output:Tensor):
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
    def forward(ctx:Context, x:Tensor):
        if not isinstance(x, Tensor):
            raise TypeError(f"Expected x to be a Tensor but got {type(x)}")
        
        if x.device == Device.CPU:
            out_data = cpu_ops.neg_forward(x.data)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {x.device} not supported")
            
        out = Tensor(out_data, device=x.device)
        
        return out

    @staticmethod
    def backward(ctx:Context, grad_output:Tensor):
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
    def forward(ctx:Context, x:Tensor, s:slice):
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
    def backward(ctx:Context, grad_output:Tensor):
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
    def forward(ctx:Context, x:list['Tensor'], dim:int):
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
        
        ctx.sections = [t.shape[dim] for t in x]
        ctx.dim = dim
        
        return out
    
    @staticmethod
    def backward(ctx:Context, grad_output:Tensor):
        sections, dim = ctx.sections, ctx.dim
        
        if grad_output.device == Device.CPU:
            a_grad = cpu_ops.concat_backward(grad_output.data, sections, dim)
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
    def forward(ctx:Context, x:list['Tensor'], dim:int=0):
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
    
    @staticmethod
    def backward(ctx:Context, grad_output:Tensor):
        dim = ctx.dim
        
        if grad_output.device == Device.CPU:
            slices_grad = cpu_ops.stack_backward(grad_output.data, dim)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {grad_output.device} not supported")
            
        x_grad = [Tensor(g, device=grad_output.device) for g in slices_grad]

        return x_grad
    

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
    return Stack.apply(x, dim)
    
    
class Unbind(Function):
    
    @staticmethod
    def forward(ctx:Context, x:Tensor, dim:int):
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
    def backward(ctx:Context, grad_output:Tensor):
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

# *******************************
# ******* Other functions *******
# *******************************

class Clone(Function):
    
    @staticmethod
    def forward(ctx:Context, x:Tensor):
        if not isinstance(x, Tensor):
            raise TypeError(f"Expected x to be a Tensor but got {type(x)}")
        
        if x.device == Device.CPU:
            out_data = cpu_ops.clone_forward(x.data)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {x.device} not supported")
        
        out = Tensor(out_data, device=x.device)
        
        return out
    
    @staticmethod
    def backward(ctx:Context, grad_output:Tensor):
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


class Exp(Function):
    
    @staticmethod
    def forward(ctx:Context, x:Tensor):
        if not isinstance(x, Tensor):
            raise TypeError(f"Expected x to be a Tensor but got {type(x)}")
        
        if x.device == Device.CPU:
            out_data = cpu_ops.exp_forward(x.data)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {x.device} not supported")
        
        out = Tensor(out_data, device=x.device)
        
        ctx.save_for_backward(out)
        
        return out
    
    @staticmethod
    def backward(ctx:Context, grad_output:Tensor):
        out, = ctx.saved_tensors
        
        if grad_output.device == Device.CPU:
            a_grad = cpu_ops.exp_backward(grad_output.data, out.data)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {grad_output.device} not supported")
        
        x_grad = Tensor(a_grad, device=grad_output.device)
        
        return x_grad
    

def exp(x:Tensor) -> 'Tensor':
    """ 
    Calculate the exponential of a tensor (e^x)

    Args:
        x (Tensor): First tensor.

    Returns:
        Tensor: The result of the exponential.
    """
    return Exp.apply(x)
    
    
class Log(Function):
    
    @staticmethod
    def forward(ctx:Context, x:Tensor):
        if not isinstance(x, Tensor):
            raise TypeError(f"Expected x to be a Tensor but got {type(x)}")
        
        if x.device == Device.CPU:
            out_data = cpu_ops.log_forward(x.data)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {x.device} not supported")
        
        out = Tensor(out_data, device=x.device)
        
        ctx.save_for_backward(x)
        
        return out
    
    @staticmethod
    def backward(ctx:Context, grad_output:Tensor):
        x, = ctx.saved_tensors
        
        if grad_output.device == Device.CPU:
            a_grad = cpu_ops.log_backward(grad_output.data, x.data)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {grad_output.device} not supported")
        
        x_grad = Tensor(a_grad, device=grad_output.device)
        
        return x_grad
    
    
def log(x:Tensor) -> 'Tensor':
    """ 
    Calculate the natural logarithm of a tensor (ln(x))

    Args:
        x (Tensor): First tensor.

    Returns:
        Tensor: The result of the natural logarithm.
    """
    return Log.apply(x)


class Sqrt(Function):
    
    @staticmethod
    def forward(ctx:Context, x:Tensor):
        if not isinstance(x, Tensor):
            raise TypeError(f"Expected x to be a Tensor but got {type(x)}")
        
        if x.device == Device.CPU:
            out_data = cpu_ops.sqrt_forward(x.data)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {x.device} not supported")
        
        out = Tensor(out_data, device=x.device)
        
        ctx.save_for_backward(out)
        
        return out
    
    @staticmethod
    def backward(ctx:Context, grad_output:Tensor):
        out, = ctx.saved_tensors
        
        if grad_output.device == Device.CPU:
            a_grad = cpu_ops.sqrt_backward(grad_output.data, out.data)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {grad_output.device} not supported")
        
        x_grad = Tensor(a_grad, device=grad_output.device)
        
        return x_grad
    

def sqrt(x:Tensor) -> 'Tensor':
    """ 
    Calculate the square root of a tensor (sqrt(x))

    Args:
        x (Tensor): First tensor.

    Returns:
        Tensor: The result of the square root.
    """
    return Sqrt.apply(x)


class Sum(Function):
    
    @staticmethod
    def forward(ctx:Context, x:Tensor, dim:'int | tuple'=None, keepdims:bool=False):
        if not isinstance(x, Tensor):
            raise TypeError(f"Expected x to be a Tensor but got {type(x)}")
        
        if x.device == Device.CPU:
            out_data = cpu_ops.sum_forward(x.data, dim, keepdims)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {x.device} not supported")
        
        out = Tensor(out_data, device=x.device)
        
        ctx.dim = dim
        ctx.keepdims = keepdims
        ctx.x_shape = x.shape
        
        return out
    
    @staticmethod
    def backward(ctx:Context, grad_output:Tensor):
        dim = ctx.dim
        keepdims = ctx.keepdims
        x_shape = ctx.x_shape
        
        if grad_output.device == Device.CPU:
            a_grad = cpu_ops.sum_backward(grad_output.data, x_shape, dim, keepdims)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {grad_output.device} not supported")
        
        x_grad = Tensor(a_grad, device=grad_output.device)
        
        return x_grad


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
    return Sum.apply(x, dim, keepdims)


class Mean(Function):
    
    @staticmethod
    def forward(ctx:Context, x:Tensor, dim:'int | tuple'=None, keepdims:bool=False):
        if not isinstance(x, Tensor):
            raise TypeError(f"Expected x to be a Tensor but got {type(x)}")
        
        if x.device == Device.CPU:
            out_data = cpu_ops.mean_forward(x.data, dim, keepdims)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {x.device} not supported")
        
        out = Tensor(out_data, device=x.device)
        
        ctx.dim = dim
        ctx.keepdims = keepdims
        ctx.x_shape = x.shape
        
        return out
    
    @staticmethod
    def backward(ctx:Context, grad_output:Tensor):
        dim = ctx.dim
        keepdims = ctx.keepdims
        x_shape = ctx.x_shape
    
        if grad_output.device == Device.CPU:
            a_grad = cpu_ops.mean_backward(grad_output.data, x_shape, dim, keepdims)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {grad_output.device} not supported")
        
        x_grad = Tensor(a_grad, device=grad_output.device)
        
        return x_grad
    
    
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
    return Mean.apply(x, dim, keepdims)


class Max(Function):
    
    @staticmethod
    def forward(ctx:Context, x:Tensor, dim:int, keepdims=False):
        if not isinstance(x, Tensor):
            raise TypeError(f"Expected x to be a Tensor but got {type(x)}")
        
        if x.device == Device.CPU:
            out_data = cpu_ops.max_forward(x.data, dim, keepdims)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {x.device} not supported")
        
        out = Tensor(out_data, device=x.device)
        
        ctx.save_for_backward(x)
        ctx.dim = dim
        ctx.keepdims = keepdims
        
        return out
    
    def backward(ctx:Context, grad_output:Tensor):
        x, = ctx.saved_tensors
        dim = ctx.dim
        keepdims = ctx.keepdims
        
        if grad_output.device == Device.CPU:
            a_grad = cpu_ops.max_backward(grad_output.data, x.data, dim, keepdims)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {grad_output.device} not supported")

        x_grad = Tensor(a_grad, device=grad_output.device)
        
        return x_grad


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
    return Max.apply(x, dim, keepdims)


class Min(Function):
    
    @staticmethod
    def forward(ctx:Context, x:Tensor, dim:int, keepdims:bool=False):
        if not isinstance(x, Tensor):
            raise TypeError(f"Expected x to be a Tensor but got {type(x)}")
        
        if x.device == Device.CPU:
            out_data = cpu_ops.min_forward(x.data, dim, keepdims)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {x.device} not supported")
        
        out = Tensor(out_data, device=x.device)
        
        ctx.x = ctx.save_for_backward(x)
        ctx.dim = dim
        ctx.keepdims = keepdims
        
        return out
    
    def backward(ctx:Context, grad_output:Tensor):
        x, = ctx.saved_tensors
        dim = ctx.dim
        keepdims = ctx.keepdims
        
        if grad_output.device == Device.CPU:
            a_grad = cpu_ops.min_backward(grad_output.data, x.data, dim, keepdims)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {grad_output.device} not supported")

        x_grad = Tensor(a_grad, device=grad_output.device)
        
        return x_grad
    
    
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
    return Min.apply(x, dim, keepdims)


class Squeeze(Function):
    
    @staticmethod
    def forward(ctx:Context, x:Tensor, dim:'int | tuple'):
        if not isinstance(x, Tensor):
            raise TypeError(f"Expected x to be a Tensor but got {type(x)}")
        
        if x.device == Device.CPU:
            out_data = cpu_ops.squeeze_forward(x.data, dim)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {x.device} not supported")
        
        out = Tensor(out_data, device=x.device)
        
        ctx.x_shape = x.shape
        
        return out
    
    def backward(ctx:Context, grad_output:Tensor):
        x_shape = ctx.x_shape
        
        if grad_output.device == Device.CPU:
            a_grad = cpu_ops.squeeze_backward(grad_output.data, x_shape)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {grad_output.device} not supported")

        x_grad = Tensor(a_grad, device=grad_output.device)
        
        return x_grad
    
    
def squeeze(x:Tensor, dim:'int | tuple'=None) -> 'Tensor':
    """ 
    Remove dimensions with size 1.

    Args:
        x (Tensor): First tensor.
        dim (int or tuple, optional): Dimension to squeeze. Defaults to None.

    Returns:
        Tensor: The result of the squeeze.
    """
    return Squeeze.apply(x, dim)


class Unsqueeze(Function):
    
    @staticmethod
    def forward(ctx:Context, x:Tensor, dim:'int | tuple'):
        if not isinstance(x, Tensor):
            raise TypeError(f"Expected x to be a Tensor but got {type(x)}")
        
        if x.device == Device.CPU:
            out_data = cpu_ops.unsqueeze_forward(x.data, dim)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {x.device} not supported")
        
        out = Tensor(out_data, device=x.device)
        
        ctx.dim = dim
        
        return out
    
    def backward(ctx:Context, grad_output:Tensor):
        dim = ctx.dim
        
        if grad_output.device == Device.CPU:
            a_grad = cpu_ops.unsqueeze_backward(grad_output.data, dim)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {grad_output.device} not supported")
        
        x_grad = Tensor(a_grad, device=grad_output.device)
        
        return x_grad


def unsqueeze(x:Tensor, dim:'int | tuple') -> 'Tensor':
    """ 
    Add an extra dimension to a tensor.

    Args:
        x (Tensor): First tensor.
        dim (int or tuple): Dimension to unsqueeze

    Returns:
        Tensor: The result of the unsqueeze.
    """
    return Unsqueeze.apply(x, dim)


class Reshape(Function):
    
    @staticmethod
    def forward(ctx:Context, x:Tensor, shape:'tuple'):
        if not isinstance(x, Tensor):
            raise TypeError(f"Expected x to be a Tensor but got {type(x)}")
        
        if x.device == Device.CPU:
            out_data = cpu_ops.reshape_forward(x.data, shape)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {x.device} not supported")
        
        out = Tensor(out_data, device=x.device)
        
        ctx.x_shape = x.shape
        
        return out
    
    @staticmethod
    def backward(ctx:Context, grad_output:Tensor):
        x_shape = ctx.x_shape
        
        if grad_output.device == Device.CPU:
            a_grad = cpu_ops.reshape_backward(grad_output.data, x_shape)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {grad_output.device} not supported")
        
        x_grad = Tensor(a_grad, device=grad_output.device)
        
        return x_grad


def reshape(x:Tensor, shape:'tuple') -> 'Tensor':
    """ 
    Reshape a tensor.

    Args:
        x (Tensor): First tensor.
        shape (tuple): New shape.

    Returns:
        Tensor: The result of the reshape.
    """
    return Reshape.apply(x, shape)


class Movedim(Function):
    
    @staticmethod
    def forward(ctx:Context, x:Tensor, source:int, destination:int):
        if not isinstance(x, Tensor):
            raise TypeError(f"Expected x to be a Tensor but got {type(x)}")
        
        if x.device == Device.CPU:
            out_data = cpu_ops.movedim_forward(x.data, source, destination)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {x.device} not supported")
        
        out = Tensor(out_data, device=x.device)
        
        ctx.source = source
        ctx.destination = destination
        
        return out
    
    @staticmethod
    def backward(ctx:Context, grad_output:Tensor):
        source = ctx.source; destination = ctx.destination
        
        if grad_output.device == Device.CPU:
            a_grad = cpu_ops.movedim_backward(grad_output.data, source, destination)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {grad_output.device} not supported")
        
        out = Tensor(a_grad, device=grad_output.device)
        
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
    return Movedim.apply(x, source, destination)


class Transpose(Function):
    
    @staticmethod
    def forward(ctx:Context, x:Tensor, dim0:int, dim1:int):
        if not isinstance(x, Tensor):
            raise TypeError(f"Expected x to be a Tensor but got {type(x)}")
        
        if x.device == Device.CPU:
            out_data = cpu_ops.transpose_forward(x.data, dim0, dim1)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {x.device} not supported")
        
        out = Tensor(out_data, device=x.device)
        
        ctx.dim0 = dim0
        ctx.dim1 = dim1
        
        return out
    
    @staticmethod
    def backward(ctx:Context, grad_output:Tensor):
        dim0 = ctx.dim0; dim1 = ctx.dim1
        
        if grad_output.device == Device.CPU:
            a_grad = cpu_ops.transpose_backward(grad_output.data, dim0, dim1)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {grad_output.device} not supported")
        
        out = Tensor(a_grad, device=grad_output.device)
        
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
    return Transpose.apply(x, dim0, dim1)


class Flatten(Function):
    
    @staticmethod
    def forward(ctx:Context, x:Tensor, start_dim:int, end_dim:int):
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
            raise RuntimeError(f"{ctx.fn_name}: {x.device} not supported")
        
        out = Tensor(out_data, device=x.device)
        
        ctx.x_shape = x.shape
        
        return out
    
    @staticmethod
    def backward(ctx:Context, grad_output:Tensor):
        x_shape = ctx.x_shape
        
        if grad_output.device == Device.CPU:
            a_grad = cpu_ops.reshape_backward(grad_output.data, x_shape)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {grad_output.device} not supported")
        
        x_grad = Tensor(a_grad, device=grad_output.device)
        
        return x_grad
    
    
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
    return Flatten.apply(x, start_dim, end_dim)


class UnfoldDim(Function):
    
    @staticmethod
    def forward(ctx:Context, x:Tensor, dimension:int, size:int, step:int):
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
            out_data = cpu_ops.unfold_forward(x.data, dimension, size, step)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {x.device} not supported")
        
        out = Tensor(out_data, device=x.device)
        
        ctx.x_shape = x.shape
        ctx.dimanesion = dimension
        ctx.size = size
        ctx.step = step
        
        return out
    
    @staticmethod
    def backward(ctx:Context, grad_output:Tensor):
        x_shape = ctx.x_shape
        dimension = ctx.dimanesion
        size = ctx.size
        step = ctx.step
        
        if grad_output.device == Device.CPU:
            a_grad = cpu_ops.unfold_backward(grad_output.data, x_shape, dimension, size, step)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {grad_output.device} not supported")
        
        x_grad = Tensor(a_grad, device=grad_output.device)
        
        return x_grad
    
    
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
    return UnfoldDim.apply(x, dimension, size, step)