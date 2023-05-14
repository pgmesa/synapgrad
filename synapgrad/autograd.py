from typing import Any
from contextlib import contextmanager
from synapgrad.tensor import Tensor
from synapgrad.device import Device
from synapgrad.tools import recursively_seek_tensors


# ********************************
# ******* Context Managers *******
# ********************************

gradient__ = True
retain_grads__ = False
retain_children__ = False

class no_grad:
    
    def __init__(self) -> None:
        self.prev = gradient__
    
    def __enter__(self):
        global gradient__
        gradient__ = False
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        global gradient__
        gradient__ = self.prev
        

class retain_grads:
    def __init__(self) -> None:
        self.prev = retain_grads__
    
    def __enter__(self):
        global retain_grads__
        retain_grads__ = True
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        global retain_grads__
        retain_grads__ = self.prev
        

class retain_children():
    def __init__(self) -> None:
        self.prev = retain_children__
    
    def __enter__(self):
        global retain_children__
        retain_children__ = True
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        global retain_children__
        retain_children__ = self.prev
        

@contextmanager
def retain_all():
    with retain_children() as a, retain_grads() as b:
        yield (a, b)

# *******************************
# ******* Autograd Engine *******
# *******************************

def backward(grad_fn, grad_output):
    """ 
    Recursively calculate the gradients of the computational graph 
    
    Args:
        grad_fn (Function): The function that computes the gradients. BackwardFunction or AccumulatedGrad
        grad_output (Tensor): The gradient of the output.
    
    Returns:
        None
    """
    if grad_fn:
        gradients = grad_fn.apply(grad_output)
        next_functions = grad_fn.next_functions
        
        if isinstance(gradients, Tensor):
            gradients = [gradients]
        
        for i in range(len(next_functions)):
            next_grad_fn, out_index = next_functions[i]
            if next_grad_fn:
                if next_grad_fn.var_out_index != out_index:
                    raise RuntimeError("Internal backprop error - next_grad_fn.var_out_index " + 
                        f"{next_grad_fn.var_out_index} != out_index {out_index}")
                backward(next_grad_fn, gradients[i])
    

class Context:
    """
    Context class for the autograd functions. It allows to save tensors and variables in the forward pass
    that are needed for the backward pass.
    """
    
    def __init__(self, function) -> None:
        self.function = function
        self.saved_tensors = []
    
    @property
    def fn_name(self):
        return self.function.__name__
    
    def save_for_backward(self, *tensors:Tensor):
        """ Saves given tensors for a future call to :func:`~Function.backward`.

        ``save_for_backward`` should be called at most once, only from inside the
        :func:`forward` method, and only with tensors. """
        
        for t in tensors:
            if not isinstance(t, Tensor):
                tp_t = type(t).__name__
                raise RuntimeError(f"Got type {tp_t} but only Tensors should be saved in save_for_backward")

            self.saved_tensors.append(t.clone())
    

class Function:
    """
    Superclass for all autograd functions. 
    
    All autograd functions must subclass this class.  
    """
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self.apply(*args, **kwds)
    
    @staticmethod
    def forward(ctx:Context, *inputs):
        raise NotImplementedError
    
    @staticmethod
    def backward(ctx:Context, *grad_outputs):
        raise NotImplementedError
    
    @classmethod
    def apply(cls, *inputs):
        """ 
        Applies the function to the given inputs and associates its backward function.
        For example: Add, Mul, Matmul, ...
        """
        input_tensors = recursively_seek_tensors(*inputs)
        
        if len(input_tensors) == 0:
            raise RuntimeError(f"{cls.__name__}: no input tensors found")
        same_device = all(x.device == input_tensors[0].device for x in input_tensors)
        if not same_device:
            raise RuntimeError(f"{cls.__name__}: all inputs must be on the same device")
        
        backward_ctx = Context(cls)
        output_tensors = cls.forward(backward_ctx, *inputs)
        
        if isinstance(output_tensors, Tensor):
            output_tensors = [output_tensors]
            
        for i, output_tensor in enumerate(output_tensors):
            output_tensor.requires_grad = any(inp.requires_grad for inp in input_tensors) and gradient__
            output_tensor._children = input_tensors if retain_children__ else ()
            output_tensor._operation = cls.__name__
            output_tensor._retain_grad = retain_grads__
            
            if output_tensor.requires_grad:
                next_functions = []
                for t in input_tensors:
                    grad_fn = t.grad_fn
                    if t.requires_grad and t.is_leaf:
                        grad_fn = AccumulateGrad(t)

                    next_functions.append((grad_fn, grad_fn.var_out_index if grad_fn else 0))

                backward_function = BackwardFunction(backward_ctx, output_tensor, i)
                backward_function.next_functions = tuple(next_functions)

                output_tensor.grad_fn = backward_function
        
        return output_tensors if len(output_tensors) > 1 else output_tensors[0]    
        
    
class AccumulateGrad:
    """
    Node where the gradient is accumulated. 
    """
    
    def __init__(self, variable:Tensor) -> None:
        self.variable = variable
        self.var_out_index = 0
        self.next_functions = ()
        self.__name = "AccumulateGrad"
        
    def name(self):
        return self.__name
    
    def apply(self, grad_tensor:Tensor):
        if not self.variable.has_grad():
            self.variable.zero_()
            
        if self.variable.device is Device.CPU:
            if not grad_tensor.matches_shape(self.variable):
                raise RuntimeWarning("AccumulateGrad - calculated gradient doesn't match " + 
                f"the shape of the variable. Expected {self.variable.shape} but got {grad_tensor.shape}")
            self.variable.grad.data += grad_tensor.data
        else:
            raise RuntimeError("AccumulateGrad is not supported for GPU tensors")
    
    def __repr__(self) -> str:
        string:str = super().__repr__()
        idx =string.find(" ")
        string = "<" + self.name() + string[idx:]
        return string
    
    
class BackwardFunction:
    """ 
    Function to be passed to :attr:`Tensor.grad_fn` to compute the gradient of a :class:`Tensor`.
    This funcition does not accumulate the gradient in the tensor
    """
    
    def __init__(self, context:Context, variable:Tensor, var_out_index:int) -> None:
        """ 
        Args:
            variable (Tensor): The tensor which will be the owner if this BackwardFunction, stored in .grad_fn attribute
            context (Context): The context with the necessary data to compute the backward pass.
            var_out_index (int): The index of this tensor in the list of outputs generated by the function in the forward pass
        """
        self.ctx = context
        self.variable = variable
        self.var_out_index = var_out_index
        self.next_functions = ()
    
    def name(self):
        return self.ctx.fn_name + "Backward"
    
    def apply(self, *grad_output):
        if self.variable._retain_grad:
            AccumulateGrad(self.variable).apply(*grad_output)
        self.ctx.forward_out_index = self.var_out_index
        return self.ctx.function.backward(self.ctx, *grad_output)
    
    def __repr__(self) -> str:
        string:str = super().__repr__()
        idx =string.find(" ")
        string = "<" + self.name() + string[idx:]
        return string
    
