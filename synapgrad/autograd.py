from typing import Any
from .tensor import Tensor
from .device import Device


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
        for i in range(len(next_functions)):
            if next_functions[i]:
                backward(next_functions[i], gradients[i])
    

class Context:
    """
    Context class for the autograd functions. It allows to save tensors and variables in the forward pass
    that are needed for the backward pass.
    """
    
    def __init__(self) -> None:
        self.saved_tensors = []
    
    def save_for_backward(self, *tensors:Tensor):
        """ Saves given tensors for a future call to :func:`~Function.backward`.

        ``save_for_backward`` should be called at most once, only from inside the
        :func:`forward` method, and only with tensors. """
        
        for t in tensors:
            if not isinstance(t, Tensor):
                raise Exception(f"Got type {t} but only Tensors should be saved in save_for_backward")

            self.saved_tensors.append(t.copy())
    

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
        """ Applies the function to the given inputs. For example Add, Mul, Matmul, ... """
        
        backward_function = BackwardFunction(cls)
        
        output_tensor = cls.forward(backward_function.ctx, *inputs)
        
        for t in inputs:
            if isinstance(t, Tensor):
                if t.grad_fn is None:
                    if t.requires_grad and (t.is_leaf or t._retain_grad):
                        t.grad_fn = AccumulateGrad(t)
                    elif t.requires_grad and not t.is_leaf:
                        t.grad_fn = BackwardFunction(cls)
                    elif not t.requires_grad and t.is_leaf:
                        t.grad_fn = None
                    else:
                        raise Exception("Tensor requires_grad is False but is_leaf is also False")
        
                backward_function.next_functions.append(t.grad_fn)
            else:
                backward_function.next_functions.append(None)
            
        output_tensor.grad_fn = backward_function
        
        return output_tensor    
        
    
class AccumulateGrad:
    """
    Node where the gradient is accumulated. 
    """
    
    def __init__(self, tensor:Tensor) -> None:
        self.tensor = tensor
        self.next_functions = [] 
    
    def apply(self, grad_tensor:Tensor):
        
        if self.tensor.grad is None:
            self.tensor.grad = Tensor(grad_tensor.data, device=self.tensor.device)
        else:
            if self.tensor.device is Device.CPU:
                self.tensor.grad.data += grad_tensor.data
    
    
class BackwardFunction:
    """ 
    Function to be passed to :attr:`Tensor.grad_fn` to compute the gradient of a :class:`Tensor`.
    This funcition does not accumulate the gradient in the tensor
    """
    
    def __init__(self, function:Function) -> None:
        self.ctx = Context()
        self.function = function
        self.next_functions = []
        self.name = function.__name__ + "Backward"
    
    def apply(self, *gradients):
        return self.function.backward(self.ctx, *gradients)
        
    def __repr__(self) -> str:
        return "<" + self.name + ">"
    
