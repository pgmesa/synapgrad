from typing import Any
from .tensor import Tensor
from .device import Device


def backward(grad_fn, grad_output):
    if grad_fn:
        gradients = grad_fn.apply(grad_output)
        functions = grad_fn.next_functions
        for i in range(len(functions)):
            if functions[i]:
                backward(functions[i], gradients[i])
    

class Context:
    
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
                if t.requires_grad and (t.is_leaf or t._retain_grad):
                    t.grad_fn = AccumulateGrad(t)
                elif t.requires_grad and not t.is_leaf:
                    t.grad_fn = BackwardFunction(cls)
                elif not t.requires_grad and t.is_leaf:
                    t.grad_fn = None
                else:
                    raise Exception("Leaf Tensors should ha")
        
                backward_function.next_functions.append(t.grad_fn)
            else:
                backward_function.next_functions.append(None)
            
        output_tensor.grad_fn = backward_function
        
        return output_tensor    
        
    
class AccumulateGrad:
    
    def __init__(self, tensor:Tensor) -> None:
        self.tensor = tensor
        self.next_functions = [] 
    
    def apply(self, grad_tensor:Tensor):
        
        if self.tensor.grad is None:
            self.tensor.grad = Tensor(grad_tensor.data, device=self.tensor.device)
        else:
            if self.tensor.device is Device.CPU:
                self.variable.grad.data += grad_tensor.data
    
    
class BackwardFunction:
    """ Function to be passed to :attr:`Tensor.grad_fn` to compute the gradient of a :class:`Tensor`.
        This funcition does not accumulate the gradient in the tensor"""
    
    def __init__(self, function:Function) -> None:
        self.function = function
        self.next_functions = []
        self.ctx = Context()
        self.name = function.__name__ + "Backward"
    
    def apply(self, *gradients):
        return self.function.backward(self.ctx, *gradients)
        
    def __repr__(self) -> str:
        return "<" + self.name + ">"
    
