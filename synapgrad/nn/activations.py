import numpy as np
from .. import Tensor, nn


class ReLU(nn.Module):
    
    def forward(self, x:Tensor) -> Tensor:
        assert isinstance(x, Tensor), "Input must be a Tensor"
        relu = np.maximum(0, x.data)
        out = Tensor(relu, (x,), '<ReLU>', requires_grad=x.requires_grad)

        def _backward():
            if x.requires_grad:
                x._grad += (out.data > 0) * out._grad
        
        out._backward = _backward

        return out


class Sigmoid(nn.Module):
    
    def forward(self, x:Tensor) -> Tensor:
        # Returning (1/(1 + np.e**-x)) should be enough, but defining explicit 
        # grad function should be faster
        assert isinstance(x, Tensor), "Input must be a Tensor"
        sigmoid = 1/(1 + np.exp(-x.data))
        out = Tensor(sigmoid, (x,), '<Sigmoid>', requires_grad=x.requires_grad)
    
        def _backward():
            if x.requires_grad:
                f = 1/(1 + np.exp(-x.data))
                x_grad = f * (1 - f)
                
                x._grad += (x_grad) * out._grad
            
        out._backward = _backward
        
        return out
    
    
class Softmax(nn.Module):
    """ Reference: https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/ """
    
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim
    
    def forward(self, x: Tensor) -> Tensor:
        assert isinstance(x, Tensor), "Input must be a Tensor"
        return x.exp() / (x.exp().sum(-1)).unsqueeze(-1)