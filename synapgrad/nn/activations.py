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
        sigmoid = 1/(1 + np.e**-x.data)
        out = Tensor(sigmoid, (x,), '<Sigmoid>', requires_grad=x.requires_grad)
    
        def _backward():
            if x.requires_grad:
                f = 1/(1 + np.exp(-x.data))
                x_grad = f * (1 - f)
                
                x._grad = (x_grad) * out._grad
            
        out._backward = _backward
        
        return out
    
    
# class Softmax:
    
#     def __init__(self) -> None:
#         super().__init__()
#         self.trainable = False
    
#     def __call__(self, x:np.ndarray) -> np.ndarray:
#         super().__call__(x)
#         softmax = []     
#         for sample in x:
#             exp = np.exp(sample)
#             out = exp / np.sum(exp)
#             softmax.append(out)
    
#         self.output = np.array(softmax)

#         return self.output
