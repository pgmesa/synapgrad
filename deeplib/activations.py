
import numpy as np
from deeplib.engine import Tensor
from deeplib.modules import Module


class ReLU(Module):
    
    def forward(self, x:Tensor) -> Tensor:
        assert isinstance(x, Tensor), "Input must be a Tensor"
        relu = np.maximum(0, x.data)
        out = Tensor(relu, (x,), '<ReLU>', requires_grad=x.requires_grad)

        def _backward():
            if x.requires_grad:
                x._grad += (out.data > 0) * out._grad
        
        out._backward = _backward

        return out


class Sigmoid(Module):
    
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
    
if __name__ == "__main__":
    l1 = [[-1.0,-2.0,4.0,5.0,1.0,7.0]]
    a = Tensor(l1, requires_grad=True)
    a.retain_grad()
    b = Sigmoid()(a)
    print(Sigmoid())
    print(b)
    b.sum().backward()
    print(a)
    print(a.grad)
    
    import torch
    a = torch.tensor(l1, requires_grad=True)
    a.retain_grad()
    out = torch.nn.Sigmoid()(a)
    out.sum().backward()
    print(out)
    print(a.grad)
    
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
