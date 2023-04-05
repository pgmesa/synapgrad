
import numpy as np
from deeplib.engine import Tensor
from deeplib.modules import Module


class ReLU(Module):
    
    def __call__(self, x:Tensor) -> Tensor:
        assert isinstance(x, Tensor), "Input must be a Tensor"
        relu = np.maximum(0, x.data)
        out = Tensor(relu, (x,), '<ReLU>', requires_grad=x.requires_grad)

        def _backward():
            x._grad += (out.data > 0) * out._grad
        
        out._backward = _backward

        return out


class Sigmoid(Module):
    
    def __call__(self, x:Tensor) -> Tensor:
        assert isinstance(x, Tensor), "Input must be a Tensor"
        sigmoid = 1/(1 + np.e**-x.data)
        out = Tensor(sigmoid, (x,), '<Sigmoid>', requires_grad=x.requires_grad)
    
        def _backward():
            f = 1/(1 + np.exp(-x.data))
            x_grad = f * (1 - f)
            
            x._grad = (x_grad) * out._grad
            
        out._backward = _backward
        
        return out
    
if __name__ == "__main__":
    a = Tensor([-1,-2,4,5,1,7], requires_grad=True)
    a.retain_grad()
    b = Sigmoid()(a)
    print(b)
    b.sum().backward()
    print(a)
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
