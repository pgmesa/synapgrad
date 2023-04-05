
from deeplib.engine import Tensor
import numpy as np


class ReLU:
    
    def __call__(self, x:Tensor) -> Tensor:
        assert isinstance(x, Tensor), "Input must be a Tensor"
        out = Tensor(np.maximum(0, x.data), (x,), 'ReLU')

        def _backward():
            x.grad += (out.data > 0) * out.grad
        
        out._backward = _backward

        return out


class Sigmoid:
    
    def __call__(self, x:Tensor) -> Tensor:
        assert isinstance(x, Tensor), "Input must be a Tensor"
        out = Tensor(1/(1 + np.e**-x))
    
        def _backward():
            f = 1/(1 + np.exp(-x.data))
            x_grad = f * (1 - f)
            
            x.grad = (x_grad) * out.grad
            
        out._backward = _backward
        
        return out
    
if __name__ == "__main__":
    a = Tensor([-1,-2,4,5,1,7])
    b = Sigmoid()(a)
    b.backward()
    print(a)
    
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
