
from .engine import Tensor

import numpy as np

class Relu:
    
    def __call__(self, x:Tensor) -> Tensor:
        out = Tensor(np.maximum(0, x.data), (x,), 'ReLU')

        def _backward():
            x.grad += (out.data > 0) * out.grad
        
        out._backward = _backward

        return out


class Sigmoid:
    
    def __call__(self, x:Tensor) -> Tensor:
        super().__call__(x)
        return (1/(1 + np.exp(-x)))
    
    def backward(self, grad):
        """ grad is the """
        f = 1/(1 + np.exp(-self.input))
        df = f * (1 - f)
        
        return df
    
class Softmax:
    
    def __init__(self) -> None:
        super().__init__()
        self.trainable = False
    
    def __call__(self, x:np.ndarray) -> np.ndarray:
        super().__call__(x)
        softmax = []     
        for sample in x:
            exp = np.exp(sample)
            out = exp / np.sum(exp)
            softmax.append(out)
    
        self.output = np.array(softmax)

        return self.output
    