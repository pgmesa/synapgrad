
from deeplib.engine import Tensor
from deeplib.neurons import Neuron
from deeplib.modules import Module

import numpy as np


class Linear(Module):
    
    def __init__(self, input_size:int, output_size:int, weight_init_method='he'):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.neurons = []
        for _ in range(self.output_size):
            n = Neuron(input_size, weight_init_method=weight_init_method)
            self.neurons.append(n)
        
    def __call__(self, x:Tensor) -> list:
        assert x.shape[0] == self.input_size, f"Expected input size '{self.input_size}' but received '{x.shape[0]}'"
        
        out = Tensor.concat([ neuron(x) for neuron in self.neurons ])
        
        return out
    
    def parameters(self) -> list[Tensor]:
        return [p for n in self.neurons for p in n.parameters()]
        
    
if __name__ == "__main__":
    list_ = [0.2, 1, 4, 2, -1, 4]
    inp = Tensor(list_, requires_grad=True)
    inp.retain_grad()
    linear = Linear(6,2)
    out = linear(inp)
    out.sum().backward()
    print(inp)
    print(inp.grad)
    
    # print(inp.shape)
    # lin = Linear(inp.shape[1], 1)
    # out = lin(inp)
    # out.backward()
    # print(inp, lin.parameters())
    
# class Conv2D(Layer):
    
#     def __init__(self, filters, kernel_size, strides=None, padding=None) -> None:
#         self.filters = filters
#         self.kernel_size = kernel_size
#         self.strides = strides
#         self.padding = padding
        
#     def __call__(self, x:np.ndarray) -> np.ndarray:
#         super().__call__(x)
#         ...
    
# class BatchNorm2d(Layer):
#     ...
    
# class MaxPool2D(Layer):
    
#     def __call__(self, x:np.ndarray) -> np.ndarray:
#         super().__call__(x)
#         ...