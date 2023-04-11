import numpy as np
from .. import nn, Tensor
from .neurons import init_weights


class Linear(nn.Module):
    
    def __init__(self, input_size:int, output_size:int, weight_init_method='he'):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        # Randomly initialize weights and biases
        weight_values = [ init_weights(input_size, output_size, weight_init_method).astype(np.float32) for _ in range(output_size) ]
        self.weights = Tensor(weight_values, requires_grad=True)
        self.biases = Tensor(np.zeros((output_size,), dtype=np.float32), requires_grad=True)
        
    def forward(self, x:Tensor) -> Tensor:
        assert x.shape[1] == self.input_size, f"Expected input size '{self.input_size}' but received '{x.shape[1]}'"

        out = (x @ self.weights.transpose(0,-1)) + self.biases
        
        return out
    
    def parameters(self) -> list[Tensor]:
        return [self.weights, self.biases]
    
    def __repr__(self) -> str:
        return f"Linear(input_size={self.input_size}, neurons={len(self.output_size)})"


class Flatten(nn.Module):
    
    def __init__(self, start_dim=1, end_dim=-1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim
    
    def forward(self, x: Tensor) -> Tensor:
        return x.flatten(self.start_dim, self.end_dim)



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