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


class Conv2D(nn.Module):
    
    def __init__(self, filters, kernel_size, strides=None, padding=None) -> None:
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
    
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x)
    
    
class BatchNorm2d(nn.Module):
    
    def __init__(self, num_features:int, eps:float=0.00001, momentum:float=0.1, affine:bool=True) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x)
    
    
class MaxPool2D(nn.Module):
    
    def __init__(self, kernel_size:'int | tuple', stride:'int | tuple | None'=None, padding:'int | tuple'=0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x)
    

class Dropout(nn.Module):
    
    def __init__(self, p=0.5, inplace=False) -> None:
        super().__init__()
        self.p = p
        self.inplace = inplace
        
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x)