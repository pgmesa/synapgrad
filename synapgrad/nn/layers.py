import numpy as np
from .. import nn, Tensor
from .neurons import init_weights
from .functional import unfold


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
    

class MaxPool2d(nn.Module):
    
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1) -> None:
        """
        Apply Max Pooling 2D to a batch of N images with C channels (N, C, H, W).
        Reference:
            https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html

        Args:
            kernel_size (int or tuple): the size of the sliding blocks
            stride (int or tuple, optional): the stride of the sliding blocks in the input spatial dimensions. Default: 1
            padding (int or tuple, optional):  implicit zero padding to be added on both sides of input. Default: 0
            dilation (int or tuple, optional): a parameter that controls the stride of elements within the neighborhood. Default: 1

        Output:
            Tensor of shape (N, C, H // stride[0], W // stride[1]).
        """
        super().__init__()
        self.kernel_size = kernel_size
        if stride is None:
            stride = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
    def forward(self, x: Tensor) -> Tensor: 
        unfolded = unfold(x.data, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation)      
        pooled = np.max(pairs, axis=(2, 3))
        folded = ...
        
        out = Tensor(folded, (x,), "<MaxPool2d>", requires_grad=x.requires_grad)
        
        def _backward():
            if x.requires_grad:
                # Esto para Conv2D
                grad = np.repeat(out._grad, pool_height, axis=2)
                grad = np.repeat(grad, pool_width, axis=3)
                # poner a 0 los no escogidos en pool (pool indices)
                # indices = pairs.reshape(N,C,H,W)
                x._grad += grad
                
        out._backward = _backward
        
        return out


class Conv2d(nn.Module):
    
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


class Dropout(nn.Module):
    
    def __init__(self, p=0.5, inplace=False) -> None:
        super().__init__()
        self.p = p
        self.inplace = inplace
        
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x)