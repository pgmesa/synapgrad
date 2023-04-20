import numpy as np
from .. import nn, Tensor
from .neurons import init_weights
from .functional import unfold, fold


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
        Applies Max Pooling 2D to a batch of N images with C channels (N, C, H, W).
        References:
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
        
        kernel_size = np.broadcast_to(kernel_size, 2)
        if stride is None:
            stride = kernel_size
        else:
            stride = np.broadcast_to(stride, 2)
        padding = np.broadcast_to(padding, 2)
        dilation = np.broadcast_to(dilation, 2)
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
    def forward(self, x: Tensor) -> Tensor: 
        N, C, H, W = x.shape
        lH = int(np.floor((H + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1))
        lW = int(np.floor((W + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1))
        
        unfolded = unfold(x.data, kernel_size=self.kernel_size, stride=self.stride, dilation=self.dilation,
                           padding=self.padding, pad_value=-np.inf)
        
        N, C_k, L = unfolded.shape
        split_per_channel = unfolded.reshape(N, C, int(C_k/C), L)
        
        unfolded_reorganized = np.moveaxis(split_per_channel, -2, -1).reshape(N, C, lH, lW, -1)
        pooled, max_indices, selected = Tensor(unfolded_reorganized).max(dim=-1, return_selected=True)
    
        out = Tensor(pooled.data, (x,), "<MaxPool2d>", requires_grad=x.requires_grad)
        
        def _backward():
            if x.requires_grad:
                grad = selected * np.expand_dims(out._grad, axis=-1)

                grad = grad.reshape(N, C, L, int(C_k/C))
                grad = np.moveaxis(grad, -1, -2)
                grad = grad.reshape(N, C_k, L)
                
                grad = fold(grad, (H,W), kernel_size=self.kernel_size, stride=self.stride,
                     dilation=self.dilation, padding=self.padding)

                x._grad += grad
                
        out._backward = _backward
        
        return out


class Conv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1) -> None:
        """
        Applies 2D Convolution to a batch of N images with C channels (N, C, H, W).
        Reference:
            https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

        Args:
            in_channels (int): the number of input channels
            out_channels (int): the number of output channels produced by the convolution
            kernel_size (int or tuple): the size of the sliding blocks
            stride (int or tuple, optional): the stride of the sliding blocks in the input spatial dimensions. Default: 1
            padding (int or tuple, optional):  implicit zero padding to be added on both sides of input. Default: 0
            dilation (int or tuple, optional): a parameter that controls the stride of elements within the neighborhood. Default: 1
        """
        super().__init__()
        
        kernel_size = np.broadcast_to(kernel_size, 2)
        if stride is None:
            stride = kernel_size
        else:
            stride = np.broadcast_to(stride, 2)
        padding = np.broadcast_to(padding, 2)
        dilation = np.broadcast_to(dilation, 2)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        weight_values = ... # np.random.rand(out_channels, in_channels, *kernel_size) * 0.01
        self.weights = Tensor(weight_values, requires_grad=True)
        self.biases = Tensor(np.zeros((out_channels,), dtype=np.float32), requires_grad=True)
    
    def forward(self, x: Tensor) -> Tensor:
        # unfolded = unfold(x.data, kernel_size=self.kernel_size, stride=self.stride, dilation=self.dilation,
        #                    padding=self.padding, pad_value=-np.inf)
        
        # out = unfolded.transpose(1,2) @ self.weights + self.biases
        
        # return out.transpose(1,2)
        return super().forward(x)
        
    def parameters(self) -> list['Tensor']:
        return [self.weights, self.biases]
    
    def __repr__(self) -> str:
        return (f"Conv2d(in_channels={self.in_channels}, out_channels={self.out_channels}, " + 
                f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, " + 
                f"dilation={self.dilation}")
    
    
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
    """ 
    Reference:
        https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html
    
    """
    
    def __init__(self, p=0.5, inplace=False) -> None:
        super().__init__()
        self.p = p
        self.inplace = inplace
        
    def forward(self, x: Tensor) -> Tensor:
        if not self.training: return x
        random_data = np.random.rand(*x.shape)
        random_data = np.where(random_data <= self.p, 0, 1)
        if self.p < 1:
            random_data = random_data / (1-self.p) # scale data
        random_t = Tensor(random_data)
        
        return x*random_t